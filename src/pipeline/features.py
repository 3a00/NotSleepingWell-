"""Multi-modal physiological feature extraction for UCDDB PSG Recordings."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from scipy.signal import find_peaks, welch  # type: ignore[import-untyped]

from src.pipeline.loader import RecordingData
from src.pipeline.windowing import WindowRecord, slice_channel_window

_RESPIRATORY_CHANNELS: tuple[str, ...] = ("Flow", "Sum", "ribcage", "abdo")
_EEG_CHANNELS: tuple[str, ...] = ("C3A2", "C4A1")
_EEG_BAND_KEYS: tuple[str, ...] = ("delta", "theta", "alpha", "sigma", "beta")
_MS_PER_SEC = 1000.0
_MS_PER_MINUTE = 60000.0
_PNN50_THRESHOLD_MS = 50.0
_HRV_INTERP_FS_HZ = 4.0
_HRV_MIN_VALID_INTERVALS = 4
_RPEAK_PROMINENCE_IQR_SCALE = 1.5
_RPEAK_PROMINENCE_STD_SCALE = 1.0
_SOUND_BURST_STD_MULT = 2.0
_EPSILON = 1e-8
_RATIO_CLIP_MAX = 10.0
_SPECTRAL_RATIO_CLIP_MAX = 100.0


@dataclass(frozen=True)
class RecordingContext:
    """Continuous Recording-level pre-computed R-peaks, RR intervals, and 5-minute rolling baselines.

    Attributes:
        rpeak_indices: 1D ``int64`` sample indices of detected ECG R-peaks.
        rpeak_times_sec: 1D ``float64`` timestamps in seconds of detected ECG R-peaks.
        rr_intervals_ms: 1D ``float64`` consecutive RR intervals in milliseconds
            (length ``max(0, len(rpeak_times_sec) - 1)``).
        rr_mid_times_sec: 1D ``float64`` midpoint timestamps in seconds of each RR interval.
        rr_valid_mask: 1D ``bool`` mask marking physiological RR intervals within
            ``[rr_min_ms, rr_max_ms]``.
        spo2_baseline_p90: 1D ``float64`` array of per-Window 5-minute rolling
            percentile baselines for ``SpO2``.
        resp_baseline_rms: Mapping from respiratory channel name (``Flow``, ``Sum``,
            ``ribcage``, ``abdo``) to 1D ``float64`` per-Window 5-minute rolling RMS.
        resp_baseline_ptp: Mapping from respiratory channel name to 1D ``float64``
            per-Window 5-minute rolling peak-to-peak amplitude.
    """

    rpeak_indices: npt.NDArray[np.int64]
    rpeak_times_sec: npt.NDArray[np.float64]
    rr_intervals_ms: npt.NDArray[np.float64]
    rr_mid_times_sec: npt.NDArray[np.float64]
    rr_valid_mask: npt.NDArray[np.bool_]
    spo2_baseline_p90: npt.NDArray[np.float64]
    resp_baseline_rms: dict[str, npt.NDArray[np.float64]]
    resp_baseline_ptp: dict[str, npt.NDArray[np.float64]]


def _compute_rms(values: npt.NDArray[np.floating[Any]]) -> float:
    """Compute root-mean-square in ``float64`` precision, returning ``0.0`` if empty."""
    if values.size == 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(arr))))


def _detect_ecg_rpeaks(
    ecg_signal: npt.NDArray[np.floating[Any]],
    fs_ecg: float,
    min_distance_sec: float,
    rr_min_ms: float,
    rr_max_ms: float,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]:
    """Detect R-peaks on continuous filtered ECG with polarity selection and adaptive prominence."""
    ecg = np.asarray(ecg_signal, dtype=np.float64)
    empty_idx: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
    empty_f: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    empty_b: npt.NDArray[np.bool_] = np.empty(0, dtype=np.bool_)
    if ecg.size == 0 or fs_ecg <= 0.0:
        return empty_idx, empty_f, empty_f, empty_f, empty_b

    p99 = float(np.percentile(ecg, 99.0))
    p1 = float(np.percentile(ecg, 1.0))
    oriented = -ecg if abs(p1) > abs(p99) else ecg

    q75, q25 = np.percentile(oriented, [75.0, 25.0])
    iqr = float(q75 - q25)
    std = float(np.std(oriented))
    prominence = max(_RPEAK_PROMINENCE_IQR_SCALE * iqr, _RPEAK_PROMINENCE_STD_SCALE * std)
    min_distance_samples = max(1, int(round(min_distance_sec * fs_ecg)))

    if prominence <= _EPSILON:
        return empty_idx, empty_f, empty_f, empty_f, empty_b

    peaks, _ = find_peaks(oriented, distance=min_distance_samples, prominence=prominence)
    rpeak_indices: npt.NDArray[np.int64] = np.asarray(peaks, dtype=np.int64)
    rpeak_times_sec: npt.NDArray[np.float64] = np.asarray(
        rpeak_indices / fs_ecg, dtype=np.float64
    )

    if rpeak_times_sec.size < 2:
        return rpeak_indices, rpeak_times_sec, empty_f, empty_f, empty_b

    rr_intervals_ms: npt.NDArray[np.float64] = np.asarray(
        np.diff(rpeak_times_sec) * _MS_PER_SEC, dtype=np.float64
    )
    rr_mid_times_sec: npt.NDArray[np.float64] = np.asarray(
        0.5 * (rpeak_times_sec[:-1] + rpeak_times_sec[1:]), dtype=np.float64
    )
    rr_valid_mask: npt.NDArray[np.bool_] = np.asarray(
        (rr_intervals_ms >= rr_min_ms) & (rr_intervals_ms <= rr_max_ms), dtype=np.bool_
    )
    return rpeak_indices, rpeak_times_sec, rr_intervals_ms, rr_mid_times_sec, rr_valid_mask


def precompute_recording_context(
    signals: Mapping[str, npt.NDArray[np.floating[Any]]],
    sampling_rates: Mapping[str, float],
    config: Mapping[str, Any],
) -> RecordingContext:
    """Precompute continuous ECG R-peaks/RR intervals and 5-minute rolling Window baselines.

    Detects R-peaks across the continuous filtered ``ECG`` waveform using
    ``scipy.signal.find_peaks`` with minimum refractory distance and adaptive
    prominence, filters RR intervals to the physiological range
    ``[rr_min_ms, rr_max_ms]``, and computes causal 5-minute (``baseline_window_sec``)
    rolling Window baselines for ``SpO2`` (rolling 90th percentile) and respiratory
    channels ``Flow``, ``Sum``, ``ribcage``, and ``abdo`` (rolling RMS and peak-to-peak
    amplitude).

    Args:
        signals: Mapping of canonical channel name to 1D filtered waveform.
        sampling_rates: Mapping of canonical channel name to sampling rate in Hz.
        config: Pipeline configuration dictionary loaded from ``config.yaml``.

    Returns:
        Populated ``RecordingContext`` instance for the Recording.
    """
    feat_cfg: Mapping[str, Any] = config["features"]
    win_cfg: Mapping[str, Any] = config["windowing"]

    window_sec = float(win_cfg["window_sec"])
    baseline_window_sec = float(feat_cfg["baseline_window_sec"])
    rpeak_min_distance_sec = float(feat_cfg["rpeak_min_distance_sec"])
    rr_min_ms = float(feat_cfg["rr_min_ms"])
    rr_max_ms = float(feat_cfg["rr_max_ms"])
    spo2_pctl = float(feat_cfg["spo2_baseline_percentile"])

    ecg_sig = signals["ECG"]
    fs_ecg = float(sampling_rates["ECG"])
    rpeak_idx, rpeak_times, rr_ms, rr_mid, rr_valid = _detect_ecg_rpeaks(
        ecg_signal=ecg_sig,
        fs_ecg=fs_ecg,
        min_distance_sec=rpeak_min_distance_sec,
        rr_min_ms=rr_min_ms,
        rr_max_ms=rr_max_ms,
    )

    spo2_sig = np.asarray(signals["SpO2"], dtype=np.float64)
    fs_spo2 = float(sampling_rates["SpO2"])
    duration_sec = spo2_sig.size / fs_spo2 if fs_spo2 > 0.0 else 0.0
    n_windows = max(1, int(duration_sec // window_sec))

    spo2_baseline_p90 = np.empty(n_windows, dtype=np.float64)
    for w_idx in range(n_windows):
        w_end = (w_idx + 1) * window_sec
        b_start = max(0.0, w_end - baseline_window_sec)
        b_slice = slice_channel_window(spo2_sig, fs_spo2, b_start, w_end)
        spo2_baseline_p90[w_idx] = (
            float(np.percentile(b_slice, spo2_pctl)) if b_slice.size > 0 else spo2_pctl
        )

    resp_baseline_rms: dict[str, npt.NDArray[np.float64]] = {}
    resp_baseline_ptp: dict[str, npt.NDArray[np.float64]] = {}

    for ch in _RESPIRATORY_CHANNELS:
        ch_sig = np.asarray(signals[ch], dtype=np.float64)
        fs_ch = float(sampling_rates[ch])
        rms_arr = np.empty(n_windows, dtype=np.float64)
        ptp_arr = np.empty(n_windows, dtype=np.float64)

        for w_idx in range(n_windows):
            w_end = (w_idx + 1) * window_sec
            b_start = max(0.0, w_end - baseline_window_sec)
            b_slice = slice_channel_window(ch_sig, fs_ch, b_start, w_end)
            if b_slice.size > 0:
                rms_arr[w_idx] = _compute_rms(b_slice)
                ptp_arr[w_idx] = float(np.ptp(b_slice))
            else:
                rms_arr[w_idx] = 0.0
                ptp_arr[w_idx] = 0.0

        resp_baseline_rms[ch] = rms_arr
        resp_baseline_ptp[ch] = ptp_arr

    return RecordingContext(
        rpeak_indices=rpeak_idx,
        rpeak_times_sec=rpeak_times,
        rr_intervals_ms=rr_ms,
        rr_mid_times_sec=rr_mid,
        rr_valid_mask=rr_valid,
        spo2_baseline_p90=spo2_baseline_p90,
        resp_baseline_rms=resp_baseline_rms,
        resp_baseline_ptp=resp_baseline_ptp,
    )


def extract_spo2_pulse_features(
    w_start: float,
    w_end: float,
    window_idx: int,
    spo2_signal: npt.NDArray[np.floating[Any]],
    pulse_signal: npt.NDArray[np.floating[Any]],
    fs_spo2: float,
    fs_pulse: float,
    context: RecordingContext,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Extract in-window SpO2 statistics, +15 s look-ahead desaturation features, and Pulse stats.

    Args:
        w_start: Window start time in seconds.
        w_end: Window end time in seconds.
        window_idx: Zero-based Window index into ``context.spo2_baseline_p90``.
        spo2_signal: Continuous cleaned 1D ``SpO2`` signal.
        pulse_signal: Continuous cleaned 1D ``Pulse`` signal.
        fs_spo2: ``SpO2`` sampling rate in Hz.
        fs_pulse: ``Pulse`` sampling rate in Hz.
        context: Pre-computed ``RecordingContext`` with 5-minute rolling ``SpO2`` baselines.
        config: Pipeline configuration dictionary loaded from ``config.yaml``.

    Returns:
        Dictionary of 13 finite ``float`` features for ``SpO2`` and ``Pulse``.
    """
    feat_cfg: Mapping[str, Any] = config["features"]
    spo2_lag_sec = float(feat_cfg["spo2_lag_sec"])
    desat_threshold_pct = float(feat_cfg["spo2_desat_threshold_pct"])
    hypoxia_threshold_pct = float(feat_cfg["spo2_hypoxia_threshold_pct"])

    spo2_win = np.asarray(
        slice_channel_window(spo2_signal, fs_spo2, w_start, w_end), dtype=np.float64
    )
    spo2_lag_win = np.asarray(
        slice_channel_window(spo2_signal, fs_spo2, w_start, w_end + spo2_lag_sec),
        dtype=np.float64,
    )
    pulse_win = np.asarray(
        slice_channel_window(pulse_signal, fs_pulse, w_start, w_end), dtype=np.float64
    )

    if context.spo2_baseline_p90.size > 0:
        clamped_idx = min(max(window_idx, 0), int(context.spo2_baseline_p90.size) - 1)
        baseline_spo2 = float(context.spo2_baseline_p90[clamped_idx])
    else:
        spo2_pctl = float(feat_cfg["spo2_baseline_percentile"])
        baseline_spo2 = (
            float(np.percentile(spo2_win, spo2_pctl))
            if spo2_win.size > 0
            else spo2_pctl
        )

    if spo2_win.size > 0:
        spo2_min = float(np.min(spo2_win))
        spo2_mean = float(np.mean(spo2_win))
        spo2_std = float(np.std(spo2_win))
        spo2_ptp = float(np.ptp(spo2_win))
        spo2_drop_from_baseline = max(0.0, baseline_spo2 - spo2_min)
        spo2_time_below_90 = float(np.mean(spo2_win < hypoxia_threshold_pct))
        if spo2_win.size >= 2 and fs_spo2 > 0.0:
            t_rel = np.arange(spo2_win.size, dtype=np.float64) / fs_spo2
            t_centered = t_rel - np.mean(t_rel)
            denom = float(np.sum(np.square(t_centered)))
            spo2_slope = (
                float(np.sum(t_centered * (spo2_win - spo2_mean)) / denom)
                if denom > _EPSILON
                else 0.0
            )
        else:
            spo2_slope = 0.0
    else:
        spo2_min = baseline_spo2
        spo2_mean = baseline_spo2
        spo2_std = 0.0
        spo2_ptp = 0.0
        spo2_drop_from_baseline = 0.0
        spo2_time_below_90 = 0.0
        spo2_slope = 0.0

    if spo2_lag_win.size > 0:
        spo2_lag_min = float(np.min(spo2_lag_win))
        spo2_lag_drop = max(0.0, baseline_spo2 - spo2_lag_min)
    else:
        spo2_lag_min = spo2_min
        spo2_lag_drop = spo2_drop_from_baseline

    spo2_ods_3pct = 1.0 if spo2_lag_drop >= desat_threshold_pct else 0.0

    if pulse_win.size > 0:
        pulse_mean = float(np.mean(pulse_win))
        pulse_std = float(np.std(pulse_win))
        pulse_ptp = float(np.ptp(pulse_win))
    else:
        pulse_mean = 0.0
        pulse_std = 0.0
        pulse_ptp = 0.0

    return {
        "spo2_min": spo2_min,
        "spo2_mean": spo2_mean,
        "spo2_std": spo2_std,
        "spo2_ptp": spo2_ptp,
        "spo2_drop_from_baseline": spo2_drop_from_baseline,
        "spo2_time_below_90": spo2_time_below_90,
        "spo2_slope": spo2_slope,
        "spo2_lag_min": spo2_lag_min,
        "spo2_lag_drop": spo2_lag_drop,
        "spo2_ods_3pct": spo2_ods_3pct,
        "pulse_mean": pulse_mean,
        "pulse_std": pulse_std,
        "pulse_ptp": pulse_ptp,
    }


def _safe_pearson_corr(
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
) -> float:
    """Compute zero-lag Pearson correlation in ``[-1, 1]``, returning ``0.0`` on flat/short arrays."""
    n = min(int(x.size), int(y.size))
    if n < 2:
        return 0.0
    x_arr = np.asarray(x[:n], dtype=np.float64)
    y_arr = np.asarray(y[:n], dtype=np.float64)
    x_centered = x_arr - np.mean(x_arr)
    y_centered = y_arr - np.mean(y_arr)
    denom = float(np.sqrt(np.sum(np.square(x_centered)) * np.sum(np.square(y_centered))))
    if denom <= _EPSILON:
        return 0.0
    corr = float(np.sum(x_centered * y_centered) / denom)
    return float(np.clip(corr, -1.0, 1.0))


def _compute_band_power(
    freqs: npt.NDArray[np.float64],
    psd: npt.NDArray[np.float64],
    f_low: float,
    f_high: float,
) -> float:
    """Integrate PSD over ``[f_low, f_high]`` using trapezoidal integration or single-bin df."""
    mask = (freqs >= f_low) & (freqs <= f_high)
    n_bins = int(np.count_nonzero(mask))
    if n_bins >= 2:
        power = float(np.trapezoid(psd[mask], freqs[mask]))
        return max(0.0, power)
    if n_bins == 1 and freqs.size >= 2:
        df = float(freqs[1] - freqs[0])
        return max(0.0, float(psd[mask][0]) * df)
    return 0.0


def extract_respiratory_features(
    w_start: float,
    w_end: float,
    window_idx: int,
    signals: Mapping[str, npt.NDArray[np.floating[Any]]],
    sampling_rates: Mapping[str, float],
    context: RecordingContext,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Extract per-channel respiratory features, 5-minute baseline ratios, and paradox metrics.

    Computes ``std``, ``rms``, ``ptp``, ``iqr``, breathing-band Welch power
    (``resp_band_hz``), and ``ratio_to_baseline`` (``window_rms / baseline_5min_rms``)
    for ``Flow``, ``Sum``, ``ribcage``, and ``abdo``, plus cross-channel paradox
    features (``rib_abdo_corr``, ``rib_abdo_rms_ratio``, ``flow_sum_corr``).

    Args:
        w_start: Window start time in seconds.
        w_end: Window end time in seconds.
        window_idx: Zero-based Window index into ``context.resp_baseline_rms``.
        signals: Mapping of canonical channel name to 1D filtered waveform.
        sampling_rates: Mapping of canonical channel name to sampling rate in Hz.
        context: Pre-computed ``RecordingContext`` with 5-minute rolling baselines.
        config: Pipeline configuration dictionary loaded from ``config.yaml``.

    Returns:
        Dictionary of 27 finite ``float`` respiratory features.
    """
    feat_cfg: Mapping[str, Any] = config["features"]
    window_sec = float(config["windowing"]["window_sec"])
    resp_band = feat_cfg["resp_band_hz"]
    f_low = float(resp_band[0])
    f_high = float(resp_band[1])

    out: dict[str, float] = {}
    win_slices: dict[str, npt.NDArray[np.float64]] = {}
    rms_by_ch: dict[str, float] = {}

    for ch in _RESPIRATORY_CHANNELS:
        prefix = ch.lower()
        fs_ch = float(sampling_rates[ch])
        ch_win = np.asarray(
            slice_channel_window(signals[ch], fs_ch, w_start, w_end), dtype=np.float64
        )
        win_slices[ch] = ch_win

        if ch_win.size > 0:
            ch_std = float(np.std(ch_win))
            ch_rms = _compute_rms(ch_win)
            ch_ptp = float(np.ptp(ch_win))
            q75, q25 = np.percentile(ch_win, [75.0, 25.0])
            ch_iqr = float(q75 - q25)
            if ch_win.size >= 8 and fs_ch > 0.0:
                nperseg = min(int(ch_win.size), max(8, int(round(window_sec * fs_ch))))
                freqs, psd = welch(ch_win, fs=fs_ch, nperseg=nperseg)
                resp_power = _compute_band_power(
                    np.asarray(freqs, dtype=np.float64),
                    np.asarray(psd, dtype=np.float64),
                    f_low,
                    f_high,
                )
            else:
                resp_power = 0.0
        else:
            ch_std = 0.0
            ch_rms = 0.0
            ch_ptp = 0.0
            ch_iqr = 0.0
            resp_power = 0.0

        rms_by_ch[ch] = ch_rms
        base_arr = context.resp_baseline_rms.get(ch)
        if base_arr is not None and base_arr.size > 0:
            clamped_idx = min(max(window_idx, 0), int(base_arr.size) - 1)
            base_rms = float(base_arr[clamped_idx])
        else:
            base_rms = ch_rms

        ratio_to_base = (
            float(np.clip(ch_rms / (base_rms + _EPSILON), 0.0, _RATIO_CLIP_MAX))
            if base_rms > _EPSILON
            else 1.0
        )

        out[f"{prefix}_std"] = ch_std
        out[f"{prefix}_rms"] = ch_rms
        out[f"{prefix}_ptp"] = ch_ptp
        out[f"{prefix}_iqr"] = ch_iqr
        out[f"{prefix}_resp_band_power"] = resp_power
        out[f"{prefix}_ratio_to_baseline"] = ratio_to_base

    out["rib_abdo_corr"] = _safe_pearson_corr(win_slices["ribcage"], win_slices["abdo"])
    abdo_rms = rms_by_ch["abdo"]
    out["rib_abdo_rms_ratio"] = (
        float(np.clip(rms_by_ch["ribcage"] / (abdo_rms + _EPSILON), 0.0, _RATIO_CLIP_MAX))
        if abdo_rms > _EPSILON
        else 1.0
    )
    out["flow_sum_corr"] = _safe_pearson_corr(win_slices["Flow"], win_slices["Sum"])
    return out


def _safe_skew_and_kurtosis(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Compute zero-guarded standardized skewness and excess kurtosis in ``float64``."""
    if values.size < 3:
        return 0.0, 0.0
    std = float(np.std(values))
    if std <= _EPSILON:
        return 0.0, 0.0
    z = (values - float(np.mean(values))) / std
    skew = float(np.mean(np.power(z, 3)))
    kurt = float(np.mean(np.power(z, 4)) - 3.0)
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt):
        kurt = 0.0
    return skew, kurt


def _compute_zcr(values: npt.NDArray[np.float64]) -> float:
    """Compute zero-crossing rate across consecutive samples."""
    if values.size < 2:
        return 0.0
    signs = values >= 0.0
    crossings = np.count_nonzero(signs[1:] != signs[:-1])
    return float(crossings / (values.size - 1))


def extract_ecg_hrv_features(
    w_start: float,
    w_end: float,
    ecg_signal: npt.NDArray[np.floating[Any]],
    fs_ecg: float,
    context: RecordingContext,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Extract 30 s ECG waveform stats, local RR/EDR metrics, and 120 s centered PSD HRV features.

    Args:
        w_start: Window start time in seconds.
        w_end: Window end time in seconds.
        ecg_signal: Continuous filtered 1D ``ECG`` waveform.
        fs_ecg: ``ECG`` sampling rate in Hz.
        context: Pre-computed ``RecordingContext`` containing R-peaks and RR intervals.
        config: Pipeline configuration dictionary loaded from ``config.yaml``.

    Returns:
        Dictionary of 21 finite ``float`` ECG and HRV features.
    """
    feat_cfg: Mapping[str, Any] = config["features"]
    hrv_context_sec = float(feat_cfg["hrv_context_sec"])
    hrv_bands: Mapping[str, Sequence[float]] = feat_cfg["hrv_bands_hz"]

    ecg_win = np.asarray(
        slice_channel_window(ecg_signal, fs_ecg, w_start, w_end), dtype=np.float64
    )
    if ecg_win.size > 0:
        ecg_mean = float(np.mean(ecg_win))
        ecg_std = float(np.std(ecg_win))
        ecg_skew, ecg_kurtosis = _safe_skew_and_kurtosis(ecg_win)
        ecg_ptp = float(np.ptp(ecg_win))
        ecg_rms = _compute_rms(ecg_win)
        q75, q25 = np.percentile(ecg_win, [75.0, 25.0])
        ecg_iqr = float(q75 - q25)
        ecg_zcr = _compute_zcr(ecg_win)
    else:
        ecg_mean = 0.0
        ecg_std = 0.0
        ecg_skew = 0.0
        ecg_kurtosis = 0.0
        ecg_ptp = 0.0
        ecg_rms = 0.0
        ecg_iqr = 0.0
        ecg_zcr = 0.0

    in_win_rr_mask = (
        context.rr_valid_mask
        & (context.rr_mid_times_sec >= w_start)
        & (context.rr_mid_times_sec <= w_end)
    )
    rr_local = context.rr_intervals_ms[in_win_rr_mask]

    if rr_local.size > 0:
        hr_local = _MS_PER_MINUTE / rr_local
        hr_mean = float(np.mean(hr_local))
        hr_std = float(np.std(hr_local)) if hr_local.size >= 2 else 0.0
        rr_mean = float(np.mean(rr_local))
        sdnn = float(np.std(rr_local)) if rr_local.size >= 2 else 0.0
        rr_min = float(np.min(rr_local))
        rr_max = float(np.max(rr_local))
        if rr_local.size >= 2:
            diffs = np.diff(rr_local)
            rmssd = float(np.sqrt(np.mean(np.square(diffs))))
            pnn50 = float(np.mean(np.abs(diffs) > _PNN50_THRESHOLD_MS))
        else:
            rmssd = 0.0
            pnn50 = 0.0
    else:
        hr_mean = 0.0
        hr_std = 0.0
        rr_mean = 0.0
        sdnn = 0.0
        rmssd = 0.0
        pnn50 = 0.0
        rr_min = 0.0
        rr_max = 0.0

    in_win_peaks_mask = (context.rpeak_times_sec >= w_start) & (
        context.rpeak_times_sec <= w_end
    )
    win_peak_indices = context.rpeak_indices[in_win_peaks_mask]
    win_peak_indices = win_peak_indices[
        (win_peak_indices >= 0) & (win_peak_indices < ecg_signal.size)
    ]
    if win_peak_indices.size >= 2:
        r_amps = np.asarray(ecg_signal[win_peak_indices], dtype=np.float64)
        edr_ptp = float(np.ptp(r_amps))
    else:
        edr_ptp = 0.0

    # Centered 120s context spectral HRV via 4 Hz tachogram interpolation + Welch PSD
    w_mid = 0.5 * (w_start + w_end)
    ctx_start = w_mid - 0.5 * hrv_context_sec
    ctx_end = w_mid + 0.5 * hrv_context_sec
    ctx_mask = (
        context.rr_valid_mask
        & (context.rr_mid_times_sec >= ctx_start)
        & (context.rr_mid_times_sec <= ctx_end)
    )
    ctx_times = context.rr_mid_times_sec[ctx_mask]
    ctx_rr = context.rr_intervals_ms[ctx_mask]

    vlf_power = 0.0
    lf_power = 0.0
    hf_power = 0.0
    lf_hf_ratio = 0.0

    if ctx_rr.size >= _HRV_MIN_VALID_INTERVALS and (ctx_times[-1] - ctx_times[0]) >= 10.0:
        grid_step = 1.0 / _HRV_INTERP_FS_HZ
        t_grid = np.arange(ctx_times[0], ctx_times[-1], grid_step, dtype=np.float64)
        if t_grid.size >= 16:
            rr_interp = np.interp(t_grid, ctx_times, ctx_rr)
            rr_detrended = rr_interp - float(np.mean(rr_interp))
            if float(np.std(rr_detrended)) > _EPSILON:
                nperseg = int(rr_detrended.size)
                freqs, psd = welch(rr_detrended, fs=_HRV_INTERP_FS_HZ, nperseg=nperseg)
                f_arr = np.asarray(freqs, dtype=np.float64)
                p_arr = np.asarray(psd, dtype=np.float64)
                vlf_band = hrv_bands["vlf"]
                lf_band = hrv_bands["lf"]
                hf_band = hrv_bands["hf"]
                vlf_power = _compute_band_power(f_arr, p_arr, float(vlf_band[0]), float(vlf_band[1]))
                lf_power = _compute_band_power(f_arr, p_arr, float(lf_band[0]), float(lf_band[1]))
                hf_power = _compute_band_power(f_arr, p_arr, float(hf_band[0]), float(hf_band[1]))
                lf_hf_ratio = (
                    float(np.clip(lf_power / (hf_power + _EPSILON), 0.0, _SPECTRAL_RATIO_CLIP_MAX))
                    if hf_power > _EPSILON
                    else 0.0
                )

    return {
        "ecg_mean": ecg_mean,
        "ecg_std": ecg_std,
        "ecg_skew": ecg_skew,
        "ecg_kurtosis": ecg_kurtosis,
        "ecg_ptp": ecg_ptp,
        "ecg_rms": ecg_rms,
        "ecg_iqr": ecg_iqr,
        "ecg_zcr": ecg_zcr,
        "hr_mean": hr_mean,
        "hr_std": hr_std,
        "rr_mean": rr_mean,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "rr_min": rr_min,
        "rr_max": rr_max,
        "edr_ptp": edr_ptp,
        "vlf_power": vlf_power,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
    }


def extract_sound_bodypos_features(
    w_start: float,
    w_end: float,
    sound_signal: npt.NDArray[np.floating[Any]],
    bodypos_signal: npt.NDArray[np.floating[Any]],
    fs_sound: float,
    fs_bodypos: float,
) -> dict[str, float]:
    """Extract snore envelope statistics from ``Sound`` and posture features from ``BodyPos``.

    Args:
        w_start: Window start time in seconds.
        w_end: Window end time in seconds.
        sound_signal: Continuous filtered 1D ``Sound`` signal.
        bodypos_signal: Continuous 1D ``BodyPos`` signal.
        fs_sound: ``Sound`` sampling rate in Hz.
        fs_bodypos: ``BodyPos`` sampling rate in Hz.

    Returns:
        Dictionary of 7 finite ``float`` features for ``Sound`` and ``BodyPos``.
    """
    sound_win = np.asarray(
        slice_channel_window(sound_signal, fs_sound, w_start, w_end), dtype=np.float64
    )
    bodypos_win = np.asarray(
        slice_channel_window(bodypos_signal, fs_bodypos, w_start, w_end), dtype=np.float64
    )

    if sound_win.size > 0:
        abs_sound = np.abs(sound_win)
        sound_rms = _compute_rms(sound_win)
        sound_max = float(np.max(abs_sound))
        sound_std = float(np.std(sound_win))
        sound_crest_factor = (
            float(np.clip(sound_max / (sound_rms + _EPSILON), 0.0, _SPECTRAL_RATIO_CLIP_MAX))
            if sound_rms > _EPSILON
            else 0.0
        )
        burst_threshold = float(np.mean(abs_sound) + _SOUND_BURST_STD_MULT * np.std(abs_sound))
        sound_burst_ratio = (
            float(np.mean(abs_sound > burst_threshold))
            if burst_threshold > _EPSILON
            else 0.0
        )
    else:
        sound_rms = 0.0
        sound_max = 0.0
        sound_std = 0.0
        sound_crest_factor = 0.0
        sound_burst_ratio = 0.0

    if bodypos_win.size > 0:
        rounded_pos = np.rint(bodypos_win).astype(np.int64)
        vals, counts = np.unique(rounded_pos, return_counts=True)
        bodypos_mode = float(vals[int(np.argmax(counts))])
        bodypos_changed = 1.0 if vals.size > 1 else 0.0
    else:
        bodypos_mode = 0.0
        bodypos_changed = 0.0

    return {
        "sound_rms": sound_rms,
        "sound_max": sound_max,
        "sound_std": sound_std,
        "sound_crest_factor": sound_crest_factor,
        "sound_burst_ratio": sound_burst_ratio,
        "bodypos_mode": bodypos_mode,
        "bodypos_changed": bodypos_changed,
    }


def extract_neuro_features(
    w_start: float,
    w_end: float,
    signals: Mapping[str, npt.NDArray[np.floating[Any]]],
    sampling_rates: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Extract EEG relative band powers/entropy, EOG movement/correlation, and EMG tone features.

    Args:
        w_start: Window start time in seconds.
        w_end: Window end time in seconds.
        signals: Mapping of canonical channel name to 1D filtered waveform.
        sampling_rates: Mapping of canonical channel name to sampling rate in Hz.
        config: Pipeline configuration dictionary loaded from ``config.yaml``.

    Returns:
        Dictionary of 20 finite ``float`` neurological features.
    """
    eeg_bands: Mapping[str, Sequence[float]] = config["features"]["eeg_bands_hz"]
    out: dict[str, float] = {}

    for ch in _EEG_CHANNELS:
        prefix = ch.lower()
        fs_eeg = float(sampling_rates[ch])
        eeg_win = np.asarray(
            slice_channel_window(signals[ch], fs_eeg, w_start, w_end), dtype=np.float64
        )
        band_powers: dict[str, float] = {b: 0.0 for b in _EEG_BAND_KEYS}
        spectral_entropy = 0.0

        if eeg_win.size >= 16 and fs_eeg > 0.0 and float(np.std(eeg_win)) > _EPSILON:
            nperseg = min(int(eeg_win.size), max(16, int(round(4.0 * fs_eeg))))
            freqs, psd = welch(eeg_win, fs=fs_eeg, nperseg=nperseg)
            f_arr = np.asarray(freqs, dtype=np.float64)
            p_arr = np.asarray(psd, dtype=np.float64)

            for band_name in _EEG_BAND_KEYS:
                bounds = eeg_bands[band_name]
                band_powers[band_name] = _compute_band_power(
                    f_arr, p_arr, float(bounds[0]), float(bounds[1])
                )

            f_min = float(eeg_bands["delta"][0])
            f_max = float(eeg_bands["beta"][1])
            band_mask = (f_arr >= f_min) & (f_arr <= f_max)
            p_sub = p_arr[band_mask]
            p_sum = float(np.sum(p_sub))
            if p_sub.size >= 2 and p_sum > _EPSILON:
                probs = p_sub / p_sum
                pos_probs = probs[probs > 0.0]
                raw_ent = -float(np.sum(pos_probs * np.log2(pos_probs)))
                spectral_entropy = float(raw_ent / np.log2(float(p_sub.size)))

        total_power = sum(band_powers.values())
        for band_name in _EEG_BAND_KEYS:
            out[f"{prefix}_{band_name}_rel"] = (
                float(band_powers[band_name] / total_power)
                if total_power > _EPSILON
                else 0.0
            )

        delta_rel = out[f"{prefix}_delta_rel"]
        alpha_rel = out[f"{prefix}_alpha_rel"]
        out[f"{prefix}_alpha_delta_ratio"] = (
            float(np.clip(alpha_rel / (delta_rel + _EPSILON), 0.0, _SPECTRAL_RATIO_CLIP_MAX))
            if total_power > _EPSILON
            else 0.0
        )
        out[f"{prefix}_spectral_entropy"] = spectral_entropy

    fs_left = float(sampling_rates["Lefteye"])
    fs_right = float(sampling_rates["RightEye"])
    left_win = np.asarray(
        slice_channel_window(signals["Lefteye"], fs_left, w_start, w_end), dtype=np.float64
    )
    right_win = np.asarray(
        slice_channel_window(signals["RightEye"], fs_right, w_start, w_end), dtype=np.float64
    )

    left_std = float(np.std(left_win)) if left_win.size > 0 else 0.0
    right_std = float(np.std(right_win)) if right_win.size > 0 else 0.0
    left_ptp = float(np.ptp(left_win)) if left_win.size > 0 else 0.0
    right_ptp = float(np.ptp(right_win)) if right_win.size > 0 else 0.0

    out["eog_std"] = 0.5 * (left_std + right_std)
    out["eog_ptp"] = 0.5 * (left_ptp + right_ptp)
    out["eog_corr"] = _safe_pearson_corr(left_win, right_win)

    fs_emg = float(sampling_rates["EMG"])
    emg_win = np.asarray(
        slice_channel_window(signals["EMG"], fs_emg, w_start, w_end), dtype=np.float64
    )
    if emg_win.size > 0:
        emg_rms = _compute_rms(emg_win)
        emg_std = float(np.std(emg_win))
        emg_median_freq = 0.0
        if emg_win.size >= 16 and fs_emg > 0.0 and emg_std > _EPSILON:
            nperseg = min(int(emg_win.size), max(16, int(round(4.0 * fs_emg))))
            freqs, psd = welch(emg_win, fs=fs_emg, nperseg=nperseg)
            f_arr = np.asarray(freqs, dtype=np.float64)
            p_arr = np.asarray(psd, dtype=np.float64)
            cumsum = np.cumsum(p_arr)
            total_psd = float(cumsum[-1]) if cumsum.size > 0 else 0.0
            if total_psd > _EPSILON:
                med_idx = int(np.searchsorted(cumsum, 0.5 * total_psd))
                med_idx = min(max(med_idx, 0), int(f_arr.size) - 1)
                emg_median_freq = float(f_arr[med_idx])
    else:
        emg_rms = 0.0
        emg_std = 0.0
        emg_median_freq = 0.0

    out["emg_rms"] = emg_rms
    out["emg_std"] = emg_std
    out["emg_median_freq"] = emg_median_freq
    return out


_METADATA_COLUMNS: tuple[str, ...] = (
    "recording_id",
    "window_idx",
    "start_sec",
    "end_sec",
    "sleep_stage",
    "is_wake",
    "overlap_sec",
    "label",
)

_FEATURE_COLUMNS: tuple[str, ...] = (
    # ECG & HRV (21)
    "ecg_mean",
    "ecg_std",
    "ecg_skew",
    "ecg_kurtosis",
    "ecg_ptp",
    "ecg_rms",
    "ecg_iqr",
    "ecg_zcr",
    "hr_mean",
    "hr_std",
    "rr_mean",
    "sdnn",
    "rmssd",
    "pnn50",
    "rr_min",
    "rr_max",
    "edr_ptp",
    "vlf_power",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    # SpO2 & Pulse (13)
    "spo2_min",
    "spo2_mean",
    "spo2_std",
    "spo2_ptp",
    "spo2_drop_from_baseline",
    "spo2_time_below_90",
    "spo2_slope",
    "spo2_lag_min",
    "spo2_lag_drop",
    "spo2_ods_3pct",
    "pulse_mean",
    "pulse_std",
    "pulse_ptp",
    # Respiratory & Paradox (27)
    "flow_std",
    "flow_rms",
    "flow_ptp",
    "flow_iqr",
    "flow_resp_band_power",
    "flow_ratio_to_baseline",
    "sum_std",
    "sum_rms",
    "sum_ptp",
    "sum_iqr",
    "sum_resp_band_power",
    "sum_ratio_to_baseline",
    "ribcage_std",
    "ribcage_rms",
    "ribcage_ptp",
    "ribcage_iqr",
    "ribcage_resp_band_power",
    "ribcage_ratio_to_baseline",
    "abdo_std",
    "abdo_rms",
    "abdo_ptp",
    "abdo_iqr",
    "abdo_resp_band_power",
    "abdo_ratio_to_baseline",
    "rib_abdo_corr",
    "rib_abdo_rms_ratio",
    "flow_sum_corr",
    # Sound & BodyPos (7)
    "sound_rms",
    "sound_max",
    "sound_std",
    "sound_crest_factor",
    "sound_burst_ratio",
    "bodypos_mode",
    "bodypos_changed",
    # Neuro: EEG, EOG, EMG (20)
    "c3a2_delta_rel",
    "c3a2_theta_rel",
    "c3a2_alpha_rel",
    "c3a2_sigma_rel",
    "c3a2_beta_rel",
    "c3a2_alpha_delta_ratio",
    "c3a2_spectral_entropy",
    "c4a1_delta_rel",
    "c4a1_theta_rel",
    "c4a1_alpha_rel",
    "c4a1_sigma_rel",
    "c4a1_beta_rel",
    "c4a1_alpha_delta_ratio",
    "c4a1_spectral_entropy",
    "eog_std",
    "eog_ptp",
    "eog_corr",
    "emg_rms",
    "emg_std",
    "emg_median_freq",
)


def extract_recording_features(
    recording_data: RecordingData,
    filtered_signals: Mapping[str, npt.NDArray[np.floating[Any]]],
    windows: Sequence[WindowRecord],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Extract the full tabular Feature vector DataFrame for all Windows of a Recording.

    Precomputes continuous Recording-level context (R-peaks, physiological RR intervals,
    and 5-minute rolling baselines), extracts all 5 modality feature groups per Window,
    enforces a deterministic column order, casts every feature column to ``float32``,
    and replaces any residual non-finite value with ``0.0``.

    Args:
        recording_data: Loaded ``RecordingData`` containing ``sampling_rates`` and metadata.
        filtered_signals: Mapping of canonical channel name to 1D filtered ``float32`` signal.
        windows: Sequence of ``WindowRecord`` metadata entries for the Recording.
        config: Pipeline configuration dictionary loaded from ``config.yaml``.

    Returns:
        ``pd.DataFrame`` with 8 metadata columns followed by 88 ``float32`` feature
        columns, guaranteed free of ``NaN`` and ``Inf`` values.
    """
    rates = recording_data.sampling_rates
    context = precompute_recording_context(filtered_signals, rates, config)

    n_wins = len(windows)
    feat_matrix = np.zeros((n_wins, len(_FEATURE_COLUMNS)), dtype=np.float32)

    ecg_sig = filtered_signals["ECG"]
    spo2_sig = filtered_signals["SpO2"]
    pulse_sig = filtered_signals["Pulse"]
    sound_sig = filtered_signals["Sound"]
    bodypos_sig = filtered_signals["BodyPos"]

    fs_ecg = float(rates["ECG"])
    fs_spo2 = float(rates["SpO2"])
    fs_pulse = float(rates["Pulse"])
    fs_sound = float(rates["Sound"])
    fs_bodypos = float(rates["BodyPos"])

    for row_idx, win in enumerate(windows):
        w_start = win.start_sec
        w_end = win.end_sec
        w_idx = win.window_idx

        combined: dict[str, float] = {}
        combined.update(
            extract_ecg_hrv_features(w_start, w_end, ecg_sig, fs_ecg, context, config)
        )
        combined.update(
            extract_spo2_pulse_features(
                w_start, w_end, w_idx, spo2_sig, pulse_sig, fs_spo2, fs_pulse, context, config
            )
        )
        combined.update(
            extract_respiratory_features(
                w_start, w_end, w_idx, filtered_signals, rates, context, config
            )
        )
        combined.update(
            extract_sound_bodypos_features(
                w_start, w_end, sound_sig, bodypos_sig, fs_sound, fs_bodypos
            )
        )
        combined.update(
            extract_neuro_features(w_start, w_end, filtered_signals, rates, config)
        )

        for col_idx, col_name in enumerate(_FEATURE_COLUMNS):
            val = float(combined[col_name])
            feat_matrix[row_idx, col_idx] = (
                np.float32(val) if np.isfinite(val) else np.float32(0.0)
            )

    np.nan_to_num(feat_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    meta_df = pd.DataFrame(
        {
            "recording_id": pd.Series(
                [w.recording_id for w in windows], dtype="string" if n_wins == 0 else "object"
            ),
            "window_idx": np.asarray([w.window_idx for w in windows], dtype=np.int64),
            "start_sec": np.asarray([w.start_sec for w in windows], dtype=np.float64),
            "end_sec": np.asarray([w.end_sec for w in windows], dtype=np.float64),
            "sleep_stage": np.asarray([w.sleep_stage for w in windows], dtype=np.int8),
            "is_wake": np.asarray([w.is_wake for w in windows], dtype=np.bool_),
            "overlap_sec": np.asarray([w.overlap_sec for w in windows], dtype=np.float64),
            "label": np.asarray([w.label for w in windows], dtype=np.int8),
        },
        columns=list(_METADATA_COLUMNS),
    )

    feat_df = pd.DataFrame(feat_matrix, columns=list(_FEATURE_COLUMNS), dtype=np.float32)
    return pd.concat([meta_df, feat_df], axis=1)



