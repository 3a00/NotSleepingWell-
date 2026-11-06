"""Modality-specific signal filtering and artifact cleaning for PSG Recordings."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, medfilt, sosfiltfilt  # type: ignore[import-untyped]

_CHANNEL_FILTER_GROUPS: Mapping[str, str] = {
    "ECG": "ecg",
    "C3A2": "neuro",
    "C4A1": "neuro",
    "Lefteye": "neuro",
    "RightEye": "neuro",
    "EMG": "emg",
    "Flow": "respiratory",
    "Sum": "respiratory",
    "ribcage": "respiratory",
    "abdo": "respiratory",
    "Sound": "sound",
    "SpO2": "spo2",
    "Pulse": "pulse",
    "BodyPos": "bodypos",
}


def _apply_sos_zero_phase(
    signal: npt.NDArray[np.floating[Any]],
    sos: npt.NDArray[np.float64],
) -> npt.NDArray[np.float32]:
    """Run ``sosfiltfilt`` in ``float64`` precision and return ``float32``."""
    filtered = sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))
    return np.asarray(filtered, dtype=np.float32)


def bandpass_filter(
    signal: npt.NDArray[np.floating[Any]],
    fs: float,
    lowcut: float,
    highcut: float,
    order: int,
) -> npt.NDArray[np.float32]:
    """Apply a zero-phase Butterworth bandpass filter to a 1D waveform.

    Uses second-order sections (``output='sos'``) and ``scipy.signal.sosfiltfilt``
    in ``float64`` precision before casting back to ``float32`` to preserve
    numerical stability at narrow normalized cutoffs.

    Args:
        signal: 1D continuous input waveform.
        fs: Sampling rate in Hz.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered 1D ``float32`` waveform of the same shape as ``signal``.

    Raises:
        ValueError: If cutoff frequencies violate ``0 < lowcut < highcut < fs / 2``
            or ``order < 1``.
    """
    nyquist = 0.5 * fs
    if order < 1:
        raise ValueError(f"Filter order must be >= 1, got {order}.")
    if not (0.0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"Bandpass cutoffs must satisfy 0 < lowcut < highcut < fs/2 ({nyquist} Hz); "
            f"got lowcut={lowcut}, highcut={highcut}."
        )

    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return _apply_sos_zero_phase(signal, sos)


def highpass_filter(
    signal: npt.NDArray[np.floating[Any]],
    fs: float,
    lowcut: float,
    order: int,
) -> npt.NDArray[np.float32]:
    """Apply a zero-phase Butterworth highpass filter to a 1D waveform.

    Uses second-order sections (``output='sos'``) and ``scipy.signal.sosfiltfilt``
    in ``float64`` precision before casting back to ``float32``.

    Args:
        signal: 1D continuous input waveform.
        fs: Sampling rate in Hz.
        lowcut: Highpass cutoff frequency in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered 1D ``float32`` waveform of the same shape as ``signal``.

    Raises:
        ValueError: If ``lowcut`` violates ``0 < lowcut < fs / 2`` or ``order < 1``.
    """
    nyquist = 0.5 * fs
    if order < 1:
        raise ValueError(f"Filter order must be >= 1, got {order}.")
    if not (0.0 < lowcut < nyquist):
        raise ValueError(
            f"Highpass cutoff must satisfy 0 < lowcut < fs/2 ({nyquist} Hz); "
            f"got lowcut={lowcut}."
        )

    sos = butter(order, lowcut, btype="highpass", fs=fs, output="sos")
    return _apply_sos_zero_phase(signal, sos)


def clean_interpolated_channel(
    signal: npt.NDArray[np.floating[Any]],
    valid_min: float,
    valid_max: float,
    median_kernel_size: int,
) -> npt.NDArray[np.float32]:
    """Remove hardware dropouts via NaN masking, linear/edge interpolation, and median filtering.

    Marks non-finite samples and values outside ``[valid_min, valid_max]`` as ``NaN``,
    linearly interpolates interior gaps while backward/forward filling leading and
    trailing edge gaps (falling back to the ``[valid_min, valid_max]`` midpoint when
    every sample is invalid), and smooths residual single-sample glitches with
    ``scipy.signal.medfilt``.

    Args:
        signal: 1D physiological channel (e.g., ``SpO2`` or ``Pulse``).
        valid_min: Minimum physiologically plausible value (inclusive).
        valid_max: Maximum physiologically plausible value (inclusive).
        median_kernel_size: Odd positive kernel width for ``scipy.signal.medfilt``.

    Returns:
        Cleaned 1D ``float32`` array with zero ``NaN`` or ``Inf`` values.

    Raises:
        ValueError: If ``valid_min > valid_max`` or ``median_kernel_size`` is not a
            positive odd integer.
    """
    if valid_min > valid_max:
        raise ValueError(
            f"valid_min ({valid_min}) must be <= valid_max ({valid_max})."
        )
    if median_kernel_size < 1 or median_kernel_size % 2 == 0:
        raise ValueError(
            f"median_kernel_size must be a positive odd integer, got {median_kernel_size}."
        )

    cleaned = np.asarray(signal, dtype=np.float64).copy()
    if cleaned.size == 0:
        return np.empty(0, dtype=np.float32)

    invalid_mask = ~np.isfinite(cleaned) | (cleaned < valid_min) | (cleaned > valid_max)
    cleaned[invalid_mask] = np.nan

    valid_indices = np.flatnonzero(~np.isnan(cleaned))
    if valid_indices.size == 0:
        fallback_value = (valid_min + valid_max) / 2.0
        interpolated = np.full(cleaned.shape, fallback_value, dtype=np.float64)
    elif valid_indices.size < cleaned.size:
        sample_indices = np.arange(cleaned.size, dtype=np.float64)
        interpolated = np.interp(
            sample_indices,
            valid_indices.astype(np.float64),
            cleaned[valid_indices],
        )
    else:
        interpolated = cleaned

    if median_kernel_size > 1:
        pad = median_kernel_size // 2
        padded = np.pad(interpolated, pad_width=pad, mode="edge")
        smoothed = medfilt(padded, kernel_size=median_kernel_size)[pad : pad + interpolated.size]
    else:
        smoothed = interpolated

    return np.asarray(smoothed, dtype=np.float32)


def filter_recording_signals(
    signals: Mapping[str, npt.NDArray[np.floating[Any]]],
    sampling_rates: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, npt.NDArray[np.float32]]:
    """Filter and clean all canonical channels of a Recording using ``config['filters']``.

    Routes continuous AC waveforms (``ECG``, neurological, ``EMG``, respiratory,
    ``Sound``) through zero-phase Butterworth ``bandpass`` or ``highpass`` filters
    (reducing a group bandpass to ``highpass_filter`` when a lower-rate channel's
    Nyquist frequency ``fs / 2`` is already at or below ``highcut_hz``, such as
    ``64 Hz`` EOG under the ``0.5-35.0 Hz`` ``neuro`` band), cleans ``SpO2`` and
    ``Pulse`` via dropout interpolation and median filtering, and passes ``BodyPos``
    through unchanged as a ``float32`` copy.

    Args:
        signals: Mapping of canonical channel name to 1D physical-unit waveform.
        sampling_rates: Mapping of canonical channel name to sampling rate in Hz.
        config: Pipeline configuration dictionary loaded from ``config.yaml``.

    Returns:
        Dictionary mapping each channel name to a filtered ``float32`` 1D array
        with identical shape and zero ``NaN`` or ``Inf`` values.

    Raises:
        ValueError: If canonical channels are missing, a channel lacks a filter
            specification or sampling rate, an unsupported filter ``type`` is given,
            or any output contains non-finite values.
    """
    canonical_channels = config["channels"]["canonical"]
    missing_channels = set(canonical_channels) - set(signals.keys())
    if missing_channels:
        raise ValueError(
            f"Missing canonical channels for filtering: {sorted(missing_channels)}"
        )

    filters_cfg: Mapping[str, Any] = config["filters"]
    filtered_signals: dict[str, npt.NDArray[np.float32]] = {}

    for channel_name, raw_signal in signals.items():
        group_key = _CHANNEL_FILTER_GROUPS.get(channel_name)
        if group_key is None or group_key not in filters_cfg:
            raise ValueError(
                f"Channel {channel_name!r} has no configured filter specification."
            )
        if channel_name not in sampling_rates:
            raise ValueError(f"Missing sampling rate for channel {channel_name!r}.")

        spec: Mapping[str, Any] = filters_cfg[group_key]
        filter_type = str(spec["type"])
        fs = float(sampling_rates[channel_name])

        if filter_type == "bandpass":
            lowcut = float(spec["lowcut_hz"])
            highcut = float(spec["highcut_hz"])
            order = int(spec["order"])
            if highcut >= 0.5 * fs:
                out = highpass_filter(
                    signal=raw_signal,
                    fs=fs,
                    lowcut=lowcut,
                    order=order,
                )
            else:
                out = bandpass_filter(
                    signal=raw_signal,
                    fs=fs,
                    lowcut=lowcut,
                    highcut=highcut,
                    order=order,
                )
        elif filter_type == "highpass":
            out = highpass_filter(
                signal=raw_signal,
                fs=fs,
                lowcut=float(spec["lowcut_hz"]),
                order=int(spec["order"]),
            )
        elif filter_type == "interpolate_median":
            out = clean_interpolated_channel(
                signal=raw_signal,
                valid_min=float(spec["valid_min"]),
                valid_max=float(spec["valid_max"]),
                median_kernel_size=int(spec["median_kernel_size"]),
            )
        elif filter_type == "passthrough":
            out = np.asarray(raw_signal, dtype=np.float32).copy()
        else:
            raise ValueError(
                f"Unsupported filter type {filter_type!r} for channel {channel_name!r}."
            )

        if not np.all(np.isfinite(out)):
            raise ValueError(
                f"Filtered output for channel {channel_name!r} contains non-finite values."
            )
        filtered_signals[channel_name] = out

    return filtered_signals
