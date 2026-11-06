"""Tests for multi-modal physiological feature extraction."""

from pathlib import Path

import numpy as np
import pytest

from src.pipeline.features import (
    RecordingContext,
    extract_ecg_hrv_features,
    extract_neuro_features,
    extract_recording_features,
    extract_respiratory_features,
    extract_sound_bodypos_features,
    extract_spo2_pulse_features,
    precompute_recording_context,
)
from src.pipeline.filters import filter_recording_signals
from src.pipeline.loader import load_config, load_recording
from src.pipeline.windowing import create_windows_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _make_synthetic_signals(duration_sec: float = 360.0) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Build synthetic 14-channel signals at canonical UCDDB sampling rates."""
    config = load_config(CONFIG_PATH)
    rates: dict[str, float] = {k: float(v) for k, v in config["channels"]["sampling_rates"].items()}
    signals: dict[str, np.ndarray] = {}
    for ch, fs in rates.items():
        n = int(round(duration_sec * fs))
        t = np.arange(n, dtype=np.float64) / fs
        if ch == "SpO2":
            signals[ch] = np.full(n, 97.0, dtype=np.float32)
        elif ch == "Pulse":
            signals[ch] = np.full(n, 65.0, dtype=np.float32)
        elif ch == "BodyPos":
            signals[ch] = np.full(n, 2.0, dtype=np.float32)
        elif ch == "ECG":
            # Periodic R-like pulses around 1 Hz with respiratory sinus arrhythmia modulation (0.1 Hz LF)
            sig = np.zeros(n, dtype=np.float64)
            t_cur = 0.5
            while t_cur < duration_sec - 0.5:
                idx = int(round(t_cur * fs))
                if 0 <= idx < n:
                    sig[idx] = 1.5
                rr_sec = 1.0 + 0.08 * np.sin(2.0 * np.pi * 0.1 * t_cur)
                t_cur += rr_sec
            signals[ch] = sig.astype(np.float32)
        else:
            signals[ch] = np.sin(2.0 * np.pi * 0.25 * t).astype(np.float32)
    return signals, rates


def test_spo2_lagged_desaturation_captures_post_window_drop() -> None:
    """Simulated +10 s delayed SpO2 drop of 5% after Window end gives spo2_lag_drop >= 4.5 and spo2_ods_3pct == 1.0."""
    config = load_config(CONFIG_PATH)
    signals, rates = _make_synthetic_signals(duration_sec=360.0)
    fs_spo2 = rates["SpO2"]

    # Window 5 covers [150.0, 180.0]. Inject a 5% drop (97% -> 92%) at +10s after Window end: [188.0, 193.0]
    drop_start = int(round(188.0 * fs_spo2))
    drop_end = int(round(193.0 * fs_spo2))
    signals["SpO2"][drop_start:drop_end] = 92.0

    context = precompute_recording_context(signals, rates, config)
    assert isinstance(context, RecordingContext)
    assert context.spo2_baseline_p90[5] == pytest.approx(97.0, abs=0.1)

    feats = extract_spo2_pulse_features(
        w_start=150.0,
        w_end=180.0,
        window_idx=5,
        spo2_signal=signals["SpO2"],
        pulse_signal=signals["Pulse"],
        fs_spo2=fs_spo2,
        fs_pulse=rates["Pulse"],
        context=context,
        config=config,
    )

    assert feats["spo2_min"] == pytest.approx(97.0)
    assert feats["spo2_drop_from_baseline"] == pytest.approx(0.0, abs=0.1)
    assert feats["spo2_lag_min"] == pytest.approx(92.0)
    assert feats["spo2_lag_drop"] >= 4.5
    assert feats["spo2_ods_3pct"] == 1.0
    assert all(np.isfinite(val) for val in feats.values())


def test_respiratory_paradox_detects_antiphase_ribcage_and_abdo() -> None:
    """Anti-phase synthetic ribcage and abdo sine waves produce rib_abdo_corr < -0.95."""
    config = load_config(CONFIG_PATH)
    signals, rates = _make_synthetic_signals(duration_sec=300.0)
    fs_resp = rates["ribcage"]
    n = signals["ribcage"].size
    t = np.arange(n, dtype=np.float64) / fs_resp

    # 0.25 Hz breathing (inside 0.1-0.5 Hz respiratory band) with anti-phase ribcage and abdo in [60.0, 90.0]
    signals["ribcage"] = np.sin(2.0 * np.pi * 0.25 * t).astype(np.float32)
    signals["abdo"] = (-np.sin(2.0 * np.pi * 0.25 * t)).astype(np.float32)
    # Attenuate Flow in Window 2 [60.0, 90.0] to 20% of baseline to verify ratio_to_baseline
    w2_start = int(round(60.0 * fs_resp))
    w2_end = int(round(90.0 * fs_resp))
    signals["Flow"][w2_start:w2_end] *= 0.2

    context = precompute_recording_context(signals, rates, config)
    feats = extract_respiratory_features(
        w_start=60.0,
        w_end=90.0,
        window_idx=2,
        signals=signals,
        sampling_rates=rates,
        context=context,
        config=config,
    )

    assert feats["rib_abdo_corr"] < -0.95
    assert feats["rib_abdo_rms_ratio"] == pytest.approx(1.0, rel=1e-2)
    assert feats["flow_ratio_to_baseline"] < 0.35
    assert feats["ribcage_resp_band_power"] > 0.0
    assert len(feats) == 27
    assert all(np.isfinite(val) for val in feats.values())


def test_ecg_hrv_sound_bodypos_and_neuro_features_on_synthetic_and_flatline_signals() -> None:
    """ECG/HRV (30s + 120s context), Sound/BodyPos, and Neuro extractors return finite expected features."""
    config = load_config(CONFIG_PATH)
    signals, rates = _make_synthetic_signals(duration_sec=240.0)

    # Inject a posture change inside [60.0, 90.0] and a snore burst in Sound
    fs_bp = rates["BodyPos"]
    bp_start = int(round(60.0 * fs_bp))
    bp_mid = int(round(75.0 * fs_bp))
    bp_end = int(round(90.0 * fs_bp))
    signals["BodyPos"][bp_start:bp_mid] = 1.0
    signals["BodyPos"][bp_mid:bp_end] = 3.0
    signals["Sound"][bp_mid : bp_mid + 8] = 10.0

    # Inject 10 Hz alpha wave on C3A2 and 2 Hz delta wave on C4A1
    fs_eeg = rates["C3A2"]
    t_eeg = np.arange(signals["C3A2"].size, dtype=np.float64) / fs_eeg
    signals["C3A2"] = np.sin(2.0 * np.pi * 10.0 * t_eeg).astype(np.float32)
    signals["C4A1"] = np.sin(2.0 * np.pi * 2.0 * t_eeg).astype(np.float32)

    context = precompute_recording_context(signals, rates, config)
    ecg_feats = extract_ecg_hrv_features(
        w_start=60.0,
        w_end=90.0,
        ecg_signal=signals["ECG"],
        fs_ecg=rates["ECG"],
        context=context,
        config=config,
    )
    assert len(ecg_feats) == 21
    assert ecg_feats["rr_mean"] == pytest.approx(1000.0, rel=2e-2)
    assert ecg_feats["hr_mean"] == pytest.approx(60.0, rel=2e-2)
    assert ecg_feats["lf_power"] > 0.0
    assert all(np.isfinite(val) for val in ecg_feats.values())

    sb_feats = extract_sound_bodypos_features(
        w_start=60.0,
        w_end=90.0,
        sound_signal=signals["Sound"],
        bodypos_signal=signals["BodyPos"],
        fs_sound=rates["Sound"],
        fs_bodypos=rates["BodyPos"],
    )
    assert len(sb_feats) == 7
    assert sb_feats["bodypos_changed"] == 1.0
    assert sb_feats["sound_burst_ratio"] > 0.0
    assert all(np.isfinite(val) for val in sb_feats.values())

    neuro_feats = extract_neuro_features(
        w_start=60.0,
        w_end=90.0,
        signals=signals,
        sampling_rates=rates,
        config=config,
    )
    assert len(neuro_feats) == 20
    assert "eog_std" in neuro_feats
    assert "eog_ptp" in neuro_feats
    assert "eog_corr" in neuro_feats
    assert neuro_feats["c3a2_alpha_rel"] > 0.7
    assert neuro_feats["c4a1_delta_rel"] > 0.7
    assert neuro_feats["c3a2_alpha_delta_ratio"] > neuro_feats["c4a1_alpha_delta_ratio"]
    assert all(np.isfinite(val) for val in neuro_feats.values())

    # Flatline signals must never produce NaN or Inf
    flat_signals = {ch: np.zeros_like(arr) for ch, arr in signals.items()}
    flat_ctx = precompute_recording_context(flat_signals, rates, config)
    flat_ecg = extract_ecg_hrv_features(60.0, 90.0, flat_signals["ECG"], rates["ECG"], flat_ctx, config)
    flat_neuro = extract_neuro_features(60.0, 90.0, flat_signals, rates, config)
    flat_sb = extract_sound_bodypos_features(
        60.0, 90.0, flat_signals["Sound"], flat_signals["BodyPos"], rates["Sound"], rates["BodyPos"]
    )
    assert all(np.isfinite(val) for val in flat_ecg.values())
    assert all(np.isfinite(val) for val in flat_neuro.values())
    assert all(np.isfinite(val) for val in flat_sb.values())


def test_extract_recording_features_deterministic_and_finite_on_ucddb002() -> None:
    """Full Recording ucddb002 produces deterministic columns, float32 feature dtypes, and zero NaN/Inf."""
    config = load_config(CONFIG_PATH)
    rec = load_recording("ucddb002", config)
    filtered = filter_recording_signals(rec.signals, rec.sampling_rates, config)
    windows = create_windows_metadata(rec, config)

    df = extract_recording_features(rec, filtered, windows, config)

    assert len(df) == len(windows) == 748
    expected_meta_cols = [
        "recording_id",
        "window_idx",
        "start_sec",
        "end_sec",
        "sleep_stage",
        "is_wake",
        "overlap_sec",
        "label",
    ]
    assert list(df.columns[: len(expected_meta_cols)]) == expected_meta_cols

    feature_cols = list(df.columns[len(expected_meta_cols) :])
    assert len(feature_cols) >= 85
    for col in feature_cols:
        assert df[col].dtype == np.float32

    feat_matrix = df[feature_cols].to_numpy()
    assert not np.isnan(feat_matrix).any()
    assert not np.isinf(feat_matrix).any()

    # Deterministic column ordering across calls (including empty window list)
    empty_df = extract_recording_features(rec, filtered, [], config)
    assert list(empty_df.columns) == list(df.columns)
    assert len(empty_df) == 0



