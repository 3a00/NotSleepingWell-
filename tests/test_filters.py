"""Tests for modality-specific signal filtering and artifact cleaning."""

from pathlib import Path

import numpy as np
import pytest

from src.pipeline.filters import (
    bandpass_filter,
    clean_interpolated_channel,
    filter_recording_signals,
    highpass_filter,
)
from src.pipeline.loader import load_config, load_recording

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def test_bandpass_filter_preserves_passband_and_attenuates_stopbands() -> None:
    """Zero-phase bandpass preserves 10 Hz wave while removing 0.1 Hz wander and 50 Hz noise."""
    fs = 128.0
    duration_sec = 30.0
    t = np.arange(0.0, duration_sec, 1.0 / fs, dtype=np.float64)

    clean_10hz = np.sin(2.0 * np.pi * 10.0 * t)
    wander_0_1hz = 2.0 * np.sin(2.0 * np.pi * 0.1 * t)
    noise_50hz = 1.5 * np.sin(2.0 * np.pi * 50.0 * t)
    composite = (clean_10hz + wander_0_1hz + noise_50hz).astype(np.float32)

    filtered = bandpass_filter(
        signal=composite,
        fs=fs,
        lowcut=0.5,
        highcut=40.0,
        order=4,
    )

    assert filtered.dtype == np.float32
    assert filtered.shape == composite.shape
    assert np.all(np.isfinite(filtered))

    interior = slice(int(5.0 * fs), int(25.0 * fs))
    max_residual = float(np.max(np.abs(filtered[interior] - clean_10hz[interior])))
    assert max_residual < 0.02

    with pytest.raises(ValueError, match="fs/2"):
        bandpass_filter(signal=composite, fs=fs, lowcut=0.5, highcut=64.0, order=4)


def test_highpass_filter_removes_dc_and_slow_drift_while_preserving_passband() -> None:
    """Zero-phase highpass removes DC/0.05 Hz drift while preserving a 2 Hz Sound-rate wave."""
    fs = 8.0
    duration_sec = 60.0
    t = np.arange(0.0, duration_sec, 1.0 / fs, dtype=np.float64)

    clean_2hz = np.sin(2.0 * np.pi * 2.0 * t)
    dc_and_drift = 5.0 + 1.5 * np.sin(2.0 * np.pi * 0.05 * t)
    composite = (clean_2hz + dc_and_drift).astype(np.float32)

    filtered = highpass_filter(
        signal=composite,
        fs=fs,
        lowcut=0.5,
        order=4,
    )

    assert filtered.dtype == np.float32
    assert filtered.shape == composite.shape
    assert np.all(np.isfinite(filtered))

    interior = slice(int(10.0 * fs), int(50.0 * fs))
    max_residual = float(np.max(np.abs(filtered[interior] - clean_2hz[interior])))
    assert max_residual < 0.01


def test_clean_interpolated_channel_interpolates_spo2_dropouts_and_handles_edges() -> None:
    """SpO2 0%/out-of-range spikes are interpolated between 96% neighbors, edge-filled, and median-filtered."""
    spo2_with_dropouts = np.array(
        [0.0, 96.0, 0.0, 0.0, 96.0, 80.0, 96.0, 105.0],
        dtype=np.float32,
    )
    cleaned = clean_interpolated_channel(
        signal=spo2_with_dropouts,
        valid_min=50.0,
        valid_max=100.0,
        median_kernel_size=3,
    )
    assert cleaned.dtype == np.float32
    assert np.allclose(cleaned, 96.0)

    ramp_with_gaps = np.array([0.0, 96.0, 0.0, 92.0, 92.0, 0.0], dtype=np.float32)
    cleaned_ramp = clean_interpolated_channel(
        signal=ramp_with_gaps,
        valid_min=50.0,
        valid_max=100.0,
        median_kernel_size=3,
    )
    expected_ramp = np.array([96.0, 96.0, 94.0, 92.0, 92.0, 92.0], dtype=np.float32)
    assert np.allclose(cleaned_ramp, expected_ramp)

    all_invalid = np.array([0.0, -1.0, np.nan, 0.0], dtype=np.float32)
    fallback = clean_interpolated_channel(
        signal=all_invalid,
        valid_min=50.0,
        valid_max=100.0,
        median_kernel_size=3,
    )
    assert fallback.dtype == np.float32
    assert np.all(np.isfinite(fallback))
    assert np.allclose(fallback, 75.0)


def test_filter_recording_signals_preserves_shapes_and_cleans_real_recording() -> None:
    """Filtering a real UCDDB Recording preserves 14 float32 shapes, cleans dropouts, and leaves zero NaN/Inf."""
    config = load_config(CONFIG_PATH)
    rec = load_recording("ucddb002", config)
    spo2_cfg = config["filters"]["spo2"]
    pulse_cfg = config["filters"]["pulse"]

    # Precondition: ucddb002 contains real hardware dropouts on SpO2 and Pulse
    assert float(np.min(rec.signals["SpO2"])) < spo2_cfg["valid_min"]
    assert float(np.min(rec.signals["Pulse"])) < pulse_cfg["valid_min"]

    filtered = filter_recording_signals(
        signals=rec.signals,
        sampling_rates=rec.sampling_rates,
        config=config,
    )

    assert list(filtered.keys()) == list(rec.signals.keys())
    for ch_name, raw_waveform in rec.signals.items():
        out_waveform = filtered[ch_name]
        assert out_waveform.dtype == np.float32, f"Expected float32 for {ch_name}"
        assert out_waveform.shape == raw_waveform.shape, f"Shape mismatch for {ch_name}"
        assert np.all(np.isfinite(out_waveform)), f"Non-finite values in {ch_name}"

    assert float(np.min(filtered["SpO2"])) >= spo2_cfg["valid_min"]
    assert float(np.max(filtered["SpO2"])) <= spo2_cfg["valid_max"]
    assert float(np.min(filtered["Pulse"])) >= pulse_cfg["valid_min"]
    assert float(np.max(filtered["Pulse"])) <= pulse_cfg["valid_max"]

    assert np.array_equal(filtered["BodyPos"], rec.signals["BodyPos"])
    assert filtered["BodyPos"] is not rec.signals["BodyPos"]
    assert abs(float(np.mean(filtered["Sound"]))) < 0.05



