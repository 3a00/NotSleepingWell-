"""Tests for Sleep stage-aligned Windowing, Apnea event overlap labeling, and Recording AHI."""

from pathlib import Path

import numpy as np
import pytest

from src.pipeline.loader import ApneaEvent, RecordingData, load_config, load_recording
from src.pipeline.windowing import (
    WindowRecord,
    compute_recording_ahi,
    compute_window_overlap,
    create_windows_metadata,
    slice_channel_window,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def test_compute_window_overlap_merges_overlapping_apnea_events_and_excludes_non_apnea() -> None:
    """Overlap merges overlapping valid Apnea events without exceeding Window length and ignores PB/POSSIBLE."""
    events = [
        # Two overlapping valid events inside [0.0, 30.0]: [5.0, 20.0] U [10.0, 25.0] = [5.0, 25.0] -> 20.0s
        ApneaEvent(start_sec=5.0, duration_sec=15.0, end_sec=20.0, event_type="APNEA-O"),
        ApneaEvent(start_sec=10.0, duration_sec=15.0, end_sec=25.0, event_type="HYP-O"),
        # Ignored non-apnea annotations fully inside [0.0, 30.0] -> 0.0s contribution
        ApneaEvent(start_sec=0.0, duration_sec=30.0, end_sec=30.0, event_type="PB"),
        ApneaEvent(start_sec=2.0, duration_sec=28.0, end_sec=30.0, event_type="POSSIBLE"),
        # Disjoint valid event partially overlapping end of [0.0, 30.0]: [28.0, 35.0] -> +2.0s
        ApneaEvent(start_sec=28.0, duration_sec=7.0, end_sec=35.0, event_type="APNEA-C"),
    ]

    overlap_sec = compute_window_overlap(0.0, 30.0, events)
    assert overlap_sec == pytest.approx(22.0)

    # Double-annotated events covering the entire 30s Window must cap at 30.0s, never 60.0s
    double_full = [
        ApneaEvent(start_sec=0.0, duration_sec=40.0, end_sec=40.0, event_type="APNEA-M"),
        ApneaEvent(start_sec=0.0, duration_sec=35.0, end_sec=35.0, event_type="HYP-M"),
    ]
    assert compute_window_overlap(0.0, 30.0, double_full) == pytest.approx(30.0)


def test_create_windows_metadata_aligns_sleep_stages_and_enforces_overlap_thresholds() -> None:
    """Windows align 1-to-1 with sleep_stages, exclude PB/POSSIBLE, and label >=5.0s overlap as 1."""
    config = load_config(CONFIG_PATH)

    # duration_sec=195.0 -> 6 full 30s intervals, capped by len(sleep_stages)=5 -> N=5 Windows
    synthetic_rec = RecordingData(
        recording_id="synthetic01",
        start_time_sec=0.0,
        duration_sec=195.0,
        signals={},
        sampling_rates={},
        events=[
            # Window 0 [0, 30]: 4.9s APNEA-O + 13.0s PB -> 4.9s overlap -> label 0
            ApneaEvent(start_sec=10.0, duration_sec=4.9, end_sec=14.9, event_type="APNEA-O"),
            ApneaEvent(start_sec=15.0, duration_sec=13.0, end_sec=28.0, event_type="PB"),
            # Window 1 [30, 60]: exact 5.0s HYP-O on decimal timestamps (35.1 - 30.1) -> label 1
            ApneaEvent(start_sec=30.1, duration_sec=5.0, end_sec=35.1, event_type="HYP-O"),
            # Spanning Window 2 [60, 90] (4.9s -> label 0) and Window 3 [90, 120] (5.0s -> label 1)
            ApneaEvent(start_sec=85.1, duration_sec=9.9, end_sec=95.0, event_type="APNEA-C"),
            # Window 4 [120, 150]: 20.0s POSSIBLE -> 0.0s overlap -> label 0
            ApneaEvent(start_sec=125.0, duration_sec=20.0, end_sec=145.0, event_type="POSSIBLE"),
        ],
        sleep_stages=np.array([0, 2, 1, 8, 3], dtype=np.int8),
    )

    windows = create_windows_metadata(synthetic_rec, config)
    assert len(windows) == 5
    assert all(isinstance(w, WindowRecord) for w in windows)

    expected_rows = [
        ("synthetic01", 0, 0.0, 30.0, 0, True, 4.9, 0),
        ("synthetic01", 1, 30.0, 60.0, 2, False, 5.0, 1),
        ("synthetic01", 2, 60.0, 90.0, 1, False, 4.9, 0),
        ("synthetic01", 3, 90.0, 120.0, 8, False, 5.0, 1),
        ("synthetic01", 4, 120.0, 150.0, 3, False, 0.0, 0),
    ]
    for win, (rec_id, idx, start, end, stage, wake, overlap, label) in zip(
        windows, expected_rows, strict=True
    ):
        assert (
            win.recording_id,
            win.window_idx,
            win.start_sec,
            win.end_sec,
            win.sleep_stage,
            win.is_wake,
            win.label,
        ) == (rec_id, idx, start, end, stage, wake, label)
        assert win.overlap_sec == pytest.approx(overlap)

    # Verify on real UCDDB Recording ucddb002 (duration 22470s -> 749, _stage.txt -> 748 => 748 Windows)
    rec_002 = load_recording("ucddb002", config)
    windows_002 = create_windows_metadata(rec_002, config)
    assert len(windows_002) == 748
    assert windows_002[0].sleep_stage == 0
    assert windows_002[0].is_wake is True
    assert windows_002[18].sleep_stage == 2
    assert windows_002[18].is_wake is False
    assert windows_002[-1].start_sec == 22410.0
    assert windows_002[-1].end_sec == 22440.0


def test_slice_channel_window_extracts_exact_samples_and_clamps_boundaries() -> None:
    """slice_channel_window returns exact 1D float32 slices and safely clamps HRV/SpO2 look-around bounds."""
    # 120s signal at 128 Hz -> 15360 samples
    ecg_sig = np.arange(15360, dtype=np.float32)

    in_bounds = slice_channel_window(ecg_sig, fs=128.0, start_sec=30.0, end_sec=60.0)
    assert in_bounds.dtype == np.float32
    assert in_bounds.shape == (3840,)
    assert float(in_bounds[0]) == 3840.0
    assert float(in_bounds[-1]) == 7679.0

    # Centered 120s HRV context for Window 0 [-45.0s, 75.0s] -> clamped to [0.0s, 75.0s] (9600 samples)
    hrv_clamped = slice_channel_window(ecg_sig, fs=128.0, start_sec=-45.0, end_sec=75.0)
    assert hrv_clamped.shape == (9600,)
    assert float(hrv_clamped[0]) == 0.0
    assert float(hrv_clamped[-1]) == 9599.0

    # +15s lagged SpO2 window at Recording end [90.0s, 135.0s] on 120s 8 Hz signal (960 samples) -> [90s, 120s]
    spo2_sig = np.arange(960, dtype=np.float32)
    spo2_clamped = slice_channel_window(spo2_sig, fs=8.0, start_sec=90.0, end_sec=135.0)
    assert spo2_clamped.shape == (240,)
    assert float(spo2_clamped[0]) == 720.0
    assert float(spo2_clamped[-1]) == 959.0

    with pytest.raises(ValueError, match="end_sec"):
        slice_channel_window(spo2_sig, fs=8.0, start_sec=30.0, end_sec=10.0)


def test_compute_recording_ahi_counts_valid_apnea_events_per_definite_sleep_hour() -> None:
    """AHI divides valid Apnea event count by hours in Sleep stages {1,2,3,4,5}, excluding 0/8 and PB/POSSIBLE."""
    config = load_config(CONFIG_PATH)

    # 120 definite sleep Windows (stages 1..5) * 30s = 3600s = 1.0 sleep hour;
    # plus 30 Wake (0) and 10 Indeterminate (8) Windows that must not count as sleep time.
    sleep_stages = np.array(
        [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20 + [5] * 40 + [0] * 30 + [8] * 10,
        dtype=np.int8,
    )
    valid_types = ["APNEA-O", "APNEA-C", "APNEA-M", "HYP-O", "HYP-C", "HYP-M"]
    events = [
        ApneaEvent(start_sec=float(i * 60), duration_sec=12.0, end_sec=float(i * 60 + 12), event_type=etype)
        for i, etype in enumerate(valid_types * 3)  # 18 valid Apnea events
    ] + [
        ApneaEvent(start_sec=2000.0, duration_sec=15.0, end_sec=2015.0, event_type="PB"),
        ApneaEvent(start_sec=2100.0, duration_sec=10.0, end_sec=2110.0, event_type="POSSIBLE"),
    ]

    synthetic_rec = RecordingData(
        recording_id="synthetic_ahi",
        start_time_sec=0.0,
        duration_sec=4800.0,
        signals={},
        sampling_rates={},
        events=events,
        sleep_stages=sleep_stages,
    )

    assert compute_recording_ahi(synthetic_rec, config) == pytest.approx(18.0)

    wake_only_rec = RecordingData(
        recording_id="wake_only",
        start_time_sec=0.0,
        duration_sec=600.0,
        signals={},
        sampling_rates={},
        events=[],
        sleep_stages=np.array([0, 0, 8, 0], dtype=np.int8),
    )
    with pytest.raises(ValueError, match="sleep"):
        compute_recording_ahi(wake_only_rec, config)



