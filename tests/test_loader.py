"""Tests for project configuration and UCDDB Recording loader."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.pipeline.loader import (
    ApneaEvent,
    RecordingData,
    discover_recording_ids,
    load_config,
    load_recording,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"

EXPECTED_RECORDING_IDS = [
    "ucddb002",
    "ucddb003",
    "ucddb005",
    "ucddb006",
    "ucddb007",
    "ucddb008",
    "ucddb009",
    "ucddb010",
    "ucddb011",
    "ucddb012",
    "ucddb013",
    "ucddb014",
    "ucddb015",
    "ucddb017",
    "ucddb018",
    "ucddb019",
    "ucddb020",
    "ucddb021",
    "ucddb022",
    "ucddb023",
    "ucddb024",
    "ucddb025",
    "ucddb026",
    "ucddb027",
    "ucddb028",
]

EXPECTED_CANONICAL_CHANNELS = {
    "C3A2": 128.0,
    "C4A1": 128.0,
    "Lefteye": 64.0,
    "RightEye": 64.0,
    "EMG": 64.0,
    "ECG": 128.0,
    "SpO2": 8.0,
    "Pulse": 8.0,
    "Flow": 8.0,
    "Sum": 8.0,
    "ribcage": 8.0,
    "abdo": 8.0,
    "Sound": 8.0,
    "BodyPos": 8.0,
}


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    """Load pipeline config.yaml once per test module."""
    return load_config(CONFIG_PATH)


def test_load_config_defines_pipeline_parameters_and_canonical_channels(
    config: dict[str, Any],
) -> None:
    """config.yaml defines all required paths, channels, filters, windowing, features, and splits."""
    assert config["seed"] == 42
    assert config["paths"]["raw_data_dir"] == "database/files"
    assert config["paths"]["processed_dir"] == "data/processed"
    assert config["paths"]["splits_path"] == "data/processed/splits.json"

    assert set(config["channels"]["canonical"]) == set(EXPECTED_CANONICAL_CHANNELS.keys())
    assert len(config["channels"]["canonical"]) == 14
    assert config["channels"]["sampling_rates"] == EXPECTED_CANONICAL_CHANNELS
    assert config["channels"]["aliases"] == {"Soud": "Sound"}

    assert "filters" in config
    assert config["windowing"]["window_sec"] == 30
    assert config["windowing"]["stride_sec"] == 30
    assert config["windowing"]["overlap_threshold_sec"] == 5.0
    assert set(config["windowing"]["valid_events"]) == {
        "APNEA-O",
        "APNEA-C",
        "APNEA-M",
        "HYP-O",
        "HYP-C",
        "HYP-M",
    }
    assert config["windowing"]["exclude_wake"] is True

    assert config["features"]["hrv_context_sec"] == 120
    assert config["features"]["spo2_lag_sec"] == 15
    assert config["features"]["baseline_window_sec"] == 300

    assert config["splitting"]["train_ratio"] == 0.70
    assert config["splitting"]["val_ratio"] == 0.15
    assert config["splitting"]["test_ratio"] == 0.15


def test_discover_recording_ids_returns_25_sorted_ucddb_recordings_ignoring_lifecard(
    config: dict[str, Any],
) -> None:
    """discover_recording_ids returns the exact 25 sorted UCDDB IDs and excludes _lifecard.edf."""
    raw_dir = REPO_ROOT / config["paths"]["raw_data_dir"]

    recording_ids = discover_recording_ids(raw_dir)

    assert recording_ids == EXPECTED_RECORDING_IDS


def test_load_recording_normalizes_channels_and_parses_waveforms_and_sleep_stages(
    config: dict[str, Any],
) -> None:
    """load_recording extracts physical float32 signals, normalizes channels, and loads int8 stages."""
    rec_002 = load_recording("ucddb002", config)
    assert isinstance(rec_002, RecordingData)
    assert rec_002.recording_id == "ucddb002"
    assert rec_002.start_time_sec == 664.0
    assert rec_002.duration_sec == 22470.0
    assert rec_002.sampling_rates == EXPECTED_CANONICAL_CHANNELS
    assert set(rec_002.signals.keys()) == set(EXPECTED_CANONICAL_CHANNELS.keys())
    assert rec_002.signals["ECG"].dtype == np.float32
    assert rec_002.signals["ECG"].shape == (2876160,)
    assert rec_002.signals["Lefteye"].shape == (1438080,)
    assert rec_002.signals["SpO2"].shape == (179760,)
    assert 90.0 <= float(np.median(rec_002.signals["SpO2"])) <= 98.0
    assert rec_002.sleep_stages.dtype == np.int8
    assert rec_002.sleep_stages.shape == (748,)
    assert int(rec_002.sleep_stages[0]) == 0
    assert int(rec_002.sleep_stages[18]) == 2

    # ucddb003 has the 'Soud' typo in EDF header -> mapped to 'Sound'
    rec_003 = load_recording("ucddb003", config)
    assert set(rec_003.signals.keys()) == set(EXPECTED_CANONICAL_CHANNELS.keys())
    assert "Soud" not in rec_003.signals

    # ucddb011 has 16 EDF channels including 'Left leg' and 'Right leg' -> dropped
    rec_011 = load_recording("ucddb011", config)
    assert rec_011.start_time_sec == 82058.0
    assert rec_011.duration_sec == 27030.0
    assert set(rec_011.signals.keys()) == set(EXPECTED_CANONICAL_CHANNELS.keys())
    assert rec_011.sleep_stages.shape == (900,)


def test_load_recording_parses_apnea_events_across_midnight_rollover(
    config: dict[str, Any],
) -> None:
    """load_recording aligns HH:MM:SS timestamps across midnight and normalizes PB/POSSIBLE."""
    # ucddb011 starts at 22:47:38; first event at 00:05:54 (HYP-O, 12s) -> 4696.0s from start
    rec_011 = load_recording("ucddb011", config)
    assert len(rec_011.events) > 0
    assert rec_011.events[0] == ApneaEvent(
        start_sec=4696.0,
        duration_sec=12.0,
        end_sec=4708.0,
        event_type="HYP-O",
    )

    # ucddb006 starts at 23:57:14 and contains 'PB EVENT  PB' at 02:52:38 (14s) -> 'PB' at 10524.0s
    rec_006 = load_recording("ucddb006", config)
    assert ApneaEvent(
        start_sec=10524.0,
        duration_sec=14.0,
        end_sec=10538.0,
        event_type="PB",
    ) in rec_006.events

    # ucddb003 starts at 23:07:50 and contains 'POSSIBLE' at 03:08:57 (9s) -> 14467.0s
    rec_003 = load_recording("ucddb003", config)
    assert ApneaEvent(
        start_sec=14467.0,
        duration_sec=9.0,
        end_sec=14476.0,
        event_type="POSSIBLE",
    ) in rec_003.events


def test_all_25_recordings_have_14_canonical_channels_and_in_bounds_valid_apnea_events(
    config: dict[str, Any],
) -> None:
    """All 25 UCDDB Recordings load with the exact 14 canonical channels and 0 OOB events."""
    raw_dir = REPO_ROOT / config["paths"]["raw_data_dir"]
    recording_ids = discover_recording_ids(raw_dir)
    assert len(recording_ids) == 25

    valid_event_types = set(config["windowing"]["valid_events"])
    known_event_types = valid_event_types | set(config["windowing"]["ignored_events"])
    total_valid_events = 0

    for recording_id in recording_ids:
        rec = load_recording(recording_id, config)
        assert list(rec.signals.keys()) == list(EXPECTED_CANONICAL_CHANNELS.keys())
        assert rec.sampling_rates == EXPECTED_CANONICAL_CHANNELS
        assert rec.sleep_stages.dtype == np.int8
        assert rec.sleep_stages.size > 0

        for ch_name, expected_fs in EXPECTED_CANONICAL_CHANNELS.items():
            waveform = rec.signals[ch_name]
            assert waveform.dtype == np.float32
            assert waveform.shape == (int(round(rec.duration_sec * expected_fs)),)

        for event in rec.events:
            assert event.event_type in known_event_types, (
                f"Unexpected event_type {event.event_type!r} in {recording_id}"
            )
            assert 0.0 <= event.start_sec < event.end_sec <= rec.duration_sec, (
                f"Out-of-bounds event in {recording_id}: {event} "
                f"(duration_sec={rec.duration_sec})"
            )
            if event.event_type in valid_event_types:
                total_valid_events += 1

    assert total_valid_events > 0
