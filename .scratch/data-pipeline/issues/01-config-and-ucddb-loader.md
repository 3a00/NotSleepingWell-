# 01: Project Configuration & UCDDB Recording Loader

**What to build:** Create the central `config.yaml` at the repo root and implement `src/pipeline/loader.py` to stream UCDDB Recordings one at a time from `database/files/`. The loader must parse `<id>.rec` (EDF format), normalize channel names to the 14 canonical PSG channels (mapping the `'Soud'` typo to `'Sound'` and dropping extra leg EMG channels present in only 2 files), parse `<id>_respevt.txt` with midnight-rollover clock arithmetic relative to the EDF header `starttime`, and parse `<id>_stage.txt` sleep stage annotations.

**Blocked by:** None (can start immediately)

**Type:** task

**Status:** resolved

## Acceptance Criteria

- [x] `config.yaml` created at repo root defining all paths (`raw_data_dir`, `processed_dir`, `splits_path`), `seed: 42`, canonical 14 channel names & sampling rates, filter parameters, windowing parameters (`window_sec: 30`, `stride_sec: 30`, `overlap_threshold_sec: 5.0`, `valid_events`, `exclude_wake: true`), feature context parameters (`hrv_context_sec: 120`, `spo2_lag_sec: 15`, `baseline_window_sec: 300`), and split ratios (`0.70 / 0.15 / 0.15`).
- [x] `.gitignore` updated to ensure `data/`, `outputs/`, `.venv/`, `venv/`, and `__pycache__/` are ignored.
- [x] `src/pipeline/loader.py` implemented with type hints and Google-style docstrings:
  - `discover_recording_ids(raw_dir)` returning the 25 sorted UCDDB IDs (`ucddb002`..`ucddb028`, ignoring `_lifecard.edf`).
  - `load_recording(recording_id, config)` returning a structured `RecordingData` dataclass/dict containing:
    - `recording_id`: `str`
    - `start_time_sec`: `float` (seconds from midnight)
    - `duration_sec`: `float`
    - `signals`: `dict[str, np.ndarray]` of physical-unit float32 waveforms for the 14 canonical channels
    - `sampling_rates`: `dict[str, float]` (`128.0`, `64.0`, or `8.0`)
    - `events`: list of parsed `ApneaEvent(start_sec, duration_sec, end_sec, event_type)` where `start_sec = (event_clock_sec - edf_start_clock_sec) % 86400`
    - `sleep_stages`: `np.ndarray` of int8 sleep stages from `<id>_stage.txt`
- [x] Unit test `tests/test_loader.py` verifying that all 25 Recordings have the exact 14 canonical channels and 0 out-of-bounds valid Apnea events.

## Comments

- 2026-11-06: Implemented `config.yaml`, `src/pipeline/loader.py`, and `tests/test_loader.py` (`pytest` 5/5 passed, `mypy --strict` clean) in commit `dcb0a5a`.
