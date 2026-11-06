# 03: Stage-Aligned 30s Windowing & Apnea Event Overlap Labeling

**What to build:** Implement `src/pipeline/windowing.py` to slice a Recording into 30-second Windows aligned 1-to-1 with `<id>_stage.txt`, compute exact overlap durations with definite Apnea events (`APNEA-O/C/M`, `HYP-O/C/M`), assign binary label `1` when total overlap in the Window is `>= overlap_threshold_sec` (`5.0 s` in `config.yaml`), and attach Window metadata (`recording_id`, `window_idx`, `start_sec`, `end_sec`, `sleep_stage`, `is_wake`, `overlap_sec`, `label`).

**Blocked by:** 01, 02

**Type:** task

**Status:** resolved

## Acceptance Criteria

- [x] `src/pipeline/windowing.py` implemented with type hints and Google-style docstrings:
  - `compute_window_overlap(w_start, w_end, events)` computing the exact overlap in seconds between `[w_start, w_end]` and all valid Apnea events (merging any overlapping event intervals first so double-annotations cannot inflate overlap beyond `30.0 s`).
  - `create_windows_metadata(recording_data, config)` generating a list of `WindowRecord` metadata objects for `N = min(int(duration_sec // window_sec), len(sleep_stages))` Windows.
  - Helper `slice_channel_window(signal, fs, start_sec, end_sec)` returning the exact 1D numpy slice for any channel and arbitrary `[start_sec, end_sec]` interval (clamped safely to recording boundaries `[0, duration_sec]`) so `features.py` can request 30 s Windows, 120 s HRV context windows, and `+15 s` lagged SpO₂ windows cleanly.
  - Helper `compute_recording_ahi(recording_data, config)` computing clinical Recording AHI (`len(valid_events) / sleep_hours` where `sleep_hours = sum(stage in {1,2,3,4,5}) * 30 / 3600`).
- [x] Unit test `tests/test_windowing.py` verifying:
  - Exact overlap boundary behavior (`4.9 s` → `0`, `5.0 s` → `1`, event spanning two consecutive Windows).
  - Exclusion of `PB` and `POSSIBLE` event types.
  - Alignment with `_stage.txt` and `is_wake` (`sleep_stage == 0`) flag accuracy.

## Comments

- 2026-11-06: Implemented `src/pipeline/windowing.py` and `tests/test_windowing.py` via TDD (`pytest` 13/13 passed across suite, `mypy --strict` clean, verified across all 25 UCDDB Recordings).
