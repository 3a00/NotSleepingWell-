# Data Pipeline — Multi-Channel PSG Signal Processing & Feature Extraction for Sleep Apnea Detection

**Status:** ready-for-agent

## Problem Statement

The UCDDB dataset (`database/files/`) contains 25 overnight polysomnography Recordings (`ucddb002` through `ucddb028`), each stored as an EDF-formatted `<id>.rec` file (~6–8 hours, multi-rate 8 Hz / 64 Hz / 128 Hz signals), paired with text annotations for respiratory events (`<id>_respevt.txt`, clock timestamps `HH:MM:SS`) and 30-second sleep stages (`<id>_stage.txt`). Classical machine learning models like XGBoost cannot directly consume raw, multi-rate 1D physiological waveforms due to phase variance, sensor drift, hardware dropouts (e.g., `0%` SpO₂ artifacts), channel naming inconsistencies (`'Soud'` vs `'Sound'`, extra leg EMG channels in 2 files), and circulatory lag between airflow cessation and oxygen desaturation. We need a deterministic, config-driven, memory-safe data pipeline that transforms raw UCDDB Recordings into clean, leak-free tabular Feature vectors ready for XGBoost training and clinical AHI evaluation.

## Solution

Build a modular, single-responsibility Python pipeline under `src/pipeline/` controlled entirely by `config.yaml`:
1. **Loader (`src/pipeline/loader.py`)**: Stream one Recording at a time from `<id>.rec`, normalize channel names to the 14 canonical PSG channels, parse `<id>_respevt.txt` using midnight-rollover clock arithmetic against the EDF header `starttime`, and load `<id>_stage.txt`.
2. **Filters (`src/pipeline/filters.py`)**: Apply modality-specific zero-phase Butterworth (`sosfiltfilt`) bandpass/highpass filters to continuous waveforms (`ECG`, `EEG`, `EOG`, `EMG`, respiratory `Flow`/`Sum`/`ribcage`/`abdo`, `Sound`) and clean/interpolate non-physiological dropouts (`<50%` SpO₂, `<20 bpm` Pulse).
3. **Windowing (`src/pipeline/windowing.py`)**: Cut 30-second Windows aligned 1-to-1 with `<id>_stage.txt`, compute exact overlap seconds with definite Apnea events (`APNEA-O/C/M`, `HYP-O/C/M`), assign binary label `1` when overlap `>= 5.0 s`, and attach `sleep_stage` metadata (supporting configurable Wake Window exclusion).
4. **Features (`src/pipeline/features.py`)**: Detect R-peaks on the continuous Recording ECG, compute 5-minute rolling physiological baselines, and extract a comprehensive tabular Feature vector per Window across all 14 channels (including a centered 120 s HRV spectral context window, a `+15 s` lagged SpO₂ desaturation look-ahead window, and thoracoabdominal paradox features).
5. **Splitting & Scaling (`src/pipeline/data_split.py`)**: Perform a deterministic 70/15/15 Recording-level split (17 train / 4 val / 4 test) stratified by Recording AHI severity class, fit a `StandardScaler` strictly on the train split Feature vectors, and persist per-Recording Parquet files (`data/processed/features/<id>.parquet`), split manifests (`data/processed/splits.json`), and scaled split tables (`train.parquet`, `val.parquet`, `test.parquet`).

## User Stories

1. As an ML engineer, I want all paths, channel definitions, filter cutoffs/orders, window lengths, overlap thresholds, lag windows, split ratios, and random seeds stored in `config.yaml`, so that no magic numbers or hardcoded paths exist in `src/`.
2. As a data engineer, I want `loader.py` to normalize the `'Soud'` typo to `'Sound'`, ignore the unaligned `_lifecard.edf` files and the 2-subject-only leg EMG channels, and align `HH:MM:SS` event timestamps across midnight using `(t_event - t_start) % 86400`, so that all 25 Recordings share an identical 14-channel schema with zero out-of-bounds events.
3. As a signal processing engineer, I want modality-specific zero-phase Butterworth filters (`scipy.signal.sosfiltfilt`) and SpO₂/Pulse dropout interpolation in `filters.py`, so that baseline wander, powerline noise, and `0%` oximeter dropouts do not corrupt downstream features.
4. As a clinical researcher, I want 30-second Windows aligned with `<id>_stage.txt` and labeled `1` when definite Apnea events (`APNEA-O/C/M`, `HYP-O/C/M`, excluding `PB` and `POSSIBLE`) overlap by `>= 5.0 s`, so that Window labels and sleep-stage metadata match clinical polysomnography conventions.
5. As an ML researcher, I want R-peaks detected across the continuous Recording ECG and combined with a centered 120 s HRV context window, a `+15 s` SpO₂ look-ahead window, and 5-minute rolling baseline ratios in `features.py`, so that XGBoost captures autonomic heart-rate variability, delayed blood oxygen desaturation, and `>=30%` hypopnea airflow drops.
6. As an ML engineer, I want the 25 Recordings split 70/15/15 (17 train / 4 val / 4 test) stratified by Recording AHI severity class and `StandardScaler` fit strictly on `train` Windows, so that zero patient-level or scaler-level data leakage occurs.
7. As an ML engineer, I want Recordings processed one at a time and saved to `data/processed/features/<id>.parquet` via a CLI runner `scripts/run_pipeline.py`, so that RAM usage stays well below the 15 GB hardware limit.

## Implementation Decisions

- **Canonical Dataset & Channels ([ADR-0001](../../docs/adr/0001-ucddb-rec-and-handcrafted-features.md))**:
  - Source files: `<id>.rec` (EDF format), `<id>_respevt.txt`, `<id>_stage.txt` for all 25 UCDDB subjects (`ucddb002`–`ucddb028`, noting `004` and `016` do not exist in UCDDB).
  - Canonical 14 channels and sampling rates:
    - Neurological: `C3A2` (128 Hz), `C4A1` (128 Hz), `Lefteye` (64 Hz), `RightEye` (64 Hz), `EMG` (64 Hz)
    - Cardiac: `ECG` (128 Hz)
    - Respiratory & Oximetry: `SpO2` (8 Hz), `Pulse` (8 Hz), `Flow` (8 Hz), `Sum` (8 Hz), `ribcage` (8 Hz), `abdo` (8 Hz), `Sound` (8 Hz; mapped from `'Soud'` when present)
    - Posture: `BodyPos` (8 Hz)
- **Modality-Specific Filtering (`src/pipeline/filters.py`)**:
  - `ECG` (128 Hz): 4th-order Butterworth bandpass `0.5–40.0 Hz`.
  - `C3A2`, `C4A1`, `Lefteye`, `RightEye` (128/64 Hz): 4th-order Butterworth bandpass `0.5–35.0 Hz`.
  - `EMG` (64 Hz): 4th-order Butterworth bandpass `10.0–30.0 Hz` (strictly below 32 Hz Nyquist).
  - `Flow`, `Sum`, `ribcage`, `abdo` (8 Hz): 4th-order Butterworth bandpass `0.05–2.0 Hz` (respiratory band).
  - `Sound` (8 Hz): 4th-order Butterworth highpass `0.5 Hz`.
  - `SpO2` (8 Hz): Clip values outside `[50.0, 100.0]%` as sensor dropouts (`NaN`), linearly interpolate gaps (with forward/backward fill at recording edges), then apply a 3-sample median filter.
  - `Pulse` (8 Hz): Clip values outside `[25.0, 220.0] bpm` as dropouts (`NaN`), linearly interpolate, and median-filter.
  - `BodyPos` (8 Hz): Unfiltered discrete signal.
- **Windowing & Labeling (`src/pipeline/windowing.py`, [ADR-0002](../../docs/adr/0002-windowing-spo2-lag-and-baseline-normalization.md))**:
  - Window length: `30.0 s`, stride: `30.0 s`. Total Windows per Recording = `min(floor(recording_duration_sec / 30), len(stage_lines))`.
  - Valid Apnea events: `["APNEA-O", "APNEA-C", "APNEA-M", "HYP-O", "HYP-C", "HYP-M"]`. Ignored: `["PB", "POSSIBLE"]`.
  - Overlap calculation: Sum of intersection durations `max(0.0, min(w_end, e_end) - max(w_start, e_start))` across all valid Apnea events in the Recording. Label `y = 1` if `overlap_sec >= 5.0`, else `0`.
  - Wake handling: Attach `sleep_stage` (`int`) and `is_wake` (`stage == 0`) to every Window row. `config.yaml` flag `exclude_wake: true` controls whether Wake Windows are filtered out when building the train/val/test matrices.
- **Feature Extraction (`src/pipeline/features.py`)**:
  - Continuous Recording pre-computation (computed once per Recording before slicing Window features):
    - **R-peaks & RR intervals**: Detect R-peaks on filtered 128 Hz `ECG` (`scipy.signal.find_peaks` with physiologically bounded minimum distance `0.35 s` and adaptive prominence), compute RR intervals (`ms`) and physiological outlier rejection (`300–2000 ms`).
    - **5-minute Rolling Baselines (`baseline_window_sec: 300`)**: Rolling 90th-percentile of `SpO2` and rolling root-mean-square / peak-to-peak amplitude of `Flow`, `Sum`, `ribcage`, and `abdo`.
  - Per-Window Feature Groups (~85 features total):
    1. **ECG & HRV (`ecg_*`, `hrv_*`)**: Waveform `mean`, `std`, `skew`, `kurtosis`, `ptp`, `rms`, `iqr`, `zcr`; 30 s local RR `hr_mean`, `hr_std`, `rr_mean`, `sdnn`, `rmssd`, `pnn50`, `rr_min`, `rr_max`, `edr_ptp` (ECG-derived respiration amplitude from R-peak heights); centered 120 s context (`[t-45s, t+75s]`) Lomb-Scargle / interpolated PSD spectral powers (`vlf_power`, `lf_power`, `hf_power`, `lf_hf_ratio`).
    2. **Oximetry & Pulse (`spo2_*`, `pulse_*`)**: In-window `spo2_min`, `spo2_mean`, `spo2_std`, `spo2_ptp`, `spo2_drop_from_baseline` (`baseline_spo2 - spo2_min`), `spo2_time_below_90`, `spo2_slope`; lagged window (`[t, min(T, t + 45s)]`, i.e., `+15 s` look-ahead) `spo2_lag_min`, `spo2_lag_drop`, `spo2_ods_3pct` (`1.0` if drop `>= 3.0%` else `0.0`); `pulse_mean`, `pulse_std`, `pulse_ptp`.
    3. **Airflow, Effort & Paradox (`resp_*`)**: For each of `Flow`, `Sum`, `ribcage`, `abdo`: `std`, `rms`, `ptp`, `iqr`, breathing-band power (`0.1–0.5 Hz`), and `ratio_to_baseline` (`window_rms / rolling_5min_rms`, capturing `>=30%` hypopnea and `>=90%` apnea reductions); plus cross-channel **thoracoabdominal paradox** features: `rib_abdo_corr` (Pearson $r$ between `ribcage` and `abdo`), `rib_abdo_rms_ratio`, and `flow_sum_corr`.
    4. **Snore & Posture (`sound_*`, `bodypos_*`)**: `sound_rms`, `sound_max`, `sound_std`, `sound_crest_factor`, `sound_burst_ratio`; `bodypos_mode`, `bodypos_change_flag`.
    5. **EEG, EOG & EMG (`eeg_*`, `eog_*`, `emg_*`)**: For `C3A2` and `C4A1`: Welch relative band powers (`delta 0.5–4 Hz`, `theta 4–8 Hz`, `alpha 8–12 Hz`, `sigma 12–16 Hz`, `beta 16–30 Hz`), `alpha_delta_ratio`, `spectral_entropy`; for `Lefteye`/`RightEye`: `eog_std`, `eog_ptp`, `eog_corr`; for `EMG`: `emg_rms`, `emg_std`, `emg_median_freq`.
- **Splitting & Scaling (`src/pipeline/data_split.py`)**:
  - Compute true Recording AHI (`total_apnea_events / total_sleep_hours`) across the 25 Recordings and stratify by clinical severity tier (`Normal/Mild < 15`, `Moderate 15–30`, `Severe >= 30`) into **17 train / 4 val / 4 test** Recordings using `seed: 42` from `config.yaml`.
  - Fit `sklearn.preprocessing.StandardScaler` strictly on `train` Feature vectors (after optional Wake filtering). Save scaler parameters to `data/processed/scaler.joblib` (or JSON stats) and output scaled `train.parquet`, `val.parquet`, `test.parquet` alongside `data/processed/splits.json`.

## Testing Decisions

- All pipeline stages are verified with deterministic unit and contract tests under `tests/`:
  1. `tests/test_loader.py`: Verifies EDF reading on `ucddb002.rec`, channel normalization (`'Soud'` → `'Sound'`, 14 canonical channels), and midnight-rollover timestamp parsing with zero out-of-bounds events across all 25 Recordings.
  2. `tests/test_filters.py`: Verifies synthetic sine-wave attenuation outside passbands, preservation inside passbands, zero NaNs, and SpO₂ `0%` dropout interpolation.
  3. `tests/test_windowing.py`: Verifies 30 s Window counts against `_stage.txt`, exact boundary overlap math (`<5s` vs `>=5s`), exclusion of `PB`/`POSSIBLE`, and `is_wake` flagging.
  4. `tests/test_features.py`: Verifies Feature vector extraction on a real Recording slice and synthetic signals, asserting deterministic column names, zero `NaN`/`Inf` values, negative `rib_abdo_corr` on anti-phase signals, and lagged SpO₂ drop detection.
  5. `tests/test_data_split.py`: Verifies 17/4/4 disjoint Recording sets (`set(train) & set(val) == set()`), stratification balance, and that `StandardScaler` mean/std match `train` only.

## Out of Scope

- XGBoost model training, hyperparameter tuning, and threshold calibration (scoped to the next feature spec: `.scratch/xgboost-baseline/`).
- Deep learning 1D-CNN / Transformer raw-waveform models.
- Using `<id>_lifecard.edf` 3-lead Holter files.
