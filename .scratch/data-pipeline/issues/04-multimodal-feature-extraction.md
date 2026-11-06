# 04: Multi-Modal Physiological Feature Extraction

**What to build:** Implement `src/pipeline/features.py` to convert each 30-second Window of a filtered Recording into a rich, interpretable tabular Feature vector (~85 features) across all 14 channels. Includes continuous Recording pre-computation (R-peak detection on 128 Hz ECG and 5-minute rolling baselines for SpO₂ and respiratory channels) and per-Window extraction of ECG/HRV (30 s local + 120 s centered spectral HRV context), SpO₂/Pulse (30 s local + `+15 s` look-ahead desaturation lag), respiratory effort & thoracoabdominal paradox (`ribcage` vs `abdo` correlation and 5-minute baseline ratios), snore envelope (`Sound`), posture (`BodyPos`), and EEG/EOG/EMG spectral band powers.

**Blocked by:** 02, 03

**Type:** task

**Status:** resolved

## Acceptance Criteria

- [x] `src/pipeline/features.py` implemented with type hints, Google-style docstrings, and modular feature-group functions controlled by `config.yaml`:
  - `precompute_recording_context(signals, sampling_rates, config)`:
    - Detects R-peaks on continuous filtered `ECG` (128 Hz) using `scipy.signal.find_peaks` with minimum distance (`0.35 s`) and adaptive prominence; computes RR intervals (`ms`) and valid physiological mask (`300–2000 ms`).
    - Precomputes 5-minute (`300 s`) rolling baseline arrays for `SpO2` (rolling 90th percentile) and respiratory channels `Flow`, `Sum`, `ribcage`, `abdo` (rolling RMS and peak-to-peak amplitude).
  - `extract_ecg_hrv_features(...)`: Waveform stats (`mean`, `std`, `skew`, `kurtosis`, `ptp`, `rms`, `iqr`, `zcr`), 30 s local RR stats (`hr_mean`, `hr_std`, `rr_mean`, `sdnn`, `rmssd`, `pnn50`, `rr_min`, `rr_max`, `edr_ptp`), and centered 120 s context (`hrv_context_sec`) Welch/Lomb-Scargle PSD HRV powers (`vlf_power`, `lf_power`, `hf_power`, `lf_hf_ratio`).
  - `extract_spo2_pulse_features(...)`: In-window `spo2_min`, `spo2_mean`, `spo2_std`, `spo2_ptp`, `spo2_drop_from_baseline`, `spo2_time_below_90`, `spo2_slope`; lagged window (`[w_start, w_end + spo2_lag_sec]`) `spo2_lag_min`, `spo2_lag_drop`, `spo2_ods_3pct`; plus `pulse_mean`, `pulse_std`, `pulse_ptp`.
  - `extract_respiratory_features(...)`: For `Flow`, `Sum`, `ribcage`, `abdo`: `std`, `rms`, `ptp`, `iqr`, breathing-band power (`0.1–0.5 Hz`), and `ratio_to_baseline` (`window_rms / baseline_5min_rms`); plus cross-channel paradox features (`rib_abdo_corr`, `rib_abdo_rms_ratio`, `flow_sum_corr`).
  - `extract_sound_bodypos_features(...)`: `sound_rms`, `sound_max`, `sound_std`, `sound_crest_factor`, `sound_burst_ratio`, `bodypos_mode`, `bodypos_changed`.
  - `extract_neuro_features(...)`: Relative Welch band powers (`delta`, `theta`, `alpha`, `sigma`, `beta`), `alpha_delta_ratio`, and `spectral_entropy` for `C3A2` and `C4A1`; `eog_std`, `eog_ptp`, `eog_corr` for `Lefteye`/`RightEye`; `emg_rms`, `emg_std`, `emg_median_freq` for `EMG`.
  - `extract_recording_features(recording_data, filtered_signals, windows, config)` returning a `pd.DataFrame` where every row is one Window with metadata columns + deterministic float32 feature columns, guaranteed free of `NaN` or `Inf` values.
- [x] Unit test `tests/test_features.py` verifying:
  - Deterministic column list and zero `NaN`/`Inf` across a full Recording (`ucddb002`).
  - Anti-phase synthetic `ribcage` and `abdo` sine waves produce `rib_abdo_corr < -0.95` (thoracoabdominal paradox verification).
  - Simulated `+10 s` delayed SpO₂ drop of `5%` after Window end is captured by `spo2_lag_drop >= 4.5` and `spo2_ods_3pct == 1.0`.

## Comments
