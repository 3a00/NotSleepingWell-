# 02: Modality-Specific Signal Filtering & Artifact Cleaning

**What to build:** Implement `src/pipeline/filters.py` to clean and filter the 14 physiological channels of a loaded Recording using parameters from `config.yaml`. Continuous AC waveforms (`ECG`, `C3A2`, `C4A1`, `Lefteye`, `RightEye`, `EMG`, `Flow`, `Sum`, `ribcage`, `abdo`, `Sound`) are filtered with zero-phase Butterworth filters (`scipy.signal.sosfiltfilt`), while oximetry and pulse rate channels (`SpO2`, `Pulse`) undergo non-physiological dropout clipping (`<50%` SpO₂, `<25 bpm` Pulse), linear interpolation across dropouts, and median filtering. `BodyPos` is passed through unchanged.

**Blocked by:** 01

**Type:** task

**Status:** ready-for-agent

## Acceptance Criteria

- [ ] `src/pipeline/filters.py` implemented with type hints, Google-style docstrings, and zero hardcoded cutoffs/orders:
  - `bandpass_filter(signal, fs, lowcut, highcut, order)` and `highpass_filter(signal, fs, lowcut, order)` using `scipy.signal.butter(..., output='sos')` and `scipy.signal.sosfiltfilt`.
  - `clean_interpolated_channel(signal, valid_min, valid_max, median_kernel_size)` that replaces out-of-range hardware dropouts (such as `0%` SpO₂) with `NaN`, linearly interpolates interior gaps, forward/backward fills edge gaps (with safe fallback if an entire array is invalid), and applies `scipy.signal.medfilt`.
  - `filter_recording_signals(signals, sampling_rates, config)` that routes each of the 14 channels through its configured filter/cleaner and returns `dict[str, np.ndarray]` in `float32` with zero `NaN` or `Inf` values.
- [ ] Unit test `tests/test_filters.py` testing:
  - Passband preservation and stopband attenuation on synthetic sine waves (e.g., 0.1 Hz baseline wander and 50 Hz noise removed from a 10 Hz ECG test wave).
  - `SpO2` dropout cleaning where `0.0%` spikes are smoothly interpolated between surrounding `96.0%` values.
  - Shape preservation and absence of `NaN`/`Inf` on a real Recording from `loader.py`.

## Comments
