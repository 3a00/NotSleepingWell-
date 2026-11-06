# 05: Stratified Recording-Level Split & Train-Only Feature Scaling

**What to build:** Implement `src/pipeline/data_split.py` to split the 25 UCDDB Recordings into disjoint `train` (17), `val` (4), and `test` (4) sets stratified by Recording AHI severity tier, filter out Wake Windows (`sleep_stage == 0`) when `exclude_wake: true` in `config.yaml`, fit a `StandardScaler` strictly on the `train` Feature vectors, and save `splits.json`, `scaler.joblib`, and scaled split datasets (`train.parquet`, `val.parquet`, `test.parquet`).

**Blocked by:** 03, 04

**Type:** task

**Status:** ready-for-agent

## Acceptance Criteria

- [ ] `src/pipeline/data_split.py` implemented with type hints and Google-style docstrings:
  - `assign_ahi_severity_tier(ahi)` mapping AHI to severity tiers (`0`: `<15` Normal/Mild, `1`: `15–30` Moderate, `2`: `>=30` Severe) so stratified splitting has sufficient samples per stratum across 25 Recordings.
  - `create_recording_splits(recording_summaries, config)` deterministically splitting the 25 Recordings into **17 train / 4 val / 4 test** using `seed` from `config.yaml` and stratified sampling by severity tier. Asserts zero overlap between `train`, `val`, and `test` Recording IDs.
  - `build_and_scale_splits(feature_dfs_by_id, splits, config)` that:
    1. Concatenates per-Recording DataFrames into `train`, `val`, and `test` tables.
    2. Applies `exclude_wake` filtering if enabled in `config.yaml` (preserving raw per-Recording Parquet files untouched).
    3. Fits `sklearn.preprocessing.StandardScaler` **strictly on `train` feature columns** and transforms `train`, `val`, and `test`.
    4. Writes `data/processed/splits.json` (with Recording IDs, AHI values, severity tiers, and window counts), `data/processed/scaler.joblib`, and `data/processed/{train,val,test}.parquet`.
- [ ] Unit test `tests/test_data_split.py` verifying:
  - Exact `17 / 4 / 4` Recording counts and pairwise disjoint Recording ID sets.
  - Zero data leakage: `scaler.mean_` and `scaler.scale_` match the unscaled `train` feature matrix exactly and are invariant to perturbations in `val` or `test` features.

## Comments
