# 06: End-to-End Streaming Pipeline Runner & Dataset Verification

**What to build:** Implement the CLI orchestration script `scripts/run_pipeline.py` that loads `config.yaml`, sets all random seeds (`random`, `numpy`), streams the 25 UCDDB Recordings one at a time through `loader` → `filters` → `windowing` → `features`, saves each Recording's unscaled feature table to `data/processed/features/<id>.parquet` (freeing memory after each Recording), then runs `data_split` to produce `splits.json`, `scaler.joblib`, and the scaled `train.parquet`, `val.parquet`, and `test.parquet` tables, along with a dataset summary report in `outputs/pipeline_summary.json`.

**Blocked by:** 01, 02, 03, 04, 05

**Type:** task

**Status:** resolved

## Acceptance Criteria

- [x] `scripts/run_pipeline.py` implemented with CLI arguments (`--config config.yaml`, optional `--force` to recompute cached per-Recording Parquet files).
- [x] Processes Recordings sequentially (one Recording in memory at a time, using `gc.collect()` between Recordings) and writes 25 Parquet files to `data/processed/features/<id>.parquet`.
- [x] Executes `data_split.py` to generate `data/processed/splits.json`, `data/processed/scaler.joblib`, and `data/processed/{train,val,test}.parquet`.
- [x] Writes `outputs/pipeline_summary.json` reporting total Recordings, Windows per split, positive Apnea Window ratios per split, feature count, and peak RAM usage.
- [x] Full test suite (`pytest tests/`) and full pipeline execution (`python scripts/run_pipeline.py --config config.yaml`) run cleanly with exit code 0 and zero `NaN`/`Inf` values in any generated Parquet file.

## Comments
