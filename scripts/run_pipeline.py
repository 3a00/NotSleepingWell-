"""End-to-end streaming data pipeline runner for UCDDB Sleep Apnea Detection."""

import argparse
from collections.abc import Mapping, Sequence
import gc
import json
from pathlib import Path
import random
import resource
import sys
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline.data_split import (  # noqa: E402
    RecordingSummary,
    assign_ahi_severity_tier,
    build_and_scale_splits,
    create_recording_splits,
)
from src.pipeline.features import extract_recording_features  # noqa: E402
from src.pipeline.filters import filter_recording_signals  # noqa: E402
from src.pipeline.loader import (  # noqa: E402
    discover_recording_ids,
    load_config,
    load_recording,
)
from src.pipeline.windowing import (  # noqa: E402
    compute_recording_ahi,
    create_windows_metadata,
)

_KB_PER_MB = 1024.0
_METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "recording_id",
        "window_idx",
        "start_sec",
        "end_sec",
        "sleep_stage",
        "is_wake",
        "overlap_sec",
        "label",
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the UCDDB streaming data pipeline runner.

    Args:
        argv: Optional list of CLI argument strings. Defaults to ``sys.argv[1:]``
            when ``None``.

    Returns:
        Parsed ``argparse.Namespace`` with ``config`` (str) and ``force`` (bool).
    """
    parser = argparse.ArgumentParser(
        description="Run the end-to-end UCDDB sleep apnea feature extraction and splitting pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML pipeline configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute per-Recording Parquet files even if cached files already exist.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _set_random_seeds(seed: int) -> None:
    """Set deterministic random seeds for ``random`` and ``numpy``."""
    random.seed(seed)
    np.random.seed(seed)


def _get_peak_ram_mb() -> float:
    """Return peak resident set size (RSS) of the current process in megabytes."""
    max_rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max_rss_kb / _KB_PER_MB


def _assert_dataframe_finite(df: pd.DataFrame, context_label: str) -> None:
    """Verify that all numeric columns of ``df`` contain zero ``NaN`` or ``Inf`` values."""
    numeric_matrix = df.select_dtypes(include=[np.number]).to_numpy()
    if not np.all(np.isfinite(numeric_matrix)):
        raise ValueError(f"Non-finite (NaN/Inf) values detected in {context_label}.")


def _process_single_recording(
    recording_id: str,
    config: Mapping[str, Any],
    features_dir: Path,
    *,
    force: bool,
) -> tuple[pd.DataFrame, RecordingSummary]:
    """Stream a single Recording, compute or load its unscaled Feature vector table, and free waveforms."""
    rec = load_recording(recording_id, config)
    ahi = compute_recording_ahi(rec, config)
    tier = assign_ahi_severity_tier(ahi, config)

    parquet_path = features_dir / f"{recording_id}.parquet"
    if parquet_path.is_file() and not force:
        del rec
        gc.collect()
        rec_df = pd.read_parquet(parquet_path)
        _assert_dataframe_finite(rec_df, str(parquet_path))
    else:
        filtered = filter_recording_signals(rec.signals, rec.sampling_rates, config)
        windows = create_windows_metadata(rec, config)
        rec_df = extract_recording_features(rec, filtered, windows, config)
        _assert_dataframe_finite(rec_df, f"extracted features for {recording_id}")
        rec_df.to_parquet(parquet_path, index=False)
        del rec, filtered, windows
        gc.collect()

    n_windows = int(len(rec_df))
    n_apnea_windows = int(rec_df["label"].sum()) if "label" in rec_df.columns else 0
    summary = RecordingSummary(
        recording_id=recording_id,
        ahi=ahi,
        severity_tier=tier,
        n_windows=n_windows,
        n_apnea_windows=n_apnea_windows,
    )
    return rec_df, summary


def run_pipeline(
    config_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Execute the end-to-end streaming UCDDB data pipeline and write dataset artifacts.

    Loads ``config_path``, seeds ``random`` and ``numpy``, streams each UCDDB Recording
    sequentially through ``loader`` -> ``filters`` -> ``windowing`` -> ``features`` (with
    ``gc.collect()`` after each Recording), caches per-Recording Parquet tables under
    ``config['paths']['features_dir']``, executes stratified Recording splitting and
    train-only ``StandardScaler`` scaling via ``data_split``, and writes a dataset summary
    report to ``config['paths']['summary_path']``.

    Args:
        config_path: Path to ``config.yaml``.
        force: When ``True``, recomputes per-Recording Parquet files even if cached on disk.

    Returns:
        Summary dictionary persisted to ``outputs/pipeline_summary.json``.
    """
    config = load_config(config_path)
    seed = int(config["seed"])
    _set_random_seeds(seed)

    paths_cfg: Mapping[str, Any] = config["paths"]
    raw_data_dir = Path(str(paths_cfg["raw_data_dir"]))
    processed_dir = Path(str(paths_cfg["processed_dir"]))
    features_dir = Path(str(paths_cfg.get("features_dir", processed_dir / "features")))
    summary_path = Path(str(paths_cfg["summary_path"]))

    features_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    recording_ids = discover_recording_ids(raw_data_dir)
    feature_dfs_by_id: dict[str, pd.DataFrame] = {}
    recording_summaries: list[RecordingSummary] = []

    for rec_id in recording_ids:
        rec_df, rec_summary = _process_single_recording(
            recording_id=rec_id,
            config=config,
            features_dir=features_dir,
            force=force,
        )
        feature_dfs_by_id[rec_id] = rec_df
        recording_summaries.append(rec_summary)
        gc.collect()

    splits = create_recording_splits(recording_summaries, config)
    scaled_splits, _ = build_and_scale_splits(
        feature_dfs_by_id=feature_dfs_by_id,
        splits=splits,
        config=config,
        recording_summaries=recording_summaries,
    )

    for split_name, split_df in scaled_splits.items():
        _assert_dataframe_finite(split_df, f"scaled {split_name}.parquet")

    sample_df = scaled_splits["train"]
    feature_columns = [c for c in sample_df.columns if c not in _METADATA_COLUMNS]

    splits_report: dict[str, dict[str, Any]] = {}
    windows_per_split: dict[str, int] = {}
    positive_ratios_per_split: dict[str, float] = {}

    for split_name in ("train", "val", "test"):
        df_s = scaled_splits[split_name]
        n_win = int(len(df_s))
        n_pos = int(df_s["label"].sum()) if n_win > 0 and "label" in df_s.columns else 0
        pos_ratio = float(n_pos / n_win) if n_win > 0 else 0.0
        windows_per_split[split_name] = n_win
        positive_ratios_per_split[split_name] = pos_ratio
        splits_report[split_name] = {
            "n_recordings": len(splits[split_name]),
            "n_windows": n_win,
            "n_apnea_windows": n_pos,
            "positive_apnea_window_ratio": pos_ratio,
        }

    peak_ram_mb = _get_peak_ram_mb()
    summary_payload: dict[str, Any] = {
        "seed": seed,
        "total_recordings": len(recording_ids),
        "feature_count": len(feature_columns),
        "peak_ram_mb": peak_ram_mb,
        "windows_per_split": windows_per_split,
        "positive_apnea_window_ratios": positive_ratios_per_split,
        "splits": splits_report,
    }

    summary_path.write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    return summary_payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``scripts/run_pipeline.py``.

    Args:
        argv: Optional list of CLI argument strings.

    Returns:
        Process exit code (``0`` on success).
    """
    args = parse_args(argv)
    run_pipeline(config_path=args.config, force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
