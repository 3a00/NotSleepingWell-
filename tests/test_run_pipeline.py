"""Tests for the end-to-end streaming pipeline runner (`scripts/run_pipeline.py`)."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from scripts.run_pipeline import main, parse_args, run_pipeline
from src.pipeline.loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _create_mini_ucddb_dataset(raw_dir: Path, n_records_sec: int = 390) -> list[str]:
    """Create a 12-Recording mini UCDDB dataset in ``raw_dir`` spanning all 3 AHI severity tiers."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_rec = REPO_ROOT / "database" / "files" / "ucddb002.rec"

    with source_rec.open("rb") as fp:
        general_hdr = bytearray(fp.read(256))
        header_bytes = int(general_hdr[184:192].decode("ascii").strip())
        record_dur = float(general_hdr[244:252].decode("ascii").strip())
        n_signals = int(general_hdr[252:256].decode("ascii").strip())
        sig_hdr = fp.read(n_signals * 256)

        samples_offset = n_signals * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
        samples_per_record = [
            int(sig_hdr[samples_offset + i * 8 : samples_offset + (i + 1) * 8].decode("ascii").strip())
            for i in range(n_signals)
        ]
        bytes_per_record = sum(samples_per_record) * 2
        n_edf_records = int(round(n_records_sec / record_dur))
        general_hdr[236:244] = f"{n_edf_records:<8d}".encode("ascii")

        fp.seek(header_bytes)
        data_bytes = fp.read(n_edf_records * bytes_per_record)

    # 13 Windows of 30s: 1 Wake Window (stage 0) + 12 Sleep Windows (stage 2) = 0.1 hours of definite sleep
    stage_text = "0\n" + ("2\n" * 12)

    # 3 tiers x 4 Recordings = 12 Recordings total:
    # Tier 0 (0 events -> AHI 0.0), Tier 1 (2 events -> AHI 20.0), Tier 2 (4 events -> AHI 40.0)
    #ucddb002 starts at 00.11.04 (00:11:04)
    event_rows = [
        "00:11:35  APNEA-O       10   92   4.0    60    8",
        "00:12:05  HYP-O         12   93   3.5    62    7",
        "00:12:35  APNEA-C       15   91   5.0    58   10",
        "00:13:05  HYP-C         10   94   3.0    64    6",
    ]

    rec_ids: list[str] = []
    for idx in range(12):
        rec_id = f"ucddb{idx + 2:03d}"
        rec_ids.append(rec_id)
        tier = idx % 3
        n_events = tier * 2  # 0, 2, or 4 events

        rec_file = raw_dir / f"{rec_id}.rec"
        rec_file.write_bytes(bytes(general_hdr) + sig_hdr + data_bytes)

        stage_file = raw_dir / f"{rec_id}_stage.txt"
        stage_file.write_text(stage_text, encoding="latin-1")

        respevt_lines = [
            "Time      Type      PB  Duration Low  %Drop  Snore Arousal",
            *event_rows[:n_events],
            "",
        ]
        respevt_file = raw_dir / f"{rec_id}_respevt.txt"
        respevt_file.write_text("\n".join(respevt_lines), encoding="latin-1")

    return sorted(rec_ids)


def test_parse_args_defaults_and_overrides() -> None:
    """CLI parser sets default config.yaml and force=False, and respects --config and --force."""
    default_args = parse_args([])
    assert Path(default_args.config) == Path("config.yaml")
    assert default_args.force is False

    custom_args = parse_args(["--config", "custom_cfg.yaml", "--force"])
    assert Path(custom_args.config) == Path("custom_cfg.yaml")
    assert custom_args.force is True


def test_run_pipeline_streams_recordings_caches_parquets_and_writes_summary(
    tmp_path: Path,
) -> None:
    """run_pipeline writes per-Recording Parquets, reuses cache unless --force, and outputs splits & summary."""
    raw_dir = tmp_path / "raw"
    rec_ids = _create_mini_ucddb_dataset(raw_dir)

    base_config = load_config(CONFIG_PATH)
    cfg = deepcopy(base_config)
    processed_dir = tmp_path / "processed"
    features_dir = processed_dir / "features"
    splits_path = processed_dir / "splits.json"
    scaler_path = processed_dir / "scaler.joblib"
    summary_path = tmp_path / "outputs" / "pipeline_summary.json"

    cfg["paths"] = {
        "raw_data_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "features_dir": str(features_dir),
        "splits_path": str(splits_path),
        "scaler_path": str(scaler_path),
        "summary_path": str(summary_path),
    }
    cfg["splitting"]["train_ratio"] = 0.50
    cfg["splitting"]["val_ratio"] = 0.25
    cfg["splitting"]["test_ratio"] = 0.25

    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    summary = run_pipeline(cfg_path, force=False)

    # 1. Verify per-Recording Parquet files exist and contain zero NaN/Inf
    for rec_id in rec_ids:
        pq_file = features_dir / f"{rec_id}.parquet"
        assert pq_file.is_file()
        rec_df = pd.read_parquet(pq_file)
        assert len(rec_df) == 13  # 1 Wake + 12 Sleep Windows in unscaled per-Recording Parquet
        num_arr = rec_df.select_dtypes(include=[np.number]).to_numpy()
        assert np.all(np.isfinite(num_arr))

    # 2. Verify splits.json, scaler.joblib, and scaled {train,val,test}.parquet exist with zero NaN/Inf
    assert splits_path.is_file()
    assert scaler_path.is_file()
    for split_name, expected_recs in (("train", 6), ("val", 3), ("test", 3)):
        split_pq = processed_dir / f"{split_name}.parquet"
        assert split_pq.is_file()
        split_df = pd.read_parquet(split_pq)
        # Wake Windows excluded (12 non-wake Windows per Recording)
        assert len(split_df) == expected_recs * 12
        assert (split_df["sleep_stage"] != 0).all()
        num_arr = split_df.select_dtypes(include=[np.number]).to_numpy()
        assert np.all(np.isfinite(num_arr))

    # 3. Verify outputs/pipeline_summary.json matches returned summary and contains required fields
    assert summary_path.is_file()
    disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert disk_summary == summary
    assert summary["total_recordings"] == 12
    assert summary["feature_count"] == 88
    assert summary["peak_ram_mb"] > 0.0
    assert set(summary["splits"].keys()) == {"train", "val", "test"}
    for split_name, expected_recs in (("train", 6), ("val", 3), ("test", 3)):
        s_info = summary["splits"][split_name]
        assert s_info["n_recordings"] == expected_recs
        assert s_info["n_windows"] == expected_recs * 12
        assert 0.0 < s_info["positive_apnea_window_ratio"] < 1.0

    # 4. Verify caching behavior: without --force, existing per-Recording Parquet mtime is preserved;
    # with --force (via CLI main), per-Recording Parquet is recomputed.
    target_pq = features_dir / "ucddb002.parquet"
    old_mtime_ns = target_pq.stat().st_mtime_ns

    exit_code_cached = main(["--config", str(cfg_path)])
    assert exit_code_cached == 0
    assert target_pq.stat().st_mtime_ns == old_mtime_ns

    # Touch mtime backwards and run with --force to verify recomputation
    import os

    os.utime(target_pq, ns=(old_mtime_ns - 10_000_000_000, old_mtime_ns - 10_000_000_000))
    backdated_mtime_ns = target_pq.stat().st_mtime_ns
    exit_code_forced = main(["--config", str(cfg_path), "--force"])
    assert exit_code_forced == 0
    assert target_pq.stat().st_mtime_ns > backdated_mtime_ns
