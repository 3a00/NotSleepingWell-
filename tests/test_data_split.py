"""Tests for stratified Recording-level splitting and train-only feature scaling."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from src.pipeline.data_split import (
    RecordingSummary,
    assign_ahi_severity_tier,
    build_and_scale_splits,
    create_recording_splits,
)
from src.pipeline.loader import discover_recording_ids, load_config, load_recording
from src.pipeline.windowing import compute_recording_ahi

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _load_real_ucddb_summaries(config: dict[str, Any]) -> list[RecordingSummary]:
    """Compute real RecordingSummary entries across all 25 UCDDB Recordings."""
    ids = discover_recording_ids(config["paths"]["raw_data_dir"])
    summaries: list[RecordingSummary] = []
    for rec_id in ids:
        rec = load_recording(rec_id, config)
        ahi = compute_recording_ahi(rec, config)
        tier = assign_ahi_severity_tier(ahi, config)
        summaries.append(
            RecordingSummary(
                recording_id=rec_id,
                ahi=ahi,
                severity_tier=tier,
            )
        )
    return summaries


def test_assign_ahi_severity_tier_maps_thresholds_correctly() -> None:
    """AHI maps to severity tier 0 (<15 Normal/Mild), 1 (15-30 Moderate), and 2 (>=30 Severe)."""
    config = load_config(CONFIG_PATH)

    assert assign_ahi_severity_tier(0.0, config) == 0
    assert assign_ahi_severity_tier(14.99, config) == 0
    assert assign_ahi_severity_tier(15.0, config) == 1
    assert assign_ahi_severity_tier(29.99, config) == 1
    assert assign_ahi_severity_tier(30.0, config) == 2
    assert assign_ahi_severity_tier(55.3, config) == 2

    # Default thresholds when config is omitted
    assert assign_ahi_severity_tier(14.9) == 0
    assert assign_ahi_severity_tier(15.0) == 1
    assert assign_ahi_severity_tier(30.0) == 2

    with pytest.raises(ValueError, match="AHI"):
        assign_ahi_severity_tier(-1.0, config)


def test_create_recording_splits_produces_deterministic_disjoint_17_4_4_stratified_sets() -> None:
    """25 UCDDB Recordings split deterministically into 17 train / 4 val / 4 test with zero overlap and tier balance."""
    config = load_config(CONFIG_PATH)
    summaries = _load_real_ucddb_summaries(config)
    assert len(summaries) == 25

    splits = create_recording_splits(summaries, config)

    assert set(splits.keys()) == {"train", "val", "test"}
    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]

    assert len(train_ids) == 17
    assert len(val_ids) == 4
    assert len(test_ids) == 4

    # Pairwise disjoint sets and exact union of all 25 Recordings
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    assert len(train_set) == 17
    assert len(val_set) == 4
    assert len(test_set) == 4
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == {s.recording_id for s in summaries}

    # Every severity tier (0, 1, 2) must be represented in train, val, and test
    tier_by_id = {s.recording_id: assign_ahi_severity_tier(s.ahi, config) for s in summaries}
    for split_name, split_ids in splits.items():
        tiers_present = {tier_by_id[rid] for rid in split_ids}
        assert tiers_present == {0, 1, 2}, f"Split {split_name!r} missing severity tiers: {tiers_present}"

    # Deterministic across repeated calls and input orderings, and accepts dict summaries / mapping
    splits_reversed = create_recording_splits(list(reversed(summaries)), config)
    assert splits_reversed == splits

    dict_summaries = [{"recording_id": s.recording_id, "ahi": s.ahi} for s in summaries]
    assert create_recording_splits(dict_summaries, config) == splits

    ahi_mapping = {s.recording_id: s.ahi for s in summaries}
    assert create_recording_splits(ahi_mapping, config) == splits


def _make_recording_feature_df(
    recording_id: str,
    sleep_stages: list[int],
    f1_values: list[float],
    f2_values: list[float],
) -> pd.DataFrame:
    """Create a synthetic per-Recording feature DataFrame with canonical metadata columns."""
    n = len(sleep_stages)
    return pd.DataFrame(
        {
            "recording_id": [recording_id] * n,
            "window_idx": np.arange(n, dtype=np.int64),
            "start_sec": (np.arange(n, dtype=np.float64) * 30.0),
            "end_sec": (np.arange(1, n + 1, dtype=np.float64) * 30.0),
            "sleep_stage": np.asarray(sleep_stages, dtype=np.int8),
            "is_wake": np.asarray([s == 0 for s in sleep_stages], dtype=np.bool_),
            "overlap_sec": np.asarray([10.0 if i % 2 == 1 else 0.0 for i in range(n)], dtype=np.float64),
            "label": np.asarray([1 if i % 2 == 1 else 0 for i in range(n)], dtype=np.int8),
            "feat_alpha": np.asarray(f1_values, dtype=np.float32),
            "feat_beta": np.asarray(f2_values, dtype=np.float32),
        }
    )


def test_build_and_scale_splits_enforces_train_only_scaler_wake_exclusion_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    """Scaler fits strictly on train non-Wake Windows, is invariant to val/test perturbations, and writes artifacts."""
    base_config = load_config(CONFIG_PATH)
    config = deepcopy(base_config)
    config["paths"]["processed_dir"] = str(tmp_path)
    config["paths"]["splits_path"] = str(tmp_path / "splits.json")
    config["paths"]["scaler_path"] = str(tmp_path / "scaler.joblib")
    config["windowing"]["exclude_wake"] = True

    # Train Recordings have Wake Windows with extreme sentinel values (9999.0) that MUST be excluded
    # before fitting the scaler, and non-Wake Windows with known exact mean & population std:
    # non-Wake feat_alpha across train = [10.0, 20.0, 30.0, 40.0] -> mean=25.0, std=sqrt(125.0)
    # non-Wake feat_beta across train  = [2.0,  4.0,  6.0,  8.0]  -> mean=5.0,  std=sqrt(5.0)
    df_train_1 = _make_recording_feature_df(
        "ucddb002",
        sleep_stages=[0, 2, 3],
        f1_values=[9999.0, 10.0, 20.0],
        f2_values=[-9999.0, 2.0, 4.0],
    )
    df_train_2 = _make_recording_feature_df(
        "ucddb003",
        sleep_stages=[0, 1, 4],
        f1_values=[8888.0, 30.0, 40.0],
        f2_values=[-8888.0, 6.0, 8.0],
    )
    df_val = _make_recording_feature_df(
        "ucddb005",
        sleep_stages=[0, 2, 2],
        f1_values=[500.0, 25.0, 50.0],
        f2_values=[500.0, 5.0, 10.0],
    )
    df_test = _make_recording_feature_df(
        "ucddb006",
        sleep_stages=[0, 3, 1],
        f1_values=[700.0, 0.0, 25.0],
        f2_values=[700.0, 0.0, 5.0],
    )

    feature_dfs = {
        "ucddb002": df_train_1,
        "ucddb003": df_train_2,
        "ucddb005": df_val,
        "ucddb006": df_test,
    }
    splits = {
        "train": ["ucddb002", "ucddb003"],
        "val": ["ucddb005"],
        "test": ["ucddb006"],
    }
    summaries = [
        RecordingSummary("ucddb002", ahi=23.73, severity_tier=1),
        RecordingSummary("ucddb003", ahi=51.26, severity_tier=2),
        RecordingSummary("ucddb005", ahi=13.23, severity_tier=0),
        RecordingSummary("ucddb006", ahi=31.34, severity_tier=2),
    ]

    scaled_dfs, scaler = build_and_scale_splits(
        feature_dfs_by_id=feature_dfs,
        splits=splits,
        config=config,
        recording_summaries=summaries,
    )

    # Caller's raw per-Recording DataFrames must remain untouched (Wake rows and unscaled values preserved)
    assert len(df_train_1) == 3
    assert float(df_train_1["feat_alpha"].iloc[0]) == pytest.approx(9999.0)

    # Wake Windows (sleep_stage == 0) must be filtered out of train, val, and test
    for split_name, df_split in scaled_dfs.items():
        assert (df_split["sleep_stage"] != 0).all(), f"Wake Window leaked into {split_name}"
        assert (~df_split["is_wake"]).all(), f"is_wake True leaked into {split_name}"

    assert len(scaled_dfs["train"]) == 4
    assert len(scaled_dfs["val"]) == 2
    assert len(scaled_dfs["test"]) == 2

    # Scaler mean_ and scale_ must match the unscaled non-Wake train feature matrix exactly
    expected_mean = np.array([25.0, 5.0], dtype=np.float64)
    expected_scale = np.array([np.sqrt(125.0), np.sqrt(5.0)], dtype=np.float64)
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scaler.scale_, expected_scale, rtol=1e-6, atol=1e-6)

    # Scaled train features have zero mean and unit population std, stored as float32
    for col in ("feat_alpha", "feat_beta"):
        assert scaled_dfs["train"][col].dtype == np.float32
        assert float(scaled_dfs["train"][col].mean()) == pytest.approx(0.0, abs=1e-6)
        assert float(np.std(scaled_dfs["train"][col].to_numpy(dtype=np.float64), ddof=0)) == pytest.approx(
            1.0, abs=1e-6
        )

    # Verify val row 0 (feat_alpha=25.0, feat_beta=5.0) transforms to (0.0, 0.0) using train stats
    assert float(scaled_dfs["val"]["feat_alpha"].iloc[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(scaled_dfs["val"]["feat_beta"].iloc[0]) == pytest.approx(0.0, abs=1e-6)

    # Zero leakage invariance test: wildly perturbing val and test features must NOT change scaler or train
    perturbed_dfs = {
        "ucddb002": df_train_1.copy(deep=True),
        "ucddb003": df_train_2.copy(deep=True),
        "ucddb005": df_val.copy(deep=True),
        "ucddb006": df_test.copy(deep=True),
    }
    perturbed_dfs["ucddb005"]["feat_alpha"] = np.array([1e6, -5e5, 9e5], dtype=np.float32)
    perturbed_dfs["ucddb006"]["feat_beta"] = np.array([-3e6, 4e6, 8e6], dtype=np.float32)

    scaled_perturbed, scaler_perturbed = build_and_scale_splits(
        feature_dfs_by_id=perturbed_dfs,
        splits=splits,
        config=config,
        recording_summaries=summaries,
    )
    np.testing.assert_array_equal(scaler.mean_, scaler_perturbed.mean_)
    np.testing.assert_array_equal(scaler.scale_, scaler_perturbed.scale_)
    pd.testing.assert_frame_equal(scaled_dfs["train"], scaled_perturbed["train"])

    # Verify persisted files: splits.json, scaler.joblib, and {train,val,test}.parquet
    splits_path = tmp_path / "splits.json"
    scaler_path = tmp_path / "scaler.joblib"
    assert splits_path.is_file()
    assert scaler_path.is_file()
    for split_name in ("train", "val", "test"):
        pq_path = tmp_path / f"{split_name}.parquet"
        assert pq_path.is_file()
        loaded_df = pd.read_parquet(pq_path)
        pd.testing.assert_frame_equal(loaded_df, scaled_perturbed[split_name])

    loaded_scaler = joblib.load(scaler_path)
    np.testing.assert_array_equal(loaded_scaler.mean_, scaler.mean_)
    np.testing.assert_array_equal(loaded_scaler.scale_, scaler.scale_)

    manifest = json.loads(splits_path.read_text(encoding="utf-8"))
    assert manifest["splits"] == splits
    assert "recordings" in manifest
    assert manifest["recordings"]["ucddb002"]["severity_tier"] == 1
    assert manifest["recordings"]["ucddb002"]["n_windows_total"] == 3
    assert manifest["recordings"]["ucddb002"]["n_windows_kept"] == 2

    # When exclude_wake is False, Wake Windows are retained
    config_keep_wake = deepcopy(config)
    config_keep_wake["windowing"]["exclude_wake"] = False
    scaled_with_wake, _ = build_and_scale_splits(
        feature_dfs_by_id=feature_dfs,
        splits=splits,
        config=config_keep_wake,
        recording_summaries=summaries,
    )
    assert len(scaled_with_wake["train"]) == 6
    assert (scaled_with_wake["train"]["sleep_stage"] == 0).sum() == 2
