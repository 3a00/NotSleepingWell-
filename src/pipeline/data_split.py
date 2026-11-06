"""Stratified Recording-level splitting and train-only feature scaling."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

_DEFAULT_AHI_MODERATE_THRESHOLD = 15.0
_DEFAULT_AHI_SEVERE_THRESHOLD = 30.0
_SEVERITY_TIER_NORMAL_MILD = 0
_SEVERITY_TIER_MODERATE = 1
_SEVERITY_TIER_SEVERE = 2
_RATIO_SUM_TOLERANCE = 1e-6
_WAKE_SLEEP_STAGE = 0
_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
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


@dataclass(frozen=True)
class RecordingSummary:
    """Recording-level clinical AHI, severity tier, and Window count summary.

    Attributes:
        recording_id: Canonical Recording identifier (e.g., ``'ucddb002'``).
        ahi: Clinical Apnea-Hypopnea Index (Apnea events per hour of sleep).
        severity_tier: Optional integer severity tier (``0`` Normal/Mild, ``1`` Moderate,
            ``2`` Severe). Computed automatically when ``None``.
        n_windows: Optional total Window count for the Recording.
        n_apnea_windows: Optional positive Apnea Window count for the Recording.
    """

    recording_id: str
    ahi: float
    severity_tier: int | None = None
    n_windows: int = 0
    n_apnea_windows: int = 0


RecordingSummaryInput = (
    Sequence[RecordingSummary | Mapping[str, Any]] | Mapping[str, float | Mapping[str, Any]]
)


def assign_ahi_severity_tier(
    ahi: float,
    config: Mapping[str, Any] | None = None,
) -> int:
    """Map Recording AHI to clinical severity tier for stratified splitting.

    Maps AHI (Apnea events per hour of sleep) to three integer severity tiers:
    ``0`` for ``< moderate_threshold`` (Normal/Mild, default ``< 15.0``),
    ``1`` for ``moderate_threshold <= ahi < severe_threshold`` (Moderate, default ``15.0–30.0``),
    and ``2`` for ``>= severe_threshold`` (Severe, default ``>= 30.0``).

    Args:
        ahi: Non-negative finite Recording AHI value.
        config: Optional pipeline configuration dictionary loaded from ``config.yaml``.
            When provided, reads ``splitting.ahi_moderate_threshold`` and
            ``splitting.ahi_severe_threshold``.

    Returns:
        Integer severity tier in ``{0, 1, 2}``.

    Raises:
        ValueError: If ``ahi`` is negative or non-finite, or if configured thresholds
            do not satisfy ``0 < moderate_threshold < severe_threshold``.
    """
    ahi_val = float(ahi)
    if not math.isfinite(ahi_val) or ahi_val < 0.0:
        raise ValueError(f"AHI must be finite and >= 0.0, got {ahi}.")

    if config is not None and "splitting" in config:
        split_cfg: Mapping[str, Any] = config["splitting"]
        moderate_thresh = float(
            split_cfg.get("ahi_moderate_threshold", _DEFAULT_AHI_MODERATE_THRESHOLD)
        )
        severe_thresh = float(
            split_cfg.get("ahi_severe_threshold", _DEFAULT_AHI_SEVERE_THRESHOLD)
        )
    else:
        moderate_thresh = _DEFAULT_AHI_MODERATE_THRESHOLD
        severe_thresh = _DEFAULT_AHI_SEVERE_THRESHOLD

    if moderate_thresh <= 0.0 or severe_thresh <= moderate_thresh:
        raise ValueError(
            f"Thresholds must satisfy 0 < moderate ({moderate_thresh}) < severe ({severe_thresh})."
        )

    if ahi_val < moderate_thresh:
        return _SEVERITY_TIER_NORMAL_MILD
    if ahi_val < severe_thresh:
        return _SEVERITY_TIER_MODERATE
    return _SEVERITY_TIER_SEVERE


def _summary_from_mapping(
    rec_id: str,
    mapping: Mapping[str, Any],
    config: Mapping[str, Any],
) -> RecordingSummary:
    """Convert a dictionary summary entry into a ``RecordingSummary``."""
    ahi = float(mapping["ahi"])
    tier_raw = mapping.get("severity_tier")
    tier = (
        int(tier_raw)
        if tier_raw is not None
        else assign_ahi_severity_tier(ahi, config)
    )
    return RecordingSummary(
        recording_id=rec_id,
        ahi=ahi,
        severity_tier=tier,
        n_windows=int(mapping.get("n_windows", 0)),
        n_apnea_windows=int(mapping.get("n_apnea_windows", 0)),
    )


def _normalize_recording_summaries(
    recording_summaries: RecordingSummaryInput,
    config: Mapping[str, Any],
) -> list[RecordingSummary]:
    """Normalize supported summary representations into sorted ``RecordingSummary`` items."""
    normalized: list[RecordingSummary] = []

    if isinstance(recording_summaries, Mapping):
        for rec_id, val in recording_summaries.items():
            if isinstance(val, Mapping):
                normalized.append(_summary_from_mapping(str(rec_id), val, config))
            else:
                ahi = float(val)
                normalized.append(
                    RecordingSummary(
                        recording_id=str(rec_id),
                        ahi=ahi,
                        severity_tier=assign_ahi_severity_tier(ahi, config),
                    )
                )
    else:
        for item in recording_summaries:
            if isinstance(item, RecordingSummary):
                tier = (
                    item.severity_tier
                    if item.severity_tier is not None
                    else assign_ahi_severity_tier(item.ahi, config)
                )
                normalized.append(
                    RecordingSummary(
                        recording_id=item.recording_id,
                        ahi=float(item.ahi),
                        severity_tier=tier,
                        n_windows=int(item.n_windows),
                        n_apnea_windows=int(item.n_apnea_windows),
                    )
                )
            elif isinstance(item, Mapping):
                normalized.append(
                    _summary_from_mapping(str(item["recording_id"]), item, config)
                )
            else:
                raise TypeError(
                    f"Unsupported recording summary element type: {type(item).__name__}"
                )

    seen_ids: set[str] = set()
    for summary in normalized:
        if summary.recording_id in seen_ids:
            raise ValueError(
                f"Duplicate recording_id in recording_summaries: {summary.recording_id!r}."
            )
        seen_ids.add(summary.recording_id)

    normalized.sort(key=lambda s: s.recording_id)
    return normalized


def _assert_disjoint_splits(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
) -> None:
    """Verify that train, val, and test Recording ID sets have zero overlap."""
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    if not (
        train_set.isdisjoint(val_set)
        and train_set.isdisjoint(test_set)
        and val_set.isdisjoint(test_set)
    ):
        raise ValueError("Recording ID overlap detected between train, val, and test splits.")


def create_recording_splits(
    recording_summaries: RecordingSummaryInput,
    config: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Deterministically split Recordings into stratified ``train``, ``val``, and ``test`` sets.

    Uses ``seed`` and ``splitting.{train_ratio,val_ratio,test_ratio}`` from ``config``
    (giving **17 train / 4 val / 4 test** across the 25 UCDDB Recordings) and stratifies
    by AHI severity tier. Enforces pairwise disjoint Recording ID sets.

    Args:
        recording_summaries: Sequence of ``RecordingSummary`` or dicts with
            ``recording_id`` and ``ahi`` (or a mapping ``{recording_id: ahi}``).
        config: Pipeline configuration dictionary loaded from ``config.yaml``.

    Returns:
        Dictionary with keys ``'train'``, ``'val'``, and ``'test'`` mapping to sorted
        lists of Recording IDs.

    Raises:
        ValueError: If split ratios do not sum to ``1.0``, if fewer than 3 Recordings
            are provided, or if any overlap occurs between splits.
    """
    seed = int(config["seed"])
    split_cfg: Mapping[str, Any] = config["splitting"]
    train_ratio = float(split_cfg["train_ratio"])
    val_ratio = float(split_cfg["val_ratio"])
    test_ratio = float(split_cfg["test_ratio"])

    if train_ratio <= 0.0 or val_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError(
            f"Split ratios must all be > 0, got ({train_ratio}, {val_ratio}, {test_ratio})."
        )
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, abs_tol=_RATIO_SUM_TOLERANCE):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}.")

    summaries = _normalize_recording_summaries(recording_summaries, config)
    n_total = len(summaries)
    if n_total < 3:
        raise ValueError(f"At least 3 Recordings required to split, got {n_total}.")

    n_val = max(1, int(round(n_total * val_ratio)))
    n_test = max(1, int(round(n_total * test_ratio)))
    n_holdout = n_val + n_test
    n_train = n_total - n_holdout
    if n_train < 1:
        raise ValueError(
            f"Computed n_train={n_train} < 1 for n_total={n_total}, ratios=({train_ratio}, {val_ratio}, {test_ratio})."
        )

    rec_ids = [s.recording_id for s in summaries]
    tiers = [
        s.severity_tier
        if s.severity_tier is not None
        else assign_ahi_severity_tier(s.ahi, config)
        for s in summaries
    ]

    train_ids_raw, holdout_ids_raw, _, holdout_tiers = train_test_split(
        rec_ids,
        tiers,
        test_size=n_holdout,
        random_state=seed,
        stratify=tiers,
    )
    val_ids_raw, test_ids_raw = train_test_split(
        holdout_ids_raw,
        test_size=n_test,
        random_state=seed,
        stratify=holdout_tiers,
    )

    train_ids = sorted(str(rid) for rid in train_ids_raw)
    val_ids = sorted(str(rid) for rid in val_ids_raw)
    test_ids = sorted(str(rid) for rid in test_ids_raw)

    if len(train_ids) != n_train or len(val_ids) != n_val or len(test_ids) != n_test:
        raise AssertionError(
            f"Expected split sizes ({n_train}, {n_val}, {n_test}), "
            f"got ({len(train_ids)}, {len(val_ids)}, {len(test_ids)})."
        )
    _assert_disjoint_splits(train_ids, val_ids, test_ids)
    if (set(train_ids) | set(val_ids) | set(test_ids)) != set(rec_ids):
        raise AssertionError("Split Recording IDs do not cover the exact input set.")

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def _filter_wake_windows(df: pd.DataFrame, exclude_wake: bool) -> pd.DataFrame:
    """Filter out Wake Windows (``sleep_stage == 0``) when ``exclude_wake`` is enabled."""
    if not exclude_wake:
        return df.copy()
    if "sleep_stage" not in df.columns:
        raise ValueError("DataFrame must contain 'sleep_stage' column.")
    mask = df["sleep_stage"] != _WAKE_SLEEP_STAGE
    return df.loc[mask].reset_index(drop=True)


def build_and_scale_splits(
    feature_dfs_by_id: Mapping[str, pd.DataFrame],
    splits: Mapping[str, Sequence[str]],
    config: Mapping[str, Any],
    recording_summaries: RecordingSummaryInput | None = None,
) -> tuple[dict[str, pd.DataFrame], StandardScaler]:
    """Concatenate per-Recording feature tables into splits, fit train-only scaler, and persist.

    1. Validates that ``train``, ``val``, and ``test`` Recording ID sets are pairwise disjoint.
    2. Concatenates per-Recording DataFrames for each split without mutating the caller's
       DataFrames (or raw per-Recording Parquet files).
    3. Filters out Wake Windows (``sleep_stage == 0``) when ``config['windowing']['exclude_wake']``
       is ``True``.
    4. Fits ``sklearn.preprocessing.StandardScaler`` strictly on the ``train`` feature columns
       and transforms ``train``, ``val``, and ``test`` feature matrices to ``float32``.
    5. Writes ``splits.json``, ``scaler.joblib``, and ``{train,val,test}.parquet`` to the paths
       configured in ``config['paths']``.

    Args:
        feature_dfs_by_id: Mapping from ``recording_id`` to unscaled per-Recording
            ``pd.DataFrame``.
        splits: Mapping with keys ``'train'``, ``'val'``, and ``'test'`` listing Recording IDs.
        config: Pipeline configuration dictionary loaded from ``config.yaml``.
        recording_summaries: Optional Recording summaries containing AHI and severity tier
            metadata to include in ``splits.json``.

    Returns:
        Tuple ``(scaled_splits, scaler)`` where ``scaled_splits`` maps ``'train'``, ``'val'``,
        and ``'test'`` to scaled ``pd.DataFrame`` tables and ``scaler`` is the train-fitted
        ``StandardScaler``.

    Raises:
        ValueError: If split sets overlap, required Recording DataFrames are missing, feature
            columns are missing or empty, or ``train`` has zero rows after filtering.
    """
    for split_name in _SPLIT_NAMES:
        if split_name not in splits:
            raise ValueError(f"Missing required split key {split_name!r} in splits.")

    train_ids = [str(rid) for rid in splits["train"]]
    val_ids = [str(rid) for rid in splits["val"]]
    test_ids = [str(rid) for rid in splits["test"]]
    _assert_disjoint_splits(train_ids, val_ids, test_ids)

    exclude_wake = bool(config["windowing"].get("exclude_wake", True))

    summary_by_id: dict[str, RecordingSummary] = {}
    if recording_summaries is not None:
        for s in _normalize_recording_summaries(recording_summaries, config):
            summary_by_id[s.recording_id] = s

    split_dfs_raw: dict[str, pd.DataFrame] = {}
    recording_manifest: dict[str, dict[str, Any]] = {}
    feature_cols: list[str] | None = None

    for split_name, rec_id_list in (
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids),
    ):
        frames_for_split: list[pd.DataFrame] = []
        for rec_id in rec_id_list:
            if rec_id not in feature_dfs_by_id:
                raise ValueError(
                    f"Recording {rec_id!r} in split {split_name!r} not found in feature_dfs_by_id."
                )
            raw_df = feature_dfs_by_id[rec_id]
            cols = [c for c in raw_df.columns if c not in _METADATA_COLUMNS]
            if not cols:
                raise ValueError(f"No feature columns found in DataFrame for Recording {rec_id!r}.")
            if feature_cols is None:
                feature_cols = cols
            elif cols != feature_cols:
                raise ValueError(
                    f"Feature column mismatch for Recording {rec_id!r}: expected {feature_cols}, got {cols}."
                )

            filtered_df = _filter_wake_windows(raw_df, exclude_wake=exclude_wake)
            frames_for_split.append(filtered_df)

            n_total_win = int(len(raw_df))
            n_kept_win = int(len(filtered_df))
            n_apnea_total = int(raw_df["label"].sum()) if "label" in raw_df.columns else 0
            n_apnea_kept = int(filtered_df["label"].sum()) if "label" in filtered_df.columns else 0

            summary_item = summary_by_id.get(rec_id)
            ahi_val: float | None = summary_item.ahi if summary_item is not None else None
            tier_val: int | None = (
                summary_item.severity_tier
                if summary_item is not None and summary_item.severity_tier is not None
                else (assign_ahi_severity_tier(ahi_val, config) if ahi_val is not None else None)
            )
            recording_manifest[rec_id] = {
                "split": split_name,
                "ahi": ahi_val,
                "severity_tier": tier_val,
                "n_windows_total": n_total_win,
                "n_windows_kept": n_kept_win,
                "n_apnea_windows_total": n_apnea_total,
                "n_apnea_windows_kept": n_apnea_kept,
            }

        if not frames_for_split:
            raise ValueError(f"Split {split_name!r} has no Recordings.")
        split_dfs_raw[split_name] = pd.concat(frames_for_split, axis=0, ignore_index=True)

    if feature_cols is None or not feature_cols:
        raise ValueError("No feature columns discovered across splits.")
    if len(split_dfs_raw["train"]) == 0:
        raise ValueError("Train split has 0 Windows after Wake filtering; cannot fit StandardScaler.")

    # Fit StandardScaler strictly on the train split feature matrix (float64 precision for exact stats)
    scaler = StandardScaler()
    train_matrix_f64 = split_dfs_raw["train"][feature_cols].to_numpy(dtype=np.float64, copy=True)
    scaler.fit(train_matrix_f64)

    scaled_splits: dict[str, pd.DataFrame] = {}
    for split_name in _SPLIT_NAMES:
        df_out = split_dfs_raw[split_name].copy(deep=True)
        if len(df_out) > 0:
            raw_mat = df_out[feature_cols].to_numpy(dtype=np.float64, copy=True)
            scaled_mat = scaler.transform(raw_mat).astype(np.float32)
            df_out[feature_cols] = scaled_mat
        else:
            for col in feature_cols:
                df_out[col] = df_out[col].astype(np.float32)
        scaled_splits[split_name] = df_out

    paths_cfg: Mapping[str, Any] = config["paths"]
    processed_dir = Path(str(paths_cfg["processed_dir"]))
    splits_path = Path(str(paths_cfg.get("splits_path", processed_dir / "splits.json")))
    scaler_path = Path(str(paths_cfg.get("scaler_path", processed_dir / "scaler.joblib")))

    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    split_window_counts = {
        split_name: {
            "n_recordings": len(splits[split_name]),
            "n_windows": int(len(scaled_splits[split_name])),
            "n_apnea_windows": (
                int(scaled_splits[split_name]["label"].sum())
                if "label" in scaled_splits[split_name].columns
                else 0
            ),
        }
        for split_name in _SPLIT_NAMES
    }

    manifest_payload = {
        "seed": int(config["seed"]),
        "exclude_wake": exclude_wake,
        "feature_columns": feature_cols,
        "splits": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
        "split_counts": split_window_counts,
        "recordings": recording_manifest,
    }
    splits_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    joblib.dump(scaler, scaler_path)

    for split_name in _SPLIT_NAMES:
        parquet_path = processed_dir / f"{split_name}.parquet"
        scaled_splits[split_name].to_parquet(parquet_path, index=False)

    return scaled_splits, scaler
