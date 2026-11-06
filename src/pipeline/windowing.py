"""Sleep stage-aligned Windowing, Apnea event overlap labeling, and Recording AHI."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from src.pipeline.loader import ApneaEvent, RecordingData

# Fixed UCDDB dataset format constants defined in CONTEXT.md (<id>_stage.txt & <id>_respevt.txt)
_CANONICAL_VALID_EVENTS: frozenset[str] = frozenset(
    {"APNEA-O", "APNEA-C", "APNEA-M", "HYP-O", "HYP-C", "HYP-M"}
)
_UCDDB_STAGE_INTERVAL_SEC = 30.0
_WAKE_SLEEP_STAGE = 0
_DEFINITE_SLEEP_STAGES: tuple[int, ...] = (1, 2, 3, 4, 5)
_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class WindowRecord:
    """Metadata and binary Apnea event label for a single Recording Window.

    Attributes:
        recording_id: Parent Recording identifier (e.g., ``'ucddb002'``).
        window_idx: Zero-based Window index aligned 1-to-1 with ``<id>_stage.txt``.
        start_sec: Window start time in seconds relative to Recording start.
        end_sec: Window end time in seconds relative to Recording start.
        sleep_stage: Annotated Sleep stage code (``0``=Wake, ``1``=REM, ``2``-``5``=NREM,
            ``8``=Indeterminate).
        is_wake: ``True`` when ``sleep_stage == 0`` (Wake), ``False`` otherwise.
        overlap_sec: Merged overlap duration in seconds with valid Apnea events.
        label: Binary target (``1`` when ``overlap_sec >= overlap_threshold_sec``, else ``0``).
    """

    recording_id: str
    window_idx: int
    start_sec: float
    end_sec: float
    sleep_stage: int
    is_wake: bool
    overlap_sec: float
    label: int


def compute_window_overlap(
    w_start: float,
    w_end: float,
    events: Sequence[ApneaEvent],
    valid_events: Collection[str] | None = None,
) -> float:
    """Compute exact total overlap in seconds between a Window and valid Apnea events.

    Filters ``events`` to ``valid_events`` (defaulting to the canonical UCDDB Apnea
    event types ``APNEA-O/C/M`` and ``HYP-O/C/M``), clips each intersecting interval
    to ``[w_start, w_end]``, and merges overlapping intervals before summing so
    double-annotations cannot inflate overlap beyond ``w_end - w_start``.

    Args:
        w_start: Window start timestamp in seconds relative to Recording start.
        w_end: Window end timestamp in seconds relative to Recording start.
        events: Parsed respiratory annotations for the Recording.
        valid_events: Allowed Apnea event type labels. When ``None``, uses the
            canonical UCDDB Apnea event set (excluding ``PB`` and ``POSSIBLE``).

    Returns:
        Exact merged overlap duration in seconds within ``[0.0, w_end - w_start]``.

    Raises:
        ValueError: If ``w_end < w_start``.
    """
    if w_end < w_start:
        raise ValueError(f"w_end ({w_end}) must be >= w_start ({w_start}).")
    if w_end == w_start or not events:
        return 0.0

    allowed_types = (
        _CANONICAL_VALID_EVENTS if valid_events is None else frozenset(valid_events)
    )

    clipped: list[tuple[float, float]] = []
    for event in events:
        if event.event_type not in allowed_types:
            continue
        inter_start = max(w_start, event.start_sec)
        inter_end = min(w_end, event.end_sec)
        if inter_end > inter_start:
            clipped.append((inter_start, inter_end))

    if not clipped:
        return 0.0

    clipped.sort()
    merged_start, merged_end = clipped[0]
    total_overlap = 0.0

    for cur_start, cur_end in clipped[1:]:
        if cur_start <= merged_end:
            merged_end = max(merged_end, cur_end)
        else:
            total_overlap += merged_end - merged_start
            merged_start, merged_end = cur_start, cur_end

    total_overlap += merged_end - merged_start
    return float(total_overlap)


def create_windows_metadata(
    recording_data: RecordingData,
    config: Mapping[str, Any],
) -> list[WindowRecord]:
    """Generate Sleep stage-aligned ``WindowRecord`` metadata for a Recording.

    Slices the Recording into ``N = min(int(duration_sec // window_sec), len(sleep_stages))``
    non-overlapping Windows aligned 1-to-1 with ``recording_data.sleep_stages``,
    computes merged valid Apnea event overlap per Window, and assigns ``label = 1``
    when ``overlap_sec >= overlap_threshold_sec``.

    Args:
        recording_data: Loaded UCDDB Recording with ``duration_sec``, ``events``,
            and ``sleep_stages``.
        config: Pipeline configuration dictionary loaded from ``config.yaml``.

    Returns:
        Ordered list of ``WindowRecord`` instances for Windows ``0 .. N - 1``.

    Raises:
        ValueError: If ``window_sec <= 0``, ``window_sec`` does not match the 30-second
            ``<id>_stage.txt`` interval, ``stride_sec != window_sec``, or
            ``overlap_threshold_sec <= 0``.
    """
    windowing_cfg: Mapping[str, Any] = config["windowing"]
    window_sec = float(windowing_cfg["window_sec"])
    stride_sec = float(windowing_cfg.get("stride_sec", window_sec))
    overlap_threshold_sec = float(windowing_cfg["overlap_threshold_sec"])
    valid_events = frozenset(windowing_cfg["valid_events"])

    if window_sec <= 0.0:
        raise ValueError(f"window_sec must be > 0, got {window_sec}.")
    if not math.isclose(window_sec, _UCDDB_STAGE_INTERVAL_SEC):
        raise ValueError(
            f"1-to-1 <id>_stage.txt alignment requires window_sec == {_UCDDB_STAGE_INTERVAL_SEC}, "
            f"got {window_sec}."
        )
    if not math.isclose(stride_sec, window_sec):
        raise ValueError(
            f"Stage-aligned Windowing requires stride_sec ({stride_sec}) == window_sec ({window_sec})."
        )
    if overlap_threshold_sec <= 0.0:
        raise ValueError(
            f"overlap_threshold_sec must be > 0, got {overlap_threshold_sec}."
        )

    n_windows = min(
        int(recording_data.duration_sec // window_sec),
        int(recording_data.sleep_stages.size),
    )

    records: list[WindowRecord] = []
    for window_idx in range(n_windows):
        start_sec = float(window_idx * window_sec)
        end_sec = float((window_idx + 1) * window_sec)
        sleep_stage = int(recording_data.sleep_stages[window_idx])
        overlap_sec = compute_window_overlap(
            w_start=start_sec,
            w_end=end_sec,
            events=recording_data.events,
            valid_events=valid_events,
        )
        is_apnea = overlap_sec >= overlap_threshold_sec or math.isclose(
            overlap_sec, overlap_threshold_sec
        )
        records.append(
            WindowRecord(
                recording_id=recording_data.recording_id,
                window_idx=window_idx,
                start_sec=start_sec,
                end_sec=end_sec,
                sleep_stage=sleep_stage,
                is_wake=(sleep_stage == _WAKE_SLEEP_STAGE),
                overlap_sec=overlap_sec,
                label=1 if is_apnea else 0,
            )
        )

    return records


def slice_channel_window(
    signal: npt.NDArray[np.floating[Any]],
    fs: float,
    start_sec: float,
    end_sec: float,
) -> npt.NDArray[np.float32]:
    """Return the 1D waveform slice for ``[start_sec, end_sec]`` clamped to Recording bounds.

    Safely clamps sample indices to ``[0, len(signal)]`` so callers can request
    30-second Windows, centered 120-second HRV context intervals (which may start
    before ``0.0 s``), and lagged ``SpO2`` look-ahead intervals (which may extend past
    ``duration_sec``) without out-of-bounds or negative-index wraparound bugs.

    Args:
        signal: 1D continuous channel waveform.
        fs: Channel sampling rate in Hz.
        start_sec: Requested interval start time in seconds.
        end_sec: Requested interval end time in seconds.

    Returns:
        1D ``float32`` numpy slice (view when ``signal`` is already ``float32``)
        corresponding to ``[start_sec, end_sec]`` clamped to ``[0, duration_sec]``.

    Raises:
        ValueError: If ``signal`` is not 1D, ``fs <= 0``, or ``end_sec < start_sec``.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D channel signal, got shape {signal.shape}.")
    if fs <= 0.0:
        raise ValueError(f"Sampling rate fs must be > 0, got {fs}.")
    if end_sec < start_sec:
        raise ValueError(f"end_sec ({end_sec}) must be >= start_sec ({start_sec}).")

    n_samples = int(signal.shape[0])
    start_idx = min(max(int(round(start_sec * fs)), 0), n_samples)
    end_idx = min(max(int(round(end_sec * fs)), start_idx), n_samples)
    sliced = signal[start_idx:end_idx]
    return np.asarray(sliced, dtype=np.float32)


def compute_recording_ahi(
    recording_data: RecordingData,
    config: Mapping[str, Any],
) -> float:
    """Compute clinical Recording AHI (valid Apnea events per hour of definite sleep).

    Counts annotated events in ``recording_data.events`` whose ``event_type`` belongs
    to ``config['windowing']['valid_events']`` and divides by total hours spent in
    definite Sleep stages ``{1, 2, 3, 4, 5}`` (REM and NREM, excluding Wake ``0`` and
    Indeterminate ``8``), where each ``<id>_stage.txt`` entry spans 30 seconds.

    Args:
        recording_data: Loaded UCDDB Recording with ``events`` and ``sleep_stages``.
        config: Pipeline configuration dictionary loaded from ``config.yaml``.

    Returns:
        Clinical Apnea-Hypopnea Index (events per hour of sleep) as a ``float``.

    Raises:
        ValueError: If the Recording has zero definite sleep hours.
    """
    valid_events = frozenset(config["windowing"]["valid_events"])
    n_sleep_windows = int(
        np.isin(recording_data.sleep_stages, _DEFINITE_SLEEP_STAGES).sum()
    )
    sleep_hours = (n_sleep_windows * _UCDDB_STAGE_INTERVAL_SEC) / _SECONDS_PER_HOUR
    if sleep_hours <= 0.0:
        raise ValueError(
            f"Cannot compute AHI for Recording {recording_data.recording_id!r} with zero sleep hours."
        )

    n_valid_events = sum(
        1 for event in recording_data.events if event.event_type in valid_events
    )
    return float(n_valid_events / sleep_hours)



