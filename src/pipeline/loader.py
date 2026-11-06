"""UCDDB Recording and pipeline configuration loader."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

# EDF-1992 fixed-width header specification constants (bytes)
_EDF_GENERAL_HEADER_BYTES = 256
_EDF_SIGNAL_HEADER_BYTES = 256
_EDF_STARTTIME_SLICE = slice(176, 184)
_EDF_HEADER_BYTES_SLICE = slice(184, 192)
_EDF_N_RECORDS_SLICE = slice(236, 244)
_EDF_RECORD_DUR_SLICE = slice(244, 252)
_EDF_N_SIGNALS_SLICE = slice(252, 256)
_EDF_SIG_LABEL_WIDTH = 16
_EDF_SIG_TRANSDUCER_WIDTH = 80
_EDF_SIG_DIM_WIDTH = 8
_EDF_SIG_EXTREMA_WIDTH = 8
_EDF_SIG_PREFILTER_WIDTH = 80
_EDF_SIG_SAMPLES_WIDTH = 8
_EDF_SIG_RESERVED_WIDTH = 32

# Clock & UCDDB text annotation format constants
_SECONDS_PER_DAY = 86400.0
_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_MINUTE = 60.0
_DOS_EOF_CHAR = "\x1a"
_EVENT_LINE_TIME_RE = re.compile(r"^(\d{1,2}:\d{2}:\d{2})\s+(.*)$")
_RESPEVT_PB_CS_QUALIFIERS = frozenset({"PB", "CS"})


@dataclass(frozen=True)
class ApneaEvent:
    """Parsed row from ``<id>_respevt.txt`` aligned to Recording start.

    Represents an annotated Apnea event (``APNEA-O/C/M`` or ``HYP-O/C/M``) or a
    non-apnea respiratory annotation (``PB`` or ``POSSIBLE``) retained for
    downstream filtering via ``config['windowing']['valid_events']``.

    Attributes:
        start_sec: Event onset in seconds relative to EDF Recording start.
        duration_sec: Event duration in seconds.
        end_sec: Event end time in seconds (``start_sec + duration_sec``).
        event_type: Normalized event type label.
    """

    start_sec: float
    duration_sec: float
    end_sec: float
    event_type: str


@dataclass
class RecordingData:
    """In-memory representation of a single UCDDB Recording.

    Attributes:
        recording_id: Recording identifier (e.g., ``'ucddb002'``).
        start_time_sec: EDF header start clock time in seconds from midnight.
        duration_sec: Total Recording duration in seconds.
        signals: Physical-unit ``float32`` 1D waveforms keyed by canonical channel name.
        sampling_rates: Sampling rate in Hz for each canonical channel.
        events: Parsed respiratory intervals aligned to ``start_time_sec``.
        sleep_stages: ``int8`` array of 30-second Sleep stage annotations.
    """

    recording_id: str
    start_time_sec: float
    duration_sec: float
    signals: dict[str, npt.NDArray[np.float32]]
    sampling_rates: dict[str, float]
    events: list[ApneaEvent]
    sleep_stages: npt.NDArray[np.int8]


@dataclass(frozen=True)
class _EdfContent:
    """Decoded EDF header metadata and canonical physical-unit waveforms."""

    start_time_sec: float
    duration_sec: float
    signals: dict[str, npt.NDArray[np.float32]]
    sampling_rates: dict[str, float]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML pipeline configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the YAML content does not parse into a dictionary.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fp:
        loaded = yaml.safe_load(fp)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at root of {path}, got {type(loaded).__name__}.")
    return loaded


def discover_recording_ids(raw_dir: str | Path) -> list[str]:
    """Discover sorted UCDDB Recording IDs from ``.rec`` files in ``raw_dir``.

    Selects ``<id>.rec`` polysomnography files and ignores ``_lifecard.edf`` Holter files.

    Args:
        raw_dir: Directory containing raw UCDDB ``<id>.rec`` files.

    Returns:
        Sorted list of Recording identifiers (e.g., ``['ucddb002', ..., 'ucddb028']``).

    Raises:
        FileNotFoundError: If ``raw_dir`` does not exist.
    """
    directory = Path(raw_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {directory}")
    return sorted(rec_path.stem for rec_path in directory.glob("*.rec") if rec_path.is_file())


def _parse_clock_seconds(clock_str: str, separator: str) -> float:
    """Convert an ``HH<sep>MM<sep>SS`` clock string into seconds from midnight."""
    parts = clock_str.strip().split(separator)
    if len(parts) != 3:
        raise ValueError(f"Invalid clock timestamp {clock_str!r}; expected separator {separator!r}.")
    hours, minutes, seconds = (int(part) for part in parts)
    return (
        hours * _SECONDS_PER_HOUR
        + minutes * _SECONDS_PER_MINUTE
        + float(seconds)
    )


def _read_edf_canonical_signals(
    rec_path: Path,
    canonical_channels: list[str],
    aliases: Mapping[str, str],
    expected_sampling_rates: Mapping[str, float],
) -> _EdfContent:
    """Read an EDF ``.rec`` file and return normalized physical-unit canonical signals."""
    canonical_set = set(canonical_channels)

    with rec_path.open("rb") as fp:
        general_hdr = fp.read(_EDF_GENERAL_HEADER_BYTES)
        if len(general_hdr) < _EDF_GENERAL_HEADER_BYTES:
            raise ValueError(f"Truncated EDF header in {rec_path}.")

        starttime_str = general_hdr[_EDF_STARTTIME_SLICE].decode("ascii").strip()
        header_bytes = int(general_hdr[_EDF_HEADER_BYTES_SLICE].decode("ascii").strip())
        n_records = int(general_hdr[_EDF_N_RECORDS_SLICE].decode("ascii").strip())
        record_duration_sec = float(general_hdr[_EDF_RECORD_DUR_SLICE].decode("ascii").strip())
        n_signals = int(general_hdr[_EDF_N_SIGNALS_SLICE].decode("ascii").strip())

        start_time_sec = _parse_clock_seconds(starttime_str, ".")
        duration_sec = n_records * record_duration_sec

        sig_hdr_bytes = fp.read(n_signals * _EDF_SIGNAL_HEADER_BYTES)
        offset = 0

        def _read_sig_field(width: int) -> list[str]:
            nonlocal offset
            values = [
                sig_hdr_bytes[offset + idx * width : offset + (idx + 1) * width]
                .decode("ascii", errors="replace")
                .strip()
                for idx in range(n_signals)
            ]
            offset += n_signals * width
            return values

        raw_labels = _read_sig_field(_EDF_SIG_LABEL_WIDTH)
        _read_sig_field(_EDF_SIG_TRANSDUCER_WIDTH)
        _read_sig_field(_EDF_SIG_DIM_WIDTH)
        phys_mins = [float(val) for val in _read_sig_field(_EDF_SIG_EXTREMA_WIDTH)]
        phys_maxs = [float(val) for val in _read_sig_field(_EDF_SIG_EXTREMA_WIDTH)]
        dig_mins = [int(val) for val in _read_sig_field(_EDF_SIG_EXTREMA_WIDTH)]
        dig_maxs = [int(val) for val in _read_sig_field(_EDF_SIG_EXTREMA_WIDTH)]
        _read_sig_field(_EDF_SIG_PREFILTER_WIDTH)
        samples_per_record = [int(val) for val in _read_sig_field(_EDF_SIG_SAMPLES_WIDTH)]
        _read_sig_field(_EDF_SIG_RESERVED_WIDTH)

        total_samples_per_record = sum(samples_per_record)
        fp.seek(header_bytes)
        raw_records = np.fromfile(
            fp,
            dtype="<i2",
            count=n_records * total_samples_per_record,
        )

    expected_total = n_records * total_samples_per_record
    if raw_records.size != expected_total:
        raise ValueError(
            f"Incomplete EDF data in {rec_path}: expected {expected_total} samples, "
            f"got {raw_records.size}."
        )
    records_2d = raw_records.reshape(n_records, total_samples_per_record)

    signals: dict[str, npt.NDArray[np.float32]] = {}
    sampling_rates: dict[str, float] = {}
    col_pos = 0

    for idx in range(n_signals):
        n_samp = samples_per_record[idx]
        sig_slice = records_2d[:, col_pos : col_pos + n_samp]
        col_pos += n_samp

        canonical_name = aliases.get(raw_labels[idx], raw_labels[idx])
        if canonical_name not in canonical_set:
            continue
        if canonical_name in signals:
            raise ValueError(f"Duplicate canonical channel {canonical_name!r} in {rec_path}.")

        fs = n_samp / record_duration_sec
        expected_fs = expected_sampling_rates.get(canonical_name)
        if expected_fs is not None and not np.isclose(fs, expected_fs):
            raise ValueError(
                f"Sampling rate mismatch for {canonical_name!r} in {rec_path}: "
                f"expected {expected_fs} Hz, got {fs} Hz."
            )

        dig_span = dig_maxs[idx] - dig_mins[idx]
        if dig_span == 0:
            raise ValueError(f"Zero digital range for channel {canonical_name!r} in {rec_path}.")
        gain = (phys_maxs[idx] - phys_mins[idx]) / dig_span
        intercept = phys_mins[idx] - dig_mins[idx] * gain
        physical = (sig_slice.reshape(-1).astype(np.float64) * gain + intercept).astype(np.float32)

        signals[canonical_name] = physical
        sampling_rates[canonical_name] = float(fs)

    missing = canonical_set - set(signals.keys())
    if missing:
        raise ValueError(f"Missing canonical channels in {rec_path}: {sorted(missing)}")

    return _EdfContent(
        start_time_sec=start_time_sec,
        duration_sec=duration_sec,
        signals={ch: signals[ch] for ch in canonical_channels},
        sampling_rates={ch: sampling_rates[ch] for ch in canonical_channels},
    )


def _parse_sleep_stages(stage_path: Path) -> npt.NDArray[np.int8]:
    """Parse ``<id>_stage.txt`` into an ``int8`` array of 30-second Sleep stages."""
    with stage_path.open("r", encoding="latin-1") as fp:
        stage_values = [
            int(cleaned)
            for line in fp
            if (cleaned := line.replace(_DOS_EOF_CHAR, "").strip())
        ]
    return np.asarray(stage_values, dtype=np.int8)


def _parse_apnea_events(
    respevt_path: Path,
    edf_start_clock_sec: float,
    event_aliases: Mapping[str, str],
) -> list[ApneaEvent]:
    """Parse ``<id>_respevt.txt`` with midnight-rollover clock arithmetic."""
    events: list[ApneaEvent] = []
    in_event_table = False

    with respevt_path.open("r", encoding="latin-1") as fp:
        for line_num, raw_line in enumerate(fp, start=1):
            cleaned = raw_line.replace(_DOS_EOF_CHAR, "").strip()
            if not cleaned:
                continue
            if not in_event_table:
                if cleaned.startswith("Time") and "Duration" in cleaned:
                    in_event_table = True
                continue

            match = _EVENT_LINE_TIME_RE.match(cleaned)
            if match is None:
                raise ValueError(
                    f"Malformed event timestamp row at {respevt_path}:{line_num}: {cleaned!r}"
                )

            clock_str, remainder = match.groups()
            tokens = remainder.split()
            if not tokens:
                raise ValueError(f"Missing event columns at {respevt_path}:{line_num}: {cleaned!r}")

            # Handle two-word 'PB EVENT' type separately from optional PB/CS column qualifier
            if len(tokens) >= 2 and f"{tokens[0]} {tokens[1]}" in event_aliases:
                raw_type = f"{tokens[0]} {tokens[1]}"
                after_type = tokens[2:]
            else:
                raw_type = tokens[0]
                after_type = tokens[1:]

            if after_type and after_type[0] in _RESPEVT_PB_CS_QUALIFIERS:
                after_type = after_type[1:]

            if not after_type:
                raise ValueError(f"Missing event duration at {respevt_path}:{line_num}: {cleaned!r}")

            duration_sec = float(after_type[0])
            event_type = event_aliases.get(raw_type, raw_type)

            event_clock_sec = _parse_clock_seconds(clock_str, ":")
            start_sec = float((event_clock_sec - edf_start_clock_sec) % _SECONDS_PER_DAY)
            events.append(
                ApneaEvent(
                    start_sec=start_sec,
                    duration_sec=duration_sec,
                    end_sec=start_sec + duration_sec,
                    event_type=event_type,
                )
            )

    if not in_event_table:
        raise ValueError(f"Missing event table header in {respevt_path}.")
    return events


def load_recording(recording_id: str, config: Mapping[str, Any]) -> RecordingData:
    """Load a single UCDDB Recording from disk using ``config`` settings.

    Reads ``<id>.rec`` physical waveforms for the 14 canonical channels (mapping
    configured aliases such as ``'Soud'`` -> ``'Sound'`` and dropping non-canonical
    channels), parses ``<id>_respevt.txt`` relative to the EDF start clock across
    midnight, and loads ``<id>_stage.txt`` Sleep stages.

    Args:
        recording_id: UCDDB Recording identifier (e.g., ``'ucddb002'``).
        config: Pipeline configuration dictionary loaded from ``config.yaml``.

    Returns:
        Populated ``RecordingData`` instance for ``recording_id``.
    """
    raw_dir = Path(config["paths"]["raw_data_dir"])
    channels_cfg = config["channels"]
    canonical_channels = list(channels_cfg["canonical"])
    aliases: Mapping[str, str] = channels_cfg.get("aliases", {})
    expected_rates: Mapping[str, float] = channels_cfg["sampling_rates"]
    event_aliases: Mapping[str, str] = config.get("windowing", {}).get("event_aliases", {})

    rec_path = raw_dir / f"{recording_id}.rec"
    respevt_path = raw_dir / f"{recording_id}_respevt.txt"
    stage_path = raw_dir / f"{recording_id}_stage.txt"

    edf = _read_edf_canonical_signals(
        rec_path=rec_path,
        canonical_channels=canonical_channels,
        aliases=aliases,
        expected_sampling_rates=expected_rates,
    )
    events = _parse_apnea_events(respevt_path, edf.start_time_sec, event_aliases)
    sleep_stages = _parse_sleep_stages(stage_path)

    return RecordingData(
        recording_id=recording_id,
        start_time_sec=edf.start_time_sec,
        duration_sec=edf.duration_sec,
        signals=edf.signals,
        sampling_rates=edf.sampling_rates,
        events=events,
        sleep_stages=sleep_stages,
    )
