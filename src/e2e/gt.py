"""Raw GAIA run-table provenance and protocol GT registry generation.

The run table is the only source of event labels.  All semantic fields are
derived from the raw ``message`` text; optional pre-parsed columns are never
trusted.  The registry keeps one row per raw injection and records the
detector-domain and split decisions as metadata rather than changing the GT
population.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


GAIA_TZ = timezone(timedelta(hours=8))
UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
TAXONOMY_VERSION = "gaia-v3-20260913"

FAULT_TYPES = (
    "login_failure",
    "memory_anomalies",
    "file_moving",
    "normal_memory_freed",
    "access_permission_denied",
    "cpu_anomalies",
)

_MESSAGE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),(\d{3})$")
_EMBEDDED_START = re.compile(
    r"start\s+(?:at|with)\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)",
    re.IGNORECASE,
)
_NUM_SECONDS = re.compile(
    r"(?:lasts|last for|wait for)\s+(\d+(?:\.\d+)?)\s+seconds",
    re.IGNORECASE,
)
_WORD_SECONDS = re.compile(
    r"(?:lasts|last for|wait for)\s+([a-z]+)\s+seconds", re.IGNORECASE
)
_WORD_MINUTES = re.compile(r"lasts\s+([a-z]+)\s+minutes", re.IGNORECASE)
_NUM_MINUTES = re.compile(
    r"lasts\s+(\d+(?:\.\d+)?)\s+minutes", re.IGNORECASE
)
_HOUR = re.compile(r"lasts?\s+an?\s+hour", re.IGNORECASE)
_LEVEL = re.compile(r"\|\s*(WARNING|ERROR|INFO|DEBUG)\s*\|", re.IGNORECASE)

_WORD_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _epoch_ms(value: datetime) -> int:
    aware = value.replace(tzinfo=GAIA_TZ)
    return int(round((aware.astimezone(timezone.utc) - UTC_EPOCH).total_seconds() * 1000.0))


def _parse_message_timestamp(value: str) -> Optional[int]:
    match = _MESSAGE_TS.match(str(value).strip())
    if not match:
        return None
    try:
        base = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return _epoch_ms(base) + int(match.group(2))


def _parse_embedded_timestamp(message: str) -> Optional[int]:
    match = _EMBEDDED_START.search(message)
    if not match:
        return None
    raw = match.group(1).replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return _epoch_ms(datetime.strptime(raw, fmt))
        except ValueError:
            continue
    return None


def _parse_duration(message: str) -> Tuple[Optional[float], Optional[str]]:
    match = _NUM_SECONDS.search(message)
    if match:
        return float(match.group(1)), "numeric_seconds"
    match = _WORD_SECONDS.search(message)
    if match and match.group(1).lower() in _WORD_NUM:
        return float(_WORD_NUM[match.group(1).lower()]), "word_seconds"
    match = _WORD_MINUTES.search(message)
    if match and match.group(1).lower() in _WORD_NUM:
        return float(_WORD_NUM[match.group(1).lower()]) * 60.0, "word_minutes"
    match = _NUM_MINUTES.search(message)
    if match:
        return float(match.group(1)) * 60.0, "numeric_minutes"
    if _HOUR.search(message):
        return 3600.0, "word_hour"
    return None, None


def _classify(payload: str, level: str) -> str:
    if "[memory_anomalies]" in payload:
        return "memory_anomalies"
    if "[cpu_anomalies]" in payload:
        return "cpu_anomalies"
    if "[normal memory freed label]" in payload:
        return "normal_memory_freed"
    lowered = payload.lower()
    if "login failure" in lowered:
        return "login_failure"
    if "file moving program" in lowered:
        return "file_moving"
    if "access permission denied exception" in lowered:
        return "access_permission_denied"
    if lowered.startswith("(background on this error at"):
        return "traceback_continuation"
    if level == "ERROR":
        return "error_record"
    if "upload" in lowered and "failed" in lowered:
        return "error_record"
    if "exception injection failed" in lowered or "object is not callable" in lowered:
        return "unsupported"
    return "normal_record"


def derive_raw_record(row: Mapping[str, object], source_index: int) -> Mapping[str, object]:
    """Derive one raw row without consulting non-raw/pre-parsed columns."""

    message = str(row.get("message", "")).strip()
    service = str(row.get("service", "")).strip()
    parts = message.split(" | ", 5)
    if len(parts) >= 6:
        prefix, level_part, node_ip, container_or_service, payload = (
            parts[0], parts[1].strip().upper(), parts[2].strip(), parts[3].strip(), parts[5]
        )
        # Some GAIA releases have five rather than six pipe-separated fields.
        # ``parts[4]`` is the service field in the six-field release and is
        # retained for provenance when it is available.
        message_service = parts[4].strip()
    else:
        prefix = parts[0] if parts else ""
        level_match = _LEVEL.search(message)
        level_part = level_match.group(1).upper() if level_match else ""
        node_ip = ""
        container_or_service = ""
        message_service = service
        payload = parts[-1] if parts else message
    if len(parts) < 6:
        # The current raw download uses five pipe-separated payload fields in
        # some business paths; split from the first level marker if possible.
        level_match = _LEVEL.search(message)
        if level_match:
            payload = message[level_match.end():].strip()

    message_ts_ms = _parse_message_timestamp(prefix)
    raw_type = _classify(payload, level_part)
    duration, duration_how = _parse_duration(payload)
    start_semantics = None
    start_ms = None
    if raw_type in ("memory_anomalies", "cpu_anomalies", "file_moving"):
        start_semantics = "embedded"
        start_ms = _parse_embedded_timestamp(payload)
    elif raw_type == "login_failure":
        start_semantics = "message_timestamp"
        start_ms = message_ts_ms
        if duration is None:
            duration, duration_how = 11.0, "login_default"
    elif raw_type == "access_permission_denied":
        start_semantics = "message_timestamp"
        start_ms = message_ts_ms
        if duration is None:
            duration, duration_how = 3600.0, "access_default"
    elif raw_type == "normal_memory_freed":
        start_semantics = "message_timestamp"
        start_ms = message_ts_ms
        if duration is None:
            duration, duration_how = 600.0, "normal_memory_freed_default"

    end_ms = None
    if start_ms is not None and duration is not None:
        end_ms = int(start_ms + round(float(duration) * 1000.0))
    parse_ok = start_ms is not None and end_ms is not None and end_ms > start_ms
    gt_candidate = raw_type in FAULT_TYPES
    gt_included = bool(gt_candidate and parse_ok and service)
    reason = "not a supported fault injection"
    if gt_candidate and not parse_ok:
        reason = "supported fault type but start/duration parsing is incomplete"
    elif gt_included:
        reason = "explicit fault-injection semantics derived from raw message"
    if raw_type == "normal_memory_freed":
        reason = "included as the protocol's sixth GT class"

    return {
        "source_index": int(source_index),
        "datetime": str(row.get("datetime", "")),
        "service": service,
        "message": message,
        "level": level_part,
        "message_service": message_service,
        "node_ip": node_ip,
        "raw_type": raw_type,
        "fault_type": raw_type if raw_type in FAULT_TYPES else "",
        "start_semantics": start_semantics or "none",
        "message_timestamp_ms": message_ts_ms,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_seconds": float(duration) if duration is not None else None,
        "duration_how": duration_how or "none",
        "gt_candidate": bool(gt_candidate),
        "gt_included": bool(gt_included),
        "decision_reason": reason,
    }


def iter_raw_records(run_table: Path) -> Iterable[Mapping[str, object]]:
    with Path(run_table).open(newline="", encoding="utf-8-sig") as handle:
        for source_index, row in enumerate(csv.DictReader(handle)):
            yield derive_raw_record(row, source_index)


def parse_raw_records(run_table: Path) -> List[Mapping[str, object]]:
    return list(iter_raw_records(run_table))


def _case_id(source_index: int) -> str:
    return "gaia-v3-{0:05d}".format(int(source_index))


def build_registry(
    records: Sequence[Mapping[str, object]],
    *,
    detector_start_ms: int,
    detector_end_ms: int,
    split_ms: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build raw, assigned, and purged GT tables.

    The first return value contains exactly one row for every parsed six-class
    injection, including rows outside the metric detector domain and rows that
    cross the split.  ``assigned`` is the only table consumed by Train/Test
    evaluation.  ``purged`` preserves the reason for every excluded row.
    """

    raw_rows: List[Dict[str, object]] = []
    assigned_rows: List[Dict[str, object]] = []
    purged_rows: List[Dict[str, object]] = []
    for record in records:
        if not bool(record["gt_included"]):
            continue
        start_ms = int(record["start_ms"])
        end_ms = int(record["end_ms"])
        in_domain = int(detector_start_ms) <= start_ms < int(detector_end_ms)
        split = ""
        purge_reason = ""
        if not in_domain:
            split = "outside_detector_domain"
            purge_reason = "outside_metric_detector_timeline"
        elif start_ms < int(split_ms) and end_ms <= int(split_ms):
            split = "train"
        elif start_ms >= int(split_ms) and end_ms <= int(detector_end_ms):
            split = "test"
        else:
            split = "boundary_purge"
            purge_reason = "raw_injection_crosses_70_30_split_boundary"
        row = {
            "case_id": _case_id(int(record["source_index"])),
            "source_index": int(record["source_index"]),
            "fault_type": str(record["fault_type"]),
            "labelled_service": str(record["service"]),
            "service": str(record["service"]),
            "level": str(record["level"]),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_seconds": float(record["duration_seconds"]),
            "taxonomy_version": TAXONOMY_VERSION,
            "detector_domain": bool(in_domain),
            "split": split,
            "purge_reason": purge_reason,
        }
        raw_rows.append(row)
        if split in ("train", "test"):
            assigned_rows.append(dict(row))
        else:
            purged_rows.append(dict(row))
    columns = [
        "case_id", "source_index", "fault_type", "labelled_service", "service", "level",
        "start_ms", "end_ms", "duration_seconds", "taxonomy_version", "detector_domain",
        "split", "purge_reason",
    ]
    sort_key = ["start_ms", "source_index", "case_id"]
    return tuple_table(raw_rows, columns, sort_key), tuple_table(assigned_rows, columns, sort_key), tuple_table(purged_rows, columns, sort_key)


def tuple_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str], sort_key: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=list(columns))
    if frame.empty:
        return frame
    return frame.sort_values(list(sort_key), kind="stable").reset_index(drop=True)


def build_provenance(
    run_table: Path,
    records: Sequence[Mapping[str, object]],
    raw_registry: pd.DataFrame,
    assigned: pd.DataFrame,
    purged: pd.DataFrame,
    *,
    detector_start_ms: int,
    detector_end_ms: int,
    split_ms: int,
    git_commit: str,
) -> Mapping[str, object]:
    six = [record for record in records if bool(record["gt_included"])]
    raw_type_counts = Counter(str(record["raw_type"]) for record in records)
    level_counts = Counter(str(record["level"]) for record in records)
    return {
        "schema_version": "p5_v3_gaia_gt_provenance_v1",
        "evidence_status": "REVALIDATED FACT",
        "taxonomy_version": TAXONOMY_VERSION,
        "generated_by_commit": str(git_commit),
        "raw_run_table": {
            "path": str(Path(run_table).resolve()),
            "sha256": sha256_file(run_table),
            "rows": int(len(records)),
            "columns_used": ["datetime", "service", "message"],
            "preparsed_columns_ignored": True,
        },
        "detector_timeline": {
            "start_ms": int(detector_start_ms),
            "end_ms": int(detector_end_ms),
            "interval": "[start_ms,end_ms)",
            "grid_seconds": 30,
        },
        "split": {
            "mode": "chronological_metric_70_30",
            "boundary_ms": int(split_ms),
            "intervals": {
                "train": "[detector_start_ms,boundary_ms)",
                "test": "[boundary_ms,detector_end_ms)",
            },
            "integer_rule": "K=(7*N)//10",
        },
        "taxonomy": {
            "fault_types": list(FAULT_TYPES),
            "raw_type_distribution": dict(sorted(raw_type_counts.items())),
            "level_distribution": dict(sorted(level_counts.items())),
            "raw_gt_count": int(len(raw_registry)),
            "detector_domain_gt_count": int(raw_registry["detector_domain"].sum()) if len(raw_registry) else 0,
            "train_gt_count": int((raw_registry["split"] == "train").sum()) if len(raw_registry) else 0,
            "test_gt_count": int((raw_registry["split"] == "test").sum()) if len(raw_registry) else 0,
            "split_boundary_purge_count": int(len(purged.loc[purged["split"] == "boundary_purge"])) if len(purged) else 0,
            "outside_detector_domain_count": int(len(purged.loc[purged["split"] == "outside_detector_domain"])) if len(purged) else 0,
            "fault_counts": dict(sorted(Counter(str(row["fault_type"]) for row in six).items())),
        },
        "semantics": {
            "gt_unit": "one raw injection equals one GT event",
            "interval": "half-open [start_ms,end_ms)",
            "normal_memory_freed": "message timestamp plus 600 seconds",
            "cpu_duration": "floating-point seconds preserved before millisecond end calculation",
            "error_traceback_exclusion": "run-table ERROR and continuation rows are not GT; business telemetry ERROR remains input",
        },
    }


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(path: Path, value: Mapping[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
