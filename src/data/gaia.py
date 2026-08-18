"""GAIA MicroSS event adapter for the standalone RCA protocol."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from src.data.schema import (
    RCACaseInput,
    RCACaseLabel,
    TelemetryRef,
    validate_case_collection,
)
from util.GAIA.constant import GAIA_SERVICES
from util.GAIA.pre_GAIA import GAIA_TZ, _read_truncated_csv, parse_anomaly_event


_VALID_FAULT_TYPES = frozenset(
    {
        "access permission denied exception",
        "cpu_anomalies",
        "file moving program",
        "login failure",
        "memory_anomalies",
    }
)


@dataclass(frozen=True)
class GAIAEventAudit:
    case_id: str
    source_index: int
    start_ms: int
    end_ms: int
    duration_seconds: float
    overlapping_case_ids: Tuple[str, ...]


@dataclass(frozen=True)
class GAIAExcludedEvent:
    source_index: int
    observed_type: str
    reason: str


@dataclass(frozen=True)
class GAIASourceLayout:
    run_table: str
    metrics_directory: str
    logs_directory: str
    traces_directory: str


@dataclass(frozen=True)
class GAIAAdapterResult:
    inputs: Tuple[RCACaseInput, ...]
    labels: Tuple[RCACaseLabel, ...]
    audit: Tuple[GAIAEventAudit, ...]
    excluded: Tuple[GAIAExcludedEvent, ...]
    source_layout: GAIASourceLayout


def _opaque_case_id(source_index: int, start_ms: int) -> str:
    payload = "gaia-2021-07:{}:{}".format(source_index, start_ms).encode("utf-8")
    return "gaia-{}".format(hashlib.sha256(payload).hexdigest()[:16])


def _fault_type(parsed_type: str) -> str:
    return parsed_type.strip().strip("[]")


def _repair_float_cpu_duration(parsed: Dict[str, object]) -> None:
    if _fault_type(str(parsed["anomaly_type"])) != "cpu_anomalies":
        return
    if parsed.get("ed_time"):
        return
    match = re.search(r"lasts\s+(\d+(?:\.\d+)?)\s+seconds", str(parsed["message"]))
    if not match or not parsed.get("st_time"):
        return
    duration = float(match.group(1))
    start = pd.Timestamp(str(parsed["st_time"]))
    parsed["duration"] = duration
    parsed["ed_time"] = str(start + pd.Timedelta(seconds=duration))


def _source_layout(raw_root: Path) -> GAIASourceLayout:
    return GAIASourceLayout(
        run_table=str(raw_root / "run" / "run" / "run" / "run_table_2021-07.csv"),
        metrics_directory=str(raw_root / "metric" / "metric_split" / "metric"),
        logs_directory=str(raw_root / "business" / "business_split" / "business"),
        traces_directory=str(raw_root / "trace" / "trace_split" / "trace"),
    )


def _validate_source_layout(layout: GAIASourceLayout) -> None:
    if not Path(layout.run_table).is_file():
        raise FileNotFoundError(layout.run_table)
    for directory in (
        layout.metrics_directory,
        layout.logs_directory,
        layout.traces_directory,
    ):
        if not Path(directory).is_dir():
            raise FileNotFoundError(directory)


def _overlap_map(intervals: Sequence[Tuple[str, int, int]]) -> Mapping[str, Tuple[str, ...]]:
    overlaps = {case_id: set() for case_id, _, _ in intervals}
    active: List[Tuple[str, int]] = []
    for case_id, start_ms, end_ms in sorted(intervals, key=lambda item: (item[1], item[2])):
        active = [(other_id, other_end) for other_id, other_end in active if other_end > start_ms]
        for other_id, _ in active:
            overlaps[case_id].add(other_id)
            overlaps[other_id].add(case_id)
        active.append((case_id, end_ms))
    return {
        case_id: tuple(sorted(other_ids))
        for case_id, other_ids in overlaps.items()
    }


def load_gaia_cases(raw_path: str) -> GAIAAdapterResult:
    """Map supported GAIA injection records to label-separated RCA cases.

    All supported, time-bounded injection records are retained. Overlap is
    reported in the audit sidecar rather than silently filtered so the final
    context and contamination policy can be selected from data diagnostics.
    """

    raw_root = Path(raw_path).resolve()
    layout = _source_layout(raw_root)
    _validate_source_layout(layout)
    raw_events = _read_truncated_csv(layout.run_table)

    provisional = []
    excluded = []
    for source_index, row in raw_events.iterrows():
        parsed = dict(parse_anomaly_event(row))
        _repair_float_cpu_duration(parsed)
        fault_type = _fault_type(str(parsed["anomaly_type"]))

        if fault_type not in _VALID_FAULT_TYPES:
            reason = {
                "normal": "normal_record",
                "normal memory freed label": "recovery_marker",
                "error_event": "non_injection_error_record",
            }.get(fault_type, "unsupported_event_type")
            excluded.append(GAIAExcludedEvent(int(source_index), fault_type, reason))
            continue
        if not parsed.get("st_time") or not parsed.get("ed_time"):
            excluded.append(
                GAIAExcludedEvent(int(source_index), fault_type, "missing_event_interval")
            )
            continue

        root_service = str(parsed["service"])
        if root_service not in GAIA_SERVICES:
            excluded.append(
                GAIAExcludedEvent(int(source_index), fault_type, "root_not_in_candidates")
            )
            continue

        start = pd.Timestamp(str(parsed["st_time"]))
        end = pd.Timestamp(str(parsed["ed_time"]))
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        if end_ms < start_ms:
            excluded.append(
                GAIAExcludedEvent(int(source_index), fault_type, "negative_duration")
            )
            continue
        case_id = _opaque_case_id(int(source_index), start_ms)
        provisional.append(
            (case_id, int(source_index), start_ms, end_ms, root_service, fault_type)
        )

    metric_ref = TelemetryRef(
        uri="gaia://micross-2021-07/metrics",
        format="csv-shards",
        timestamp_column="timestamp",
        metadata={"timestamp_unit": "ms", "timezone": GAIA_TZ},
    )
    log_ref = TelemetryRef(
        uri="gaia://micross-2021-07/business-logs",
        format="csv-shards",
        timestamp_column="message",
        service_column="service",
        metadata={
            "timezone": GAIA_TZ,
            "timestamp_extraction": (
                "leading YYYY-MM-DD HH:MM:SS,mmm prefix from message"
            ),
            "declared_datetime_column": "date-only; not used for event slicing",
        },
    )
    trace_ref = TelemetryRef(
        uri="gaia://micross-2021-07/traces",
        format="csv-shards",
        timestamp_column="start_time",
        service_column="service_name",
        metadata={"timezone": GAIA_TZ},
    )

    inputs = []
    labels = []
    for case_id, _, start_ms, _, root_service, fault_type in provisional:
        inputs.append(
            RCACaseInput(
                case_id=case_id,
                dataset="GAIA-MicroSS-2021-07",
                anchor_time=start_ms,
                services=tuple(GAIA_SERVICES),
                metrics=metric_ref,
                logs=log_ref,
                traces=trace_ref,
                topology=None,
                metadata={
                    "timestamp_unit": "ms",
                    "timezone": GAIA_TZ,
                    "candidate_rule": "GAIA application service instances",
                },
            )
        )
        labels.append(RCACaseLabel(case_id, root_service, fault_type))

    validate_case_collection(inputs, labels)
    overlap_by_id = _overlap_map(
        [(case_id, start_ms, end_ms) for case_id, _, start_ms, end_ms, _, _ in provisional]
    )
    audit = tuple(
        GAIAEventAudit(
            case_id=case_id,
            source_index=source_index,
            start_ms=start_ms,
            end_ms=end_ms,
            duration_seconds=(end_ms - start_ms) / 1000.0,
            overlapping_case_ids=overlap_by_id[case_id],
        )
        for case_id, source_index, start_ms, end_ms, _, _ in provisional
    )
    return GAIAAdapterResult(
        inputs=tuple(inputs),
        labels=tuple(labels),
        audit=audit,
        excluded=tuple(excluded),
        source_layout=layout,
    )
