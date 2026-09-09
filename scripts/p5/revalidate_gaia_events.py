#!/usr/bin/env python3
"""Revalidate GAIA run-table taxonomy and injection intervals without training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import re
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p5.common import (  # noqa: E402
    GAIA_SERVICES,
    SUPPORTED_FAULTS,
    opaque_case_id,
    percentile_summary,
    sha256_file,
    write_csv,
    write_json,
)
from util.GAIA.pre_GAIA import (  # noqa: E402
    _read_truncated_csv,
    parse_anomaly_event,
)


SCHEMA_VERSION = "p5_g0r2_event_revalidation_v1"
CPU_DURATION_RE = re.compile(r"lasts\s+(\d+(?:\.\d+)?)\s+seconds")

TAXONOMY_RULES = {
    "login failure": {
        "parsing_rule": "WARNING message contains login failure; integer 'wait for N seconds', default 11 seconds",
        "start_time_source": "millisecond message prefix",
        "end_time_source": "start + parsed/default duration",
        "duration_rule": "integer seconds, default 11",
        "explicit_injection": True,
        "official_semantics": "run directory contains anomaly-injection records; message says simulate/login failure",
    },
    "memory_anomalies": {
        "parsing_rule": "WARNING message contains [memory_anomalies]",
        "start_time_source": "embedded 'start at' timestamp",
        "end_time_source": "start + parsed duration",
        "duration_rule": "integer seconds in current main",
        "explicit_injection": True,
        "official_semantics": "official run example is an anomaly injection; message says trigger high-memory program",
    },
    "cpu_anomalies": {
        "parsing_rule": "WARNING message contains [cpu_anomalies]",
        "start_time_source": "embedded 'start at' timestamp",
        "end_time_source": "start + duration when parsed",
        "duration_rule": "current main accepts integer only; audit interpretation accepts decimal seconds",
        "explicit_injection": True,
        "official_semantics": "run directory is documented as system log and anomaly-injection records; message is explicit anomaly trigger",
    },
    "file moving program": {
        "parsing_rule": "WARNING contains file moving program",
        "start_time_source": "embedded 'start with' timestamp",
        "end_time_source": "start + parsed duration",
        "duration_rule": "integer seconds",
        "explicit_injection": True,
        "official_semantics": "run directory is documented as system log and anomaly-injection records; message says trigger program",
    },
    "access permission denied exception": {
        "parsing_rule": "WARNING contains access permission denied exception",
        "start_time_source": "millisecond message prefix",
        "end_time_source": "start + duration",
        "duration_rule": "3600 seconds for 'an hour', otherwise integer seconds/default 3600",
        "explicit_injection": True,
        "official_semantics": "run directory is documented as system log and anomaly-injection records; message names injected exception",
    },
    "normal": {
        "parsing_rule": "all INFO records",
        "start_time_source": "not assigned by current parser",
        "end_time_source": "not assigned",
        "duration_rule": "0",
        "explicit_injection": False,
        "official_semantics": "system-log record, not established as an anomaly injection",
    },
    "error_event": {
        "parsing_rule": "all ERROR records",
        "start_time_source": "second-resolution message prefix",
        "end_time_source": "equal to start",
        "duration_rule": "0",
        "explicit_injection": False,
        "official_semantics": "ERROR system-log record; no official evidence that every ERROR row is an injection",
    },
    "normal memory freed label": {
        "parsing_rule": "WARNING contains [normal memory freed label]",
        "start_time_source": "millisecond message prefix",
        "end_time_source": "synthetic start + 600 seconds in current main",
        "duration_rule": "hard-coded 600 seconds",
        "explicit_injection": False,
        "official_semantics": "message names a normal memory-freed label; recovery interpretation requires temporal evidence",
    },
    "unknown_warning": {
        "parsing_rule": "WARNING not matched by a named current-main rule",
        "start_time_source": "millisecond message prefix when present",
        "end_time_source": "equal to start",
        "duration_rule": "0",
        "explicit_injection": False,
        "official_semantics": "unsupported without an explicit supported-injection rule",
    },
    "unknown": {
        "parsing_rule": "message level not recognized",
        "start_time_source": "none",
        "end_time_source": "none",
        "duration_rule": "0",
        "explicit_injection": False,
        "official_semantics": "unsupported",
    },
}


def _fault_type(value: object) -> str:
    return str(value).strip().strip("[]")


def _timestamp_ms(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    timestamp = pd.Timestamp(str(value))
    if pd.isna(timestamp):
        return None
    return int(timestamp.timestamp() * 1000)


def _repair_cpu(parsed: Mapping[str, object]) -> Mapping[str, object]:
    result = dict(parsed)
    if _fault_type(result["anomaly_type"]) != "cpu_anomalies" or result.get("ed_time"):
        result["audit_cpu_repair_applied"] = False
        return result
    match = CPU_DURATION_RE.search(str(result.get("message", "")))
    if not match or not result.get("st_time"):
        result["audit_cpu_repair_applied"] = False
        return result
    duration = float(match.group(1))
    start = pd.Timestamp(str(result["st_time"]))
    result["duration"] = duration
    result["ed_time"] = str(start + pd.Timedelta(seconds=duration))
    result["audit_cpu_repair_applied"] = True
    return result


def _classify_exclusion(fault_type: str) -> str:
    return {
        "normal": "normal_record",
        "normal memory freed label": "recovery_marker_candidate",
        "error_event": "non_injection_error_record",
    }.get(fault_type, "unsupported_event_type")


def parse_registry(run_table: Path) -> Tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]:
    frame = _read_truncated_csv(str(run_table))
    raw_rows = []
    injections = []
    for source_index, row in frame.iterrows():
        current = dict(parse_anomaly_event(row))
        repaired = _repair_cpu(current)
        fault_type = _fault_type(current["anomaly_type"])
        current_start = _timestamp_ms(current.get("st_time"))
        current_end = _timestamp_ms(current.get("ed_time"))
        audit_start = _timestamp_ms(repaired.get("st_time"))
        audit_end = _timestamp_ms(repaired.get("ed_time"))
        supported_taxonomy = fault_type in SUPPORTED_FAULTS
        service_supported = str(current["service"]) in GAIA_SERVICES
        current_interval_valid = (
            current_start is not None and current_end is not None and current_end >= current_start
        )
        audit_interval_valid = (
            audit_start is not None and audit_end is not None and audit_end >= audit_start
        )
        eligible = supported_taxonomy and service_supported and audit_interval_valid
        exclusion = "" if eligible else (
            "missing_event_interval"
            if supported_taxonomy and service_supported and not audit_interval_valid
            else "service_not_in_candidates"
            if supported_taxonomy and not service_supported
            else _classify_exclusion(fault_type)
        )
        raw_record = {
            "source_index": int(source_index),
            "datetime": str(current["datetime"]),
            "service": str(current["service"]),
            "level": str(current["level"]),
            "fault_type": fault_type,
            "current_start_ms": current_start,
            "current_end_ms": current_end,
            "current_duration_seconds": float(current.get("duration", 0)),
            "current_interval_valid": current_interval_valid,
            "audit_start_ms": audit_start,
            "audit_end_ms": audit_end,
            "audit_duration_seconds": float(repaired.get("duration", 0)),
            "audit_cpu_repair_applied": bool(repaired["audit_cpu_repair_applied"]),
            "supported_injection": eligible,
            "exclusion_reason": exclusion,
            "message": str(current["message"]),
        }
        raw_rows.append(raw_record)
        if eligible:
            start_ms = int(audit_start)
            end_ms = int(audit_end)
            injections.append(
                {
                    "case_id": opaque_case_id(int(source_index), start_ms),
                    "source_index": int(source_index),
                    "service": str(current["service"]),
                    "fault_type": fault_type,
                    "level": str(current["level"]),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_seconds": (end_ms - start_ms) / 1000.0,
                    "current_main_interval_valid": current_interval_valid,
                    "audit_cpu_repair_applied": bool(repaired["audit_cpu_repair_applied"]),
                }
            )
    return tuple(raw_rows), tuple(injections)


def _overlap_analysis(injections: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    parent = list(range(len(injections)))
    neighbors = [set() for _ in injections]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    active = []
    same_root_pairs = 0
    different_root_pairs = 0
    for index in sorted(range(len(injections)), key=lambda i: (injections[i]["start_ms"], injections[i]["end_ms"], injections[i]["case_id"])):
        start_ms = int(injections[index]["start_ms"])
        active = [other for other in active if int(injections[other]["end_ms"]) > start_ms]
        for other in active:
            neighbors[index].add(other)
            neighbors[other].add(index)
            union(index, other)
            if injections[index]["service"] == injections[other]["service"]:
                same_root_pairs += 1
            else:
                different_root_pairs += 1
        active.append(index)

    categories = Counter()
    for index, adjacent in enumerate(neighbors):
        if not adjacent:
            categories["isolated"] += 1
        elif any(injections[index]["service"] != injections[other]["service"] for other in adjacent):
            categories["different_root_overlap"] += 1
        else:
            categories["same_root_only_overlap"] += 1
    components = Counter(find(index) for index in range(len(injections)))
    size_distribution = Counter(components.values())
    return {
        "interval_semantics": "half-open [start_ms, end_ms); boundary touching is not overlap",
        "case_categories": dict(sorted(categories.items())),
        "overlap_cases": sum(value for key, value in categories.items() if key != "isolated"),
        "overlap_pairs": same_root_pairs + different_root_pairs,
        "same_root_overlap_pairs": same_root_pairs,
        "different_root_overlap_pairs": different_root_pairs,
        "connected_components": len(components),
        "component_size_distribution": {str(key): value for key, value in sorted(size_distribution.items())},
        "maximum_component_size": max(components.values(), default=0),
        "maximum_overlap_degree": max((len(value) for value in neighbors), default=0),
    }


def _recovery_analysis(raw_rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    memories = [row for row in raw_rows if row["fault_type"] == "memory_anomalies" and row["audit_end_ms"] is not None]
    recoveries = [row for row in raw_rows if row["fault_type"] == "normal memory freed label" and row["audit_start_ms"] is not None]
    gaps = []
    matched = []
    for recovery in recoveries:
        prior = [
            memory
            for memory in memories
            if memory["service"] == recovery["service"] and int(memory["audit_start_ms"]) <= int(recovery["audit_start_ms"])
        ]
        if not prior:
            continue
        memory = max(prior, key=lambda value: int(value["audit_start_ms"]))
        gap = (int(recovery["audit_start_ms"]) - int(memory["audit_end_ms"])) / 1000.0
        gaps.append(gap)
        matched.append(
            {
                "recovery_source_index": recovery["source_index"],
                "memory_source_index": memory["source_index"],
                "service": recovery["service"],
                "gap_from_memory_end_seconds": gap,
            }
        )
    return {
        "recovery_marker_rows": len(recoveries),
        "same_service_preceding_memory_rows": len(matched),
        "gap_from_preceding_memory_end_seconds": percentile_summary(gaps),
        "within_1_second_of_memory_end": sum(abs(value) <= 1 for value in gaps),
        "within_30_seconds_after_memory_end": sum(0 <= value <= 30 for value in gaps),
        "examples": matched[:10],
        "interpretation_boundary": "temporal association is evidence for a recovery marker, not proof of official semantics",
    }


def _taxonomy(raw_rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    by_type = defaultdict(list)
    for row in raw_rows:
        by_type[str(row["fault_type"])].append(row)
    result = {}
    for fault_type in sorted(set(TAXONOMY_RULES) | set(by_type)):
        rows = by_type.get(fault_type, [])
        result[fault_type] = {
            **TAXONOMY_RULES.get(fault_type, TAXONOMY_RULES["unknown"]),
            "raw_count": len(rows),
            "service_distribution": dict(sorted(Counter(str(row["service"]) for row in rows).items())),
            "current_main_interval_valid": sum(bool(row["current_interval_valid"]) for row in rows),
            "audit_supported_injections": sum(bool(row["supported_injection"]) for row in rows),
            "duration_examples_seconds": sorted({float(row["audit_duration_seconds"]) for row in rows})[:12],
            "message_examples": [str(row["message"])[:240] for row in rows[:3]],
        }
    return result


def build_report(run_table: Path) -> Tuple[Mapping[str, object], Sequence[Mapping[str, object]]]:
    raw_rows, injections = parse_registry(run_table)
    root_counts = Counter(str(row["service"]) for row in injections)
    fault_counts = Counter(str(row["fault_type"]) for row in injections)
    joint_counts = Counter((str(row["service"]), str(row["fault_type"])) for row in injections)
    total = len(injections)
    report = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "REVALIDATED FACT",
        "source": {
            "run_table": str(run_table.resolve()),
            "sha256": sha256_file(run_table),
            "bytes": run_table.stat().st_size,
            "official_repository": "https://github.com/CloudWise-OpenSource/GAIA-DataSet",
            "official_semantics": "MicroSS run contains system logs and all anomaly-injection records",
        },
        "raw_record_count": len(raw_rows),
        "level_distribution": dict(sorted(Counter(str(row["level"]) for row in raw_rows).items())),
        "raw_type_distribution": dict(sorted(Counter(str(row["fault_type"]) for row in raw_rows).items())),
        "current_main_parseable_start_end": sum(bool(row["current_interval_valid"]) for row in raw_rows),
        "audit_interpretation_parseable_start_end": sum(row["audit_start_ms"] is not None and row["audit_end_ms"] is not None for row in raw_rows),
        "supported_injections": total,
        "excluded_records": len(raw_rows) - total,
        "exclusion_reason_distribution": dict(sorted(Counter(str(row["exclusion_reason"]) for row in raw_rows if row["exclusion_reason"]).items())),
        "taxonomy": _taxonomy(raw_rows),
        "root_distribution": {
            service: {"count": count, "percentage": 100.0 * count / total}
            for service, count in sorted(root_counts.items())
        },
        "fault_distribution": {
            fault: {"count": count, "percentage": 100.0 * count / total}
            for fault, count in sorted(fault_counts.items())
        },
        "root_fault_distribution": [
            {
                "service": service,
                "fault_type": fault,
                "count": count,
                "percentage": 100.0 * count / total,
            }
            for (service, fault), count in sorted(joint_counts.items())
        ],
        "raw_interval_overlap": _overlap_analysis(injections),
        "cpu_parser_audit": {
            "cpu_event_count": sum(row["fault_type"] == "cpu_anomalies" for row in raw_rows),
            "current_main_parse_success": sum(row["fault_type"] == "cpu_anomalies" and row["current_interval_valid"] for row in raw_rows),
            "current_main_parse_failure": sum(row["fault_type"] == "cpu_anomalies" and not row["current_interval_valid"] for row in raw_rows),
            "audit_float_repair_success": sum(row["fault_type"] == "cpu_anomalies" and row["audit_cpu_repair_applied"] for row in raw_rows),
            "duration_examples_seconds": sorted({float(row["audit_duration_seconds"]) for row in raw_rows if row["fault_type"] == "cpu_anomalies"}),
            "status": "REVALIDATED PARSER DEFECT" if any(row["fault_type"] == "cpu_anomalies" and row["audit_cpu_repair_applied"] for row in raw_rows) else "NOT REPRODUCED",
        },
        "recovery_marker_audit": _recovery_analysis(raw_rows),
        "error_record_audit": {
            "error_rows": sum(row["fault_type"] == "error_event" for row in raw_rows),
            "current_main_zero_duration_intervals": sum(row["fault_type"] == "error_event" and row["current_interval_valid"] for row in raw_rows),
            "would_enter_deal_label_filter": sum(row["fault_type"] == "error_event" and row["current_interval_valid"] and row["service"] in GAIA_SERVICES for row in raw_rows),
            "official_injection_status": "not established",
        },
        "frequency_shortcut": {
            "majority_root": root_counts.most_common(1)[0][0],
            "majority_root_count": root_counts.most_common(1)[0][1],
            "full_inventory_majority_root_accuracy": root_counts.most_common(1)[0][1] / total,
            "majority_fault": fault_counts.most_common(1)[0][0],
            "majority_fault_count": fault_counts.most_common(1)[0][1],
            "full_inventory_majority_fault_accuracy": fault_counts.most_common(1)[0][1] / total,
            "scope": "dataset-shortcut sanity check only; not used for configuration selection",
        },
        "limitations": [
            "The audit-only decimal CPU interpretation does not modify Ada-MGAD production preprocessing.",
            "Official GAIA documentation establishes the run directory role but does not define every message subtype.",
            "label.csv membership also depends on the metric-derived timestamp index; that exact index is audited separately.",
        ],
    }
    return report, injections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaia-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_table = args.gaia_root / "run" / "run" / "run" / "run_table_2021-07.csv"
    if not run_table.is_file():
        raise FileNotFoundError(run_table)
    report, injections = build_report(run_table)
    write_json(args.output_dir / "event_revalidation.json", report)
    write_csv(
        args.output_dir / "event_registry.csv",
        (
            "case_id",
            "source_index",
            "service",
            "fault_type",
            "level",
            "start_ms",
            "end_ms",
            "duration_seconds",
            "current_main_interval_valid",
            "audit_cpu_repair_applied",
        ),
        injections,
    )
    print(
        "records={}; supported={}; excluded={}; run_table_sha256={}".format(
            report["raw_record_count"],
            report["supported_injections"],
            report["excluded_records"],
            report["source"]["sha256"],
        )
    )


if __name__ == "__main__":
    main()
