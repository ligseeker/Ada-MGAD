#!/usr/bin/env python3
"""Audit run-table -> Ada-MGAD labels -> historical RCA label semantics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p5.common import (  # noqa: E402
    GAIA_SERVICES,
    SUPPORTED_FAULTS,
    percentile_summary,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.p5.revalidate_gaia_events import parse_registry  # noqa: E402


SCHEMA_VERSION = "p5_g0r2_label_reconciliation_v1"
INTERVAL_MS = 30_000


def _ad_bins(start_ms: int, end_ms: int) -> tuple[int, ...]:
    """Replicate current deal_label floor alignment and inclusive end-bin mask."""

    start = (int(start_ms) // INTERVAL_MS) * INTERVAL_MS
    end = (int(end_ms) // INTERVAL_MS) * INTERVAL_MS
    return tuple(range(start, end + INTERVAL_MS, INTERVAL_MS))


def _row_semantics(fault_type: str) -> Mapping[str, str]:
    values = {
        "login failure": (
            "explicit simulated login failure",
            "positive on parsed interval at event service",
            "yes",
            "labelled injected service",
            "raw message + current parser + historical adapter",
        ),
        "memory_anomalies": (
            "explicit high-memory injection",
            "positive on parsed interval at event service",
            "yes",
            "labelled injected service",
            "official run example + current parser + historical adapter",
        ),
        "cpu_anomalies": (
            "explicit CPU anomaly injection with decimal duration",
            "not labelled because current-main duration regex rejects decimal",
            "yes after audit-only duration repair",
            "labelled injected service; parser defect must be resolved before final taxonomy",
            "raw rows + current parser + historical adapter repair",
        ),
        "file moving program": (
            "explicit file-moving program trigger",
            "positive on parsed interval at event service",
            "yes",
            "labelled fault service",
            "raw message + current parser + historical adapter",
        ),
        "access permission denied exception": (
            "explicit permission-denied exception",
            "positive on parsed interval at event service",
            "yes",
            "labelled fault service",
            "raw message + current parser + historical adapter",
        ),
        "normal": (
            "INFO/system-log record",
            "not positive",
            "no",
            "normal/system record; not a diagnosis event",
            "current parser; official run directory contains both logs and injections",
        ),
        "error_event": (
            "generic ERROR system-log record; injection status not established",
            "positive for one aligned bin because st=end and type is non-normal",
            "no",
            "semantically unresolved non-injection error observation",
            "current parser + deal_label filter; no subtype-specific official support",
        ),
        "normal memory freed label": (
            "normal memory-freed marker; exact role not officially established",
            "positive for synthetic 600s interval at event service",
            "no (historically excluded as recovery marker)",
            "OPEN QUESTION: recovery evidence, not an independent anomaly unless frozen later",
            "raw name + current hard-coded duration + targeted temporal audit",
        ),
        "unknown": (
            "unparsed/unsupported record",
            "not positive because no interval",
            "no",
            "unsupported/open",
            "current parser",
        ),
    }
    raw_semantics, ad_label, rca, meaning, evidence = values[fault_type]
    return {
        "raw_semantics": raw_semantics,
        "current_ad_label": ad_label,
        "historical_rca_eligible": rca,
        "candidate_e2e_meaning": meaning,
        "evidence": evidence,
    }


def build(run_table: Path):
    raw_rows, injections = parse_registry(run_table)
    injections_by_source = {int(row["source_index"]): row for row in injections}

    current_ad_events = []
    for row in raw_rows:
        if (
            row["fault_type"] != "normal"
            and row["current_interval_valid"]
            and row["service"] in GAIA_SERVICES
        ):
            bins = _ad_bins(int(row["current_start_ms"]), int(row["current_end_ms"]))
            current_ad_events.append({**row, "bins": bins})

    ad_bin_events = defaultdict(list)
    for row in current_ad_events:
        for timestamp in row["bins"]:
            ad_bin_events[timestamp].append(row)

    injection_bin_events = defaultdict(list)
    injection_bin_counts = []
    for row in injections:
        bins = _ad_bins(int(row["start_ms"]), int(row["end_ms"]))
        injection_bin_counts.append(len(bins))
        for timestamp in bins:
            injection_bin_events[timestamp].append(row)

    rca_total = len(injections)
    current_mappable = [row for row in injections if row["current_main_interval_valid"]]
    # The historical adapter assigns parsed["service"] directly to RCACaseLabel.root_service.
    exact_service = sum(
        raw_rows[int(row["source_index"])]["service"] == row["service"]
        for row in current_mappable
    )
    positive_bin_service_counts = Counter()
    positive_bin_injection_counts = Counter()
    same_root_multi_bins = 0
    different_root_multi_bins = 0
    for timestamp, rows in injection_bin_events.items():
        distinct_services = {str(row["service"]) for row in rows}
        positive_bin_service_counts[len(distinct_services)] += 1
        positive_bin_injection_counts[len(rows)] += 1
        if len(rows) > 1:
            if len(distinct_services) > 1:
                different_root_multi_bins += 1
            else:
                same_root_multi_bins += 1

    interval_groups = Counter(
        (row["service"], _ad_bins(int(row["start_ms"]), int(row["end_ms"])))
        for row in injections
    )
    shared_groups = [count for count in interval_groups.values() if count > 1]

    type_counts = Counter(str(row["fault_type"]) for row in raw_rows)
    reconciliation_rows = []
    for fault_type in (
        "login failure",
        "memory_anomalies",
        "cpu_anomalies",
        "file moving program",
        "access permission denied exception",
        "normal",
        "error_event",
        "normal memory freed label",
        "unknown",
    ):
        current_rows = [row for row in raw_rows if row["fault_type"] == fault_type]
        reconciliation_rows.append(
            {
                "raw_type": fault_type,
                "raw_count": type_counts[fault_type],
                **_row_semantics(fault_type),
                "current_main_interval_valid_count": sum(row["current_interval_valid"] for row in current_rows),
                "current_ad_mapped_count": sum(
                    row["fault_type"] != "normal"
                    and row["current_interval_valid"]
                    and row["service"] in GAIA_SERVICES
                    for row in current_rows
                ),
                "p5_status": (
                    "REVALIDATED FACT"
                    if fault_type not in {"normal memory freed label", "error_event", "unknown"}
                    else "OPEN QUESTION"
                    if fault_type == "normal memory freed label"
                    else "REVALIDATED FACT (mapping); semantics unresolved"
                ),
            }
        )

    examples = []
    for fault_type in SUPPORTED_FAULTS:
        candidates = [row for row in injections if row["fault_type"] == fault_type]
        if not candidates:
            continue
        row = min(candidates, key=lambda value: int(value["source_index"]))
        bins = _ad_bins(int(row["start_ms"]), int(row["end_ms"]))
        examples.append(
            {
                "source_index": row["source_index"],
                "fault_type": fault_type,
                "event_service": row["service"],
                "label_matrix_column": row["service"],
                "aligned_first_bin_ms": bins[0],
                "aligned_last_bin_ms": bins[-1],
                "aligned_bin_count": len(bins),
                "sliding_window_groundtruth": "label row at global_end - 1; same service one-hot state",
                "current_main_mappable": row["current_main_interval_valid"],
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "REVALIDATED FACT",
        "source": {
            "run_table": str(run_table.resolve()),
            "run_table_sha256": sha256_file(run_table),
            "grid_seconds": 30,
        },
        "code_trace": [
            "run_table row -> util/GAIA/pre_GAIA.py::parse_anomaly_event",
            "deal_label uses event.instance, which is assigned from row.service",
            "deal_label writes label_matrix[aligned_timestamp, service] = 1",
            "util/GAIA/data_GAIA.py::_transform selects label[global_end - 1] as groundtruth_real",
            "historical src/data/gaia.py assigns parsed['service'] directly to RCACaseLabel.root_service",
        ],
        "ad_label_definition": {
            "positive_node": "parsed run-table service/instance column",
            "time_alignment": "floor start and end to 30s; current code includes the aligned end bin",
            "sliding_window_target": "last timestep of each 10-step input window",
            "real_sample_checks": examples,
        },
        "rca_label_definition": {
            "root_service_expression": "str(parsed['service'])",
            "additional_causal_inference": False,
            "preferred_term": "labelled injected service or labelled fault service",
            "causal_root_cause_claim_supported": False,
        },
        "coupling": {
            "rca_supported_injections": rca_total,
            "current_ad_mappable_supported_injections": len(current_mappable),
            "current_ad_missing_supported_injections": rca_total - len(current_mappable),
            "missing_reason": "12 CPU rows have decimal durations rejected by current-main integer-only regex",
            "service_exact_matches_among_current_ad_mappable": exact_service,
            "service_exact_match_ratio_among_current_ad_mappable": exact_service / len(current_mappable),
            "service_exact_match_ratio_over_all_rca_injections_treating_missing_as_mismatch": exact_service / rca_total,
            "semantic_statement": "The service identity is shared by construction, but event identity is not one-to-one after 30s rasterization.",
        },
        "ad_timestep_multiplicity": {
            "positive_supported_injection_bins": len(injection_bin_events),
            "distinct_positive_service_count_distribution": {str(key): value for key, value in sorted(positive_bin_service_counts.items())},
            "injection_count_distribution": {str(key): value for key, value in sorted(positive_bin_injection_counts.items())},
            "same_root_multi_injection_bins": same_root_multi_bins,
            "different_root_multi_injection_bins": different_root_multi_bins,
            "maximum_injections_in_one_bin": max(positive_bin_injection_counts, default=0),
            "maximum_positive_services_in_one_bin": max(positive_bin_service_counts, default=0),
        },
        "injection_to_ad_timesteps": {
            **percentile_summary(injection_bin_counts),
            "mean": float(np.mean(injection_bin_counts)),
            "production_semantics_note": "counts reproduce current inclusive aligned-end implementation, not half-open GT semantics",
        },
        "shared_ad_positive_intervals": {
            "same_service_exact_interval_groups": len(shared_groups),
            "injections_in_shared_same_service_exact_intervals": sum(shared_groups),
            "maximum_group": max(shared_groups, default=1),
        },
        "current_noninjection_ad_entries": {
            "error_event_rows": sum(row["fault_type"] == "error_event" for row in current_ad_events),
            "normal_memory_freed_rows": sum(row["fault_type"] == "normal memory freed label" for row in current_ad_events),
            "unknown_rows": sum(row["fault_type"] == "unknown" for row in current_ad_events),
        },
        "limitations": [
            "Exact membership in a generated label.csv also depends on the metric-derived timestamp index.",
            "This audit reproduces the current alignment formula on real events without writing production label files.",
            "Service equality is annotation coupling, not independent causal validation.",
        ],
    }
    return report, reconciliation_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaia-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_table = args.gaia_root / "run" / "run" / "run" / "run_table_2021-07.csv"
    report, rows = build(run_table)
    write_json(args.output_dir / "label_reconciliation.json", report)
    write_csv(
        args.output_dir / "label_reconciliation.csv",
        (
            "raw_type",
            "raw_count",
            "raw_semantics",
            "current_ad_label",
            "historical_rca_eligible",
            "candidate_e2e_meaning",
            "evidence",
            "current_main_interval_valid_count",
            "current_ad_mapped_count",
            "p5_status",
        ),
        rows,
    )
    print(
        "rca_events={}; ad_mappable={}; all_event_service_match_ratio={:.8f}".format(
            report["coupling"]["rca_supported_injections"],
            report["coupling"]["current_ad_mappable_supported_injections"],
            report["coupling"]["service_exact_match_ratio_over_all_rca_injections_treating_missing_as_mismatch"],
        )
    )


if __name__ == "__main__":
    main()
