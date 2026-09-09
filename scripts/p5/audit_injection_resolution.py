#!/usr/bin/env python3
"""Quantify GAIA injection density and compatibility with a 30-second detector grid."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import itertools
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p5.common import percentile_summary, sha256_file, write_json  # noqa: E402


SCHEMA_VERSION = "p5_g0r2_temporal_resolution_v1"
THRESHOLDS = (15, 30, 60, 120, 300)
GRID_MS = 30_000


def _read_registry(path: Path) -> Sequence[Mapping[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "source_index": int(row["source_index"]),
                    "start_ms": int(row["start_ms"]),
                    "end_ms": int(row["end_ms"]),
                    "duration_seconds": float(row["duration_seconds"]),
                    "current_main_interval_valid": row["current_main_interval_valid"] == "True",
                }
            )
        return tuple(rows)


def _summary(deltas_seconds: Sequence[float]) -> Mapping[str, object]:
    result = dict(percentile_summary(deltas_seconds))
    total = len(deltas_seconds)
    result["thresholds"] = {
        "lt_{}s".format(threshold): {
            "count": sum(value < threshold for value in deltas_seconds),
            "ratio": sum(value < threshold for value in deltas_seconds) / total if total else None,
        }
        for threshold in THRESHOLDS
    }
    return result


def _consecutive_deltas(rows: Iterable[Mapping[str, object]]) -> list[float]:
    starts = sorted(int(row["start_ms"]) for row in rows)
    return [(right - left) / 1000.0 for left, right in zip(starts, starts[1:])]


def _same_root_deltas(rows: Sequence[Mapping[str, object]]) -> list[float]:
    by_root = defaultdict(list)
    for row in rows:
        by_root[str(row["service"])].append(row)
    return [value for service_rows in by_root.values() for value in _consecutive_deltas(service_rows)]


def _different_root_adjacent_deltas(rows: Sequence[Mapping[str, object]]) -> list[float]:
    ordered = sorted(rows, key=lambda row: (int(row["start_ms"]), int(row["source_index"])))
    return [
        (int(right["start_ms"]) - int(left["start_ms"])) / 1000.0
        for left, right in zip(ordered, ordered[1:])
        if left["service"] != right["service"]
    ]


def _nearest_other_seconds(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    starts = np.asarray(sorted(int(row["start_ms"]) for row in rows), dtype=np.int64)
    previous = np.full(len(starts), np.iinfo(np.int64).max, dtype=np.int64)
    following = previous.copy()
    if len(starts) > 1:
        previous[1:] = starts[1:] - starts[:-1]
        following[:-1] = starts[1:] - starts[:-1]
    return np.minimum(previous, following).astype(float) / 1000.0


def _aligned_bins(start_ms: int, end_ms: int) -> tuple[int, ...]:
    start = (start_ms // GRID_MS) * GRID_MS
    end = (end_ms // GRID_MS) * GRID_MS
    return tuple(range(start, end + GRID_MS, GRID_MS))


def build(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    all_deltas = _consecutive_deltas(rows)
    by_fault = defaultdict(list)
    for row in rows:
        by_fault[str(row["fault_type"])].append(row)

    start_bins = defaultdict(list)
    service_start_bins = defaultdict(list)
    for row in rows:
        aligned = (int(row["start_ms"]) // GRID_MS) * GRID_MS
        start_bins[aligned].append(row)
        service_start_bins[(aligned, str(row["service"]))].append(row)

    month_start = int(pd.Timestamp("2021-07-01 00:00:00", tz="Asia/Shanghai").timestamp() * 1000)
    month_end = int(pd.Timestamp("2021-08-01 00:00:00", tz="Asia/Shanghai").timestamp() * 1000)
    calendar_bin_count = (month_end - month_start) // GRID_MS
    occupancy = Counter(len(values) for values in start_bins.values())
    zero_bins = calendar_bin_count - len(start_bins)
    bucket_counts = {
        "0": zero_bins,
        "1": occupancy.get(1, 0),
        "2": occupancy.get(2, 0),
        "ge_3": sum(count for multiplicity, count in occupancy.items() if multiplicity >= 3),
    }
    bucket_report = {
        key: {"count": count, "percentage": 100.0 * count / calendar_bin_count}
        for key, count in bucket_counts.items()
    }

    colliding_bins = {key: values for key, values in start_bins.items() if len(values) > 1}
    service_colliding = {key: values for key, values in service_start_bins.items() if len(values) > 1}
    injections_in_collision = {str(row["case_id"]) for values in colliding_bins.values() for row in values}
    injections_in_service_collision = {str(row["case_id"]) for values in service_colliding.values() for row in values}

    cover_counts = [len(_aligned_bins(int(row["start_ms"]), int(row["end_ms"]))) for row in rows]
    current_cover_counts = [
        len(_aligned_bins(int(row["start_ms"]), int(row["end_ms"])))
        for row in rows
        if row["current_main_interval_valid"]
    ]
    aligned_interval_groups = Counter(
        (str(row["service"]), _aligned_bins(int(row["start_ms"]), int(row["end_ms"])))
        for row in rows
    )
    shared_interval_sizes = [value for value in aligned_interval_groups.values() if value > 1]
    service_positive_bins = defaultdict(list)
    for row in rows:
        for aligned_bin in _aligned_bins(int(row["start_ms"]), int(row["end_ms"])):
            service_positive_bins[(str(row["service"]), aligned_bin)].append(str(row["case_id"]))
    shared_service_bin_pairs = set()
    for case_ids in service_positive_bins.values():
        for left, right in itertools.combinations(sorted(set(case_ids)), 2):
            shared_service_bin_pairs.add((left, right))
    injections_sharing_service_bin = {
        case_id for pair in shared_service_bin_pairs for case_id in pair
    }

    nearest = _nearest_other_seconds(rows)
    resolution_subsets = {
        "nearest_other_gt_{}s".format(threshold): {
            "count": int(np.count_nonzero(nearest > threshold)),
            "ratio": float(np.mean(nearest > threshold)),
        }
        for threshold in (30, 60, 120)
    }

    same_root_collision_bins = 0
    different_root_collision_bins = 0
    mixed_collision_bins = 0
    for values in colliding_bins.values():
        roots = Counter(str(row["service"]) for row in values)
        has_same = any(count > 1 for count in roots.values())
        has_different = len(roots) > 1
        same_root_collision_bins += int(has_same and not has_different)
        different_root_collision_bins += int(has_different and not has_same)
        mixed_collision_bins += int(has_same and has_different)

    durations = [float(row["duration_seconds"]) for row in rows]
    report = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "REVALIDATED FACT",
        "injection_count": len(rows),
        "interarrival_seconds": {
            "all_supported_consecutive": _summary(all_deltas),
            "same_root_consecutive_pooled": _summary(_same_root_deltas(rows)),
            "different_root_adjacent_global": _summary(_different_root_adjacent_deltas(rows)),
            "by_fault_type": {
                fault_type: _summary(_consecutive_deltas(fault_rows))
                for fault_type, fault_rows in sorted(by_fault.items())
            },
        },
        "start_bin_collision_30s": {
            "alignment": "Unix-epoch floor(timestamp_ms / 30000) * 30000, matching current Ada-MGAD",
            "denominator": "all 30-second calendar bins in July 2021 Asia/Shanghai",
            "calendar_bins": int(calendar_bin_count),
            "bin_injection_count_distribution": bucket_report,
            "event_bearing_bins": len(start_bins),
            "colliding_bins": len(colliding_bins),
            "same_root_only_collision_bins": same_root_collision_bins,
            "different_root_only_collision_bins": different_root_collision_bins,
            "mixed_same_and_different_root_collision_bins": mixed_collision_bins,
            "maximum_injections_in_one_start_bin": max((len(values) for values in start_bins.values()), default=0),
            "injections_in_any_start_bin_collision": len(injections_in_collision),
            "injections_in_any_start_bin_collision_ratio": len(injections_in_collision) / len(rows),
            "injections_in_same_service_start_bin_collision": len(injections_in_service_collision),
            "injections_in_same_service_start_bin_collision_ratio": len(injections_in_service_collision) / len(rows),
        },
        "interval_to_ad_grid": {
            "duration_seconds": {
                **percentile_summary(durations),
                "mean": float(np.mean(durations)),
                "lt_30s_count": sum(value < 30 for value in durations),
                "lt_30s_ratio": sum(value < 30 for value in durations) / len(rows),
                "lt_60s_count": sum(value < 60 for value in durations),
                "lt_60s_ratio": sum(value < 60 for value in durations) / len(rows),
            },
            "audit_repaired_all_injections_inclusive_end_bin_count": {
                **percentile_summary(cover_counts),
                "mean": float(np.mean(cover_counts)),
            },
            "current_main_mappable_injections": len(current_cover_counts),
            "current_main_inclusive_end_bin_count": {
                **percentile_summary(current_cover_counts),
                "mean": float(np.mean(current_cover_counts)),
            },
            "same_service_exact_aligned_interval_groups": len(shared_interval_sizes),
            "injections_in_shared_same_service_exact_aligned_intervals": sum(shared_interval_sizes),
            "maximum_shared_group": max(shared_interval_sizes, default=1),
            "same_service_event_pairs_sharing_any_ad_positive_bin": len(shared_service_bin_pairs),
            "injections_sharing_any_same_service_ad_positive_bin": len(injections_sharing_service_bin),
            "injections_sharing_any_same_service_ad_positive_bin_ratio": len(injections_sharing_service_bin)
            / len(rows),
        },
        "resolution_compatible_subsets": resolution_subsets,
        "one_to_one_resolution": {
            "strict_time_bin_definition": {
                "not_one_to_one_count": len(injections_in_collision),
                "not_one_to_one_ratio": len(injections_in_collision) / len(rows),
                "meaning": "injection start shares a 30s output time bin with at least one other injection",
            },
            "service_aware_lower_bound": {
                "not_one_to_one_count": len(injections_in_service_collision),
                "not_one_to_one_ratio": len(injections_in_service_collision) / len(rows),
                "meaning": "injection shares both service and 30s start bin; node-time labels cannot separate them",
            },
            "decision": "30s node-time outputs are not a one-injection-per-timestep event registry",
        },
        "limitations": [
            "The zero-bin denominator is the full July calendar, not the metric-derived observed timestamp subset.",
            "Different-service events in one time bin can remain distinct node positives, while same-service collisions cannot.",
            "Current Ada-MGAD includes the aligned end bin; raw injection GT remains half-open and is not merged here.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build(_read_registry(args.event_registry))
    report["source"] = {
        "event_registry": str(args.event_registry.resolve()),
        "event_registry_sha256": sha256_file(args.event_registry),
        "grid_seconds": GRID_MS // 1000,
    }
    write_json(args.output, report)
    strict = report["one_to_one_resolution"]["strict_time_bin_definition"]
    lower = report["one_to_one_resolution"]["service_aware_lower_bound"]
    print(
        "strict_collision={}/{} ({:.4%}); same_service_collision={}/{} ({:.4%})".format(
            strict["not_one_to_one_count"],
            report["injection_count"],
            strict["not_one_to_one_ratio"],
            lower["not_one_to_one_count"],
            report["injection_count"],
            lower["not_one_to_one_ratio"],
        )
    )


if __name__ == "__main__":
    main()
