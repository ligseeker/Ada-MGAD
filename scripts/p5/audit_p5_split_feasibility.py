#!/usr/bin/env python3
"""Audit split feasibility for the GAIA P5 two-stage protocol.

This is an audit-only implementation.  It consumes the already reconciled
``event_registry.csv`` and never reads or writes production preprocessing or
model artifacts.  Protocol S keeps expanded context groups intact; Protocol T
uses chronological contiguous blocks and reports purge requirements.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p5.common import GAIA_SERVICES, SUPPORTED_FAULTS, sha256_file, write_json  # noqa: E402


SCHEMA_VERSION = "p5_g0r2_split_feasibility_v1"
FOLDS = 5
CONTEXT_WINDOWS = (300, 600)
RARE_FAULTS = ("cpu_anomalies", "access permission denied exception")
GRID_MS = 30_000
DETECTOR_BINS = 10


@dataclass(frozen=True)
class Event:
    case_id: str
    source_index: int
    service: str
    fault_type: str
    start_ms: int
    end_ms: int


def read_registry(path: Path) -> Tuple[Event, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                Event(
                    case_id=str(row["case_id"]),
                    source_index=int(row["source_index"]),
                    service=str(row["service"]),
                    fault_type=str(row["fault_type"]),
                    start_ms=int(row["start_ms"]),
                    end_ms=int(row["end_ms"]),
                )
            )
    events = tuple(sorted(rows, key=lambda e: (e.start_ms, e.end_ms, e.source_index)))
    if not events:
        raise ValueError("event registry is empty")
    if len({e.case_id for e in events}) != len(events):
        raise ValueError("event registry has duplicate case_id")
    if any(e.start_ms >= e.end_ms for e in events):
        raise ValueError("event registry has non-positive interval")
    if any(e.service not in GAIA_SERVICES for e in events):
        raise ValueError("event registry contains service outside GAIA_SERVICES")
    if any(e.fault_type not in SUPPORTED_FAULTS for e in events):
        raise ValueError("event registry contains unsupported fault type")
    return events


def _expanded_interval(event: Event, seconds: int) -> Tuple[int, int]:
    radius = seconds * 1000
    return min(event.start_ms, event.start_ms - radius), max(event.end_ms, event.start_ms + radius)


def build_groups(events: Sequence[Event], seconds: int) -> Tuple[Tuple[int, ...], ...]:
    """Build connected components of raw injection union expanded context."""

    parent = list(range(len(events)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    active: List[Tuple[int, int]] = []
    expanded = [_expanded_interval(event, seconds) for event in events]
    order = sorted(range(len(events)), key=lambda i: (expanded[i][0], expanded[i][1], events[i].source_index))
    for index in order:
        start, end = expanded[index]
        active = [(other, other_end) for other, other_end in active if other_end > start]
        for other, _ in active:
            union(index, other)
        active.append((index, end))
    groups: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(events)):
        groups[find(index)].append(index)
    return tuple(sorted((tuple(sorted(values)) for values in groups.values()), key=lambda g: (g[0], len(g))))


def _group_counts(group: Sequence[int], events: Sequence[Event]) -> Mapping[str, Counter]:
    return {
        "root": Counter(events[i].service for i in group),
        "fault": Counter(events[i].fault_type for i in group),
    }


def grouped_stratified_assignment(events: Sequence[Event], groups: Sequence[Sequence[int]]) -> Mapping[int, int]:
    """Deterministic, performance-blind greedy grouped stratification."""

    total_root = Counter(e.service for e in events)
    total_fault = Counter(e.fault_type for e in events)
    fold_root = [Counter() for _ in range(FOLDS)]
    fold_fault = [Counter() for _ in range(FOLDS)]
    fold_sizes = [0] * FOLDS
    group_counts = {tuple(group): _group_counts(group, events) for group in groups}

    def rarity(group: Sequence[int]) -> Tuple[float, int]:
        counts = group_counts[tuple(group)]
        rare = min(
            [total_root[key] / max(value, 1) for key, value in counts["root"].items()]
            + [total_fault[key] / max(value, 1) for key, value in counts["fault"].items()]
        )
        return (rare, group[0])

    def objective(candidate_root: Sequence[Counter], candidate_fault: Sequence[Counter], candidate_sizes: Sequence[int]) -> float:
        # Squared deviation across all labels makes empty-fold and rare-label
        # deficits visible, while the size term keeps groups distributed.
        value = 5.0 * sum((size / max(len(events), 1) - 1 / FOLDS) ** 2 for size in candidate_sizes)
        value += sum(
            (candidate_root[fold][key] / total_root[key] - 1 / FOLDS) ** 2
            for key in total_root
            for fold in range(FOLDS)
        )
        value += sum(
            (candidate_fault[fold][key] / total_fault[key] - 1 / FOLDS) ** 2
            for key in total_fault
            for fold in range(FOLDS)
        )
        return value

    assignment: Dict[int, int] = {}
    # Place large groups first.  This is deterministic and avoids the common
    # failure mode where a late large component makes one fold unusably large.
    ordered = sorted(groups, key=lambda group: (-len(group), rarity(group), group[0]))
    for group in ordered:
        counts = group_counts[tuple(group)]
        scores = []
        for fold in range(FOLDS):
            size_after = fold_sizes[fold] + len(group)
            fold_root[fold].update(counts["root"])
            fold_fault[fold].update(counts["fault"])
            fold_sizes[fold] = size_after
            scores.append((objective(fold_root, fold_fault, fold_sizes), fold))
            fold_root[fold].subtract(counts["root"])
            fold_fault[fold].subtract(counts["fault"])
            fold_sizes[fold] -= len(group)
        fold = min(scores)[1]
        for index in group:
            assignment[index] = fold
        fold_sizes[fold] += len(group)
        fold_root[fold].update(counts["root"])
        fold_fault[fold].update(counts["fault"])
    return assignment


def _distribution(values: Iterable[int]) -> Mapping[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def _fold_summary(events: Sequence[Event], assignment: Mapping[int, int]) -> Mapping[str, object]:
    folds = []
    for fold in range(FOLDS):
        selected = [events[i] for i, value in assignment.items() if value == fold]
        folds.append(
            {
                "fold": fold,
                "event_count": len(selected),
                "root_counts": dict(sorted(Counter(e.service for e in selected).items())),
                "fault_counts": dict(sorted(Counter(e.fault_type for e in selected).items())),
                "missing_roots": sorted(set(GAIA_SERVICES) - {e.service for e in selected}),
                "missing_faults": sorted(set(SUPPORTED_FAULTS) - {e.fault_type for e in selected}),
            }
        )
    groups_by_fold = defaultdict(set)
    for index, fold in assignment.items():
        groups_by_fold[fold].add(index)
    return {"folds": folds, "fold_event_counts": [row["event_count"] for row in folds]}


def protocol_s(events: Sequence[Event], seconds: int) -> Mapping[str, object]:
    groups = build_groups(events, seconds)
    assignment = grouped_stratified_assignment(events, groups)
    integrity_violations = []
    for group_id, group in enumerate(groups):
        folds = {assignment[index] for index in group}
        if len(folds) != 1:
            integrity_violations.append({"group_id": group_id, "folds": sorted(folds)})
    sizes = [len(group) for group in groups]
    return {
        "context_seconds": seconds,
        "context_definition": "raw [start,end) union [start-W,start+W)",
        "group_count": len(groups),
        "non_singleton_group_count": sum(size > 1 for size in sizes),
        "group_size_distribution": _distribution(sizes),
        "maximum_group_size": max(sizes),
        "grouped_stratified_5fold": _fold_summary(events, assignment),
        "group_integrity": {
            "violations": integrity_violations,
            "passed": not integrity_violations,
        },
        "feasible": not integrity_violations
        and all(not row["missing_roots"] and not row["missing_faults"] for row in _fold_summary(events, assignment)["folds"]),
        "selection_basis": "coverage, group integrity and deterministic balance only; no model performance",
    }


def _local_date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone(timedelta(hours=8))).date().isoformat()


def _collision_stats(selected: Sequence[Event]) -> Mapping[str, object]:
    bins: Dict[int, List[Event]] = defaultdict(list)
    for event in selected:
        bins[(event.start_ms // GRID_MS) * GRID_MS].append(event)
    collision = [values for values in bins.values() if len(values) > 1]
    multi_root = [values for values in collision if len({e.service for e in values}) > 1]
    return {
        "event_bearing_start_bins": len(bins),
        "collision_bins": len(collision),
        "collision_events": sum(len(values) for values in collision),
        "maximum_events_per_start_bin": max((len(values) for values in bins.values()), default=0),
        "multi_root_collision_bins": len(multi_root),
        "multi_root_collision_events": sum(len(values) for values in multi_root),
    }


def _rare_distribution(events: Sequence[Event]) -> Mapping[str, object]:
    by_date_root_fault = Counter((_local_date(e.start_ms), e.service, e.fault_type) for e in events)
    by_date = defaultdict(Counter)
    for (date, root, fault), count in by_date_root_fault.items():
        by_date[date][f"{root}::{fault}"] += count
    rare_dates = {
        date: dict(sorted(values.items()))
        for date, values in sorted(by_date.items())
        if any(key.split("::", 1)[1] in RARE_FAULTS for key in values)
    }
    return {
        "rare_faults": list(RARE_FAULTS),
        "by_date_root_fault": rare_dates,
        "rare_event_rows": [
            {
                "case_id": e.case_id,
                "date": _local_date(e.start_ms),
                "root": e.service,
                "fault": e.fault_type,
                "start_ms": e.start_ms,
            }
            for e in events
            if e.fault_type in RARE_FAULTS
        ],
    }


def _block_summary(events: Sequence[Event], indices: Sequence[int], name: str) -> Mapping[str, object]:
    selected = [events[i] for i in indices]
    if not selected:
        return {"name": name, "event_count": 0, "empty": True}
    return {
        "name": name,
        "event_count": len(selected),
        "start_min_ms": min(e.start_ms for e in selected),
        "start_max_ms": max(e.start_ms for e in selected),
        "end_max_ms": max(e.end_ms for e in selected),
        "root_counts": dict(sorted(Counter(e.service for e in selected).items())),
        "fault_counts": dict(sorted(Counter(e.fault_type for e in selected).items())),
        "missing_roots": sorted(set(GAIA_SERVICES) - {e.service for e in selected}),
        "missing_faults": sorted(set(SUPPORTED_FAULTS) - {e.fault_type for e in selected}),
        "rare_fault_counts": dict(sorted(Counter(e.fault_type for e in selected if e.fault_type in RARE_FAULTS).items())),
        "collision_stats": _collision_stats(selected),
    }


def protocol_t(events: Sequence[Event], mode: str) -> Mapping[str, object]:
    ordered = sorted(range(len(events)), key=lambda i: (events[i].start_ms, events[i].source_index))
    starts = [events[i].start_ms for i in ordered]
    first_start, last_end = min(e.start_ms for e in events), max(e.end_ms for e in events)
    if mode == "duration_60_20_20":
        b1 = first_start + (last_end - first_start) * 3 // 5
        b2 = first_start + (last_end - first_start) * 4 // 5
        boundaries = (b1, b2)
    elif mode == "event_count_60_20_20":
        boundaries = (starts[(3 * len(starts)) // 5], starts[(4 * len(starts)) // 5])
    else:
        raise ValueError(mode)
    block_indices = [
        [i for i in ordered if events[i].start_ms < boundaries[0]],
        [i for i in ordered if boundaries[0] <= events[i].start_ms < boundaries[1]],
        [i for i in ordered if events[i].start_ms >= boundaries[1]],
    ]
    blocks = [_block_summary(events, values, name) for values, name in zip(block_indices, ("train", "validation", "test"))]
    for block, (left, right) in zip(blocks, ((first_start, boundaries[0]), boundaries, (boundaries[1], last_end))):
        block["time_block_start_ms"] = left
        block["time_block_end_ms"] = right
        block["duration_seconds"] = (right - left) / 1000.0

    crossing = {}
    purge = {}
    for seconds in CONTEXT_WINDOWS:
        radius = seconds * 1000
        expanded = [_expanded_interval(e, seconds) for e in events]
        cross_indices = {i for i, (start, end) in enumerate(expanded) if any(start < b < end for b in boundaries)}
        kept = [i for i in ordered if i not in cross_indices]
        crossing[str(seconds)] = {
            "boundary_crossing_events": len(cross_indices),
            "boundary_crossing_ratio": len(cross_indices) / len(events),
            "crossing_case_ids": sorted(events[i].case_id for i in cross_indices),
        }
        purge[str(seconds)] = {
            "before_events": len(events),
            "purged_events": len(cross_indices),
            "retained_events": len(kept),
            "retention_ratio": len(kept) / len(events),
            "retained_by_block": [
                sum(events[i].start_ms < boundaries[0] for i in kept),
                sum(boundaries[0] <= events[i].start_ms < boundaries[1] for i in kept),
                sum(events[i].start_ms >= boundaries[1] for i in kept),
            ],
            "purge_rule": "remove event if expanded raw/context interval crosses either chronological boundary",
        }
    # Current Ada-MGAD selects the label at global_end - 1 while its 10-bin
    # input contains that target bin and the preceding nine bins.
    detector_window = DETECTOR_BINS * GRID_MS
    detector_cross = {
        "window_bins": DETECTOR_BINS,
        "window_seconds": detector_window / 1000,
        "definition": "[floor(start/30s)*30s - 9*30s, floor(start/30s)*30s + 30s)",
        "boundary_crossing_events": sum(
            1
            for e in events
            if any(
                (e.start_ms // GRID_MS) * GRID_MS - (DETECTOR_BINS - 1) * GRID_MS
                < b
                < (e.start_ms // GRID_MS) * GRID_MS + GRID_MS
                for b in boundaries
            )
        ),
    }
    return {
        "mode": mode,
        "boundary_definition": "chronological contiguous blocks; no event-window concatenation",
        "boundaries_ms": list(boundaries),
        "blocks": blocks,
        "context_crossing": crossing,
        "purge": purge,
        "ada_mgad_sliding_window_crossing": detector_cross,
    }


def build(events: Sequence[Event], registry_path: Path) -> Mapping[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "REVALIDATED FACT",
        "event_registry": {
            "path": str(registry_path.resolve()),
            "sha256": sha256_file(registry_path),
            "event_count": len(events),
            "start_min_ms": min(e.start_ms for e in events),
            "end_max_ms": max(e.end_ms for e in events),
        },
        "protocol_s_component_integration": {
            "purpose": "component/integration evaluation; not temporal deployment generalization",
            "windows": {str(seconds): protocol_s(events, seconds) for seconds in CONTEXT_WINDOWS},
        },
        "protocol_t_continuous_temporal_e2e": {
            "purpose": "continuous chronological trigger-to-RCA evaluation feasibility",
            "protocols": {
                mode: protocol_t(events, mode)
                for mode in ("duration_60_20_20", "event_count_60_20_20")
            },
        },
        "rare_fault_distribution": _rare_distribution(events),
        "interpretation": {
            "selection_rule": "No AC@1, Avg@5, MRR, F1, or model performance was used.",
            "raw_injection_gt": "Events remain raw registry units; no event merging was performed.",
            "context_vs_raw": "Protocol S grouping uses expanded context; Protocol T reports raw chronological blocks plus explicit purge.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    events = read_registry(args.event_registry)
    report = build(events, args.event_registry)
    write_json(args.output, report)
    print("wrote {} ({} events)".format(args.output, len(events)))


if __name__ == "__main__":
    main()
