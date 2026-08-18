"""Event concurrency flags for designated-root RCA cohorts."""

from dataclasses import dataclass
from bisect import bisect_left
from typing import Dict, Mapping, Sequence, Tuple

from src.data.split import CaseGroup, SplitIntegrityError


@dataclass(frozen=True)
class LabeledEventInterval:
    case_id: str
    start_ms: int
    end_ms: int
    root_service: str
    fault_type: str


@dataclass(frozen=True)
class EventPurityFlag:
    case_id: str
    actual_overlap_degree: int
    anchor_concurrent_degree: int
    anchor_concurrent_other_root_count: int
    anchor_has_multiple_root_services: bool
    context_group_size: int
    context_overlap_degree: int
    context_group_root_service_count: int
    context_group_fault_type_count: int
    other_event_starts_in_context: int


def build_event_purity_flags(
    events: Sequence[LabeledEventInterval],
    context_groups: Sequence[CaseGroup],
    context_radius_ms: int,
) -> Tuple[EventPurityFlag, ...]:
    """Describe actual concurrency separately from expanded context overlap."""

    if context_radius_ms <= 0:
        raise SplitIntegrityError("context_radius_ms must be positive")
    rows = tuple(events)
    case_ids = [row.case_id for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise SplitIntegrityError("events contain duplicate case_id")
    by_case = {}
    for row in rows:
        if (
            not row.case_id
            or not row.root_service
            or not row.fault_type
            or int(row.start_ms) >= int(row.end_ms)
        ):
            raise SplitIntegrityError("events contain invalid labeled intervals")
        by_case[row.case_id] = row

    group_by_case = {}
    members_by_group: Dict[str, list] = {}
    for group in context_groups:
        if group.case_id in group_by_case:
            raise SplitIntegrityError("context groups contain duplicate case_id")
        group_by_case[group.case_id] = group
        members_by_group.setdefault(group.group_id, []).append(group.case_id)
    if set(group_by_case) != set(by_case):
        raise SplitIntegrityError("event/context-group case IDs do not match")
    for group_id, members in members_by_group.items():
        declared = {group_by_case[case_id].group_size for case_id in members}
        if declared != {len(members)}:
            raise SplitIntegrityError(
                "context group {!r} has inconsistent size".format(group_id)
            )

    neighbors = {case_id: set() for case_id in case_ids}
    active = []
    for row in sorted(rows, key=lambda value: (value.start_ms, value.end_ms, value.case_id)):
        active = [other for other in active if other.end_ms > row.start_ms]
        for other in active:
            neighbors[row.case_id].add(other.case_id)
            neighbors[other.case_id].add(row.case_id)
        active.append(row)

    sorted_starts = sorted(int(row.start_ms) for row in rows)
    flags = []
    for case_id in sorted(case_ids):
        row = by_case[case_id]
        concurrent = [
            neighbor_id
            for neighbor_id in neighbors[case_id]
            if by_case[neighbor_id].start_ms <= row.start_ms
            < by_case[neighbor_id].end_ms
        ]
        other_roots = {
            by_case[neighbor_id].root_service
            for neighbor_id in concurrent
            if by_case[neighbor_id].root_service != row.root_service
        }
        group = group_by_case[case_id]
        group_members = members_by_group[group.group_id]
        left = bisect_left(sorted_starts, row.start_ms - context_radius_ms)
        right = bisect_left(sorted_starts, row.start_ms + context_radius_ms)
        flags.append(
            EventPurityFlag(
                case_id=case_id,
                actual_overlap_degree=len(neighbors[case_id]),
                anchor_concurrent_degree=len(concurrent),
                anchor_concurrent_other_root_count=len(other_roots),
                anchor_has_multiple_root_services=bool(other_roots),
                context_group_size=group.group_size,
                context_overlap_degree=group.overlap_degree,
                context_group_root_service_count=len(
                    {by_case[member].root_service for member in group_members}
                ),
                context_group_fault_type_count=len(
                    {by_case[member].fault_type for member in group_members}
                ),
                other_event_starts_in_context=max(right - left - 1, 0),
            )
        )
    return tuple(flags)
