"""Leakage-resistant grouping and split validation for RCA cases."""

from dataclasses import dataclass
import hashlib
from typing import Dict, Iterable, Sequence, Tuple


class SplitIntegrityError(ValueError):
    """Raised when a case split violates identity or grouping constraints."""


@dataclass(frozen=True)
class CaseGroup:
    """Stable group assignment used to keep related cases in one split."""

    case_id: str
    group_id: str
    group_size: int
    overlap_degree: int


@dataclass(frozen=True)
class SplitAssignment:
    """Assignment of one case to a named partition or fold."""

    case_id: str
    split: str


def _stable_group_id(member_ids: Sequence[str]) -> str:
    payload = "rca-overlap-group-v1:" + "\n".join(sorted(member_ids))
    return "group-{}".format(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16])


def build_overlap_groups(events: Iterable[object]) -> Tuple[CaseGroup, ...]:
    """Return connected components from event overlap adjacency.

    Each event must expose case_id and overlapping_case_ids. The function
    validates references, treats the supplied graph as undirected, and assigns
    a deterministic content-derived ID to every component (including
    singletons).
    """

    rows = tuple(events)
    case_ids = [str(row.case_id) for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise SplitIntegrityError("overlap events contain duplicate case_id")
    known = set(case_ids)
    parent: Dict[str, str] = {case_id: case_id for case_id in case_ids}

    def find(case_id: str) -> str:
        while parent[case_id] != case_id:
            parent[case_id] = parent[parent[case_id]]
            case_id = parent[case_id]
        return case_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    degrees: Dict[str, int] = {}
    for row in rows:
        case_id = str(row.case_id)
        neighbors = tuple(str(value) for value in row.overlapping_case_ids)
        if len(set(neighbors)) != len(neighbors):
            raise SplitIntegrityError(
                "overlap event {!r} contains duplicate neighbors".format(case_id)
            )
        if case_id in neighbors:
            raise SplitIntegrityError(
                "overlap event {!r} references itself".format(case_id)
            )
        unknown = sorted(set(neighbors) - known)
        if unknown:
            raise SplitIntegrityError(
                "overlap event {!r} references unknown cases {}".format(
                    case_id, unknown
                )
            )
        degrees[case_id] = len(neighbors)
        for neighbor in neighbors:
            union(case_id, neighbor)

    components: Dict[str, list] = {}
    for case_id in case_ids:
        components.setdefault(find(case_id), []).append(case_id)

    group_by_case: Dict[str, Tuple[str, int]] = {}
    for members in components.values():
        members = sorted(members)
        group_id = _stable_group_id(members)
        for case_id in members:
            group_by_case[case_id] = (group_id, len(members))

    return tuple(
        CaseGroup(
            case_id=case_id,
            group_id=group_by_case[case_id][0],
            group_size=group_by_case[case_id][1],
            overlap_degree=degrees[case_id],
        )
        for case_id in sorted(case_ids)
    )


def build_singleton_groups(case_ids: Iterable[str]) -> Tuple[CaseGroup, ...]:
    """Build one deterministic group per independent case."""

    values = tuple(str(case_id) for case_id in case_ids)
    if len(set(values)) != len(values):
        raise SplitIntegrityError("case_ids contain duplicates")
    return tuple(
        CaseGroup(case_id, _stable_group_id((case_id,)), 1, 0)
        for case_id in sorted(values)
    )


def validate_split_integrity(
    assignments: Sequence[SplitAssignment],
    groups: Sequence[CaseGroup],
) -> None:
    """Require complete case coverage and prevent any group crossing splits."""

    group_ids_by_case: Dict[str, str] = {}
    declared_sizes: Dict[str, int] = {}
    observed_sizes: Dict[str, int] = {}
    for group in groups:
        if group.case_id in group_ids_by_case:
            raise SplitIntegrityError("group manifest contains duplicate case_id")
        if not group.group_id or group.group_size < 1 or group.overlap_degree < 0:
            raise SplitIntegrityError("group manifest contains invalid values")
        group_ids_by_case[group.case_id] = group.group_id
        previous_size = declared_sizes.setdefault(group.group_id, group.group_size)
        if previous_size != group.group_size:
            raise SplitIntegrityError("group_size is inconsistent within a group")
        observed_sizes[group.group_id] = observed_sizes.get(group.group_id, 0) + 1

    if declared_sizes != observed_sizes:
        raise SplitIntegrityError("declared group_size does not match member count")

    split_by_case: Dict[str, str] = {}
    for assignment in assignments:
        if not assignment.case_id or not assignment.split:
            raise SplitIntegrityError("split assignments require non-empty values")
        if assignment.case_id in split_by_case:
            raise SplitIntegrityError("split manifest contains duplicate case_id")
        split_by_case[assignment.case_id] = assignment.split

    expected = set(group_ids_by_case)
    observed = set(split_by_case)
    if expected != observed:
        raise SplitIntegrityError(
            "split/group case_id mismatch; missing={}, extra={}".format(
                sorted(expected - observed), sorted(observed - expected)
            )
        )

    split_by_group: Dict[str, str] = {}
    for case_id, split in split_by_case.items():
        group_id = group_ids_by_case[case_id]
        previous = split_by_group.setdefault(group_id, split)
        if previous != split:
            raise SplitIntegrityError(
                "group {!r} crosses splits {!r} and {!r}".format(
                    group_id, previous, split
                )
            )
