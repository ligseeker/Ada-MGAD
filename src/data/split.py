"""Leakage-resistant grouping, fold assignment, and split validation."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple


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
class CaseInterval:
    """Half-open telemetry/injection interval used for leakage grouping."""

    case_id: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SplitAssignment:
    """Assignment of one case to a named partition or fold."""

    case_id: str
    split: str


def _stable_group_id(member_ids: Sequence[str]) -> str:
    payload = "rca-overlap-group-v1:" + "\n".join(sorted(member_ids))
    return "group-{}".format(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16])


def _stable_seed_value(seed: int, *parts: str) -> int:
    payload = "rca-fold-v1:{}:{}".format(int(seed), "\n".join(parts))
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    )


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


def build_interval_overlap_groups(
    intervals: Iterable[CaseInterval],
) -> Tuple[CaseGroup, ...]:
    """Group cases whose half-open intervals overlap directly or transitively.

    Boundary-touching intervals do not overlap. ``overlap_degree`` counts
    direct interval overlaps, while ``group_size`` reflects the full connected
    component. This is suitable for grouping expanded telemetry contexts before
    assigning cases to folds.
    """

    rows = tuple(intervals)
    case_ids = [str(row.case_id) for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise SplitIntegrityError("intervals contain duplicate case_id")

    normalized = []
    for row in rows:
        case_id = str(row.case_id)
        start_ms = int(row.start_ms)
        end_ms = int(row.end_ms)
        if not case_id or start_ms >= end_ms:
            raise SplitIntegrityError(
                "intervals require non-empty case_id and start_ms < end_ms"
            )
        normalized.append((case_id, start_ms, end_ms))

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

    degrees: Dict[str, int] = {case_id: 0 for case_id in case_ids}
    active = []
    for case_id, start_ms, end_ms in sorted(
        normalized, key=lambda item: (item[1], item[2], item[0])
    ):
        active = [
            (other_id, other_end)
            for other_id, other_end in active
            if other_end > start_ms
        ]
        for other_id, _ in active:
            degrees[case_id] += 1
            degrees[other_id] += 1
            union(case_id, other_id)
        active.append((case_id, end_ms))

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


def assign_balanced_group_folds(
    groups: Sequence[CaseGroup],
    strata_by_case: Mapping[str, Mapping[str, str]],
    n_folds: int = 5,
    seed: int = 0,
    axis_weights: Mapping[str, float] = None,
    size_weight: float = 1.0,
) -> Tuple[SplitAssignment, ...]:
    """Assign whole groups to deterministic, approximately stratified folds.

    ``strata_by_case`` maps each case to one category per named axis. The
    greedy objective minimizes the change in normalized squared imbalance for
    fold size and every axis. Rare-label-bearing groups are placed first, while
    ``seed`` is used only for deterministic tie breaking.
    """

    rows = tuple(groups)
    if not rows:
        raise SplitIntegrityError("groups must not be empty")
    if n_folds < 2:
        raise SplitIntegrityError("n_folds must be at least 2")

    validate_split_integrity(
        tuple(SplitAssignment(row.case_id, "all") for row in rows), rows
    )
    members_by_group: Dict[str, list] = {}
    for row in rows:
        members_by_group.setdefault(row.group_id, []).append(row.case_id)
    if len(members_by_group) < n_folds:
        raise SplitIntegrityError("n_folds exceeds the number of atomic groups")

    expected = {row.case_id for row in rows}
    observed = {str(case_id) for case_id in strata_by_case}
    if expected != observed:
        raise SplitIntegrityError(
            "strata/group case_id mismatch; missing={}, extra={}".format(
                sorted(expected - observed), sorted(observed - expected)
            )
        )

    first_case = min(expected)
    axes = tuple(sorted(strata_by_case[first_case]))
    if not axes:
        raise SplitIntegrityError("at least one stratification axis is required")
    normalized: Dict[str, Dict[str, str]] = {}
    for case_id in sorted(expected):
        values = strata_by_case[case_id]
        if tuple(sorted(values)) != axes:
            raise SplitIntegrityError(
                "stratification axes must be identical for every case"
            )
        normalized[case_id] = {}
        for axis in axes:
            category = str(values[axis])
            if not axis or not category:
                raise SplitIntegrityError(
                    "stratification axes and categories must be non-empty"
                )
            normalized[case_id][axis] = category

    weights = {axis: 1.0 for axis in axes}
    for axis, value in (axis_weights or {}).items():
        if axis not in weights:
            raise SplitIntegrityError(
                "axis_weights contains unknown axis {!r}".format(axis)
            )
        weights[axis] = float(value)
    if (
        not math.isfinite(float(size_weight))
        or float(size_weight) < 0
        or any(not math.isfinite(value) or value < 0 for value in weights.values())
        or (float(size_weight) == 0 and not any(weights.values()))
    ):
        raise SplitIntegrityError("balance weights must be finite and non-negative")

    totals = {axis: Counter() for axis in axes}
    group_counts = {}
    for group_id, members in members_by_group.items():
        counts = {axis: Counter() for axis in axes}
        for case_id in members:
            for axis in axes:
                category = normalized[case_id][axis]
                totals[axis][category] += 1
                counts[axis][category] += 1
        group_counts[group_id] = counts

    def rarity_score(group_id: str) -> float:
        return sum(
            weights[axis]
            * sum(
                count / totals[axis][category]
                for category, count in group_counts[group_id][axis].items()
            )
            for axis in axes
        )

    ordered_groups = sorted(
        members_by_group,
        key=lambda group_id: (
            -rarity_score(group_id),
            -len(members_by_group[group_id]),
            _stable_seed_value(seed, "group-order", group_id),
            group_id,
        ),
    )

    target_size = len(expected) / float(n_folds)
    target_counts = {
        axis: {
            category: count / float(n_folds)
            for category, count in totals[axis].items()
        }
        for axis in axes
    }
    fold_sizes = [0] * n_folds
    fold_counts = [
        {axis: Counter() for axis in axes}
        for _ in range(n_folds)
    ]
    fold_groups = [set() for _ in range(n_folds)]
    fold_by_group = {}

    def fold_loss(fold_index: int) -> float:
        loss = float(size_weight) * (
            (fold_sizes[fold_index] - target_size) / target_size
        ) ** 2
        for axis in axes:
            categories = target_counts[axis]
            axis_loss = 0.0
            for category, target in categories.items():
                observed_count = fold_counts[fold_index][axis][category]
                axis_loss += ((observed_count - target) / max(target, 1.0)) ** 2
            loss += weights[axis] * axis_loss / len(categories)
        return loss

    for group_id in ordered_groups:
        group_size = len(members_by_group[group_id])
        candidates = []
        for fold_index in range(n_folds):
            before = fold_loss(fold_index)
            fold_sizes[fold_index] += group_size
            for axis in axes:
                fold_counts[fold_index][axis].update(group_counts[group_id][axis])
            after = fold_loss(fold_index)
            fold_sizes[fold_index] -= group_size
            for axis in axes:
                fold_counts[fold_index][axis].subtract(group_counts[group_id][axis])
            candidates.append(
                (
                    after - before,
                    fold_sizes[fold_index],
                    _stable_seed_value(seed, group_id, str(fold_index)),
                    fold_index,
                )
            )
        fold_index = min(candidates)[-1]
        fold_by_group[group_id] = fold_index
        fold_groups[fold_index].add(group_id)
        fold_sizes[fold_index] += group_size
        for axis in axes:
            fold_counts[fold_index][axis].update(group_counts[group_id][axis])

    def adjust_group(group_id: str, fold_index: int, direction: int) -> None:
        fold_sizes[fold_index] += direction * len(members_by_group[group_id])
        for axis in axes:
            for category, count in group_counts[group_id][axis].items():
                fold_counts[fold_index][axis][category] += direction * count

    # Deterministic local improvement repairs avoidable greedy imbalances. Moves
    # are general; pair swaps are bounded to smaller problems such as RE2-OB.
    sorted_group_ids = tuple(sorted(ordered_groups))
    for _ in range(len(ordered_groups) * 2):
        best_move = None
        for group_id in sorted_group_ids:
            source = fold_by_group[group_id]
            if len(fold_groups[source]) <= 1:
                continue
            for destination in range(n_folds):
                if destination == source:
                    continue
                before = fold_loss(source) + fold_loss(destination)
                adjust_group(group_id, source, -1)
                adjust_group(group_id, destination, 1)
                delta = fold_loss(source) + fold_loss(destination) - before
                adjust_group(group_id, destination, -1)
                adjust_group(group_id, source, 1)
                candidate = (
                    delta,
                    _stable_seed_value(
                        seed, "move", group_id, str(source), str(destination)
                    ),
                    group_id,
                    destination,
                )
                if best_move is None or candidate < best_move:
                    best_move = candidate
        if best_move is None or best_move[0] >= -1e-12:
            break
        _, _, group_id, destination = best_move
        source = fold_by_group[group_id]
        adjust_group(group_id, source, -1)
        adjust_group(group_id, destination, 1)
        fold_groups[source].remove(group_id)
        fold_groups[destination].add(group_id)
        fold_by_group[group_id] = destination

    if len(ordered_groups) <= 128:
        for _ in range(len(ordered_groups) * 2):
            best_swap = None
            for left_index, left_group in enumerate(sorted_group_ids):
                left_fold = fold_by_group[left_group]
                for right_group in sorted_group_ids[left_index + 1 :]:
                    right_fold = fold_by_group[right_group]
                    if left_fold == right_fold:
                        continue
                    before = fold_loss(left_fold) + fold_loss(right_fold)
                    adjust_group(left_group, left_fold, -1)
                    adjust_group(right_group, right_fold, -1)
                    adjust_group(left_group, right_fold, 1)
                    adjust_group(right_group, left_fold, 1)
                    delta = fold_loss(left_fold) + fold_loss(right_fold) - before
                    adjust_group(right_group, left_fold, -1)
                    adjust_group(left_group, right_fold, -1)
                    adjust_group(right_group, right_fold, 1)
                    adjust_group(left_group, left_fold, 1)
                    candidate = (
                        delta,
                        _stable_seed_value(
                            seed, "swap", left_group, right_group
                        ),
                        left_group,
                        right_group,
                    )
                    if best_swap is None or candidate < best_swap:
                        best_swap = candidate
            if best_swap is None or best_swap[0] >= -1e-12:
                break
            _, _, left_group, right_group = best_swap
            left_fold = fold_by_group[left_group]
            right_fold = fold_by_group[right_group]
            adjust_group(left_group, left_fold, -1)
            adjust_group(right_group, right_fold, -1)
            adjust_group(left_group, right_fold, 1)
            adjust_group(right_group, left_fold, 1)
            fold_groups[left_fold].remove(left_group)
            fold_groups[right_fold].remove(right_group)
            fold_groups[left_fold].add(right_group)
            fold_groups[right_fold].add(left_group)
            fold_by_group[left_group] = right_fold
            fold_by_group[right_group] = left_fold

    assignments = tuple(
        SplitAssignment(row.case_id, "fold-{}".format(fold_by_group[row.group_id]))
        for row in sorted(rows, key=lambda value: value.case_id)
    )
    validate_split_integrity(assignments, rows)
    return assignments


def assign_contiguous_group_folds(
    groups: Sequence[CaseGroup],
    order_by_case: Mapping[str, int],
    n_folds: int = 5,
) -> Tuple[SplitAssignment, ...]:
    """Partition ordered, non-interleaving atomic groups into balanced folds.

    Dynamic programming chooses contiguous group boundaries that minimize the
    sum of squared fold-size deviations. Group order ranges may touch but must
    not interleave, which makes the result suitable for context-purged temporal
    blocks when ``order_by_case`` contains case anchor times.
    """

    rows = tuple(groups)
    if not rows:
        raise SplitIntegrityError("groups must not be empty")
    if n_folds < 2:
        raise SplitIntegrityError("n_folds must be at least 2")
    validate_split_integrity(
        tuple(SplitAssignment(row.case_id, "all") for row in rows), rows
    )

    expected = {row.case_id for row in rows}
    observed = {str(case_id) for case_id in order_by_case}
    if expected != observed:
        raise SplitIntegrityError(
            "order/group case_id mismatch; missing={}, extra={}".format(
                sorted(expected - observed), sorted(observed - expected)
            )
        )

    members_by_group: Dict[str, list] = {}
    for row in rows:
        members_by_group.setdefault(row.group_id, []).append(row.case_id)
    if len(members_by_group) < n_folds:
        raise SplitIntegrityError("n_folds exceeds the number of atomic groups")

    ordered_groups = []
    for group_id, members in members_by_group.items():
        values = [int(order_by_case[case_id]) for case_id in members]
        ordered_groups.append(
            (min(values), max(values), group_id, tuple(sorted(members)))
        )
    ordered_groups.sort(key=lambda value: (value[0], value[1], value[2]))
    for previous, current in zip(ordered_groups, ordered_groups[1:]):
        if previous[1] > current[0]:
            raise SplitIntegrityError("atomic group order ranges interleave")

    group_sizes = [len(value[3]) for value in ordered_groups]
    group_count = len(group_sizes)
    target_size = sum(group_sizes) / float(n_folds)
    prefix = [0]
    for size in group_sizes:
        prefix.append(prefix[-1] + size)

    infinity = float("inf")
    costs = [[infinity] * (group_count + 1) for _ in range(n_folds + 1)]
    previous_cut = [[None] * (group_count + 1) for _ in range(n_folds + 1)]
    costs[0][0] = 0.0
    for part_count in range(1, n_folds + 1):
        minimum_end = part_count
        maximum_end = group_count - (n_folds - part_count)
        for end in range(minimum_end, maximum_end + 1):
            best_cost = infinity
            best_start = None
            for start in range(part_count - 1, end):
                if costs[part_count - 1][start] == infinity:
                    continue
                segment_size = prefix[end] - prefix[start]
                candidate = costs[part_count - 1][start] + (
                    segment_size - target_size
                ) ** 2
                if candidate < best_cost or (
                    candidate == best_cost
                    and (best_start is None or start < best_start)
                ):
                    best_cost = candidate
                    best_start = start
            costs[part_count][end] = best_cost
            previous_cut[part_count][end] = best_start

    segments = []
    end = group_count
    for part_count in range(n_folds, 0, -1):
        start = previous_cut[part_count][end]
        if start is None:
            raise SplitIntegrityError("could not construct contiguous folds")
        segments.append((part_count - 1, start, end))
        end = start
    segments.reverse()

    fold_by_group = {}
    for fold_index, start, end in segments:
        for _, _, group_id, _ in ordered_groups[start:end]:
            fold_by_group[group_id] = fold_index
    assignments = tuple(
        SplitAssignment(row.case_id, "fold-{}".format(fold_by_group[row.group_id]))
        for row in sorted(rows, key=lambda value: value.case_id)
    )
    validate_split_integrity(assignments, rows)
    return assignments


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
