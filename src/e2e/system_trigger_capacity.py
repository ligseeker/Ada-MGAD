"""P6-C0F structural capacity: label reference, grid bound and episode bound.

These are pure analysis helpers for the P6-C0F failure-mechanism audit.  They
never train, never change a threshold, never modify the frozen episode/matching
semantics and never write anything: the driver decides where results go.

Three quantities are computed on a split's **legal prediction slot grid** (the
``prediction_available_time`` values of the windows that actually exist in that
split):

``R_label``
    Label direct-decoding reference.  The frozen tri-state trigger label is turned
    into a binary sequence (POSITIVE -> 1, NEGATIVE -> 0, IGNORE -> 0), then fed
    through the *original* episode construction and matching.  This only measures
    whether the label decoding and the event evaluation are aligned; it is **not**
    a model-reachable recall ceiling (see the counterexample test).

``U_grid``
    Relaxed upper bound: every legal prediction timestamp is a capacity-1 slot and
    may match any GT event with ``0 <= t - onset <= tolerance`` (causal,
    one-to-one, maximum cardinality).  No episode-merging constraint is applied,
    so this is usually not realisable by a single binary sequence.

``U_episode``
    Upper bound under the complete protocol: the maximum, over binary sequences on
    the legal grid, of the number of GT events matched after the original episode
    construction and the original causal matching.  Solved exactly by dynamic
    programming (see ``episode_bound``) and verified by replaying the witness
    through the frozen pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .event_detection import _min_cost_max_cardinality
from .system_trigger import (
    DEFAULT_GRID_SECONDS,
    DEFAULT_TOLERANCE_SECONDS,
    TRIGGER_POSITIVE,
    evaluate_system_threshold,
    system_score_frame,
)


# ---------------------------------------------------------------------------
# shared geometry
# ---------------------------------------------------------------------------


def adjacency_ranges(
    slot_times_ms: np.ndarray,
    onsets_ms: np.ndarray,
    *,
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
) -> np.ndarray:
    """Per GT event, the half-open range of legal slot indices it may match.

    Returns an ``(n_events, 2)`` int array of ``[left, right)`` index pairs with
    ``0 <= slot_time - onset <= tolerance``.  An empty range is ``left == right``.
    """

    slots = np.asarray(slot_times_ms, dtype=np.int64)
    onsets = np.asarray(onsets_ms, dtype=np.int64)
    tolerance_ms = int(tolerance_seconds) * 1000
    if len(slots) > 1 and not np.all(np.diff(slots) > 0):
        raise ValueError("prediction slots must be strictly increasing")
    left = np.searchsorted(slots, onsets, side="left")
    right = np.searchsorted(slots, onsets + tolerance_ms, side="right")
    return np.stack([left, right], axis=1).astype(np.int64)


def assert_contiguous_slots(slot_times_ms: np.ndarray, *, grid_seconds: int = DEFAULT_GRID_SECONDS) -> None:
    """The capacity model assumes one contiguous grid per split (no interior gaps)."""

    slots = np.asarray(slot_times_ms, dtype=np.int64)
    if len(slots) > 1 and not np.all(np.diff(slots) == int(grid_seconds) * 1000):
        raise ValueError("P6-C0F capacity requires a contiguous legal slot grid inside a split")


def edf_matching(
    adjacency: np.ndarray, n_slots: int
) -> Tuple[int, Dict[int, int]]:
    """Maximum-cardinality one-to-one matching between slots and events.

    Classic earliest-deadline-first greedy over the slots in increasing order: at
    every slot, serve the still-open event with the earliest last-legal slot and
    drop events whose window has closed.  For interval windows this is optimal.
    Returns ``(matched_count, {slot_index: event_index})``.
    """

    adjacency = np.asarray(adjacency, dtype=np.int64)
    opens: List[List[Tuple[int, int]]] = [[] for _ in range(int(n_slots))]
    for event_index, (left, right) in enumerate(adjacency):
        if right > left:
            opens[int(left)].append((int(right) - 1, int(event_index)))
    heap: List[Tuple[int, int]] = []
    assignment: Dict[int, int] = {}
    for slot in range(int(n_slots)):
        for deadline, event_index in opens[slot]:
            heapq.heappush(heap, (deadline, event_index))
        while heap and heap[0][0] < slot:
            heapq.heappop(heap)
        if heap:
            _, event_index = heapq.heappop(heap)
            assignment[slot] = event_index
    return len(assignment), assignment


# ---------------------------------------------------------------------------
# R_label — label direct-decoding reference
# ---------------------------------------------------------------------------


def label_reference(
    slot_times_ms: np.ndarray,
    labels: np.ndarray,
    ground_truth: pd.DataFrame,
    *,
    split: str = "unknown",
    grid_seconds: int = DEFAULT_GRID_SECONDS,
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
) -> Mapping[str, object]:
    """Decode the frozen tri-state label into a binary sequence and evaluate it.

    POSITIVE -> 1, NEGATIVE -> 0, IGNORE -> 0.  The IGNORE rows stay on the
    timeline (they are decoded, not deleted).  The original episode construction
    and the original causal matching are reused unchanged.
    """

    slots = np.asarray(slot_times_ms, dtype=np.int64)
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(slots) != len(values):
        raise ValueError("label reference requires one label per legal prediction slot")
    binary = (values == TRIGGER_POSITIVE).astype(float)
    frame = system_score_frame(split, slots, binary)
    episodes, matching, metrics = evaluate_system_threshold(
        frame, ground_truth, 0.5, grid_seconds=grid_seconds, tolerance_seconds=tolerance_seconds
    )
    return {
        "status": "EXACT",
        "decoding_rule": "POSITIVE -> 1, NEGATIVE -> 0, IGNORE -> 0 (IGNORE rows kept on the timeline)",
        "n_slots": int(len(slots)),
        "positive_slots": int(binary.sum()),
        "positive_regions": int(_region_count(binary > 0)),
        "episode_count": int(len(episodes)),
        "event_metrics": metrics,
        "ground_truth_events": int(len(ground_truth)),
        "matching": matching,
        "episodes": episodes,
        "note": (
            "label-decoding reference only: it measures whether the frozen label decoding and the "
            "event evaluation agree.  It is NOT an upper bound on model-reachable recall."
        ),
    }


def _region_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not mask.any():
        return 0
    starts = mask.copy()
    starts[1:] &= ~mask[:-1]
    return int(starts.sum())


# ---------------------------------------------------------------------------
# U_grid — relaxed slot-level bound
# ---------------------------------------------------------------------------


def grid_bound(
    slot_times_ms: np.ndarray,
    ground_truth: pd.DataFrame,
    *,
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
    cross_check_max_slots: int = 2_000,
) -> Mapping[str, object]:
    """Maximum-cardinality causal matching over every legal prediction slot.

    The cardinality comes from the earliest-deadline-first greedy, which is exact
    for interval windows (equivalently the frozen causal matcher; cross-validated
    against exhaustive enumeration in the unit tests).  The frozen min-cost-flow
    implementation is re-run as an independent cross-check only on small
    instances, because its successive-shortest-path loop is quadratic in the
    number of prediction slots and does not scale to a full split grid.
    """

    slots = np.asarray(slot_times_ms, dtype=np.int64)
    onsets = ground_truth["start_ms"].to_numpy(dtype=np.int64)
    adjacency = adjacency_ranges(slots, onsets, tolerance_seconds=tolerance_seconds)
    count, assignment = edf_matching(adjacency, len(slots))

    cross_check: Optional[int] = None
    if len(slots) <= int(cross_check_max_slots):
        predictions = pd.DataFrame({
            "prediction_id": ["slot-{:08d}".format(index) for index in range(len(slots))],
            "t_hat": slots,
            "split": "audit",
        })
        cross_check = int(len(_min_cost_max_cardinality(
            predictions, ground_truth.reset_index(drop=True), int(tolerance_seconds) * 1000
        )))
    return {
        "status": "EXACT",
        "n_slots": int(len(slots)),
        "ground_truth_events": int(len(ground_truth)),
        "matched_events": int(count),
        "recall_upper_bound": (count / len(ground_truth)) if len(ground_truth) else None,
        "frozen_matching_matched_events": cross_check,
        "cross_check_agrees": (None if cross_check is None else bool(cross_check == count)),
        "cross_check_scope": (
            "frozen min-cost flow re-run on this instance"
            if cross_check is not None else
            "skipped on this full split grid; the greedy is validated against the frozen matcher "
            "and against exhaustive enumeration on synthetic instances in tests/test_p6_c0f_capacity.py"
        ),
        "assignment": {int(slot): int(event) for slot, event in sorted(assignment.items())},
        "relaxation": (
            "slots are matched individually; the episode-merging constraint of the real protocol is "
            "not applied, so this bound is usually not attainable by one binary sequence"
        ),
    }


# ---------------------------------------------------------------------------
# U_episode — complete-protocol bound
# ---------------------------------------------------------------------------


@dataclass
class EpisodeBoundResult:
    upper_bound_tp: int
    n_events: int
    n_slots: int
    witness_slots: List[int]
    status: str
    dp_states: int = 0
    proof: str = ""
    extras: Dict[str, object] = field(default_factory=dict)


def episode_bound(
    slot_times_ms: np.ndarray,
    onsets_ms: np.ndarray,
    *,
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
    grid_seconds: int = DEFAULT_GRID_SECONDS,
) -> EpisodeBoundResult:
    """Exact maximum TP over binary sequences under the complete protocol.

    Structure of the problem (all three steps are proved in the docstring of
    ``docs/P6_C0F_FAILURE_MECHANISM_AUDIT_PLAN.md`` section 6.4 and re-verified by
    the synthetic tests):

    1. The set of episode starts realisable by *some* binary sequence is exactly
       the set of slot subsets with no two grid-adjacent members: an episode start
       is positive while its predecessor is not (or the grid is broken), and a
       pulse-only sequence realises any such subset.
    2. For a fixed anchor set the maximum number of matched GT events is obtained
       by the earliest-deadline-first greedy (interval windows, unit capacities).
    3. The DP below enumerates every admissible anchor set while carrying exactly
       the state EDF needs: the number of still-open unserved events per deadline
       offset.  Identities are interchangeable once an event is open, so the counts
       are sufficient and the recursion is exact.
    """

    slots = np.asarray(slot_times_ms, dtype=np.int64)
    onsets = np.asarray(onsets_ms, dtype=np.int64)
    assert_contiguous_slots(slots, grid_seconds=grid_seconds)
    n_slots = int(len(slots))
    adjacency = adjacency_ranges(slots, onsets, tolerance_seconds=tolerance_seconds)
    # the DP needs window lengths in slot indices; windows longer than one grid
    # step plus the tolerance would break the "deadline offset <= 2" bookkeeping.
    offsets = adjacency[:, 1] - adjacency[:, 0] - 1
    if len(offsets) and offsets.max() > 2:
        raise ValueError(
            "episode bound assumes windows of at most three legal slots "
            "(60 s tolerance on a 30 s grid); observed {}".format(int(offsets.max()))
        )

    opens: List[List[int]] = [[] for _ in range(n_slots + 1)]
    for event_index, (left, right) in enumerate(adjacency):
        if right > left:
            opens[int(left)].append(int(right) - 1 - int(left))  # deadline offset 0/1/2

    def _openings(slot: int) -> Tuple[int, int, int]:
        counts = [0, 0, 0]
        for offset in opens[slot]:
            counts[offset] += 1
        return counts[0], counts[1], counts[2]

    # precompute the per-slot opening counts: the inner DP loop must not rescan them
    opening_counts = [_openings(slot) for slot in range(n_slots + 1)]
    tail_openings = opening_counts[-1]

    # state at the start of a slot = (previous_slot_selected, expiring_now, expiring_next,
    # expiring_next2); the counts already include the events that open at this slot.
    initial_openings = opening_counts[0]
    initial_state = (0, initial_openings[0], initial_openings[1], initial_openings[2])
    frontier: Dict[Tuple[int, int, int, int], int] = {initial_state: 0}
    traces: List[Dict[Tuple[int, int, int, int], Tuple[Tuple[int, int, int, int], bool]]] = []
    expanded = 0
    for slot in range(n_slots):
        open0, open1, open2 = opening_counts[slot + 1] if slot + 1 < n_slots else tail_openings
        next_frontier: Dict[Tuple[int, int, int, int], int] = {}
        trace: Dict[Tuple[int, int, int, int], Tuple[Tuple[int, int, int, int], bool]] = {}
        for state, value in frontier.items():
            previous_selected, now, nxt, nxt2 = state
            expanded += 1
            # option A: do not select this slot; events expiring now are dropped
            key = (0, nxt + open0, nxt2 + open1, open2)
            candidate = value
            if key not in next_frontier or next_frontier[key] < candidate:
                next_frontier[key] = candidate
                trace[key] = (state, False)
            if previous_selected:
                continue
            # option B: select this slot and serve the earliest-deadline open event
            if now > 0:
                remaining = nxt, nxt2
            elif nxt > 0:
                remaining = nxt - 1, nxt2
            elif nxt2 > 0:
                remaining = nxt, nxt2 - 1
            else:
                continue  # selecting without serving is dominated by option A
            key = (1, remaining[0] + open0, remaining[1] + open1, open2)
            candidate = value + 1
            if key not in next_frontier or next_frontier[key] < candidate:
                next_frontier[key] = candidate
                trace[key] = (state, True)
        traces.append(trace)
        frontier = next_frontier
        if not frontier:
            raise RuntimeError("episode bound DP lost all states")

    best_state = max(frontier.items(), key=lambda item: item[1])[0]
    best_value = int(frontier[best_state])

    witness: List[int] = []
    state: Optional[Tuple[int, int, int, int]] = best_state
    for slot in range(len(traces) - 1, -1, -1):
        if state is None:
            raise RuntimeError("episode bound backtracking lost a state")
        previous_state, selected = traces[slot][state]
        if selected:
            witness.append(slot)
        state = previous_state
    witness = sorted(witness)
    return EpisodeBoundResult(
        upper_bound_tp=best_value,
        n_events=int(len(onsets)),
        n_slots=n_slots,
        witness_slots=[int(slots[index]) for index in witness],
        status="EXACT",
        dp_states=expanded,
        proof=(
            "episode starts form a stable set of the slot path; pulse-only sequences realise every "
            "stable set; EDF is optimal for a fixed anchor set; the DP enumerates all stable sets "
            "with the exact EDF state"
        ),
    )


# ---------------------------------------------------------------------------
# witness replay
# ---------------------------------------------------------------------------


def replay_witness(
    witness_slots_ms: Sequence[int],
    slot_times_ms: np.ndarray,
    ground_truth: pd.DataFrame,
    *,
    split: str = "unknown",
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
    grid_seconds: int = DEFAULT_GRID_SECONDS,
) -> Mapping[str, object]:
    """Replay a witness anchor set through the frozen episode/matching pipeline."""

    slots = np.asarray(slot_times_ms, dtype=np.int64)
    anchor_set = set(int(value) for value in witness_slots_ms)
    binary = np.asarray([1.0 if int(slot) in anchor_set else 0.0 for slot in slots], dtype=float)
    frame = system_score_frame(split, slots, binary)
    episodes, matching, metrics = evaluate_system_threshold(
        frame, ground_truth, 0.5, grid_seconds=grid_seconds, tolerance_seconds=tolerance_seconds
    )
    realised = int(metrics["true_positive_events"])
    return {
        "witness_anchors": int(len(anchor_set)),
        "episode_count": int(len(episodes)),
        "episode_anchors_match_witness": bool(
            sorted(int(value) for value in episodes["t_hat"].tolist()) == sorted(anchor_set)
        ),
        "realised_tp": realised,
        "event_metrics": metrics,
        "episodes": episodes,
        "matching": matching,
    }
