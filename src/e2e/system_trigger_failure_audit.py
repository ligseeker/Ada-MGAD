"""P6-C0F failure accounting, score trajectories and confounding analysis.

Pure read-only analysis helpers for the C0F audit.  They consume the frozen
checkpoint's scores (or the saved Test scores), the frozen GT registry and the
frozen episode/matching artifacts.  Nothing here trains, selects, thresholds or
writes; the audit driver owns all I/O.

Terminology follows ``docs/P6_C0F_FAILURE_MECHANISM_AUDIT_PLAN.md``:

* a **slot** is a legal prediction timestamp of a split (``prediction_available_time``
  of a window that exists in that split);
* the **failure ledger** assigns every complete GT event exactly one location code;
* a **trajectory** is the score/logit path of one event around its onset, with
  other-event concurrency markers;
* **collision** statistics describe several onsets sharing a 30 s bin.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .system_trigger import (
    DEFAULT_GRID_SECONDS,
    DEFAULT_TOLERANCE_SECONDS,
    DURATION_STRATUM_LABELS,
    SMALL_N_THRESHOLD,
    duration_stratum,
    onset_density,
)
from .system_trigger_capacity import adjacency_ranges


FAILURE_CATEGORIES = (
    "MATCHED",
    "NO_LEGAL_PREDICTION",
    "BELOW_THRESHOLD",
    "NO_NEW_EPISODE",
    "MATCHING_COMPETITION",
)

WINDOW_TAGS = ("pre_onset", "response", "early", "mid", "event_rest")

RESPONSE_BANDS = (
    "le_60s", "60_120s", "120_300s", "gt_300s", "no_grid_observation", "never", "censored",
)

LEDGER_COLUMNS = (
    "case_id", "split", "onset_ms", "end_ms", "duration_seconds", "fault_type", "service",
    "onset_bin", "onset_on_grid_boundary", "failure_category", "failure_basis",
    "eligible_slots", "positive_slots", "candidate_episode_ids", "candidate_episode_count",
    "matched_prediction_id", "official_matched_time", "detection_delay_seconds",
    "first_positive_time", "first_positive_delta_ms", "first_causal_positive_time",
    "first_new_episode_time", "episode_active_before_onset", "observation_end_ms",
    "trajectory_end_ms", "response_band", "episode_band",
    "censored_by_split_end", "censored_by_score_coverage",
    "other_onset_count_60s", "other_active_event_count", "in_other_recent_onset_window",
    "duration_stratum", "onset_bin_count", "onset_bin_root_service_count",
)


def _slot_lookup(slots: np.ndarray) -> Mapping[int, int]:
    return {int(value): index for index, value in enumerate(slots)}


def build_failure_ledger(
    case_frame: pd.DataFrame,
    *,
    split: str,
    slot_times_ms: np.ndarray,
    slot_scores: np.ndarray,
    threshold: float,
    episode_anchors_ms: np.ndarray,
    episode_end_times_ms: np.ndarray,
    matching: pd.DataFrame,
    origin_ms: int,
    grid_seconds: int = DEFAULT_GRID_SECONDS,
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    """Assign every complete GT event of one split to exactly one location code.

    ``MATCHED`` / ``NO_LEGAL_PREDICTION`` / ``BELOW_THRESHOLD`` / ``NO_NEW_EPISODE`` /
    ``MATCHING_COMPETITION``.  The official global episode construction and the
    official global matching are used as they are; events are never re-matched
    individually.
    """

    slots = np.asarray(slot_times_ms, dtype=np.int64)
    scores = np.asarray(slot_scores, dtype=float)
    if len(slots) != len(scores):
        raise ValueError("every legal slot needs exactly one score")
    anchors = np.sort(np.asarray(episode_anchors_ms, dtype=np.int64))
    anchor_ends = np.asarray(episode_end_times_ms, dtype=np.int64)
    if len(anchor_ends) != len(np.asarray(episode_anchors_ms)):
        raise ValueError("episode anchors and end times must be aligned")
    order = np.argsort(np.asarray(episode_anchors_ms, dtype=np.int64))
    anchor_end_by_start = {
        int(np.asarray(episode_anchors_ms)[index]): int(anchor_ends[index]) for index in order
    }
    tolerance_ms = int(tolerance_seconds) * 1000

    events = case_frame.reset_index(drop=True).copy()
    events["case_id"] = events["case_id"].astype(str)
    onsets = events["start_ms"].to_numpy(dtype=np.int64)
    ends = events["end_ms"].to_numpy(dtype=np.int64)
    adjacency = adjacency_ranges(slots, onsets, tolerance_seconds=tolerance_seconds)

    matching_status = dict(zip(matching["case_id"].astype(str), matching["match_status"].astype(str)))
    matching_prediction = dict(zip(matching["case_id"].astype(str), matching["prediction_id"]))
    matching_time = dict(zip(matching["case_id"].astype(str), matching["t_hat"]))
    matching_delay = dict(zip(matching["case_id"].astype(str), matching["detection_delay_seconds"]))

    density = onset_density(events, origin_ms=origin_ms, grid_seconds=grid_seconds)
    bin_counts = dict(zip(density["onset_bin"].astype(int), density["onset_count"].astype(int)))
    bin_services = dict(zip(density["onset_bin"].astype(int), density["root_service_count"].astype(int)))

    score_coverage_end = int(slots[-1]) if len(slots) else None
    rows: List[Dict[str, object]] = []
    categories: Dict[str, int] = {name: 0 for name in FAILURE_CATEGORIES}
    for index in range(len(events)):
        onset = int(onsets[index])
        end = int(ends[index])
        event_id = str(events.at[index, "case_id"])
        left, right = int(adjacency[index][0]), int(adjacency[index][1])
        eligible = right - left
        eligible_slot_times = slots[left:right]
        positive_mask = scores[left:right] >= float(threshold) if eligible else np.zeros(0, dtype=bool)
        positive_slots = int(positive_mask.sum())
        candidates = anchors[(anchors >= onset) & (anchors <= onset + tolerance_ms)]
        candidate_ids = ["anchor-{}".format(int(value)) for value in candidates]

        status = matching_status.get(event_id, "absent")
        if status == "matched":
            category = "MATCHED"
            basis = "official causal matching"
        elif eligible == 0:
            category = "NO_LEGAL_PREDICTION"
            basis = "no legal prediction timestamp inside [onset, onset+tolerance]"
        elif positive_slots == 0:
            category = "BELOW_THRESHOLD"
            basis = "eligible timestamps exist but none reached the frozen threshold"
        elif len(candidates) == 0:
            category = "NO_NEW_EPISODE"
            basis = "positive timestamps exist but none starts a new episode inside the window"
        else:
            category = "MATCHING_COMPETITION"
            basis = "an episode starts inside the window but the global matching assigned it elsewhere"
        categories[category] += 1

        first_causal_positive = None
        if positive_slots:
            first_causal_positive = int(eligible_slot_times[np.argmax(positive_mask)])
        # response tracking spans the trajectory range: onset .. max(onset+300 s, observable end)
        coverage_end = score_coverage_end if score_coverage_end is not None else end
        trajectory_end = max(onset + 300_000, min(end, coverage_end))
        after_left = int(np.searchsorted(slots, onset, side="left"))
        after_right = int(np.searchsorted(slots, trajectory_end, side="right"))
        after_slots = slots[after_left:after_right]
        after_scores = scores[after_left:after_right]
        observed_positive = after_scores >= float(threshold) if len(after_slots) else np.zeros(0, dtype=bool)
        first_positive = int(after_slots[int(np.argmax(observed_positive))]) if observed_positive.any() else None
        anchors_after = anchors[(anchors >= onset) & (anchors <= trajectory_end)]
        first_episode = int(anchors_after[0]) if len(anchors_after) else None
        censored = bool(end > coverage_end)
        has_observation = bool(len(after_slots))
        active_before = any(
            int(start) <= onset < int(anchor_end_by_start.get(int(start), int(start)))
            for start in anchors if int(start) <= onset
        )
        observation_end = min(end, coverage_end)
        bin_index = int((onset - int(origin_ms)) // (int(grid_seconds) * 1000))
        rows.append({
            "case_id": event_id,
            "split": str(split),
            "onset_ms": onset,
            "end_ms": end,
            "duration_seconds": (end - onset) / 1000.0,
            "fault_type": str(events.at[index, "fault_type"]),
            "service": str(events.at[index, "service"]),
            "onset_bin": bin_index,
            "onset_on_grid_boundary": bool((onset - int(origin_ms)) % (int(grid_seconds) * 1000) == 0),
            "failure_category": category,
            "failure_basis": basis,
            "eligible_slots": int(eligible),
            "positive_slots": positive_slots,
            "candidate_episode_ids": ";".join(candidate_ids),
            "candidate_episode_count": int(len(candidates)),
            "matched_prediction_id": matching_prediction.get(event_id),
            "official_matched_time": int(matching_time[event_id]) if status == "matched" else None,
            "detection_delay_seconds": matching_delay.get(event_id),
            "first_positive_time": first_positive,
            "first_positive_delta_ms": None if first_positive is None else int(first_positive) - onset,
            "first_causal_positive_time": first_causal_positive,
            "first_new_episode_time": first_episode,
            "episode_active_before_onset": bool(active_before),
            "observation_end_ms": int(observation_end),
            "trajectory_end_ms": int(trajectory_end),
            "response_band": _response_band(first_positive, onset, has_observation, censored),
            "episode_band": _response_band(first_episode, onset, has_observation, censored),
            "censored_by_split_end": bool(end > coverage_end),
            "censored_by_score_coverage": bool(score_coverage_end is not None and end > score_coverage_end),
            "other_onset_count_60s": _other_onset_count(onsets, onset, index),
            "other_active_event_count": _active_count(onsets, ends, onset, index),
            "in_other_recent_onset_window": _other_onset_count(onsets, onset, index) > 0,
            "duration_stratum": str(duration_stratum([(end - onset) / 1000.0])[0]),
            "onset_bin_count": int(bin_counts.get(bin_index, 0)),
            "onset_bin_root_service_count": int(bin_services.get(bin_index, 0)),
        })

    ledger = pd.DataFrame(rows, columns=list(LEDGER_COLUMNS))
    total = int(len(ledger))
    summary = {
        "split": str(split),
        "complete_gt_events": total,
        "category_counts": categories,
        "category_fractions": {name: (value / total if total else None) for name, value in categories.items()},
        "categories_sum_to_total": bool(sum(categories.values()) == total),
        "matched_equals_tp": bool(
            categories["MATCHED"] == int((matching["match_status"].astype(str) == "matched").sum())
        ),
        "unmatched_events_equal_fn": bool(
            total - categories["MATCHED"] == int((matching["match_status"].astype(str) == "miss").sum())
        ),
        "unmatched_episodes_equal_fp": bool(
            int(len(anchors)) - categories["MATCHED"]
            == int((matching["match_status"].astype(str) == "false_alarm").sum())
        ),
        "definition": (
            "every complete GT event gets exactly one location code; the codes are occurrence "
            "locations, not a full causal explanation"
        ),
    }
    return ledger, summary


def _response_band(
    first_response: Optional[int], onset: int, has_observation: bool, censored: bool
) -> str:
    """Band of the first response after the onset, or why there is none.

    ``first_response`` is the first threshold crossing (or first new episode) at
    or after the onset inside the trajectory range.  A short event whose own
    interval contains no grid point is reported as ``no_grid_observation`` rather
    than as a zero score, and an incomplete observation interval is ``censored``
    rather than ``never``.
    """

    if first_response is not None:
        delta = int(first_response) - int(onset)
        if delta <= 60_000:
            return "le_60s"
        if delta <= 120_000:
            return "60_120s"
        if delta <= 300_000:
            return "120_300s"
        return "gt_300s"
    if not has_observation:
        return "no_grid_observation"
    if censored:
        return "censored"
    return "never"


def _other_onset_count(onsets: np.ndarray, onset: int, self_index: int) -> int:
    left = int(np.searchsorted(onsets, onset - 60_000, side="left"))
    right = int(np.searchsorted(onsets, onset, side="right"))
    count = right - left
    if left <= self_index < right:
        count -= 1
    return int(count)


def _active_count(onsets: np.ndarray, ends: np.ndarray, onset: int, self_index: int) -> int:
    count = 0
    for index in range(len(onsets)):
        if index == self_index:
            continue
        if int(onsets[index]) <= onset < int(ends[index]):
            count += 1
    return int(count)


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------


def build_trajectories(
    ledger: pd.DataFrame,
    *,
    split: str,
    slot_times_ms: np.ndarray,
    slot_scores: np.ndarray,
    slot_logits: np.ndarray,
    threshold: float,
    leading_seconds: int = 300,
    trailing_seconds: int = 300,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-event score path plus a per-event, per-window response summary."""

    slots = np.asarray(slot_times_ms, dtype=np.int64)
    scores = np.asarray(slot_scores, dtype=float)
    logits = np.asarray(slot_logits, dtype=float)
    if not (len(slots) == len(scores) == len(logits)):
        raise ValueError("slots, scores and logits must be aligned")
    onsets_all = ledger["onset_ms"].to_numpy(dtype=np.int64)
    ends_all = ledger["end_ms"].to_numpy(dtype=np.int64)
    score_coverage_end = int(slots[-1]) if len(slots) else 0

    rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for index, event in enumerate(ledger.itertuples(index=False)):
        onset = int(event.onset_ms)
        end = int(event.end_ms)
        left = int(np.searchsorted(slots, onset - leading_seconds * 1000, side="left"))
        right = int(np.searchsorted(slots, onset + trailing_seconds * 1000, side="right"))
        event_right = int(np.searchsorted(slots, end, side="left"))
        right = max(right, event_right)
        deltas = slots[left:right] - onset
        other_onsets = _other_onset_vector(onsets_all, slots[left:right], index)
        other_active = _active_vector(onsets_all, ends_all, slots[left:right], index)
        for position, delta in enumerate(deltas):
            slot_index = left + position
            rows.append({
                "case_id": event.case_id,
                "split": str(split),
                "onset_ms": onset,
                "delta_ms": int(delta),
                "prediction_available_time": int(slots[slot_index]),
                "score": float(scores[slot_index]),
                "logit": float(logits[slot_index]),
                "binary_prediction": int(scores[slot_index] >= float(threshold)),
                "window_tag": _window_tag(int(delta), end - onset),
                "other_onset_count_60s": int(other_onsets[position]),
                "other_active_event_count": int(other_active[position]),
                "in_other_recent_onset_window": bool(other_onsets[position] > 0),
            })
        summary_rows.append(_event_window_summary(
            event, split=split, slots=slots, scores=scores, logits=logits,
            threshold=float(threshold), onset=onset, end=end, coverage_end=score_coverage_end,
        ))
    trajectories = pd.DataFrame(rows, columns=(
        "case_id", "split", "onset_ms", "delta_ms", "prediction_available_time",
        "score", "logit", "binary_prediction", "window_tag",
        "other_onset_count_60s", "other_active_event_count", "in_other_recent_onset_window",
    ))
    summary = pd.DataFrame(summary_rows)
    return trajectories, summary


def _window_tag(delta_ms: int, duration_ms: int) -> str:
    if delta_ms < 0:
        return "pre_onset"
    if delta_ms <= 60_000:
        return "response"
    if delta_ms <= 120_000:
        return "early"
    if delta_ms <= 300_000:
        return "mid"
    return "event_rest"


def _other_onset_vector(onsets: np.ndarray, slot_times: np.ndarray, self_index: int) -> np.ndarray:
    left = np.searchsorted(onsets, slot_times - 60_000, side="left")
    right = np.searchsorted(onsets, slot_times, side="right")
    counts = right - left
    self_onset = int(onsets[self_index])
    inside = (slot_times - 60_000 <= self_onset) & (self_onset <= slot_times)
    return counts - inside.astype(np.int64)


def _active_vector(onsets: np.ndarray, ends: np.ndarray, slot_times: np.ndarray, self_index: int) -> np.ndarray:
    started = np.searchsorted(onsets, slot_times, side="right")
    finished = np.searchsorted(ends, slot_times, side="right")
    counts = started - finished
    self_active = (int(onsets[self_index]) <= slot_times) & (slot_times < int(ends[self_index]))
    return counts - self_active.astype(np.int64)


def _event_window_summary(event, *, split, slots, scores, logits, threshold, onset, end, coverage_end):
    duration_ms = end - onset
    observation_end = min(end, coverage_end)
    response = {
        "case_id": event.case_id,
        "split": str(split),
        "onset_ms": int(onset),
        "end_ms": int(end),
        "duration_seconds": duration_ms / 1000.0,
        "duration_stratum": str(duration_stratum([duration_ms / 1000.0])[0]),
        "failure_category": str(event.failure_category),
        "response_band": str(event.response_band),
        "first_positive_time": event.first_positive_time,
        "first_new_episode_time": event.first_new_episode_time,
        "official_matched_time": event.official_matched_time,
        "observation_end_ms": int(observation_end),
        "censored_by_split_end": bool(event.censored_by_split_end),
        "censored_by_score_coverage": bool(event.censored_by_score_coverage),
        "other_onset_count_60s": int(event.other_onset_count_60s),
        "other_active_event_count": int(event.other_active_event_count),
        "in_other_recent_onset_window": bool(event.in_other_recent_onset_window),
        "onset_bin_count": int(event.onset_bin_count),
    }
    for tag in WINDOW_TAGS:
        if tag == "pre_onset":
            mask = (slots >= onset - 300_000) & (slots < onset)
        elif tag == "response":
            mask = (slots >= onset) & (slots <= onset + 60_000)
        elif tag == "early":
            mask = (slots > onset + 60_000) & (slots <= onset + 120_000)
        elif tag == "mid":
            mask = (slots > onset + 120_000) & (slots <= onset + 300_000)
        else:
            mask = (slots >= onset) & (slots < end)
        values = scores[mask]
        logit_values = logits[mask]
        response["{}_count".format(tag)] = int(mask.sum())
        response["{}_max_score".format(tag)] = float(values.max()) if len(values) else None
        response["{}_median_score".format(tag)] = float(np.median(values)) if len(values) else None
        response["{}_max_logit".format(tag)] = float(logit_values.max()) if len(logit_values) else None
        response["{}_positive".format(tag)] = int((values >= float(threshold)).sum()) if len(values) else 0
        response["{}_censored".format(tag)] = bool(tag != "pre_onset" and end > coverage_end and mask.any())
    return response


# ---------------------------------------------------------------------------
# stratified and confounding tables
# ---------------------------------------------------------------------------


def stratified_summary(ledger: pd.DataFrame, *, split: str) -> Mapping[str, object]:
    """Recall and failure-category mix by duration, fault, service and collision."""

    def _group(column: str, order: Optional[Sequence[str]] = None):
        rows = []
        values = order if order is not None else sorted(ledger[column].astype(str).unique())
        for value in values:
            group = ledger.loc[ledger[column].astype(str) == str(value)]
            if group.empty and order is None:
                continue
            total = int(len(group))
            matched = int((group["failure_category"] == "MATCHED").sum())
            rows.append({
                column: str(value),
                "n": total,
                "matched": matched,
                "fn": total - matched,
                "recall": (matched / total) if total else None,
                "small_n": bool(total < SMALL_N_THRESHOLD),
                "failure_mix": {
                    name: int((group["failure_category"] == name).sum()) for name in FAILURE_CATEGORIES
                },
            })
        return rows

    return {
        "split": str(split),
        "by_duration_stratum": _group("duration_stratum", order=list(DURATION_STRATUM_LABELS)),
        "by_fault_type": _group("fault_type"),
        "by_service": _group("service"),
        "by_onset_bin_count": _group("onset_bin_count"),
        "by_response_band": _group("response_band"),
        "response_band_totals": {
            band: int((ledger["response_band"].astype(str) == band).sum()) for band in RESPONSE_BANDS
        },
        "definition": (
            "recall and failure-category mix per stratum; empty strata keep n=0 with a null recall; "
            "categories are occurrence locations, not causal explanations"
        ),
    }


def confounding_tables(events: pd.DataFrame, *, origin_ms: int, grid_seconds: int = DEFAULT_GRID_SECONDS) -> Mapping[str, object]:
    """fault x duration x service cross counts with explicit empty cells."""

    frame = events.copy()
    frame["duration_stratum"] = duration_stratum(
        (frame["end_ms"] - frame["start_ms"]).to_numpy(dtype=float) / 1000.0
    )
    fault_duration = (
        frame.groupby(["fault_type", "duration_stratum"], dropna=False).size().rename("n").reset_index()
    )
    fault_service = frame.groupby(["fault_type", "service"], dropna=False).size().rename("n").reset_index()
    service_duration = (
        frame.groupby(["service", "duration_stratum"], dropna=False).size().rename("n").reset_index()
    )
    triple = (
        frame.groupby(["fault_type", "duration_stratum", "service"], dropna=False)
        .size().rename("n").reset_index()
    )
    return {
        "fault_x_duration": fault_duration.to_dict(orient="records"),
        "fault_x_service": fault_service.to_dict(orient="records"),
        "service_x_duration": service_duration.to_dict(orient="records"),
        "fault_x_duration_x_service": triple.to_dict(orient="records"),
        "empty_cell_policy": "only observed combinations are listed; the full cartesian product was checked to be empty elsewhere",
        "cartesian_product_size": int(
            frame["fault_type"].nunique() * len(DURATION_STRATUM_LABELS) * frame["service"].nunique()
        ),
        "observed_combination_count": int(len(triple)),
    }


def collision_audit(events: pd.DataFrame, *, population_name: str, origin_ms: int, grid_seconds: int = DEFAULT_GRID_SECONDS) -> Mapping[str, object]:
    """Onset collisions: bin ids, exact grid-boundary onsets and bin multiplicities."""

    if events.empty:
        return {
            "population": str(population_name), "events": 0, "onsets_on_grid_boundary": 0,
            "distinct_onset_bins": 0, "onsets_per_bin_histogram": {},
            "bins_with_exactly_two_onsets": 0, "bins_with_exactly_three_onsets": 0,
            "events_in_two_onset_bins": 0, "events_in_three_onset_bins": 0,
            "events_in_multi_onset_bins": 0,
        }
    frame = events.copy()
    grid_ms = int(grid_seconds) * 1000
    onsets = frame["start_ms"].to_numpy(dtype=np.int64)
    bins = np.floor_divide(onsets - int(origin_ms), grid_ms)
    frame["onset_bin"] = bins
    frame["on_grid_boundary"] = ((onsets - int(origin_ms)) % grid_ms) == 0
    counts = frame.groupby("onset_bin").size()
    histogram = {str(int(key)): int(value) for key, value in counts.value_counts().sort_index().items()}
    by_bin_count = {int(key): int(value) for key, value in counts.value_counts().sort_index().items()}
    frame["bin_count"] = frame["onset_bin"].map(counts)
    return {
        "population": str(population_name),
        "events": int(len(frame)),
        "onsets_on_grid_boundary": int(frame["on_grid_boundary"].sum()),
        "distinct_onset_bins": int(counts.size),
        "onsets_per_bin_histogram": histogram,
        "bins_with_exactly_two_onsets": int(by_bin_count.get(2, 0)),
        "bins_with_exactly_three_onsets": int(by_bin_count.get(3, 0)),
        "events_in_two_onset_bins": int((frame["bin_count"] == 2).sum()),
        "events_in_three_onset_bins": int((frame["bin_count"] == 3).sum()),
        "events_in_multi_onset_bins": int((frame["bin_count"] >= 2).sum()),
        "definition": (
            "onset bin = floor((onset_ms - grid_origin) / 30 s); collisions are bins with more than "
            "one onset, independent of whether the onsets belong to the same fault or service"
        ),
    }
