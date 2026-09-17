"""P6-C0 root-agnostic system event trigger: labels, split, purge, selection.

These helpers are P6-C0-only.  They *re-slice* the frozen GAIA V2 Ada-MGAD
arrays and build a **system-level** (recent-onset) trigger label on top of the
frozen GT event registry.  They never:

* re-preprocess or modify raw telemetry,
* read a node anomaly label as a supervision target,
* read the labelled root service as a feature or a target,
* change the frozen 30 s grid, 300 s input history, 60 s causal tolerance or the
  causal max-cardinality / minimum-delay event matching,
* write into a frozen P5/P6 artifact directory.

Split protocol (P6-C0, chronological only):

```text
first 50%   Detector-Fit          gradient updates + Fit-only statistics
next  20%   Detector-Validation   checkpoint selection + threshold selection
last  30%   Final Test            observation only, frozen once
```

The Test block is bit-identical to the frozen P5 Test block, so the final Test
population is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .event_detection import (
    _evaluate_threshold_candidates,
    construct_predicted_episodes,
    event_metrics,
    match_events,
)
from .protocol import TemporalBlock, assign_event_blocks


TRIGGER_NEGATIVE = 0
TRIGGER_POSITIVE = 1
TRIGGER_IGNORE = 2
TRIGGER_LABEL_VALUES = (TRIGGER_NEGATIVE, TRIGGER_POSITIVE, TRIGGER_IGNORE)
TRIGGER_LABEL_NAMES = {
    TRIGGER_NEGATIVE: "negative",
    TRIGGER_POSITIVE: "positive",
    TRIGGER_IGNORE: "ignore",
}

SPLIT_NAMES = ("fit", "validation", "test")
FIT_NUMERATOR = 5
FIT_DENOMINATOR = 10
CUMULATIVE_VALIDATION_NUMERATOR = 7
CUMULATIVE_VALIDATION_DENOMINATOR = 10

DEFAULT_GRID_SECONDS = 30
DEFAULT_WINDOW_BINS = 10
DEFAULT_HISTORY_SECONDS = 300
DEFAULT_POSITIVE_WINDOW_SECONDS = 60
DEFAULT_TOLERANCE_SECONDS = 60

DURATION_STRATUM_EDGES = (15.0, 30.0, 60.0, 300.0)
DURATION_STRATUM_LABELS = ("le_15s", "15_30s", "30_60s", "60_300s", "gt_300s")

SMALL_N_THRESHOLD = 30


# ---------------------------------------------------------------------------
# split geometry
# ---------------------------------------------------------------------------


def trigger_temporal_blocks(config: Mapping[str, object]) -> Tuple[TemporalBlock, ...]:
    """Derive the 50/20/30 chronological blocks from the frozen timeline.

    The rule is the same deterministic rule the frozen 70/30 split uses
    (``K = (numerator * N) // denominator`` on the 30 s metric grid), applied at
    50% and 70%.  The Test block start/end must equal the frozen boundary, so the
    final Test population is untouched.
    """

    split = config["split"]
    grid_ms = int(config["ad"]["grid_seconds"]) * 1000
    start = int(split["absolute_start_ms"])
    end = int(split["absolute_end_ms"])
    duration = end - start
    if duration <= 0 or duration % grid_ms:
        raise ValueError("metric detector timeline must contain whole 30-second bins")
    n_bins = duration // grid_ms
    fit_bins = (FIT_NUMERATOR * n_bins) // FIT_DENOMINATOR
    validation_bins = (CUMULATIVE_VALIDATION_NUMERATOR * n_bins) // CUMULATIVE_VALIDATION_DENOMINATOR
    if not 0 < fit_bins < validation_bins < n_bins:
        raise ValueError("50/20/30 split must strictly partition the metric timeline")
    fit_end = start + fit_bins * grid_ms
    validation_end = start + validation_bins * grid_ms
    declared = split.get("boundary_ms")
    if declared is not None and int(declared) != validation_end:
        raise ValueError("P6-C0 validation/Test boundary must equal the frozen 70/30 boundary")
    return (
        TemporalBlock("fit", start, fit_end),
        TemporalBlock("validation", fit_end, validation_end),
        TemporalBlock("test", validation_end, end, is_final=True),
    )


def block_bounds(blocks: Sequence[TemporalBlock]) -> Mapping[str, Mapping[str, int]]:
    return {
        block.name: {"start_ms": int(block.start_ms), "end_ms": int(block.end_ms)}
        for block in blocks
    }


def window_split_assignment(
    timestamps_ms: np.ndarray,
    blocks: Sequence[TemporalBlock],
    *,
    grid_seconds: int = DEFAULT_GRID_SECONDS,
    window_bins: int = DEFAULT_WINDOW_BINS,
) -> pd.DataFrame:
    """Assign every sliding window to a split with a >=history purge at boundaries.

    A window is identified by its sample index and by its prediction time
    ``t = prediction_available_time = target_bin_end``.  Its input interval is
    ``[t - window_bins*grid, t)``.  The window is kept only when that whole
    interval is inside the block that owns ``t``; otherwise it is purged.  With
    the frozen 300 s history this removes exactly ``window_bins`` windows at each
    internal boundary, which is the same purge the frozen preprocessing applied
    to the original Train/Test boundary.
    """

    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    grid_ms = int(grid_seconds) * 1000
    window_bins = int(window_bins)
    history_ms = window_bins * grid_ms
    starts = np.array([int(block.start_ms) for block in blocks], dtype=np.int64)
    names = [block.name for block in blocks]
    absolute_end = int(blocks[-1].end_ms)
    sample_count = max(0, len(timestamps) - window_bins + 1)
    rows: List[Dict[str, object]] = []
    for index in range(sample_count):
        target_index = index + window_bins - 1
        prediction_time = int(timestamps[target_index]) + grid_ms
        window_start = int(timestamps[index])
        owner = int(np.searchsorted(starts, prediction_time, side="right")) - 1
        if owner < 0 or prediction_time > absolute_end:
            split = "outside"
            keep = False
            reason = "prediction_time_outside_blocks"
        else:
            block = blocks[owner]
            split = names[owner]
            keep = window_start >= int(block.start_ms) and prediction_time <= int(block.end_ms)
            reason = "" if keep else "input_history_crosses_split_boundary"
        rows.append({
            "sample_index": int(index),
            "split": split,
            "window_start_time": window_start,
            "window_end_time": prediction_time,
            "target_bin_start": prediction_time - grid_ms,
            "target_bin_end": prediction_time,
            "prediction_available_time": prediction_time,
            "keep": bool(keep),
            "purge_reason": reason,
            "_history_ms": history_ms,
        })
    frame = pd.DataFrame(rows, columns=[
        "sample_index", "split", "window_start_time", "window_end_time",
        "target_bin_start", "target_bin_end", "prediction_available_time",
        "keep", "purge_reason",
    ] + ["_history_ms"])
    return frame


# ---------------------------------------------------------------------------
# trigger label
# ---------------------------------------------------------------------------


def build_trigger_labels(
    prediction_times_ms: np.ndarray,
    events: pd.DataFrame,
    *,
    positive_window_seconds: int = DEFAULT_POSITIVE_WINDOW_SECONDS,
) -> np.ndarray:
    """Rasterize the recent-onset trigger target, one tri-state value per time.

    ``prediction_times_ms`` must be the **prediction-time** grid, i.e. the array
    of ``prediction_available_time = target_bin_end`` values of the sliding
    windows.  For a prediction time ``t`` and a legal GT event onset ``t_start``:

    ```text
    POSITIVE : exists an event with 0 <= t - t_start <= positive_window
    IGNORE   : t_start + positive_window < t < t_end for some event, and no
               POSITIVE condition holds at t   (that event is already > 60 s old
               but is still ongoing; do not supervise it as NEGATIVE)
    NEGATIVE : neither of the above
    ```

    Indexing this array with the window's target bin index therefore yields the
    label of the state at that window's ``prediction_available_time``.  Passing
    the bin-start grid instead would silently lag the supervision target by one
    bin; ``assert_prediction_time_grid`` guards that contract.

    The label is system-level: several events, several root services or overlapping
    events in the same bin collapse into a single POSITIVE.  No service-specific
    target is produced, and an 11 s event and a 3600 s event contribute the same
    (bounded) number of positive supervision bins.
    """

    timestamps = np.asarray(prediction_times_ms, dtype=np.int64)
    labels = np.full(len(timestamps), TRIGGER_NEGATIVE, dtype=np.int8)
    if len(timestamps) == 0:
        return labels
    if isinstance(events, pd.DataFrame):
        missing = {"start_ms", "end_ms"} - set(events.columns)
        if missing:
            raise ValueError("trigger label events are missing columns: {}".format(sorted(missing)))
        starts = events["start_ms"].to_numpy(dtype=np.int64)
        ends = events["end_ms"].to_numpy(dtype=np.int64)
    else:
        array = np.asarray(events, dtype=np.int64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("trigger label events must be a (n, 2) interval array")
        starts, ends = array[:, 0], array[:, 1]
    if len(starts) and np.any(ends <= starts):
        raise ValueError("trigger label intervals must be positive half-open spans")
    window_ms = int(positive_window_seconds) * 1000
    # IGNORE first so that POSITIVE always wins where the two overlap.
    for start, end in zip(starts, ends):
        left = int(np.searchsorted(timestamps, int(start) + window_ms, side="right"))
        right = int(np.searchsorted(timestamps, int(end), side="left"))
        if right > left:
            labels[left:right] = TRIGGER_IGNORE
    for start in starts:
        left = int(np.searchsorted(timestamps, int(start), side="left"))
        right = int(np.searchsorted(timestamps, int(start) + window_ms, side="right"))
        if right > left:
            labels[left:right] = TRIGGER_POSITIVE
    return labels


def prediction_time_grid(timestamps_ms: np.ndarray, *, grid_seconds: int = DEFAULT_GRID_SECONDS) -> np.ndarray:
    """Map a bin-start grid to the prediction-time grid ``target_bin_end``.

    ``prediction_available_time = target_bin_start + grid``, so the trigger label
    must be rasterized on ``timestamps + grid``.  Using the raw bin-start grid
    lags the supervision target by one bin (see ``build_trigger_labels``).
    """

    values = np.asarray(timestamps_ms, dtype=np.int64)
    return values + int(grid_seconds) * 1000


def assert_prediction_time_grid(
    prediction_times_ms: np.ndarray, timestamps_ms: np.ndarray, *, grid_seconds: int = DEFAULT_GRID_SECONDS
) -> None:
    """Fail closed when a caller hands a bin-start grid to the label builder."""

    step = int(grid_seconds) * 1000
    prediction_times = np.asarray(prediction_times_ms, dtype=np.int64)
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if len(prediction_times) != len(timestamps):
        raise ValueError("trigger label grid must have the same length as the split timeline")
    if len(timestamps) == 0:
        return
    if np.array_equal(prediction_times, timestamps):
        raise ValueError(
            "trigger labels must be rasterized on the prediction-time grid "
            "(target_bin_end = target_bin_start + grid), not on the bin-start grid"
        )
    if not np.array_equal(prediction_times, timestamps + step):
        raise ValueError("trigger label grid must equal timestamps + grid_seconds")


def trigger_label_counts(labels: np.ndarray) -> Mapping[str, object]:
    values = np.asarray(labels).reshape(-1)
    total = int(len(values))
    counts = {TRIGGER_LABEL_NAMES[value]: int(np.sum(values == value)) for value in TRIGGER_LABEL_VALUES}
    return {
        "total": total,
        "counts": counts,
        "fractions": {name: (count / total if total else 0.0) for name, count in counts.items()},
        "positive_region_count": int(_region_count(values == TRIGGER_POSITIVE)),
        "ignore_region_count": int(_region_count(values == TRIGGER_IGNORE)),
    }


def _region_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not mask.any():
        return 0
    starts = mask.copy()
    starts[1:] &= ~mask[:-1]
    return int(starts.sum())


def trigger_binary_mask(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(target, loss_mask)``: IGNORE bins are masked out of the loss."""

    values = np.asarray(labels).reshape(-1)
    target = (values == TRIGGER_POSITIVE).astype(np.float32)
    mask = (values != TRIGGER_IGNORE).astype(np.float32)
    if not np.isin(values, TRIGGER_LABEL_VALUES).all():
        raise ValueError("trigger labels must be NEGATIVE/POSITIVE/IGNORE")
    return target, mask


# ---------------------------------------------------------------------------
# onset density and duration strata
# ---------------------------------------------------------------------------


def onset_density(events: pd.DataFrame, *, origin_ms: int, grid_seconds: int = DEFAULT_GRID_SECONDS) -> pd.DataFrame:
    """Count onsets per 30 s detector bin and per distinct root service."""

    grid_ms = int(grid_seconds) * 1000
    frame = events.copy()
    frame["onset_bin"] = np.floor_divide(
        frame["start_ms"].to_numpy(dtype=np.int64) - int(origin_ms), grid_ms
    ).astype(np.int64)
    grouped = frame.groupby("onset_bin", sort=True)
    counts = grouped.size().rename("onset_count")
    services = grouped["service"].nunique().rename("root_service_count")
    summary = pd.concat([counts, services], axis=1).reset_index()
    summary["bin_start_ms"] = int(origin_ms) + summary["onset_bin"] * grid_ms
    return summary


def duration_stratum(durations_seconds: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(durations_seconds), dtype=float)
    edges = np.asarray(DURATION_STRATUM_EDGES, dtype=float)
    indices = np.searchsorted(edges, values, side="left")
    return np.asarray(DURATION_STRATUM_LABELS, dtype=object)[indices]


def duration_stratified_metrics(
    matching: pd.DataFrame,
    case_frame: pd.DataFrame,
    *,
    grid_seconds: int = DEFAULT_GRID_SECONDS,
) -> List[Mapping[str, object]]:
    """GT-event recall and detection delay per duration stratum."""

    frame = _gt_rows(matching).merge(
        case_frame[["case_id", "duration_seconds"]], on="case_id", how="left", validate="one_to_one"
    )
    if frame["duration_seconds"].isna().any():
        raise ValueError("duration stratification is missing GT case durations")
    frame["duration_stratum"] = duration_stratum(frame["duration_seconds"])
    rows: List[Mapping[str, object]] = []
    for label in DURATION_STRATUM_LABELS:
        group = frame.loc[frame["duration_stratum"] == label]
        matched = group.loc[group["match_status"] == "matched"]
        true_positive = int(len(matched))
        false_negative = int((group["match_status"] == "miss").sum())
        rows.append({
            "duration_stratum": label,
            "ground_truth_events": int(len(group)),
            "true_positive_events": true_positive,
            "false_negative_events": false_negative,
            "recall": (true_positive / len(group)) if len(group) else None,
            "small_n": bool(len(group) < SMALL_N_THRESHOLD),
            **_delay_summary(matched["detection_delay_seconds"]),
        })
    return rows


def field_stratified_metrics(matching: pd.DataFrame, field: str) -> List[Mapping[str, object]]:
    """GT-event recall per categorical GT field (``fault_type`` or ``gt_service``)."""

    frame = _gt_rows(matching)
    if field not in frame.columns:
        raise ValueError("matching frame has no field {}".format(field))
    rows: List[Mapping[str, object]] = []
    for value, group in frame.groupby(field, sort=True):
        matched = group.loc[group["match_status"] == "matched"]
        true_positive = int(len(matched))
        false_negative = int((group["match_status"] == "miss").sum())
        rows.append({
            field: str(value),
            "ground_truth_events": int(len(group)),
            "true_positive_events": true_positive,
            "false_negative_events": false_negative,
            "recall": (true_positive / len(group)) if len(group) else None,
            "small_n": bool(len(group) < SMALL_N_THRESHOLD),
            **_delay_summary(matched["detection_delay_seconds"]),
        })
    return rows


def onset_density_stratified_metrics(
    matching: pd.DataFrame,
    *,
    legal_events: pd.DataFrame,
    origin_ms: int,
    grid_seconds: int = DEFAULT_GRID_SECONDS,
) -> Mapping[str, object]:
    """Single-onset versus multi-onset event recall, plus same-bin overlap counts.

    The onset density is computed over the label-legal GT population (the same
    population the trigger label was built from), so "this onset shares its 30 s
    bin with another onset" means the same thing at label time and at metric time.
    """

    density = onset_density(legal_events, origin_ms=origin_ms, grid_seconds=grid_seconds)
    frame = _gt_rows(matching)
    grid_ms = int(grid_seconds) * 1000
    frame["onset_bin"] = np.floor_divide(
        frame["gt_start_ms"].to_numpy(dtype=np.int64) - int(origin_ms), grid_ms
    ).astype(np.int64)
    frame = frame.merge(
        density[["onset_bin", "onset_count", "root_service_count"]], on="onset_bin", how="left"
    )
    if frame["onset_count"].isna().any():
        raise ValueError("onset density join lost GT events")
    strata = {
        "single_onset": frame["onset_count"] == 1,
        "multi_onset": frame["onset_count"] >= 2,
    }
    report: Dict[str, object] = {}
    for name, mask in strata.items():
        group = frame.loc[mask]
        matched = group.loc[group["match_status"] == "matched"]
        true_positive = int(len(matched))
        false_negative = int((group["match_status"] == "miss").sum())
        report[name] = {
            "ground_truth_events": int(len(group)),
            "true_positive_events": true_positive,
            "false_negative_events": false_negative,
            "recall": (true_positive / len(group)) if len(group) else None,
            "small_n": bool(len(group) < SMALL_N_THRESHOLD),
            **_delay_summary(matched["detection_delay_seconds"]),
        }
    report["onset_count_histogram"] = {
        str(int(key)): int(value)
        for key, value in frame["onset_count"].value_counts().sort_index().items()
    }
    report["density_scope"] = "label-legal GT population, 30 s detector bins"
    report["bins_with_onsets"] = int(len(density))
    report["bins_with_multiple_events"] = int((density["onset_count"] >= 2).sum())
    report["bins_with_multiple_root_services"] = int((density["root_service_count"] >= 2).sum())
    report["same_bin_multi_event_events"] = int((frame["onset_count"] >= 2).sum())
    report["same_bin_multi_root_events"] = int((frame["root_service_count"] >= 2).sum())
    return report


def _gt_rows(matching: pd.DataFrame) -> pd.DataFrame:
    frame = matching.loc[matching["match_status"].astype(str).isin(("matched", "miss"))].copy()
    frame["detection_delay_seconds"] = pd.to_numeric(frame["detection_delay_seconds"], errors="coerce")
    return frame


def _delay_summary(delays: Iterable[float]) -> Mapping[str, object]:
    values = pd.to_numeric(
        pd.Series(list(delays), dtype="float64"), errors="coerce"
    ).dropna().to_numpy(dtype=float)
    if not len(values):
        return {
            "mean_delay_seconds": None, "median_delay_seconds": None,
            "p95_delay_seconds": None, "delay_count": 0,
        }
    return {
        "mean_delay_seconds": float(np.mean(values)),
        "median_delay_seconds": float(np.median(values)),
        "p95_delay_seconds": float(np.percentile(values, 95)),
        "delay_count": int(len(values)),
    }


# ---------------------------------------------------------------------------
# frozen-rule threshold selection on a system score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerThresholdSelection:
    """Validation-only threshold decision for the system trigger score."""

    threshold: float
    metrics: Mapping[str, object]
    candidate_count: int
    tie_break: str = "highest threshold among equal Validation event F1"

    @property
    def validation_metrics(self) -> Mapping[str, object]:
        return self.metrics


def system_score_frame(
    split: str, prediction_available_time: np.ndarray, system_score: np.ndarray
) -> pd.DataFrame:
    """One system-level trigger score per prediction timestamp (no service axis)."""

    frame = pd.DataFrame({
        "split": str(split),
        "prediction_available_time": np.asarray(prediction_available_time, dtype=np.int64),
        "system_score": np.asarray(system_score, dtype=float),
    })
    frame["prediction_timestamp"] = frame["prediction_available_time"]
    return frame


def evaluate_system_threshold(
    system_scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    threshold: float,
    *,
    grid_seconds: int = DEFAULT_GRID_SECONDS,
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Episode construction + causal matching + metrics via the frozen functions."""

    episodes = construct_predicted_episodes(system_scores, threshold, grid_seconds=grid_seconds)
    matching = match_events(episodes, ground_truth, tolerance_seconds=tolerance_seconds)
    return episodes, matching, event_metrics(matching)


def select_system_threshold(
    system_scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    grid_seconds: int = DEFAULT_GRID_SECONDS,
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
    workers: int = 1,
    start_method: str = "spawn",
) -> TriggerThresholdSelection:
    """Exact unique-score sweep maximising event F1; ties take the highest threshold.

    This is the frozen ``select_train_threshold`` decision rule evaluated on the
    Detector-Validation score series instead of the Train score series.  Only the
    score source changes; the candidate enumeration, the causal counts, the F1
    objective and the tie-break are reused verbatim.
    """

    frame = system_scores.copy()
    frame["prediction_available_time"] = frame["prediction_available_time"].astype(np.int64)
    frame["system_score"] = frame["system_score"].astype(float)
    frame = frame.sort_values("prediction_available_time", kind="stable").reset_index(drop=True)
    scores = frame["system_score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("system scores contain non-finite values")
    finite = scores[np.isfinite(scores)]
    if len(finite):
        candidates = sorted(set(float(value) for value in finite), reverse=True)
        candidates.insert(0, float(np.nextafter(max(candidates), math.inf)))
    else:
        candidates = [float("inf")]
    gt = ground_truth
    if "start_ms" not in gt:
        raise ValueError("event registry is missing columns: ['start_ms']")
    evaluated = _evaluate_threshold_candidates(
        candidates,
        frame["prediction_available_time"].to_numpy(dtype=np.int64),
        scores,
        gt["start_ms"].to_numpy(dtype=np.int64),
        grid_seconds=grid_seconds,
        tolerance_seconds=tolerance_seconds,
        workers=int(workers),
        start_method=str(start_method),
    )
    best_index, best_threshold, _ = max(evaluated, key=lambda row: (row[2], -row[0]))
    del best_index
    _, _, metrics = evaluate_system_threshold(
        frame, gt, best_threshold,
        grid_seconds=grid_seconds, tolerance_seconds=tolerance_seconds,
    )
    return TriggerThresholdSelection(
        threshold=float(best_threshold), metrics=metrics, candidate_count=len(candidates),
    )


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------


def to_builtin(value):
    """Recursively convert numpy/pandas scalars into JSON-safe builtins."""

    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def assign_legal_events(
    registry: pd.DataFrame, blocks: Sequence[TemporalBlock]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the registry into label-legal, block-assigned and purged events.

    * ``legal`` rows lie inside the frozen metric detector timeline.  They define
      the ground-truth fault *state* used by the trigger label, including the
      single event that crosses the frozen 70/30 boundary, so no ongoing fault is
      mislabelled as NEGATIVE.  They are never used as a metric population.
    * ``assigned`` rows are complete injections contained in exactly one P6-C0
      block.  They are the metric population of each split.
    * ``purged`` rows cross a block boundary (or the timeline) and are excluded
      from every metric population, so no GT event is counted in two splits.
    """

    frame = registry.copy()
    if "detector_domain" not in frame.columns:
        raise ValueError("event registry must carry the frozen detector_domain flag")
    legal = frame.loc[frame["detector_domain"].astype(bool)].copy()
    assigned, purged = assign_event_blocks(frame, blocks)
    if len(purged) and "detector_domain" in purged.columns:
        purged = purged.loc[purged["detector_domain"].astype(bool)].copy()
    if len(assigned) and set(assigned["split"]) - set(SPLIT_NAMES):
        raise ValueError("assigned events use a split outside fit/validation/test")
    return legal, assigned, purged
