"""Validation-only event triggering and deterministic GAIA event matching.

This module deliberately contains no detector or RCA model.  It consumes the
timestamp-preserving Ada-MGAD-G node probabilities and implements the frozen
P5-I1 event protocol:

* ``S(t) = max_i p_i(t)``;
* choose the threshold using validation event F1 only;
* merge consecutive positive 30-second bins into one predicted episode;
* use the first positive bin as ``t_hat``; and
* greedily match predictions, in chronological order, to the nearest
  unmatched injection onset within the fixed tolerance.

All functions are deterministic.  In particular, no test rows are inspected
by :func:`select_validation_threshold`.
"""

from __future__ import annotations

import bisect
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .protocol import GAIA_SERVICES, TemporalBlock, assign_event_blocks


PREDICTION_COLUMNS = (
    "split",
    "sample_index",
    "window_start_time",
    "window_end_time",
    "prediction_timestamp",
    "service",
    "service_registry_index",
    "anomaly_score",
    "node_label",
    "binary_prediction",
)


@dataclass(frozen=True)
class ThresholdSelection:
    """The validation-only threshold decision and its audit information."""

    threshold: float
    validation_metrics: Mapping[str, object]
    candidate_count: int
    tie_break: str = "highest threshold among equal validation Event F1"


def _as_frame(value: pd.DataFrame | Iterable[Mapping[str, object]]) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(list(value))


def load_prediction_table(path: Path) -> pd.DataFrame:
    """Load and validate a timestamped Ada-MGAD prediction CSV."""

    frame = pd.read_csv(path)
    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("prediction table is missing columns: {}".format(missing))
    if frame.empty:
        raise ValueError("prediction table is empty: {}".format(path))
    if not set(frame["service"].astype(str)).issubset(GAIA_SERVICES):
        raise ValueError("prediction table contains a service outside GAIA_SERVICES")
    frame["prediction_timestamp"] = frame["prediction_timestamp"].astype(np.int64)
    frame["anomaly_score"] = frame["anomaly_score"].astype(float)
    if not np.isfinite(frame["anomaly_score"].to_numpy()).all():
        raise ValueError("prediction table contains non-finite anomaly scores")
    return frame


def aggregate_system_scores(
    predictions: pd.DataFrame,
    *,
    split: Optional[str] = None,
    grid_seconds: int = 30,
) -> pd.DataFrame:
    """Compute ``S(t)=max_i p_i(t)`` for one split.

    The service rows are checked for uniqueness.  Missing services are allowed
    for an adapter smoke fixture, but are represented by the maximum of the
    available service scores and are counted in the returned audit columns.
    """

    frame = _as_frame(predictions)
    if split is not None and "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str) == str(split)].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["split", "prediction_timestamp", "system_score", "service_count", "timestamp_gap_ms"]
        )
    required = {"prediction_timestamp", "service", "anomaly_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("prediction table is missing columns: {}".format(missing))
    duplicate = frame.duplicated(["prediction_timestamp", "service"], keep=False)
    if duplicate.any():
        raise ValueError("duplicate prediction timestamp/service rows detected")
    frame = frame.sort_values(["prediction_timestamp", "service"], kind="stable")
    grouped = frame.groupby("prediction_timestamp", sort=True, as_index=False).agg(
        system_score=("anomaly_score", "max"),
        service_count=("service", "nunique"),
    )
    if not (grouped["service_count"] == len(GAIA_SERVICES)).all():
        raise ValueError("every prediction timestamp must contain all canonical GAIA services")
    if "split" in frame.columns:
        split_values = frame.groupby("prediction_timestamp", sort=True)["split"].first()
        grouped["split"] = grouped["prediction_timestamp"].map(split_values)
    else:
        grouped["split"] = str(split) if split is not None else "unknown"
    grouped = grouped[["split", "prediction_timestamp", "system_score", "service_count"]]
    grouped["timestamp_gap_ms"] = grouped["prediction_timestamp"].diff()
    grouped["timestamp_gap_ms"] = grouped["timestamp_gap_ms"].fillna(int(grid_seconds) * 1000)
    return grouped


def construct_predicted_episodes(
    system_scores: pd.DataFrame,
    threshold: float,
    *,
    grid_seconds: int = 30,
) -> pd.DataFrame:
    """Merge consecutive positive bins and emit one row per episode."""

    frame = _as_frame(system_scores)
    required = {"prediction_timestamp", "system_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("system score table is missing columns: {}".format(missing))
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "prediction_id", "split", "t_hat", "episode_end_time",
                "positive_bins", "system_score", "threshold",
            ]
        )
    frame = frame.sort_values("prediction_timestamp", kind="stable").reset_index(drop=True)
    timestamps = frame["prediction_timestamp"].to_numpy(dtype=np.int64)
    scores = frame["system_score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("system scores contain non-finite values")
    positive = scores >= float(threshold)
    step_ms = int(grid_seconds) * 1000
    starts = positive.copy()
    if len(starts) > 1:
        starts[1:] &= (~positive[:-1]) | ((timestamps[1:] - timestamps[:-1]) != step_ms)
    episode_indices = np.flatnonzero(starts)
    rows: List[Dict[str, object]] = []
    for episode_number, start in enumerate(episode_indices):
        stop = int(start) + 1
        while stop < len(frame) and positive[stop] and timestamps[stop] - timestamps[stop - 1] == step_ms:
            stop += 1
        episode_scores = scores[start:stop]
        split_name = str(frame.iloc[start].get("split", "unknown"))
        rows.append({
            "prediction_id": "{}-pred-{:06d}".format(split_name, episode_number),
            "split": split_name,
            "t_hat": int(timestamps[start]),
            "episode_end_time": int(timestamps[stop - 1] + step_ms),
            "positive_bins": int(stop - start),
            "system_score": float(np.max(episode_scores)),
            "threshold": float(threshold),
        })
    return pd.DataFrame(rows, columns=[
        "prediction_id", "split", "t_hat", "episode_end_time",
        "positive_bins", "system_score", "threshold",
    ])


def _gt_rows_for_split(
    registry: pd.DataFrame,
    split: Optional[str],
    *,
    blocks: Optional[Sequence[TemporalBlock]] = None,
) -> pd.DataFrame:
    """Return raw injections assigned to a split without changing their spans."""

    frame = _as_frame(registry)
    required = {"case_id", "source_index", "service", "fault_type", "start_ms", "end_ms"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("event registry is missing columns: {}".format(missing))
    if blocks is not None and "split" not in frame.columns:
        assigned, _ = assign_event_blocks(frame, blocks)
        frame = assigned
    if split is not None and "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str) == str(split)].copy()
    return frame.sort_values(["start_ms", "source_index", "case_id"], kind="stable").reset_index(drop=True)


def match_events(
    predicted_episodes: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    tolerance_seconds: int = 60,
) -> pd.DataFrame:
    """Perform deterministic one-to-one onset matching.

    Predictions are considered in increasing ``t_hat`` order.  A prediction
    chooses the nearest currently-unmatched GT onset within the tolerance;
    equal distances are resolved by earlier onset, ``source_index``, then
    ``case_id``.  Long fault durations never widen the matching interval.
    """

    predictions = _as_frame(predicted_episodes)
    gt = _as_frame(ground_truth)
    for name, required in (
        ("predicted episodes", {"prediction_id", "t_hat"}),
        ("ground truth", {"case_id", "source_index", "service", "fault_type", "start_ms", "end_ms"}),
    ):
        missing = sorted(required - set((predictions if name.startswith("predicted") else gt).columns))
        if missing:
            raise ValueError("{} is missing columns: {}".format(name, missing))
    tolerance_ms = int(tolerance_seconds) * 1000
    predictions = predictions.sort_values(["t_hat", "prediction_id"], kind="stable").reset_index(drop=True)
    gt = gt.sort_values(["start_ms", "source_index", "case_id"], kind="stable").reset_index(drop=True)
    # Search only the onset-sorted tolerance slice.  This is semantically
    # identical to scanning every unmatched GT, but keeps validation threshold
    # selection practical for the full month.
    gt_starts = gt["start_ms"].to_numpy(dtype=np.int64)
    unmatched = np.ones(len(gt), dtype=bool)
    rows: List[Dict[str, object]] = []
    for prediction in predictions.itertuples(index=False):
        t_hat = int(prediction.t_hat)
        left = int(np.searchsorted(gt_starts, t_hat - tolerance_ms, side="left"))
        right = int(np.searchsorted(gt_starts, t_hat + tolerance_ms, side="right"))
        candidates = np.flatnonzero(unmatched[left:right]).astype(np.int64) + left
        base = {
            "split": getattr(prediction, "split", "unknown"),
            "prediction_id": str(prediction.prediction_id),
            "t_hat": t_hat,
            "episode_end_time": getattr(prediction, "episode_end_time", None),
            "positive_bins": getattr(prediction, "positive_bins", None),
            "system_score": getattr(prediction, "system_score", None),
            "threshold": getattr(prediction, "threshold", None),
        }
        if len(candidates) == 0:
            rows.append({
                **base,
                "match_status": "false_alarm",
                "case_id": None,
                "source_index": None,
                "gt_service": None,
                "fault_type": None,
                "gt_start_ms": None,
                "gt_end_ms": None,
                "detection_delay_seconds": None,
                "absolute_onset_error_seconds": None,
            })
            continue
        selected = min(
            candidates,
            key=lambda index: (
                abs(int(gt.iloc[index]["start_ms"]) - t_hat),
                int(gt.iloc[index]["start_ms"]),
                int(gt.iloc[index]["source_index"]),
                str(gt.iloc[index]["case_id"]),
            ),
        )
        target = gt.iloc[selected]
        unmatched[selected] = False
        delay_seconds = (t_hat - int(target["start_ms"])) / 1000.0
        rows.append({
            **base,
            "match_status": "matched",
            "case_id": str(target["case_id"]),
            "source_index": int(target["source_index"]),
            "gt_service": str(target["service"]),
            "fault_type": str(target["fault_type"]),
            "gt_start_ms": int(target["start_ms"]),
            "gt_end_ms": int(target["end_ms"]),
            "detection_delay_seconds": float(delay_seconds),
            "absolute_onset_error_seconds": float(abs(delay_seconds)),
        })
    for index in sorted(np.flatnonzero(unmatched), key=lambda value: (
        int(gt.iloc[value]["start_ms"]),
        int(gt.iloc[value]["source_index"]),
        str(gt.iloc[value]["case_id"]),
    )):
        target = gt.iloc[index]
        rows.append({
            "split": str(target.get("split", "unknown")),
            "prediction_id": None,
            "t_hat": None,
            "episode_end_time": None,
            "positive_bins": None,
            "system_score": None,
            "threshold": None,
            "match_status": "miss",
            "case_id": str(target["case_id"]),
            "source_index": int(target["source_index"]),
            "gt_service": str(target["service"]),
            "fault_type": str(target["fault_type"]),
            "gt_start_ms": int(target["start_ms"]),
            "gt_end_ms": int(target["end_ms"]),
            "detection_delay_seconds": None,
            "absolute_onset_error_seconds": None,
        })
    columns = [
        "split", "prediction_id", "t_hat", "episode_end_time", "positive_bins",
        "system_score", "threshold", "match_status", "case_id", "source_index",
        "gt_service", "fault_type", "gt_start_ms", "gt_end_ms",
        "detection_delay_seconds", "absolute_onset_error_seconds",
    ]
    return pd.DataFrame(rows, columns=columns)


def event_metrics(matching: pd.DataFrame) -> Dict[str, object]:
    """Summarize event precision/recall/F1 and matched detection delay."""

    frame = _as_frame(matching)
    status = frame["match_status"].astype(str) if "match_status" in frame else pd.Series(dtype=str)
    tp = int((status == "matched").sum())
    fp = int((status == "false_alarm").sum())
    fn = int((status == "miss").sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = pd.to_numeric(
        frame.loc[status == "matched", "detection_delay_seconds"], errors="coerce"
    ).dropna().to_numpy(dtype=float) if len(frame) else np.empty(0, dtype=float)
    delay_metrics = {
        "mean_seconds": float(np.mean(delays)) if len(delays) else None,
        "median_seconds": float(np.median(delays)) if len(delays) else None,
        "p90_seconds": float(np.percentile(delays, 90)) if len(delays) else None,
        "p95_seconds": float(np.percentile(delays, 95)) if len(delays) else None,
        "matched_delay_count": int(len(delays)),
    }
    return {
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "true_positive_events": tp,
        "false_positive_events": fp,
        "false_negative_events": fn,
        "predicted_episode_count": int(tp + fp),
        "ground_truth_event_count": int(tp + fn),
        "detection_delay": delay_metrics,
    }


def evaluate_threshold(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    threshold: float,
    *,
    grid_seconds: int = 30,
    tolerance_seconds: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Construct episodes, match them, and calculate event metrics."""

    scores = aggregate_system_scores(predictions, grid_seconds=grid_seconds)
    episodes = construct_predicted_episodes(scores, threshold, grid_seconds=grid_seconds)
    matching = match_events(episodes, ground_truth, tolerance_seconds=tolerance_seconds)
    return episodes, matching, event_metrics(matching)


def _event_confusion_counts(
    timestamps: np.ndarray,
    system_scores: np.ndarray,
    gt_starts: np.ndarray,
    threshold: float,
    *,
    grid_seconds: int,
    tolerance_seconds: int,
) -> Tuple[int, int, int]:
    """Return event TP/FP/FN without materializing matching data frames.

    This is the selection-only equivalent of ``construct_predicted_episodes``
    followed by ``match_events``.  Ground-truth onsets enter an ordered active
    set as predictions are scanned chronologically.  Only the nearest onset on
    either side of an anchor can win the frozen distance/timestamp tie-break.
    The full matching table is still constructed once after the winning
    validation threshold has been selected.
    """

    timestamp_values = np.asarray(timestamps, dtype=np.int64)
    score_values = np.asarray(system_scores, dtype=float)
    onset_values = np.sort(np.asarray(gt_starts, dtype=np.int64), kind="stable")
    if timestamp_values.shape != score_values.shape:
        raise ValueError("timestamp and system-score arrays must have identical shape")
    if not np.isfinite(score_values).all():
        raise ValueError("system scores contain non-finite values")

    positive = score_values >= float(threshold)
    step_ms = int(grid_seconds) * 1000
    starts = positive.copy()
    if len(starts) > 1:
        starts[1:] &= (~positive[:-1]) | (
            (timestamp_values[1:] - timestamp_values[:-1]) != step_ms
        )
    anchors = timestamp_values[np.flatnonzero(starts)]

    tolerance_ms = int(tolerance_seconds) * 1000
    available: List[Tuple[int, int]] = []
    next_gt = 0
    matched = 0
    for anchor_value in anchors:
        anchor = int(anchor_value)
        upper = anchor + tolerance_ms
        while next_gt < len(onset_values) and int(onset_values[next_gt]) <= upper:
            available.append((int(onset_values[next_gt]), next_gt))
            next_gt += 1

        expired = bisect.bisect_left(available, (anchor - tolerance_ms, -1))
        if expired:
            del available[:expired]
        if not available:
            continue

        right = bisect.bisect_left(available, (anchor, -1))
        candidate_positions: List[int] = []
        if right < len(available):
            candidate_positions.append(right)
        if right:
            predecessor_start = available[right - 1][0]
            candidate_positions.append(
                bisect.bisect_left(available, (predecessor_start, -1), 0, right)
            )
        selected = min(
            candidate_positions,
            key=lambda position: (
                abs(available[position][0] - anchor),
                available[position][0],
                available[position][1],
            ),
        )
        available.pop(selected)
        matched += 1

    predicted = int(len(anchors))
    return matched, predicted - matched, int(len(onset_values)) - matched


def _threshold_chunk(arguments):
    """Evaluate an ordered threshold chunk in a worker process."""

    (
        indexed_candidates,
        timestamps,
        system_scores,
        gt_starts,
        grid_seconds,
        tolerance_seconds,
    ) = arguments
    rows = []
    for index, threshold in indexed_candidates:
        tp, fp, fn = _event_confusion_counts(
            timestamps,
            system_scores,
            gt_starts,
            threshold,
            grid_seconds=grid_seconds,
            tolerance_seconds=tolerance_seconds,
        )
        denominator = 2 * tp + fp + fn
        f1 = (2.0 * tp / denominator) if denominator else 0.0
        rows.append((index, float(threshold), float(f1)))
    return rows


def _evaluate_threshold_candidates(
    candidates: Sequence[float],
    timestamps: np.ndarray,
    system_scores: np.ndarray,
    gt_starts: np.ndarray,
    *,
    grid_seconds: int,
    tolerance_seconds: int,
    workers: int,
    start_method: str,
) -> List[Tuple[int, float, float]]:
    """Evaluate exact threshold candidates, optionally in deterministic chunks."""

    indexed = list(enumerate(float(value) for value in candidates))
    worker_count = max(1, min(int(workers), len(indexed)))
    if worker_count == 1:
        return _threshold_chunk((
            indexed, timestamps, system_scores, gt_starts,
            grid_seconds, tolerance_seconds,
        ))

    # Several chunks per worker balance candidate-dependent episode counts.
    chunk_size = max(1, math.ceil(len(indexed) / (worker_count * 4)))
    tasks = [
        (
            indexed[offset:offset + chunk_size],
            timestamps,
            system_scores,
            gt_starts,
            grid_seconds,
            tolerance_seconds,
        )
        for offset in range(0, len(indexed), chunk_size)
    ]
    context = mp.get_context(str(start_method))
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        chunks = list(pool.map(_threshold_chunk, tasks))
    return sorted((row for chunk in chunks for row in chunk), key=lambda row: row[0])


def select_validation_threshold(
    validation_predictions: pd.DataFrame,
    validation_ground_truth: pd.DataFrame,
    *,
    grid_seconds: int = 30,
    tolerance_seconds: int = 60,
    workers: int = 1,
    start_method: str = "spawn",
) -> ThresholdSelection:
    """Select a threshold by validation Event F1 only.

    Event F1 is piecewise constant between observed system scores, therefore
    evaluating every unique finite score (plus an explicit no-positive
    candidate) is exact for the frozen threshold rule.  Equal-F1 candidates
    retain the first candidate in descending order, i.e. the highest
    threshold, as a conservative deterministic tie-break.
    """

    scores = aggregate_system_scores(validation_predictions, grid_seconds=grid_seconds)
    finite_scores = scores["system_score"].to_numpy(dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if len(finite_scores):
        candidates = sorted(set(float(value) for value in finite_scores), reverse=True)
        candidates.insert(0, float(np.nextafter(max(candidates), math.inf)))
    else:
        candidates = [float("inf")]
    gt = _as_frame(validation_ground_truth)
    if "start_ms" not in gt:
        raise ValueError("event registry is missing columns: ['start_ms']")
    evaluated = _evaluate_threshold_candidates(
        candidates,
        scores["prediction_timestamp"].to_numpy(dtype=np.int64),
        scores["system_score"].to_numpy(dtype=float),
        gt["start_ms"].to_numpy(dtype=np.int64),
        grid_seconds=grid_seconds,
        tolerance_seconds=tolerance_seconds,
        workers=workers,
        start_method=start_method,
    )
    # Candidates are descending.  Strict improvement preserves the frozen
    # highest-threshold tie-break.
    best_index, best_threshold, _ = max(evaluated, key=lambda row: (row[2], -row[0]))
    del best_index
    _, matching, best_metrics = evaluate_threshold(
        validation_predictions,
        validation_ground_truth,
        best_threshold,
        grid_seconds=grid_seconds,
        tolerance_seconds=tolerance_seconds,
    )
    return ThresholdSelection(
        threshold=float(best_threshold),
        validation_metrics=best_metrics,
        candidate_count=len(candidates),
    )


def run_event_detection(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    registry: pd.DataFrame,
    blocks: Sequence[TemporalBlock],
    *,
    grid_seconds: int = 30,
    tolerance_seconds: int = 60,
    threshold_workers: int = 1,
    threshold_start_method: str = "spawn",
) -> Mapping[str, object]:
    """Run validation threshold selection and test event evaluation."""

    validation_gt = _gt_rows_for_split(registry, "validation", blocks=blocks)
    test_gt = _gt_rows_for_split(registry, "test", blocks=blocks)
    selection = select_validation_threshold(
        validation_predictions,
        validation_gt,
        grid_seconds=grid_seconds,
        tolerance_seconds=tolerance_seconds,
        workers=threshold_workers,
        start_method=threshold_start_method,
    )
    val_episodes, val_matching, val_metrics = evaluate_threshold(
        validation_predictions,
        validation_gt,
        selection.threshold,
        grid_seconds=grid_seconds,
        tolerance_seconds=tolerance_seconds,
    )
    test_episodes, test_matching, test_metrics = evaluate_threshold(
        test_predictions,
        test_gt,
        selection.threshold,
        grid_seconds=grid_seconds,
        tolerance_seconds=tolerance_seconds,
    )
    all_episodes = pd.concat([val_episodes, test_episodes], ignore_index=True)
    all_matching = pd.concat([val_matching, test_matching], ignore_index=True)
    return {
        "threshold_selection": selection,
        "validation_episodes": val_episodes,
        "test_episodes": test_episodes,
        "event_predictions": all_episodes,
        "validation_matching": val_matching,
        "test_matching": test_matching,
        "matching": all_matching,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
