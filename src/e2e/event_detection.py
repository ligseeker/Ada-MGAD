"""Train-only event triggering and causal one-to-one GAIA matching.

The detector consumes timestamped Ada-MGAD service probabilities. V3 keeps
the model output timestamp separate from the target-bin start: an episode
anchor is ``t_hat = prediction_available_time = target_bin_end``. Threshold
selection uses Train rows only. Test rows are transformed with that frozen
threshold and are never used by fitting or selection.
"""

from __future__ import annotations

import heapq
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .protocol import GAIA_SERVICES, TemporalBlock, assign_event_blocks


PREDICTION_COLUMNS = (
    "split", "sample_index", "window_start_time", "window_end_time",
    "target_bin_start", "target_bin_end", "prediction_available_time",
    "prediction_timestamp", "service", "service_registry_index",
    "anomaly_score", "node_label", "binary_prediction",
)


@dataclass(frozen=True)
class ThresholdSelection:
    """The frozen Train-only threshold decision and its audit information."""

    threshold: float
    train_metrics: Mapping[str, object]
    candidate_count: int
    tie_break: str = "highest threshold among equal Train event F1"

    @property
    def validation_metrics(self) -> Mapping[str, object]:
        """Compatibility alias; V3 has no validation split."""

        return self.train_metrics


def _as_frame(value: pd.DataFrame | Iterable[Mapping[str, object]]) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(list(value))


def _prediction_time_column(frame: pd.DataFrame) -> str:
    if "prediction_available_time" in frame.columns:
        return "prediction_available_time"
    if "prediction_timestamp" in frame.columns:
        # Legacy fixtures are accepted at this API boundary. Formal V3
        # writers always provide prediction_available_time explicitly.
        return "prediction_timestamp"
    raise ValueError(
        "prediction table is missing prediction_available_time (or compatibility prediction_timestamp)"
    )


def load_prediction_table(path) -> pd.DataFrame:
    """Load and validate a timestamped Ada-MGAD prediction CSV."""

    frame = pd.read_csv(path)
    required = {"split", "service", "anomaly_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("prediction table is missing columns: {}".format(missing))
    time_column = _prediction_time_column(frame)
    if frame.empty:
        raise ValueError("prediction table is empty: {}".format(path))
    if not set(frame["service"].astype(str)).issubset(GAIA_SERVICES):
        raise ValueError("prediction table contains a service outside GAIA_SERVICES")
    frame[time_column] = frame[time_column].astype(np.int64)
    if time_column != "prediction_available_time":
        frame["prediction_available_time"] = frame[time_column]
    frame["prediction_timestamp"] = frame["prediction_available_time"]
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
    """Compute ``S(t)=max_i p_i(t)`` with complete service-row validation."""

    frame = _as_frame(predictions)
    if split is not None and "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str) == str(split)].copy()
    if frame.empty:
        return pd.DataFrame(columns=[
            "split", "prediction_available_time", "prediction_timestamp",
            "system_score", "service_count", "timestamp_gap_ms",
        ])
    required = {"service", "anomaly_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("prediction table is missing columns: {}".format(missing))
    time_column = _prediction_time_column(frame)
    frame["prediction_available_time"] = frame[time_column].astype(np.int64)
    frame["prediction_timestamp"] = frame["prediction_available_time"]
    duplicate = frame.duplicated(["prediction_available_time", "service"], keep=False)
    if duplicate.any():
        raise ValueError("duplicate prediction time/service rows detected")
    frame = frame.sort_values(["prediction_available_time", "service"], kind="stable")
    grouped = frame.groupby("prediction_available_time", sort=True, as_index=False).agg(
        system_score=("anomaly_score", "max"),
        service_count=("service", "nunique"),
    )
    if not (grouped["service_count"] == len(GAIA_SERVICES)).all():
        raise ValueError("every prediction timestamp must contain all canonical GAIA services")
    if "split" in frame.columns:
        split_values = frame.groupby("prediction_available_time", sort=True)["split"].first()
        grouped["split"] = grouped["prediction_available_time"].map(split_values)
    else:
        grouped["split"] = str(split) if split is not None else "unknown"
    grouped["prediction_timestamp"] = grouped["prediction_available_time"]
    grouped = grouped[[
        "split", "prediction_available_time", "prediction_timestamp",
        "system_score", "service_count",
    ]]
    grouped["timestamp_gap_ms"] = grouped["prediction_available_time"].diff()
    grouped["timestamp_gap_ms"] = grouped["timestamp_gap_ms"].fillna(int(grid_seconds) * 1000)
    return grouped


def construct_predicted_episodes(
    system_scores: pd.DataFrame,
    threshold: float,
    *,
    grid_seconds: int = 30,
) -> pd.DataFrame:
    """Merge consecutive positive prediction-available bins into episodes."""

    frame = _as_frame(system_scores)
    if frame.empty:
        return pd.DataFrame(columns=[
            "prediction_id", "split", "t_hat", "episode_end_time", "positive_bins",
            "system_score", "threshold",
        ])
    time_column = _prediction_time_column(frame)
    required = {"system_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("system score table is missing columns: {}".format(missing))
    frame["prediction_available_time"] = frame[time_column].astype(np.int64)
    frame = frame.sort_values("prediction_available_time", kind="stable").reset_index(drop=True)
    timestamps = frame["prediction_available_time"].to_numpy(dtype=np.int64)
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
        while (
            stop < len(frame)
            and positive[stop]
            and timestamps[stop] - timestamps[stop - 1] == step_ms
        ):
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
        "prediction_id", "split", "t_hat", "episode_end_time", "positive_bins",
        "system_score", "threshold",
    ])


def _gt_rows_for_split(
    registry: pd.DataFrame,
    split: Optional[str],
    *,
    blocks: Optional[Sequence[TemporalBlock]] = None,
) -> pd.DataFrame:
    """Return complete raw injections assigned to one Train/Test split."""

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
    starts = pd.to_numeric(frame["start_ms"], errors="coerce")
    ends = pd.to_numeric(frame["end_ms"], errors="coerce")
    if (ends <= starts).any():
        raise ValueError("ground-truth intervals must be positive half-open spans")
    return frame.sort_values(["start_ms", "source_index", "case_id"], kind="stable").reset_index(drop=True)


class _FlowEdge:
    __slots__ = ("to", "reverse", "capacity", "cost")

    def __init__(self, to: int, reverse: int, capacity: int, cost: int):
        self.to = int(to)
        self.reverse = int(reverse)
        self.capacity = int(capacity)
        self.cost = int(cost)


def _add_flow_edge(graph: List[List[_FlowEdge]], source: int, target: int, cost: int) -> None:
    forward = _FlowEdge(target, len(graph[target]), 1, cost)
    reverse = _FlowEdge(source, len(graph[source]), 0, -int(cost))
    graph[source].append(forward)
    graph[target].append(reverse)


def _min_cost_max_cardinality(
    predictions: pd.DataFrame,
    gt: pd.DataFrame,
    tolerance_ms: int,
) -> Mapping[int, int]:
    """Solve sparse causal bipartite matching exactly.

    Successive shortest augmenting paths first maximize flow because the
    augmenting loop continues until no path remains. Edge costs encode delay
    as the primary objective and stable GT order as a secondary objective.
    """

    n_pred, n_gt = len(predictions), len(gt)
    if not n_pred or not n_gt:
        return {}
    pred_times = predictions["t_hat"].to_numpy(dtype=np.int64)
    gt_times = gt["start_ms"].to_numpy(dtype=np.int64)
    source = 0
    pred_offset = 1
    gt_offset = pred_offset + n_pred
    sink = gt_offset + n_gt
    graph: List[List[_FlowEdge]] = [[] for _ in range(sink + 1)]
    tie_scale = (n_gt + 1) * (n_pred + 1)
    for pred_index in range(n_pred):
        _add_flow_edge(graph, source, pred_offset + pred_index, 0)
    for gt_index in range(n_gt):
        _add_flow_edge(graph, gt_offset + gt_index, sink, 0)
    for pred_index, t_hat in enumerate(pred_times):
        left = int(np.searchsorted(gt_times, int(t_hat) - tolerance_ms, side="left"))
        right = int(np.searchsorted(gt_times, int(t_hat), side="right"))
        for gt_index in range(left, right):
            delay = int(t_hat) - int(gt_times[gt_index])
            if delay < 0 or delay > tolerance_ms:
                continue
            _add_flow_edge(
                graph, pred_offset + pred_index, gt_offset + gt_index,
                delay * tie_scale + gt_index,
            )

    node_count = len(graph)
    potential = [0] * node_count
    matches: Dict[int, int] = {}
    while True:
        distances = [math.inf] * node_count
        previous: List[Optional[Tuple[int, int]]] = [None] * node_count
        distances[source] = 0
        heap: List[Tuple[int, int]] = [(0, source)]
        while heap:
            distance, node = heapq.heappop(heap)
            if distance != distances[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                reduced = int(edge.cost) + potential[node] - potential[edge.to]
                candidate = distance + reduced
                if candidate < distances[edge.to] or (
                    candidate == distances[edge.to]
                    and (previous[edge.to] is None or node < previous[edge.to][0])
                ):
                    distances[edge.to] = candidate
                    previous[edge.to] = (node, edge_index)
                    heapq.heappush(heap, (candidate, edge.to))
        if previous[sink] is None:
            break
        for node in range(node_count):
            if distances[node] < math.inf:
                potential[node] += int(distances[node])
        node = sink
        while node != source:
            previous_node, edge_index = previous[node]
            edge = graph[previous_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = previous_node

    for pred_index in range(n_pred):
        node = pred_offset + pred_index
        for edge in graph[node]:
            if gt_offset <= edge.to < sink and edge.capacity == 0:
                matches[pred_index] = edge.to - gt_offset
                break
    return matches


def match_events(
    predicted_episodes: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    tolerance_seconds: int = 60,
) -> pd.DataFrame:
    """Causally match episodes using max-cardinality then minimum total delay."""

    predictions = _as_frame(predicted_episodes)
    gt = _as_frame(ground_truth)
    if predictions.empty and not len(predictions.columns):
        predictions = pd.DataFrame(columns=["prediction_id", "t_hat"])
    if gt.empty and not len(gt.columns):
        gt = pd.DataFrame(columns=[
            "case_id", "source_index", "service", "fault_type", "start_ms", "end_ms",
        ])
    pred_required = {"prediction_id", "t_hat"}
    gt_required = {"case_id", "source_index", "service", "fault_type", "start_ms", "end_ms"}
    missing_pred = sorted(pred_required - set(predictions.columns))
    missing_gt = sorted(gt_required - set(gt.columns))
    if missing_pred:
        raise ValueError("predicted episodes is missing columns: {}".format(missing_pred))
    if missing_gt:
        raise ValueError("ground truth is missing columns: {}".format(missing_gt))
    if predictions["prediction_id"].duplicated().any():
        raise ValueError("predicted episode IDs must be unique")
    tolerance_ms = int(tolerance_seconds) * 1000
    predictions = predictions.sort_values(["t_hat", "prediction_id"], kind="stable").reset_index(drop=True)
    gt = gt.sort_values(["start_ms", "source_index", "case_id"], kind="stable").reset_index(drop=True)
    matches = _min_cost_max_cardinality(predictions, gt, tolerance_ms)
    matched_gt = set(matches.values())
    rows: List[Dict[str, object]] = []
    for pred_index, prediction in enumerate(predictions.itertuples(index=False)):
        base = {
            "split": getattr(prediction, "split", "unknown"),
            "prediction_id": str(prediction.prediction_id), "t_hat": int(prediction.t_hat),
            "episode_end_time": getattr(prediction, "episode_end_time", None),
            "positive_bins": getattr(prediction, "positive_bins", None),
            "system_score": getattr(prediction, "system_score", None),
            "threshold": getattr(prediction, "threshold", None),
        }
        if pred_index not in matches:
            rows.append({
                **base, "match_status": "false_alarm", "case_id": None,
                "source_index": None, "gt_service": None, "fault_type": None,
                "gt_start_ms": None, "gt_end_ms": None,
                "detection_delay_seconds": None, "absolute_onset_error_seconds": None,
            })
            continue
        target = gt.iloc[matches[pred_index]]
        delay = int(prediction.t_hat) - int(target["start_ms"])
        rows.append({
            **base, "match_status": "matched", "case_id": str(target["case_id"]),
            "source_index": int(target["source_index"]), "gt_service": str(target["service"]),
            "fault_type": str(target["fault_type"]), "gt_start_ms": int(target["start_ms"]),
            "gt_end_ms": int(target["end_ms"]),
            "detection_delay_seconds": float(delay / 1000.0),
            "absolute_onset_error_seconds": float(delay / 1000.0),
        })
    for gt_index in range(len(gt)):
        if gt_index in matched_gt:
            continue
        target = gt.iloc[gt_index]
        rows.append({
            "split": str(target.get("split", "unknown")), "prediction_id": None,
            "t_hat": None, "episode_end_time": None, "positive_bins": None,
            "system_score": None, "threshold": None, "match_status": "miss",
            "case_id": str(target["case_id"]), "source_index": int(target["source_index"]),
            "gt_service": str(target["service"]), "fault_type": str(target["fault_type"]),
            "gt_start_ms": int(target["start_ms"]), "gt_end_ms": int(target["end_ms"]),
            "detection_delay_seconds": None, "absolute_onset_error_seconds": None,
        })
    columns = [
        "split", "prediction_id", "t_hat", "episode_end_time", "positive_bins",
        "system_score", "threshold", "match_status", "case_id", "source_index",
        "gt_service", "fault_type", "gt_start_ms", "gt_end_ms",
        "detection_delay_seconds", "absolute_onset_error_seconds",
    ]
    return pd.DataFrame(rows, columns=columns)


def event_metrics(matching: pd.DataFrame) -> Dict[str, object]:
    """Summarize event precision/recall/F1 and causal detection delay."""

    frame = _as_frame(matching)
    status = frame["match_status"].astype(str) if "match_status" in frame else pd.Series(dtype=str)
    tp = int((status == "matched").sum())
    fp = int((status == "false_alarm").sum())
    fn = int((status == "miss").sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = (
        pd.to_numeric(frame.loc[status == "matched", "detection_delay_seconds"], errors="coerce")
        .dropna().to_numpy(dtype=float)
        if len(frame) else np.empty(0, dtype=float)
    )
    delay_metrics = {
        "mean_seconds": float(np.mean(delays)) if len(delays) else None,
        "median_seconds": float(np.median(delays)) if len(delays) else None,
        "p90_seconds": float(np.percentile(delays, 90)) if len(delays) else None,
        "p95_seconds": float(np.percentile(delays, 95)) if len(delays) else None,
        "matched_delay_count": int(len(delays)),
        "causal_rule": "0 <= t_hat - gt_start_ms <= tolerance_seconds",
    }
    return {
        "event_precision": float(precision), "event_recall": float(recall),
        "event_f1": float(f1), "true_positive_events": tp,
        "false_positive_events": fp, "false_negative_events": fn,
        "predicted_episode_count": int(tp + fp),
        "ground_truth_event_count": int(tp + fn), "detection_delay": delay_metrics,
    }


def evaluate_threshold(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    threshold: float,
    *,
    grid_seconds: int = 30,
    tolerance_seconds: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
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
    """Fast causal maximum-cardinality counts for threshold selection."""

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
        starts[1:] &= (~positive[:-1]) | ((timestamp_values[1:] - timestamp_values[:-1]) != step_ms)
    anchors = timestamp_values[np.flatnonzero(starts)]
    tolerance_ms = int(tolerance_seconds) * 1000
    available: List[Tuple[int, int]] = []
    next_gt = 0
    matched = 0
    for anchor in anchors:
        anchor = int(anchor)
        while next_gt < len(onset_values) and int(onset_values[next_gt]) <= anchor:
            heapq.heappush(available, (int(onset_values[next_gt]), next_gt))
            next_gt += 1
        while available and available[0][0] < anchor - tolerance_ms:
            heapq.heappop(available)
        if available:
            heapq.heappop(available)
            matched += 1
    predicted = int(len(anchors))
    return matched, predicted - matched, int(len(onset_values)) - matched


def _threshold_chunk(arguments):
    indexed_candidates, timestamps, system_scores, gt_starts, grid_seconds, tolerance_seconds = arguments
    rows = []
    for index, threshold in indexed_candidates:
        tp, fp, fn = _event_confusion_counts(
            timestamps, system_scores, gt_starts, threshold,
            grid_seconds=grid_seconds, tolerance_seconds=tolerance_seconds,
        )
        denominator = 2 * tp + fp + fn
        rows.append((index, float(threshold), float(2.0 * tp / denominator) if denominator else 0.0))
    return rows


def _evaluate_threshold_candidates(
    candidates: Sequence[float], timestamps: np.ndarray, system_scores: np.ndarray,
    gt_starts: np.ndarray, *, grid_seconds: int, tolerance_seconds: int,
    workers: int, start_method: str,
) -> List[Tuple[int, float, float]]:
    indexed = list(enumerate(float(value) for value in candidates))
    if not indexed:
        return []
    worker_count = max(1, min(int(workers), len(indexed)))
    if worker_count == 1:
        return _threshold_chunk((indexed, timestamps, system_scores, gt_starts, grid_seconds, tolerance_seconds))
    chunk_size = max(1, math.ceil(len(indexed) / (worker_count * 4)))
    tasks = [
        (indexed[offset:offset + chunk_size], timestamps, system_scores, gt_starts,
         grid_seconds, tolerance_seconds)
        for offset in range(0, len(indexed), chunk_size)
    ]
    context = mp.get_context(str(start_method))
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        chunks = list(pool.map(_threshold_chunk, tasks))
    return sorted((row for chunk in chunks for row in chunk), key=lambda row: row[0])


def select_train_threshold(
    train_predictions: pd.DataFrame,
    train_ground_truth: pd.DataFrame,
    *,
    grid_seconds: int = 30,
    tolerance_seconds: int = 60,
    workers: int = 1,
    start_method: str = "spawn",
) -> ThresholdSelection:
    """Select the highest threshold attaining maximum Train event F1."""

    scores = aggregate_system_scores(train_predictions, split="train", grid_seconds=grid_seconds)
    finite_scores = scores["system_score"].to_numpy(dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if len(finite_scores):
        candidates = sorted(set(float(value) for value in finite_scores), reverse=True)
        candidates.insert(0, float(np.nextafter(max(candidates), math.inf)))
    else:
        candidates = [float("inf")]
    gt = _as_frame(train_ground_truth)
    if "start_ms" not in gt:
        raise ValueError("event registry is missing columns: ['start_ms']")
    evaluated = _evaluate_threshold_candidates(
        candidates, scores["prediction_available_time"].to_numpy(dtype=np.int64),
        scores["system_score"].to_numpy(dtype=float),
        gt["start_ms"].to_numpy(dtype=np.int64),
        grid_seconds=grid_seconds, tolerance_seconds=tolerance_seconds,
        workers=workers, start_method=start_method,
    )
    best_index, best_threshold, _ = max(evaluated, key=lambda row: (row[2], -row[0]))
    del best_index
    _, _, best_metrics = evaluate_threshold(
        train_predictions, train_ground_truth, best_threshold,
        grid_seconds=grid_seconds, tolerance_seconds=tolerance_seconds,
    )
    return ThresholdSelection(
        threshold=float(best_threshold), train_metrics=best_metrics,
        candidate_count=len(candidates),
    )


def select_validation_threshold(*args, **kwargs) -> ThresholdSelection:
    """Deprecated compatibility name mapped to the Train-only selector."""

    if len(args) >= 1:
        predictions = args[0].copy()
        if "split" in predictions.columns:
            predictions["split"] = "train"
        args = (predictions,) + args[1:]
    if len(args) >= 2:
        ground_truth = args[1].copy()
        if "split" in ground_truth.columns:
            ground_truth["split"] = "train"
        args = args[:1] + (ground_truth,) + args[2:]
    return select_train_threshold(*args, **kwargs)


def run_event_detection(
    train_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    registry: pd.DataFrame,
    blocks: Sequence[TemporalBlock],
    *,
    grid_seconds: int = 30,
    tolerance_seconds: int = 60,
    threshold_workers: int = 1,
    threshold_start_method: str = "spawn",
) -> Mapping[str, object]:
    """Select on Train and evaluate both Train diagnostics and frozen Test."""

    train_gt = _gt_rows_for_split(registry, "train", blocks=blocks)
    test_gt = _gt_rows_for_split(registry, "test", blocks=blocks)
    selection = select_train_threshold(
        train_predictions, train_gt, grid_seconds=grid_seconds,
        tolerance_seconds=tolerance_seconds, workers=threshold_workers,
        start_method=threshold_start_method,
    )
    train_episodes, train_matching, train_metrics = evaluate_threshold(
        train_predictions, train_gt, selection.threshold,
        grid_seconds=grid_seconds, tolerance_seconds=tolerance_seconds,
    )
    test_episodes, test_matching, test_metrics = evaluate_threshold(
        test_predictions, test_gt, selection.threshold,
        grid_seconds=grid_seconds, tolerance_seconds=tolerance_seconds,
    )
    all_episodes = pd.concat([train_episodes, test_episodes], ignore_index=True)
    all_matching = pd.concat([train_matching, test_matching], ignore_index=True)
    return {
        "threshold_selection": selection, "train_episodes": train_episodes,
        "test_episodes": test_episodes, "event_predictions": all_episodes,
        "train_matching": train_matching, "test_matching": test_matching,
        "matching": all_matching, "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
