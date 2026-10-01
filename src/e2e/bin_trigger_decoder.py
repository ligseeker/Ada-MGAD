"""Independent completed-bin candidates; all duplicates incur normal FP costs."""

import numpy as np
import pandas as pd

from .event_detection import match_events, event_metrics


def construct_bin_candidates(frame, threshold, grid_seconds=30):
    times = frame["prediction_available_time"].to_numpy(dtype=np.int64)
    scores = frame["system_score"].to_numpy(dtype=float)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("candidate times must be strictly increasing")
    selected = np.flatnonzero(scores >= float(threshold))
    split = str(frame["split"].iloc[0]) if len(frame) else "validation"
    return pd.DataFrame({
        "prediction_id": ["{}-bin-{:06d}".format(split, int(i)) for i in selected],
        "split": [split] * len(selected), "t_hat": times[selected],
        "episode_end_time": times[selected] + int(grid_seconds) * 1000,
        "positive_bins": np.ones(len(selected), dtype=np.int64),
        "system_score": scores[selected],
    })


def independent_bin_counts(times, scores, starts, threshold, tolerance_ms=60_000):
    """Exact maximum cardinality for ordered interval matching (counts only).

    Consume the earliest still-eligible GT onset per chronological candidate.
    The official matcher separately chooses identities and minimum total delay.
    """
    anchors = np.asarray(times, dtype=np.int64)[np.asarray(scores) >= float(threshold)]
    starts = np.asarray(starts, dtype=np.int64)
    if (len(starts) > 1 and np.any(np.diff(starts) < 0)) or (
            len(anchors) > 1 and np.any(np.diff(anchors) <= 0)):
        raise ValueError("matching counts require sorted times")
    index, tp = 0, 0
    for anchor in anchors:
        while index < len(starts) and starts[index] < anchor - tolerance_ms:
            index += 1
        if index < len(starts) and starts[index] <= anchor:
            index += 1
            tp += 1
    return tp, len(anchors) - tp, len(starts) - tp


def select_bin_threshold(frame, gt, tolerance_seconds=60):
    """Same Validation F1 objective/highest-threshold tie-break, new decoder."""
    scores = frame["system_score"].to_numpy(dtype=float)
    times = frame["prediction_available_time"].to_numpy(dtype=np.int64)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("finite nonempty scores required")
    starts = np.sort(gt.start_ms.to_numpy(dtype=np.int64), kind="stable")
    candidates = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
    best_tp, best_denominator, best_threshold = 0, max(1, len(starts)), float(candidates[0])
    for threshold in candidates:
        tp, fp, _ = independent_bin_counts(times, scores, starts, threshold, tolerance_seconds * 1000)
        denominator = len(starts) + tp + fp
        # Compare event F1=2TP/(GT+predictions) without floating-point ties.
        if tp * best_denominator > best_tp * denominator:
            best_tp, best_denominator, best_threshold = tp, denominator, float(threshold)
    return best_threshold, int(len(candidates))


def evaluate_bin_threshold(frame, gt, threshold, tolerance_seconds=60):
    candidates = construct_bin_candidates(frame, threshold)
    matching = match_events(candidates, gt, tolerance_seconds=tolerance_seconds)
    metrics = event_metrics(matching)
    return candidates, matching, metrics
