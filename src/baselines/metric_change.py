"""Within-case metric mean-shift sanity baseline primitives."""

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np

from src.data.schema import RCACaseInput, assert_label_free


@dataclass(frozen=True)
class MetricChangePrediction:
    ranking: Tuple[str, ...]
    service_scores: Mapping[str, float]
    observed_feature_counts: Mapping[str, int]
    fallback_services: Tuple[str, ...]


def window_mean_shift_scores(
    timestamps_ms: Sequence[int],
    values: Sequence[float],
    anchors_ms: Sequence[int],
    radius_ms: int,
    min_samples_per_side: int = 2,
    score_cap: float = 20.0,
) -> np.ndarray:
    """Compute standardized pre/post mean shifts for many anchors.

    Duplicate timestamps are averaged before scoring. Windows are half-open:
    pre is ``[t0-radius, t0)`` and post is ``[t0, t0+radius)``. The scale is
    the pooled population standard deviation, with a relative epsilon; scores
    are capped to keep constant-to-shift features from dominating aggregation.
    Invalid or under-covered anchors return NaN.
    """

    if radius_ms <= 0 or min_samples_per_side < 1 or score_cap <= 0:
        raise ValueError("metric-change parameters must be positive")
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    numeric_values = np.asarray(values, dtype=np.float64)
    anchors = np.asarray(anchors_ms, dtype=np.int64)
    if timestamps.shape != numeric_values.shape:
        raise ValueError("timestamps and values must have identical shape")
    result = np.full(anchors.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numeric_values)
    if not valid.any() or anchors.size == 0:
        return result
    timestamps = timestamps[valid]
    numeric_values = numeric_values[valid]
    order = np.argsort(timestamps, kind="mergesort")
    timestamps = timestamps[order]
    numeric_values = numeric_values[order]

    unique_timestamps, starts, counts = np.unique(
        timestamps, return_index=True, return_counts=True
    )
    reduced_values = np.add.reduceat(numeric_values, starts) / counts
    prefix_sum = np.concatenate(([0.0], np.cumsum(reduced_values)))
    prefix_square = np.concatenate(
        ([0.0], np.cumsum(reduced_values * reduced_values))
    )

    left = np.searchsorted(unique_timestamps, anchors - radius_ms, side="left")
    middle = np.searchsorted(unique_timestamps, anchors, side="left")
    right = np.searchsorted(unique_timestamps, anchors + radius_ms, side="left")
    pre_count = middle - left
    post_count = right - middle
    covered = (pre_count >= min_samples_per_side) & (
        post_count >= min_samples_per_side
    )
    if not covered.any():
        return result

    pre_sum = prefix_sum[middle] - prefix_sum[left]
    post_sum = prefix_sum[right] - prefix_sum[middle]
    pre_square = prefix_square[middle] - prefix_square[left]
    post_square = prefix_square[right] - prefix_square[middle]
    safe_pre_count = np.maximum(pre_count, 1)
    safe_post_count = np.maximum(post_count, 1)
    pre_mean = pre_sum / safe_pre_count
    post_mean = post_sum / safe_post_count
    total_count = safe_pre_count + safe_post_count
    total_sum = pre_sum + post_sum
    total_square = pre_square + post_square
    pooled_variance = np.maximum(
        total_square / total_count - (total_sum / total_count) ** 2, 0.0
    )
    pooled_scale = np.sqrt(pooled_variance)
    epsilon = 1e-6 * np.maximum.reduce(
        (np.abs(pre_mean), np.abs(post_mean), np.ones_like(pre_mean))
    )
    scores = np.minimum(
        np.abs(post_mean - pre_mean) / (pooled_scale + epsilon), score_cap
    )
    result[covered] = scores[covered]
    return result


def metric_change_prediction(
    case_input: RCACaseInput,
    service_scores: Mapping[str, float],
    observed_feature_counts: Mapping[str, int],
) -> MetricChangePrediction:
    """Create a complete ranking with missing services placed last."""

    assert_label_free(case_input)
    unknown = set(service_scores) - set(case_input.services)
    if unknown:
        raise ValueError("metric scores contain unknown candidate services")
    normalized_scores = {}
    normalized_counts = {}
    fallback = []
    for service in case_input.services:
        score = float(service_scores.get(service, float("nan")))
        count = int(observed_feature_counts.get(service, 0))
        if count < 0:
            raise ValueError("observed feature count cannot be negative")
        normalized_counts[service] = count
        if count == 0 or not np.isfinite(score):
            fallback.append(service)
            normalized_scores[service] = float("nan")
        else:
            normalized_scores[service] = score
    ranking = tuple(
        sorted(
            case_input.services,
            key=lambda service: (
                not np.isfinite(normalized_scores[service]),
                -normalized_scores[service]
                if np.isfinite(normalized_scores[service])
                else 0.0,
                service,
            ),
        )
    )
    return MetricChangePrediction(
        ranking=ranking,
        service_scores=normalized_scores,
        observed_feature_counts=normalized_counts,
        fallback_services=tuple(sorted(fallback)),
    )
