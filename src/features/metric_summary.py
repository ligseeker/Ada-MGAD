"""Dataset-independent whole/staged metric summary primitives for P2."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_ONSET_SECONDS = (30, 60, 120)
CONTRAST_FIELDS = ("signed_mean_shift", "log_scale_ratio", "slope_shift")
AGGREGATE_STATISTICS = (
    "abs_max",
    "abs_top3_mean",
    "abs_top5_mean",
    "signed_mean",
    "positive_fraction",
)


@dataclass(frozen=True)
class MetricContrast:
    """Dimensionless changes between two sufficiently observed metric segments."""

    observed: bool
    signed_mean_shift: float = 0.0
    log_scale_ratio: float = 0.0
    slope_shift: float = 0.0


@dataclass(frozen=True)
class MetricContrastBatch:
    """Vectorized contrasts for one raw series over many case anchors."""

    observed: np.ndarray
    signed_mean_shift: np.ndarray
    log_scale_ratio: np.ndarray
    slope_shift: np.ndarray


@dataclass(frozen=True)
class _SegmentBatch:
    count: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    slope_per_second: np.ndarray
    duration_seconds: float


def _contrast_names() -> Tuple[str, ...]:
    return ("whole",) + tuple(
        "stage{}.{}".format(onset, comparison)
        for onset in DEFAULT_ONSET_SECONDS
        for comparison in ("pre_onset", "pre_impact", "onset_impact")
    )


@lru_cache(maxsize=1)
def metric_feature_names() -> Tuple[str, ...]:
    """Return the fixed streamable feature schema used by every service row."""

    names = []
    for contrast_name in _contrast_names():
        names.extend(
            (contrast_name + ".valid_count", contrast_name + ".valid_ratio")
        )
        for field in CONTRAST_FIELDS:
            names.extend(
                "{}.{}.{}".format(contrast_name, field, statistic)
                for statistic in AGGREGATE_STATISTICS
            )
    return tuple(names)


def _deduplicate(timestamps_ms, values) -> Tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    numeric = np.asarray(values, dtype=np.float64)
    if timestamps.shape != numeric.shape:
        raise ValueError("timestamps and values must have identical shape")
    valid = np.isfinite(numeric)
    timestamps = timestamps[valid]
    numeric = numeric[valid]
    if timestamps.size == 0:
        return timestamps, numeric
    order = np.argsort(timestamps, kind="mergesort")
    timestamps = timestamps[order]
    numeric = numeric[order]
    unique, starts, counts = np.unique(
        timestamps, return_index=True, return_counts=True
    )
    reduced = np.add.reduceat(numeric, starts) / counts
    return unique, reduced


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def _segment_many(
    timestamps: np.ndarray,
    values: np.ndarray,
    starts_ms: np.ndarray,
    ends_ms: np.ndarray,
) -> _SegmentBatch:
    left = np.searchsorted(timestamps, starts_ms, side="left")
    right = np.searchsorted(timestamps, ends_ms, side="left")
    count = right - left
    safe_count = np.maximum(count, 1)

    value_sum = _prefix(values)
    square_sum = _prefix(values * values)
    sums = value_sum[right] - value_sum[left]
    squares = square_sum[right] - square_sum[left]
    mean = sums / safe_count
    variance = np.maximum(squares / safe_count - mean * mean, 0.0)

    if timestamps.size:
        time_seconds = (timestamps - timestamps[0]).astype(np.float64) / 1000.0
    else:
        time_seconds = np.empty(0, dtype=np.float64)
    time_sum = _prefix(time_seconds)
    time_square_sum = _prefix(time_seconds * time_seconds)
    time_value_sum = _prefix(time_seconds * values)
    sums_t = time_sum[right] - time_sum[left]
    sums_tt = time_square_sum[right] - time_square_sum[left]
    sums_ty = time_value_sum[right] - time_value_sum[left]
    denominator = safe_count * sums_tt - sums_t * sums_t
    numerator = safe_count * sums_ty - sums_t * sums
    slope = np.zeros(count.shape, dtype=np.float64)
    np.divide(numerator, denominator, out=slope, where=np.abs(denominator) > 1e-12)
    duration_seconds = float(np.max(ends_ms - starts_ms) / 1000.0)
    return _SegmentBatch(
        count=count,
        mean=mean,
        std=np.sqrt(variance),
        slope_per_second=slope,
        duration_seconds=max(duration_seconds, 1e-3),
    )


def _contrast_many(
    left: _SegmentBatch,
    right: _SegmentBatch,
    min_samples: int,
    score_cap: float,
) -> MetricContrastBatch:
    observed = (left.count >= min_samples) & (right.count >= min_samples)
    total = np.maximum(left.count + right.count, 1)
    pooled_variance = (
        left.count * (left.std ** 2 + left.mean ** 2)
        + right.count * (right.std ** 2 + right.mean ** 2)
    ) / total - (
        (left.count * left.mean + right.count * right.mean) / total
    ) ** 2
    pooled_scale = np.sqrt(np.maximum(pooled_variance, 0.0))
    epsilon = 1e-6 * np.maximum.reduce(
        (np.abs(left.mean), np.abs(right.mean), np.ones_like(left.mean))
    )
    signed_mean = (right.mean - left.mean) / (pooled_scale + epsilon)
    scale_epsilon = 1e-6 * np.maximum.reduce(
        (left.std, right.std, np.ones_like(left.std))
    )
    log_scale = np.log(
        (right.std + scale_epsilon) / (left.std + scale_epsilon)
    )
    comparable_duration = min(left.duration_seconds, right.duration_seconds)
    slope = (
        (right.slope_per_second - left.slope_per_second)
        * comparable_duration
        / (pooled_scale + epsilon)
    )
    clipped_values = []
    for raw in (signed_mean, log_scale, slope):
        clipped = np.clip(raw, -score_cap, score_cap)
        clipped[~observed] = 0.0
        clipped_values.append(clipped)
    return MetricContrastBatch(
        observed,
        clipped_values[0],
        clipped_values[1],
        clipped_values[2],
    )


def metric_series_contrasts_many(
    timestamps_ms: Sequence[int],
    values: Sequence[float],
    anchors_ms: Sequence[int],
    pre_ms: int = 300000,
    post_ms: int = 300000,
    onset_seconds: Sequence[int] = DEFAULT_ONSET_SECONDS,
    min_samples: int = 2,
    score_cap: float = 20.0,
) -> Mapping[str, MetricContrastBatch]:
    """Summarize one raw metric series for many anchors using prefix statistics."""

    if pre_ms <= 0 or post_ms <= 0 or min_samples < 1 or score_cap <= 0:
        raise ValueError("metric summary parameters must be positive")
    onsets = tuple(int(value) for value in onset_seconds)
    if tuple(sorted(onsets)) != DEFAULT_ONSET_SECONDS:
        raise ValueError("P2 metric schema requires onset boundaries 30/60/120 s")
    timestamps, numeric = _deduplicate(timestamps_ms, values)
    anchors = np.asarray(anchors_ms, dtype=np.int64)
    pre = _segment_many(timestamps, numeric, anchors - pre_ms, anchors)
    post = _segment_many(timestamps, numeric, anchors, anchors + post_ms)
    contrasts: Dict[str, MetricContrastBatch] = {
        "whole": _contrast_many(pre, post, min_samples, score_cap)
    }
    for onset_seconds_value in onsets:
        onset_end = anchors + onset_seconds_value * 1000
        onset = _segment_many(timestamps, numeric, anchors, onset_end)
        impact = _segment_many(timestamps, numeric, onset_end, anchors + post_ms)
        prefix = "stage{}".format(onset_seconds_value)
        contrasts[prefix + ".pre_onset"] = _contrast_many(
            pre, onset, min_samples, score_cap
        )
        contrasts[prefix + ".pre_impact"] = _contrast_many(
            pre, impact, min_samples, score_cap
        )
        contrasts[prefix + ".onset_impact"] = _contrast_many(
            onset, impact, min_samples, score_cap
        )
    return contrasts


def metric_series_contrasts(
    timestamps_ms: Sequence[int],
    values: Sequence[float],
    anchor_ms: int,
    pre_ms: int = 300000,
    post_ms: int = 300000,
    onset_seconds: Sequence[int] = DEFAULT_ONSET_SECONDS,
    min_samples: int = 2,
    score_cap: float = 20.0,
) -> Mapping[str, MetricContrast]:
    """Scalar convenience wrapper with the same implementation as batch extraction."""

    batches = metric_series_contrasts_many(
        timestamps_ms,
        values,
        (anchor_ms,),
        pre_ms=pre_ms,
        post_ms=post_ms,
        onset_seconds=onset_seconds,
        min_samples=min_samples,
        score_cap=score_cap,
    )
    return {
        name: MetricContrast(
            observed=bool(batch.observed[0]),
            signed_mean_shift=float(batch.signed_mean_shift[0]),
            log_scale_ratio=float(batch.log_scale_ratio[0]),
            slope_shift=float(batch.slope_shift[0]),
        )
        for name, batch in batches.items()
    }


def _top_mean(values: np.ndarray, count: int) -> float:
    if values.size == 0:
        return 0.0
    selected_count = min(count, values.size)
    selected = np.partition(values, values.size - selected_count)[-selected_count:]
    return float(selected.mean())


def aggregate_metric_contrasts(
    series_contrasts: Sequence[Mapping[str, MetricContrast]],
    total_series_count: int,
) -> Tuple[Tuple[str, ...], Tuple[float, ...], Tuple[bool, ...]]:
    """Aggregate arbitrary metric dimensions into a fixed service vector."""

    if total_series_count < len(series_contrasts) or total_series_count < 0:
        raise ValueError("total_series_count must cover supplied series")
    expected = set(_contrast_names())
    if any(set(series) != expected for series in series_contrasts):
        raise ValueError("series contrasts do not use the default fixed schema")

    values = []
    masks = []
    for contrast_name in _contrast_names():
        contrasts = [
            series[contrast_name]
            for series in series_contrasts
            if series[contrast_name].observed
        ]
        valid_count = len(contrasts)
        values.extend(
            (
                float(valid_count),
                float(valid_count / total_series_count)
                if total_series_count
                else 0.0,
            )
        )
        masks.extend((True, True))
        for field in CONTRAST_FIELDS:
            signed = np.asarray(
                [float(getattr(contrast, field)) for contrast in contrasts],
                dtype=np.float64,
            )
            absolute = np.abs(signed)
            values.extend(
                (
                    float(absolute.max()) if absolute.size else 0.0,
                    _top_mean(absolute, 3),
                    _top_mean(absolute, 5),
                    float(signed.mean()) if signed.size else 0.0,
                    float((signed > 0).mean()) if signed.size else 0.0,
                )
            )
            masks.extend((bool(signed.size),) * len(AGGREGATE_STATISTICS))
    return metric_feature_names(), tuple(values), tuple(masks)


class MetricFeatureAccumulator:
    """Stream exact top-k/mean metric aggregates without retaining raw scores."""

    def __init__(self, case_count: int, services: Sequence[str]):
        if case_count < 1 or not services or len(set(services)) != len(services):
            raise ValueError("accumulator requires cases and unique services")
        self.case_count = case_count
        self.services = tuple(services)
        self.service_index = {
            service: index for index, service in enumerate(self.services)
        }
        shape = (case_count, len(self.services), len(_contrast_names()))
        self.total_series = np.zeros(shape[:2], dtype=np.int32)
        self.valid_counts = np.zeros(shape, dtype=np.int32)
        self.signed_sums = np.zeros(
            shape + (len(CONTRAST_FIELDS),), dtype=np.float64
        )
        self.positive_counts = np.zeros(
            shape + (len(CONTRAST_FIELDS),), dtype=np.int32
        )
        self.top_absolute = np.full(
            shape + (len(CONTRAST_FIELDS), 5),
            -np.inf,
            dtype=np.float32,
        )

    def update(
        self,
        service: str,
        batches: Mapping[str, MetricContrastBatch],
        case_mask: Optional[Sequence[bool]] = None,
    ) -> None:
        if service not in self.service_index:
            raise ValueError("unknown service: {}".format(service))
        if set(batches) != set(_contrast_names()):
            raise ValueError("metric contrast batch schema mismatch")
        mask = (
            np.ones(self.case_count, dtype=bool)
            if case_mask is None
            else np.asarray(case_mask, dtype=bool)
        )
        if mask.shape != (self.case_count,):
            raise ValueError("case mask shape mismatch")
        service_index = self.service_index[service]
        self.total_series[mask, service_index] += 1
        row_indices = np.arange(self.case_count)
        for contrast_index, contrast_name in enumerate(_contrast_names()):
            batch = batches[contrast_name]
            if batch.observed.shape != (self.case_count,):
                raise ValueError("metric contrast batch does not align with cases")
            valid = batch.observed & mask
            self.valid_counts[valid, service_index, contrast_index] += 1
            for field_index, field in enumerate(CONTRAST_FIELDS):
                field_values = np.asarray(getattr(batch, field), dtype=np.float64)
                if field_values.shape != (self.case_count,):
                    raise ValueError("metric field batch does not align with cases")
                self.signed_sums[
                    valid, service_index, contrast_index, field_index
                ] += field_values[valid]
                self.positive_counts[
                    valid, service_index, contrast_index, field_index
                ] += (field_values[valid] > 0).astype(np.int32)
                slots = self.top_absolute[
                    :, service_index, contrast_index, field_index, :
                ]
                minimum_slot = np.argmin(slots, axis=1)
                absolute = np.abs(field_values)
                replace = valid & (absolute > slots[row_indices, minimum_slot])
                slots[row_indices[replace], minimum_slot[replace]] = absolute[
                    replace
                ].astype(np.float32)

    def update_case(
        self,
        service: str,
        case_index: int,
        contrasts: Mapping[str, MetricContrast],
    ) -> None:
        """Add one raw series for one case without allocating a case-sized batch."""

        if service not in self.service_index:
            raise ValueError("unknown service: {}".format(service))
        if not 0 <= case_index < self.case_count:
            raise IndexError("case index is out of range")
        if set(contrasts) != set(_contrast_names()):
            raise ValueError("metric contrast schema mismatch")
        service_index = self.service_index[service]
        self.total_series[case_index, service_index] += 1
        for contrast_index, contrast_name in enumerate(_contrast_names()):
            contrast = contrasts[contrast_name]
            if not contrast.observed:
                continue
            self.valid_counts[case_index, service_index, contrast_index] += 1
            for field_index, field in enumerate(CONTRAST_FIELDS):
                field_value = float(getattr(contrast, field))
                self.signed_sums[
                    case_index, service_index, contrast_index, field_index
                ] += field_value
                self.positive_counts[
                    case_index, service_index, contrast_index, field_index
                ] += int(field_value > 0)
                slots = self.top_absolute[
                    case_index, service_index, contrast_index, field_index, :
                ]
                minimum_slot = int(np.argmin(slots))
                absolute = abs(field_value)
                if absolute > slots[minimum_slot]:
                    slots[minimum_slot] = np.float32(absolute)

    def feature_vector(
        self, case_index: int, service: str
    ) -> Tuple[Tuple[str, ...], Tuple[float, ...], Tuple[bool, ...]]:
        if not 0 <= case_index < self.case_count:
            raise IndexError("case index is out of range")
        service_index = self.service_index[service]
        values = []
        masks = []
        total = int(self.total_series[case_index, service_index])
        for contrast_index, _ in enumerate(_contrast_names()):
            count = int(
                self.valid_counts[case_index, service_index, contrast_index]
            )
            values.extend((float(count), float(count / total) if total else 0.0))
            masks.extend((True, True))
            for field_index, _ in enumerate(CONTRAST_FIELDS):
                slots = self.top_absolute[
                    case_index, service_index, contrast_index, field_index, :
                ]
                finite = slots[np.isfinite(slots)].astype(np.float64)
                if count:
                    signed_mean = (
                        self.signed_sums[
                            case_index, service_index, contrast_index, field_index
                        ]
                        / count
                    )
                    positive_fraction = (
                        self.positive_counts[
                            case_index, service_index, contrast_index, field_index
                        ]
                        / count
                    )
                    field_values = (
                        float(finite.max()),
                        _top_mean(finite, 3),
                        _top_mean(finite, 5),
                        float(signed_mean),
                        float(positive_fraction),
                    )
                else:
                    field_values = (0.0,) * len(AGGREGATE_STATISTICS)
                values.extend(field_values)
                masks.extend((bool(count),) * len(AGGREGATE_STATISTICS))
        return metric_feature_names(), tuple(values), tuple(masks)
