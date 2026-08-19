"""Label-free whole/staged summaries for raw log and trace event streams."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_EVENT_ONSET_SECONDS = (30, 60, 120)
RE2_TRACE_SERVICE_ALIASES = {"frontendservice": "frontend"}


@dataclass(frozen=True)
class _Segment:
    start_ms: int
    end_ms: int
    left: int
    right: int

    @property
    def duration_seconds(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0


def normalize_trace_service(
    raw_service: str,
    candidate_services: Sequence[str],
    aliases: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Map only explicit raw aliases into the legal candidate service set."""

    candidates = set(candidate_services)
    if raw_service in candidates:
        return raw_service
    mapped = dict(aliases or {}).get(raw_service)
    return mapped if mapped in candidates else None


def _comparison_intervals(anchor_ms: int, pre_ms: int, post_ms: int, onsets):
    result = [("whole", anchor_ms - pre_ms, anchor_ms, anchor_ms, anchor_ms + post_ms)]
    for onset in onsets:
        onset_end = anchor_ms + onset * 1000
        result.extend(
            (
                (
                    "stage{}.pre_onset".format(onset),
                    anchor_ms - pre_ms,
                    anchor_ms,
                    anchor_ms,
                    onset_end,
                ),
                (
                    "stage{}.pre_impact".format(onset),
                    anchor_ms - pre_ms,
                    anchor_ms,
                    onset_end,
                    anchor_ms + post_ms,
                ),
                (
                    "stage{}.onset_impact".format(onset),
                    anchor_ms,
                    onset_end,
                    onset_end,
                    anchor_ms + post_ms,
                ),
            )
        )
    return tuple(result)


def _validate_common(timestamps_ms, anchor_ms, pre_ms, post_ms, onset_seconds):
    if pre_ms <= 0 or post_ms <= 0:
        raise ValueError("event windows must be positive")
    onsets = tuple(int(value) for value in onset_seconds)
    if tuple(sorted(set(onsets))) != DEFAULT_EVENT_ONSET_SECONDS:
        raise ValueError("event schema requires onset boundaries 30/60/120 s")
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if timestamps.ndim != 1:
        raise ValueError("event timestamps must be one-dimensional")
    order = np.argsort(timestamps, kind="mergesort")
    return timestamps[order], order, _comparison_intervals(
        int(anchor_ms), int(pre_ms), int(post_ms), onsets
    )


def _validate_parameters(pre_ms, post_ms, onset_seconds, score_cap):
    if pre_ms <= 0 or post_ms <= 0:
        raise ValueError("event windows must be positive")
    onsets = tuple(int(value) for value in onset_seconds)
    if tuple(sorted(set(onsets))) != DEFAULT_EVENT_ONSET_SECONDS:
        raise ValueError("event schema requires onset boundaries 30/60/120 s")
    if score_cap <= 0:
        raise ValueError("event score cap must be positive")
    return onsets


def _prepare_timestamps(timestamps_ms):
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if timestamps.ndim != 1:
        raise ValueError("event timestamps must be one-dimensional")
    order = np.argsort(timestamps, kind="mergesort")
    return timestamps[order], order


def _segments(timestamps, interval):
    _, left_start, left_end, right_start, right_end = interval
    return (
        _Segment(
            left_start,
            left_end,
            int(np.searchsorted(timestamps, left_start, side="left")),
            int(np.searchsorted(timestamps, left_end, side="left")),
        ),
        _Segment(
            right_start,
            right_end,
            int(np.searchsorted(timestamps, right_start, side="left")),
            int(np.searchsorted(timestamps, right_end, side="left")),
        ),
    )


def _finite_aligned(values, order, expected_length, field_name):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected_length,):
        raise ValueError("{} must align with timestamps".format(field_name))
    return array[order]


def _binary_aligned(values, order, expected_length, field_name):
    array = _finite_aligned(values, order, expected_length, field_name)
    finite = array[np.isfinite(array)]
    if np.any((finite != 0.0) & (finite != 1.0)):
        raise ValueError("{} must contain only 0/1/NaN".format(field_name))
    return array


def _nonnegative_aligned(values, order, expected_length, field_name):
    array = _finite_aligned(values, order, expected_length, field_name)
    if np.any(array[np.isfinite(array)] < 0):
        raise ValueError("{} must be nonnegative or NaN".format(field_name))
    return array


def _text_aligned(values, order, expected_length, field_name):
    raw = np.asarray(values, dtype=object)
    if raw.shape != (expected_length,):
        raise ValueError("{} must align with timestamps".format(field_name))
    normalized = []
    for value in raw:
        missing = value is None
        if not missing:
            try:
                missing = bool(np.isnan(value))
            except (TypeError, ValueError):
                missing = False
        normalized.append("" if missing else str(value))
    return np.asarray(normalized, dtype=str)[order]


def _code_aligned(values, order, expected_length, field_name):
    array = np.asarray(values)
    if array.shape != (expected_length,) or not np.issubdtype(
        array.dtype, np.integer
    ):
        raise ValueError("{} must be aligned integer codes".format(field_name))
    result = array.astype(np.int64, copy=False)[order]
    if np.any(result < -1):
        raise ValueError("{} may use only -1 as a missing code".format(field_name))
    return result


def _fraction(values, segment):
    selected = values[segment.left : segment.right]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        return 0.0, False
    return float(finite.mean()), True


def _mean_std(values, segment):
    selected = values[segment.left : segment.right]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        return 0.0, 0.0, False
    return float(finite.mean()), float(finite.std()), True


def _log_rate_shift(left_count, left_duration, right_count, right_duration, cap):
    left_rate = (left_count + 0.5) / left_duration
    right_rate = (right_count + 0.5) / right_duration
    return float(np.clip(np.log(right_rate / left_rate), -cap, cap))


def _standardized_mean_shift(left_mean, left_std, right_mean, right_std, cap):
    scale = np.sqrt((left_std * left_std + right_std * right_std) / 2.0)
    epsilon = 1e-6 * max(abs(left_mean), abs(right_mean), 1.0)
    return float(np.clip((right_mean - left_mean) / (scale + epsilon), -cap, cap))


def _log_scale_shift(left, right, cap):
    epsilon = 1e-6 * max(abs(left), abs(right), 1.0)
    return float(np.clip(np.log((right + epsilon) / (left + epsilon)), -cap, cap))


def _active_fraction(timestamps, segment):
    selected = timestamps[segment.left : segment.right]
    if selected.size == 0:
        return 0.0
    active_seconds = np.unique(selected // 1000).size
    return float(min(active_seconds / segment.duration_seconds, 1.0))


@lru_cache(maxsize=1)
def log_feature_names() -> Tuple[str, ...]:
    fields = (
        "log_event_rate_ratio",
        "active_second_fraction_shift",
        "error_fraction_shift",
        "message_length_mean_shift",
        "message_length_log_scale_ratio",
    )
    names = []
    for interval in _comparison_intervals(0, 300000, 300000, DEFAULT_EVENT_ONSET_SECONDS):
        names.extend("{}.{}".format(interval[0], field) for field in fields)
    return tuple(names)


def _log_features_sorted(
    timestamps,
    errors,
    lengths,
    anchor_ms,
    entity_observed,
    pre_ms,
    post_ms,
    onset_seconds,
    score_cap,
):
    intervals = _comparison_intervals(
        int(anchor_ms), int(pre_ms), int(post_ms), onset_seconds
    )
    segment_cache = {}

    def summarize(segment):
        if segment not in segment_cache:
            error, error_ok = _fraction(errors, segment)
            mean, std, length_ok = _mean_std(lengths, segment)
            segment_cache[segment] = (
                segment.right - segment.left,
                _active_fraction(timestamps, segment),
                error,
                error_ok,
                mean,
                std,
                length_ok,
            )
        return segment_cache[segment]

    values = []
    masks = []
    for interval in intervals:
        left, right = _segments(timestamps, interval)
        (
            left_count,
            left_active,
            left_error,
            left_error_ok,
            left_mean,
            left_std,
            left_length_ok,
        ) = summarize(left)
        (
            right_count,
            right_active,
            right_error,
            right_error_ok,
            right_mean,
            right_std,
            right_length_ok,
        ) = summarize(right)
        block_values = (
            _log_rate_shift(
                left_count,
                left.duration_seconds,
                right_count,
                right.duration_seconds,
                score_cap,
            ),
            right_active - left_active,
            right_error - left_error,
            _standardized_mean_shift(
                left_mean, left_std, right_mean, right_std, score_cap
            ),
            _log_scale_shift(left_std, right_std, score_cap),
        )
        block_masks = (
            entity_observed,
            entity_observed,
            entity_observed and left_error_ok and right_error_ok,
            entity_observed and left_length_ok and right_length_ok,
            entity_observed and left_length_ok and right_length_ok,
        )
        values.extend(
            value if observed else 0.0
            for value, observed in zip(block_values, block_masks)
        )
        masks.extend(block_masks)
    return log_feature_names(), tuple(values), tuple(masks)


class PreparedLogStream:
    """Validated log arrays sorted once for many case-window queries."""

    def __init__(
        self,
        timestamps_ms: Sequence[int],
        error_flags: Sequence[float],
        message_lengths: Sequence[float],
        entity_observed: bool = True,
        pre_ms: int = 300000,
        post_ms: int = 300000,
        onset_seconds: Sequence[int] = DEFAULT_EVENT_ONSET_SECONDS,
        score_cap: float = 20.0,
    ):
        if not isinstance(entity_observed, bool):
            raise ValueError("log entity flag must be bool")
        self.onset_seconds = _validate_parameters(
            pre_ms, post_ms, onset_seconds, score_cap
        )
        self.pre_ms = int(pre_ms)
        self.post_ms = int(post_ms)
        self.score_cap = float(score_cap)
        self.entity_observed = entity_observed
        self.timestamps, order = _prepare_timestamps(timestamps_ms)
        self.errors = _binary_aligned(
            error_flags, order, len(order), "error flags"
        )
        self.lengths = _nonnegative_aligned(
            message_lengths, order, len(order), "message lengths"
        )

    def features(self, anchor_ms: int, entity_observed: Optional[bool] = None):
        observed = self.entity_observed if entity_observed is None else entity_observed
        if not isinstance(observed, bool):
            raise ValueError("log entity flag must be bool")
        return _log_features_sorted(
            self.timestamps,
            self.errors,
            self.lengths,
            anchor_ms,
            observed,
            self.pre_ms,
            self.post_ms,
            self.onset_seconds,
            self.score_cap,
        )


def log_stream_features(
    timestamps_ms: Sequence[int],
    error_flags: Sequence[float],
    message_lengths: Sequence[float],
    anchor_ms: int,
    entity_observed: bool = True,
    pre_ms: int = 300000,
    post_ms: int = 300000,
    onset_seconds: Sequence[int] = DEFAULT_EVENT_ONSET_SECONDS,
    score_cap: float = 20.0,
) -> Tuple[Tuple[str, ...], Tuple[float, ...], Tuple[bool, ...]]:
    """Compute raw, vocabulary-free log change features for one service."""

    stream = PreparedLogStream(
        timestamps_ms,
        error_flags,
        message_lengths,
        entity_observed=entity_observed,
        pre_ms=pre_ms,
        post_ms=post_ms,
        onset_seconds=onset_seconds,
        score_cap=score_cap,
    )
    return stream.features(anchor_ms)


@lru_cache(maxsize=1)
def trace_feature_names() -> Tuple[str, ...]:
    fields = (
        "span_rate_ratio",
        "error_fraction_shift",
        "duration_median_ratio",
        "duration_p90_ratio",
        "duration_p99_ratio",
        "unique_trace_rate_ratio",
        "unique_operation_rate_ratio",
        "parent_fraction_shift",
    )
    names = []
    for interval in _comparison_intervals(0, 300000, 300000, DEFAULT_EVENT_ONSET_SECONDS):
        names.extend("{}.{}".format(interval[0], field) for field in fields)
    return tuple(names)


def _quantiles(values, segment):
    selected = values[segment.left : segment.right]
    finite = selected[np.isfinite(selected) & (selected >= 0)]
    if finite.size == 0:
        return (0.0, 0.0, 0.0), False
    result = np.quantile(finite, (0.5, 0.9, 0.99))
    return tuple(float(value) for value in result), True


def _unique_rate(values, segment):
    selected = values[segment.left : segment.right]
    if np.issubdtype(selected.dtype, np.integer):
        valid = selected[selected >= 0]
    else:
        valid = selected[selected != ""]
    if valid.size == 0:
        return 0.0, False
    return float(np.unique(valid).size / segment.duration_seconds), True


def _trace_features_sorted(
    timestamps,
    durations,
    errors,
    traces,
    operations_array,
    parents,
    anchor_ms,
    entity_observed,
    pre_ms,
    post_ms,
    onset_seconds,
    score_cap,
):
    intervals = _comparison_intervals(
        int(anchor_ms), int(pre_ms), int(post_ms), onset_seconds
    )
    segment_cache = {}

    def summarize(segment):
        if segment not in segment_cache:
            error, error_ok = _fraction(errors, segment)
            duration, duration_ok = _quantiles(durations, segment)
            trace_rate, trace_ok = _unique_rate(traces, segment)
            operation_rate, operation_ok = _unique_rate(
                operations_array, segment
            )
            parent, parent_ok = _fraction(parents, segment)
            segment_cache[segment] = (
                segment.right - segment.left,
                error,
                error_ok,
                duration,
                duration_ok,
                trace_rate,
                trace_ok,
                operation_rate,
                operation_ok,
                parent,
                parent_ok,
            )
        return segment_cache[segment]

    values = []
    masks = []
    for interval in intervals:
        left, right = _segments(timestamps, interval)
        (
            left_count,
            left_error,
            left_error_ok,
            left_duration,
            left_duration_ok,
            left_trace_rate,
            left_trace_ok,
            left_operation_rate,
            left_operation_ok,
            left_parent,
            left_parent_ok,
        ) = summarize(left)
        (
            right_count,
            right_error,
            right_error_ok,
            right_duration,
            right_duration_ok,
            right_trace_rate,
            right_trace_ok,
            right_operation_rate,
            right_operation_ok,
            right_parent,
            right_parent_ok,
        ) = summarize(right)
        block_values = (
            _log_rate_shift(
                left_count,
                left.duration_seconds,
                right_count,
                right.duration_seconds,
                score_cap,
            ),
            right_error - left_error,
            *(
                _log_scale_shift(left_value, right_value, score_cap)
                for left_value, right_value in zip(left_duration, right_duration)
            ),
            _log_scale_shift(left_trace_rate, right_trace_rate, score_cap),
            _log_scale_shift(
                left_operation_rate, right_operation_rate, score_cap
            ),
            right_parent - left_parent,
        )
        block_masks = (
            entity_observed,
            entity_observed and left_error_ok and right_error_ok,
            entity_observed and left_duration_ok and right_duration_ok,
            entity_observed and left_duration_ok and right_duration_ok,
            entity_observed and left_duration_ok and right_duration_ok,
            entity_observed and left_trace_ok and right_trace_ok,
            entity_observed and left_operation_ok and right_operation_ok,
            entity_observed and left_parent_ok and right_parent_ok,
        )
        values.extend(
            value if observed else 0.0
            for value, observed in zip(block_values, block_masks)
        )
        masks.extend(block_masks)
    return trace_feature_names(), tuple(values), tuple(masks)


class PreparedTraceStream:
    """Validated trace arrays sorted once for many case-window queries."""

    def __init__(
        self,
        timestamps_ms: Sequence[int],
        durations_seconds: Sequence[float],
        error_flags: Sequence[float],
        trace_ids,
        operations,
        parent_present: Sequence[float],
        entity_observed: bool = True,
        pre_ms: int = 300000,
        post_ms: int = 300000,
        onset_seconds: Sequence[int] = DEFAULT_EVENT_ONSET_SECONDS,
        score_cap: float = 20.0,
        identifier_codes: bool = False,
    ):
        if not isinstance(entity_observed, bool):
            raise ValueError("trace entity flag must be bool")
        if not isinstance(identifier_codes, bool):
            raise ValueError("identifier_codes must be bool")
        self.onset_seconds = _validate_parameters(
            pre_ms, post_ms, onset_seconds, score_cap
        )
        self.pre_ms = int(pre_ms)
        self.post_ms = int(post_ms)
        self.score_cap = float(score_cap)
        self.entity_observed = entity_observed
        self.timestamps, order = _prepare_timestamps(timestamps_ms)
        length = len(order)
        self.durations = _nonnegative_aligned(
            durations_seconds, order, length, "trace durations"
        )
        self.errors = _binary_aligned(
            error_flags, order, length, "trace error flags"
        )
        self.parents = _binary_aligned(
            parent_present, order, length, "parent-presence flags"
        )
        align = _code_aligned if identifier_codes else _text_aligned
        self.traces = align(trace_ids, order, length, "trace identifiers")
        self.operations = align(operations, order, length, "trace operations")

    @classmethod
    def from_identifier_codes(cls, *args, **kwargs):
        kwargs["identifier_codes"] = True
        return cls(*args, **kwargs)

    def features(self, anchor_ms: int, entity_observed: Optional[bool] = None):
        observed = self.entity_observed if entity_observed is None else entity_observed
        if not isinstance(observed, bool):
            raise ValueError("trace entity flag must be bool")
        return _trace_features_sorted(
            self.timestamps,
            self.durations,
            self.errors,
            self.traces,
            self.operations,
            self.parents,
            anchor_ms,
            observed,
            self.pre_ms,
            self.post_ms,
            self.onset_seconds,
            self.score_cap,
        )


def trace_stream_features(
    timestamps_ms: Sequence[int],
    durations_seconds: Sequence[float],
    error_flags: Sequence[float],
    trace_ids: Sequence[str],
    operations: Sequence[str],
    parent_present: Sequence[float],
    anchor_ms: int,
    entity_observed: bool = True,
    pre_ms: int = 300000,
    post_ms: int = 300000,
    onset_seconds: Sequence[int] = DEFAULT_EVENT_ONSET_SECONDS,
    score_cap: float = 20.0,
) -> Tuple[Tuple[str, ...], Tuple[float, ...], Tuple[bool, ...]]:
    """Compute non-graph trace change features for one service."""

    stream = PreparedTraceStream(
        timestamps_ms,
        durations_seconds,
        error_flags,
        trace_ids,
        operations,
        parent_present,
        entity_observed=entity_observed,
        pre_ms=pre_ms,
        post_ms=post_ms,
        onset_seconds=onset_seconds,
        score_cap=score_cap,
    )
    return stream.features(anchor_ms)
