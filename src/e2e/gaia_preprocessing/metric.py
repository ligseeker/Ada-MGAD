"""Leakage-safe Metric transformation primitives for GAIA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricScaler:
    lower: float
    upper: float
    median: float

    def as_dict(self):
        return {
            "lower": self.lower,
            "upper": self.upper,
            "median": self.median,
        }


def fit_metric_scaler(
    train_values: np.ndarray,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> MetricScaler:
    """Fit the frozen robust min-max parameters from Train values only."""

    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    values = np.asarray(train_values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot fit a metric scaler without finite Train values")
    lower, upper = np.quantile(finite, [lower_quantile, upper_quantile])
    median = float(np.median(finite))
    if not float(upper) > float(lower):
        raise ValueError("metric scaler has zero dynamic range")
    return MetricScaler(lower=float(lower), upper=float(upper), median=median)


def apply_metric_scaler(
    values: np.ndarray,
    scaler: MetricScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """Impute with the frozen Train median, clip, and scale to [0, 1]."""

    if not isinstance(scaler, MetricScaler):
        raise TypeError("scaler must be MetricScaler")
    if not scaler.upper > scaler.lower:
        raise ValueError("metric scaler upper must exceed lower")
    source = np.asarray(values, dtype=np.float64)
    observed = np.isfinite(source)
    imputed = np.where(observed, source, scaler.median)
    scaled = (np.clip(imputed, scaler.lower, scaler.upper) - scaler.lower) / (
        scaler.upper - scaler.lower
    )
    return scaled.astype(np.float32), observed


def aggregate_multicore(core_values: np.ndarray):
    """Return the two approved core summaries without adding a std channel."""

    values = np.asarray(core_values, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] == 0:
        raise ValueError("core_values must have a non-empty final core dimension")
    finite = np.isfinite(values)
    counts = finite.sum(axis=-1)
    means = np.full(values.shape[:-1], np.nan, dtype=np.float64)
    np.divide(
        np.where(finite, values, 0.0).sum(axis=-1),
        counts,
        out=means,
        where=counts > 0,
    )
    maxima = np.max(np.where(finite, values, -np.inf), axis=-1)
    maxima[counts == 0] = np.nan
    return {"mean": means, "max": maxima}


def assemble_metric_tensor(
    *,
    global_values: np.ndarray,
    global_observed: np.ndarray,
    host_values: np.ndarray,
    host_observed: np.ndarray,
    host_applicable: np.ndarray,
    neutral_value: float,
) -> np.ndarray:
    """Assemble uniform slots plus three aggregate observability channels."""

    global_array = np.asarray(global_values, dtype=np.float32)
    global_mask = np.asarray(global_observed, dtype=bool)
    host_array = np.asarray(host_values, dtype=np.float32)
    host_mask = np.asarray(host_observed, dtype=bool)
    applicable = np.asarray(host_applicable, dtype=bool)
    if global_array.ndim != 3 or host_array.ndim != 3:
        raise ValueError("Metric arrays must have shape [time, node, feature]")
    if global_array.shape != global_mask.shape or host_array.shape != host_mask.shape:
        raise ValueError("Metric observed masks must match their value arrays")
    if global_array.shape[:2] != host_array.shape[:2]:
        raise ValueError("global and host Metric arrays must share time/node axes")
    if applicable.shape != (global_array.shape[1],):
        raise ValueError("host_applicable must have one value per node")
    if global_array.shape[-1] == 0 or host_array.shape[-1] == 0:
        raise ValueError("global and host Metric groups must both be non-empty")
    if not np.isfinite(neutral_value):
        raise ValueError("neutral_value must be finite")

    global_finite = np.isfinite(global_array)
    host_finite = np.isfinite(host_array)
    global_mask = global_mask & global_finite
    host_mask = host_mask & host_finite & applicable[None, :, None]
    global_array = np.where(global_finite, global_array, neutral_value)
    host_array = np.where(
        host_finite & applicable[None, :, None], host_array, neutral_value
    )
    global_fraction = global_mask.mean(axis=-1, keepdims=True, dtype=np.float64)
    host_fraction = host_mask.mean(axis=-1, keepdims=True, dtype=np.float64)
    applicability = np.broadcast_to(
        applicable[None, :, None], global_fraction.shape
    ).astype(np.float32)
    return np.concatenate(
        (
            global_array,
            host_array,
            global_fraction.astype(np.float32),
            applicability,
            host_fraction.astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32)


def derive_reset_aware_rate(
    timestamps_ms: np.ndarray,
    values: np.ndarray,
    *,
    max_gap_ms: int,
) -> np.ndarray:
    """Convert a cumulative counter to per-second rate.

    Resets, non-increasing timestamps, non-finite pairs, and intervals longer
    than the frozen maximum gap remain missing instead of becoming spikes.
    """

    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    counters = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or counters.ndim != 1 or timestamps.shape != counters.shape:
        raise ValueError("timestamps_ms and values must be equal-length 1D arrays")
    if max_gap_ms <= 0:
        raise ValueError("max_gap_ms must be positive")

    result = np.full(counters.shape, np.nan, dtype=np.float64)
    if counters.size < 2:
        return result
    elapsed_ms = np.diff(timestamps)
    delta = np.diff(counters)
    valid = (
        np.isfinite(counters[:-1])
        & np.isfinite(counters[1:])
        & (elapsed_ms > 0)
        & (elapsed_ms <= max_gap_ms)
        & (delta >= 0)
    )
    result[1:][valid] = delta[valid] / (elapsed_ms[valid] / 1000.0)
    return result


def forward_fill_by_age(
    timestamps_ms: np.ndarray,
    values: np.ndarray,
    *,
    max_age_ms: int,
    split_slices: Sequence[slice],
) -> np.ndarray:
    """Forward-fill within explicit split boundaries and a wall-clock age."""

    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    source = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or source.ndim != 1 or timestamps.shape != source.shape:
        raise ValueError("timestamps_ms and values must be equal-length 1D arrays")
    if max_age_ms <= 0:
        raise ValueError("max_age_ms must be positive")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_ms must be strictly increasing")

    result = source.copy()
    covered = np.zeros(source.size, dtype=bool)
    for split_slice in split_slices:
        if not isinstance(split_slice, slice) or split_slice.step not in (None, 1):
            raise ValueError("split_slices must contain contiguous slices")
        start = 0 if split_slice.start is None else split_slice.start
        stop = source.size if split_slice.stop is None else split_slice.stop
        if start < 0 or stop < start or stop > source.size or covered[start:stop].any():
            raise ValueError("split_slices must be non-overlapping and in bounds")
        covered[start:stop] = True
        last_index = None
        for index in range(start, stop):
            if np.isfinite(source[index]):
                last_index = index
            elif last_index is not None and timestamps[index] - timestamps[last_index] <= max_age_ms:
                result[index] = source[last_index]
    return result
