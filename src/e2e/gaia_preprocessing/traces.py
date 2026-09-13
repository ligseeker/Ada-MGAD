"""Split-local Trace parent resolution and 8D aggregation for GAIA."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .schema import TRACE_SLOTS, TRACE_STATUSES


@dataclass(frozen=True)
class TraceAggregation:
    features: np.ndarray
    diagnostics: Mapping[str, int]
    observed_directed_edges: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class FrozenTraceScalers:
    scales: Mapping[str, float]
    fallback_sources: Mapping[str, str]


def _as_int(value: Any):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def aggregate_trace_split(
    rows: Iterable[Mapping[str, Any]],
    *,
    services: Sequence[str],
    grid_ms: np.ndarray,
    split_start_ms: int,
    split_end_ms: int,
    frozen_directed_edges: Sequence[Tuple[str, str]],
) -> TraceAggregation:
    """Resolve and aggregate one split without consulting any other split.

    Input rows use millisecond timestamps. The returned feature order is the
    fixed status-major ``count, mean_latency_seconds`` 8D representation.
    Scaling is a separate frozen-schema operation.
    """

    service_names = tuple(services)
    if len(service_names) != len(set(service_names)) or not service_names:
        raise ValueError("services must be unique and non-empty")
    service_index = {service: index for index, service in enumerate(service_names)}
    grid = np.asarray(grid_ms, dtype=np.int64)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid_ms must be a strictly increasing 1D array")
    steps = np.diff(grid)
    if np.any(steps != steps[0]):
        raise ValueError("grid_ms must have a fixed interval")
    if split_end_ms <= split_start_ms:
        raise ValueError("split_end_ms must exceed split_start_ms")

    frozen_edges = {tuple(edge) for edge in frozen_directed_edges}
    if any(
        len(edge) != 2
        or edge[0] not in service_index
        or edge[1] not in service_index
        or edge[0] == edge[1]
        for edge in frozen_edges
    ):
        raise ValueError("frozen_directed_edges must contain directed non-self edges")

    diagnostics = {
        "rows_in_split": 0,
        "invalid_rows": 0,
        "root_rows": 0,
        "unmatched_parent_rows": 0,
        "ambiguous_parent_rows": 0,
        "self_loop_rows": 0,
        "unknown_service_rows": 0,
        "unknown_status_rows": 0,
        "matched_cross_service_rows": 0,
        "unseen_directed_edge_rows": 0,
    }
    selected = []
    for raw in rows:
        end_ms = _as_int(raw.get("end_time_ms"))
        if end_ms is None or not split_start_ms <= end_ms < split_end_ms:
            continue
        selected.append(raw)
        diagnostics["rows_in_split"] += 1

    parent_services: Dict[Tuple[str, str], set] = {}
    for raw in selected:
        trace_id = str(raw.get("trace_id", ""))
        span_id = str(raw.get("span_id", ""))
        service = str(raw.get("service", ""))
        if trace_id and span_id and service in service_index:
            parent_services.setdefault((trace_id, span_id), set()).add(service)

    counts = np.zeros(
        (len(grid), len(service_names), len(service_names), len(TRACE_STATUSES)),
        dtype=np.float64,
    )
    duration_sums = np.zeros_like(counts)
    status_index = {status: index for index, status in enumerate(TRACE_STATUSES)}
    observed_edges = set()
    bin_ms = int(steps[0])

    for raw in selected:
        parent_id = str(raw.get("parent_id", ""))
        if not parent_id:
            diagnostics["root_rows"] += 1
            continue
        trace_id = str(raw.get("trace_id", ""))
        destination = str(raw.get("service", ""))
        if destination not in service_index:
            diagnostics["unknown_service_rows"] += 1
            continue
        matches = parent_services.get((trace_id, parent_id), set())
        if not matches:
            diagnostics["unmatched_parent_rows"] += 1
            continue
        if len(matches) != 1:
            diagnostics["ambiguous_parent_rows"] += 1
            continue
        source = next(iter(matches))
        if source == destination:
            diagnostics["self_loop_rows"] += 1
            continue

        status = str(raw.get("status_code", "")).strip()
        if status not in status_index:
            diagnostics["unknown_status_rows"] += 1
            continue
        start_ms = _as_int(raw.get("start_time_ms"))
        end_ms = _as_int(raw.get("end_time_ms"))
        if start_ms is None or end_ms is None or end_ms < start_ms:
            diagnostics["invalid_rows"] += 1
            continue
        position = int(np.searchsorted(grid, end_ms, side="right") - 1)
        if (
            position < 0
            or position >= len(grid)
            or not grid[position] <= end_ms < grid[position] + bin_ms
        ):
            diagnostics["invalid_rows"] += 1
            continue

        source_index = service_index[source]
        destination_index = service_index[destination]
        status_position = status_index[status]
        counts[position, source_index, destination_index, status_position] += 1.0
        duration_sums[position, source_index, destination_index, status_position] += (
            end_ms - start_ms
        ) / 1000.0
        edge = (source, destination)
        observed_edges.add(edge)
        diagnostics["matched_cross_service_rows"] += 1
        if edge not in frozen_edges:
            diagnostics["unseen_directed_edge_rows"] += 1

    features = np.zeros(
        (len(grid), len(service_names), len(service_names), len(TRACE_SLOTS)),
        dtype=np.float32,
    )
    for status_position in range(len(TRACE_STATUSES)):
        count_position = status_position * 2
        mean_position = count_position + 1
        status_counts = counts[..., status_position]
        status_durations = duration_sums[..., status_position]
        features[..., count_position] = status_counts.astype(np.float32)
        np.divide(
            status_durations,
            status_counts,
            out=features[..., mean_position],
            where=status_counts > 0,
        )

    return TraceAggregation(
        features=features,
        diagnostics=diagnostics,
        observed_directed_edges=tuple(sorted(observed_edges)),
    )


def _trace_scale_key(source: str, destination: str, slot: str) -> str:
    return "{}->{}::{}".format(source, destination, slot)


def _supported_log_q99(values: np.ndarray, min_positive_bins: int):
    positive = values[values > 0]
    if positive.size < min_positive_bins:
        return None
    return float(np.quantile(np.log1p(positive), 0.99))


def fit_trace_scalers(
    train_features: np.ndarray,
    *,
    services: Sequence[str],
    frozen_directed_edges: Sequence[Tuple[str, str]],
    min_positive_bins: int,
) -> FrozenTraceScalers:
    """Fit edge scales with status/statistic Train-only fallback pools."""

    values = np.asarray(train_features, dtype=np.float64)
    service_names = tuple(services)
    if values.ndim != 4 or values.shape[1:3] != (len(service_names), len(service_names)):
        raise ValueError("train_features must have shape [time, node, node, 8]")
    if values.shape[-1] != len(TRACE_SLOTS):
        raise ValueError("Trace feature dimension must be 8")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Trace Train features must be finite and non-negative")
    if min_positive_bins < 1:
        raise ValueError("min_positive_bins must be positive")
    service_index = {service: index for index, service in enumerate(service_names)}
    edges = tuple(tuple(edge) for edge in frozen_directed_edges)
    if len(edges) != len(set(edges)):
        raise ValueError("frozen_directed_edges contains duplicates")
    if any(
        len(edge) != 2
        or edge[0] not in service_index
        or edge[1] not in service_index
        or edge[0] == edge[1]
        for edge in edges
    ):
        raise ValueError("frozen_directed_edges contains an invalid edge")

    status_pools = {}
    statistic_pools = {"count": [], "mean_latency": []}
    for slot_index, slot in enumerate(TRACE_SLOTS):
        edge_parts = [
            values[:, service_index[source], service_index[destination], slot_index]
            for source, destination in edges
        ]
        pooled = np.concatenate(edge_parts) if edge_parts else np.empty(0)
        status_pools[slot] = _supported_log_q99(pooled, min_positive_bins)
        statistic = "mean_latency" if "mean_latency" in slot else "count"
        statistic_pools[statistic].append(pooled)
    statistic_scales = {
        statistic: _supported_log_q99(
            np.concatenate(parts) if parts else np.empty(0), min_positive_bins
        )
        for statistic, parts in statistic_pools.items()
    }

    scales = {}
    sources = {}
    for source, destination in edges:
        source_index = service_index[source]
        destination_index = service_index[destination]
        for slot_index, slot in enumerate(TRACE_SLOTS):
            key = _trace_scale_key(source, destination, slot)
            direct = _supported_log_q99(
                values[:, source_index, destination_index, slot_index],
                min_positive_bins,
            )
            if direct is not None:
                scales[key] = direct
                sources[key] = key
            elif status_pools[slot] is not None:
                scales[key] = status_pools[slot]
                sources[key] = "__status_pool__::{}".format(slot)
            else:
                statistic = "mean_latency" if "mean_latency" in slot else "count"
                if statistic_scales[statistic] is not None:
                    scales[key] = statistic_scales[statistic]
                    sources[key] = "__statistic_pool__::{}".format(statistic)
                else:
                    scales[key] = 1.0
                    sources[key] = "__fixed_1__"
    return FrozenTraceScalers(
        scales=MappingProxyType(scales),
        fallback_sources=MappingProxyType(sources),
    )


def apply_trace_scalers(
    features: np.ndarray,
    *,
    services: Sequence[str],
    frozen_directed_edges: Sequence[Tuple[str, str]],
    scales: Mapping[str, float],
) -> np.ndarray:
    """Apply frozen log1p scales to the graph's directed Trace features."""

    values = np.asarray(features, dtype=np.float64)
    service_names = tuple(services)
    if values.ndim != 4 or values.shape[1:3] != (len(service_names), len(service_names)):
        raise ValueError("features must have shape [time, node, node, 8]")
    if values.shape[-1] != len(TRACE_SLOTS):
        raise ValueError("Trace feature dimension must be 8")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Trace features must be finite and non-negative")
    service_index = {service: index for index, service in enumerate(service_names)}
    result = np.zeros(values.shape, dtype=np.float32)
    for source, destination in frozen_directed_edges:
        source_index = service_index[source]
        destination_index = service_index[destination]
        for slot_index, slot in enumerate(TRACE_SLOTS):
            key = _trace_scale_key(source, destination, slot)
            scale = float(scales[key])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("invalid frozen Trace scale for {}".format(key))
            result[:, source_index, destination_index, slot_index] = np.clip(
                np.log1p(values[:, source_index, destination_index, slot_index]) / scale,
                0.0,
                1.0,
            )
    return result
