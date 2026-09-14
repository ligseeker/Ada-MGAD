"""Raw GAIA readers for the V2 Ada-MGAD preprocessing path.

This module is deliberately independent from :mod:`src.e2e.ad_preprocess`.
It contains the file boundary of the V2 protocol: all selection, vocabulary,
scaler and graph decisions are fitted from an explicit Train interval and the
returned objects are immutable descriptions of those decisions.  The readers
are chunked; no Test row is inspected while a ``fit_*`` function is running.

The public entry points are ``fit_metric``/``transform_metric``,
``fit_logs``/``transform_logs`` and ``fit_trace``/``transform_trace``.  The
adapters intentionally return raw modal tensors, not Ada-MGAD windows.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import jsonpickle
import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from .logs import apply_log_scalers, fit_log_scalers, log_slot_names, normalize_log_level
from .metric import (
    MetricScaler,
    apply_metric_scaler,
    aggregate_multicore,
    derive_reset_aware_rate,
    fit_metric_scaler,
    forward_fill_by_age,
    metric_semantic_kind,
)
from .schema import LOG_LEVELS, TRACE_SLOTS, TRACE_STATUSES
from .traces import TraceAggregation, apply_trace_scalers, fit_trace_scalers


GAIA_SERVICES = (
    "dbservice1", "dbservice2", "logservice1", "logservice2",
    "mobservice1", "mobservice2", "redisservice1", "redisservice2",
    "webservice1", "webservice2",
)
GAIA_IP_MAP = {
    "0.0.0.1": ("mobservice1", "redisservice1", "webservice1"),
    "0.0.0.2": ("dbservice2", "logservice2", "redisservice2"),
    "0.0.0.3": ("logservice1", "webservice2"),
    "0.0.0.4": ("dbservice1", "mobservice2"),
}
_SERVICE_TYPES = {
    "db": ("dbservice1", "dbservice2"),
    "log": ("logservice1", "logservice2"),
    "mob": ("mobservice1", "mobservice2"),
    "redis": ("redisservice1", "redisservice2"),
    "web": ("webservice1", "webservice2"),
}
_METRIC_RE = re.compile(
    r"^(?P<source>[^_]+)_(?P<ip>[0-9.]+)_(?P<feature>.+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.csv$"
)
_CORE_RE = re.compile(r"_core_(?P<core>\d+)(?P<tail>_)" )


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _grid(start_ms: int, end_ms: int, grid_ms: int) -> np.ndarray:
    if end_ms <= start_ms or grid_ms <= 0 or (end_ms - start_ms) % grid_ms:
        raise ValueError("split interval must be a positive whole number of grid bins")
    return np.arange(int(start_ms), int(end_ms), int(grid_ms), dtype=np.int64)


@dataclass(frozen=True)
class MetricFile:
    path: str
    source: str
    ip: str
    logical_feature: str
    core_id: Optional[int]
    targets: Tuple[str, ...]


@dataclass(frozen=True)
class MetricSlot:
    name: str
    scope: str
    logical_feature: str
    statistic: str
    semantic_kind: str
    targets: Tuple[str, ...]
    source_keys: Tuple[str, ...]


@dataclass(frozen=True)
class MetricFit:
    grid_ms: int
    train_start_ms: int
    train_end_ms: int
    services: Tuple[str, ...]
    slots: Tuple[MetricSlot, ...]
    scalers: Mapping[str, MetricScaler]
    source_index: Mapping[str, Tuple[MetricFile, ...]]
    fill_max_age_ms: int
    neutral_value: float = 0.5

    @property
    def slot_names(self) -> Tuple[str, ...]:
        return tuple(slot.name for slot in self.slots)


def _canonical_metric_feature(feature: str) -> Tuple[str, Optional[int]]:
    match = _CORE_RE.search(feature)
    if not match:
        return feature, None
    core = int(match.group("core"))
    canonical = feature[: match.start()] + "_core_X" + feature[match.end() - 1 :]
    return canonical, core


def index_metric_files(metric_dir: Path, services: Sequence[str] = GAIA_SERVICES) -> Mapping[str, Tuple[MetricFile, ...]]:
    """Index raw metric files without reading their bodies.

    Service-type slots require both instances of a type.  A single-instance
    service metric is therefore not silently copied to its sibling.  Host
    files are mapped through GAIA's published IP map and retain their host
    scope in the slot name.
    """
    service_set = set(services)
    grouped: Dict[str, List[MetricFile]] = {}
    root = Path(metric_dir)
    for path in sorted(root.glob("*.csv")):
        match = _METRIC_RE.match(path.name)
        if match is None:
            continue
        source, ip = match.group("source"), match.group("ip")
        logical, core_id = _canonical_metric_feature(match.group("feature"))
        if source in service_set:
            targets = (source,)
        elif source == "system":
            targets = tuple(service for service in GAIA_IP_MAP.get(ip, ()) if service in service_set)
        else:
            continue
        if not targets:
            continue
        record = MetricFile(str(path), source, ip, logical, core_id, targets)
        scope = "host" if source == "system" else "service"
        key = "{}::{}".format(scope, logical)
        grouped.setdefault(key, []).append(record)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def _read_metric_series(path: Path, grid: np.ndarray, start_ms: int, end_ms: int) -> np.ndarray:
    sums = np.zeros(len(grid), dtype=np.float64)
    counts = np.zeros(len(grid), dtype=np.int64)
    grid_ms = int(grid[1] - grid[0]) if len(grid) > 1 else 30_000
    for chunk in pd.read_csv(path, usecols=["timestamp", "value"], chunksize=100_000,
                             keep_default_na=False, on_bad_lines="error"):
        timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        values = pd.to_numeric(chunk["value"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(timestamps) & np.isfinite(values)
        valid &= (timestamps >= start_ms) & (timestamps < end_ms)
        if not np.any(valid):
            continue
        positions = ((timestamps[valid].astype(np.int64) - int(grid[0])) // grid_ms)
        inside = (positions >= 0) & (positions < len(grid))
        positions = positions[inside].astype(np.int64)
        selected = values[valid][inside]
        np.add.at(sums, positions, selected)
        np.add.at(counts, positions, 1)
    result = np.full(len(grid), np.nan, dtype=np.float64)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def _metric_values(records: Sequence[MetricFile], target: str, grid: np.ndarray,
                   start_ms: int, end_ms: int, semantic: str, statistic: str = "value") -> np.ndarray:
    by_core: Dict[Optional[int], List[np.ndarray]] = {}
    for record in records:
        if target not in record.targets:
            continue
        series = _read_metric_series(Path(record.path), grid, start_ms, end_ms)
        by_core.setdefault(record.core_id, []).append(series)
    if not by_core:
        return np.full(len(grid), np.nan, dtype=np.float64)
    core_series = []
    for core_id in sorted(by_core, key=lambda value: (-1 if value is None else value)):
        values = np.vstack(by_core[core_id])
        finite = np.isfinite(values)
        result = np.full(values.shape[1], np.nan, dtype=np.float64)
        np.divide(np.where(finite, values, 0.0).sum(axis=0), finite.sum(axis=0), out=result, where=finite.sum(axis=0) > 0)
        # A cumulative counter must be differenced independently per core.
        # Differencing only after aggregation can hide a reset on one core or
        # turn it into a false spike in the aggregate.
        if semantic == "rate":
            result = derive_reset_aware_rate(
                grid, result,
                max_gap_ms=int(grid[1] - grid[0]) * 2 if len(grid) > 1 else 60_000,
            )
        core_series.append(result)
    matrix = np.vstack(core_series).T
    if matrix.shape[1] == 1:
        values = matrix[:, 0]
    else:
        values = aggregate_multicore(matrix)["max"] if statistic == "max" else aggregate_multicore(matrix)["mean"]
    return values


def _metric_quality(values: np.ndarray, min_coverage: float, min_unique: int,
                    min_dynamic_ratio: float) -> Optional[float]:
    finite = values[np.isfinite(values)]
    coverage = float(len(finite) / len(values)) if len(values) else 0.0
    if coverage < min_coverage or len(np.unique(finite)) < min_unique:
        return None
    q05, q95 = np.quantile(finite, [0.05, 0.95])
    dynamic = float(q95 - q05)
    dynamic_ratio = dynamic / (float(np.mean(np.abs(finite))) + 1e-6)
    if dynamic <= 0.0 or dynamic_ratio < min_dynamic_ratio:
        return None
    return coverage * (np.log1p(dynamic) + 0.1 * np.log1p(len(np.unique(finite))))


def _candidate_correlation(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray],
                           services: Sequence[str]) -> Tuple[float, float]:
    """Return pairwise-complete absolute Pearson and Spearman correlations.

    Only common applicable nodes are compared.  This keeps a host slot from
    being declared redundant merely because its non-applicable neutral values
    match a service slot.  The values are still Train-only candidate arrays.
    """
    common = [service for service in services if service in left and service in right]
    if not common:
        return np.nan, np.nan
    x = np.concatenate([np.asarray(left[service], dtype=np.float64) for service in common])
    y = np.concatenate([np.asarray(right[service], dtype=np.float64) for service in common])
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.ptp(x) <= 0.0 or np.ptp(y) <= 0.0:
        return np.nan, np.nan
    pearson = float(np.corrcoef(x, y)[0, 1])
    # pandas rank gives deterministic average ranks for ties and avoids a
    # scipy dependency in the raw reader.
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if np.ptp(rx) > 0 and np.ptp(ry) > 0 else np.nan
    return pearson, spearman


def _select_metric_candidates(candidates, services: Sequence[str], *, max_slots: Optional[int],
                              required_slots: Optional[int], scope_quotas: Optional[Mapping[str, int]]):
    """Apply deterministic scope quotas, then use real overflow slots only if needed."""
    ordered = list(candidates)
    quotas = dict(scope_quotas or {})

    def quota_for(scope):
        if scope in quotas:
            return int(quotas[scope])
        if scope.startswith("service_type_") and "service_type" in quotas:
            return int(quotas["service_type"])
        return None

    selected = []
    selected_ids = set()
    counts = {}
    limit = int(max_slots) if max_slots is not None else None
    for candidate in ordered:
        slot = candidate[0]
        quota = quota_for(slot.scope)
        if quota is not None and counts.get(slot.scope, 0) >= quota:
            continue
        if limit is not None and len(selected) >= limit:
            break
        selected.append(candidate)
        selected_ids.add(slot.name)
        counts[slot.scope] = counts.get(slot.scope, 0) + 1
    if required_slots is not None and len(selected) < int(required_slots):
        # A scope quota is a preference for the normal budget, not permission
        # to invent slots.  Emergency fill draws only from still-qualified
        # real slots, in the same score/name order.
        for candidate in ordered:
            if candidate[0].name in selected_ids:
                continue
            if limit is not None and len(selected) >= limit:
                break
            selected.append(candidate)
            selected_ids.add(candidate[0].name)
            if len(selected) >= int(required_slots):
                break
    if required_slots is not None and len(selected) < int(required_slots):
        raise ValueError("qualified real Metric slots {} below required budget {}; refusing padding".format(len(ordered), required_slots))
    return selected


def _scope_for_targets(targets: Sequence[str], source_scope: str, services: Sequence[str]) -> Optional[str]:
    target_set = set(targets)
    all_set = set(services)
    if source_scope == "host":
        return "host"
    if target_set == all_set:
        return "global"
    for name, members in _SERVICE_TYPES.items():
        if target_set == set(members).intersection(all_set):
            return "service_type_{}".format(name)
    return None


def fit_metric(metric_dir: Path, train_start_ms: int, train_end_ms: int, *, grid_ms: int = 30_000,
               services: Sequence[str] = GAIA_SERVICES, min_coverage: float = 0.20,
               min_unique: int = 2, min_dynamic_ratio: float = 1e-4,
               max_slots: Optional[int] = None, required_slots: Optional[int] = None,
               fill_max_intervals: int = 10, semantic_overrides: Optional[Mapping[str, str]] = None,
               pearson_threshold: Optional[float] = None, spearman_threshold: Optional[float] = None,
               correlation_threshold: Optional[float] = None,
               scope_quotas: Optional[Mapping[str, int]] = None,
               scope_max_quotas: Optional[Mapping[str, int]] = None) -> MetricFit:
    """Fit real Metric slots and scalers from Train only.

    ``required_slots`` is a fail-closed budget gate.  It is intentionally
    separate from ``max_slots`` so a caller cannot satisfy a dimension budget
    by manufacturing padding features.
    """
    if scope_quotas is not None and scope_max_quotas is not None and dict(scope_quotas) != dict(scope_max_quotas):
        raise ValueError("scope_quotas and scope_max_quotas disagree")
    if scope_quotas is None:
        scope_quotas = scope_max_quotas
    if scope_quotas is not None and any(int(value) < 0 for value in scope_quotas.values()):
        raise ValueError("scope quotas must be non-negative")
    grid = _grid(train_start_ms, train_end_ms, grid_ms)
    source_index = index_metric_files(metric_dir, services)
    candidates: List[Tuple[MetricSlot, float, Dict[str, np.ndarray]]] = []
    overrides = dict(semantic_overrides or {})
    for key, records in source_index.items():
        source_scope = "host" if key.startswith("host::") else "service"
        logical = key.split("::", 1)[1]
        targets = tuple(sorted(set(target for record in records for target in record.targets), key=lambda s: list(services).index(s)))
        scope = _scope_for_targets(targets, source_scope, services)
        if scope is None:
            continue
        applicable = tuple(service for service in services if service in targets)
        semantic_kind = overrides.get(logical, metric_semantic_kind(logical))
        if semantic_kind == "counter":
            semantic = "rate"
        elif semantic_kind == "direct_rate":
            semantic = "direct"
        else:
            semantic = "value"
        core_count = len(set(record.core_id for record in records if record.core_id is not None))
        statistics = ("mean", "max") if core_count > 1 else ("value",)
        for statistic in statistics:
            by_target = {}
            scores = []
            for target in applicable:
                values = _metric_values(records, target, grid, train_start_ms, train_end_ms, semantic, statistic)
                # Quality is evaluated before imputation; Train fill is applied
                # only after the slot has passed this availability gate.
                score = _metric_quality(values, min_coverage, min_unique, min_dynamic_ratio)
                if score is None:
                    break
                by_target[target] = values
                scores.append(score)
            if len(by_target) != len(applicable):
                continue
            slot_name = "{}::{}::{}".format(scope, logical, statistic)
            slot = MetricSlot(slot_name, scope, logical, statistic, semantic_kind, applicable, (key,))
            candidates.append((slot, float(np.mean(scores)), by_target))
    candidates.sort(key=lambda item: (-item[1], item[0].name))
    if correlation_threshold is not None:
        if pearson_threshold is None:
            pearson_threshold = float(correlation_threshold)
        if spearman_threshold is None:
            spearman_threshold = float(correlation_threshold)
    for threshold, name in ((pearson_threshold, "pearson_threshold"), (spearman_threshold, "spearman_threshold")):
        if threshold is not None and not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("{} must be within [0, 1]".format(name))
    reduced = []
    for candidate in candidates:
        redundant = False
        for kept in reduced:
            pearson, spearman = _candidate_correlation(candidate[2], kept[2], services)
            pearson_hit = pearson_threshold is not None and np.isfinite(pearson) and abs(pearson) >= float(pearson_threshold)
            spearman_hit = spearman_threshold is not None and np.isfinite(spearman) and abs(spearman) >= float(spearman_threshold)
            if pearson_hit or spearman_hit:
                redundant = True
                break
        if not redundant:
            reduced.append(candidate)
    candidates = _select_metric_candidates(
        reduced, services, max_slots=max_slots, required_slots=required_slots,
        scope_quotas=scope_quotas,
    )
    slots = tuple(item[0] for item in candidates)
    scalers: Dict[str, MetricScaler] = {}
    fill_age = int(fill_max_intervals) * int(grid_ms)
    for slot, _, by_target in candidates:
        pooled = []
        for target in slot.targets:
            values = forward_fill_by_age(grid, by_target[target], max_age_ms=fill_age, split_slices=(slice(0, len(grid)),))
            pooled.append(values)
        finite = np.concatenate(pooled)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            raise ValueError("selected Metric slot has no finite Train values: {}".format(slot.name))
        scalers[slot.name] = fit_metric_scaler(finite)
    return MetricFit(int(grid_ms), int(train_start_ms), int(train_end_ms), tuple(services), slots, scalers, source_index, fill_age)


def transform_metric(metric_fit: MetricFit, metric_dir: Path, split_start_ms: int, split_end_ms: int) -> np.ndarray:
    """Transform one split with a previously frozen Metric schema."""
    if not isinstance(metric_fit, MetricFit):
        raise TypeError("metric_fit must be MetricFit")
    grid = _grid(split_start_ms, split_end_ms, metric_fit.grid_ms)
    result, _ = _transform_metric_with_masks(metric_fit, metric_dir, split_start_ms, split_end_ms)
    return result


def _transform_metric_with_masks(metric_fit: MetricFit, metric_dir: Path, split_start_ms: int, split_end_ms: int):
    grid = _grid(split_start_ms, split_end_ms, metric_fit.grid_ms)
    result = np.full((len(grid), len(metric_fit.services), len(metric_fit.slots)), metric_fit.neutral_value, dtype=np.float32)
    observed = np.zeros(result.shape, dtype=bool)
    service_index = {service: index for index, service in enumerate(metric_fit.services)}
    for slot_index, slot in enumerate(metric_fit.slots):
        records = metric_fit.source_index[slot.source_keys[0]]
        for service in slot.targets:
            semantic = "rate" if slot.semantic_kind == "counter" else "direct" if slot.semantic_kind == "direct_rate" else "value"
            values = _metric_values(records, service, grid, split_start_ms, split_end_ms, semantic, slot.statistic)
            filled = forward_fill_by_age(grid, values, max_age_ms=metric_fit.fill_max_age_ms, split_slices=(slice(0, len(grid)),))
            scaled, _ = apply_metric_scaler(filled, metric_fit.scalers[slot.name])
            result[:, service_index[service], slot_index] = np.where(np.isfinite(scaled), scaled, metric_fit.neutral_value)
            observed[:, service_index[service], slot_index] = np.isfinite(filled)
    return result, observed


def transform_metric_with_observability(metric_fit: MetricFit, metric_dir: Path, split_start_ms: int, split_end_ms: int):
    """Return Metric values plus explicit applicability/observability channels.

    The three returned arrays are ``global_observed_fraction``,
    ``host_applicable`` and ``host_observed_fraction``.  They are derived from
    the frozen slot scopes and raw availability, not from transformed numeric
    values, so a genuine normalized zero cannot be confused with missingness.
    """
    values, observed = _transform_metric_with_masks(metric_fit, metric_dir, split_start_ms, split_end_ms)
    service_index = {service: index for index, service in enumerate(metric_fit.services)}
    global_positions = [index for index, slot in enumerate(metric_fit.slots) if slot.scope == "global"]
    host_positions = [index for index, slot in enumerate(metric_fit.slots) if slot.scope == "host"]
    global_fraction = observed[..., global_positions].mean(axis=-1, keepdims=True) if global_positions else np.zeros(values.shape[:2] + (1,), dtype=np.float32)
    host_applicable = np.zeros((len(metric_fit.services),), dtype=np.float32)
    for slot in metric_fit.slots:
        if slot.scope == "host":
            for service in slot.targets:
                host_applicable[service_index[service]] = 1.0
    host_mask = observed[..., host_positions] if host_positions else np.zeros(values.shape[:2] + (1,), dtype=bool)
    host_fraction = host_mask.mean(axis=-1, keepdims=True) if host_positions else np.zeros(values.shape[:2] + (1,), dtype=np.float32)
    applicability = np.broadcast_to(host_applicable[None, :, None], values.shape[:2] + (1,)).astype(np.float32)
    augmented = np.concatenate((values, global_fraction.astype(np.float32), applicability, host_fraction.astype(np.float32)), axis=-1)
    return augmented.astype(np.float32), {
        "values": values,
        "observed": observed,
        "global_observed_fraction": global_fraction.astype(np.float32),
        "host_applicable": applicability,
        "host_observed_fraction": host_fraction.astype(np.float32),
        "slot_names": metric_fit.slot_names + ("global_observed_fraction", "host_applicable", "host_observed_fraction"),
    }


@dataclass(frozen=True)
class LogFit:
    grid_ms: int
    train_start_ms: int
    train_end_ms: int
    drain_state: str
    stable_cluster_ids: Tuple[int, ...]
    slot_names: Tuple[str, ...]
    scalers: Mapping[str, float]
    cluster_stats: Mapping[str, Mapping[str, Any]]
    config_path: Optional[str] = None


def _new_miner(config_path: Optional[Path] = None) -> TemplateMiner:
    config = TemplateMinerConfig()
    if config_path is not None:
        config.load(str(config_path))
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def _restore_miner(state: str, config_path: Optional[Path]) -> TemplateMiner:
    miner = _new_miner(config_path)
    loaded = jsonpickle.loads(state)
    if loaded.id_to_cluster and isinstance(next(iter(loaded.id_to_cluster.keys())), str):
        converted = {}
        for key, value in loaded.id_to_cluster.items():
            text = str(key).replace("json://", "")
            try:
                converted[int(text)] = value
            except ValueError:
                converted[key] = value
        loaded.id_to_cluster = converted
    miner.drain.id_to_cluster = loaded.id_to_cluster
    miner.drain.clusters_counter = loaded.clusters_counter
    miner.drain.root_node = loaded.root_node
    return miner


def _log_payload(message: Any) -> str:
    text = str(message).strip()
    return text.split(" | ", 5)[-1].strip()


def _log_time(message: Any) -> Optional[int]:
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})", str(message).strip())
    if not match:
        return None
    try:
        value = pd.Timestamp("{}.{}".format(match.group(1), match.group(2)), tz="Asia/Shanghai")
        return int(value.timestamp() * 1000)
    except (TypeError, ValueError):
        return None


def _iter_log_rows(business_dir: Path, services: Sequence[str], chunk_rows: int) -> Iterator[Tuple[str, str, int]]:
    for service in services:
        path = Path(business_dir) / "business_table_{}_2021-07.csv".format(service)
        if not path.is_file():
            raise FileNotFoundError(path)
        for chunk in pd.read_csv(path, usecols=["message"], chunksize=int(chunk_rows), keep_default_na=False, on_bad_lines="error"):
            for message in chunk["message"].to_numpy(dtype=str):
                timestamp = _log_time(message)
                if timestamp is not None:
                    yield service, message, timestamp


def fit_logs(business_dir: Path, train_start_ms: int, train_end_ms: int, *, grid_ms: int = 30_000,
             services: Sequence[str] = GAIA_SERVICES, chunk_rows: int = 100_000,
             config_path: Optional[Path] = None, min_template_count: int = 2,
             min_template_bins: int = 2, max_stable_templates: Optional[int] = None) -> LogFit:
    """Fit Drain3, stable-template routing and Train q99 scales."""
    miner = _new_miner(config_path)
    for service, message, timestamp in _iter_log_rows(business_dir, services, chunk_rows):
        if train_start_ms <= timestamp < train_end_ms:
            miner.add_log_message(_log_payload(message))
    stats: Dict[int, Dict[str, Any]] = {}
    grid_ids = set(int(x) for x in _grid(train_start_ms, train_end_ms, grid_ms))
    for service, message, timestamp in _iter_log_rows(business_dir, services, chunk_rows):
        if not train_start_ms <= timestamp < train_end_ms:
            continue
        result = miner.match(_log_payload(message), full_search_strategy="always")
        if result is None:
            continue
        cluster_id = int(result.cluster_id)
        item = stats.setdefault(cluster_id, {"count": 0, "bins": set(), "services": set()})
        item["count"] += 1
        item["bins"].add(int((timestamp - train_start_ms) // grid_ms))
        item["services"].add(service)
    stable = [cluster_id for cluster_id, item in stats.items()
              if item["count"] >= int(min_template_count) and len(item["bins"]) >= int(min_template_bins)]
    stable.sort(key=lambda cluster_id: (-stats[cluster_id]["count"], cluster_id))
    if max_stable_templates is not None:
        stable = stable[: int(max_stable_templates)]
    names = log_slot_names(stable)
    train_grid = _grid(train_start_ms, train_end_ms, grid_ms)
    counts = np.zeros((len(train_grid) * len(services), len(names)), dtype=np.float64)
    # Reuse the frozen matcher and produce a Train matrix for q99 fitting.
    service_index = {service: index for index, service in enumerate(services)}
    for service, message, timestamp in _iter_log_rows(business_dir, services, chunk_rows):
        if not train_start_ms <= timestamp < train_end_ms:
            continue
        result = miner.match(_log_payload(message), full_search_strategy="always")
        cluster_id = int(result.cluster_id) if result is not None else None
        level = normalize_log_level(re.search(r"\|\s*([A-Za-z]+)\s*\|", message).group(1) if re.search(r"\|\s*([A-Za-z]+)\s*\|", message) else None)
        if cluster_id in stable:
            route = "stable_template_{}".format(cluster_id)
        elif cluster_id is None:
            route = "UNK_{}".format(level)
        else:
            route = "RARE_{}".format(level)
        row = int((timestamp - train_start_ms) // grid_ms) * len(services) + service_index[service]
        counts[row, names.index(route)] += 1.0
        counts[row, names.index("level_{}".format(level))] += 1.0
    fitted = fit_log_scalers(counts, names)
    serial_stats = {str(key): {"count": int(value["count"]), "bins": len(value["bins"]), "services": sorted(value["services"])} for key, value in stats.items()}
    return LogFit(int(grid_ms), int(train_start_ms), int(train_end_ms), jsonpickle.dumps(miner.drain, keys=True), tuple(stable), tuple(names), fitted.scales, serial_stats, str(config_path) if config_path else None)


def transform_logs(log_fit: LogFit, business_dir: Path, split_start_ms: int, split_end_ms: int, *, services: Sequence[str] = GAIA_SERVICES, chunk_rows: int = 100_000) -> np.ndarray:
    if not isinstance(log_fit, LogFit):
        raise TypeError("log_fit must be LogFit")
    grid = _grid(split_start_ms, split_end_ms, log_fit.grid_ms)
    result = np.zeros((len(grid), len(services), len(log_fit.slot_names)), dtype=np.float32)
    service_index = {service: index for index, service in enumerate(services)}
    positions = {name: index for index, name in enumerate(log_fit.slot_names)}
    miners = {
        service: _restore_miner(log_fit.drain_state, Path(log_fit.config_path) if log_fit.config_path else None)
        for service in services
    }
    for service, message, timestamp in _iter_log_rows(business_dir, services, chunk_rows):
        if timestamp < split_start_ms or timestamp >= split_end_ms:
            continue
        position = int((timestamp - split_start_ms) // log_fit.grid_ms)
        if position < 0 or position >= len(grid):
            continue
        match = miners[service].match(_log_payload(message), full_search_strategy="always")
        cluster_id = int(match.cluster_id) if match is not None else None
        level_match = re.search(r"\|\s*([A-Za-z]+)\s*\|", message)
        level = normalize_log_level(level_match.group(1) if level_match else None)
        if cluster_id in log_fit.stable_cluster_ids:
            route = "stable_template_{}".format(cluster_id)
        elif cluster_id is None:
            route = "UNK_{}".format(level)
        else:
            route = "RARE_{}".format(level)
        result[position, service_index[service], positions[route]] += 1.0
        result[position, service_index[service], positions["level_{}".format(level)]] += 1.0
    return apply_log_scalers(result, log_fit.slot_names, log_fit.scalers)


@dataclass(frozen=True)
class TraceFit:
    grid_ms: int
    train_start_ms: int
    train_end_ms: int
    services: Tuple[str, ...]
    directed_edges: Tuple[Tuple[str, str], ...]
    scalers: Mapping[str, float]
    diagnostics: Mapping[str, int]


def _iter_trace_rows(trace_dir: Path, services: Sequence[str], chunk_rows: int) -> Iterator[Mapping[str, Any]]:
    for service in services:
        path = Path(trace_dir) / "trace_table_{}_2021-07.csv".format(service)
        if not path.is_file():
            raise FileNotFoundError(path)
        usecols = ["trace_id", "span_id", "parent_id", "service_name", "start_time", "end_time", "status_code"]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunk_rows), keep_default_na=False, on_bad_lines="error"):
            starts = pd.to_datetime(chunk["start_time"], errors="coerce")
            ends = pd.to_datetime(chunk["end_time"], errors="coerce")
            for index, row in chunk.iterrows():
                start = starts.loc[index]
                end = ends.loc[index]
                if pd.isna(start) or pd.isna(end):
                    start_ms = end_ms = None
                else:
                    start_ms = int(pd.Timestamp(start, tz="Asia/Shanghai").timestamp() * 1000) if pd.Timestamp(start).tzinfo is None else int(pd.Timestamp(start).timestamp() * 1000)
                    end_ms = int(pd.Timestamp(end, tz="Asia/Shanghai").timestamp() * 1000) if pd.Timestamp(end).tzinfo is None else int(pd.Timestamp(end).timestamp() * 1000)
                yield {"trace_id": str(row["trace_id"]), "span_id": str(row["span_id"]), "parent_id": str(row["parent_id"]), "service": str(row["service_name"]), "start_time_ms": start_ms, "end_time_ms": end_ms, "status_code": str(row["status_code"])}


def _aggregate_trace_files(trace_dir: Path, services: Sequence[str], grid: np.ndarray, split_start_ms: int, split_end_ms: int, chunk_rows: int) -> TraceAggregation:
    """Aggregate a split using a disk-backed, split-local parent index.

    GAIA Trace rows are too large to materialize as Python dictionaries.  The
    first pass indexes only parent identity/service for spans whose *end* lies
    in this split.  The second pass resolves each child against that index and
    updates the fixed tensor.  Consequently a parent from another split is
    intentionally unmatched and cross-split state never enters the transform.
    """
    service_names = tuple(services)
    if len(grid) < 2:
        raise ValueError("Trace grid must contain at least two bins")
    service_index = {service: index for index, service in enumerate(service_names)}
    status_index = {status: index for index, status in enumerate(TRACE_STATUSES)}
    bin_ms = int(grid[1] - grid[0])
    counts = np.zeros((len(grid), len(service_names), len(service_names), len(TRACE_STATUSES)), dtype=np.float64)
    durations = np.zeros_like(counts)
    diagnostics = {
        "rows_in_split": 0, "invalid_rows": 0, "root_rows": 0,
        "unmatched_parent_rows": 0, "ambiguous_parent_rows": 0,
        "self_loop_rows": 0, "unknown_service_rows": 0,
        "unknown_status_rows": 0, "matched_cross_service_rows": 0,
        "unseen_directed_edge_rows": 0,
    }
    observed_edges = set()
    db_file = tempfile.NamedTemporaryFile(prefix="gaia_trace_parent_", suffix=".sqlite3", delete=False)
    db_path = db_file.name
    db_file.close()
    try:
        connection = sqlite3.connect(db_path)
        connection.execute("CREATE TABLE parents (trace_id TEXT NOT NULL, span_id TEXT NOT NULL, service TEXT NOT NULL, PRIMARY KEY(trace_id, span_id, service))")
        for raw in _iter_trace_rows(trace_dir, service_names, chunk_rows):
            end_ms = _as_int(raw.get("end_time_ms"))
            service = str(raw.get("service", ""))
            if end_ms is None or not split_start_ms <= end_ms < split_end_ms or service not in service_index:
                continue
            trace_id, span_id = str(raw.get("trace_id", "")), str(raw.get("span_id", ""))
            if trace_id and span_id:
                connection.execute("INSERT OR IGNORE INTO parents(trace_id, span_id, service) VALUES (?, ?, ?)", (trace_id, span_id, service))
        connection.commit()
        for raw in _iter_trace_rows(trace_dir, service_names, chunk_rows):
            end_ms = _as_int(raw.get("end_time_ms"))
            if end_ms is None or not split_start_ms <= end_ms < split_end_ms:
                continue
            diagnostics["rows_in_split"] += 1
            parent_id = str(raw.get("parent_id", "")).strip()
            if parent_id in ("", "0", "None", "nan"):
                diagnostics["root_rows"] += 1
                continue
            destination = str(raw.get("service", ""))
            if destination not in service_index:
                diagnostics["unknown_service_rows"] += 1
                continue
            trace_id = str(raw.get("trace_id", ""))
            matches = [row[0] for row in connection.execute("SELECT service FROM parents WHERE trace_id = ? AND span_id = ?", (trace_id, parent_id))]
            if not matches:
                diagnostics["unmatched_parent_rows"] += 1
                continue
            if len(matches) != 1:
                diagnostics["ambiguous_parent_rows"] += 1
                continue
            source = matches[0]
            if source == destination:
                diagnostics["self_loop_rows"] += 1
                continue
            status = str(raw.get("status_code", "")).strip()
            if status not in status_index:
                diagnostics["unknown_status_rows"] += 1
                continue
            start_ms = _as_int(raw.get("start_time_ms"))
            if start_ms is None or end_ms < start_ms:
                diagnostics["invalid_rows"] += 1
                continue
            position = int((end_ms - int(grid[0])) // bin_ms)
            if position < 0 or position >= len(grid):
                diagnostics["invalid_rows"] += 1
                continue
            source_position, destination_position = service_index[source], service_index[destination]
            status_position = status_index[status]
            counts[position, source_position, destination_position, status_position] += 1.0
            durations[position, source_position, destination_position, status_position] += (end_ms - start_ms) / 1000.0
            observed_edges.add((source, destination))
            diagnostics["matched_cross_service_rows"] += 1
        connection.close()
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass
    features = np.zeros((len(grid), len(service_names), len(service_names), len(TRACE_SLOTS)), dtype=np.float32)
    for status_position in range(len(TRACE_STATUSES)):
        count_position = status_position * 2
        features[..., count_position] = counts[..., status_position].astype(np.float32)
        np.divide(durations[..., status_position], counts[..., status_position], out=features[..., count_position + 1], where=counts[..., status_position] > 0)
    return TraceAggregation(features=features, diagnostics=diagnostics, observed_directed_edges=tuple(sorted(observed_edges)))


def fit_trace(trace_dir: Path, train_start_ms: int, train_end_ms: int, *, grid_ms: int = 30_000,
              services: Sequence[str] = GAIA_SERVICES, chunk_rows: int = 100_000,
              min_edge_rows: int = 1, min_positive_bins: int = 2) -> TraceFit:
    """Fit Train-only call graph and count/mean-latency scalers."""
    grid = _grid(train_start_ms, train_end_ms, grid_ms)
    aggregation = _aggregate_trace_files(trace_dir, services, grid, train_start_ms, train_end_ms, chunk_rows)
    edge_counts = {}
    for source, destination in aggregation.observed_directed_edges:
        source_index = list(services).index(source)
        destination_index = list(services).index(destination)
        edge_counts[(source, destination)] = int(np.sum(aggregation.features[:, source_index, destination_index, ::2]))
    edges = tuple(sorted(edge for edge, count in edge_counts.items() if count >= int(min_edge_rows)))
    if not edges:
        raise ValueError("Train Trace contains no directed edge meeting min_edge_rows")
    scalers = fit_trace_scalers(aggregation.features, services=services, frozen_directed_edges=edges, min_positive_bins=min_positive_bins)
    return TraceFit(int(grid_ms), int(train_start_ms), int(train_end_ms), tuple(services), edges, scalers.scales, aggregation.diagnostics)


def transform_trace(trace_fit: TraceFit, trace_dir: Path, split_start_ms: int, split_end_ms: int, *, chunk_rows: int = 100_000) -> np.ndarray:
    if not isinstance(trace_fit, TraceFit):
        raise TypeError("trace_fit must be TraceFit")
    grid = _grid(split_start_ms, split_end_ms, trace_fit.grid_ms)
    aggregation = _aggregate_trace_files(trace_dir, trace_fit.services, grid, split_start_ms, split_end_ms, chunk_rows)
    return apply_trace_scalers(aggregation.features, services=trace_fit.services, frozen_directed_edges=trace_fit.directed_edges, scales=trace_fit.scalers)


__all__ = [
    "MetricFile", "MetricSlot", "MetricFit", "index_metric_files", "fit_metric", "transform_metric",
    "transform_metric_with_observability",
    "LogFit", "fit_logs", "transform_logs", "TraceFit", "fit_trace", "transform_trace",
]
