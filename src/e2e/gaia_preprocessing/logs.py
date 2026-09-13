"""Frozen-vocabulary Log routing and count scaling for GAIA."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .schema import LOG_LEVELS


@dataclass(frozen=True)
class RoutedLogBin:
    slot_names: Tuple[str, ...]
    counts: np.ndarray
    audit_total: int


@dataclass(frozen=True)
class FrozenLogScalers:
    scales: Mapping[str, float]
    fallback_sources: Mapping[str, str]


def log_slot_names(stable_cluster_ids: Sequence[Any]) -> Tuple[str, ...]:
    stable_ids = list(stable_cluster_ids)
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("stable_cluster_ids contains duplicates")
    return tuple(
        ["stable_template_{}".format(cluster_id) for cluster_id in stable_ids]
        + ["RARE_{}".format(level) for level in LOG_LEVELS]
        + ["UNK_{}".format(level) for level in LOG_LEVELS]
        + ["level_{}".format(level) for level in LOG_LEVELS]
    )


def normalize_log_level(level: Any) -> str:
    normalized = str(level).strip().upper() if level is not None else "UNKNOWN"
    if normalized == "WARN":
        normalized = "WARNING"
    return normalized if normalized in LOG_LEVELS else "UNKNOWN"


def route_log_bin(
    *,
    events: Iterable[Tuple[Any, Any]],
    stable_cluster_ids: Sequence[Any],
) -> RoutedLogBin:
    """Route already matched events through a frozen vocabulary.

    ``cluster_id=None`` means that the frozen Drain matcher did not match. Any
    non-stable cluster ID is a Train-known rare template. Cluster creation is
    intentionally outside this transform boundary.
    """

    names = log_slot_names(stable_cluster_ids)
    positions = {name: index for index, name in enumerate(names)}
    stable = set(stable_cluster_ids)
    counts = np.zeros(len(names), dtype=np.int64)
    total = 0
    for cluster_id, raw_level in events:
        level = normalize_log_level(raw_level)
        if cluster_id in stable:
            route = "stable_template_{}".format(cluster_id)
        elif cluster_id is None:
            route = "UNK_{}".format(level)
        else:
            route = "RARE_{}".format(level)
        counts[positions[route]] += 1
        counts[positions["level_{}".format(level)]] += 1
        total += 1
    return RoutedLogBin(slot_names=names, counts=counts, audit_total=total)


def _positive_log_q99(values: np.ndarray) -> float:
    positive = values[values > 0]
    if positive.size == 0:
        return 0.0
    return float(np.quantile(np.log1p(positive), 0.99))


def fit_log_scalers(
    train_counts: np.ndarray,
    slot_names: Sequence[str],
) -> FrozenLogScalers:
    """Fit log1p/q99 scales and resolve zero-support fallbacks on Train."""

    names = tuple(slot_names)
    counts = np.asarray(train_counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(names):
        raise ValueError("train_counts must have shape [samples, len(slot_names)]")
    if len(names) != len(set(names)):
        raise ValueError("slot_names contains duplicates")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("train_counts must be finite and non-negative")

    direct = {
        name: _positive_log_q99(counts[:, index])
        for index, name in enumerate(names)
    }
    level_indices = [index for index, name in enumerate(names) if name.startswith("level_")]
    pooled = _positive_log_q99(counts[:, level_indices].sum(axis=1)) if level_indices else 0.0
    scales = {}
    sources = {}
    for name in names:
        if direct[name] > 0:
            scales[name] = direct[name]
            sources[name] = name
            continue
        if name.startswith("stable_template_"):
            raise ValueError("stable template {} has zero Train support".format(name))

        candidates = []
        if name.startswith("UNK_"):
            level = name[len("UNK_") :]
            candidates.extend(("RARE_{}".format(level), "level_{}".format(level)))
        elif name.startswith("RARE_"):
            level = name[len("RARE_") :]
            candidates.append("level_{}".format(level))
        for candidate in candidates:
            if direct.get(candidate, 0.0) > 0:
                scales[name] = direct[candidate]
                sources[name] = candidate
                break
        else:
            if pooled > 0:
                scales[name] = pooled
                sources[name] = "__pooled_train_messages__"
            else:
                scales[name] = 1.0
                sources[name] = "__fixed_1__"

    return FrozenLogScalers(
        scales=MappingProxyType(scales),
        fallback_sources=MappingProxyType(sources),
    )


def apply_log_scalers(
    counts: np.ndarray,
    slot_names: Sequence[str],
    scales: Mapping[str, float],
) -> np.ndarray:
    """Apply frozen log1p/q99 scales without learning from transformed data."""

    names = tuple(slot_names)
    values = np.asarray(counts, dtype=np.float64)
    if values.shape[-1] != len(names):
        raise ValueError("counts last dimension must match slot_names")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("counts must be finite and non-negative")
    scale_array = np.asarray([scales[name] for name in names], dtype=np.float64)
    if not np.isfinite(scale_array).all() or np.any(scale_array <= 0):
        raise ValueError("all frozen Log scales must be finite and positive")
    return np.clip(np.log1p(values) / scale_array, 0.0, 1.0).astype(np.float32)
