#!/usr/bin/env python3
"""Train-only, audit-only GAIA multimodal preprocessing statistics.

This command is deliberately separate from ``src.e2e.ad_preprocess``.  The
default mode only binds the inputs and writes metadata.  ``--full`` performs
streaming audit passes and writes diagnostic tables; it never writes detector
arrays, checkpoints, predictions, or formal preprocessing artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.log_templates import (
    LEVELS,
    _miner_from_config,
    _restore_drain,
    fit_train_templates,
    message_level,
    message_payload,
    parse_message_timestamp_ms,
)
from src.e2e.protocol import GAIA_SERVICES, layout_digest, sha256_file
from util.GAIA.constant import GAIA_IP_MAP, GAIA_MULTI_CORE_PREFIXES
from util.GAIA.pre_GAIA import (
    _metric_duplicate_reduce_mode,
    _parse_metric_filename,
    _reduce_duplicate_timestamp_values,
)


SCRIPT_SCHEMA = "gaia_multimodal_preprocessing_audit_v1"
METRIC_COLUMNS = ("timestamp", "value")
STATUS_CODES = ("200", "300", "400", "500")
_LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}


def _json_dump(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _config_split(config: Mapping[str, object], split: str) -> Tuple[int, int]:
    section = config["split"]
    start = int(section["absolute_start_ms"])
    end = int(section["absolute_end_ms"])
    boundary = int(section["boundary_ms"])
    if split == "train":
        return start, boundary
    return boundary, end


def _source_binding(config_path: Path, raw_root: Path, config: Mapping[str, object], split: str) -> Mapping[str, object]:
    roots = {
        "metrics": raw_root / "metric/metric_split/metric",
        "logs": raw_root / "business/business_split/business",
        "traces": raw_root / "trace/trace_split/trace",
    }
    missing = [str(path) for path in roots.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("GAIA raw root is missing modality directories: {}".format(missing))
    run_table = raw_root / "run/run/run/run_table_2021-07.csv"
    if not run_table.is_file():
        raise FileNotFoundError(run_table)
    return {
        "schema": SCRIPT_SCHEMA,
        "git_commit": _git_head(config_path.parents[2]),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "raw_root": str(raw_root.resolve()),
        "run_table": str(run_table.resolve()),
        "run_table_sha256": sha256_file(run_table),
        "split": split,
        "split_interval_ms": list(_config_split(config, split)),
        "grid_seconds": int(config["ad"]["grid_seconds"]),
        "services": list(GAIA_SERVICES),
        "layout": {
            name: layout_digest(path, path.glob("*.csv")) for name, path in roots.items()
        },
        "decision_inputs": ["train"],
        "gt_labels_used": False,
        "test_used_for_selection": False,
    }


def _metric_registry(metric_root: Path, train_start: int, train_end: int) -> Tuple[pd.DataFrame, Mapping[str, Mapping[str, object]]]:
    records = []
    groups: MutableMapping[str, MutableMapping[str, MutableMapping[str, list[Path]]]] = {
        service: defaultdict(lambda: defaultdict(list)) for service in GAIA_SERVICES
    }
    for path in sorted(metric_root.glob("*.csv")):
        info = _parse_metric_filename(path.name)
        if info is None:
            records.append({"path": str(path), "parse_status": "unparsed"})
            continue
        date_start = int(pd.Timestamp(info["date_start"], tz="Asia/Shanghai").timestamp() * 1000)
        date_end = int(pd.Timestamp(info["date_end"], tz="Asia/Shanghai").timestamp() * 1000)
        source_service = str(info["service"])
        feature = str(info["feature"])
        is_core = any(prefix in feature for prefix in GAIA_MULTI_CORE_PREFIXES)
        logical = re.sub(r"core_\d+", "core_X", feature) if is_core else feature
        targets = []
        scope = "unknown_namespace"
        if source_service in GAIA_SERVICES:
            targets = [(source_service, logical)]
            scope = "global_container"
        elif source_service == "system":
            targets = [
                (service, "host_" + logical)
                for service in GAIA_IP_MAP.get(str(info["ip"]), [])
                if service in GAIA_SERVICES
            ]
            scope = "host_of_node" if targets else "unmapped_system"
        for service, target in targets:
            groups[service][target][str(info["full_name"])].append(path)
            records.append({
                "path": str(path.resolve()), "source_service": source_service,
                "target_service": service, "scope": scope, "ip": str(info["ip"]),
                "feature": feature, "logical_feature": target, "full_name": info["full_name"],
                "date_start_ms": date_start, "date_end_ms": date_end,
                "overlaps_train": bool(date_end > train_start and date_start < train_end),
                "multi_core": is_core, "parse_status": "mapped",
            })
        if not targets:
            records.append({
                "path": str(path.resolve()), "source_service": source_service,
                "target_service": "", "scope": scope, "ip": str(info["ip"]),
                "feature": feature, "logical_feature": logical, "full_name": info["full_name"],
                "date_start_ms": date_start, "date_end_ms": date_end,
                "overlaps_train": bool(date_end > train_start and date_start < train_end),
                "multi_core": is_core, "parse_status": "unmapped",
            })
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame([{"parse_status": "empty"}])
    return frame, groups


def _read_metric_group(paths: Sequence[Path], feature: str, train_start: int, train_end: int, grid: np.ndarray, chunk_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return one aligned array per physical full_name group and raw timestamps."""
    frames = []
    raw_timestamps = []
    for path in paths:
        for chunk in pd.read_csv(path, usecols=list(METRIC_COLUMNS), chunksize=int(chunk_rows), low_memory=False):
            timestamp = pd.to_numeric(chunk["timestamp"], errors="coerce")
            value = pd.to_numeric(chunk["value"], errors="coerce")
            valid = timestamp.notna() & value.notna()
            valid &= timestamp >= train_start
            valid &= timestamp < train_end
            if valid.any():
                frames.append(pd.DataFrame({"timestamp": timestamp[valid].astype(np.int64), "value": value[valid].astype(float)}))
                raw_timestamps.append(timestamp[valid].to_numpy(dtype=np.int64))
    if not frames:
        return np.full(len(grid), np.nan, dtype=np.float32), np.empty(0, dtype=np.int64)
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp", "value"])
    if merged.duplicated(subset=["timestamp"]).any():
        mode = _metric_duplicate_reduce_mode(feature)
        merged = merged.groupby("timestamp", as_index=False)["value"].agg(
            lambda values: _reduce_duplicate_timestamp_values(values, mode)
        )
    aligned = (merged["timestamp"].to_numpy(dtype=np.int64) // 30000) * 30000
    compact = pd.DataFrame({"timestamp": aligned, "value": merged["value"].to_numpy(dtype=float)}).groupby("timestamp")["value"].mean()
    result = np.full(len(grid), np.nan, dtype=np.float32)
    positions = np.searchsorted(grid, compact.index.to_numpy(dtype=np.int64))
    inside = positions < len(grid)
    exact = np.zeros(len(positions), dtype=bool)
    exact[inside] = grid[positions[inside]] == compact.index.to_numpy(dtype=np.int64)[inside]
    result[positions[exact]] = compact.to_numpy(dtype=np.float32)[exact]
    return result, np.concatenate(raw_timestamps) if raw_timestamps else np.empty(0, dtype=np.int64)


def _quality(values: np.ndarray, raw_timestamps: np.ndarray, grid_ms: int = 30000) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    finite = np.asarray(values, dtype=float)
    observed = np.isfinite(finite)
    valid = finite[observed]
    if valid.size:
        q05, q95 = np.quantile(valid, [0.05, 0.95])
        dynamic_span = float(q95 - q05)
        dynamic_ratio = float(dynamic_span / (np.mean(np.abs(valid)) + 1e-6))
        variance = float(np.var(valid))
        zero_ratio = float(np.mean(np.abs(valid) <= 1e-12))
        q01, q99 = np.quantile(valid, [0.01, 0.99])
    else:
        q05 = q95 = q01 = q99 = dynamic_span = dynamic_ratio = variance = zero_ratio = 0.0
    gaps = []
    missing = ~observed
    if missing.any():
        starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
        ends = np.flatnonzero(missing & np.r_[~missing[1:], True])
        gaps = (ends - starts + 1).astype(int).tolist()
    intervals = np.diff(np.unique(np.asarray(raw_timestamps, dtype=np.int64)))
    intervals = intervals[intervals > 0]
    quality = {
        "coverage": float(observed.mean()) if len(values) else 0.0,
        "observed_bins": int(observed.sum()), "total_bins": int(len(values)),
        "unique_count": int(pd.Series(valid).nunique()) if valid.size else 0,
        "zero_ratio": zero_ratio, "variance": variance,
        "q01": float(q01), "q05": float(q05), "q95": float(q95), "q99": float(q99),
        "dynamic_span": dynamic_span, "dynamic_ratio": dynamic_ratio,
        "constant": bool(valid.size > 0 and pd.Series(valid).nunique() <= 1),
        "near_constant": bool(valid.size > 1 and dynamic_ratio < 1e-4),
        "median_sampling_interval_ms": float(np.median(intervals)) if intervals.size else None,
        "p95_sampling_interval_ms": float(np.quantile(intervals, 0.95)) if intervals.size else None,
    }
    gap_stats = {
        "gap_count": int(len(gaps)), "max_gap_bins": int(max(gaps) if gaps else 0),
        "max_gap_seconds": float(max(gaps) * grid_ms / 1000.0 if gaps else 0.0),
        "gap_ge_30s": int(sum(length * grid_ms >= 30000 for length in gaps)),
        "gap_ge_60s": int(sum(length * grid_ms >= 60000 for length in gaps)),
        "gap_ge_90s": int(sum(length * grid_ms >= 90000 for length in gaps)),
        "gap_ge_120s": int(sum(length * grid_ms >= 120000 for length in gaps)),
        "gap_ge_300s": int(sum(length * grid_ms >= 300000 for length in gaps)),
        "gap_lengths_bins": gaps,
    }
    return quality, gap_stats


def _union_find_clusters(matrix: np.ndarray, names: Sequence[str], threshold: float) -> list[list[str]]:
    parent = list(range(len(names)))
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) >= threshold:
                union(i, j)
    clusters: MutableMapping[int, list[str]] = defaultdict(list)
    for index, name in enumerate(names):
        clusters[find(index)].append(name)
    return [sorted(group) for group in clusters.values() if len(group) > 1]


def _corr_pair(left: np.ndarray, right: np.ndarray, method: str) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return None
    left, right = left[valid], right[valid]
    if method == "spearman":
        left = pd.Series(left).rank(method="average").to_numpy(dtype=float)
        right = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _correlation_matrices(feature_map: Mapping[str, np.ndarray]) -> Tuple[list[str], np.ndarray, np.ndarray]:
    names = []
    vectors = []
    for name in sorted(feature_map):
        values = np.asarray(feature_map[name], dtype=np.float64)
        finite = np.isfinite(values)
        if finite.sum() <= 1:
            continue
        replacement = float(np.nanmedian(values[finite]))
        names.append(name)
        vectors.append(np.where(finite, values, replacement))
    if not vectors:
        empty = np.empty((0, 0))
        return names, empty, empty
    matrix = np.asarray(vectors, dtype=np.float64)
    return names, np.corrcoef(matrix), np.corrcoef(
        pd.DataFrame(matrix).rank(axis=1, method="average").to_numpy(dtype=np.float64)
    )


def _threshold_summary(names: Sequence[str], matrix: np.ndarray, thresholds: Sequence[float]) -> Mapping[str, object]:
    result = {}
    for threshold in thresholds:
        pairs = int(sum(abs(matrix[i, j]) >= threshold for i in range(len(names)) for j in range(i + 1, len(names)) if np.isfinite(matrix[i, j])))
        result[str(threshold)] = {
            "absolute_pair_count": pairs,
            "clusters": _union_find_clusters(matrix, names, threshold),
        }
    return result


def _clusters_from_pairs(pairs: Iterable[Tuple[str, str]]) -> list[list[str]]:
    names = sorted({name for pair in pairs for name in pair})
    index = {name: position for position, name in enumerate(names)}
    parent = list(range(len(names)))
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    for left, right in pairs:
        a, b = find(index[left]), find(index[right])
        if a != b:
            parent[b] = a
    groups: MutableMapping[int, list[str]] = defaultdict(list)
    for name, position in index.items():
        groups[find(position)].append(name)
    return [sorted(group) for group in groups.values() if len(group) > 1]


def audit_metrics(metric_root: Path, output: Path, train_start: int, train_end: int, chunk_rows: int) -> Mapping[str, object]:
    registry, groups = _metric_registry(metric_root, train_start, train_end)
    registry.to_csv(output / "metric_registry.csv", index=False)
    grid = np.arange((train_start // 30000) * 30000, train_end, 30000, dtype=np.int64)
    quality_rows = []
    gap_rows = []
    multicore_rows = []
    # Keep one aligned vector per service/feature for per-service and physical
    # host correlation.  The bound is the GAIA registry (10 services x 474
    # logical names x 60,984 Train bins), not the raw row count.
    service_arrays: MutableMapping[str, MutableMapping[str, np.ndarray]] = defaultdict(dict)
    for service in GAIA_SERVICES:
        for feature, full_names in sorted(groups[service].items()):
            train_groups = []
            timestamp_parts = []
            for full_name, paths in sorted(full_names.items()):
                array, timestamps = _read_metric_group(paths, feature, train_start, train_end, grid, chunk_rows)
                train_groups.append(array)
                timestamp_parts.append(timestamps)
            if not train_groups:
                continue
            stack = np.asarray(train_groups, dtype=np.float32)
            combined = np.nanmean(stack, axis=0) if len(train_groups) > 1 else stack[0]
            quality, gap_stats = _quality(combined, np.concatenate(timestamp_parts) if timestamp_parts else np.empty(0, dtype=np.int64))
            scope = "host_of_node" if feature.startswith("host_") else "global_container"
            base = {"service": service, "scope": scope, "logical_feature": feature, "source_count": len(train_groups)}
            quality_rows.append({**base, **quality})
            gap_rows.append({**base, **{key: value for key, value in gap_stats.items() if key != "gap_lengths_bins"}, "gap_lengths_bins": json.dumps(gap_stats["gap_lengths_bins"])})
            if scope == "global_container" or feature.startswith("host_"):
                service_arrays[service][feature] = combined
            if len(train_groups) > 1:
                mean_values = np.nanmean(stack, axis=0)
                max_values = np.nanmax(stack, axis=0)
                valid = np.isfinite(mean_values) & np.isfinite(max_values)
                correlation = _corr_pair(mean_values, max_values, "pearson")
                spearman = _corr_pair(mean_values, max_values, "spearman")
                multicore_rows.append({**base, "core_count": len(train_groups), "mean_max_pearson": correlation,
                                       "mean_max_spearman": spearman,
                                       "mean_q95": float(np.nanquantile(mean_values, 0.95)) if np.isfinite(mean_values).any() else None,
                                       "max_q95": float(np.nanquantile(max_values, 0.95)) if np.isfinite(max_values).any() else None})
    quality_frame = pd.DataFrame(quality_rows)
    gap_frame = pd.DataFrame(gap_rows)
    quality_frame.to_csv(output / "metric_quality_train.csv", index=False)
    gap_frame.to_csv(output / "metric_missing_gaps_train.csv", index=False)
    pd.DataFrame(multicore_rows).to_csv(output / "metric_multicore_train.csv", index=False)

    # Average same logical feature over available nodes for a compact, label-free
    # correlation audit.  Correlation is descriptive; no representatives are selected here.
    feature_arrays: MutableMapping[str, list[np.ndarray]] = defaultdict(list)
    for arrays in service_arrays.values():
        for feature, values in arrays.items():
            feature_arrays[feature].append(values)
    names = sorted(feature_arrays)
    vectors = []
    valid_names = []
    for name in names:
        arrays = feature_arrays[name]
        vector = np.nanmean(np.asarray(arrays, dtype=np.float32), axis=0)
        if np.isfinite(vector).sum() > 1:
            vector = np.where(np.isfinite(vector), vector, np.nanmedian(vector[np.isfinite(vector)]))
            vectors.append(vector)
            valid_names.append(name)
    thresholds = (0.90, 0.95, 0.98, 0.995)
    if vectors:
        vector_matrix = np.asarray(vectors, dtype=np.float64)
        pearson = np.corrcoef(vector_matrix)
        spearman = np.corrcoef(pd.DataFrame(vector_matrix).rank(axis=1, method="average").to_numpy(dtype=np.float64))
    else:
        pearson = spearman = np.empty((0, 0))
    payload = {
        "schema": "metric_train_correlation_audit_v1", "feature_count": len(valid_names),
        "aggregation": "mean over available services, Train aligned 30s bins; descriptive only",
        "thresholds": {}, "selection_performed": False,
    }
    for threshold in thresholds:
        pearson_pairs = int(sum(abs(pearson[i, j]) >= threshold for i in range(len(valid_names)) for j in range(i + 1, len(valid_names)) if np.isfinite(pearson[i, j])))
        spearman_pairs = int(sum(abs(spearman[i, j]) >= threshold for i in range(len(valid_names)) for j in range(i + 1, len(valid_names)) if np.isfinite(spearman[i, j])))
        payload["thresholds"][str(threshold)] = {
            "pearson_absolute_pair_count": pearson_pairs,
            "pearson_clusters": _union_find_clusters(pearson, valid_names, threshold),
            "spearman_absolute_pair_count": spearman_pairs,
            "spearman_clusters": _union_find_clusters(spearman, valid_names, threshold),
        }
    # Per-service summaries prevent a strong signal in one node from being
    # mistaken for a cross-node redundant family.
    payload["per_service"] = {}
    global_support = {
        "pearson": {threshold: Counter() for threshold in thresholds},
        "spearman": {threshold: Counter() for threshold in thresholds},
    }
    for service in GAIA_SERVICES:
        service_features = {name: values for name, values in service_arrays[service].items() if not name.startswith("host_")}
        service_names, service_pearson, service_spearman = _correlation_matrices(service_features)
        payload["per_service"][service] = {
            "feature_count": len(service_names),
            "pearson": _threshold_summary(service_names, service_pearson, thresholds),
            "spearman": _threshold_summary(service_names, service_spearman, thresholds),
        }
        for method, matrix in (("pearson", service_pearson), ("spearman", service_spearman)):
            for threshold in thresholds:
                for i in range(len(service_names)):
                    for j in range(i + 1, len(service_names)):
                        if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) >= threshold:
                            global_support[method][threshold][(service_names[i], service_names[j])] += 1
    consensus = {}
    for threshold in thresholds:
        threshold_result = {"required_service_count": 8}
        for method in ("pearson", "spearman"):
            support = global_support[method][threshold]
            selected = [pair for pair, count in support.items() if count >= 8]
            threshold_result[method] = {
                "consensus_pair_count": len(selected),
                "consensus_clusters": _clusters_from_pairs(selected),
                "support_histogram": {str(number): int(sum(value == number for value in support.values())) for number in range(1, len(GAIA_SERVICES) + 1)},
            }
        consensus[str(threshold)] = threshold_result
    payload["global_consensus_at_least_8_of_10_services"] = consensus
    host_by_physical_host = {}
    for ip, mapped_services in sorted(GAIA_IP_MAP.items()):
        available = [service for service in mapped_services if service in GAIA_SERVICES and any(name.startswith("host_") for name in service_arrays[service])]
        representative = available[0] if available else None
        host_features = {name: values for name, values in service_arrays[representative].items() if name.startswith("host_")} if representative else {}
        host_names, host_pearson, host_spearman = _correlation_matrices(host_features)
        host_by_physical_host[ip] = {
            "mapped_services": available, "available_service_count": len(available),
            "representative_service": representative, "feature_count": len(host_names),
            "pearson": _threshold_summary(host_names, host_pearson, thresholds),
            "spearman": _threshold_summary(host_names, host_spearman, thresholds),
            "deduplication": "one representative service per physical host; no service replication in correlation",
        }
    payload["host_by_physical_host"] = host_by_physical_host
    _json_dump(output / "metric_correlation_clusters_train.json", payload)
    return {"registry_rows": int(len(registry)), "quality_rows": int(len(quality_frame)), "feature_count_for_correlation": len(valid_names)}


def _iter_log_messages(path: Path, chunk_rows: int) -> Iterable[Tuple[pd.Series, np.ndarray, np.ndarray]]:
    for chunk in pd.read_csv(path, usecols=["message"], chunksize=int(chunk_rows), keep_default_na=False, on_bad_lines="error"):
        messages = chunk["message"].astype("string")
        timestamps, valid = parse_message_timestamp_ms(messages)
        yield messages, timestamps, valid


def audit_logs(log_root: Path, output: Path, train_start: int, train_end: int, test_end: int, config_path: Path, chunk_rows: int) -> Mapping[str, object]:
    config = config_path.parent.parent.parent / "util/GAIA/gaia.ini"
    fit = fit_train_templates(log_root, train_start, train_end, config_path=config, chunk_rows=chunk_rows)
    miner = _miner_from_config(config)
    _restore_drain(miner, fit["state"])
    cluster_ids = tuple(int(value) for value in fit["cluster_ids"])
    cluster_set = set(cluster_ids)
    records: MutableMapping[int, Counter] = defaultdict(Counter)
    active_node_bins: MutableMapping[int, set[Tuple[str, int]]] = defaultdict(set)
    active_time_bins: MutableMapping[int, set[int]] = defaultdict(set)
    services_seen: MutableMapping[int, set[str]] = defaultdict(set)
    level_counts: MutableMapping[int, Counter] = defaultdict(Counter)
    train_level_counts = Counter()
    train_total = 0
    train_unseen = 0
    for service in GAIA_SERVICES:
        source = log_root / ("business_table_{}_2021-07.csv".format(service))
        for messages, timestamps, valid in _iter_log_messages(source, chunk_rows):
            selected = valid & (timestamps >= train_start) & (timestamps < train_end)
            for message, timestamp in zip(messages.to_numpy(dtype=str)[selected], timestamps[selected]):
                result = miner.match(message_payload(message), full_search_strategy="always")
                cluster_id = int(result.cluster_id) if result is not None and int(result.cluster_id) in cluster_set else -1
                if cluster_id < 0:
                    train_unseen += 1
                records[cluster_id]["records"] += 1
                active_node_bins[cluster_id].add((service, int(timestamp // 30000)))
                active_time_bins[cluster_id].add(int(timestamp // 30000))
                services_seen[cluster_id].add(service)
                level = message_level(message)
                level_counts[cluster_id][level] += 1
                train_level_counts[level] += 1
                train_total += 1
    rows = []
    template_map = {int(key): value for key, value in fit["templates"].items()}
    for cluster_id in cluster_ids:
        counter = records[cluster_id]
        rows.append({"cluster_id": cluster_id, "template": template_map.get(cluster_id, ""),
                     "records": int(counter["records"]), "active_node_bins": len(active_node_bins[cluster_id]),
                     "active_time_bins": len(active_time_bins[cluster_id]),
                     "active_services": len(services_seen[cluster_id]),
                     "active_days": len({bin_id // 2880 for bin_id in active_time_bins[cluster_id]}),
                     "level_INFO": level_counts[cluster_id]["INFO"], "level_WARNING": level_counts[cluster_id]["WARNING"],
                     "level_ERROR": level_counts[cluster_id]["ERROR"], "level_DEBUG": level_counts[cluster_id]["DEBUG"],
                     "level_UNKNOWN": level_counts[cluster_id]["UNKNOWN"],
                     "diagnostic_rare_records_le_5": bool(counter["records"] <= 5)})
    pd.DataFrame(rows).to_csv(output / "log_templates_train.csv", index=False)
    def _histogram(values: Sequence[int], bins: Sequence[Tuple[str, int | None, int | None]]) -> Mapping[str, int]:
        result = {}
        for label, lower, upper in bins:
            result[label] = int(sum((value >= lower) and (upper is None or value <= upper) for value in values))
        return result

    record_values = [row["records"] for row in rows]
    service_values = [row["active_services"] for row in rows]
    node_bin_values = [row["active_node_bins"] for row in rows]
    time_bin_values = [row["active_time_bins"] for row in rows]
    distribution = {
        "schema": "log_train_template_distribution_v1", "fit_split": "train",
        "train_rows_fitted": int(fit["train_rows_fitted"]), "train_rows_transformed": int(train_total),
        "train_unseen_template_rows": int(train_unseen), "cluster_count": len(cluster_ids),
        "level_counts": dict(train_level_counts),
        "records_quantiles": {str(q): float(np.quantile([row["records"] for row in rows], q)) if rows else 0.0 for q in (0, .5, .9, .99, 1)},
        "records_histogram": _histogram(record_values, (("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21-100", 21, 100), (">100", 101, None))),
        "active_service_count_histogram": _histogram(service_values, (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10))),
        "active_node_bin_count_histogram": _histogram(node_bin_values, (("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21-100", 21, 100), (">100", 101, None))),
        "active_global_time_bin_count_histogram": _histogram(time_bin_values, (("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21-100", 21, 100), (">100", 101, None))),
        "rare_diagnostic_only": "records<=5; not a frozen selection rule",
        "selection_performed": False,
    }
    _json_dump(output / "log_template_distribution_train.json", distribution)
    # A second frozen matcher pass is deliberately Test-transform-only.  It is
    # reported, but never used to choose stable/rare/UNK dimensions or scales.
    test_counts = Counter(); test_levels = Counter(); test_unseen_by_level = Counter(); test_rows = 0
    test_unseen = 0
    test_miner = _miner_from_config(config); _restore_drain(test_miner, fit["state"])
    for service in GAIA_SERVICES:
        source = log_root / ("business_table_{}_2021-07.csv".format(service))
        for messages, timestamps, valid in _iter_log_messages(source, chunk_rows):
            selected = valid & (timestamps >= train_end) & (timestamps < test_end)
            for message in messages.to_numpy(dtype=str)[selected]:
                result = test_miner.match(message_payload(message), full_search_strategy="always")
                level = message_level(message)
                if result is None or int(result.cluster_id) not in cluster_set:
                    test_unseen += 1; test_counts[service] += 1; test_unseen_by_level[level] += 1
                test_levels[level] += 1; test_rows += 1
    _json_dump(output / "log_frozen_transform_stats.json", {
        "schema": "log_frozen_transform_stats_v1", "vocabulary_fit_split": "train",
        "transform_split": "test", "test_rows_transformed": int(test_rows),
        "test_unseen_template_rows": int(test_unseen), "test_unseen_by_service": dict(test_counts),
        "test_unseen_by_level": dict(test_unseen_by_level),
        "test_level_counts": dict(test_levels), "test_used_for_selection": False,
    })
    return {"cluster_count": len(cluster_ids), "train_rows": train_total, "test_rows": test_rows}


def _trace_time_ms(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    parsed = pd.to_datetime(values.astype("string"), errors="coerce")
    localized = parsed.dt.tz_localize("Asia/Shanghai", ambiguous="NaT", nonexistent="NaT")
    valid = localized.notna().to_numpy()
    return localized.astype("int64").to_numpy(dtype=np.int64) // 1_000_000, valid


def _trace_pair_hash(trace_id: object, span_id: object) -> bytes:
    """Stable 128-bit key for a trace-local span identity."""
    return hashlib.blake2b(
        (str(trace_id) + "\0" + str(span_id)).encode("utf-8"), digest_size=16
    ).digest()


def _reservoir_add(sample: list[float], value: float, seen: int, limit: int = 512) -> None:
    """Deterministic, bounded reservoir update (no global duration list)."""
    if len(sample) < limit:
        sample.append(float(value))
        return
    # A deterministic LCG gives a reproducible replacement position while
    # keeping at most ``limit`` values per edge/global stream.
    position = ((int(seen) * 1103515245 + 12345) & 0x7FFFFFFF) % int(seen)
    if position < limit:
        sample[position] = float(value)


def _open_parent_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE IF NOT EXISTS parent (key BLOB PRIMARY KEY, service TEXT NOT NULL)")
    connection.execute("CREATE TEMP TABLE IF NOT EXISTS query_keys (key BLOB PRIMARY KEY)")
    return connection


def _parent_lookup(connection: sqlite3.Connection, keys: Sequence[bytes]) -> Mapping[bytes, str]:
    connection.execute("DELETE FROM query_keys")
    connection.executemany("INSERT OR IGNORE INTO query_keys(key) VALUES (?)", ((key,) for key in keys))
    return dict(connection.execute(
        "SELECT query_keys.key, parent.service FROM query_keys JOIN parent USING(key)"
    ).fetchall())


def audit_traces(trace_root: Path, output: Path, train_start: int, train_end: int, chunk_rows: int) -> Mapping[str, object]:
    # First pass: split-local on-disk parent map keyed by a 128-bit digest of
    # (trace_id, span_id), never by span_id alone.  A SQLite B-tree keeps RAM
    # bounded even for tens of millions of spans; only one input chunk is held.
    index_path = output / ".trace_parent_train.sqlite3"
    # A killed prior run must never be treated as a resumable parent map: the
    # audit is intentionally recomputed from the current source binding.
    try:
        index_path.unlink()
    except FileNotFoundError:
        pass
    connection = _open_parent_index(index_path)
    duplicate_keys = 0; parent_rows = 0; invalid_parent_rows = 0
    for source in sorted(trace_root.glob("trace_table_*_2021-07.csv")):
        for chunk in pd.read_csv(source, usecols=["trace_id", "span_id", "service_name", "end_time"], chunksize=chunk_rows, keep_default_na=False, on_bad_lines="error"):
            end_ms, valid_time = _trace_time_ms(chunk["end_time"])
            selected = valid_time & (end_ms >= train_start) & (end_ms < train_end)
            rows = []
            for trace_id, span_id, service, valid in zip(chunk["trace_id"].astype(str), chunk["span_id"].astype(str), chunk["service_name"].astype(str), selected):
                if not valid or service not in GAIA_SERVICES:
                    if valid: invalid_parent_rows += 1
                    continue
                rows.append((_trace_pair_hash(trace_id, span_id), service))
            if rows:
                before = int(connection.execute("SELECT COUNT(*) FROM parent").fetchone()[0])
                connection.executemany("INSERT OR IGNORE INTO parent(key, service) VALUES (?, ?)", rows)
                connection.commit()
                after = int(connection.execute("SELECT COUNT(*) FROM parent").fetchone()[0])
                inserted = after - before
                parent_rows += inserted
                duplicate_keys += len(rows) - inserted
    edge_stats: MutableMapping[Tuple[str, str, str], MutableMapping[str, object]] = defaultdict(
        lambda: {"count": 0, "duration_sum": 0.0, "duration_min": math.inf,
                 "duration_max": -math.inf, "sample": [], "bins": set(), "days": set()}
    )
    counters = Counter(); status_counts = Counter(); duration_sample: list[float] = []
    duration_seen = 0
    for source in sorted(trace_root.glob("trace_table_*_2021-07.csv")):
        for chunk in pd.read_csv(source, usecols=["trace_id", "span_id", "parent_id", "service_name", "start_time", "end_time", "status_code"], chunksize=chunk_rows, keep_default_na=False, on_bad_lines="error"):
            end_ms, end_valid = _trace_time_ms(chunk["end_time"])
            status = pd.to_numeric(chunk["status_code"], errors="coerce").to_numpy(dtype=float)
            status_int = np.where(np.isfinite(status), status, -1).astype(np.int64)
            starts = pd.to_datetime(chunk["start_time"], errors="coerce")
            ends = pd.to_datetime(chunk["end_time"], errors="coerce")
            duration = (ends - starts).dt.total_seconds().to_numpy(dtype=float)
            selected = end_valid & (end_ms >= train_start) & (end_ms < train_end) & np.isfinite(duration) & (duration >= 0) & np.isfinite(status) & np.isin(status_int, np.asarray([200, 300, 400, 500], dtype=np.int64))
            counters["rows_train"] += int(selected.sum())
            keys = [_trace_pair_hash(trace_id, parent_id) for trace_id, parent_id in zip(chunk["trace_id"].astype(str), chunk["parent_id"].astype(str))]
            found = _parent_lookup(connection, keys) if np.any(selected) else {}
            for key_hash, trace_id, parent_id, dst, end, code, seconds, valid in zip(keys, chunk["trace_id"].astype(str), chunk["parent_id"].astype(str), chunk["service_name"].astype(str), end_ms, status, duration, selected):
                if not valid: continue
                source_service = found.get(key_hash)
                counters["parent_resolved"] += int(source_service is not None)
                counters["parent_unmatched"] += int(source_service is None)
                if source_service is None or source_service == dst or dst not in GAIA_SERVICES:
                    counters["same_service_or_unusable"] += 1
                    continue
                code_text = str(int(code)); key = (source_service, dst, code_text)
                item = edge_stats[key]; item["count"] += 1; item["duration_sum"] += float(seconds)
                item["duration_min"] = min(float(item["duration_min"]), float(seconds))
                item["duration_max"] = max(float(item["duration_max"]), float(seconds))
                _reservoir_add(item["sample"], float(seconds), int(item["count"]))
                item["bins"].add(int(end // 30000)); item["days"].add(int(end // 86400000))
                status_counts[code_text] += 1; duration_seen += 1
                _reservoir_add(duration_sample, float(seconds), duration_seen)
    connection.close()
    try:
        index_path.unlink()
    except FileNotFoundError:
        pass
    edge_rows = []
    for (source, destination, status), item in sorted(edge_stats.items()):
        values = np.asarray(item["sample"], dtype=float)
        edge_rows.append({"src_service": source, "dst_service": destination, "status_code": status,
                          "call_count": int(item["count"]), "active_bins": len(item["bins"]), "active_days": len(item["days"]),
                          "duration_sum_seconds": float(item["duration_sum"]), "duration_mean_seconds": float(item["duration_sum"] / item["count"]),
                          "duration_min_seconds": float(item["duration_min"]), "duration_max_seconds": float(item["duration_max"]),
                          "duration_p50_seconds": float(np.quantile(values, .50)) if len(values) else None,
                          "duration_p95_seconds": float(np.quantile(values, .95)) if len(values) else None,
                          "duration_quantile_sample_size": int(len(values))})
    pd.DataFrame(edge_rows).to_csv(output / "trace_edges_train.csv", index=False)
    _json_dump(output / "trace_parent_resolution_train.json", {
        "schema": "trace_parent_resolution_train_v1", "key": "(trace_id, span_id)",
        "parent_map_split_local": True, "parent_rows": parent_rows, "duplicate_parent_keys": duplicate_keys,
        "invalid_parent_rows": invalid_parent_rows, **dict(counters), "graph_selection_performed": False,
    })
    values = np.asarray(duration_sample, dtype=float)
    observed_pairs = {(row["src_service"], row["dst_service"]) for row in edge_rows}
    pair_counts: MutableMapping[Tuple[str, str], MutableMapping[str, object]] = defaultdict(lambda: {"count": 0, "active_bins": set()})
    for (source, destination, _status), item in edge_stats.items():
        pair = pair_counts[(source, destination)]
        pair["count"] += int(item["count"])
        pair["active_bins"].update(item["bins"])
    low_frequency_curve = []
    for threshold in (1, 2, 3, 5, 10, 20, 50, 100):
        low_frequency_curve.append({
            "threshold": threshold,
            "directed_edges_call_count_le": int(sum(item["count"] <= threshold for item in pair_counts.values())),
            "directed_edges_active_bins_le": int(sum(len(item["active_bins"]) <= threshold for item in pair_counts.values())),
        })
    _json_dump(output / "trace_feature_distribution_train.json", {
        "schema": "trace_feature_distribution_train_v1", "status_counts": dict(status_counts),
        "possible_directed_edges": len(GAIA_SERVICES) * (len(GAIA_SERVICES) - 1),
        "directed_status_edge_rows": len(edge_rows), "directed_edges": len(observed_pairs),
        "edge_sparsity": 1.0 - (len(observed_pairs) / float(len(GAIA_SERVICES) * (len(GAIA_SERVICES) - 1))),
        "low_frequency_curve": low_frequency_curve,
        "duration_summary_seconds": {"count": int(duration_seen), "mean": float(sum(float(item["duration_sum"]) for item in edge_stats.values()) / duration_seen) if duration_seen else None,
                                      "p50": float(np.quantile(values, .5)) if len(values) else None,
                                      "p95": float(np.quantile(values, .95)) if len(values) else None,
                                      "max": float(max(float(item["duration_max"]) for item in edge_stats.values())) if edge_stats else None,
                                      "quantile_sample_size": int(len(values)), "quantile_method": "deterministic bounded reservoir"},
        "features_audited": ["call_count", "duration_sum", "duration_mean", "duration_p50", "duration_p95"],
        "selection_performed": False,
    })
    return {"train_rows": int(counters["rows_train"]), "parent_unmatched": int(counters["parent_unmatched"]), "edge_rows": len(edge_rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v3.json")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--chunk-rows", type=int, default=150000)
    parser.add_argument("--workers", type=int, default=1, help="Reserved audit metadata; full audit is deterministic and serial.")
    parser.add_argument("--start-method", choices=("spawn", "forkserver"), default="spawn")
    parser.add_argument("--full", action="store_true", help="Run streaming Train audit and frozen Test log transform statistics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_rows < 1 or args.workers < 1:
        raise ValueError("chunk-rows and workers must be positive")
    config_path = Path(args.config).resolve()
    if args.split != "train":
        raise ValueError("this audit's decisions and outputs are Train-only; use --split train")
    raw_root = Path(args.raw_root).resolve()
    output = Path(args.output_root).resolve()
    if output == raw_root or output == (raw_root / "data/p5/v3/ad").resolve():
        raise ValueError("audit output must be independent of the raw root and formal detector output")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = _source_binding(config_path, raw_root, config, args.split)
    _json_dump(output / "source_binding.json", source)
    manifest = {
        "schema": SCRIPT_SCHEMA, "status": "FULL_AUDIT_PENDING" if args.full else "METADATA_ONLY",
        "full_scan_executed": bool(args.full), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": source["config_sha256"], "run_table_sha256": source["run_table_sha256"],
        "decision_inputs": ["train"], "gt_labels_used": False, "test_used_for_selection": False,
        "split": args.split, "chunk_rows": args.chunk_rows, "workers": args.workers, "start_method": args.start_method,
        "output_root": str(output), "outputs": [], "results": {},
    }
    if args.full:
        train_start, train_end = _config_split(config, "train")
        manifest["results"]["metric"] = audit_metrics(raw_root / "metric/metric_split/metric", output, train_start, train_end, args.chunk_rows)
        _, test_end = _config_split(config, "test")
        manifest["results"]["logs"] = audit_logs(raw_root / "business/business_split/business", output, train_start, train_end, test_end, config_path, args.chunk_rows)
        manifest["results"]["traces"] = audit_traces(raw_root / "trace/trace_split/trace", output, train_start, train_end, args.chunk_rows)
        manifest["status"] = "AUDIT_FULL_COMPLETE"
    manifest["outputs"] = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "audit_manifest.json")
    _json_dump(output / "audit_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output_root": str(output), "outputs": manifest["outputs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
