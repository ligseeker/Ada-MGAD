"""Deterministic raw-GAIA preprocessing for timestamped Ada-MGAD-G input.

The detector architecture and loss are untouched.  This adapter replaces the
historical generic label path with the frozen supported-injection registry,
fits telemetry scaling on Train only, and materializes split-local arrays so no
window can cross a chronological boundary.
"""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

from util.GAIA.constant import GAIA_MULTI_CORE_PREFIXES
from util.GAIA.pre_GAIA import (
    _metric_duplicate_reduce_mode,
    _parse_metric_filename,
    _reduce_duplicate_timestamp_values,
    _target_services_for_metric,
)

from .ad_data import (
    build_registry_node_labels,
    build_semisupervised_mask,
    grid_for_block,
    save_split_arrays,
)
from .protocol import (
    GAIA_SERVICES,
    assign_event_blocks,
    layout_digest,
    load_registry,
    sha256_file,
    temporal_blocks,
    write_json,
)


LOG_FEATURES = (
    "level_INFO", "level_WARNING", "level_ERROR", "level_DEBUG",
    "level_UNKNOWN", "log_total",
)
TRACE_STATUS_CODES = ("200", "300", "400", "500")
HASH_DTYPE = np.dtype([("h1", "<u8"), ("h2", "<u8")])


def _read_metric_group_strict(full_name: str, paths: Sequence[Path], feature: str) -> pd.DataFrame:
    """Read every E2E metric shard and fail rather than accept partial input."""

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = {"timestamp", "value"} - set(frame.columns)
        if missing:
            raise ValueError(
                "metric shard {} for {} lacks columns {}".format(path, full_name, sorted(missing))
            )
        frames.append(frame)
    if not frames:
        raise ValueError("metric group {} contains no source shards".format(full_name))
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp", "value"])
    merged["timestamp"] = pd.to_numeric(merged["timestamp"], errors="coerce")
    merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
    merged = merged.dropna(subset=["timestamp", "value"])
    if merged.duplicated(subset=["timestamp"]).any():
        reduce_mode = _metric_duplicate_reduce_mode(feature)
        merged = merged.groupby("timestamp", as_index=False)["value"].agg(
            lambda values: _reduce_duplicate_timestamp_values(values, reduce_mode)
        )
    return merged.sort_values("timestamp").reset_index(drop=True)


def _logical_metric_schema(metric_dir: Path, excluded_features: Sequence[str] = ()):
    by_service = {service: defaultdict(lambda: defaultdict(list)) for service in GAIA_SERVICES}
    for path in sorted(Path(metric_dir).glob("*.csv")):
        info = _parse_metric_filename(path.name)
        if info is None:
            continue
        for service, feature in _target_services_for_metric(info):
            if service not in by_service:
                continue
            base = feature
            if any(prefix in feature for prefix in GAIA_MULTI_CORE_PREFIXES):
                base = re.sub(r"core_\d+", "core_X", feature)
            if base not in set(excluded_features):
                by_service[service][base][info["full_name"]].append(path)
    common = set.intersection(*(set(by_service[service]) for service in GAIA_SERVICES))
    return by_service, tuple(sorted(common))


def _aligned_mean(frame: pd.DataFrame, grid: np.ndarray, grid_ms: int) -> np.ndarray:
    result = np.full(len(grid), np.nan, dtype=np.float32)
    if frame is None or frame.empty:
        return result
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(timestamps) & np.isfinite(values)
    if not np.any(valid):
        return result
    aligned = (timestamps[valid].astype(np.int64) // grid_ms) * grid_ms
    compact = pd.DataFrame({"timestamp": aligned, "value": values[valid]}).groupby(
        "timestamp", sort=True
    )["value"].mean()
    compact_ts = compact.index.to_numpy(dtype=np.int64)
    positions = np.searchsorted(grid, compact_ts)
    inside = positions < len(grid)
    exact = np.zeros(len(positions), dtype=bool)
    exact[inside] = grid[positions[inside]] == compact_ts[inside]
    result[positions[exact]] = compact.to_numpy(dtype=np.float32)[exact]
    return result


def _quality(series: np.ndarray) -> Mapping[str, object]:
    finite = np.asarray(series, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"coverage": 0.0, "unique_count": 0, "dynamic_span": 0.0, "dynamic_ratio": 0.0, "keep": False}
    q05, q95 = np.quantile(finite, [0.05, 0.95]) if finite.size > 1 else (finite[0], finite[0])
    span = float(q95 - q05)
    ratio = float(span / (float(np.mean(np.abs(finite))) + 1e-6))
    record = {
        "coverage": float(finite.size / len(series)),
        "unique_count": int(pd.Series(finite).nunique(dropna=True)),
        "dynamic_span": span,
        "dynamic_ratio": ratio,
    }
    record["keep"] = bool(
        record["coverage"] >= 0.20
        and record["unique_count"] >= 2
        and record["dynamic_span"] >= 1e-10
        and record["dynamic_ratio"] >= 1e-4
    )
    return record


def _forward_fill_by_split(values: np.ndarray, slices: Mapping[str, slice]) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    for block_slice in slices.values():
        output[block_slice] = pd.Series(output[block_slice]).ffill(limit=10).to_numpy(dtype=np.float32)
    return output


def build_metric_arrays(
    metric_dir: Path,
    grid: np.ndarray,
    slices: Mapping[str, slice],
    cache_dir: Path,
    excluded_features: Sequence[str] = (),
    progress_every: int = 25,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    """Preserve the historical metric schema/aggregation with Train-only fit."""

    groups, schema_common = _logical_metric_schema(metric_dir, excluded_features)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_slice = slices["train"]
    grid_ms = 30000
    series: Dict[Tuple[str, str], np.ndarray] = {}
    health: Dict[str, Dict[str, Mapping[str, object]]] = {service: {} for service in GAIA_SERVICES}
    total = len(GAIA_SERVICES) * len(schema_common)
    processed = 0
    for service in GAIA_SERVICES:
        for base in schema_common:
            source_records = []
            for full_name, paths in sorted(groups[service][base].items()):
                source_records.append({
                    "full_name": full_name,
                    "files": [
                        {
                            "path": str(Path(path).resolve()),
                            "size": Path(path).stat().st_size,
                            "mtime_ns": Path(path).stat().st_mtime_ns,
                        }
                        for path in sorted(paths)
                    ],
                })
            cache_binding = {
                "service": service,
                "feature": base,
                "grid_start_ms": int(grid[0]),
                "grid_end_ms": int(grid[-1]),
                "grid_rows": len(grid),
                "sources": source_records,
            }
            cache_key = hashlib.sha256(
                json.dumps(cache_binding, sort_keys=True).encode("utf-8")
            ).hexdigest()
            cache_path = cache_dir / (cache_key + ".npy")
            binding_path = cache_dir / (cache_key + ".json")
            if cache_path.is_file() and binding_path.is_file():
                cached_binding = json.loads(binding_path.read_text(encoding="utf-8"))
                logical = np.load(cache_path, allow_pickle=False)
                if cached_binding != cache_binding or logical.shape != (len(grid),):
                    raise ValueError("metric resume-cache binding mismatch")
            else:
                sums = np.zeros(len(grid), dtype=np.float64)
                counts = np.zeros(len(grid), dtype=np.int16)
                for full_name, paths in sorted(groups[service][base].items()):
                    frame = _read_metric_group_strict(full_name, paths, base)
                    aligned = _aligned_mean(frame, grid, grid_ms)
                    observed = np.isfinite(aligned)
                    sums[observed] += aligned[observed]
                    counts[observed] += 1
                logical = np.full(len(grid), np.nan, dtype=np.float32)
                observed = counts > 0
                logical[observed] = (sums[observed] / counts[observed]).astype(np.float32)
                np.save(cache_path, logical, allow_pickle=False)
                binding_path.write_text(
                    json.dumps(cache_binding, sort_keys=True) + "\n", encoding="utf-8"
                )
            series[(service, base)] = logical
            health[service][base] = _quality(logical[train_slice])
            processed += 1
            if progress_every and (processed % progress_every == 0 or processed == total):
                logging.info("Ada-MGAD metric logical series %d/%d", processed, total)

    kept = tuple(sorted(
        base for base in schema_common
        if all(bool(health[service][base]["keep"]) for service in GAIA_SERVICES)
    ))
    if not kept:
        raise ValueError("Train-only metric quality filter removed every shared feature")
    output = np.zeros((len(grid), len(GAIA_SERVICES), len(kept)), dtype=np.float32)
    normalization = {}
    for service_index, service in enumerate(GAIA_SERVICES):
        for feature_index, base in enumerate(kept):
            values = _forward_fill_by_split(series[(service, base)], slices)
            train = values[train_slice]
            finite_train = train[np.isfinite(train)]
            if finite_train.size == 0:
                raise ValueError("kept metric feature lacks finite Train values")
            minimum = float(np.min(finite_train))
            maximum = float(np.max(finite_train))
            scale = maximum - minimum
            if scale == 0:
                scale = 1.0
            transformed = (values.astype(np.float64) - minimum) / scale
            output[:, service_index, feature_index] = np.nan_to_num(
                transformed, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            normalization["{}::{}".format(service, base)] = {
                "train_min": minimum, "train_max": maximum, "scale": scale
            }
    stats = {
        "schema_rule": "filename/service registry only; labels are unavailable to mapping",
        "globally_excluded_features": list(excluded_features),
        "shared_schema_features_before_train_quality": len(schema_common),
        "features_per_node_after_train_quality": len(kept),
        "feature_names": list(kept),
        "quality_thresholds": {
            "coverage": 0.20, "unique_values": 2,
            "dynamic_span": 1e-10, "dynamic_ratio": 1e-4,
        },
        "quality_fit_split": "train",
        "normalization_fit_split": "train",
        "normalization": normalization,
        "resume_cache": {
            "path": str(cache_dir.resolve()),
            "policy": "source-metadata and grid bound; execution-only and excluded from Git",
        },
    }
    return output, stats


def _local_ms(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    parsed = pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S,%f", errors="coerce")
    localized = parsed.dt.tz_localize("Asia/Shanghai", ambiguous="NaT", nonexistent="NaT")
    valid = localized.notna().to_numpy()
    numeric = localized.astype("int64").to_numpy(dtype=np.int64) // 1_000_000
    return numeric, valid


def _grid_positions(timestamps_ms: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    aligned = (timestamps_ms // 30000) * 30000
    positions = np.searchsorted(grid, aligned)
    valid = positions < len(grid)
    exact = np.zeros(len(positions), dtype=bool)
    exact[valid] = grid[positions[valid]] == aligned[valid]
    return positions, exact


def build_log_arrays(
    business_dir: Path,
    grid: np.ndarray,
    slices: Mapping[str, slice],
    chunk_rows: int = 500000,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    """Vectorize GAIA message-prefix time and numeric level/volume indicators."""

    values = np.zeros((len(grid), len(GAIA_SERVICES), len(LOG_FEATURES)), dtype=np.float32)
    rows_scanned = 0
    invalid_timestamps = 0
    retained = 0
    for service_index, service in enumerate(GAIA_SERVICES):
        path = Path(business_dir) / "business_table_{}_2021-07.csv".format(service)
        for chunk_index, chunk in enumerate(pd.read_csv(
            path, usecols=["message"], chunksize=chunk_rows,
            keep_default_na=False, on_bad_lines="skip",
        ), start=1):
            messages = chunk["message"].astype("string")
            rows_scanned += len(messages)
            numeric, valid_time = _local_ms(messages.str.slice(0, 23))
            invalid_timestamps += int((~valid_time).sum())
            positions, in_grid = _grid_positions(numeric, grid)
            wanted = valid_time & in_grid
            if not np.any(wanted):
                continue
            selected = messages[wanted]
            selected_pos = positions[wanted]
            levels = np.full(len(selected), 4, dtype=np.int64)
            for level_index, level in enumerate(("INFO", "WARNING", "ERROR", "DEBUG")):
                is_level = selected.str.contains(
                    "| {} |".format(level), regex=False, na=False
                ).to_numpy(dtype=bool, na_value=False)
                levels[is_level] = level_index
            np.add.at(values, (selected_pos, service_index, levels), 1.0)
            np.add.at(values, (selected_pos, service_index, np.full(len(selected_pos), 5)), 1.0)
            retained += len(selected_pos)
            if chunk_index % 20 == 0:
                logging.info("Ada-MGAD logs %s rows %,d", service, rows_scanned)

    train = values[slices["train"]]
    maxima = np.max(train, axis=(0, 1))
    maxima[maxima == 0] = 1.0
    values /= maxima[None, None, :]
    return values, {
        "raw_rows_scanned": rows_scanned,
        "invalid_message_prefix_timestamps": invalid_timestamps,
        "retained_protocol_rows": retained,
        "feature_names": list(LOG_FEATURES),
        "timestamp_source": "message prefix YYYY-MM-DD HH:MM:SS,mmm",
        "raw_text_model_input": False,
        "normalization_fit_split": "train",
        "train_maxima": maxima.astype(float).tolist(),
        "adapter_compromise": "numeric level counts plus total count; no full-month Drain3 template learning",
    }


def _hash_pair(values: pd.Series) -> np.ndarray:
    strings = values.astype("string").fillna("")
    result = np.empty(len(strings), dtype=HASH_DTYPE)
    result["h1"] = pd.util.hash_pandas_object(
        strings, index=False, hash_key="p5i1spanhashkey1"
    ).to_numpy(dtype=np.uint64)
    result["h2"] = pd.util.hash_pandas_object(
        strings, index=False, hash_key="p5i1spanhashkey2"
    ).to_numpy(dtype=np.uint64)
    return result


def _membership(reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(reference, queries)
    inside = positions < len(reference)
    result = np.zeros(len(queries), dtype=bool)
    result[inside] = reference[positions[inside]] == queries[inside]
    return result


def _build_span_hashes(
    trace_dir: Path, cache_dir: Path, chunk_rows: int
) -> Tuple[Mapping[str, np.ndarray], Mapping[str, object]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    stats = {}
    for service in GAIA_SERVICES:
        source = Path(trace_dir) / "trace_table_{}_2021-07.csv".format(service)
        parts = []
        rows = 0
        for chunk in pd.read_csv(
            source, usecols=["span_id"], chunksize=chunk_rows,
            keep_default_na=False, on_bad_lines="skip",
        ):
            rows += len(chunk)
            parts.append(_hash_pair(chunk["span_id"]))
        hashes = np.concatenate(parts) if parts else np.empty(0, dtype=HASH_DTYPE)
        unique = np.unique(hashes)
        path = cache_dir / (service + ".npy")
        np.save(path, unique, allow_pickle=False)
        arrays[service] = np.load(path, mmap_mode="r")
        stats[service] = {
            "span_rows": rows,
            "unique_128bit_span_hashes": len(unique),
            "duplicate_span_rows": rows - len(unique),
            "cache_path": str(path.resolve()),
            "cache_sha256": sha256_file(path),
        }
        logging.info("Ada-MGAD trace span index %s: %,d rows", service, rows)
    return arrays, stats


def build_trace_arrays(
    trace_dir: Path,
    cache_dir: Path,
    grid: np.ndarray,
    slices: Mapping[str, slice],
    chunk_rows: int = 500000,
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    """Build original directed duration-by-status tensors with bounded memory."""

    span_hashes, span_stats = _build_span_hashes(trace_dir, cache_dir, chunk_rows)
    trace = np.zeros((
        len(grid), len(GAIA_SERVICES), len(GAIA_SERVICES), len(TRACE_STATUS_CODES)
    ), dtype=np.float32)
    graph = np.zeros((len(GAIA_SERVICES), len(GAIA_SERVICES)), dtype=np.float32)
    status_index = {status: index for index, status in enumerate(TRACE_STATUS_CODES)}
    rows_scanned = invalid_timestamps = invalid_durations = negative_durations = 0
    unmatched_parents = retained_cross_service = unknown_status = 0
    train_stop = slices["train"].stop
    for child_index, child in enumerate(GAIA_SERVICES):
        source = Path(trace_dir) / "trace_table_{}_2021-07.csv".format(child)
        for chunk_index, chunk in enumerate(pd.read_csv(
            source,
            usecols=["start_time", "end_time", "parent_id", "status_code"],
            chunksize=chunk_rows, keep_default_na=False, on_bad_lines="skip",
        ), start=1):
            rows_scanned += len(chunk)
            end_prefix = chunk["end_time"].astype("string").str.replace(".", ",", n=1, regex=False)
            end_ms, valid_time = _local_ms(end_prefix)
            start = pd.to_datetime(chunk["start_time"], errors="coerce")
            end = pd.to_datetime(chunk["end_time"], errors="coerce")
            durations = (end - start).dt.total_seconds().to_numpy(dtype=np.float64)
            finite_duration = np.isfinite(durations)
            invalid_durations += int((~finite_duration).sum())
            negative_durations += int((finite_duration & (durations < 0)).sum())
            invalid_timestamps += int((~valid_time).sum())
            positions, in_grid = _grid_positions(end_ms, grid)
            valid = valid_time & in_grid & finite_duration & (durations >= 0)
            status = chunk["status_code"].astype("string").str.strip().to_numpy(dtype=str)
            known_status = np.isin(status, TRACE_STATUS_CODES)
            unknown_status += int((valid & ~known_status).sum())
            valid &= known_status
            if not np.any(valid):
                continue
            selected_rows = np.flatnonzero(valid)
            parents = _hash_pair(chunk.loc[valid, "parent_id"])
            sources = np.full(len(parents), -1, dtype=np.int16)
            unresolved = np.ones(len(parents), dtype=bool)
            for parent_index, parent_service in enumerate(GAIA_SERVICES):
                found = unresolved & _membership(span_hashes[parent_service], parents)
                sources[found] = parent_index
                unresolved[found] = False
            unmatched_parents += int(unresolved.sum())
            selected_positions = positions[valid]
            selected_durations = durations[valid]
            selected_status = status[valid]
            cross = (sources >= 0) & (sources != child_index)
            retained_cross_service += int(cross.sum())
            if np.any(cross):
                src = sources[cross].astype(np.int64)
                dst = np.full(int(cross.sum()), child_index, dtype=np.int64)
                pos = selected_positions[cross]
                status_ids = np.asarray([status_index[value] for value in selected_status[cross]], dtype=np.int64)
                np.add.at(trace, (pos, src, dst, status_ids), selected_durations[cross].astype(np.float32))
                train_edge = pos < train_stop
                for source_index in np.unique(src[train_edge]):
                    graph[int(source_index), child_index] = 1.0
                    graph[child_index, int(source_index)] = 1.0
            if chunk_index % 10 == 0:
                logging.info("Ada-MGAD traces %s rows %,d", child, rows_scanned)

    train_mean = np.mean(trace[slices["train"]], axis=0)
    trace /= (train_mean[None, :, :, :] * 10.0 + 1e-6)
    if not np.any(graph):
        raise ValueError("Train trace graph has no cross-service edges")
    return trace, graph, {
        "raw_rows_scanned": rows_scanned,
        "invalid_timestamp_rows": invalid_timestamps,
        "invalid_duration_rows": invalid_durations,
        "negative_duration_rows": negative_durations,
        "unknown_status_rows_in_protocol_grid": unknown_status,
        "unmatched_parent_rows_in_protocol_grid": unmatched_parents,
        "retained_cross_service_rows": retained_cross_service,
        "timestamp_alignment": "trace end_time to 30-second detector bin (historical Ada-MGAD convention)",
        "duration_unit": "seconds",
        "status_codes": list(TRACE_STATUS_CODES),
        "normalization_fit_split": "train",
        "graph_fit_split": "train",
        "span_lookup": "two independent stable uint64 hashes; label-free parent span to service mapping",
        "span_indexes": span_stats,
        "graph_edges": int(graph.sum()),
    }


def build_ad_data(
    config: Mapping[str, object],
    project_root: Path,
    data_root: Path,
    artifact_root: Path,
    chunk_rows: int = 500000,
    raw_root_override: Path = None,
) -> Mapping[str, object]:
    blocks = temporal_blocks(config)
    grids = {
        block.name: grid_for_block(block, int(config["ad"]["grid_seconds"]))
        for block in blocks
    }
    offsets = {}
    cursor = 0
    for block in blocks:
        offsets[block.name] = slice(cursor, cursor + len(grids[block.name]))
        cursor += len(grids[block.name])
    grid = np.concatenate([grids[block.name] for block in blocks])
    canonical_raw_root = Path(str(config["gaia_raw_root"]))
    raw_root = Path(raw_root_override) if raw_root_override is not None else canonical_raw_root
    source_timing = json.loads(
        (Path(project_root) / "artifacts/p5/g0r2/raw_telemetry_timing.json").read_text()
    )
    expected_inventory = source_timing["source_binding"]["current_inventory"]
    execution_inventory = {
        "metrics": layout_digest(
            raw_root / "metric/metric_split/metric",
            (raw_root / "metric/metric_split/metric").glob("*.csv"),
        ),
        "logs": layout_digest(
            raw_root / "business/business_split/business",
            (raw_root / "business/business_split/business").glob("*.csv"),
        ),
        "traces": layout_digest(
            raw_root / "trace/trace_split/trace",
            (raw_root / "trace/trace_split/trace").glob("*.csv"),
        ),
    }
    if execution_inventory != expected_inventory:
        raise ValueError("execution GAIA raw root differs from the P5-G0R2 byte-layout binding")
    metric, metric_stats = build_metric_arrays(
        raw_root / "metric/metric_split/metric", grid, offsets,
        Path(data_root) / "metric_cache",
        tuple(config["ad_preprocessing"]["metric_excluded_features"]),
    )
    metric_stats["exclusions"] = list(config["ad_preprocessing"]["metric_exclusions"])
    logs, log_stats = build_log_arrays(
        raw_root / "business/business_split/business", grid, offsets, chunk_rows
    )
    trace, graph, trace_stats = build_trace_arrays(
        raw_root / "trace/trace_split/trace", Path(data_root) / "span_hashes",
        grid, offsets, chunk_rows
    )

    registry = load_registry(config, project_root)
    assigned, raw_purged = assign_event_blocks(registry, blocks)
    split_files = {}
    split_counts = {}
    for block in blocks:
        block_slice = offsets[block.name]
        block_registry = assigned[assigned["split"] == block.name]
        timestamps = grid[block_slice]
        labels = build_registry_node_labels(
            timestamps, block_registry, GAIA_SERVICES,
            int(config["ad"]["grid_seconds"]),
        )
        label_mask = build_semisupervised_mask(
            labels, float(config["ad_model"]["label_percent"]),
            int(config["ad"]["window_bins"]),
        )
        split_files[block.name] = save_split_arrays(data_root, block.name, {
            "timestamps": timestamps,
            "metric": metric[block_slice],
            "log": logs[block_slice],
            "trace": trace[block_slice],
            "labels": labels,
            "label_mask": label_mask,
        })
        split_counts[block.name] = {
            "timestamps": len(timestamps),
            "windows": max(0, len(timestamps) - int(config["ad"]["window_bins"]) + 1),
            "positive_node_bins": int(labels.sum()),
            "event_count": len(block_registry),
            "first_timestamp_ms": int(timestamps[0]),
            "last_timestamp_ms": int(timestamps[-1]),
        }

    graph_path = Path(data_root) / "graph.npy"
    np.save(graph_path, graph, allow_pickle=False)
    manifest = {
        "schema_version": "p5_i1_ad_data_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(project_root), text=True
        ).strip(),
        "random_seed": int(config["random_seed"]),
        "config_sha256": sha256_file(Path(project_root) / "configs/e2e/gaia_p5_v1.yaml"),
        "event_registry_sha256": str(config["event_registry"]["sha256"]),
        "raw_layout_binding": expected_inventory,
        "execution_raw_layout": execution_inventory,
        "canonical_raw_root": str(canonical_raw_root.resolve()),
        "execution_raw_root": str(raw_root.resolve()),
        "raw_injection_boundary_purged": len(raw_purged),
        "services": list(GAIA_SERVICES),
        "grid_seconds": int(config["ad"]["grid_seconds"]),
        "window_bins": int(config["ad"]["window_bins"]),
        "split_counts": split_counts,
        "dimensions": {
            "raw_node": int(metric.shape[-1]),
            "log_len": int(logs.shape[-1]),
            "raw_edge": int(trace.shape[-1]),
        },
        "metric": metric_stats,
        "logs": log_stats,
        "traces": trace_stats,
        "graph": {
            "path": str(graph_path.resolve()),
            "sha256": sha256_file(graph_path),
            "shape": list(graph.shape),
            "edges": int(graph.sum()),
        },
        "split_files": split_files,
        "label_source": "frozen 16,200 supported injection registry only",
        "excluded_label_sources": ["normal", "ERROR", "normal memory freed label", "unsupported or unknown"],
        "normalization_firewall": "all telemetry scaling and quality decisions fit on Train only",
    }
    write_json(Path(artifact_root) / "ad_data_manifest.json", manifest)
    return manifest
