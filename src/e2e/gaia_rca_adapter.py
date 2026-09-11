"""Label-free GAIA raw telemetry index for Ada-RCA-G.

The index is independent of Ada-MGAD tensors.  It preserves exact raw
timestamps so each case is binned relative to its own anchor, then exposes the
four frozen feature channels (metric, log, trace-error, trace-latency).
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from util.GAIA.pre_GAIA import (
    _metric_duplicate_reduce_mode,
    _reduce_duplicate_timestamp_values,
)

from .ad_preprocess import _local_ms, _logical_metric_schema
from .protocol import GAIA_SERVICES, layout_digest, sha256_file, write_json


LOG_LEVELS = ("INFO", "WARNING", "ERROR", "DEBUG", "UNKNOWN")
CHANNELS = ("metric", "log", "trace-error", "trace-latency")


def _safe_key(service: str, indicator: str) -> str:
    digest = hashlib.sha256((service + "\0" + indicator).encode("utf-8")).hexdigest()[:20]
    return "{}-{}".format(service, digest)


def _window_indices(timestamps_ms: np.ndarray, anchor_ms: int, spec) -> Tuple[np.ndarray, np.ndarray]:
    start_ms = int(anchor_ms) - int(round(float(spec.window_seconds) * 1000.0))
    width_ms = int(round(float(spec.bin_seconds) * 1000.0))
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    indices = np.floor_divide(timestamps - start_ms, width_ms)
    valid = (timestamps >= start_ms) & (timestamps < start_ms + int(spec.n_bins) * width_ms)
    return indices.astype(np.int64), valid


def _window_slice(timestamps_ms: np.ndarray, anchor_ms: int, spec) -> slice:
    """Locate the event window in a sorted timestamp array without scanning it."""

    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    start_ms = int(anchor_ms) - int(round(float(spec.window_seconds) * 1000.0))
    end_ms = start_ms + int(spec.n_bins) * int(round(float(spec.bin_seconds) * 1000.0))
    return slice(
        int(np.searchsorted(timestamps, start_ms, side="left")),
        int(np.searchsorted(timestamps, end_ms, side="left")),
    )


def binned_sum_count(
    timestamps_ms: np.ndarray,
    values: np.ndarray,
    anchor_ms: int,
    spec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite value sums/counts after slicing the sorted raw series."""

    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    numeric = np.asarray(values, dtype=np.float64)
    if timestamps.shape != numeric.shape:
        raise ValueError("timestamps and values must have identical shapes")
    window = _window_slice(timestamps, anchor_ms, spec)
    timestamps = timestamps[window]
    numeric = numeric[window]
    indices, valid_time = _window_indices(timestamps, anchor_ms, spec)
    valid = valid_time & np.isfinite(numeric)
    sums = np.bincount(
        indices[valid], weights=numeric[valid], minlength=int(spec.n_bins)
    ).astype(np.float64)
    counts = np.bincount(indices[valid], minlength=int(spec.n_bins)).astype(np.int64)
    return sums, counts


def binned_mean(
    timestamps_ms: np.ndarray,
    values: np.ndarray,
    anchor_ms: int,
    spec,
) -> np.ndarray:
    """Average finite observations into the exact event-relative bins."""

    sums, counts = binned_sum_count(timestamps_ms, values, anchor_ms, spec)
    result = np.full(int(spec.n_bins), np.nan, dtype=np.float64)
    observed = counts > 0
    result[observed] = sums[observed] / counts[observed]
    return result


def binned_count(
    timestamps_ms: np.ndarray,
    selected: np.ndarray,
    anchor_ms: int,
    spec,
) -> np.ndarray:
    """Count selected raw records; absence is an observed numeric zero."""

    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    selected_values = np.asarray(selected, dtype=bool)
    if timestamps.shape != selected_values.shape:
        raise ValueError("timestamps and selection must have identical shapes")
    window = _window_slice(timestamps, anchor_ms, spec)
    timestamps = timestamps[window]
    selected_values = selected_values[window]
    indices, valid_time = _window_indices(timestamps, anchor_ms, spec)
    chosen = valid_time & selected_values
    return np.bincount(indices[chosen], minlength=int(spec.n_bins)).astype(np.float64)


def binned_category_counts(
    timestamps_ms: np.ndarray,
    categories: np.ndarray,
    category_count: int,
    anchor_ms: int,
    spec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Count categorical rows and all rows with one timestamp slice."""

    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    values = np.asarray(categories, dtype=np.int64)
    if timestamps.shape != values.shape:
        raise ValueError("timestamps and categories must have identical shapes")
    window = _window_slice(timestamps, anchor_ms, spec)
    timestamps = timestamps[window]
    values = values[window]
    indices, valid_time = _window_indices(timestamps, anchor_ms, spec)
    valid = valid_time & (values >= 0) & (values < int(category_count))
    output = np.zeros((int(category_count), int(spec.n_bins)), dtype=np.float64)
    if np.any(valid):
        np.add.at(output, (values[valid], indices[valid]), 1.0)
    total = np.bincount(indices[valid_time], minlength=int(spec.n_bins)).astype(np.float64)
    return output, total


def parse_trace_chunk(frame: pd.DataFrame) -> Tuple[Mapping[str, np.ndarray], Mapping[str, int]]:
    """Validate GAIA trace time/status/latency and return deterministic arrays."""

    required = {"start_time", "end_time", "status_code", "service_name"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("trace chunk is missing columns: {}".format(missing))
    end_prefix = frame["end_time"].astype("string").str.replace(".", ",", n=1, regex=False)
    end_ms, valid_time = _local_ms(end_prefix)
    starts = pd.to_datetime(frame["start_time"], errors="coerce")
    ends = pd.to_datetime(frame["end_time"], errors="coerce")
    latency = (ends - starts).dt.total_seconds().to_numpy(dtype=np.float64)
    finite = np.isfinite(latency)
    nonnegative = finite & (latency >= 0.0)
    status = pd.to_numeric(frame["status_code"], errors="coerce").to_numpy(dtype=np.float64)
    valid_status = np.isfinite(status)
    services = frame["service_name"].astype("string").fillna("").to_numpy(dtype=str)
    known_service = np.isin(services, GAIA_SERVICES)
    valid = valid_time & nonnegative & valid_status & known_service
    arrays = {
        "timestamp_ms": end_ms[valid].astype(np.int64),
        "service": services[valid],
        "trace_error": (status[valid] != 200.0),
        "latency_seconds": latency[valid].astype(np.float64),
        "status_code": status[valid].astype(np.int64),
    }
    stats = {
        "rows": int(len(frame)),
        "invalid_timestamp_rows": int((~valid_time).sum()),
        "invalid_latency_rows": int((~finite).sum()),
        "negative_latency_rows": int((finite & (latency < 0.0)).sum()),
        "invalid_status_rows": int((~valid_status).sum()),
        "unknown_service_rows": int((~known_service).sum()),
        "retained_rows": int(valid.sum()),
        "trace_error_rows": int((status[valid] != 200.0).sum()),
    }
    return arrays, stats


@dataclass(frozen=True)
class IndexedSeries:
    service: str
    indicator: str
    timestamps_ms: np.ndarray
    values: np.ndarray


class GaiaRcaRawIndex:
    """Memory-mapped exact-timestamp telemetry used to build per-case inputs."""

    def __init__(
        self,
        metric_series: Sequence[IndexedSeries],
        logs: Mapping[str, Tuple[np.ndarray, np.ndarray]],
        traces: Mapping[str, Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    ):
        self.metric_series = tuple(metric_series)
        self.logs = dict(logs)
        self.traces = {service: tuple(parts) for service, parts in traces.items()}

    @classmethod
    def from_manifest(cls, path: Path) -> "GaiaRcaRawIndex":
        manifest_path = Path(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
        metric = []
        for record in manifest["metric_series"]:
            metric.append(IndexedSeries(
                service=str(record["service"]),
                indicator=str(record["indicator"]),
                timestamps_ms=np.load(root / record["timestamps"], mmap_mode="r"),
                values=np.load(root / record["values"], mmap_mode="r"),
            ))
        logs = {}
        for service, record in manifest["logs"].items():
            logs[service] = (
                np.load(root / record["timestamps"], mmap_mode="r"),
                np.load(root / record["levels"], mmap_mode="r"),
            )
        traces: MutableMapping[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
        for record in manifest["traces"]["parts"]:
            traces[str(record["service"])].append((
                np.load(root / record["timestamps"], mmap_mode="r"),
                np.load(root / record["errors"], mmap_mode="r"),
                np.load(root / record["latencies"], mmap_mode="r"),
            ))
        return cls(metric, logs, traces)

    def case_indicators(self, anchor_ms: int, spec) -> Mapping[str, Mapping[str, np.ndarray]]:
        """Return label-free exact-anchor indicator series for four channels."""

        result: Dict[str, Dict[str, np.ndarray]] = {channel: {} for channel in CHANNELS}
        for series in self.metric_series:
            key = "{}::{}".format(series.service, series.indicator)
            result["metric"][key] = binned_mean(
                series.timestamps_ms, series.values, anchor_ms, spec
            )
        for service in GAIA_SERVICES:
            timestamps, levels = self.logs.get(
                service,
                (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)),
            )
            level_counts, total_counts = binned_category_counts(
                timestamps, levels, len(LOG_LEVELS), anchor_ms, spec
            )
            for level_index, level in enumerate(LOG_LEVELS):
                result["log"]["{}::level_{}".format(service, level)] = level_counts[level_index]
            result["log"]["{}::log_total".format(service)] = total_counts

            error_counts = np.zeros(int(spec.n_bins), dtype=np.float64)
            latency_sums = np.zeros(int(spec.n_bins), dtype=np.float64)
            latency_counts = np.zeros(int(spec.n_bins), dtype=np.int64)
            for timestamps_part, errors_part, latencies_part in self.traces.get(service, ()):
                timestamps_part = np.asarray(timestamps_part, dtype=np.int64)
                error_counts += binned_count(
                    timestamps_part, np.asarray(errors_part, dtype=bool), anchor_ms, spec
                )
                sums, counts = binned_sum_count(
                    timestamps_part, np.asarray(latencies_part, dtype=np.float64),
                    anchor_ms, spec,
                )
                latency_sums += sums
                latency_counts += counts
            latency_mean = np.full(int(spec.n_bins), np.nan, dtype=np.float64)
            observed_latency = latency_counts > 0
            latency_mean[observed_latency] = (
                latency_sums[observed_latency] / latency_counts[observed_latency]
            )
            result["trace-error"]["{}::status_not_200_count".format(service)] = error_counts
            result["trace-latency"]["{}::latency_seconds_mean".format(service)] = latency_mean
        return result


def _save_array(path: Path, values: np.ndarray) -> Mapping[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values), allow_pickle=False)
    return {
        "path": str(path.name),
        "shape": list(np.asarray(values).shape),
        "dtype": str(np.asarray(values).dtype),
        "sha256": sha256_file(path),
    }


def _read_metric_group_strict(
    full_name: str, paths: Sequence[Path], indicator: str
) -> pd.DataFrame:
    """Read every shard in a logical metric group without swallowing I/O errors.

    The historical Ada-MGAD helper deliberately ignored unreadable shards.  A
    formal RCA raw index must instead fail closed: accepting a partially read
    group would make missing telemetry indistinguishable from real sparsity.
    Duplicate reduction is kept identical to the historical helper.
    """

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = {"timestamp", "value"} - set(frame.columns)
        if missing:
            raise ValueError(
                "metric shard {} for {} lacks columns {}".format(
                    path, full_name, sorted(missing)
                )
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
        reduce_mode = _metric_duplicate_reduce_mode(indicator)
        merged = merged.groupby("timestamp", as_index=False)["value"].agg(
            lambda values: _reduce_duplicate_timestamp_values(values, reduce_mode)
        )
    return merged.sort_values("timestamp").reset_index(drop=True)


def _verify_raw_layout(raw_root: Path, expected_inventory: Mapping[str, object]) -> Mapping[str, object]:
    roots = {
        "metrics": raw_root / "metric/metric_split/metric",
        "logs": raw_root / "business/business_split/business",
        "traces": raw_root / "trace/trace_split/trace",
    }
    actual = {
        modality: layout_digest(root, root.glob("*.csv"))
        for modality, root in roots.items()
    }
    if actual != expected_inventory:
        raise ValueError("GAIA raw root differs from the P5-G0R2 byte-layout binding")
    return actual


def build_raw_index(
    raw_root: Path,
    output_root: Path,
    expected_inventory: Mapping[str, object],
    *,
    chunk_rows: int = 500000,
    provenance: Optional[Mapping[str, object]] = None,
) -> Mapping[str, object]:
    """Build a reusable raw index; this is the intentionally long full-data step."""

    raw_root = Path(raw_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    actual_inventory = _verify_raw_layout(raw_root, expected_inventory)
    metric_dir = raw_root / "metric/metric_split/metric"
    groups, common = _logical_metric_schema(metric_dir)
    metric_records = []
    for service in GAIA_SERVICES:
        for indicator in common:
            core_series = []
            for full_name, paths in sorted(groups[service][indicator].items()):
                frame = _read_metric_group_strict(full_name, paths, indicator)
                if len(frame):
                    core_series.append(pd.Series(
                        pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float),
                        index=pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=float),
                    ))
            if not core_series:
                raise ValueError(
                    "metric series {}::{} has no finite observations".format(
                        service, indicator
                    )
                )
            combined = pd.concat(core_series, axis=1).mean(axis=1, skipna=True).sort_index()
            valid = np.isfinite(combined.index.to_numpy(dtype=float)) & np.isfinite(combined.to_numpy(dtype=float))
            timestamps = combined.index.to_numpy(dtype=float)[valid].astype(np.int64)
            values = combined.to_numpy(dtype=float)[valid].astype(np.float32)
            order = np.argsort(timestamps, kind="stable")
            key = _safe_key(service, indicator)
            ts_path = output_root / (key + ".metric.timestamps.npy")
            value_path = output_root / (key + ".metric.values.npy")
            ts_meta = _save_array(ts_path, timestamps[order])
            value_meta = _save_array(value_path, values[order])
            metric_records.append({
                "service": service,
                "indicator": indicator,
                "timestamps": ts_path.name,
                "values": value_path.name,
                "rows": int(len(order)),
                "timestamps_sha256": ts_meta["sha256"],
                "values_sha256": value_meta["sha256"],
            })

    log_records = {}
    invalid_log_timestamps = 0
    for service in GAIA_SERVICES:
        source = raw_root / "business/business_split/business" / (
            "business_table_{}_2021-07.csv".format(service)
        )
        timestamp_parts = []
        level_parts = []
        for chunk in pd.read_csv(
            source, usecols=["message"], chunksize=chunk_rows,
            keep_default_na=False, on_bad_lines="skip",
        ):
            messages = chunk["message"].astype("string")
            timestamps, valid = _local_ms(messages.str.slice(0, 23))
            invalid_log_timestamps += int((~valid).sum())
            selected = messages[valid]
            levels = np.full(len(selected), len(LOG_LEVELS) - 1, dtype=np.uint8)
            for level_index, level in enumerate(LOG_LEVELS[:-1]):
                found = selected.str.contains(
                    "| {} |".format(level), regex=False, na=False
                ).to_numpy(dtype=bool, na_value=False)
                levels[found] = level_index
            timestamp_parts.append(timestamps[valid].astype(np.int64))
            level_parts.append(levels)
        timestamps = np.concatenate(timestamp_parts) if timestamp_parts else np.empty(0, dtype=np.int64)
        levels = np.concatenate(level_parts) if level_parts else np.empty(0, dtype=np.uint8)
        order = np.argsort(timestamps, kind="stable")
        ts_path = output_root / (service + ".log.timestamps.npy")
        level_path = output_root / (service + ".log.levels.npy")
        ts_meta = _save_array(ts_path, timestamps[order])
        level_meta = _save_array(level_path, levels[order])
        log_records[service] = {
            "timestamps": ts_path.name,
            "levels": level_path.name,
            "rows": int(len(order)),
            "timestamps_sha256": ts_meta["sha256"],
            "levels_sha256": level_meta["sha256"],
        }

    trace_parts = []
    trace_stats = defaultdict(int)
    trace_status_counts = defaultdict(int)
    trace_dir = raw_root / "trace/trace_split/trace"
    for source in sorted(trace_dir.glob("trace_table_*_2021-07.csv")):
        part_number = 0
        for chunk in pd.read_csv(
            source,
            usecols=["start_time", "end_time", "status_code", "service_name"],
            chunksize=chunk_rows, keep_default_na=False, on_bad_lines="skip",
        ):
            arrays, stats = parse_trace_chunk(chunk)
            for key, value in stats.items():
                trace_stats[key] += int(value)
            for status in arrays["status_code"]:
                trace_status_counts[str(int(status))] += 1
            for service in GAIA_SERVICES:
                selected = arrays["service"] == service
                if not np.any(selected):
                    continue
                prefix = "{}.{}.trace".format(source.stem, part_number)
                ts_path = output_root / (prefix + ".{}.timestamps.npy".format(service))
                error_path = output_root / (prefix + ".{}.errors.npy".format(service))
                latency_path = output_root / (prefix + ".{}.latencies.npy".format(service))
                timestamps = arrays["timestamp_ms"][selected]
                order = np.argsort(timestamps, kind="stable")
                ts_meta = _save_array(ts_path, timestamps[order])
                error_meta = _save_array(
                    error_path, arrays["trace_error"][selected][order].astype(np.uint8)
                )
                latency_meta = _save_array(
                    latency_path, arrays["latency_seconds"][selected][order].astype(np.float32)
                )
                trace_parts.append({
                    "service": service,
                    "timestamps": ts_path.name,
                    "errors": error_path.name,
                    "latencies": latency_path.name,
                    "rows": int(selected.sum()),
                    "timestamps_sha256": ts_meta["sha256"],
                    "errors_sha256": error_meta["sha256"],
                    "latencies_sha256": latency_meta["sha256"],
                })
            part_number += 1

    manifest = {
        "schema_version": "p5_i1_gaia_rca_raw_index_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root.resolve()),
        "raw_layout": actual_inventory,
        "services": list(GAIA_SERVICES),
        "label_firewall": "no root service, fault type, event label, or Ada-MGAD tensor is accepted",
        "provenance": dict(provenance or {}),
        "metric_mapping": "filename telemetry schema plus canonical service registry only",
        "metric_series": metric_records,
        "logs": log_records,
        "log_timestamp_source": "message prefix YYYY-MM-DD HH:MM:SS,mmm",
        "invalid_log_timestamp_rows": int(invalid_log_timestamps),
        "traces": {
            "parts": trace_parts,
            "stats": dict(trace_stats),
            "status_counts": dict(sorted(trace_status_counts.items())),
            "error_rule": "status_code != 200",
            "latency_rule": "(end_time - start_time).total_seconds()",
            "timestamp_source": "end_time; parsed to epoch milliseconds in Asia/Shanghai",
            "latency_unit": "seconds",
        },
    }
    manifest_path = output_root / "index_manifest.json"
    write_json(manifest_path, manifest)
    return manifest
