"""Streaming telemetry diagnostics for the P1 standalone RCA benchmark.

The scanner deliberately separates three concepts:

* scheduled metric missingness (empty value cells or absent timestamps);
* required-field/parse failures for event streams;
* activity coverage of logs/traces in candidate windows.

An inactive event stream is not silently classified as missing telemetry.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.data.gaia import GAIAAdapterResult
from src.data.rcaeval import RCAEvalAdapterResult
from src.data.schema import RCACaseInput
from util.GAIA.constant import GAIA_SERVICES
from util.GAIA.pre_GAIA import (
    GAIA_TZ,
    _parse_metric_filename,
    _target_services_for_metric,
)


DIAGNOSTIC_SCHEMA_VERSION = "p1_telemetry_diagnostics_v1"
DEFAULT_WINDOWS_SECONDS = (30, 60, 300, 600, 1800)
_BIN_SIZE_MS = 30_000
_MAX_DELTA_SAMPLE = 100_000
_DELTA_SAMPLE_PER_UPDATE = 256


def _finite_summary(values: Iterable[float]) -> Mapping[str, Optional[float]]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


@dataclass
class TimestampAccumulator:
    """Constant-memory timestamp quality summary plus 30-second activity bins."""

    valid_rows: int = 0
    invalid_rows: int = 0
    minimum_ms: Optional[int] = None
    maximum_ms: Optional[int] = None
    positive_deltas: int = 0
    zero_deltas: int = 0
    negative_deltas: int = 0
    bins: Set[int] = field(default_factory=set)
    _previous_ms: Optional[int] = None
    _delta_sample: List[int] = field(default_factory=list)

    def begin_file(self) -> None:
        self._previous_ms = None

    def update(self, values_ms: np.ndarray, invalid_rows: int = 0) -> None:
        values = np.asarray(values_ms, dtype=np.int64)
        self.invalid_rows += int(invalid_rows)
        if values.size == 0:
            return

        self.valid_rows += int(values.size)
        current_min = int(values.min())
        current_max = int(values.max())
        self.minimum_ms = (
            current_min if self.minimum_ms is None else min(self.minimum_ms, current_min)
        )
        self.maximum_ms = (
            current_max if self.maximum_ms is None else max(self.maximum_ms, current_max)
        )
        self.bins.update(int(value) for value in np.unique(values // _BIN_SIZE_MS))

        if self._previous_ms is None:
            deltas = np.diff(values)
        else:
            deltas = np.diff(np.concatenate(([self._previous_ms], values)))
        self._previous_ms = int(values[-1])

        if deltas.size == 0:
            return
        self.zero_deltas += int((deltas == 0).sum())
        self.negative_deltas += int((deltas < 0).sum())
        positive = deltas[deltas > 0]
        self.positive_deltas += int(positive.size)
        if positive.size:
            if positive.size > _DELTA_SAMPLE_PER_UPDATE:
                indices = np.linspace(
                    0, positive.size - 1, _DELTA_SAMPLE_PER_UPDATE, dtype=np.int64
                )
                positive = positive[indices]
            self._delta_sample.extend(int(value) for value in positive)
            if len(self._delta_sample) > _MAX_DELTA_SAMPLE:
                indices = np.linspace(
                    0,
                    len(self._delta_sample) - 1,
                    _MAX_DELTA_SAMPLE,
                    dtype=np.int64,
                )
                self._delta_sample = [self._delta_sample[index] for index in indices]

    def to_record(self) -> Mapping[str, Any]:
        total_transitions = (
            self.positive_deltas + self.zero_deltas + self.negative_deltas
        )
        return {
            "valid_timestamp_rows": self.valid_rows,
            "invalid_timestamp_rows": self.invalid_rows,
            "timestamp_min_ms": self.minimum_ms,
            "timestamp_max_ms": self.maximum_ms,
            "occupied_30s_bins": len(self.bins),
            "sequential_delta_semantics": (
                "source-order transitions; reset at every source file"
            ),
            "positive_delta_rows": self.positive_deltas,
            "zero_delta_rows": self.zero_deltas,
            "negative_delta_rows": self.negative_deltas,
            "zero_delta_ratio": (
                self.zero_deltas / total_transitions if total_transitions else None
            ),
            "negative_delta_ratio": (
                self.negative_deltas / total_transitions if total_transitions else None
            ),
            "positive_delta_ms_approx": _finite_summary(self._delta_sample),
            "delta_sample_method": (
                "deterministic evenly spaced sample, capped at {} transitions".format(
                    _MAX_DELTA_SAMPLE
                )
            ),
        }


@dataclass
class ModalityAccumulator:
    """Aggregate scalar quality counts and timestamp activity by service."""

    name: str
    timestamps: TimestampAccumulator = field(default_factory=TimestampAccumulator)
    per_service: Dict[str, TimestampAccumulator] = field(default_factory=dict)
    service_features: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    files_scanned: int = 0
    source_bytes: int = 0
    rows: int = 0
    value_cells: int = 0
    missing_value_cells: int = 0
    nonnumeric_value_cells: int = 0
    scheduled_files: int = 0
    scheduled_expected_timestamps: int = 0
    scheduled_observed_timestamps: int = 0
    scheduled_missing_timestamps: int = 0
    scheduled_duplicate_timestamp_rows: int = 0
    scheduled_nominal_interval_ms: List[int] = field(default_factory=list)
    required_missing: Counter = field(default_factory=Counter)
    incomplete_files: List[Mapping[str, str]] = field(default_factory=list)
    headers: Set[Tuple[str, ...]] = field(default_factory=set)

    def service_stats(self, service: str) -> TimestampAccumulator:
        if service not in self.per_service:
            self.per_service[service] = TimestampAccumulator()
        return self.per_service[service]

    def register_file(
        self,
        path: Path,
        services: Sequence[str],
        feature: Optional[str] = None,
        header: Optional[Sequence[str]] = None,
    ) -> None:
        self.files_scanned += 1
        self.source_bytes += path.stat().st_size
        self.timestamps.begin_file()
        for service in services:
            self.service_stats(service).begin_file()
            if feature is not None:
                self.service_features[service].add(feature)
        if header is not None:
            self.headers.add(tuple(header))

    def update_timestamps(
        self,
        values_ms: np.ndarray,
        invalid_rows: int,
        services: Sequence[str],
    ) -> None:
        self.timestamps.update(values_ms, invalid_rows=invalid_rows)
        for service in services:
            self.service_stats(service).update(values_ms, invalid_rows=invalid_rows)

    def record_scheduled_file(
        self,
        unique_timestamps_ms: Set[int],
        valid_timestamp_rows: int,
    ) -> None:
        """Record expected-vs-observed samples for one scheduled metric file."""

        if not unique_timestamps_ms:
            return
        ordered = np.asarray(sorted(unique_timestamps_ms), dtype=np.int64)
        self.scheduled_files += 1
        observed = int(ordered.size)
        self.scheduled_observed_timestamps += observed
        self.scheduled_duplicate_timestamp_rows += max(
            0, int(valid_timestamp_rows) - observed
        )

        if observed < 2:
            expected = observed
        else:
            positive = np.diff(ordered)
            positive = positive[positive > 0]
            if positive.size == 0:
                expected = observed
            else:
                positive.sort()
                nominal = max(1, int(positive[(positive.size - 1) // 2]))
                self.scheduled_nominal_interval_ms.append(nominal)
                expected = int((ordered[-1] - ordered[0]) // nominal) + 1
                expected = max(expected, observed)
        self.scheduled_expected_timestamps += expected
        self.scheduled_missing_timestamps += max(0, expected - observed)

    def to_record(self) -> Mapping[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "source_bytes": self.source_bytes,
            "rows": self.rows,
            "value_cells": self.value_cells,
            "missing_value_cells": self.missing_value_cells,
            "missing_value_ratio": (
                self.missing_value_cells / self.value_cells
                if self.value_cells
                else None
            ),
            "nonnumeric_value_cells": self.nonnumeric_value_cells,
            "scheduled_timestamp_quality": {
                "files": self.scheduled_files,
                "expected_timestamps": self.scheduled_expected_timestamps,
                "observed_unique_timestamps": self.scheduled_observed_timestamps,
                "missing_timestamps": self.scheduled_missing_timestamps,
                "missing_timestamp_ratio": (
                    self.scheduled_missing_timestamps
                    / self.scheduled_expected_timestamps
                    if self.scheduled_expected_timestamps
                    else None
                ),
                "duplicate_timestamp_rows": self.scheduled_duplicate_timestamp_rows,
                "nominal_interval_ms": _finite_summary(
                    self.scheduled_nominal_interval_ms
                ),
                "semantics": (
                    "per-file grid inferred from lower-median positive timestamp delta"
                ),
            },
            "required_field_missing": dict(sorted(self.required_missing.items())),
            "incomplete_files": self.incomplete_files,
            "distinct_headers": len(self.headers),
            "timestamp": self.timestamps.to_record(),
            "per_service": {
                service: {
                    **stats.to_record(),
                    "source_features": len(self.service_features.get(service, set())),
                }
                for service, stats in sorted(self.per_service.items())
            },
        }


def _header(path: Path) -> List[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _numeric_timestamp_ms(
    series: pd.Series,
    source_unit: str,
) -> Tuple[np.ndarray, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric.astype(float))
    values = numeric[valid].to_numpy(dtype=float)
    factors = {"s": 1000.0, "ms": 1.0, "us": 0.001, "ns": 0.000001}
    if source_unit not in factors:
        raise ValueError("unsupported timestamp unit {!r}".format(source_unit))
    converted = np.floor(values * factors[source_unit]).astype(np.int64)
    return converted, int((~valid).sum())


def _local_datetime_ms(
    series: pd.Series,
    format_string: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    parsed = pd.to_datetime(series, format=format_string, errors="coerce")
    valid = parsed.notna()
    localized = parsed[valid].dt.tz_localize(
        GAIA_TZ, ambiguous="NaT", nonexistent="NaT"
    )
    localized_valid = localized.notna()
    values = (
        localized[localized_valid].astype("int64").to_numpy(dtype=np.int64)
        // 1_000_000
    )
    invalid = int((~valid).sum() + (~localized_valid).sum())
    return values, invalid


def _gaia_log_timestamp_ms(messages: pd.Series) -> Tuple[np.ndarray, int]:
    extracted = messages.astype("string").str.extract(
        r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})",
        expand=False,
    )
    return _local_datetime_ms(extracted, "%Y-%m-%d %H:%M:%S,%f")


def _layout_inventory(paths: Sequence[Path], root: Path) -> Mapping[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(paths):
        relative = str(path.resolve().relative_to(root.resolve()))
        size = path.stat().st_size
        total_bytes += size
        digest.update("{}\0{}\n".format(relative, size).encode("utf-8"))
    return {
        "files": len(paths),
        "bytes": total_bytes,
        "digest_scope": "sha256(sorted relative_path + NUL + byte_size)",
        "layout_sha256": digest.hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _window_bins(anchor_ms: int, window_seconds: int) -> range:
    first = (int(anchor_ms) - window_seconds * 1000) // _BIN_SIZE_MS
    last = (int(anchor_ms) + window_seconds * 1000) // _BIN_SIZE_MS
    return range(first, last + 1)


def _case_window_activity(
    cases: Sequence[RCACaseInput],
    modality: ModalityAccumulator,
    windows_seconds: Sequence[int],
) -> Mapping[str, Any]:
    by_window = {}
    for window_seconds in windows_seconds:
        occupied_ratios = []
        service_presence_ratios = []
        any_activity = 0
        for case in cases:
            bins = tuple(_window_bins(int(case.anchor_time), window_seconds))
            occupied = sum(value in modality.timestamps.bins for value in bins)
            ratio = occupied / len(bins)
            occupied_ratios.append(ratio)
            if occupied:
                any_activity += 1

            present_services = 0
            for service in case.services:
                stats = modality.per_service.get(service)
                if stats is not None and any(value in stats.bins for value in bins):
                    present_services += 1
            service_presence_ratios.append(present_services / len(case.services))

        by_window[str(window_seconds)] = {
            "window_semantics": "inclusive [t0-window, t0+window], 30s occupancy bins",
            "cases_with_any_activity": any_activity,
            "case_any_activity_ratio": (
                any_activity / len(cases) if cases else None
            ),
            "occupied_bin_ratio": _finite_summary(occupied_ratios),
            "candidate_service_presence_ratio": _finite_summary(
                service_presence_ratios
            ),
        }
    return by_window


def _progress(label: str, current: int, total: int, every: int) -> None:
    if every > 0 and (current == total or current % every == 0):
        print(
            "[telemetry-diagnostics] {}: {}/{}".format(label, current, total),
            file=sys.stderr,
            flush=True,
        )


def _scan_gaia_metrics(
    directory: Path,
    chunk_rows: int,
    max_files: Optional[int],
    progress_every: int,
) -> Tuple[ModalityAccumulator, Mapping[str, Any]]:
    accumulator = ModalityAccumulator("metrics")
    all_paths = sorted(directory.glob("*.csv"))
    paths = all_paths[:max_files] if max_files is not None else all_paths
    unrecognized = 0

    for index, path in enumerate(paths, 1):
        info = _parse_metric_filename(path.name)
        targets = _target_services_for_metric(info) if info is not None else []
        if not targets:
            unrecognized += 1
            _progress("GAIA metrics", index, len(paths), progress_every)
            continue
        services = tuple(service for service, _ in targets)
        feature = str(info["feature"])
        file_timestamps: Set[int] = set()
        file_valid_rows = 0
        completed = False
        try:
            header = _header(path)
            accumulator.register_file(path, services, feature=feature, header=header)
            for chunk in pd.read_csv(
                path,
                usecols=["timestamp", "value"],
                chunksize=chunk_rows,
            ):
                accumulator.rows += len(chunk)
                timestamps, invalid = _numeric_timestamp_ms(
                    chunk["timestamp"], "ms"
                )
                accumulator.update_timestamps(timestamps, invalid, services)
                file_timestamps.update(np.unique(timestamps).tolist())
                file_valid_rows += len(timestamps)

                accumulator.value_cells += len(chunk)
                raw_missing = chunk["value"].isna()
                numeric = pd.to_numeric(chunk["value"], errors="coerce")
                nonnumeric = numeric.isna() & ~raw_missing
                accumulator.missing_value_cells += int(raw_missing.sum())
                accumulator.nonnumeric_value_cells += int(nonnumeric.sum())
            completed = True
        except Exception as exc:
            accumulator.incomplete_files.append(
                {"path": str(path), "error": type(exc).__name__}
            )
        if completed:
            accumulator.record_scheduled_file(
                file_timestamps, valid_timestamp_rows=file_valid_rows
            )
        _progress("GAIA metrics", index, len(paths), progress_every)

    inventory = _layout_inventory(paths, directory)
    return accumulator, {
        "inventory": inventory,
        "available_files": len(all_paths),
        "scan_limited": max_files is not None,
        "unrecognized_or_noncandidate_files": unrecognized,
    }


def _scan_gaia_logs(
    directory: Path,
    chunk_rows: int,
    max_files: Optional[int],
    progress_every: int,
) -> Tuple[ModalityAccumulator, Mapping[str, Any]]:
    accumulator = ModalityAccumulator("logs")
    all_paths = [
        directory / "business_table_{}_2021-07.csv".format(service)
        for service in GAIA_SERVICES
    ]
    existing = [path for path in all_paths if path.is_file()]
    paths = existing[:max_files] if max_files is not None else existing
    declared_datetime_has_clock = 0
    source_service_mismatches = 0

    for index, path in enumerate(paths, 1):
        source_service = path.name[len("business_table_") : -len("_2021-07.csv")]
        try:
            header = _header(path)
            accumulator.register_file(path, (source_service,), header=header)
            for chunk in pd.read_csv(
                path,
                usecols=["datetime", "service", "message"],
                chunksize=chunk_rows,
                keep_default_na=False,
            ):
                accumulator.rows += len(chunk)
                missing_message = chunk["message"].astype(str).str.len().eq(0)
                missing_service = chunk["service"].astype(str).str.len().eq(0)
                accumulator.required_missing["message"] += int(missing_message.sum())
                accumulator.required_missing["service"] += int(missing_service.sum())
                accumulator.required_missing["declared_datetime"] += int(
                    chunk["datetime"].astype(str).str.len().eq(0).sum()
                )
                declared_datetime_has_clock += int(
                    chunk["datetime"]
                    .astype(str)
                    .str.contains(r"\d{2}:\d{2}:\d{2}", regex=True)
                    .sum()
                )
                source_service_mismatches += int(
                    (
                        chunk["service"].astype(str).str.strip()
                        != source_service
                    ).sum()
                )
                timestamps, invalid = _gaia_log_timestamp_ms(chunk["message"])
                accumulator.update_timestamps(
                    timestamps, invalid, (source_service,)
                )
        except Exception as exc:
            accumulator.incomplete_files.append(
                {"path": str(path), "error": type(exc).__name__}
            )
        _progress("GAIA logs", index, len(paths), progress_every)

    return accumulator, {
        "inventory": _layout_inventory(paths, directory),
        "expected_service_files": len(all_paths),
        "available_service_files": len(existing),
        "scan_limited": max_files is not None,
        "declared_datetime_rows_with_clock": declared_datetime_has_clock,
        "source_filename_service_mismatches": source_service_mismatches,
        "timestamp_source": "millisecond prefix parsed from message",
    }


def _scan_gaia_traces(
    directory: Path,
    chunk_rows: int,
    max_files: Optional[int],
    progress_every: int,
) -> Tuple[ModalityAccumulator, Mapping[str, Any]]:
    accumulator = ModalityAccumulator("traces")
    all_paths = [
        directory / "trace_table_{}_2021-07.csv".format(service)
        for service in GAIA_SERVICES
    ]
    existing = [path for path in all_paths if path.is_file()]
    paths = existing[:max_files] if max_files is not None else existing
    negative_duration_rows = 0
    parent_id_missing = 0
    topology_fields_complete_files = 0

    columns = [
        "service_name",
        "trace_id",
        "span_id",
        "parent_id",
        "start_time",
        "end_time",
        "status_code",
    ]
    for index, path in enumerate(paths, 1):
        source_service = path.name[len("trace_table_") : -len("_2021-07.csv")]
        try:
            header = _header(path)
            if {"trace_id", "span_id", "parent_id", "service_name"}.issubset(header):
                topology_fields_complete_files += 1
            accumulator.register_file(path, (source_service,), header=header)
            for chunk in pd.read_csv(
                path,
                usecols=columns,
                chunksize=chunk_rows,
                keep_default_na=False,
            ):
                accumulator.rows += len(chunk)
                for field_name in (
                    "service_name",
                    "trace_id",
                    "span_id",
                    "start_time",
                    "end_time",
                ):
                    accumulator.required_missing[field_name] += int(
                        chunk[field_name].astype(str).str.len().eq(0).sum()
                    )
                parent_id_missing += int(
                    chunk["parent_id"].astype(str).str.len().eq(0).sum()
                )
                start_ms, invalid = _local_datetime_ms(chunk["start_time"])
                accumulator.update_timestamps(
                    start_ms, invalid, (source_service,)
                )

                start = pd.to_datetime(chunk["start_time"], errors="coerce")
                end = pd.to_datetime(chunk["end_time"], errors="coerce")
                valid_duration = start.notna() & end.notna()
                negative_duration_rows += int(
                    (end[valid_duration] < start[valid_duration]).sum()
                )
        except Exception as exc:
            accumulator.incomplete_files.append(
                {"path": str(path), "error": type(exc).__name__}
            )
        _progress("GAIA traces", index, len(paths), progress_every)

    return accumulator, {
        "inventory": _layout_inventory(paths, directory),
        "expected_service_files": len(all_paths),
        "available_service_files": len(existing),
        "scan_limited": max_files is not None,
        "negative_duration_rows": negative_duration_rows,
        "parent_id_missing_rows": parent_id_missing,
        "topology_fields_complete_files": topology_fields_complete_files,
    }


def diagnose_gaia(
    raw_path: str,
    adapter: GAIAAdapterResult,
    chunk_rows: int = 100_000,
    windows_seconds: Sequence[int] = DEFAULT_WINDOWS_SECONDS,
    max_files_per_modality: Optional[int] = None,
    progress_every: int = 100,
) -> Mapping[str, Any]:
    root = Path(raw_path).resolve()
    metric_directory = Path(adapter.source_layout.metrics_directory)
    log_directory = Path(adapter.source_layout.logs_directory)
    trace_directory = Path(adapter.source_layout.traces_directory)

    metrics, metric_extra = _scan_gaia_metrics(
        metric_directory,
        chunk_rows,
        max_files_per_modality,
        progress_every,
    )
    logs, log_extra = _scan_gaia_logs(
        log_directory,
        chunk_rows,
        max_files_per_modality,
        progress_every,
    )
    traces, trace_extra = _scan_gaia_traces(
        trace_directory,
        chunk_rows,
        max_files_per_modality,
        progress_every,
    )

    modalities = {"metrics": metrics, "logs": logs, "traces": traces}
    extras = {
        "metrics": metric_extra,
        "logs": log_extra,
        "traces": trace_extra,
    }
    run_table = Path(adapter.source_layout.run_table)
    return {
        "dataset": "GAIA-MicroSS-2021-07",
        "scan_scope": (
            "limited-smoke" if max_files_per_modality is not None else "full"
        ),
        "cases": len(adapter.inputs),
        "source_root": str(root),
        "critical_annotation_sha256": {
            "run_table": _sha256_file(run_table),
        },
        "modalities": {
            name: {
                **accumulator.to_record(),
                **extras[name],
                "case_window_activity": _case_window_activity(
                    adapter.inputs, accumulator, windows_seconds
                ),
            }
            for name, accumulator in modalities.items()
        },
        "topology": {
            "explicit_static_service_graph": False,
            "adapter_topology_ref_populated": any(
                case.topology is not None for case in adapter.inputs
            ),
            "trace_derived_dynamic_graph_possible": (
                trace_extra["topology_fields_complete_files"]
                == trace_extra["available_service_files"]
                and trace_extra["available_service_files"] > 0
            ),
            "source": (
                "trace_id/span_id/parent_id/service_name fields; graph is not "
                "materialized by this diagnostic"
            ),
        },
        "diagnostic_cautions": [
            "GAIA log timestamps come from the message prefix, not the coarse datetime column",
            "log/trace empty activity bins indicate no observed event, not scheduled-sample missingness",
            "layout digests bind relative paths and sizes, not full telemetry bytes",
        ],
    }


def _service_from_metric_column(column: str, candidates: Sequence[str]) -> Optional[str]:
    for service in candidates:
        if column.startswith(service + "_"):
            return service
    return None


def _update_service_timestamps_from_rows(
    accumulator: ModalityAccumulator,
    chunk: pd.DataFrame,
    timestamp_ms_by_index: pd.Series,
    service_column: str,
) -> None:
    valid = timestamp_ms_by_index.notna()
    if not valid.any():
        return
    values = chunk.loc[valid, service_column].astype(str).str.strip()
    times = timestamp_ms_by_index[valid]
    for service, indices in values.groupby(values).groups.items():
        if not service:
            continue
        stats = accumulator.service_stats(service)
        selected = times.loc[indices].to_numpy(dtype=np.int64)
        stats.update(selected, invalid_rows=0)


def _scan_re2_metrics(
    path: Path,
    candidates: Sequence[str],
    chunk_rows: int,
) -> Tuple[ModalityAccumulator, Mapping[str, Any]]:
    accumulator = ModalityAccumulator("metrics")
    header = _header(path)
    accumulator.register_file(path, (), header=header)
    value_columns = [column for column in header if column != "time"]
    columns_by_service: Dict[str, List[str]] = defaultdict(list)
    for column in value_columns:
        service = _service_from_metric_column(column, candidates)
        if service is not None:
            columns_by_service[service].append(column)
            accumulator.service_features[service].add(column)

    file_timestamps: Set[int] = set()
    file_valid_rows = 0
    completed = False
    try:
        for chunk in pd.read_csv(path, chunksize=chunk_rows):
            accumulator.rows += len(chunk)
            timestamps, invalid = _numeric_timestamp_ms(chunk["time"], "s")
            accumulator.timestamps.update(timestamps, invalid_rows=invalid)
            file_timestamps.update(np.unique(timestamps).tolist())
            file_valid_rows += len(timestamps)

            values = chunk[value_columns]
            accumulator.value_cells += int(values.shape[0] * values.shape[1])
            accumulator.missing_value_cells += int(values.isna().sum().sum())

            numeric_time = pd.to_numeric(chunk["time"], errors="coerce")
            time_ms = pd.Series(np.nan, index=chunk.index)
            valid_time = numeric_time.notna()
            time_ms.loc[valid_time] = np.floor(
                numeric_time.loc[valid_time].astype(float) * 1000.0
            )
            for service, columns in columns_by_service.items():
                observed = chunk[columns].notna().any(axis=1) & valid_time
                selected = time_ms.loc[observed].to_numpy(dtype=np.int64)
                accumulator.service_stats(service).update(selected, invalid_rows=0)
        completed = True
    except Exception as exc:
        accumulator.incomplete_files.append(
            {"path": str(path), "error": type(exc).__name__}
        )
    if completed:
        accumulator.record_scheduled_file(
            file_timestamps, valid_timestamp_rows=file_valid_rows
        )

    return accumulator, {
        "metric_columns": len(value_columns),
        "candidate_service_metric_columns": {
            service: len(columns)
            for service, columns in sorted(columns_by_service.items())
        },
    }


def _scan_re2_event_stream(
    path: Path,
    modality_name: str,
    timestamp_column: str,
    timestamp_unit: str,
    service_column: str,
    extra_required: Sequence[str],
    chunk_rows: int,
) -> Tuple[ModalityAccumulator, Mapping[str, Any]]:
    accumulator = ModalityAccumulator(modality_name)
    header = _header(path)
    accumulator.register_file(path, (), header=header)
    columns = [timestamp_column, service_column] + list(extra_required)
    service_entities = set()

    try:
        for chunk in pd.read_csv(
            path,
            usecols=columns,
            chunksize=chunk_rows,
            keep_default_na=False,
        ):
            accumulator.rows += len(chunk)
            for field_name in columns:
                accumulator.required_missing[field_name] += int(
                    chunk[field_name].astype(str).str.len().eq(0).sum()
                )
            timestamps, invalid = _numeric_timestamp_ms(
                chunk[timestamp_column], timestamp_unit
            )
            accumulator.timestamps.update(timestamps, invalid_rows=invalid)

            numeric = pd.to_numeric(chunk[timestamp_column], errors="coerce")
            valid = numeric.notna()
            factor = {"s": 1000.0, "ms": 1.0, "us": 0.001, "ns": 0.000001}[
                timestamp_unit
            ]
            time_ms = pd.Series(np.nan, index=chunk.index)
            time_ms.loc[valid] = np.floor(
                numeric.loc[valid].astype(float) * factor
            )
            _update_service_timestamps_from_rows(
                accumulator, chunk, time_ms, service_column
            )
            service_entities.update(
                value
                for value in chunk[service_column].astype(str).str.strip().unique()
                if value
            )
    except Exception as exc:
        accumulator.incomplete_files.append(
            {"path": str(path), "error": type(exc).__name__}
        )

    return accumulator, {
        "service_entities": sorted(service_entities),
        "topology_fields_present": all(field in header for field in extra_required),
    }


def _relative_time_record(
    accumulator: ModalityAccumulator,
    anchor_ms: int,
) -> Mapping[str, Any]:
    minimum = accumulator.timestamps.minimum_ms
    maximum = accumulator.timestamps.maximum_ms
    return {
        "start_relative_seconds": (
            (minimum - anchor_ms) / 1000.0 if minimum is not None else None
        ),
        "end_relative_seconds": (
            (maximum - anchor_ms) / 1000.0 if maximum is not None else None
        ),
        "t0_within_observed_range": (
            minimum <= anchor_ms <= maximum
            if minimum is not None and maximum is not None
            else False
        ),
    }


def _single_case_window_activity(
    accumulator: ModalityAccumulator,
    anchor_ms: int,
    services: Sequence[str],
    windows_seconds: Sequence[int],
) -> Mapping[str, Any]:
    result = {}
    for window_seconds in windows_seconds:
        bins = tuple(_window_bins(anchor_ms, window_seconds))
        occupied = sum(value in accumulator.timestamps.bins for value in bins)
        present_services = sum(
            1
            for service in services
            if service in accumulator.per_service
            and any(value in accumulator.per_service[service].bins for value in bins)
        )
        result[str(window_seconds)] = {
            "occupied_30s_bins": occupied,
            "possible_30s_bins": len(bins),
            "occupied_bin_ratio": occupied / len(bins),
            "candidate_services_with_activity": present_services,
            "candidate_service_presence_ratio": present_services / len(services),
        }
    return result


def _deployment_metadata(case_directory: Path, candidates: Sequence[str]) -> Mapping[str, Any]:
    pod_files = [
        case_directory / "pod-node-1.csv",
        case_directory / "pod-node-2.csv",
    ]
    existing = [path for path in pod_files if path.is_file()]
    mapped_services = set()
    node_names = set()
    readable = 0
    for path in existing:
        try:
            table = pd.read_csv(path)
            if {"POD", "NODE_NAME"}.issubset(table.columns):
                readable += 1
                node_names.update(table["NODE_NAME"].dropna().astype(str))
                pods = table["POD"].dropna().astype(str)
                for service in candidates:
                    if pods.str.startswith(service + "-").any():
                        mapped_services.add(service)
        except Exception:
            continue
    return {
        "pod_node_files_present": len(existing),
        "pod_node_files_readable": readable,
        "deployment_nodes": len(node_names),
        "candidate_services_mapped": len(mapped_services),
        "candidate_services_total": len(candidates),
        "cluster_info_classification": (
            "log-template metadata, not service topology"
            if (case_directory / "cluster_info.json").is_file()
            else "absent"
        ),
    }


def _aggregate_re2_cases(
    case_records: Sequence[Mapping[str, Any]],
    modality_name: str,
) -> Mapping[str, Any]:
    modality_rows = [record["modalities"][modality_name] for record in case_records]
    total_rows = sum(row["rows"] for row in modality_rows)
    total_valid = sum(
        row["timestamp"]["valid_timestamp_rows"] for row in modality_rows
    )
    total_invalid = sum(
        row["timestamp"]["invalid_timestamp_rows"] for row in modality_rows
    )
    total_value_cells = sum(row["value_cells"] for row in modality_rows)
    total_missing_values = sum(row["missing_value_cells"] for row in modality_rows)
    scheduled_expected = sum(
        row["scheduled_timestamp_quality"]["expected_timestamps"]
        for row in modality_rows
    )
    scheduled_observed = sum(
        row["scheduled_timestamp_quality"]["observed_unique_timestamps"]
        for row in modality_rows
    )
    scheduled_missing = sum(
        row["scheduled_timestamp_quality"]["missing_timestamps"]
        for row in modality_rows
    )
    scheduled_duplicates = sum(
        row["scheduled_timestamp_quality"]["duplicate_timestamp_rows"]
        for row in modality_rows
    )
    required = Counter()
    for row in modality_rows:
        required.update(row["required_field_missing"])

    return {
        "cases_scanned": len(modality_rows),
        "rows": total_rows,
        "valid_timestamp_rows": total_valid,
        "invalid_timestamp_rows": total_invalid,
        "value_cells": total_value_cells,
        "missing_value_cells": total_missing_values,
        "missing_value_ratio": (
            total_missing_values / total_value_cells if total_value_cells else None
        ),
        "scheduled_timestamp_quality": {
            "expected_timestamps": scheduled_expected,
            "observed_unique_timestamps": scheduled_observed,
            "missing_timestamps": scheduled_missing,
            "missing_timestamp_ratio": (
                scheduled_missing / scheduled_expected
                if scheduled_expected
                else None
            ),
            "duplicate_timestamp_rows": scheduled_duplicates,
        },
        "required_field_missing": dict(sorted(required.items())),
        "case_rows": _finite_summary(row["rows"] for row in modality_rows),
        "case_start_relative_seconds": _finite_summary(
            record["relative_time"][modality_name]["start_relative_seconds"]
            for record in case_records
        ),
        "case_end_relative_seconds": _finite_summary(
            record["relative_time"][modality_name]["end_relative_seconds"]
            for record in case_records
        ),
        "cases_with_t0_in_observed_range": sum(
            bool(record["relative_time"][modality_name]["t0_within_observed_range"])
            for record in case_records
        ),
        "case_positive_delta_median_ms": _finite_summary(
            row["timestamp"]["positive_delta_ms_approx"]["median"]
            for row in modality_rows
            if row["timestamp"]["positive_delta_ms_approx"]["median"] is not None
        ),
        "incomplete_files": [
            value
            for row in modality_rows
            for value in row["incomplete_files"]
        ],
    }


def diagnose_re2ob(
    raw_path: str,
    adapter: RCAEvalAdapterResult,
    chunk_rows: int = 100_000,
    windows_seconds: Sequence[int] = DEFAULT_WINDOWS_SECONDS,
    max_cases: Optional[int] = None,
    progress_every: int = 10,
    dataset: str = "RCAEval-RE2-OB",
    progress_label: str = "RE2-OB cases",
) -> Mapping[str, Any]:
    """Diagnose one RCAEval release. Defaults reproduce the frozen RE2-OB report.

    ``dataset``/``progress_label`` exist so the RE2-TT protocol extension can reuse
    this scanner verbatim instead of forking it; passing neither leaves the P1
    output byte-identical.
    """

    root = Path(raw_path).resolve()
    cases_by_id = {case.case_id: case for case in adapter.inputs}
    sources = adapter.sources[:max_cases] if max_cases is not None else adapter.sources
    case_records = []
    source_paths = []
    annotation_digest = hashlib.sha256()

    for index, source in enumerate(sources, 1):
        case = cases_by_id[source.case_id]
        case_directory = root / source.relative_directory
        metrics, metric_extra = _scan_re2_metrics(
            Path(source.metrics_path), case.services, chunk_rows
        )
        logs, log_extra = _scan_re2_event_stream(
            Path(source.logs_path),
            "logs",
            "timestamp",
            "ns",
            "container_name",
            (),
            chunk_rows,
        )
        traces, trace_extra = _scan_re2_event_stream(
            Path(source.traces_path),
            "traces",
            "startTime",
            "us",
            "serviceName",
            ("traceID", "spanID", "parentSpanID"),
            chunk_rows,
        )
        modalities = {"metrics": metrics, "logs": logs, "traces": traces}
        extras = {
            "metrics": metric_extra,
            "logs": log_extra,
            "traces": trace_extra,
        }
        anchor_ms = int(float(case.anchor_time) * 1000)
        relative_time = {
            name: _relative_time_record(accumulator, anchor_ms)
            for name, accumulator in modalities.items()
        }
        case_records.append(
            {
                "case_id": case.case_id,
                "relative_time": relative_time,
                "window_activity": {
                    name: _single_case_window_activity(
                        accumulator,
                        anchor_ms,
                        case.services,
                        windows_seconds,
                    )
                    for name, accumulator in modalities.items()
                },
                "modalities": {
                    name: {**accumulator.to_record(), **extras[name]}
                    for name, accumulator in modalities.items()
                },
                "deployment_metadata": _deployment_metadata(
                    case_directory, case.services
                ),
            }
        )
        case_source_paths = [
            Path(source.metrics_path),
            Path(source.logs_path),
            Path(source.traces_path),
            Path(source.inject_time_path),
        ]
        source_paths.extend(case_source_paths)
        relative_inject = str(
            Path(source.inject_time_path).resolve().relative_to(root)
        )
        annotation_digest.update(relative_inject.encode("utf-8"))
        annotation_digest.update(b"\0")
        annotation_digest.update(Path(source.inject_time_path).read_bytes())
        annotation_digest.update(b"\n")
        _progress(progress_label, index, len(sources), progress_every)

    deployment_rows = [record["deployment_metadata"] for record in case_records]
    return {
        "dataset": dataset,
        "scan_scope": "limited-smoke" if max_cases is not None else "full",
        "cases": len(case_records),
        "available_adapter_cases": len(adapter.inputs),
        "source_root": str(root),
        "source_inventory": _layout_inventory(source_paths, root),
        "critical_annotation_sha256": {
            "inject_time_set": annotation_digest.hexdigest(),
            "digest_scope": "relative inject_time path + NUL + file bytes",
        },
        "modalities": {
            name: _aggregate_re2_cases(case_records, name)
            for name in ("metrics", "logs", "traces")
        },
        "case_diagnostics": case_records,
        "topology": {
            "explicit_static_service_graph": False,
            "trace_derived_dynamic_graph_possible_cases": sum(
                bool(
                    record["modalities"]["traces"].get(
                        "topology_fields_present"
                    )
                )
                for record in case_records
            ),
            "deployment_metadata_cases": sum(
                row["pod_node_files_readable"] > 0 for row in deployment_rows
            ),
            "deployment_candidate_service_coverage": _finite_summary(
                row["candidate_services_mapped"]
                / row["candidate_services_total"]
                for row in deployment_rows
                if row["candidate_services_total"]
            ),
            "cluster_info_classification": (
                "log-template metadata, not service topology"
            ),
        },
        "diagnostic_cautions": [
            "event-stream timestamp collisions reflect multiple events and are not duplicate-record proof",
            "log/trace empty activity bins indicate no observed event, not scheduled-sample missingness",
            "layout digests bind relative paths and sizes, not full telemetry bytes",
        ],
    }


def build_telemetry_diagnostics(
    gaia_path: str,
    re2ob_path: str,
    gaia_adapter: GAIAAdapterResult,
    re2ob_adapter: RCAEvalAdapterResult,
    chunk_rows: int = 100_000,
    windows_seconds: Sequence[int] = DEFAULT_WINDOWS_SECONDS,
    max_gaia_files_per_modality: Optional[int] = None,
    max_re2ob_cases: Optional[int] = None,
    progress_every: int = 100,
) -> Mapping[str, Any]:
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    windows = tuple(sorted(set(int(value) for value in windows_seconds)))
    if not windows or any(value <= 0 for value in windows):
        raise ValueError("windows_seconds must contain positive values")

    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "evidence_scope": "raw telemetry streaming diagnostics",
        "definitions": {
            "metric_missingness": (
                "empty/NaN value cells plus separately reported per-file "
                "scheduled timestamp gaps"
            ),
            "event_parse_failure": "missing required field or unparseable timestamp",
            "event_activity_coverage": (
                "occupied 30s bins; absence is not classified as missingness"
            ),
            "candidate_windows_seconds": list(windows),
        },
        "gaia": diagnose_gaia(
            gaia_path,
            gaia_adapter,
            chunk_rows=chunk_rows,
            windows_seconds=windows,
            max_files_per_modality=max_gaia_files_per_modality,
            progress_every=progress_every,
        ),
        "re2ob": diagnose_re2ob(
            re2ob_path,
            re2ob_adapter,
            chunk_rows=chunk_rows,
            windows_seconds=windows,
            max_cases=max_re2ob_cases,
            progress_every=max(1, min(progress_every, 10)),
        ),
        "limitations": [
            "no T_pre/T_post or split strategy is selected by this diagnostic",
            "full telemetry content checksums are not computed",
            "trace-derived graphs are identified as possible but not materialized",
        ],
    }
