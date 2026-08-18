#!/usr/bin/env python3
"""Extract real-data L0/T0 smoke bundles without reading any labels."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_manifest_cases
from src.features import (
    FeatureRow,
    RE2_TRACE_SERVICE_ALIASES,
    log_stream_features,
    normalize_trace_service,
    trace_stream_features,
    verify_feature_bundle,
    write_feature_bundle,
)


LOG_EXTRACTOR = "p2_log_l0_v1"
TRACE_EXTRACTOR = "p2_trace_t0_v1"
ERROR_LEVELS = frozenset({"critical", "error", "fatal"})
KNOWN_LEVELS = frozenset(
    {"critical", "debug", "error", "fatal", "info", "trace", "warn", "warning"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> tuple:
    with path.open("r", encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle)


def _local_ms(values, date_format=None):
    parsed = pd.to_datetime(values, format=date_format, errors="coerce")
    localized = parsed.dt.tz_localize(
        "Asia/Shanghai", ambiguous="NaT", nonexistent="NaT"
    )
    valid = localized.notna().to_numpy()
    numeric = localized.astype("int64").to_numpy(dtype=np.int64) // 1_000_000
    return numeric, valid


def _severity_flags(levels, explicit_error=None):
    normalized = levels.astype("string").str.strip().str.lower()
    known = normalized.isin(KNOWN_LEVELS).to_numpy()
    error = normalized.isin(ERROR_LEVELS).to_numpy()
    if explicit_error is not None:
        explicit = explicit_error.notna().to_numpy()
        error |= explicit
        known |= explicit
    result = np.full(len(levels), np.nan, dtype=np.float64)
    result[known] = error[known].astype(np.float64)
    return result


def _gaia_log_window(path, start_ms, end_ms, chunk_rows):
    timestamps = []
    errors = []
    lengths = []
    rows_scanned = 0
    for chunk_index, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=["message"],
            chunksize=chunk_rows,
            keep_default_na=False,
        ),
        start=1,
    ):
        rows_scanned += len(chunk)
        messages = chunk["message"].astype("string")
        prefixes = messages.str.extract(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})",
            expand=False,
        )
        numeric, valid = _local_ms(prefixes, "%Y-%m-%d %H:%M:%S,%f")
        wanted = valid & (numeric >= start_ms) & (numeric < end_ms)
        if wanted.any():
            selected = messages[wanted]
            levels = selected.str.extract(
                r"\|\s*([A-Za-z]+)\s*\|", expand=False
            )
            timestamps.append(numeric[wanted])
            errors.append(_severity_flags(levels))
            lengths.append(selected.str.len().to_numpy(dtype=np.float64))
        if chunk_index % 10 == 0:
            print(
                "[event-smoke] GAIA log rows {:,}".format(rows_scanned),
                flush=True,
            )
    return (
        np.concatenate(timestamps) if timestamps else np.empty(0, dtype=np.int64),
        np.concatenate(errors) if errors else np.empty(0, dtype=np.float64),
        np.concatenate(lengths) if lengths else np.empty(0, dtype=np.float64),
        rows_scanned,
    )


def _gaia_trace_window(path, start_ms, end_ms, chunk_rows):
    collected = {name: [] for name in ("timestamps", "durations", "errors", "traces", "operations", "parents")}
    rows_scanned = 0
    columns = [
        "start_time",
        "end_time",
        "status_code",
        "trace_id",
        "url",
        "parent_id",
    ]
    for chunk_index, chunk in enumerate(
        pd.read_csv(path, usecols=columns, chunksize=chunk_rows, keep_default_na=False),
        start=1,
    ):
        rows_scanned += len(chunk)
        start_numeric, valid_start = _local_ms(chunk["start_time"])
        wanted = valid_start & (start_numeric >= start_ms) & (start_numeric < end_ms)
        if wanted.any():
            selected = chunk.loc[wanted]
            start = pd.to_datetime(selected["start_time"], errors="coerce")
            end = pd.to_datetime(selected["end_time"], errors="coerce")
            duration = (end - start).dt.total_seconds().to_numpy(dtype=np.float64)
            status = pd.to_numeric(selected["status_code"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            error = np.full(len(selected), np.nan, dtype=np.float64)
            finite_status = np.isfinite(status)
            error[finite_status] = (status[finite_status] >= 500).astype(np.float64)
            operations = (
                selected["url"]
                .astype("string")
                .str.replace(r"^https?://[^/]+", "", regex=True)
                .str.split("?", n=1, regex=False)
                .str[0]
                .fillna("")
                .to_numpy(dtype=str)
            )
            collected["timestamps"].append(start_numeric[wanted])
            collected["durations"].append(duration)
            collected["errors"].append(error)
            collected["traces"].append(selected["trace_id"].astype(str).to_numpy())
            collected["operations"].append(operations)
            collected["parents"].append(
                selected["parent_id"].astype(str).str.len().gt(0).to_numpy(dtype=np.float64)
            )
        if chunk_index % 10 == 0:
            print(
                "[event-smoke] GAIA trace rows {:,}".format(rows_scanned),
                flush=True,
            )
    result = {}
    dtypes = {
        "timestamps": np.int64,
        "durations": np.float64,
        "errors": np.float64,
        "traces": str,
        "operations": str,
        "parents": np.float64,
    }
    for name, parts in collected.items():
        result[name] = np.concatenate(parts) if parts else np.empty(0, dtype=dtypes[name])
    return result, rows_scanned


def _re2_logs(frame, case_input):
    numeric = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    valid_time = np.isfinite(numeric)
    timestamps = np.floor(numeric[valid_time] / 1_000_000.0).astype(np.int64)
    services = frame.loc[valid_time, "container_name"].astype(str).to_numpy()
    flags = _severity_flags(
        frame.loc[valid_time, "level"], frame.loc[valid_time, "error"]
    )
    lengths = frame.loc[valid_time, "message"].astype("string").str.len().to_numpy(dtype=np.float64)
    rows = []
    observed_entities = set(services)
    anchor_ms = int(float(case_input.anchor_time) * 1000)
    for service in case_input.services:
        wanted = services == service
        names, values, observed = log_stream_features(
            timestamps[wanted], flags[wanted], lengths[wanted], anchor_ms,
            entity_observed=service in observed_entities,
        )
        rows.append(FeatureRow(case_input.case_id, service, LOG_EXTRACTOR, names, values, observed))
    return tuple(rows), sorted(observed_entities)


def _re2_traces(frame, case_input):
    numeric = pd.to_numeric(frame["startTime"], errors="coerce").to_numpy(dtype=np.float64)
    valid_time = np.isfinite(numeric)
    timestamps = np.floor(numeric[valid_time] / 1000.0).astype(np.int64)
    raw_services = frame.loc[valid_time, "serviceName"].astype(str).to_numpy()
    services = np.asarray(
        [normalize_trace_service(value, case_input.services, RE2_TRACE_SERVICE_ALIASES) or "" for value in raw_services],
        dtype=str,
    )
    durations = pd.to_numeric(frame.loc[valid_time, "duration"], errors="coerce").to_numpy(dtype=np.float64) / 1_000_000.0
    status = pd.to_numeric(frame.loc[valid_time, "statusCode"], errors="coerce").to_numpy(dtype=np.float64)
    flags = np.full(len(status), np.nan, dtype=np.float64)
    finite_status = np.isfinite(status)
    flags[finite_status] = (status[finite_status] != 0).astype(np.float64)
    traces = frame.loc[valid_time, "traceID"].fillna("").astype(str).to_numpy()
    operations = frame.loc[valid_time, "operationName"].fillna("").astype(str).to_numpy()
    parents = frame.loc[valid_time, "parentSpanID"].notna().to_numpy(dtype=np.float64)
    rows = []
    observed_entities = set(services) - {""}
    anchor_ms = int(float(case_input.anchor_time) * 1000)
    for service in case_input.services:
        wanted = services == service
        names, values, observed = trace_stream_features(
            timestamps[wanted], durations[wanted], flags[wanted], traces[wanted],
            operations[wanted], parents[wanted], anchor_ms,
            entity_observed=service in observed_entities,
        )
        rows.append(FeatureRow(case_input.case_id, service, TRACE_EXTRACTOR, names, values, observed))
    return tuple(rows), sorted(observed_entities), sorted(set(raw_services))


def _write_bundle(path, dataset, inputs, rows, config, source):
    manifest = write_feature_bundle(str(path), dataset, inputs, rows, config, source)
    verify_feature_bundle(str(path), inputs)
    return {
        "case_count": manifest["case_count"],
        "feature_count": manifest["feature_count"],
        "manifest_sha256": _sha256(path / "manifest.json"),
        "observed_value_count": sum(sum(row.observed) for row in rows),
        "service_row_count": manifest["service_row_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="artifacts/p1/manifests")
    parser.add_argument("--output-root", default="artifacts/p2/event_features_smoke")
    parser.add_argument("--chunk-rows", type=int, default=200000)
    parser.add_argument("--gaia-service", default="dbservice1")
    args = parser.parse_args()

    manifest_root = Path(args.manifest_root)
    output_root = Path(args.output_root)
    telemetry_path = Path("artifacts/p1/telemetry_diagnostics.json")
    telemetry_sha = _sha256(telemetry_path)
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    gaia_source = _read_jsonl(manifest_root / "gaia" / "sources.jsonl")[0]
    gaia_inputs, _ = read_manifest_cases(str(manifest_root / "gaia"))
    main_ids = {row["case_id"] for row in _read_jsonl(Path("artifacts/p1/inclusion/gaia/main_cohort.jsonl"))}
    log_time = telemetry["gaia"]["modalities"]["logs"]["per_service"][args.gaia_service]
    trace_time = telemetry["gaia"]["modalities"]["traces"]["per_service"][args.gaia_service]
    shared_start = max(log_time["timestamp_min_ms"], trace_time["timestamp_min_ms"]) + 300000
    shared_end = min(log_time["timestamp_max_ms"], trace_time["timestamp_max_ms"]) - 300000
    shared_midpoint = (shared_start + shared_end) // 2
    eligible_gaia = tuple(
        row
        for row in gaia_inputs
        if row.case_id in main_ids and shared_start <= int(row.anchor_time) < shared_end
    )
    if not eligible_gaia:
        raise ValueError("no GAIA main case lies inside selected service coverage")
    gaia_case = min(
        eligible_gaia,
        key=lambda row: (abs(int(row.anchor_time) - shared_midpoint), row.case_id),
    )
    gaia_case = replace(gaia_case, services=(args.gaia_service,))
    window_start = int(gaia_case.anchor_time) - 300000
    window_end = int(gaia_case.anchor_time) + 300000

    log_path = Path(gaia_source["logs_directory"]) / "business_table_{}_2021-07.csv".format(args.gaia_service)
    timestamps, errors, lengths, log_rows_scanned = _gaia_log_window(
        log_path, window_start, window_end, args.chunk_rows
    )
    names, values, observed = log_stream_features(
        timestamps, errors, lengths, int(gaia_case.anchor_time), entity_observed=True
    )
    gaia_log_rows = (FeatureRow(gaia_case.case_id, args.gaia_service, LOG_EXTRACTOR, names, values, observed),)

    trace_path = Path(gaia_source["traces_directory"]) / "trace_table_{}_2021-07.csv".format(args.gaia_service)
    trace_data, trace_rows_scanned = _gaia_trace_window(
        trace_path, window_start, window_end, args.chunk_rows
    )
    names, values, observed = trace_stream_features(
        trace_data["timestamps"], trace_data["durations"], trace_data["errors"],
        trace_data["traces"], trace_data["operations"], trace_data["parents"],
        int(gaia_case.anchor_time), entity_observed=True,
    )
    gaia_trace_rows = (FeatureRow(gaia_case.case_id, args.gaia_service, TRACE_EXTRACTOR, names, values, observed),)

    re2_inputs, _ = read_manifest_cases(str(manifest_root / "re2ob"))
    re2_case = sorted(re2_inputs, key=lambda row: row.case_id)[0]
    source_by_case = {row["case_id"]: row for row in _read_jsonl(manifest_root / "re2ob" / "sources.jsonl")}
    re2_source = source_by_case[re2_case.case_id]
    re2_log_frame = pd.read_csv(re2_source["logs_path"])
    re2_log_rows, re2_log_entities = _re2_logs(re2_log_frame, re2_case)
    re2_trace_frame = pd.read_csv(re2_source["traces_path"])
    re2_trace_rows, re2_trace_entities, re2_raw_trace_entities = _re2_traces(re2_trace_frame, re2_case)

    common = {"onset_candidates_seconds": [30, 60, 120], "pre_seconds": 300, "post_seconds": 300, "smoke": True}
    source_common = {"telemetry_diagnostics_sha256": telemetry_sha}
    summary = {
        "gaia_log": _write_bundle(
            output_root / "gaia_log", "GAIA-MicroSS-2021-07/main", (gaia_case,), gaia_log_rows,
            {**common, "raw_rows_scanned": log_rows_scanned, "selected_window_events": len(timestamps)},
            {**source_common, "dataset_manifest_sha256": _sha256(manifest_root / "gaia" / "manifest.json"), "raw_file": str(log_path), "raw_file_bytes": log_path.stat().st_size},
        ),
        "gaia_trace": _write_bundle(
            output_root / "gaia_trace", "GAIA-MicroSS-2021-07/main", (gaia_case,), gaia_trace_rows,
            {**common, "raw_rows_scanned": trace_rows_scanned, "selected_window_events": len(trace_data["timestamps"])},
            {**source_common, "dataset_manifest_sha256": _sha256(manifest_root / "gaia" / "manifest.json"), "raw_file": str(trace_path), "raw_file_bytes": trace_path.stat().st_size},
        ),
        "re2ob_log": _write_bundle(
            output_root / "re2ob_log", "RCAEval-RE2-OB", (re2_case,), re2_log_rows,
            {**common, "observed_entities": re2_log_entities, "raw_rows_scanned": len(re2_log_frame)},
            {**source_common, "dataset_manifest_sha256": _sha256(manifest_root / "re2ob" / "manifest.json"), "source_snapshot_manifest_sha256": _sha256(Path("artifacts/p1/source_snapshots/re2ob/manifest.json"))},
        ),
        "re2ob_trace": _write_bundle(
            output_root / "re2ob_trace", "RCAEval-RE2-OB", (re2_case,), re2_trace_rows,
            {**common, "observed_entities": re2_trace_entities, "raw_entities": re2_raw_trace_entities, "raw_rows_scanned": len(re2_trace_frame), "service_aliases": RE2_TRACE_SERVICE_ALIASES},
            {**source_common, "dataset_manifest_sha256": _sha256(manifest_root / "re2ob" / "manifest.json"), "source_snapshot_manifest_sha256": _sha256(Path("artifacts/p1/source_snapshots/re2ob/manifest.json"))},
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
