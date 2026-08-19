#!/usr/bin/env python3
"""Extract recoverable full P2 L0/T0 bundles for GAIA main and RE2-OB."""

import argparse
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
    DEFAULT_EVENT_ONSET_SECONDS,
    FeatureRow,
    PreparedLogStream,
    PreparedTraceStream,
    RE2_TRACE_SERVICE_ALIASES,
    load_event_checkpoint,
    load_feature_matrices,
    log_feature_names,
    log_stream_features,
    normalize_trace_service,
    trace_feature_names,
    trace_stream_features,
    verify_feature_bundle,
    write_event_checkpoint,
    write_feature_bundle,
)


LOG_EXTRACTOR = "p2_log_l0_v1"
TRACE_EXTRACTOR = "p2_trace_t0_v1"
ERROR_LEVELS = frozenset({"critical", "error", "fatal"})
KNOWN_LEVELS = frozenset(
    {"critical", "debug", "error", "fatal", "info", "trace", "warn", "warning"}
)
PRE_MS = 300000
POST_MS = 300000
SCORE_CAP = 20.0
# Frozen before the full coverage results are read. This is an extraction-support
# diagnostic, not a model-selection or claim threshold.
STAGE_ABSOLUTE_COVERAGE_MIN = 0.80
STAGE_RELATIVE_TO_WHOLE_MIN = 0.90


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> tuple:
    with path.open("r", encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle)


def _write_json(path: Path, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_digest(record) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_binding(path: Path) -> dict:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return {"raw_file": str(resolved), "raw_file_bytes": resolved.stat().st_size}


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


def _factorize_identifiers(parts):
    if not parts:
        return np.empty(0, dtype=np.int32), 0
    values = np.concatenate(parts).astype(object, copy=False)
    missing = pd.isna(values)
    try:
        missing |= values == ""
    except TypeError:
        pass
    if np.any(missing):
        values = values.copy()
        values[missing] = None
    codes, uniques = pd.factorize(values, sort=False, use_na_sentinel=True)
    if len(uniques) >= np.iinfo(np.int32).max:
        raise ValueError("identifier cardinality exceeds int32 checkpoint boundary")
    return codes.astype(np.int32, copy=False), len(uniques)


def _gaia_log_stream(path, minimum_ms, maximum_ms, chunk_rows, progress_every):
    timestamps = []
    errors = []
    lengths = []
    rows_scanned = 0
    invalid_timestamps = 0
    raw_min = None
    raw_max = None
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
        prefixes = messages.str.slice(0, 23)
        numeric, valid = _local_ms(prefixes, "%Y-%m-%d %H:%M:%S,%f")
        invalid_timestamps += int((~valid).sum())
        if valid.any():
            valid_times = numeric[valid]
            chunk_min = int(valid_times.min())
            chunk_max = int(valid_times.max())
            raw_min = chunk_min if raw_min is None else min(raw_min, chunk_min)
            raw_max = chunk_max if raw_max is None else max(raw_max, chunk_max)
        wanted = valid & (numeric >= minimum_ms) & (numeric < maximum_ms)
        if wanted.any():
            selected = messages[wanted]
            levels = selected.str.extract(r"\|\s*([A-Za-z]+)\s*\|", expand=False)
            timestamps.append(numeric[wanted])
            errors.append(_severity_flags(levels))
            lengths.append(selected.str.len().to_numpy(dtype=np.float64))
        if progress_every > 0 and chunk_index % progress_every == 0:
            print(
                "[p2-event] GAIA log {} rows {:,}".format(path.stem, rows_scanned),
                flush=True,
            )
    if raw_min is None or raw_max is None:
        raise ValueError("GAIA log source has no valid timestamps: {}".format(path))
    timestamp_array = (
        np.concatenate(timestamps) if timestamps else np.empty(0, dtype=np.int64)
    )
    error_array = (
        np.concatenate(errors) if errors else np.empty(0, dtype=np.float64)
    )
    length_array = (
        np.concatenate(lengths) if lengths else np.empty(0, dtype=np.float64)
    )
    return PreparedLogStream(timestamp_array, error_array, length_array), {
        "invalid_timestamp_rows": invalid_timestamps,
        "raw_rows_scanned": rows_scanned,
        "retained_window_envelope_events": len(timestamp_array),
        "timestamp_max_ms": raw_max,
        "timestamp_min_ms": raw_min,
    }


def _gaia_trace_stream(path, minimum_ms, maximum_ms, chunk_rows, progress_every):
    collected = {
        name: []
        for name in (
            "timestamps",
            "durations",
            "errors",
            "traces",
            "operations",
            "parents",
        )
    }
    rows_scanned = 0
    invalid_timestamps = 0
    raw_min = None
    raw_max = None
    columns = [
        "start_time",
        "end_time",
        "status_code",
        "trace_id",
        "url",
        "parent_id",
    ]
    for chunk_index, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=columns,
            chunksize=chunk_rows,
            keep_default_na=False,
        ),
        start=1,
    ):
        rows_scanned += len(chunk)
        start_numeric, valid_start = _local_ms(chunk["start_time"])
        invalid_timestamps += int((~valid_start).sum())
        if valid_start.any():
            valid_times = start_numeric[valid_start]
            chunk_min = int(valid_times.min())
            chunk_max = int(valid_times.max())
            raw_min = chunk_min if raw_min is None else min(raw_min, chunk_min)
            raw_max = chunk_max if raw_max is None else max(raw_max, chunk_max)
        wanted = valid_start & (start_numeric >= minimum_ms) & (start_numeric < maximum_ms)
        if wanted.any():
            selected = chunk.loc[wanted]
            start = pd.to_datetime(selected["start_time"], errors="coerce")
            end = pd.to_datetime(selected["end_time"], errors="coerce")
            duration = (end - start).dt.total_seconds().to_numpy(dtype=np.float64)
            status = pd.to_numeric(
                selected["status_code"], errors="coerce"
            ).to_numpy(dtype=np.float64)
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
                .to_numpy(dtype=object)
            )
            collected["timestamps"].append(start_numeric[wanted])
            collected["durations"].append(duration)
            collected["errors"].append(error)
            collected["traces"].append(
                selected["trace_id"].astype(str).to_numpy(dtype=object)
            )
            collected["operations"].append(operations)
            collected["parents"].append(
                selected["parent_id"]
                .astype(str)
                .str.len()
                .gt(0)
                .to_numpy(dtype=np.float64)
            )
        if progress_every > 0 and chunk_index % progress_every == 0:
            print(
                "[p2-event] GAIA trace {} rows {:,}".format(path.stem, rows_scanned),
                flush=True,
            )
    if raw_min is None or raw_max is None:
        raise ValueError("GAIA trace source has no valid timestamps: {}".format(path))
    arrays = {}
    dtypes = {
        "timestamps": np.int64,
        "durations": np.float64,
        "errors": np.float64,
        "parents": np.float64,
    }
    for name, dtype in dtypes.items():
        arrays[name] = (
            np.concatenate(collected[name])
            if collected[name]
            else np.empty(0, dtype=dtype)
        )
    trace_codes, trace_cardinality = _factorize_identifiers(collected["traces"])
    operation_codes, operation_cardinality = _factorize_identifiers(
        collected["operations"]
    )
    stream = PreparedTraceStream.from_identifier_codes(
        arrays["timestamps"],
        arrays["durations"],
        arrays["errors"],
        trace_codes,
        operation_codes,
        arrays["parents"],
    )
    return stream, {
        "invalid_timestamp_rows": invalid_timestamps,
        "operation_cardinality": operation_cardinality,
        "raw_rows_scanned": rows_scanned,
        "retained_window_envelope_events": len(arrays["timestamps"]),
        "timestamp_max_ms": raw_max,
        "timestamp_min_ms": raw_min,
        "trace_cardinality": trace_cardinality,
    }


def _checkpoint_or_extract_gaia(
    inputs,
    service,
    modality,
    path,
    checkpoint_root,
    source_binding,
    minimum_ms,
    maximum_ms,
    chunk_rows,
    chunk_progress_every,
    feature_progress_every,
):
    extractor = LOG_EXTRACTOR if modality == "log" else TRACE_EXTRACTOR
    names = log_feature_names() if modality == "log" else trace_feature_names()
    checkpoint_id = "gaia_main/{}/{}".format(modality, service)
    output = checkpoint_root / "gaia_main" / modality / service
    pairs = tuple((case_input.case_id, service) for case_input in inputs)
    if (output / "manifest.json").is_file():
        values, observed, manifest = load_event_checkpoint(
            str(output), checkpoint_id, extractor, names, pairs, source_binding
        )
        print("[p2-event] resume {}".format(checkpoint_id), flush=True)
        return values, observed, manifest, True

    if modality == "log":
        stream, stats = _gaia_log_stream(
            path,
            minimum_ms,
            maximum_ms,
            chunk_rows,
            chunk_progress_every,
        )
    else:
        stream, stats = _gaia_trace_stream(
            path,
            minimum_ms,
            maximum_ms,
            chunk_rows,
            chunk_progress_every,
        )
    values = np.empty((len(inputs), len(names)), dtype="<f4")
    observed = np.empty((len(inputs), len(names)), dtype=np.bool_)
    fully_supported = 0
    for index, case_input in enumerate(inputs, start=1):
        anchor_ms = int(case_input.anchor_time)
        source_supported = (
            stats["timestamp_min_ms"] <= anchor_ms - PRE_MS
            and stats["timestamp_max_ms"] >= anchor_ms + POST_MS
        )
        fully_supported += int(source_supported)
        feature_names, row_values, row_observed = stream.features(
            anchor_ms, entity_observed=source_supported
        )
        if feature_names != names:
            raise ValueError("event feature schema drift")
        values[index - 1] = row_values
        observed[index - 1] = row_observed
        if feature_progress_every > 0 and (
            index % feature_progress_every == 0 or index == len(inputs)
        ):
            print(
                "[p2-event] {} features {}/{}".format(
                    checkpoint_id, index, len(inputs)
                ),
                flush=True,
            )
    stats["fully_source_supported_cases"] = fully_supported
    manifest = write_event_checkpoint(
        str(output),
        checkpoint_id,
        extractor,
        names,
        pairs,
        values,
        observed,
        source_binding,
        stats,
    )
    verified_values, verified_observed, verified_manifest = load_event_checkpoint(
        str(output), checkpoint_id, extractor, names, pairs, source_binding
    )
    return verified_values, verified_observed, verified_manifest, False


def _re2_log_features(frame, case_input):
    numeric = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid_time = np.isfinite(numeric)
    timestamps = np.floor(numeric[valid_time] / 1_000_000.0).astype(np.int64)
    services = frame.loc[valid_time, "container_name"].astype(str).to_numpy()
    flags = _severity_flags(
        frame.loc[valid_time, "level"], frame.loc[valid_time, "error"]
    )
    lengths = (
        frame.loc[valid_time, "message"]
        .astype("string")
        .str.len()
        .to_numpy(dtype=np.float64)
    )
    observed_entities = set(services)
    anchor_ms = int(float(case_input.anchor_time) * 1000)
    values = np.empty((len(case_input.services), len(log_feature_names())), dtype="<f4")
    observed = np.empty_like(values, dtype=np.bool_)
    for index, service in enumerate(case_input.services):
        wanted = services == service
        names, row_values, row_observed = log_stream_features(
            timestamps[wanted],
            flags[wanted],
            lengths[wanted],
            anchor_ms,
            entity_observed=service in observed_entities,
        )
        if names != log_feature_names():
            raise ValueError("RE2 log feature schema drift")
        values[index] = row_values
        observed[index] = row_observed
    return values, observed, {
        "invalid_timestamp_rows": int((~valid_time).sum()),
        "observed_entities": sorted(observed_entities),
        "raw_rows_scanned": len(frame),
        "timestamp_max_ms": int(timestamps.max()) if len(timestamps) else None,
        "timestamp_min_ms": int(timestamps.min()) if len(timestamps) else None,
    }


def _re2_trace_features(frame, case_input):
    numeric = pd.to_numeric(frame["startTime"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid_time = np.isfinite(numeric)
    timestamps = np.floor(numeric[valid_time] / 1000.0).astype(np.int64)
    raw_services = frame.loc[valid_time, "serviceName"].astype(str).to_numpy()
    services = np.asarray(
        [
            normalize_trace_service(
                value, case_input.services, RE2_TRACE_SERVICE_ALIASES
            )
            or ""
            for value in raw_services
        ],
        dtype=str,
    )
    durations = (
        pd.to_numeric(frame.loc[valid_time, "duration"], errors="coerce")
        .to_numpy(dtype=np.float64)
        / 1_000_000.0
    )
    status = pd.to_numeric(
        frame.loc[valid_time, "statusCode"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    flags = np.full(len(status), np.nan, dtype=np.float64)
    finite_status = np.isfinite(status)
    flags[finite_status] = (status[finite_status] != 0).astype(np.float64)
    traces = frame.loc[valid_time, "traceID"].fillna("").astype(str).to_numpy()
    operations = (
        frame.loc[valid_time, "operationName"].fillna("").astype(str).to_numpy()
    )
    parents = frame.loc[valid_time, "parentSpanID"].notna().to_numpy(dtype=np.float64)
    observed_entities = set(services) - {""}
    anchor_ms = int(float(case_input.anchor_time) * 1000)
    values = np.empty((len(case_input.services), len(trace_feature_names())), dtype="<f4")
    observed = np.empty_like(values, dtype=np.bool_)
    for index, service in enumerate(case_input.services):
        wanted = services == service
        names, row_values, row_observed = trace_stream_features(
            timestamps[wanted],
            durations[wanted],
            flags[wanted],
            traces[wanted],
            operations[wanted],
            parents[wanted],
            anchor_ms,
            entity_observed=service in observed_entities,
        )
        if names != trace_feature_names():
            raise ValueError("RE2 trace feature schema drift")
        values[index] = row_values
        observed[index] = row_observed
    return values, observed, {
        "invalid_timestamp_rows": int((~valid_time).sum()),
        "observed_entities": sorted(observed_entities),
        "raw_entities": sorted(set(raw_services)),
        "raw_rows_scanned": len(frame),
        "timestamp_max_ms": int(timestamps.max()) if len(timestamps) else None,
        "timestamp_min_ms": int(timestamps.min()) if len(timestamps) else None,
        "unknown_raw_entities": sorted(
            set(raw_services)
            - set(case_input.services)
            - set(RE2_TRACE_SERVICE_ALIASES)
        ),
    }


def _checkpoint_or_extract_re2(
    case_input,
    source,
    modality,
    checkpoint_root,
    source_binding,
):
    extractor = LOG_EXTRACTOR if modality == "log" else TRACE_EXTRACTOR
    names = log_feature_names() if modality == "log" else trace_feature_names()
    checkpoint_id = "re2ob/{}/{}".format(modality, case_input.case_id)
    output = checkpoint_root / "re2ob" / modality / case_input.case_id
    pairs = tuple((case_input.case_id, service) for service in case_input.services)
    if (output / "manifest.json").is_file():
        values, observed, manifest = load_event_checkpoint(
            str(output), checkpoint_id, extractor, names, pairs, source_binding
        )
        return values, observed, manifest, True
    if modality == "log":
        frame = pd.read_csv(
            source["logs_path"],
            usecols=["timestamp", "container_name", "message", "level", "error"],
        )
        values, observed, stats = _re2_log_features(frame, case_input)
    else:
        frame = pd.read_csv(
            source["traces_path"],
            usecols=[
                "traceID",
                "serviceName",
                "operationName",
                "startTime",
                "duration",
                "statusCode",
                "parentSpanID",
            ],
        )
        values, observed, stats = _re2_trace_features(frame, case_input)
    manifest = write_event_checkpoint(
        str(output),
        checkpoint_id,
        extractor,
        names,
        pairs,
        values,
        observed,
        source_binding,
        stats,
    )
    verified_values, verified_observed, verified_manifest = load_event_checkpoint(
        str(output), checkpoint_id, extractor, names, pairs, source_binding
    )
    return verified_values, verified_observed, verified_manifest, False


def _checkpoint_rows(checkpoints, extractor, feature_names):
    for pairs, values, observed, _ in checkpoints:
        for index, (case_id, service) in enumerate(pairs):
            yield FeatureRow(
                case_id,
                service,
                extractor,
                feature_names,
                tuple(float(value) for value in values[index]),
                tuple(bool(value) for value in observed[index]),
            )


def _checkpoint_set_digest(checkpoints):
    records = [
        {
            "checkpoint_id": manifest["checkpoint_id"],
            "observed_sha256": manifest["files"]["observed.npy"]["sha256"],
            "values_sha256": manifest["files"]["values.npy"]["sha256"],
        }
        for _, _, _, manifest in checkpoints
    ]
    return _canonical_digest(sorted(records, key=lambda row: row["checkpoint_id"]))


def _write_dataset_bundle(
    output,
    dataset,
    inputs,
    modality,
    checkpoints,
    source_bindings,
):
    extractor = LOG_EXTRACTOR if modality == "log" else TRACE_EXTRACTOR
    names = log_feature_names() if modality == "log" else trace_feature_names()
    stats = [checkpoint[3]["extraction_stats"] for checkpoint in checkpoints]
    config = {
        "checkpoint_count": len(checkpoints),
        "checkpoint_unit": "service" if dataset.startswith("GAIA") else "case",
        "onset_candidates_seconds": list(DEFAULT_EVENT_ONSET_SECONDS),
        "post_seconds": POST_MS // 1000,
        "pre_seconds": PRE_MS // 1000,
        "raw_rows_scanned": sum(row["raw_rows_scanned"] for row in stats),
        "score_cap": SCORE_CAP,
        "source_availability_rule": "full [-300s,+300s) coverage; missing entities fully masked",
    }
    manifest = write_feature_bundle(
        str(output),
        dataset,
        inputs,
        _checkpoint_rows(checkpoints, extractor, names),
        config,
        {
            **source_bindings,
            "checkpoint_set_sha256": _checkpoint_set_digest(checkpoints),
        },
    )
    verify_feature_bundle(str(output), inputs)
    return {
        "case_count": manifest["case_count"],
        "checkpoint_count": len(checkpoints),
        "feature_count": manifest["feature_count"],
        "manifest_sha256": _sha256(output / "manifest.json"),
        "observed_sha256": manifest["files"]["observed.npy"]["sha256"],
        "raw_rows_scanned": config["raw_rows_scanned"],
        "service_row_count": manifest["service_row_count"],
        "values_sha256": manifest["files"]["values.npy"]["sha256"],
    }


def _content_fields(modality):
    if modality == "log":
        return (
            "error_fraction_shift",
            "message_length_mean_shift",
            "message_length_log_scale_ratio",
        )
    return (
        "error_fraction_shift",
        "duration_median_ratio",
        "duration_p90_ratio",
        "duration_p99_ratio",
        "unique_trace_rate_ratio",
        "unique_operation_rate_ratio",
        "parent_fraction_shift",
    )


def _coverage_audit(output, inputs, modality):
    manifest, index, values, observed = load_feature_matrices(str(output))
    verify_feature_bundle(str(output), inputs)
    names = tuple(manifest["feature_names"])
    entity_column = names.index(
        "whole.log_event_rate_ratio" if modality == "log" else "whole.span_rate_ratio"
    )
    entity_rows = np.asarray(observed[:, entity_column], dtype=bool)
    entity_count = int(entity_rows.sum())
    intervals = tuple(dict.fromkeys(name.rsplit(".", 1)[0] for name in names))
    content_fields = _content_fields(modality)
    interval_coverage = {}
    for interval in intervals:
        columns = [
            names.index("{}.{}".format(interval, field)) for field in content_fields
        ]
        block = np.asarray(observed[:, columns], dtype=bool)
        if entity_count:
            entity_block = block[entity_rows]
            cell_ratio = float(entity_block.mean())
            complete_ratio = float(entity_block.all(axis=1).mean())
        else:
            cell_ratio = 0.0
            complete_ratio = 0.0
        interval_coverage[interval] = {
            "content_cell_ratio_among_observed_entities": cell_ratio,
            "content_complete_row_ratio_among_observed_entities": complete_ratio,
        }
    whole = interval_coverage["whole"][
        "content_complete_row_ratio_among_observed_entities"
    ]
    onset_support = {}
    for onset in DEFAULT_EVENT_ONSET_SECONDS:
        blocks = [
            "stage{}.pre_onset".format(onset),
            "stage{}.pre_impact".format(onset),
            "stage{}.onset_impact".format(onset),
        ]
        minimum = min(
            interval_coverage[name][
                "content_complete_row_ratio_among_observed_entities"
            ]
            for name in blocks
        )
        relative = minimum / whole if whole > 0 else 0.0
        onset_support[str(onset)] = {
            "absolute_min_complete_ratio": minimum,
            "relative_to_whole": relative,
            "supported": bool(
                minimum >= STAGE_ABSOLUTE_COVERAGE_MIN
                and relative >= STAGE_RELATIVE_TO_WHOLE_MIN
            ),
        }
    return {
        "all_values_finite": bool(np.isfinite(values).all()),
        "entity_observed_row_count": entity_count,
        "entity_observed_row_ratio": entity_count / len(index),
        "feature_observed_ratios": {
            name: float(np.asarray(observed[:, column], dtype=bool).mean())
            for column, name in enumerate(names)
        },
        "interval_coverage": interval_coverage,
        "masked_values_zero": bool(np.all(values[~observed] == 0.0)),
        "onset_support": onset_support,
        "onset_support_thresholds": {
            "absolute_min": STAGE_ABSOLUTE_COVERAGE_MIN,
            "relative_to_whole_min": STAGE_RELATIVE_TO_WHOLE_MIN,
        },
        "service_row_count": len(index),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="artifacts/p1/manifests")
    parser.add_argument("--inclusion-root", default="artifacts/p1/inclusion/gaia")
    parser.add_argument("--split-root", default="artifacts/p1/splits")
    parser.add_argument(
        "--source-snapshot", default="artifacts/p1/source_snapshots/re2ob"
    )
    parser.add_argument(
        "--telemetry-diagnostics", default="artifacts/p1/telemetry_diagnostics.json"
    )
    parser.add_argument("--output-root", default="artifacts/p2/event_features")
    parser.add_argument(
        "--checkpoint-root", default="artifacts/p2/event_feature_checkpoints"
    )
    parser.add_argument("--summary", default="artifacts/p2/event_feature_summary.json")
    parser.add_argument("--audit", default="artifacts/p2/event_feature_audit.json")
    parser.add_argument("--chunk-rows", type=int, default=200000)
    parser.add_argument("--chunk-progress-every", type=int, default=5)
    parser.add_argument("--feature-progress-every", type=int, default=2000)
    parser.add_argument("--re2-progress-every", type=int, default=1)
    args = parser.parse_args()
    if min(
        args.chunk_rows,
        args.chunk_progress_every,
        args.feature_progress_every,
        args.re2_progress_every,
    ) <= 0:
        raise ValueError("progress and chunk arguments must be positive")

    manifest_root = Path(args.manifest_root)
    inclusion_root = Path(args.inclusion_root)
    split_root = Path(args.split_root)
    source_snapshot = Path(args.source_snapshot)
    telemetry_path = Path(args.telemetry_diagnostics)
    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    required = (
        manifest_root / "gaia" / "manifest.json",
        manifest_root / "re2ob" / "manifest.json",
        inclusion_root / "manifest.json",
        split_root / "gaia" / "assignments.jsonl",
        split_root / "gaia" / "split_manifest.json",
        split_root / "re2ob" / "assignments.jsonl",
        split_root / "re2ob" / "split_manifest.json",
        source_snapshot / "manifest.json",
        telemetry_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required source bindings are missing: {}".format(missing))

    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry_sha = _sha256(telemetry_path)
    gaia_inputs, _ = read_manifest_cases(str(manifest_root / "gaia"))
    main_ids = {
        row["case_id"] for row in _read_jsonl(inclusion_root / "main_cohort.jsonl")
    }
    gaia_inputs = tuple(
        sorted(
            (case_input for case_input in gaia_inputs if case_input.case_id in main_ids),
            key=lambda row: row.case_id,
        )
    )
    if len(gaia_inputs) != len(main_ids):
        raise ValueError("GAIA main cohort does not match prediction inputs")
    gaia_services = gaia_inputs[0].services
    if any(case_input.services != gaia_services for case_input in gaia_inputs):
        raise ValueError("GAIA candidate service set is not fixed")
    gaia_source = _read_jsonl(manifest_root / "gaia" / "sources.jsonl")[0]
    gaia_common_binding = {
        "dataset_manifest_sha256": _sha256(manifest_root / "gaia" / "manifest.json"),
        "inclusion_manifest_sha256": _sha256(inclusion_root / "manifest.json"),
        "split_assignment_sha256": _sha256(split_root / "gaia" / "assignments.jsonl"),
        "split_manifest_sha256": _sha256(split_root / "gaia" / "split_manifest.json"),
        "telemetry_diagnostics_sha256": telemetry_sha,
    }
    anchors = np.asarray(
        [int(case_input.anchor_time) for case_input in gaia_inputs], dtype=np.int64
    )
    minimum_ms = int(anchors.min() - PRE_MS)
    maximum_ms = int(anchors.max() + POST_MS)
    gaia_checkpoints = {"log": [], "trace": []}
    gaia_resumed = {"log": 0, "trace": 0}
    for modality in ("log", "trace"):
        directory_key = "logs_directory" if modality == "log" else "traces_directory"
        prefix = "business_table" if modality == "log" else "trace_table"
        layout = telemetry["gaia"]["modalities"][
            "logs" if modality == "log" else "traces"
        ]["inventory"]
        for service in gaia_services:
            path = Path(gaia_source[directory_key]) / "{}_{}_2021-07.csv".format(
                prefix, service
            )
            binding = {
                **gaia_common_binding,
                **_file_binding(path),
                "raw_layout_sha256": layout["layout_sha256"],
            }
            values, observed, manifest, resumed = _checkpoint_or_extract_gaia(
                gaia_inputs,
                service,
                modality,
                path,
                checkpoint_root,
                binding,
                minimum_ms,
                maximum_ms,
                args.chunk_rows,
                args.chunk_progress_every,
                args.feature_progress_every,
            )
            pairs = tuple((case_input.case_id, service) for case_input in gaia_inputs)
            gaia_checkpoints[modality].append((pairs, values, observed, manifest))
            gaia_resumed[modality] += int(resumed)
            print(
                "[p2-event] GAIA {} checkpoint {}/{}".format(
                    modality, len(gaia_checkpoints[modality]), len(gaia_services)
                ),
                flush=True,
            )

    re2_inputs, _ = read_manifest_cases(str(manifest_root / "re2ob"))
    re2_inputs = tuple(sorted(re2_inputs, key=lambda row: row.case_id))
    re2_source_by_case = {
        row["case_id"]: row
        for row in _read_jsonl(manifest_root / "re2ob" / "sources.jsonl")
    }
    re2_common_binding = {
        "dataset_manifest_sha256": _sha256(manifest_root / "re2ob" / "manifest.json"),
        "source_snapshot_manifest_sha256": _sha256(source_snapshot / "manifest.json"),
        "split_assignment_sha256": _sha256(split_root / "re2ob" / "assignments.jsonl"),
        "split_manifest_sha256": _sha256(split_root / "re2ob" / "split_manifest.json"),
        "telemetry_diagnostics_sha256": telemetry_sha,
    }
    re2_checkpoints = {"log": [], "trace": []}
    re2_resumed = {"log": 0, "trace": 0}
    for index, case_input in enumerate(re2_inputs, start=1):
        source = re2_source_by_case[case_input.case_id]
        for modality, source_key in (("log", "logs_path"), ("trace", "traces_path")):
            binding = {
                **re2_common_binding,
                **_file_binding(Path(source[source_key])),
                "relative_directory": source["relative_directory"],
            }
            values, observed, manifest, resumed = _checkpoint_or_extract_re2(
                case_input,
                source,
                modality,
                checkpoint_root,
                binding,
            )
            pairs = tuple(
                (case_input.case_id, service) for service in case_input.services
            )
            re2_checkpoints[modality].append((pairs, values, observed, manifest))
            re2_resumed[modality] += int(resumed)
        if index % args.re2_progress_every == 0 or index == len(re2_inputs):
            print(
                "[p2-event] RE2 cases {}/{}".format(index, len(re2_inputs)),
                flush=True,
            )

    summaries = {}
    bundle_paths = {}
    for dataset_key, dataset_name, inputs, checkpoints, bindings in (
        (
            "gaia_main",
            "GAIA-MicroSS-2021-07/main",
            gaia_inputs,
            gaia_checkpoints,
            gaia_common_binding,
        ),
        ("re2ob", "RCAEval-RE2-OB", re2_inputs, re2_checkpoints, re2_common_binding),
    ):
        summaries[dataset_key] = {}
        bundle_paths[dataset_key] = {}
        for modality in ("log", "trace"):
            extractor = LOG_EXTRACTOR if modality == "log" else TRACE_EXTRACTOR
            output = output_root / dataset_key / extractor
            bundle_paths[dataset_key][modality] = output
            summaries[dataset_key][modality] = _write_dataset_bundle(
                output,
                dataset_name,
                inputs,
                modality,
                checkpoints[modality],
                bindings,
            )

    audit = {
        "gaia_main": {
            modality: _coverage_audit(
                bundle_paths["gaia_main"][modality], gaia_inputs, modality
            )
            for modality in ("log", "trace")
        },
        "re2ob": {
            modality: _coverage_audit(
                bundle_paths["re2ob"][modality], re2_inputs, modality
            )
            for modality in ("log", "trace")
        },
    }
    audit["gates"] = {
        "all_bundles_verified": True,
        "all_values_finite": all(
            audit[dataset][modality]["all_values_finite"]
            for dataset in ("gaia_main", "re2ob")
            for modality in ("log", "trace")
        ),
        "masked_values_zero": all(
            audit[dataset][modality]["masked_values_zero"]
            for dataset in ("gaia_main", "re2ob")
            for modality in ("log", "trace")
        ),
        "labels_read_by_extractor": False,
    }
    summary = {
        "extractors": {"log": LOG_EXTRACTOR, "trace": TRACE_EXTRACTOR},
        "gaia_main": summaries["gaia_main"],
        "re2ob": summaries["re2ob"],
    }
    _write_json(Path(args.summary), summary)
    _write_json(Path(args.audit), audit)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "audit_sha256": _sha256(Path(args.audit)),
                "gaia_resumed": gaia_resumed,
                "re2_resumed": re2_resumed,
                "summary_sha256": _sha256(Path(args.summary)),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
