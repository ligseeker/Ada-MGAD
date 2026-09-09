#!/usr/bin/env python3
"""Bind historical full-scan telemetry evidence to the migrated GAIA bytes."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p5.common import layout_digest, percentile_summary, sha256_file, write_json  # noqa: E402
from util.GAIA.pre_GAIA import _parse_metric_filename, _target_services_for_metric  # noqa: E402


SCHEMA_VERSION = "p5_g0r2_raw_telemetry_timing_v1"


def _header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _trace_status_and_duration(trace_paths, chunk_rows: int) -> Mapping[str, object]:
    status = Counter()
    rows = invalid_status = 0
    duration_count = duration_invalid = duration_negative = duration_zero = 0
    duration_sum = 0.0
    duration_min = None
    duration_max = None
    duration_sample = []
    global_offset = 0
    sample_stride = 257
    for file_index, path in enumerate(trace_paths, 1):
        for chunk in pd.read_csv(
            path,
            usecols=["start_time", "end_time", "status_code"],
            chunksize=chunk_rows,
            low_memory=False,
        ):
            numeric_status = pd.to_numeric(chunk["status_code"], errors="coerce")
            invalid_status += int(numeric_status.isna().sum())
            status.update(str(int(value)) for value in numeric_status.dropna().to_numpy())
            start = pd.to_datetime(chunk["start_time"], errors="coerce")
            end = pd.to_datetime(chunk["end_time"], errors="coerce")
            duration = (end - start).dt.total_seconds().to_numpy(dtype=float)
            finite = duration[np.isfinite(duration)]
            duration_invalid += len(duration) - len(finite)
            duration_count += len(finite)
            duration_negative += int(np.count_nonzero(finite < 0))
            duration_zero += int(np.count_nonzero(finite == 0))
            duration_sum += float(np.sum(finite))
            if finite.size:
                local_min = float(np.min(finite))
                local_max = float(np.max(finite))
                duration_min = local_min if duration_min is None else min(duration_min, local_min)
                duration_max = local_max if duration_max is None else max(duration_max, local_max)
            positions = np.arange(global_offset, global_offset + len(duration), dtype=np.int64)
            sampled = duration[(positions % sample_stride) == 0]
            duration_sample.extend(float(value) for value in sampled if np.isfinite(value))
            global_offset += len(duration)
            rows += len(chunk)
        print("trace {}/{} {} rows={}".format(file_index, len(trace_paths), path.name, rows), flush=True)
    status_distribution = dict(sorted(status.items(), key=lambda item: int(item[0])))
    not_200 = sum(count for code, count in status.items() if int(code) != 200)
    ge_500 = sum(count for code, count in status.items() if int(code) >= 500)
    sample_summary = dict(percentile_summary(duration_sample))
    sample_summary["sample_stride"] = sample_stride
    sample_summary["sampling"] = "global source-order every 257th row"
    return {
        "rows": rows,
        "invalid_status_rows": invalid_status,
        "status_code_distribution": status_distribution,
        "official_not_200_count": not_200,
        "official_not_200_ratio": not_200 / rows,
        "historical_ge_500_count": ge_500,
        "historical_ge_500_ratio": ge_500 / rows,
        "rules_equivalent": not_200 == ge_500,
        "rows_missed_by_ge_500": not_200 - ge_500,
        "duration_seconds": {
            "valid_rows": duration_count,
            "invalid_rows": duration_invalid,
            "negative_rows": duration_negative,
            "zero_rows": duration_zero,
            "mean": duration_sum / duration_count if duration_count else None,
            "min": duration_min,
            "max": duration_max,
            "quantile_sample": sample_summary,
        },
    }


def build(gaia_root: Path, historical_path: Path, scan_traces: bool, chunk_rows: int) -> Mapping[str, object]:
    import json

    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    old_gaia = historical["gaia"]
    old_modalities = old_gaia["modalities"]
    run_table = gaia_root / "run" / "run" / "run" / "run_table_2021-07.csv"
    metric_dir = gaia_root / "metric" / "metric_split" / "metric"
    log_dir = gaia_root / "business" / "business_split" / "business"
    trace_dir = gaia_root / "trace" / "trace_split" / "trace"
    metric_paths = sorted(metric_dir.glob("*.csv"))
    log_paths = sorted(log_dir.glob("*.csv"))
    trace_paths = sorted(trace_dir.glob("*.csv"))
    candidate_metric_paths = []
    for path in metric_paths:
        info = _parse_metric_filename(path.name)
        if info is not None and _target_services_for_metric(info):
            candidate_metric_paths.append(path)

    current_inventory = {
        "metrics": layout_digest(metric_dir, metric_paths),
        "logs": layout_digest(log_dir, log_paths),
        "traces": layout_digest(trace_dir, trace_paths),
    }
    inventory_match = {
        modality: current_inventory[modality] == old_modalities[modality]["inventory"]
        for modality in ("metrics", "logs", "traces")
    }
    run_hash = sha256_file(run_table)
    run_hash_match = run_hash == old_gaia["critical_annotation_sha256"]["run_table"]
    binding_valid = all(inventory_match.values()) and run_hash_match
    if not binding_valid:
        raise ValueError("migrated GAIA source binding differs from historical full-scan inventory")

    metric_grid = old_modalities["metrics"]["scheduled_timestamp_quality"]
    trace_scan = _trace_status_and_duration(trace_paths, chunk_rows) if scan_traces else {
        "status": "NOT SCANNED",
        "reproduction_command_required": "rerun with --scan-traces",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "REVALIDATED FACT",
        "source_binding": {
            "gaia_root": str(gaia_root.resolve()),
            "run_table_sha256": run_hash,
            "historical_run_table_sha256": old_gaia["critical_annotation_sha256"]["run_table"],
            "run_table_match": run_hash_match,
            "current_inventory": current_inventory,
            "historical_inventory": {
                modality: old_modalities[modality]["inventory"]
                for modality in ("metrics", "logs", "traces")
            },
            "inventory_match": inventory_match,
            "binding_valid": binding_valid,
            "historical_artifact": str(historical_path.resolve()),
            "historical_artifact_sha256": sha256_file(historical_path),
            "reuse_rule": "full-scan facts are upgraded only because run-table hash and all size-bound layout digests match",
        },
        "metrics": {
            "header": _header(metric_paths[0]),
            "all_files": len(metric_paths),
            "candidate_files": len(candidate_metric_paths),
            "candidate_source_bytes": sum(path.stat().st_size for path in candidate_metric_paths),
            "rows": old_modalities["metrics"]["rows"],
            "valid_timestamp_rows": old_modalities["metrics"]["timestamp"]["valid_timestamp_rows"],
            "invalid_timestamp_rows": old_modalities["metrics"]["timestamp"]["invalid_timestamp_rows"],
            "positive_delta_median_ms": old_modalities["metrics"]["timestamp"]["positive_delta_ms_approx"]["median"],
            "positive_delta_p95_ms": old_modalities["metrics"]["timestamp"]["positive_delta_ms_approx"]["p95"],
            "scheduled_nominal_interval_median_ms": metric_grid["nominal_interval_ms"]["median"],
            "scheduled_nominal_interval_p95_ms": metric_grid["nominal_interval_ms"]["p95"],
            "duplicate_timestamp_rows": metric_grid["duplicate_timestamp_rows"],
            "inferred_sampling_gap_count": metric_grid["missing_timestamps"],
            "inferred_sampling_gap_ratio": metric_grid["missing_timestamp_ratio"],
            "gap_semantics": metric_grid["semantics"],
        },
        "logs": {
            "header": _header(log_paths[0]),
            "files": len(log_paths),
            "rows": old_modalities["logs"]["rows"],
            "timestamp_source": old_modalities["logs"]["timestamp_source"],
            "timestamp_resolution": "millisecond message prefix YYYY-MM-DD HH:MM:SS,mmm",
            "declared_datetime_rows_with_clock": old_modalities["logs"]["declared_datetime_rows_with_clock"],
            "valid_timestamp_rows": old_modalities["logs"]["timestamp"]["valid_timestamp_rows"],
            "invalid_timestamp_rows": old_modalities["logs"]["timestamp"]["invalid_timestamp_rows"],
        },
        "traces": {
            "header": _header(trace_paths[0]),
            "files": len(trace_paths),
            "historical_rows": old_modalities["traces"]["rows"],
            "timestamp_field": "timestamp is second-resolution display time; start_time/end_time carry microseconds",
            "duration_rule": "(end_time - start_time).total_seconds()",
            "historical_negative_duration_rows": old_modalities["traces"]["negative_duration_rows"],
            "required_field_missing": old_modalities["traces"]["required_field_missing"],
            "parent_id_missing_rows": old_modalities["traces"]["parent_id_missing_rows"],
            "span_parent_fields": ["trace_id", "span_id", "parent_id", "service_name"],
            "status_official_semantics": "200 normal, all other codes anomalous",
            "targeted_full_status_and_duration_scan": trace_scan,
        },
        "limitations": [
            "Historical timestamp counts are not blindly copied: they are reused only after exact run-table and layout binding.",
            "Layout digests bind relative paths and byte sizes, not every telemetry file byte; run-table uses a full SHA-256.",
            "Trace duration quantiles use a deterministic every-257th-row sample; row counts, invalid/negative counts, min/max, and mean are full-scan.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaia-root", required=True, type=Path)
    parser.add_argument("--historical-telemetry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scan-traces", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    args = parser.parse_args()
    report = build(args.gaia_root, args.historical_telemetry, args.scan_traces, args.chunk_rows)
    write_json(args.output, report)
    print(
        "binding_valid={}; trace_rules_equivalent={}".format(
            report["source_binding"]["binding_valid"],
            report["traces"]["targeted_full_status_and_duration_scan"].get("rules_equivalent"),
        )
    )


if __name__ == "__main__":
    main()
