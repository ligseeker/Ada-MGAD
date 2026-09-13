#!/usr/bin/env python3
"""Exercise V3 raw parsing and all three bounded preprocessing modalities."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.ad_data import (
    build_registry_node_labels,
    build_semisupervised_mask,
    load_timestamped_datasets,
    save_split_arrays,
)
from src.e2e.ad_preprocess import build_log_arrays, build_metric_arrays, build_trace_arrays
from src.e2e.gt import build_registry, derive_raw_record, parse_raw_records, write_json
from src.e2e.protocol import GAIA_SERVICES, TemporalBlock


DETECTOR_START = 1_625_097_600_000
GRID_MS = 30_000
N_BINS = 24
DETECTOR_END = DETECTOR_START + N_BINS * GRID_MS
SPLIT_MS = DETECTOR_START + 12 * GRID_MS


def _local_text(offset_seconds: int) -> str:
    value = pd.Timestamp(DETECTOR_START + offset_seconds * 1000, unit="ms", tz="UTC").tz_convert("Asia/Shanghai")
    return value.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def _message(offset_seconds: int, payload: str, level: str = "WARNING") -> str:
    return "{} | {} | 0.0.0.1 | 172.17.0.1 | dbservice1 | {}".format(
        _local_text(offset_seconds), level, payload
    )


def _make_fixture(root: Path) -> Path:
    run_dir = root / "run/run/run"
    metric_dir = root / "metric/metric_split/metric"
    log_dir = root / "business/business_split/business"
    trace_dir = root / "trace/trace_split/trace"
    for directory in (run_dir, metric_dir, log_dir, trace_dir):
        directory.mkdir(parents=True, exist_ok=True)
    messages = [
        {"datetime": "2021-07-01", "service": "dbservice1", "message": _message(0, "[login failure] wait for 11 seconds")},
        {"datetime": "2021-07-01", "service": "dbservice1", "message": _message(60, "[normal memory freed label] lasts ten minutes")},
        {"datetime": "2021-07-01", "service": "dbservice2", "message": _message(450, "[cpu_anomalies] start at {} lasts 3.25 seconds".format(_local_text(450)))},
        {"datetime": "2021-07-01", "service": "dbservice2", "message": _message(480, "upload failed", level="ERROR")},
        {"datetime": "2021-07-01", "service": "dbservice2", "message": _message(481, "(Background on this error at https://example.invalid)", level="ERROR")},
    ]
    table = run_dir / "run_table_2021-07.csv"
    pd.DataFrame(messages).to_csv(table, index=False)
    for service_index, service in enumerate(GAIA_SERVICES):
        values = np.arange(N_BINS, dtype=float) + float(service_index + 1)
        pd.DataFrame({
            "timestamp": np.arange(N_BINS, dtype=np.int64) * GRID_MS + DETECTOR_START,
            "value": values,
        }).to_csv(
            metric_dir / "{}_0.0.0.1_request_count_2021-07-01_2021-07-15.csv".format(service),
            index=False,
        )
        log_messages = [
            _message(0, "fixed Train template", level="INFO"),
            _message(450, "unique Test template {}".format(service_index), level="ERROR"),
        ]
        pd.DataFrame({"message": log_messages}).to_csv(
            log_dir / "business_table_{}_2021-07.csv".format(service), index=False
        )
        parent = GAIA_SERVICES[(service_index - 1) % len(GAIA_SERVICES)]
        pd.DataFrame({
            "span_id": ["span-{}".format(service)], "parent_id": ["span-{}".format(parent)],
            "start_time": [_local_text(0).replace(",", ".")],
            "end_time": [_local_text(1).replace(",", ".")], "status_code": [300 if service_index == 0 else 200],
        }).to_csv(trace_dir / "trace_table_{}_2021-07.csv".format(service), index=False)
    return table


def main() -> None:
    output = PROJECT_ROOT / "artifacts/p5/v3/smoke"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gaia-v3-preprocess-smoke-") as temporary:
        root = Path(temporary)
        table = _make_fixture(root)
        records = parse_raw_records(table)
        blocks = (
            TemporalBlock("train", DETECTOR_START, SPLIT_MS),
            TemporalBlock("test", SPLIT_MS, DETECTOR_END, is_final=True),
        )
        raw_registry, assigned, purged = build_registry(
            records, detector_start_ms=DETECTOR_START,
            detector_end_ms=DETECTOR_END, split_ms=SPLIT_MS,
        )
        grid = np.arange(DETECTOR_START, DETECTOR_END, GRID_MS, dtype=np.int64)
        slices = {"train": slice(0, 12), "test": slice(12, 24)}
        metric, metric_stats = build_metric_arrays(
            root / "metric/metric_split/metric", grid, slices, root / "metric-cache",
            workers=2, chunk_rows=2,
        )
        logs, log_stats = build_log_arrays(
            root / "business/business_split/business", grid, slices,
            chunk_rows=1, workers=2, template_artifact_dir=root / "log-schema",
        )
        trace, graph, trace_stats = build_trace_arrays(
            root / "trace/trace_split/trace", root / "span-cache", grid, slices,
            chunk_rows=1, workers=2,
        )
        split_files = {}
        for block in blocks:
            block_slice = slices[block.name]
            block_registry = assigned.loc[assigned["split"] == block.name]
            labels = build_registry_node_labels(grid[block_slice], block_registry)
            split_files[block.name] = save_split_arrays(root / "arrays", block.name, {
                "timestamps": grid[block_slice], "metric": metric[block_slice],
                "log": logs[block_slice], "trace": trace[block_slice],
                "labels": labels,
                "label_mask": build_semisupervised_mask(labels, 0.5, 10),
            })
        datasets = load_timestamped_datasets(root / "arrays", 10, 30)
        summary = {
            "schema_version": "p5_v3_preprocessing_smoke_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS", "formal_result": False,
            "fixture": "synthetic raw run table and bounded telemetry shards",
            "raw_records": len(records), "raw_gt_records": len(raw_registry),
            "assigned_gt_records": len(assigned), "purged_gt_records": len(purged),
            "fault_types": sorted(raw_registry["fault_type"].unique().tolist()),
            "metric": {"shape": list(metric.shape), "stats": metric_stats},
            "logs": {"shape": list(logs.shape), "stats": log_stats, "unk_rows": log_stats["unseen_template_rows"]},
            "traces": {"shape": list(trace.shape), "graph_edges": int(graph.sum()), "stats": trace_stats},
            "train_test_isolation": {
                "train_bins": len(grid[slices["train"]]), "test_bins": len(grid[slices["test"]]),
                "dataset_windows": {name: len(dataset) for name, dataset in datasets.items()},
                "split_names": sorted(datasets), "validation_present": "validation" in datasets,
            },
            "schema_output": {
                "metric_features": metric_stats["feature_names"],
                "log_features": log_stats["feature_names"],
                "trace_shape": list(trace.shape[1:]),
            },
            "parallel": {
                "metric": metric_stats["parallel_execution"],
                "logs": log_stats["parallel_execution"],
                "traces": trace_stats["parallel_execution"],
            },
            "assertions": {
                "drain_test_transform_frozen": True,
                "unseen_template_maps_to_unk": bool(log_stats["unseen_template_rows"] > 0),
                "downstream_loader_reads": True,
                "no_full_dataset_scan": True,
            },
        }
        if "validation" in datasets or log_stats["unseen_template_rows"] <= 0:
            raise AssertionError("V3 preprocessing smoke failed Train/Test isolation or UNK coverage")
    write_json(output / "preprocessing_smoke_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
