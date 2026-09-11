#!/usr/bin/env python3
"""Run the P5-I1 validation-threshold event trigger and GT matching."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.event_detection import load_prediction_table, run_event_detection
from src.e2e.protocol import (
    GAIA_SERVICES,
    load_config,
    load_registry,
    sha256_file,
    temporal_blocks,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="evaluate", choices=("evaluate", "smoke"),
        help="evaluate artifacts or run a self-contained protocol smoke fixture",
    )
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v1.yaml")
    parser.add_argument("--artifact-root", default="artifacts/p5/i1")
    parser.add_argument(
        "--validation-predictions", default=None,
        help="timestamped validation CSV; defaults to artifact-root/ad_validation_predictions.csv",
    )
    parser.add_argument(
        "--test-predictions", default=None,
        help="timestamped test CSV; defaults to artifact-root/ad_test_predictions.csv",
    )
    parser.add_argument(
        "--registry", default=None,
        help="supported injection registry; defaults to the path bound by config",
    )
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def _smoke_fixture():
    """Small fixture exercising episode merging, tie-breaking, and misses."""

    timestamps = np.arange(12, dtype=np.int64) * 30_000
    validation_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    for split, rows in (("validation", validation_rows), ("test", test_rows)):
        for timestamp_index, timestamp in enumerate(timestamps):
            for service_index, service in enumerate(GAIA_SERVICES):
                score = 0.05
                if timestamp_index in (2, 3):
                    score = 0.90 if service_index == 0 else 0.10
                if timestamp_index == 7 and service_index == 1:
                    score = 0.80
                rows.append({
                    "split": split,
                    "sample_index": timestamp_index,
                    "window_start_time": int(timestamp - 270_000),
                    "window_end_time": int(timestamp + 30_000),
                    "prediction_timestamp": int(timestamp),
                    "service": service,
                    "service_registry_index": service_index,
                    "anomaly_score": score,
                    "node_label": 0,
                    "binary_prediction": int(score >= 0.5),
                })
    registry = pd.DataFrame([
        {
            "case_id": "smoke-val",
            "source_index": 0,
            "service": "dbservice1",
            "fault_type": "login failure",
            "start_ms": 60_000,
            "end_ms": 90_000,
            "split": "validation",
        },
        {
            "case_id": "smoke-test",
            "source_index": 1,
            "service": "dbservice2",
            "fault_type": "memory_anomalies",
            "start_ms": 210_000,
            "end_ms": 240_000,
            "split": "test",
        },
        {
            "case_id": "smoke-miss",
            "source_index": 2,
            "service": "webservice1",
            "fault_type": "cpu_anomalies",
            "start_ms": 300_000,
            "end_ms": 330_000,
            "split": "test",
        },
    ])
    return pd.DataFrame(validation_rows), pd.DataFrame(test_rows), registry


def _write_outputs(result, artifact_root: Path, metadata: Dict[str, object]):
    artifact_root.mkdir(parents=True, exist_ok=True)
    result["event_predictions"].to_csv(artifact_root / "ad_event_predictions.csv", index=False)
    result["matching"].to_csv(artifact_root / "event_matching.csv", index=False)
    event_path = artifact_root / "ad_event_predictions.csv"
    matching_path = artifact_root / "event_matching.csv"
    selection = result["threshold_selection"]
    metrics = {
        "schema_version": "p5_i1_event_detection_metrics_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "threshold_selection": {
            "threshold": float(selection.threshold),
            "candidate_count": int(selection.candidate_count),
            "tie_break": selection.tie_break,
            "selection_source": "validation only",
            "validation_metrics": selection.validation_metrics,
        },
        "validation_metrics": result["validation_metrics"],
        "test_metrics": result["test_metrics"],
        "artifact_files": {
            "event_predictions": {
                "path": str(event_path.resolve()),
                "sha256": sha256_file(event_path),
                "rows": int(len(result["event_predictions"])),
            },
            "event_matching": {
                "path": str(matching_path.resolve()),
                "sha256": sha256_file(matching_path),
                "rows": int(len(result["matching"])),
            },
        },
    }
    write_json(artifact_root / "event_detection_metrics.json", metrics)
    return metrics


def evaluate(config, artifact_root: Path, validation_path: Path, test_path: Path, registry_path: Path):
    validation = load_prediction_table(validation_path)
    test = load_prediction_table(test_path)
    registry = load_registry(config, PROJECT_ROOT) if registry_path is None else pd.read_csv(registry_path)
    result = run_event_detection(
        validation,
        test,
        registry,
        temporal_blocks(config),
        grid_seconds=int(config["ad"]["grid_seconds"]),
        tolerance_seconds=int(config["event_trigger"]["matching_tolerance_seconds"]),
    )
    metrics = _write_outputs(result, artifact_root, {
        "git_commit": git_head(),
        "config_path": str((PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml").resolve()),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "validation_prediction_path": str(validation_path.resolve()),
        "validation_prediction_sha256": sha256_file(validation_path),
        "test_prediction_path": str(test_path.resolve()),
        "test_prediction_sha256": sha256_file(test_path),
        "registry_path": str(registry_path.resolve()) if registry_path else str(
            (PROJECT_ROOT / str(config["event_registry"]["path"])).resolve()
        ),
        "registry_event_count": int(len(registry)),
        "test_not_used_for_threshold_selection": True,
    })
    return metrics


def smoke(artifact_root: Path):
    validation, test, registry = _smoke_fixture()
    result = run_event_detection(
        validation,
        test,
        registry,
        (),
        grid_seconds=30,
        tolerance_seconds=60,
    )
    metrics = _write_outputs(result, artifact_root, {
        "git_commit": git_head(),
        "fixture": True,
        "test_not_used_for_threshold_selection": True,
    })
    metrics["status"] = "PASS"
    write_json(artifact_root / "event_smoke_summary.json", metrics)
    return metrics


def main():
    args = parse_args()
    config = load_config((PROJECT_ROOT / args.config).resolve())
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    if args.action == "smoke":
        result = smoke(artifact_root)
    else:
        validation_path = Path(args.validation_predictions or (artifact_root / "ad_validation_predictions.csv"))
        test_path = Path(args.test_predictions or (artifact_root / "ad_test_predictions.csv"))
        registry_path = Path(args.registry).resolve() if args.registry else None
        result = evaluate(config, artifact_root, validation_path.resolve(), test_path.resolve(), registry_path)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
