#!/usr/bin/env python3
"""Run V3 Train threshold selection and causal Train/Test event evaluation."""

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
from src.e2e.protocol import GAIA_SERVICES, load_config, load_registry, sha256_file, temporal_blocks, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="evaluate", choices=("evaluate", "smoke"))
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v3.json")
    parser.add_argument("--artifact-root", default="artifacts/p5/v3/events")
    parser.add_argument("--train-predictions", default=None)
    parser.add_argument("--test-predictions", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--start-method", choices=("spawn", "forkserver"), default="spawn")
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()


def _smoke_fixture():
    timestamps = np.arange(12, dtype=np.int64) * 30_000 + 30_000
    train_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    for split, rows in (("train", train_rows), ("test", test_rows)):
        for timestamp_index, timestamp in enumerate(timestamps):
            for service_index, service in enumerate(GAIA_SERVICES):
                score = 0.05
                if timestamp_index in (2, 3):
                    score = 0.90 if service_index == 0 else 0.10
                if timestamp_index == 7 and service_index == 1:
                    score = 0.80
                rows.append({
                    "split": split, "sample_index": timestamp_index,
                    "window_start_time": int(timestamp - 300_000),
                    "window_end_time": int(timestamp),
                    "target_bin_start": int(timestamp - 30_000),
                    "target_bin_end": int(timestamp),
                    "prediction_available_time": int(timestamp),
                    "prediction_timestamp": int(timestamp),
                    "service": service, "service_registry_index": service_index,
                    "anomaly_score": score, "node_label": 0,
                    "binary_prediction": int(score >= 0.5),
                })
    registry = pd.DataFrame([
        {"case_id": "smoke-train", "source_index": 0, "service": "dbservice1", "fault_type": "login_failure", "start_ms": 90_000, "end_ms": 120_000, "split": "train"},
        {"case_id": "smoke-test", "source_index": 1, "service": "dbservice2", "fault_type": "memory_anomalies", "start_ms": 240_000, "end_ms": 270_000, "split": "test"},
        {"case_id": "smoke-miss", "source_index": 2, "service": "webservice1", "fault_type": "cpu_anomalies", "start_ms": 600_000, "end_ms": 630_000, "split": "test"},
    ])
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows), registry


def _write_outputs(result, artifact_root: Path, metadata: Dict[str, object]):
    artifact_root.mkdir(parents=True, exist_ok=True)
    event_path = artifact_root / "ad_event_predictions.csv"
    matching_path = artifact_root / "event_matching.csv"
    result["event_predictions"].to_csv(event_path, index=False, lineterminator="\n")
    result["matching"].to_csv(matching_path, index=False, lineterminator="\n")
    selection = result["threshold_selection"]
    metrics = {
        "schema_version": "p5_v3_event_detection_metrics_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), **metadata,
        "status": "FORMAL_FULL_DATA",
        "formal_result": True,
        "threshold_selection": {
            "threshold": float(selection.threshold), "candidate_count": int(selection.candidate_count),
            "tie_break": selection.tie_break, "selection_source": "Train only",
            "train_metrics": selection.train_metrics,
        },
        "train_metrics": result["train_metrics"], "test_metrics": result["test_metrics"],
        "matching_semantics": "causal_max_cardinality_minimum_delay",
        "prediction_time": "t_hat=prediction_available_time=target_bin_end",
        "artifact_files": {
            "event_predictions": {"path": str(event_path.resolve()), "sha256": sha256_file(event_path), "rows": int(len(result["event_predictions"]))},
            "event_matching": {"path": str(matching_path.resolve()), "sha256": sha256_file(matching_path), "rows": int(len(result["matching"]))},
        },
    }
    write_json(artifact_root / "event_detection_metrics.json", metrics)
    return metrics


def evaluate(
    config, artifact_root: Path, train_path: Path, test_path: Path,
    registry_path: Path, workers=1, start_method="spawn", config_path: Path = None,
):
    train = load_prediction_table(train_path)
    test = load_prediction_table(test_path)
    if str(config.get("schema_version", "")).startswith("p5_v3_"):
        if set(train["split"].astype(str)) != {"train"}:
            raise ValueError("V3 Train prediction artifact must contain only Train rows")
        if set(test["split"].astype(str)) != {"test"}:
            raise ValueError("V3 Test prediction artifact must contain only Test rows")
    registry = load_registry(config, PROJECT_ROOT) if registry_path is None else pd.read_csv(registry_path)
    result = run_event_detection(
        train, test, registry, temporal_blocks(config),
        grid_seconds=int(config["ad"]["grid_seconds"]),
        tolerance_seconds=int(config["event_trigger"]["matching_tolerance_seconds"]),
        threshold_workers=int(workers), threshold_start_method=str(start_method),
    )
    config_path = Path(
        config_path or (PROJECT_ROOT / "configs/e2e/gaia_p5_v3.json")
    ).resolve()
    return _write_outputs(result, artifact_root, {
        "git_commit": git_head(), "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "train_prediction_path": str(train_path.resolve()), "train_prediction_sha256": sha256_file(train_path),
        "test_prediction_path": str(test_path.resolve()), "test_prediction_sha256": sha256_file(test_path),
        "registry_path": str(registry_path.resolve()) if registry_path else str((PROJECT_ROOT / str(config["event_registry"]["path"])).resolve()),
        "registry_sha256": sha256_file(registry_path) if registry_path else sha256_file((PROJECT_ROOT / str(config["event_registry"]["path"])).resolve()),
        "test_not_used_for_threshold_selection": True,
        "threshold_execution": {"workers": int(workers), "start_method": str(start_method), "selection_semantics": "exact unique Train-score sweep"},
    })


def smoke(config, artifact_root: Path):
    train, test, registry = _smoke_fixture()
    result = run_event_detection(train, test, registry, (), grid_seconds=30, tolerance_seconds=60)
    selection = result["threshold_selection"]
    summary = {
        "schema_version": "p5_v3_event_detection_smoke_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS", "formal_result": False, "fixture": "synthetic Train/Test only",
        "git_commit": git_head(), "random_seed": int(config["random_seed"]),
        "test_not_used_for_threshold_selection": True,
        "matching_semantics": "causal_max_cardinality_minimum_delay",
        "threshold_selection": {"threshold": float(selection.threshold), "candidate_count": int(selection.candidate_count), "tie_break": selection.tie_break, "selection_source": "Train only"},
        "train_metrics": result["train_metrics"], "test_metrics": result["test_metrics"],
        "event_prediction_rows": int(len(result["event_predictions"])), "matching_rows": int(len(result["matching"])),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "event_smoke_summary.json", summary)
    return summary


def main():
    args = parse_args()
    config = load_config((PROJECT_ROOT / args.config).resolve())
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    if args.action == "smoke":
        result = smoke(config, artifact_root)
    else:
        train_path = Path(args.train_predictions or (PROJECT_ROOT / "artifacts/p5/v3/ad/ad_train_predictions.csv")).resolve()
        test_path = Path(args.test_predictions or (PROJECT_ROOT / "artifacts/p5/v3/ad/ad_test_predictions.csv")).resolve()
        registry_path = Path(args.registry).resolve() if args.registry else None
        result = evaluate(
            config, artifact_root, train_path, test_path, registry_path,
            args.workers, args.start_method, (PROJECT_ROOT / args.config).resolve(),
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
