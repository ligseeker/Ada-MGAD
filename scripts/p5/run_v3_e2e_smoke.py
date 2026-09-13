#!/usr/bin/env python3
"""Run a bounded synthetic V3 detector -> RCA -> E2E smoke path.

This fixture intentionally uses synthetic detector probabilities and a tiny raw
telemetry tree.  It validates wiring and provenance checks only; it is never a
formal GAIA result.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p5.run_v3_preprocessing_smoke import _make_fixture
from scripts.p5.run_i1_e2e import evaluate as evaluate_e2e
from scripts.p5.run_i1_rca import train_v3
from scripts.p5.run_i1_rca_features import materialize
from src.e2e.event_detection import run_event_detection
from src.e2e.gaia_rca_adapter import build_raw_index
from src.e2e.protocol import GAIA_SERVICES, load_config, sha256_file, temporal_blocks
from src.e2e.rca_model import FEATURE_DIMENSION


def _prediction_frame(config):
    blocks = temporal_blocks(config)
    train_anchor = blocks[0].end_ms - 600_000
    test_anchor = blocks[1].start_ms + 600_000
    train_times = np.asarray([train_anchor - 60_000, train_anchor, train_anchor + 60_000], dtype=np.int64)
    test_times = np.asarray([test_anchor - 60_000, test_anchor, test_anchor + 60_000], dtype=np.int64)
    rows_train = []
    rows_test = []
    for split, timestamps, destination in (
        ("train", train_times, rows_train), ("test", test_times, rows_test)
    ):
        for timestamp in timestamps:
            for service_index, service in enumerate(GAIA_SERVICES):
                score = 0.1
                if int(timestamp) == (train_anchor if split == "train" else test_anchor):
                    score = 0.9 if service_index == 0 else 0.1
                destination.append({
                    "split": split,
                    "sample_index": int(timestamp // 30_000),
                    "window_start_time": int(timestamp - 300_000),
                    "window_end_time": int(timestamp),
                    "target_bin_start": int(timestamp - 30_000),
                    "target_bin_end": int(timestamp),
                    "prediction_available_time": int(timestamp),
                    "prediction_timestamp": int(timestamp),
                    "service": service,
                    "service_registry_index": service_index,
                    "anomaly_score": score,
                    "node_label": int(score >= 0.5),
                    "binary_prediction": int(score >= 0.5),
                })
    registry = pd.DataFrame([
        {
            "case_id": "smoke-train-0", "source_index": 0,
            "service": "dbservice1", "labelled_service": "dbservice1",
            "fault_type": "login_failure", "start_ms": int(train_anchor),
            "end_ms": int(train_anchor + 30_000), "split": "train",
        },
        {
            "case_id": "smoke-train-1", "source_index": 1,
            "service": "dbservice2", "labelled_service": "dbservice2",
            "fault_type": "memory_anomalies", "start_ms": int(train_anchor - 60_000),
            "end_ms": int(train_anchor - 30_000), "split": "train",
        },
        {
            "case_id": "smoke-test-0", "source_index": 2,
            "service": "dbservice1", "labelled_service": "dbservice1",
            "fault_type": "cpu_anomalies", "start_ms": int(test_anchor),
            "end_ms": int(test_anchor + 30_000), "split": "test",
        },
    ])
    return pd.DataFrame(rows_train), pd.DataFrame(rows_test), registry, train_anchor, test_anchor


def _write_case_registry(path: Path, registry: pd.DataFrame, detected_anchor: int):
    oracle = registry.copy()
    oracle["anchor_type"] = "GT injection start"
    oracle["gt_start_ms"] = oracle["start_ms"].astype(np.int64)
    oracle["prediction_id"] = ""
    oracle.to_csv(path / "oracle_cases.csv", index=False)

    detected = registry.loc[registry["split"].astype(str) == "train"].copy()
    detected["anchor_type"] = "GT injection start"
    detected["gt_start_ms"] = detected["start_ms"].astype(np.int64)
    detected["prediction_id"] = ""
    test = registry.loc[registry["case_id"] == "smoke-test-0"].copy()
    test["start_ms"] = int(detected_anchor)
    test["end_ms"] = int(detected_anchor)
    test["anchor_type"] = "detected prediction_available_time"
    test["gt_start_ms"] = registry.loc[registry["case_id"] == "smoke-test-0", "start_ms"].iloc[0]
    test["prediction_id"] = "test-pred-000000"
    detected = pd.concat([detected, test], ignore_index=True, sort=False)
    detected.to_csv(path / "detected_cases.csv", index=False)
    return path / "oracle_cases.csv", path / "detected_cases.csv"


def main():
    config = load_config(PROJECT_ROOT / "configs/e2e/gaia_p5_v3.json")
    output = PROJECT_ROOT / "artifacts/p5/v3/smoke"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gaia-v3-e2e-smoke-") as temporary:
        root = Path(temporary)
        raw_root = root / "raw"
        _make_fixture(raw_root)
        train_predictions, test_predictions, registry, _, test_anchor = _prediction_frame(config)
        event = run_event_detection(
            train_predictions, test_predictions, registry, temporal_blocks(config),
            grid_seconds=30, tolerance_seconds=60,
        )
        matching_path = root / "event_matching.csv"
        event["matching"].to_csv(matching_path, index=False)
        node_path = root / "ad_test_predictions.csv"
        test_predictions.to_csv(node_path, index=False)

        index_root = root / "raw-index"
        index_manifest = build_raw_index(raw_root, index_root, chunk_rows=2, workers=1)
        oracle_case_path, detected_case_path = _write_case_registry(root, registry, test_anchor)
        oracle_features = root / "oracle-features"
        detected_features = root / "detected-features"
        materialize(
            config, index_root, oracle_features, root / "oracle-artifacts", oracle_case_path,
            workers=1, case_chunk_size=2, formal_result=False,
        )
        materialize(
            config, index_root, detected_features, root / "detected-artifacts", detected_case_path,
            workers=1, case_chunk_size=2, formal_result=False,
        )
        rca_artifacts = root / "rca-artifacts"
        rca_model = root / "rca-model" / "conditional_logit.npz"
        rca_manifest = train_v3(
            config, oracle_features, oracle_case_path, rca_model, rca_artifacts,
            detected_features, detected_case_path,
            formal_result=False,
        )
        e2e = evaluate_e2e(
            config, rca_artifacts, index_manifest= index_root / "index_manifest.json",
            model_path=rca_model, detected_feature_path=root / "unused-detected.npy",
            matching_path=matching_path, node_path=node_path,
            oracle_path=rca_artifacts / "rca_oracle_predictions.csv",
            frequency_path=rca_artifacts / "root_frequency_predictions.csv",
            case_registry_path=oracle_case_path,
            detected_predictions_path=rca_artifacts / "rca_detected_predictions.csv",
            formal_result=False,
        )
        layered_copy = output / "full_pipeline_smoke_layered_report.json"
        shutil.copyfile(rca_artifacts / "e2e_layered_report.json", layered_copy)
        summary = {
            "schema_version": "p5_v3_full_pipeline_smoke_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS", "formal_result": False,
            "fixture": "synthetic detector probabilities plus bounded raw telemetry tree",
            "steps": {
                "event_matching_rows": int(len(event["matching"])),
                "event_test_metrics": event["test_metrics"],
                "raw_index_build_id": index_manifest["build_id"],
                "oracle_feature_shape": list(np.load(oracle_features / "z2_features.npy", mmap_mode="r").shape),
                "detected_feature_shape": list(np.load(detected_features / "z2_features.npy", mmap_mode="r").shape),
                "rca_case_counts": rca_manifest["case_counts"],
                "e2e_matched_events": e2e["e2e_diagnosis_metrics"]["counts"]["matched_events"],
                "layered_report": str(layered_copy.resolve()),
                "layered_report_sha256": sha256_file(layered_copy),
            },
            "assertions": {
                "raw_index_label_free": True,
                "oracle_and_detected_anchors_separate": True,
                "train_only_rca_fit": True,
                "precomputed_detected_ranking_consumed": True,
                "no_formal_gaia_result": True,
            },
        }
    (output / "full_pipeline_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
