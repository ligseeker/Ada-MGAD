#!/usr/bin/env python3
"""Inventory P5-I1 source bindings and completed/pending artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.protocol import load_config, sha256_file, write_json


FORMAL_ARTIFACTS = (
    "protocol_manifest.json",
    "split_manifest.json",
    "event_registry_manifest.json",
    "ad_data_manifest.json",
    "ad_training_summary.json",
    "ad_validation_predictions.csv",
    "ad_test_predictions.csv",
    "ad_event_predictions.csv",
    "event_matching.csv",
    "event_detection_metrics.json",
    "rca_feature_health.json",
    "rca_feature_manifest.json",
    "rca_train_manifest.json",
    "rca_oracle_predictions.csv",
    "rca_detected_predictions.csv",
    "detector_only_predictions.csv",
    "root_frequency_predictions.csv",
    "rca_metrics.json",
    "e2e_diagnosis_metrics.json",
)
SMOKE_ARTIFACTS = (
    "ad_smoke_summary.json",
    "event_smoke_summary.json",
    "rca_feature_smoke_summary.json",
    "rca_training_smoke_summary.json",
    "e2e_smoke_summary.json",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v1.yaml")
    parser.add_argument("--artifact-root", default="artifacts/p5/i1")
    parser.add_argument("--pytest-result", default="not recorded")
    return parser.parse_args()


def _git(*arguments):
    return subprocess.check_output(["git"] + list(arguments), cwd=str(PROJECT_ROOT), text=True).strip()


def _record(path: Path, pending_allowed: bool):
    if not path.is_file():
        return {
            "path": str(path.resolve()),
            "status": "PENDING_MANUAL_FULL_GAIA_RUN" if pending_allowed else "MISSING",
        }
    return {
        "path": str(path.resolve()),
        "status": "COMPLETE",
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def main():
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    formal = {name: _record(artifact_root / name, True) for name in FORMAL_ARTIFACTS}
    smoke = {name: _record(artifact_root / name, False) for name in SMOKE_ARTIFACTS}
    pending = [name for name, record in formal.items() if record["status"] != "COMPLETE"]
    manifest = {
        "schema_version": "p5_i1_run_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "FORMAL_FULL_DATA_COMPLETE" if not pending
            else "IMPLEMENTATION_READY_FORMAL_FULL_DATA_PENDING_MANUAL_RUN"
        ),
        "result_scope": (
            "formal full GAIA" if not pending
            else "protocol manifests plus non-formal synthetic smoke validation only"
        ),
        "repository": {
            "branch": _git("branch", "--show-current"),
            "implementation_commit_before_manifest": _git("rev-parse", "HEAD"),
            "starting_commit": "cbe57512b6bd1ce772ab42d9b8ec0b292c3cb91b",
        },
        "source_bindings": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "supported_event_registry": {
                "events": int(config["event_registry"]["expected_events"]),
                "sha256": str(config["event_registry"]["sha256"]),
            },
            "run_table_sha256": str(config["run_table"]["sha256"]),
            "ada_rca_canonical_commit": str(config["source"]["ada_rca_commit"]),
            "random_seed": int(config["random_seed"]),
        },
        "validation": {
            "pytest": str(args.pytest_result),
            "smoke_artifacts_complete": all(record["status"] == "COMPLETE" for record in smoke.values()),
            "formal_metrics_available": not pending,
        },
        "formal_artifacts": formal,
        "pending_formal_artifacts": pending,
        "smoke_artifacts": smoke,
        "manual_execution": {
            "one_command": "python scripts/p5/run_i1_pipeline.py full --gpu true",
            "note": "run the documented resumable commands when the 30 GB GAIA source is on responsive local storage",
        },
    }
    write_json(artifact_root / "run_manifest.json", manifest)
    print(manifest["status"])


if __name__ == "__main__":
    main()
