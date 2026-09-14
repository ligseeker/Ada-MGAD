#!/usr/bin/env python3
"""Finalize the machine-readable run manifest while preserving RCA paths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.protocol import load_config, sha256_file, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v3.json")
    parser.add_argument("--artifact-root", default="artifacts/p5/v3")
    parser.add_argument("--protocol-root", default=None)
    parser.add_argument("--ad-artifact-root", default=None)
    parser.add_argument("--ad-checkpoint-root", default=None)
    parser.add_argument("--event-artifact-root", default=None)
    parser.add_argument("--pytest-result", default="not recorded")
    return parser.parse_args()


def git(*arguments):
    return subprocess.check_output(
        ["git"] + list(arguments), cwd=str(PROJECT_ROOT), text=True
    ).strip()


def record(path: Path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path.resolve()), "status": "PENDING_MANUAL_FULL_RUN"}
    return {
        "path": str(path.resolve()),
        "status": "COMPLETE",
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def json_record(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def config_sha_from_record(record):
    if not isinstance(record, dict):
        return None
    if record.get("config_sha256") is not None:
        return str(record["config_sha256"])
    config = record.get("config")
    if isinstance(config, dict) and config.get("sha256") is not None:
        return str(config["sha256"])
    return None


def resolve_output_roots(config, *, artifact_root, protocol_root=None,
                         ad_artifact_root=None, ad_checkpoint_root=None,
                         event_artifact_root=None):
    """Resolve protocol, AD, and event trees without changing RCA history paths."""

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    configured_ad = config.get("ad_paths", {})
    root = resolve(artifact_root)
    return {
        "root": root.resolve(),
        "protocol": resolve(protocol_root or config.get("gt_output_dir", root / "protocol")).resolve(),
        "ad": resolve(ad_artifact_root or configured_ad.get("artifact_root", "artifacts/p5/v3/ad")).resolve(),
        "checkpoint": resolve(ad_checkpoint_root or configured_ad.get("checkpoint_root", "data/p5/v3/checkpoint")).resolve(),
        "event": resolve(event_artifact_root or configured_ad.get("event_root", root / "events")).resolve(),
    }


def main():
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)
    roots = resolve_output_roots(
        config, artifact_root=args.artifact_root,
        protocol_root=args.protocol_root,
        ad_artifact_root=args.ad_artifact_root,
        ad_checkpoint_root=args.ad_checkpoint_root,
        event_artifact_root=args.event_artifact_root,
    )
    root, protocol_root = roots["root"], roots["protocol"]
    ad_root, checkpoint_root, event_root = roots["ad"], roots["checkpoint"], roots["event"]
    required = {
        "protocol_manifest": protocol_root / "protocol_manifest.json",
        "split_manifest": protocol_root / "split_manifest.json",
        "gt_provenance": protocol_root / "provenance.json",
        "ad_data_manifest": ad_root / "ad_data_manifest.json",
        "ad_training_summary": ad_root / "ad_training_summary.json",
        "ad_train_predictions": ad_root / "ad_train_predictions.csv",
        "ad_test_predictions": ad_root / "ad_test_predictions.csv",
        "ad_calibration": ad_root / "reconstruction_calibration.json",
        "ad_checkpoint_best_train_loss": checkpoint_root / "best_train_loss.pt",
        "ad_checkpoint_best_train_f1": checkpoint_root / "best_train_f1.pt",
        "ad_checkpoint_last": checkpoint_root / "last.pt",
        "event_metrics": event_root / "event_detection_metrics.json",
        "ad_event_predictions": event_root / "ad_event_predictions.csv",
        "event_matching": event_root / "event_matching.csv",
        "rca_raw_index_manifest": PROJECT_ROOT / "data/p5/v3/rca_raw_index/index_manifest.json",
        "rca_gt_case_registry": root / "rca/rca_case_registry_gt.csv",
        "rca_detected_case_registry": root / "rca/rca_case_registry_detected.csv",
        "rca_gt_feature_manifest": root / "rca_gt_features/rca_feature_manifest.json",
        "rca_detected_feature_manifest": root / "rca_detected_features/rca_feature_manifest.json",
        "rca_train_manifest": root / "rca/rca_train_manifest.json",
        "rca_metrics": root / "rca/rca_metrics.json",
        "rca_model": PROJECT_ROOT / "data/p5/v3/rca_model/conditional_logit.npz",
        "rca_oracle_predictions": root / "rca/rca_oracle_predictions.csv",
        "rca_detected_predictions": root / "rca/rca_detected_predictions.csv",
        "detector_only_predictions": root / "rca/detector_only_predictions.csv",
        "root_frequency_predictions": root / "rca/root_frequency_predictions.csv",
        "e2e_diagnosis": root / "rca/e2e_diagnosis_metrics.json",
        "e2e_layered_report": root / "rca/e2e_layered_report.json",
    }
    artifacts = {name: record(path) for name, path in required.items()}
    pending = [name for name, value in artifacts.items() if value["status"] != "COMPLETE"]
    expected_config_sha = sha256_file(config_path)
    provenance_checks = {}
    for name in (
        "protocol_manifest", "split_manifest", "gt_provenance", "ad_data_manifest",
        "ad_training_summary", "event_metrics", "rca_gt_feature_manifest",
        "rca_detected_feature_manifest", "rca_train_manifest", "rca_metrics",
        "e2e_diagnosis", "e2e_layered_report",
    ):
        artifact_path = Path(artifacts[name]["path"])
        if artifacts[name]["status"] != "COMPLETE":
            continue
        value = json_record(artifact_path)
        actual = config_sha_from_record(value)
        provenance_checks[name] = {
            "config_sha256": actual,
            "matches_execution_config": actual == expected_config_sha,
        }
        if actual != expected_config_sha:
            raise ValueError(
                "{} has config SHA {} but execution config is {}".format(
                    name, actual, expected_config_sha
                )
            )
    for name in (
        "ad_data_manifest", "ad_training_summary", "event_metrics", "rca_metrics",
        "rca_gt_feature_manifest", "rca_detected_feature_manifest",
    ):
        if artifacts[name]["status"] == "COMPLETE":
            status = json_record(Path(artifacts[name]["path"])).get("status")
            allowed_statuses = ("FORMAL", "FORMAL_FULL_DATA")
            if name == "ad_data_manifest":
                # V2 preprocessing is published before model training. COMPLETE
                # is its terminal status, not a training/evaluation claim.
                allowed_statuses = allowed_statuses + ("COMPLETE",)
            if status not in allowed_statuses:
                raise ValueError("{} has non-formal status {}".format(name, status))
    for name in ("rca_train_manifest", "e2e_diagnosis", "e2e_layered_report"):
        if artifacts[name]["status"] == "COMPLETE":
            status = json_record(Path(artifacts[name]["path"])).get("status")
            if status != "FORMAL_FULL_DATA":
                raise ValueError("{} has non-formal status {}".format(name, status))
    run_table_path = Path(config["run_table"]["path"]).resolve()
    run_table_binding = {
        "path": str(run_table_path),
        "sha256": str(config["run_table"]["sha256"]),
        "configured_sha256": str(config["run_table"]["sha256"]),
        "actual_sha256": sha256_file(run_table_path) if run_table_path.is_file() else None,
    }
    run_table_binding["matches_config"] = (
        run_table_binding["actual_sha256"] == run_table_binding["configured_sha256"]
    )
    if not run_table_binding["matches_config"]:
        raise ValueError("configured GAIA run table is missing or has a SHA mismatch")
    formal_complete = not pending
    manifest = {
        "schema_version": "p5_v3_gaia_run_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL_FULL_DATA_COMPLETE" if formal_complete else "IMPLEMENTATION_READY_FORMAL_FULL_DATA_PENDING_MANUAL_RUN",
        "result_scope": "formal full GAIA" if formal_complete else "protocol binding and smoke validation only",
        "repository": {
            "branch": git("branch", "--show-current"),
            "head_commit": git("rev-parse", "HEAD"),
            "audit_baseline": str(config["source"]["audit_baseline"]),
        },
        "source_bindings": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "run_table": {
                **run_table_binding,
            },
            "event_registry": {
                "path": str((PROJECT_ROOT / str(config["event_registry"]["path"])).resolve()),
                "sha256": str(config["event_registry"]["sha256"]),
                "taxonomy_version": str(config["event_registry"]["taxonomy_version"]),
            },
            "ada_rca_source_commit": str(config["source"]["ada_rca_commit"]),
            "random_seed": int(config["random_seed"]),
        },
        "validation": {
            "pytest": str(args.pytest_result),
            "formal_artifacts_complete": formal_complete,
            "pending_formal_artifacts": pending,
            "full_preprocessing_executed": formal_complete,
            "full_training_executed": formal_complete,
            "full_test_executed": formal_complete,
            "provenance_checks": provenance_checks,
        },
        "formal_artifacts": artifacts,
        "manual_execution": {
            "note": "Run the documented V3 commands in order; this finalizer never executes preprocessing or training.",
            "documentation": "docs/GAIA_V3_IMPLEMENTATION.md",
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "run_manifest.json", manifest)
    report_lines = [
        "# GAIA V3 run report",
        "",
        "- Status: `{}`".format(manifest["status"]),
        "- Result scope: {}".format(manifest["result_scope"]),
        "- Config SHA-256: `{}`".format(manifest["source_bindings"]["config"]["sha256"]),
        "- Run-table SHA-256: `{}`".format(manifest["source_bindings"]["run_table"]["sha256"]),
        "- Pytest: {}".format(args.pytest_result),
        "",
        "## Artifact evidence",
        "",
    ]
    for name, value in artifacts.items():
        report_lines.append("- `{}`: {}".format(name, value["status"]))
    if pending:
        report_lines.extend([
            "",
            "Formal full-data preprocessing, training, and Test execution remain pending manual operator execution.",
            "No smoke artifact is promoted to a formal result.",
        ])
    (root / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(manifest["status"])


if __name__ == "__main__":
    main()
