#!/usr/bin/env python3
"""Validate the V3 bindings and write the deterministic protocol manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.gt import sha256_file, write_json
from src.e2e.protocol import (
    count_crossing_ad_windows,
    distribution,
    load_config,
    load_registry,
    temporal_blocks,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v3.json")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = (project_root / args.config).resolve()
    config = load_config(config_path)
    registry = load_registry(config, project_root)
    blocks = temporal_blocks(config)
    assigned = registry.loc[registry["split"].isin(["train", "test"])].copy()
    purged = registry.loc[~registry["split"].isin(["train", "test"])].copy()
    window_purge = count_crossing_ad_windows(
        blocks, int(config["ad"]["grid_seconds"]), int(config["ad"]["window_bins"])
    )
    output_dir = (project_root / str(config["gt_output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "schema_version": "p5_v3_gaia_split_manifest_v1",
        "protocol_id": str(config["protocol_id"]),
        "config_sha256": sha256_file(config_path),
        "event_registry_sha256": sha256_file(
            (project_root / str(config["event_registry"]["path"])).resolve()
        ),
        "blocks": [
            {
                "name": block.name,
                "start_ms": block.start_ms,
                "end_ms": block.end_ms,
                "interval": "[start_ms,end_ms)",
            }
            for block in blocks
        ],
        "metric_grid": {
            "grid_seconds": int(config["ad"]["grid_seconds"]),
            "N": int(
                (blocks[-1].end_ms - blocks[0].start_ms)
                // (int(config["ad"]["grid_seconds"]) * 1000)
            ),
            "K": int(
                7
                * (
                    (blocks[-1].end_ms - blocks[0].start_ms)
                    // (int(config["ad"]["grid_seconds"]) * 1000)
                )
                // 10
            ),
            "integer_rule": "K=(7*N)//10",
        },
        "event_counts": {
            "raw_registry_rows": int(len(registry)),
            "assigned_rows": int(len(assigned)),
            "train_rows": int((assigned["split"] == "train").sum()),
            "test_rows": int((assigned["split"] == "test").sum()),
            "purged_rows": int(len(purged)),
            "by_fault_type": distribution(registry, "fault_type"),
            "by_split": distribution(registry, "split"),
        },
        "ad_window_purge": window_purge,
        "semantics": {
            "train_test_isolation": True,
            "validation_split": False,
            "label_interval": str(config["ad"]["label_interval"]),
            "prediction_time": str(config["ad"]["prediction_time"]),
            "test_fit_or_select": False,
        },
    }
    protocol_manifest = {
        "schema_version": "p5_v3_gaia_protocol_manifest_v1",
        "evidence_status": "IMPLEMENTATION_BINDING",
        "protocol_id": str(config["protocol_id"]),
        "generated_by_commit": git_head(),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "raw_run_table": {
            "path": str(Path(config["run_table"]["path"]).resolve()),
            "sha256": str(config["run_table"]["sha256"]),
        },
        "event_registry": {
            "path": str(
                (project_root / str(config["event_registry"]["path"])).resolve()
            ),
            "sha256": sha256_file(
                (project_root / str(config["event_registry"]["path"])).resolve()
            ),
            "taxonomy_version": str(config["event_registry"]["taxonomy_version"]),
        },
        "split_manifest": str((output_dir / "split_manifest.json").resolve()),
        "formal_execution": {
            "full_preprocessing_executed": False,
            "full_training_executed": False,
            "full_test_executed": False,
            "smoke_only_until_explicit_operator_run": True,
        },
    }
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "protocol_manifest.json", protocol_manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "protocol_id": config["protocol_id"],
                "config_sha256": sha256_file(config_path),
                "registry_sha256": protocol_manifest["event_registry"]["sha256"],
                "train_rows": split_manifest["event_counts"]["train_rows"],
                "test_rows": split_manifest["event_counts"]["test_rows"],
                "purged_rows": split_manifest["event_counts"]["purged_rows"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
