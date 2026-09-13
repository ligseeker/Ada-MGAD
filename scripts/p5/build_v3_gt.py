#!/usr/bin/env python3
"""Generate the provenance-bound six-class GAIA V3 GT registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.gt import (
    build_provenance,
    build_registry,
    parse_raw_records,
    sha256_file,
    write_csv,
    write_json,
)
from src.e2e.protocol import load_config, temporal_blocks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v3.json")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def main() -> None:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)
    raw_root = Path(args.raw_root or config["gaia_raw_root"]).resolve()
    run_table = raw_root / "run/run/run/run_table_2021-07.csv"
    if not run_table.is_file():
        raise FileNotFoundError(run_table)
    configured_sha = str(config["run_table"]["sha256"])
    actual_sha = sha256_file(run_table)
    if actual_sha != configured_sha:
        raise ValueError(
            "raw run-table SHA-256 differs from frozen config: expected {}, got {}".format(
                configured_sha, actual_sha
            )
        )
    blocks = temporal_blocks(config)
    records = parse_raw_records(run_table)
    raw_registry, assigned, purged = build_registry(
        records,
        detector_start_ms=blocks[0].start_ms,
        detector_end_ms=blocks[-1].end_ms,
        split_ms=blocks[1].start_ms,
    )
    output_dir = (PROJECT_ROOT / (args.output_dir or config["gt_output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = (PROJECT_ROOT / config["event_registry"]["path"]).resolve()
    write_csv(registry_path, raw_registry)
    write_csv(output_dir / "gt_event_registry.csv", raw_registry)
    write_csv(output_dir / "assigned_event_registry.csv", assigned)
    write_csv(output_dir / "purged_event_registry.csv", purged)
    provenance = build_provenance(
        run_table,
        records,
        raw_registry,
        assigned,
        purged,
        detector_start_ms=blocks[0].start_ms,
        detector_end_ms=blocks[-1].end_ms,
        split_ms=blocks[1].start_ms,
        git_commit=git_head(),
    )
    provenance = {
        **provenance,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
            "rows": int(len(raw_registry)),
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in (
                ("gt_event_registry.csv", output_dir / "gt_event_registry.csv"),
                ("assigned_event_registry.csv", output_dir / "assigned_event_registry.csv"),
                ("purged_event_registry.csv", output_dir / "purged_event_registry.csv"),
            )
        },
    }
    write_json(output_dir / "provenance.json", provenance)
    print(json.dumps({
        "status": "PASS",
        "raw_run_table_sha256": actual_sha,
        "raw_gt_count": len(raw_registry),
        "assigned_train": int((assigned["split"] == "train").sum()) if len(assigned) else 0,
        "assigned_test": int((assigned["split"] == "test").sum()) if len(assigned) else 0,
        "purged": len(purged),
        "registry_sha256": sha256_file(registry_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
