#!/usr/bin/env python3
"""Bind P5-I1 to G0R2 evidence and emit chronological split manifests."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.protocol import (
    assign_event_blocks,
    count_crossing_ad_windows,
    distribution,
    load_config,
    load_registry,
    purge_rca_cases,
    sha256_file,
    temporal_blocks,
    write_json,
)
from util.GAIA.pre_GAIA import _read_truncated_csv, parse_anomaly_event


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v1.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def iso_ms(value: int) -> str:
    return pd.Timestamp(value, unit="ms", tz="UTC").isoformat()


def cpu_parser_regression(run_table: Path) -> dict:
    rows = _read_truncated_csv(str(run_table))
    cpu = rows[rows["message"].astype(str).str.contains("[cpu_anomalies]", regex=False)]
    legacy = cpu["message"].astype(str).str.contains(
        r"lasts\s+\d+\s+seconds", regex=True
    )
    current = cpu.apply(
        lambda row: bool(parse_anomaly_event(row)["ed_time"]), axis=1
    )
    return {
        "cpu_rows": len(cpu),
        "before_integer_only_parse_success": int(legacy.sum()),
        "before_integer_only_parse_failure": int((~legacy).sum()),
        "after_float_duration_parse_success": int(current.sum()),
        "after_float_duration_parse_failure": int((~current).sum()),
        "change_scope": "CPU duration regex and float conversion only",
    }


def main() -> None:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)
    registry = load_registry(config, PROJECT_ROOT)
    blocks = temporal_blocks(config)
    assigned, raw_purged = assign_event_blocks(registry, blocks)
    retained, rca_purged = purge_rca_cases(
        assigned, blocks, int(config["rca"]["window_seconds"])
    )

    output_dir = (PROJECT_ROOT / (args.output_dir or str(config["output_dir"]))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    common = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "generated_at_utc": generated_at,
        "git_commit": git_head(),
        "random_seed": int(config["random_seed"]),
    }

    registry_manifest = dict(common)
    run_table_path = Path(str(config["run_table"]["path"]))
    registry_manifest.update({
        "schema_version": "p5_i1_event_registry_manifest_v1",
        "event_count": len(registry),
        "event_registry_path": str((PROJECT_ROOT / config["event_registry"]["path"]).resolve()),
        "event_registry_sha256": str(config["event_registry"]["sha256"]),
        "run_table_path": str(config["run_table"]["path"]),
        "run_table_sha256": sha256_file(run_table_path),
        "cpu_parser_regression": cpu_parser_regression(run_table_path),
        "fault_counts": distribution(registry, "fault_type"),
        "service_counts": distribution(registry, "service"),
        "semantics": str(config["event_registry"]["semantics"]),
        "excluded_from_e2e_gt": list(config["event_registry"]["excluded"]),
    })
    if registry_manifest["run_table_sha256"] != str(config["run_table"]["sha256"]):
        raise ValueError("run-table SHA-256 mismatch")

    block_records = []
    for block in blocks:
        raw_block = assigned[assigned["split"] == block.name]
        retained_block = retained[retained["split"] == block.name]
        rca_purged_block = rca_purged[rca_purged["split"] == block.name]
        block_records.append({
            "split": block.name,
            "absolute_start_ms": block.start_ms,
            "absolute_start_utc": iso_ms(block.start_ms),
            "absolute_end_ms": block.end_ms,
            "absolute_end_utc": iso_ms(block.end_ms),
            "duration_seconds": (block.end_ms - block.start_ms) / 1000.0,
            "event_count_after_raw_boundary_purge": len(raw_block),
            "rca_case_count_after_w300_purge": len(retained_block),
            "fault_counts": distribution(raw_block, "fault_type"),
            "service_counts": distribution(raw_block, "service"),
            "purged_rca_cases": len(rca_purged_block),
            "purged_rca_case_ids": sorted(rca_purged_block.get("case_id", pd.Series(dtype=str)).astype(str).tolist()),
        })

    split_manifest = dict(common)
    split_manifest.update({
        "schema_version": "p5_i1_split_manifest_v1",
        "split_mode": str(config["split"]["mode"]),
        "split_definition": str(config["split"]["definition"]),
        "blocks": block_records,
        "raw_injection_boundary_purge": {
            "count": len(raw_purged),
            "case_ids": sorted(raw_purged.get("case_id", pd.Series(dtype=str)).astype(str).tolist()),
            "rule": "exclude complete raw injection interval if it crosses a split boundary",
        },
        "ad_window_boundary_purge": count_crossing_ad_windows(
            blocks,
            int(config["ad"]["grid_seconds"]),
            int(config["ad"]["window_bins"]),
        ),
        "rca_w300_boundary_purge": {
            "count": len(rca_purged),
            "case_ids": sorted(rca_purged.get("case_id", pd.Series(dtype=str)).astype(str).tolist()),
            "rule": "exclude raw-injection union [t0-300s,t0+300s) context crossing a split boundary",
        },
    })

    protocol_manifest = dict(common)
    protocol_manifest.update({
        "schema_version": "p5_i1_protocol_manifest_v1",
        "protocol": config,
        "bindings": {
            "event_registry_sha256": registry_manifest["event_registry_sha256"],
            "run_table_sha256": registry_manifest["run_table_sha256"],
            "event_count": len(registry),
        },
        "selection_firewall": {
            "split_selected_without_performance": True,
            "rca_temporal_spec_selected_without_performance": True,
            "test_for_checkpoint_selection": False,
            "test_for_threshold_selection": False,
        },
    })

    write_json(output_dir / "event_registry_manifest.json", registry_manifest)
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "protocol_manifest.json", protocol_manifest)
    retained.to_csv(output_dir / "rca_case_registry.csv", index=False)
    print(json.dumps({
        "event_count": len(registry),
        "raw_purged": len(raw_purged),
        "rca_purged": len(rca_purged),
        "rca_retained": len(retained),
        "output_dir": str(output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
