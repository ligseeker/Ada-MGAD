#!/usr/bin/env python3
"""All-seed, denominator-preserving stability decision on frozen Validation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.p6.analyze_c0_causal_development import (paired_table, verify_completion,
                                                     verify_gt_identity)
from scripts.p6.tcn_replication import REPLICATION_SPEC, check_replication_config
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import trigger_temporal_blocks


def temporal_pairs(gt, baseline_matching, candidate_matching, start_ms, end_ms):
    """Stratify one global matching; never rematch at an artificial boundary."""
    baseline_hits = set(baseline_matching.loc[baseline_matching.match_status.eq("matched"), "case_id"])
    candidate_hits = set(candidate_matching.loc[candidate_matching.match_status.eq("matched"), "case_id"])
    midpoint = start_ms + (end_ms - start_ms) // 2
    strata = {}
    for name, lower, upper in (("first_half", start_ms, midpoint), ("second_half", midpoint, end_ms)):
        ids = gt.loc[(gt.start_ms >= lower) & (gt.start_ms < upper), "case_id"]
        paired = paired_table(ids, baseline_hits, candidate_hits)
        days = (upper - lower) / 86400000.0
        for role, frame in (("baseline", baseline_matching), ("candidate", candidate_matching)):
            false_alarms = frame.loc[frame.match_status.eq("false_alarm")]
            count = int(((false_alarms.t_hat >= lower) & (false_alarms.t_hat < upper)).sum())
            paired[role + "_global_fp"] = count
            paired[role + "_fp_per_day"] = count / days
        paired.update({"start_ms": lower, "end_ms": upper})
        strata[name] = paired
    if sum(item["n"] for item in strata.values()) != len(gt):
        raise ValueError("time strata do not preserve the full GT denominator")
    for role, frame in (("baseline", baseline_matching), ("candidate", candidate_matching)):
        if sum(item[role + "_global_fp"] for item in strata.values()) != int(frame.match_status.eq("false_alarm").sum()):
            raise ValueError("time strata do not preserve all global false alarms")
    return strata


def matching_identity(frame, gt, expected_metrics):
    cases = frame.loc[frame.case_id.notna()]
    recorded = cases.rename(columns={"gt_start_ms": "start_ms", "gt_end_ms": "end_ms",
                                     "gt_service": "service"})
    # Matching CSV has float interval/source columns because FP rows lack GT.
    for name in ("start_ms", "end_ms", "source_index"):
        recorded[name] = recorded[name].astype("int64")
    verify_gt_identity(recorded, gt)
    for status, key in (("matched", "true_positive_events"), ("miss", "false_negative_events"),
                        ("false_alarm", "false_positive_events")):
        if int(frame.match_status.eq(status).sum()) != expected_metrics[key]:
            raise ValueError("matching counts differ from sealed decoder report")
    if set(frame.match_status) - {"matched", "miss", "false_alarm"}:
        raise ValueError("unknown matching status")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed17-decoder", type=Path, required=True)
    parser.add_argument("--seed2026-decoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = json.loads(args.config.read_text())
    spec = config["replication"]
    reference = json.loads(Path(spec["reference_config"]).read_text())
    reference_selection = json.loads((Path(spec["reference_model_run"]) / "validation_selection.json").read_text())
    check_replication_config(config, reference, 17)
    baseline_dir = Path(config["baseline_run"])
    verify_completion(baseline_dir)
    gt = pd.read_csv(baseline_dir / "validation_gt.csv")
    if len(gt) != 2901:
        raise ValueError("full Validation GT denominator drift")
    base_path = ROOT / config["base_config"]
    block = trigger_temporal_blocks(load_config(base_path))[1]
    paths = [Path(__file__), args.config, base_path, Path(spec["reference_config"]),
             ROOT / "scripts/p6/tcn_replication.py", ROOT / "docs/P6_TCN_REPLICATION_PROTOCOL_20261001.md",
             ROOT / "scripts/p6/analyze_c0_causal_development.py", ROOT / "src/e2e/system_trigger.py",
             baseline_dir / "completion_manifest.json", baseline_dir / "validation_gt.csv",
             Path(spec["reference_model_run"]) / "validation_selection.json"]
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    results = {}
    baseline_reference = None
    for seed, directory in ((42, Path(spec["reference_decoder_run"])), (17, args.seed17_decoder),
                            (2026, args.seed2026_decoder)):
        verify_completion(directory)
        report = json.loads((directory / "decoder_comparison.json").read_text())
        model_path = [Path(path).parent for path in report["source_sha256"]
                      if path.endswith("/validation_selection.json") and Path(path).parent != baseline_dir]
        if len(model_path) != 1:
            raise ValueError("ambiguous decoder model binding")
        model_dir = model_path[0]
        verify_completion(model_dir)
        selection = json.loads((model_dir / "validation_selection.json").read_text())
        if (selection["model_args"]["random_seed"] != seed
                or dict(selection["model_args"], random_seed=42) != reference_selection["model_args"]
                or selection["training"] != reference["training"]
                or selection["tcn_spec"] != reference["tcn_spec"] or selection["target"] != reference["trigger_label"]
                or report["gt_denominator"] != 2901 or report["gate_margins"] != reference["development_gate"]
                or report["model_completion_sha256"] != sha256_file(model_dir / "completion_manifest.json")
                or report["protocol_id"] != (reference["protocol_id"] if seed == 42 else config["protocol_id"])
                or report["full_e2e_run"] or selection["model_class"] != "WindowCausalTCNTrigger"):
            raise ValueError("seed/model/recipe/scope drift")
        if seed != 42:
            lock = json.loads((model_dir / "development_input_lock.json").read_text())
            if lock.get("effective_seed") != seed or lock.get("replication_changes") != ["random_seed"]:
                raise ValueError("not a registered initialization-only replication")
        if json.loads((model_dir / "real_window_causality_gate.json").read_text()).get("passed") is not True:
            raise ValueError("actual-window independence gate failed")
        for path, expected in report["source_sha256"].items():
            if sha256_file(Path(path)) != expected:
                raise ValueError("archived decoder input drift: " + path)
            bindings[str(Path(path).resolve())] = expected
        for name in ("completion_manifest.json", "decoder_comparison.json",
                     "baseline_merged_matching.csv", "onset_bins_fixed_threshold_matching.csv"):
            bindings[str((directory / name).resolve())] = sha256_file(directory / name)
        candidate = pd.read_csv(directory / "onset_bins_fixed_threshold_matching.csv")
        baseline = pd.read_csv(directory / "baseline_merged_matching.csv")
        variant = report["results"]["onset_bins_fixed_threshold"]
        if variant["threshold"] != selection["selected_validation_threshold"]:
            raise ValueError("primary threshold changed after the merged selector")
        matching_identity(candidate, gt, variant["metrics"])
        matching_identity(baseline, gt, report["results"]["baseline_merged"]["metrics"])
        baseline_hits = set(baseline.loc[baseline.match_status.eq("matched"), "case_id"])
        if baseline_reference is None:
            baseline_reference = baseline_hits
        elif baseline_reference != baseline_hits:
            raise ValueError("baseline global matching differs across seeds")
        strata = temporal_pairs(gt, baseline, candidate, block.start_ms, block.end_ms)
        time_gates = {name: value["delta_recall"] is not None
                      and value["delta_recall"] >= REPLICATION_SPEC["minimum_half_recall_gain"]
                      for name, value in strata.items()}
        gate = report["gates"]["onset_bins_fixed_threshold"]
        if set(gate) != {"precision_floor", "recall_gain", "f1_gain", "single_onset_recall_decline_limit"}:
            raise ValueError("incomplete global gate")
        results[str(seed)] = {"metrics": variant["metrics"], "threshold": variant["threshold"],
                              "global_gates": gate, "time_gates": time_gates, "time_strata": strata,
                              "passed": all(gate.values()) and all(time_gates.values())}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "analysis_input_lock.json", {"execution_commit": head, "source_sha256": bindings})
    report = {"status": "GO_TO_FIT_OOS_AND_VALIDATION_E2E" if all(item["passed"] for item in results.values())
              else "STABILITY_NO_GO", "execution_commit": head, "gt_denominator": len(gt),
              "seeds": results, "primary_seed": 42, "best_seed_selection": False,
              "test_arrays_read": False, "test_inference_run": False, "full_e2e_run": False,
              "evidence_grade": "reused Validation development; not independent confirmation",
              "spec": REPLICATION_SPEC}
    write_json(args.output_dir / "replication_comparison.json", report)
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != head
            or any(sha256_file(Path(path)) != expected for path, expected in bindings.items())):
        raise ValueError("source/input drift during stability analysis")
    write_json(args.output_dir / "completion_manifest.json", {"status": "COMPLETE_DEVELOPMENT_ONLY",
        "execution_commit": head, "test_inference_run": False,
        "files": {str(path.relative_to(args.output_dir)): sha256_file(path)
                  for path in args.output_dir.rglob("*") if path.is_file()}})
    print(json.dumps({"status": report["status"], "seeds_passed": {s: v["passed"] for s, v in results.items()}}))


if __name__ == "__main__":
    main()
