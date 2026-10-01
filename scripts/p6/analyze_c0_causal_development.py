#!/usr/bin/env python3
"""Denominator-preserving Validation comparison of window NN and flat XGB."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import (evaluate_system_threshold, onset_density,
                                    system_score_frame, to_builtin)
from src.e2e.system_trigger_failure_audit import build_failure_ledger, stratified_summary
from src.e2e.trigger_development import TriggerDevelopmentState


def verify_completion(directory):
    completion = json.loads((directory / "completion_manifest.json").read_text())
    if completion["status"] != "COMPLETE_DEVELOPMENT_ONLY" or completion["test_inference_run"]:
        raise ValueError("not a completed development-only run")
    for name, sha in completion["files"].items():
        if sha256_file(directory / name) != sha:
            raise ValueError("completed artifact drift: " + name)
    return completion


def paired_table(ids, baseline_hits, candidate_hits):
    ids = set(ids)
    both = ids & baseline_hits & candidate_hits
    gained = ids & (candidate_hits - baseline_hits)
    lost = ids & (baseline_hits - candidate_hits)
    missed = ids - baseline_hits - candidate_hits
    return {"n": len(ids), "baseline_tp": len(ids & baseline_hits), "candidate_tp": len(ids & candidate_hits),
            "both_hit": len(both), "candidate_only": len(gained), "baseline_only": len(lost),
            "both_miss": len(missed), "gained_case_ids": sorted(gained), "lost_case_ids": sorted(lost),
            "delta_recall": (len(gained) - len(lost)) / len(ids) if ids else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--archived-validation-matching", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = load_config(args.base_config)
    if sha256_file(args.registry) != config["event_registry"]["sha256"]:
        raise ValueError("registry drift")
    state = TriggerDevelopmentState(config, args.data_root, args.registry)
    gt = state.gt_events("validation").set_index("case_id", drop=False)
    ids = sorted(gt.index)
    if len(ids) != 2901 or not gt.index.is_unique:
        raise ValueError("Validation GT denominator drift")
    historical = pd.read_csv(args.archived_validation_matching)
    historical = historical.loc[historical.case_id.notna()].set_index("case_id")
    if set(historical.index) != set(ids) or not historical.index.is_unique:
        raise ValueError("archived Validation event identity drift")
    for new, old in (("service", "gt_service"), ("fault_type", "fault_type"),
                     ("start_ms", "gt_start_ms"), ("end_ms", "gt_end_ms")):
        if gt.loc[ids, new].tolist() != historical.loc[ids, old].tolist():
            raise ValueError("archived event metadata mismatch: " + new)
    args.output_dir.mkdir(parents=True)
    results, hits = {}, {}
    for name, directory in (("baseline", args.baseline_run), ("candidate", args.candidate_run)):
        completion = verify_completion(directory)
        selection = json.loads((directory / "validation_selection.json").read_text())
        if name == "baseline" and selection["model_args"].get("graph_batch_scope") != "window":
            raise ValueError("baseline graph is not window independent")
        scores = pd.read_csv(directory / "validation_predictions.csv", float_precision="round_trip")
        indices = scores.sample_index.to_numpy(dtype=np.int64)
        if (not np.array_equal(indices, state.sample_indices("validation"))
                or not np.array_equal(scores.prediction_available_time,
                                       state.timestamps["train"][indices + 9] + 30_000)
                or not np.array_equal(scores.trigger_label,
                                       state.labels["train"][scores.sample_index.to_numpy() + 9])):
            raise ValueError("Validation windows/labels drift")
        threshold = float(selection["selected_validation_threshold"])
        frame = system_score_frame("validation", scores.prediction_available_time, scores.system_trigger_score)
        episodes, matching, metrics = evaluate_system_threshold(frame, gt.reset_index(drop=True), threshold)
        cases = matching.loc[matching.case_id.notna()].set_index("case_id")
        if not cases.index.is_unique or set(cases.index) != set(ids):
            raise ValueError("matching omitted or duplicated a GT event")
        for key in ("true_positive_events", "false_positive_events", "false_negative_events", "event_f1"):
            if abs(metrics[key] - selection["selected_validation_metrics"][key]) > 1e-12:
                raise ValueError("selected metric does not replay from exported scores: " + key)
        ledger, invariants = build_failure_ledger(
            gt.reset_index(drop=True), split="validation",
            slot_times_ms=scores.prediction_available_time.to_numpy(dtype=np.int64),
            slot_scores=scores.system_trigger_score.to_numpy(), threshold=threshold,
            episode_anchors_ms=episodes.t_hat.to_numpy(dtype=np.int64),
            episode_end_times_ms=episodes.episode_end_time.to_numpy(dtype=np.int64),
            matching=matching, origin_ms=state.blocks[0].start_ms,
            split_end_ms=state.blocks[1].end_ms, context_events=state.legal_events)
        for key in ("categories_sum_to_total", "matched_equals_tp", "unmatched_events_equal_fn",
                    "unmatched_episodes_equal_fp"):
            if not invariants[key]:
                raise ValueError("failure decomposition does not close: " + key)
        ledger.to_csv(args.output_dir / (name + "_failure_ledger.csv"), index=False)
        hits[name] = set(cases.index[cases.match_status.eq("matched")])
        results[name] = {"metrics": metrics, "threshold": threshold,
                         "execution_commit": completion["execution_commit"],
                         "completion_sha256": sha256_file(directory / "completion_manifest.json"),
                         "failure": invariants, "stratified": stratified_summary(ledger, split="validation")}
    density = onset_density(state.legal_events, origin_ms=state.blocks[0].start_ms).set_index("onset_bin")
    bins = (gt.start_ms.astype(np.int64) - state.blocks[0].start_ms) // 30_000
    multiplicity = bins.map(density.onset_count)
    groups = {"all": ids, "single_onset": gt.index[multiplicity.to_numpy() == 1],
              "multiple_onsets": gt.index[multiplicity.to_numpy() > 1],
              "all_memory": gt.index[gt.fault_type.eq("memory_anomalies")],
              "long_gt300s": gt.index[(gt.end_ms - gt.start_ms) > 300000]}
    paired = {name: paired_table(group, hits["baseline"], hits["candidate"]) for name, group in groups.items()}
    by_fault = {fault: paired_table(gt.index[gt.fault_type.eq(fault)], hits["baseline"], hits["candidate"])
                for fault in sorted(gt.fault_type.unique())}
    by_service = {service: paired_table(gt.index[gt.service.eq(service)], hits["baseline"], hits["candidate"])
                  for service in config["services"]}
    base, candidate = (results[name]["metrics"] for name in ("baseline", "candidate"))
    gates = {"recall_gain_at_least_0_02": candidate["event_recall"] - base["event_recall"] >= .02,
             "f1_gain_at_least_0_005": candidate["event_f1"] - base["event_f1"] >= .005,
             "precision_at_least_0_90": candidate["event_precision"] >= .90,
             "single_onset_recall_decline_at_most_0_03": paired["single_onset"]["delta_recall"] >= -.03}
    report = {"status": "GO_TO_E2E_DEVELOPMENT" if all(gates.values()) else "NO_GO_DEVELOPMENT",
              "test_read": False, "full_e2e_run": False, "gt_denominator": len(ids),
              "results": results, "paired": paired, "by_fault": by_fault, "by_service": by_service,
              "gates": gates, "limitations": ["reused Validation development screening",
              "70 percent Train preprocessing includes detector Validation inputs",
              "window graph independence does not certify online trace-parent availability"],
              "source_sha256": {str(path.resolve()): sha256_file(path) for path in
                                (Path(__file__), args.base_config, args.registry, args.archived_validation_matching)}}
    write_json(args.output_dir / "comparison.json", to_builtin(report))
    print(json.dumps({"status": report["status"], "gates": gates,
                      "metrics": {name: row["metrics"] for name, row in results.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
