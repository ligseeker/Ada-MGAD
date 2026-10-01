#!/usr/bin/env python3
"""Separate decoder ablation on frozen onset-model Validation probabilities."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.p6.analyze_c0_causal_development import paired_table, verify_completion
from scripts.p6.run_c0_trigger import git_head
from src.e2e.bin_trigger_decoder import evaluate_bin_threshold, select_bin_threshold
from src.e2e.onset_trigger import OnsetDevelopmentState
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import evaluate_system_threshold, onset_density, system_score_frame, to_builtin
from src.e2e.system_trigger_failure_audit import build_failure_ledger, stratified_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onset-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = json.loads(args.config.read_text())
    baseline_dir = Path(config["baseline_run"])
    for directory in (baseline_dir, args.onset_run):
        verify_completion(directory)
    selection = json.loads((args.onset_run / "validation_selection.json").read_text())
    if selection["protocol_id"] != config["protocol_id"]:
        raise ValueError("onset model protocol mismatch")
    base_path = ROOT / config["base_config"]
    base = load_config(base_path)
    if sha256_file(args.registry) != base["event_registry"]["sha256"]:
        raise ValueError("registry digest drift")
    state = OnsetDevelopmentState(base, args.data_root, args.registry)
    gt = state.gt_events("validation")
    scores = pd.read_csv(args.onset_run / "validation_predictions.csv", float_precision="round_trip")
    dataset = state.build_dataset("validation")
    if (len(gt) != config["expected_cohort"]["validation_events"]
            or not np.array_equal(scores.sample_index, dataset.sample_indices)
            or not np.array_equal(scores.prediction_available_time, dataset.prediction_times())
            or not np.array_equal(scores.trigger_label, dataset.labels_at())):
        raise ValueError("onset score window/time/label/GT mismatch")
    sources = (Path(__file__), ROOT / "src/e2e/bin_trigger_decoder.py", ROOT / "src/e2e/onset_trigger.py",
               ROOT / "src/e2e/trigger_development.py", ROOT / "src/e2e/event_detection.py",
               ROOT / "src/e2e/system_trigger.py", ROOT / "src/e2e/system_trigger_failure_audit.py",
               ROOT / "scripts/p6/analyze_c0_causal_development.py", args.config, base_path, args.registry,
               args.data_root / "train/timestamps.npy", args.onset_run / "completion_manifest.json",
               args.onset_run / "validation_predictions.csv", args.onset_run / "validation_selection.json",
               baseline_dir / "completion_manifest.json", baseline_dir / "validation_predictions.csv",
               baseline_dir / "validation_selection.json")
    bindings = {str(path.resolve()): sha256_file(path) for path in sources}
    execution_commit = git_head()
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "analysis_input_lock.json", {"source_sha256": bindings,
               "execution_commit": execution_commit,
               "test_read": False, "candidate_rule": "one independent candidate per positive bin"})
    frame = system_score_frame("validation", scores.prediction_available_time, scores.system_trigger_score)
    fixed_threshold = float(selection["selected_validation_threshold"])
    best_threshold, candidate_count = select_bin_threshold(frame, gt)
    results, hits = {}, {}
    baseline_selection = json.loads((baseline_dir / "validation_selection.json").read_text())
    baseline_scores = pd.read_csv(baseline_dir / "validation_predictions.csv", float_precision="round_trip")
    if (not np.array_equal(baseline_scores.sample_index, dataset.sample_indices)
            or not np.array_equal(baseline_scores.prediction_available_time, dataset.prediction_times())):
        raise ValueError("baseline window/time mismatch")
    baseline_frame = system_score_frame("validation", baseline_scores.prediction_available_time,
                                        baseline_scores.system_trigger_score)
    variants = (("baseline_merged", baseline_frame, float(baseline_selection["selected_validation_threshold"]), False),
                ("onset_merged", frame, fixed_threshold, False),
                ("onset_bins_fixed_threshold", frame, fixed_threshold, True),
                ("onset_bins_validation_selected", frame, best_threshold, True))
    for name, variant_frame, threshold, independent in variants:
        evaluator = evaluate_bin_threshold if independent else evaluate_system_threshold
        episodes, matching, metrics = evaluator(variant_frame, gt, threshold)
        cases = matching.loc[matching.case_id.notna()]
        if not cases.case_id.is_unique or set(cases.case_id) != set(gt.case_id):
            raise ValueError("GT denominator not preserved")
        if not independent:
            reference = baseline_selection if name == "baseline_merged" else selection
            for key in ("true_positive_events", "false_positive_events", "false_negative_events", "event_f1"):
                if abs(metrics[key] - reference["selected_validation_metrics"][key]) > 1e-12:
                    raise ValueError("merged result does not replay: " + name)
        ledger, invariants = build_failure_ledger(
            gt, split="validation", slot_times_ms=variant_frame.prediction_available_time.to_numpy(),
            slot_scores=variant_frame.system_score.to_numpy(), threshold=threshold,
            episode_anchors_ms=episodes.t_hat.to_numpy(dtype=np.int64),
            episode_end_times_ms=episodes.episode_end_time.to_numpy(dtype=np.int64), matching=matching,
            origin_ms=state.blocks[0].start_ms, split_end_ms=state.blocks[1].end_ms,
            context_events=state.legal_events)
        if not all(invariants[key] for key in ("categories_sum_to_total", "matched_equals_tp",
                                             "unmatched_events_equal_fn", "unmatched_episodes_equal_fp")):
            raise ValueError("failure ledger does not close")
        hits[name] = set(cases.loc[cases.match_status.eq("matched"), "case_id"])
        episodes.to_csv(args.output_dir / (name + "_candidates.csv"), index=False)
        matching.to_csv(args.output_dir / (name + "_matching.csv"), index=False)
        ledger.to_csv(args.output_dir / (name + "_failure_ledger.csv"), index=False)
        results[name] = {"threshold": threshold, "metrics": metrics, "failure": invariants,
                         "stratified": stratified_summary(ledger, split="validation")}
    pairs = {name: {"vs_baseline": paired_table(gt.case_id, hits["baseline_merged"], hits[name]),
                    "vs_onset_merged": paired_table(gt.case_id, hits["onset_merged"], hits[name])}
             for name in ("onset_bins_fixed_threshold", "onset_bins_validation_selected")}
    margins = config["development_gate"]
    baseline = results["baseline_merged"]["metrics"]
    density = onset_density(state.legal_events, origin_ms=state.blocks[0].start_ms).set_index("onset_bin")
    multiplicity = ((gt.start_ms - state.blocks[0].start_ms) // 30_000).map(density.onset_count)
    single_ids = gt.loc[multiplicity.eq(1), "case_id"]
    gates = {}
    for name in pairs:
        metrics = results[name]["metrics"]
        single = paired_table(single_ids, hits["baseline_merged"], hits[name])
        pairs[name]["single_onset_vs_baseline"] = single
        gates[name] = {"precision_floor": metrics["event_precision"] >= margins["precision_floor"],
                       "recall_gain": metrics["event_recall"] - baseline["event_recall"] >= margins["recall_gain"],
                       "f1_gain": metrics["event_f1"] - baseline["event_f1"] >= margins["f1_gain"],
                       "single_onset_recall_decline_limit": single["delta_recall"] >= -margins["single_onset_recall_decline_limit"]}
    report = {"status": "DEVELOPMENT_ANALYSIS_COMPLETE", "test_read": False, "full_e2e_run": False,
              "execution_commit": execution_commit,
              "gt_denominator": len(gt), "results": results, "paired": pairs, "gates": gates,
              "threshold_candidate_count": candidate_count, "source_sha256": bindings,
              "controlled_contrast": "fixed threshold: only the decoder changes",
              "operating_point_contrast": "Validation-selected: decoder and its derived threshold; reused Validation"}
    write_json(args.output_dir / "decoder_comparison.json", to_builtin(report))
    if git_head() != execution_commit or any(sha256_file(Path(path)) != sha for path, sha in bindings.items()):
        raise ValueError("source/input drift during analysis")
    write_json(args.output_dir / "completion_manifest.json", {
        "status": "COMPLETE_DEVELOPMENT_ONLY", "test_inference_run": False,
        "execution_commit": execution_commit,
        "files": {str(path.relative_to(args.output_dir)): sha256_file(path)
                  for path in args.output_dir.rglob("*") if path.is_file()},
    })
    print(json.dumps({"status": report["status"], "gates": gates,
                      "metrics": {name: value["metrics"] for name, value in results.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
