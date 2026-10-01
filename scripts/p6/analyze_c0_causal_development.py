#!/usr/bin/env python3
"""Denominator-preserving Validation comparison of window NN and flat XGB."""

import argparse
import json
from pathlib import Path
import subprocess
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
from src.e2e.onset_trigger import OnsetDevelopmentState


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


def development_bindings(directory):
    """Bind archived scores, metadata and the Train inputs they actually used."""
    paths = [directory / name for name in ("completion_manifest.json", "validation_selection.json",
             "validation_predictions.csv", "development_input_lock.json", "validation_gt.csv")]
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    lock = json.loads((directory / "development_input_lock.json").read_text())
    for path, expected in lock["source_sha256"].items():
        if ("/train/" in path or path.endswith("/graph.npy") or path.endswith("/ad_data_manifest.json")
                or ("/configs/" in path and path.endswith(".json"))):
            if sha256_file(Path(path)) != expected:
                raise ValueError("archived Train input drift: " + path)
            bindings[str(Path(path).resolve())] = expected
    return bindings


def verify_target(selection, target, directory=None):
    recorded = selection.get("target", {}).get("target")
    expected = "one_bin_onset_frozen_ignore" if target == "onset30" else "recent_onset60"
    if recorded is None:
        if target != "recent_onset60" or directory is None:
            raise ValueError("archived target missing without a bound legacy configuration")
        lock = json.loads((directory / "development_input_lock.json").read_text())
        labels = []
        for path, digest in lock["source_sha256"].items():
            if "/configs/" in path and path.endswith(".json"):
                if sha256_file(Path(path)) != digest:
                    raise ValueError("legacy target configuration drift")
                config = json.loads(Path(path).read_text())
                if "trigger_label" in config:
                    labels.append(config["trigger_label"])
        if (len(labels) != 1 or labels[0].get("target") != "recent_onset_trigger"
                or labels[0].get("positive_window_seconds") != 60):
            raise ValueError("legacy target is not the frozen recent-onset60 task")
        recorded = expected
    if recorded != expected:
        raise ValueError("declared trigger target differs from archived selection")


def verify_gt_identity(recorded, current):
    columns = ["case_id", "source_index", "service", "fault_type", "start_ms", "end_ms"]
    if not recorded.case_id.is_unique or set(recorded.case_id) != set(current.case_id):
        raise ValueError("archived GT denominator/identity mismatch")
    left = recorded.sort_values("case_id")[columns].reset_index(drop=True)
    right = current.sort_values("case_id")[columns].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("archived GT annotations differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--archived-validation-matching", type=Path, required=True)
    parser.add_argument("--candidate-trigger-target", choices=("recent_onset60", "onset30"),
                        default="recent_onset60")
    parser.add_argument("--baseline-trigger-target", choices=("recent_onset60", "onset30"),
                        default="recent_onset60")
    parser.add_argument("--comparison-role", choices=("development_gate", "encoder_attribution"),
                        default="development_gate")
    parser.add_argument("--gate-config", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = load_config(args.base_config)
    if sha256_file(args.registry) != config["event_registry"]["sha256"]:
        raise ValueError("registry drift")
    state = (OnsetDevelopmentState(config, args.data_root, args.registry)
             if args.baseline_trigger_target == "onset30"
             else TriggerDevelopmentState(config, args.data_root, args.registry))
    candidate_state = (OnsetDevelopmentState(config, args.data_root, args.registry)
                       if args.candidate_trigger_target == "onset30"
                       else TriggerDevelopmentState(config, args.data_root, args.registry))
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
    execution_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = [Path(__file__), ROOT / "src/e2e/onset_trigger.py", ROOT / "src/e2e/trigger_development.py",
               ROOT / "src/e2e/system_trigger.py", ROOT / "src/e2e/system_trigger_data.py",
               ROOT / "src/e2e/event_detection.py", ROOT / "src/e2e/system_trigger_failure_audit.py",
               args.base_config, args.registry, args.archived_validation_matching]
    if args.gate_config:
        sources.append(args.gate_config)
    bindings = {str(path.resolve()): sha256_file(path) for path in sources}
    selections = {}
    for name, directory, target in (("baseline", args.baseline_run, args.baseline_trigger_target),
                                    ("candidate", args.candidate_run, args.candidate_trigger_target)):
        verify_completion(directory)
        selections[name] = json.loads((directory / "validation_selection.json").read_text())
        verify_target(selections[name], target, directory)
        verify_gt_identity(pd.read_csv(directory / "validation_gt.csv"), gt.reset_index(drop=True))
        bindings.update(development_bindings(directory))
    if args.comparison_role == "encoder_attribution":
        from src.e2e.system_trigger_tcn import TCN_SPEC
        baseline, successor = selections["baseline"], selections["candidate"]
        lock = json.loads((args.candidate_run / "development_input_lock.json").read_text())
        real_gate_path = args.candidate_run / "real_window_causality_gate.json"
        real_gate = json.loads(real_gate_path.read_text())
        if (args.baseline_trigger_target != "onset30" or args.candidate_trigger_target != "onset30"
                or baseline["model_args"] != successor["model_args"]
                or baseline["training"] != successor["training"]
                or baseline["target"] != successor["target"]
                or successor.get("model_class") != "WindowCausalTCNTrigger"
                or lock.get("model_class") != "WindowCausalTCNTrigger"
                or successor.get("tcn_spec") != TCN_SPEC or lock.get("tcn_spec") != TCN_SPEC
                or not lock.get("training_factory_only_verified")):
            raise ValueError("encoder attribution changes more than the declared encoder package")
        # The training completion manifest covers this actual-input gate.
        if real_gate.get("passed") is not True:
            raise ValueError("candidate actual-window independence gate failed")
        bindings[str(real_gate_path.resolve())] = sha256_file(real_gate_path)
        bindings[str((ROOT / "src/e2e/system_trigger_tcn.py").resolve())] = sha256_file(ROOT / "src/e2e/system_trigger_tcn.py")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "analysis_input_lock.json", {"execution_commit": execution_commit,
               "source_sha256": bindings, "test_read": False, "comparison_role": args.comparison_role})
    results, hits = {}, {}
    for name, directory in (("baseline", args.baseline_run), ("candidate", args.candidate_run)):
        completion = verify_completion(directory)
        selection = selections[name]
        if name == "baseline" and selection["model_args"].get("graph_batch_scope") != "window":
            raise ValueError("baseline graph is not window independent")
        scores = pd.read_csv(directory / "validation_predictions.csv", float_precision="round_trip")
        indices = scores.sample_index.to_numpy(dtype=np.int64)
        label_state = state if name == "baseline" else candidate_state
        if (not np.array_equal(indices, state.sample_indices("validation"))
                or not np.array_equal(scores.prediction_available_time,
                                       state.timestamps["train"][indices + 9] + 30_000)
                or not np.array_equal(scores.trigger_label,
                                       label_state.labels["train"][scores.sample_index.to_numpy() + 9])):
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
    if args.gate_config:
        margins = json.loads(args.gate_config.read_text())["development_gate"]
        gates = {"recall_gain": candidate["event_recall"] - base["event_recall"] >= margins["recall_gain"],
                 "f1_gain": candidate["event_f1"] - base["event_f1"] >= margins["f1_gain"],
                 "precision_floor": candidate["event_precision"] >= margins["precision_floor"],
                 "single_onset_recall_decline_limit": paired["single_onset"]["delta_recall"] >= -margins["single_onset_recall_decline_limit"]}
    else:
        margins = {"recall_gain": .02, "f1_gain": .005, "precision_floor": .90,
                   "single_onset_recall_decline_limit": .03}
        gates = {"recall_gain_at_least_0_02": candidate["event_recall"] - base["event_recall"] >= .02,
                 "f1_gain_at_least_0_005": candidate["event_f1"] - base["event_f1"] >= .005,
                 "precision_at_least_0_90": candidate["event_precision"] >= .90,
                 "single_onset_recall_decline_at_most_0_03": paired["single_onset"]["delta_recall"] >= -.03}
    status = ("ENCODER_ATTRIBUTION_COMPLETE" if args.comparison_role == "encoder_attribution" else
              ("GO_TO_E2E_DEVELOPMENT" if all(gates.values()) else "NO_GO_DEVELOPMENT"))
    if args.comparison_role == "encoder_attribution":
        gates = {}
    report = {"status": status, "execution_commit": execution_commit,
              "test_read": False, "full_e2e_run": False, "gt_denominator": len(ids),
              "results": results, "paired": paired, "by_fault": by_fault, "by_service": by_service,
              "gates": gates, "gate_margins": margins,
              "candidate_trigger_target": args.candidate_trigger_target,
              "baseline_trigger_target": args.baseline_trigger_target,
              "comparison_role": args.comparison_role,
              "limitations": ["reused Validation development screening",
              "70 percent Train preprocessing includes detector Validation inputs",
              "window graph independence does not certify online trace-parent availability"],
              "source_sha256": bindings}
    write_json(args.output_dir / "comparison.json", to_builtin(report))
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != execution_commit
            or any(sha256_file(Path(path)) != sha for path, sha in bindings.items())):
        raise ValueError("source/input drift; comparison remains incomplete")
    write_json(args.output_dir / "completion_manifest.json", {
        "status": "COMPLETE_DEVELOPMENT_ONLY", "execution_commit": execution_commit,
        "test_inference_run": False,
        "files": {str(path.relative_to(args.output_dir)): sha256_file(path)
                  for path in args.output_dir.rglob("*") if path.is_file()},
    })
    print(json.dumps({"status": report["status"], "gates": gates,
                      "metrics": {name: row["metrics"] for name, row in results.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
