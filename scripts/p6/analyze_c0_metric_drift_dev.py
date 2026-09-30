"""Paired Fit/Validation-only gate for the single C0 Metric-drift candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.protocol import sha256_file, write_json


def case_table(path: Path):
    frame = pd.read_csv(path)
    if set(frame["match_status"]) - {"matched", "miss", "false_alarm"}:
        raise ValueError("unexpected matching status")
    cases = frame[frame["case_id"].notna()].copy()
    if cases["case_id"].duplicated().any() or (cases["match_status"] == "false_alarm").any():
        raise ValueError("GT cases must appear exactly once and cannot be false alarms")
    if frame[frame["case_id"].isna()]["match_status"].ne("false_alarm").any():
        raise ValueError("only false alarms may lack a GT case")
    return frame, cases.set_index("case_id", drop=False)


def counts(frame, cases):
    tp = int((cases["match_status"] == "matched").sum())
    fp = int((frame["match_status"] == "false_alarm").sum())
    fn = int((cases["match_status"] == "miss").sum())
    if tp + fn != len(cases):
        raise ValueError("GT denominator does not close")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(cases) if len(cases) else 0.0
    f1 = 2 * tp / (len(cases) + tp + fp) if len(cases) + tp + fp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "gt": int(len(cases)), "episodes": tp + fp}


def group_results(ids, base_cases, candidate_cases):
    ids = sorted(set(ids))
    if set(ids) - set(base_cases.index):
        raise ValueError("group contains an unknown case")
    base_hits = {case_id for case_id in ids if base_cases.loc[case_id, "match_status"] == "matched"}
    candidate_hits = {case_id for case_id in ids
                      if candidate_cases.loc[case_id, "match_status"] == "matched"}
    return {"n": len(ids), "baseline_tp": len(base_hits), "candidate_tp": len(candidate_hits),
            "gained": sorted(candidate_hits - base_hits), "lost": sorted(base_hits - candidate_hits)}


def analyze(baseline_path: Path, candidate_path: Path, clean_path: Path,
            split_manifest_path: Path, selection_path: Path):
    base_frame, base_cases = case_table(baseline_path)
    candidate_frame, candidate_cases = case_table(candidate_path)
    if set(base_cases.index) != set(candidate_cases.index):
        raise ValueError("baseline and candidate GT cohorts differ")
    for column in ("fault_type", "gt_service", "gt_start_ms", "gt_end_ms"):
        left = base_cases.loc[sorted(base_cases.index), column].tolist()
        right = candidate_cases.loc[sorted(base_cases.index), column].tolist()
        if left != right:
            raise ValueError("GT metadata differs: " + column)
    baseline = counts(base_frame, base_cases)
    candidate = counts(candidate_frame, candidate_cases)
    if baseline["gt"] != 2901 or (baseline["tp"], baseline["fp"], baseline["fn"]) != (2124, 13, 777):
        raise ValueError("sealed C0 Validation baseline did not replay")
    selection = json.loads(selection_path.read_text())
    selected = selection["selected_validation_metrics"]
    for key, value in (("true_positive_events", candidate["tp"]),
                       ("false_positive_events", candidate["fp"]),
                       ("false_negative_events", candidate["fn"])):
        if int(selected[key]) != value:
            raise ValueError("candidate matching differs from selected checkpoint: " + key)

    clean = pd.read_csv(clean_path)
    if clean["case_id"].duplicated().any() or set(clean["case_id"]) - set(base_cases.index):
        raise ValueError("clean audit case IDs are invalid")
    clean_memory = set(clean.loc[(clean["fault_type"] == "memory_anomalies") & clean["supported"], "case_id"])
    if len(clean_memory) != 30:
        raise ValueError("clean supported memory cohort changed")
    onset = base_cases["gt_start_ms"].astype("int64")
    manifest = json.loads(split_manifest_path.read_text())
    origin = int(manifest["blocks"]["fit"]["start_ms"])
    onset_bin = (onset - origin) // 30000
    per_bin = onset_bin.value_counts()
    multiplicity = onset_bin.map(per_bin)

    groups = {
        "clean_supported_memory": clean_memory,
        "all_memory": set(base_cases.index[base_cases["fault_type"] == "memory_anomalies"]),
        "long_gt300s": set(base_cases.index[(base_cases["gt_end_ms"] - base_cases["gt_start_ms"]) > 300000]),
        "login_failure": set(base_cases.index[base_cases["fault_type"] == "login_failure"]),
        "single_onset_bin": set(base_cases.index[multiplicity == 1]),
        "multi_onset_bin": set(base_cases.index[multiplicity > 1]),
        "all": set(base_cases.index),
    }
    paired = {name: group_results(ids, base_cases, candidate_cases) for name, ids in groups.items()}
    days = pd.to_datetime(base_cases["gt_start_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    day_results = {day: group_results(set(base_cases.index[days == day]), base_cases, candidate_cases)
                   for day in sorted(days.unique())}
    memory = paired["clean_supported_memory"]
    gates = {
        "clean_memory_gained_at_least_5": len(memory["gained"]) >= 5,
        "clean_memory_lost_zero": len(memory["lost"]) == 0,
        "overall_tp_gain_at_least_10": candidate["tp"] - baseline["tp"] >= 10,
        "overall_f1_at_least_0_8432": candidate["f1"] >= 0.8432,
        "overall_precision_at_least_0_98": candidate["precision"] >= 0.98,
    }
    return {
        "schema_version": "p6_c0_metric_drift_development_comparison_v1",
        "status": "GO_TO_EXPLORATORY_TEST" if all(gates.values()) else "NO_GO_DEVELOPMENT",
        "test_inference_run": False,
        "source_sha256": {str(p): sha256_file(p) for p in
                          (baseline_path, candidate_path, clean_path, split_manifest_path, selection_path)},
        "selection": {"epoch": int(selection["selected_epoch"]),
                      "threshold": float(selection["selected_validation_threshold"]),
                      "fit_metric_drift_stats": selection.get("metric_drift_stats"),
                      "metric_drift_alpha": selection.get("metric_drift_alpha")},
        "baseline": baseline, "candidate": candidate, "paired": paired,
        "utc_onset_days": day_results, "gates": gates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-matching", "candidate-matching", "clean-events",
                 "split-manifest", "selection", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.baseline_matching, args.candidate_matching,
                     args.clean_events, args.split_manifest, args.selection)
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "baseline": result["baseline"],
                      "candidate": result["candidate"], "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
