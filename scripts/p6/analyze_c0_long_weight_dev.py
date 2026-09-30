"""Paired Validation gate for GPU C0 control versus long-onset weighting."""

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


def read_matching(path: Path):
    frame = pd.read_csv(path)
    if set(frame["match_status"]) - {"matched", "miss", "false_alarm"}:
        raise ValueError("unexpected matching status")
    cases = frame[frame["case_id"].notna()].copy()
    if cases["case_id"].duplicated().any() or (cases["match_status"] == "false_alarm").any():
        raise ValueError("GT identity is not unique")
    if frame[frame["case_id"].isna()]["match_status"].ne("false_alarm").any():
        raise ValueError("only false alarms may lack a GT case")
    return frame, cases.set_index("case_id", drop=False)


def metrics(frame, cases):
    tp = int(cases["match_status"].eq("matched").sum())
    fn = int(cases["match_status"].eq("miss").sum())
    fp = int(frame["match_status"].eq("false_alarm").sum())
    n = len(cases)
    if tp + fn != n:
        raise ValueError("GT denominator does not close")
    return {"tp": tp, "fp": fp, "fn": fn, "gt": n, "episodes": tp + fp,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / n if n else 0.0,
            "f1": 2 * tp / (n + tp + fp) if n + tp + fp else 0.0}


def paired_group(ids, arms):
    ids = sorted(set(ids))
    hits = {name: {case_id for case_id in ids if cases.loc[case_id, "match_status"] == "matched"}
            for name, cases in arms.items()}
    return {"n": len(ids), "tp": {name: len(group) for name, group in hits.items()},
            "weighted_vs_gpu_gained": sorted(hits["weighted"] - hits["gpu_control"]),
            "weighted_vs_gpu_lost": sorted(hits["gpu_control"] - hits["weighted"]),
            "weighted_vs_historical_gained": sorted(hits["weighted"] - hits["historical"]),
            "weighted_vs_historical_lost": sorted(hits["historical"] - hits["weighted"])}


def analyze(paths, clean_path: Path, manifest_path: Path, selection_paths):
    frames, arms = {}, {}
    for name, path in paths.items():
        frames[name], arms[name] = read_matching(path)
    ids = set(arms["historical"].index)
    if any(set(cases.index) != ids for cases in arms.values()) or len(ids) != 2901:
        raise ValueError("Validation GT cohorts differ")
    for column in ("fault_type", "gt_service", "gt_start_ms", "gt_end_ms"):
        reference = arms["historical"].loc[sorted(ids), column].tolist()
        if any(cases.loc[sorted(ids), column].tolist() != reference for cases in arms.values()):
            raise ValueError("GT metadata differs: " + column)
    scores = {name: metrics(frames[name], arms[name]) for name in arms}
    if (scores["historical"]["tp"], scores["historical"]["fp"], scores["historical"]["fn"]) != (2124, 13, 777):
        raise ValueError("historical C0 matching failed to replay")
    selections = {name: json.loads(path.read_text()) for name, path in selection_paths.items()}
    for name in ("gpu_control", "weighted"):
        selected = selections[name]["selected_validation_metrics"]
        for key, count in (("true_positive_events", "tp"),
                           ("false_positive_events", "fp"), ("false_negative_events", "fn")):
            if int(selected[key]) != scores[name][count]:
                raise ValueError(name + " selected checkpoint does not replay matching")
    weight_stats = selections["weighted"].get("long_onset_weight_stats")
    if weight_stats is None or selections["gpu_control"].get("long_onset_weight_stats") is not None:
        raise ValueError("weighting/control identity differs from the plan")
    for key, value in (("fit_events", 7443), ("long_events_gt300s", 362),
                       ("positive_bins", 12897), ("long_positive_bins", 718)):
        if int(weight_stats[key]) != value:
            raise ValueError("Fit weight count drift: " + key)
    if abs(float(weight_stats["weighted_positive_mass"]) - 12897) > 0.001:
        raise ValueError("positive loss mass was not preserved")

    clean = pd.read_csv(clean_path)
    if clean["case_id"].duplicated().any() or set(clean["case_id"]) - ids:
        raise ValueError("clean cohort identity invalid")
    clean_memory = set(clean.loc[clean["fault_type"].eq("memory_anomalies") & clean["supported"], "case_id"])
    if len(clean_memory) != 30:
        raise ValueError("clean supported memory cohort drift")
    root = arms["historical"]
    origin = int(json.loads(manifest_path.read_text())["blocks"]["fit"]["start_ms"])
    onset_bin = (root["gt_start_ms"].astype("int64") - origin) // 30000
    multiplicity = onset_bin.map(onset_bin.value_counts())
    groups = {
        "all": ids,
        "clean_supported_memory": clean_memory,
        "all_memory": set(root.index[root["fault_type"].eq("memory_anomalies")]),
        "long_gt300s": set(root.index[(root["gt_end_ms"] - root["gt_start_ms"]) > 300000]),
        "login_failure": set(root.index[root["fault_type"].eq("login_failure")]),
        "single_onset_bin": set(root.index[multiplicity == 1]),
        "multi_onset_bin": set(root.index[multiplicity > 1]),
    }
    paired = {name: paired_group(group, arms) for name, group in groups.items()}
    days = pd.to_datetime(root["gt_start_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    by_day = {day: paired_group(set(root.index[days == day]), arms) for day in sorted(days.unique())}
    clean_group = paired["clean_supported_memory"]
    weighted, control = scores["weighted"], scores["gpu_control"]
    gates = {
        "clean_memory_gain_at_least_5_vs_gpu": len(clean_group["weighted_vs_gpu_gained"]) >= 5,
        "clean_memory_loss_zero_vs_gpu": len(clean_group["weighted_vs_gpu_lost"]) == 0,
        "tp_gain_at_least_10_vs_gpu": weighted["tp"] - control["tp"] >= 10,
        "f1_at_least_gpu": weighted["f1"] >= control["f1"],
        "precision_at_least_0_98": weighted["precision"] >= 0.98,
        "tp_at_least_2134": weighted["tp"] >= 2134,
        "f1_at_least_historical": weighted["f1"] >= 4248 / 5038,
    }
    sources = {str(path): sha256_file(path) for path in
               list(paths.values()) + [clean_path, manifest_path] + list(selection_paths.values())}
    return {
        "schema_version": "p6_c0_long_weight_comparison_v1",
        "status": "GO_TO_EXPLORATORY_TEST" if all(gates.values()) else "NO_GO_DEVELOPMENT",
        "test_inference_run": False, "source_sha256": sources,
        "selections": {name: {"epoch": int(row["selected_epoch"]),
                              "threshold": float(row["selected_validation_threshold"]),
                              "checkpoint_sha256": row["checkpoint"]["sha256"]}
                       for name, row in selections.items()},
        "fit_weight_stats": weight_stats,
        "metrics": scores, "paired": paired, "utc_onset_days": by_day, "gates": gates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("historical-matching", "control-matching", "weighted-matching",
                 "clean-events", "split-manifest", "control-selection",
                 "weighted-selection", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(
        {"historical": args.historical_matching, "gpu_control": args.control_matching,
         "weighted": args.weighted_matching}, args.clean_events, args.split_manifest,
        {"gpu_control": args.control_selection, "weighted": args.weighted_selection},
    )
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"],
                      "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
