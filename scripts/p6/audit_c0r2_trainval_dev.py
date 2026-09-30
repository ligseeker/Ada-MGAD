#!/usr/bin/env python3
"""Exploratory Fit/Validation audit of a fixed positive-logit-rise re-trigger.

The existing C0F score bytes are reused. This script never opens Test files,
trains a model, changes the score threshold, or chooses a candidate by outcome.
Validation was inspected before this audit was frozen; evidence is development
only. The original C0/C0F results remain immutable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.e2e.event_detection import construct_predicted_episodes, event_metrics, match_events

SOURCE = Path("/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2")
C0F = SOURCE / "experiments/p6/c0f_failure_audit/c0f-correction-20260923T0755Z"
OUTPUT = ROOT / "experiments/p6/c0r2_trainval_dev/positive-logit-rise-v2"
INPUTS = {
    "fit": {
        "scores": "140c58508263cd5c5fca280cac284c6d3e52798275eb40b36c8921f9a797687b",
        "matching": "3c3b5239515e4c9cb55aebed87d4aad3c5df125c982cdbf5b1071fabf79a98ba",
        "baseline_tp": 5626, "baseline_fp": 44, "baseline_fn": 1817,
    },
    "validation": {
        "scores": "5b5e592636753ada3615ff4c373ef739660dcc5196908a78bceb9eb85892ce2f",
        "matching": "8fff3e6b303a0ce11b60bcd0ae22cd84389c168d1e9d3cf647c4f36bfe1ada10",
        "baseline_tp": 2124, "baseline_fp": 13, "baseline_fn": 777,
    },
}
THRESHOLD = 0.9998264908790588
TOLERANCE_MS = 60000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(split: str):
    binding = INPUTS[split]
    score_path = C0F / "scores" / (split + ".csv")
    matching_path = C0F / "observed" / split / "matching.csv"
    if sha256(score_path) != binding["scores"] or sha256(matching_path) != binding["matching"]:
        raise ValueError("C0F development score or observed matching drift: " + split)
    scores = pd.read_csv(score_path)
    observed = pd.read_csv(matching_path)
    if (not scores["split"].eq(split).all() or scores["prediction_available_time"].duplicated().any()
            or not np.all(np.diff(scores["prediction_available_time"].to_numpy(dtype=np.int64)) == 30000)
            or not np.allclose(scores["fixed_threshold"].to_numpy(dtype=float), THRESHOLD, rtol=0, atol=0)
            or not np.array_equal(scores["binary_prediction"].to_numpy(dtype=int),
                                  (scores["score"].to_numpy(dtype=float) >= THRESHOLD).astype(int))):
        raise ValueError("C0F fixed threshold, chronology or scores drift: " + split)
    gt_rows = observed.loc[observed["match_status"].isin(("matched", "miss"))].copy()
    if gt_rows["case_id"].duplicated().any():
        raise ValueError("C0F GT case identity drift: " + split)
    gt = gt_rows[["case_id", "source_index", "gt_service", "fault_type",
                  "gt_start_ms", "gt_end_ms"]].rename(columns={
                      "gt_service": "service", "gt_start_ms": "start_ms",
                      "gt_end_ms": "end_ms"})
    for name in ("source_index", "start_ms", "end_ms"):
        gt[name] = gt[name].astype(np.int64)
    return scores, observed, gt


def _candidate(scores: pd.DataFrame, split: str) -> pd.DataFrame:
    """Emit at each positive run start and at a strictly rising positive logit."""
    times = scores["prediction_available_time"].to_numpy(dtype=np.int64)
    values = scores["score"].to_numpy(dtype=float)
    logits = scores["logit"].to_numpy(dtype=float)
    positive = values >= THRESHOLD
    if not np.isfinite(values).all() or not np.isfinite(logits).all():
        raise ValueError("non-finite frozen C0F score")
    starts = positive.copy()
    starts[1:] &= ~positive[:-1]
    rises = np.zeros(len(positive), dtype=bool)
    rises[1:] = positive[1:] & positive[:-1] & (logits[1:] > logits[:-1])
    anchors = np.flatnonzero(starts | rises)
    return pd.DataFrame({
        "prediction_id": ["{}-rise-pred-{:06d}".format(split, i) for i in range(len(anchors))],
        "split": split, "t_hat": times[anchors], "episode_end_time": times[anchors] + 30000,
        "positive_bins": 1, "system_score": values[anchors], "threshold": THRESHOLD,
        "trigger_kind": np.where(starts[anchors], "run_start", "positive_logit_rise"),
    })


def match_by_time_component(predictions: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Run the unchanged exact matcher on components separated by >60 seconds."""
    ptime = predictions["t_hat"].to_numpy(dtype=np.int64)
    gtime = gt["start_ms"].to_numpy(dtype=np.int64)
    times = np.concatenate((ptime, gtime))
    order = np.argsort(times, kind="stable")
    sorted_comp = np.zeros(len(times), dtype=np.int64)
    if len(times) > 1:
        sorted_comp[1:] = np.cumsum(np.diff(times[order]) > TOLERANCE_MS)
    component = np.empty(len(times), dtype=np.int64)
    component[order] = sorted_comp
    pred_groups = predictions.groupby(component[:len(ptime)], sort=True)
    gt_groups = gt.groupby(component[len(ptime):], sort=True)
    pred_index = pred_groups.indices
    gt_index = gt_groups.indices
    frames = []
    for key in sorted(set(pred_index) | set(gt_index)):
        p = predictions.iloc[pred_index[key]] if key in pred_index else predictions.iloc[:0]
        g = gt.iloc[gt_index[key]] if key in gt_index else gt.iloc[:0]
        frames.append(match_events(p, g, tolerance_seconds=60))
    result = pd.concat(frames, ignore_index=True)
    if (result["prediction_id"].notna().sum() != len(predictions)
            or result.loc[result["match_status"].isin(("matched", "miss")),
                          "case_id"].nunique() != len(gt)):
        raise ValueError("component matching failed to preserve prediction or GT universe")
    return result


def _pairs(frame: pd.DataFrame):
    matched = frame.loc[frame["match_status"] == "matched"]
    return {(str(row.case_id), int(row.t_hat)) for row in matched.itertuples(index=False)}


def audit(split: str):
    scores, observed, gt = _load(split)
    score_frame = scores.rename(columns={"score": "system_score"})
    baseline_episodes = construct_predicted_episodes(score_frame, THRESHOLD)
    baseline = match_by_time_component(baseline_episodes, gt)
    base_metrics = event_metrics(baseline)
    expected = INPUTS[split]
    if (base_metrics["true_positive_events"] != expected["baseline_tp"]
            or base_metrics["false_positive_events"] != expected["baseline_fp"]
            or base_metrics["false_negative_events"] != expected["baseline_fn"]
            or _pairs(baseline) != _pairs(observed)):
        raise ValueError("unchanged C0F baseline replay failed: " + split)
    candidate_episodes = _candidate(scores, split)
    candidate = match_by_time_component(candidate_episodes, gt)
    candidate_metrics = event_metrics(candidate)
    baseline_cases = set(baseline.loc[baseline["match_status"] == "matched", "case_id"])
    candidate_cases = set(candidate.loc[candidate["match_status"] == "matched", "case_id"])
    candidate_m = candidate.loc[candidate["match_status"] == "matched"]
    kinds = candidate_m.merge(candidate_episodes[["prediction_id", "trigger_kind"]],
                               on="prediction_id", validate="one_to_one")
    gained = gt.loc[gt["case_id"].isin(candidate_cases - baseline_cases)]
    lost = gt.loc[gt["case_id"].isin(baseline_cases - candidate_cases)]
    return {
        "split": split, "baseline": base_metrics, "candidate": candidate_metrics,
        "paired_case_change": {
            "gained": len(gained), "lost": len(lost),
            "gained_fault_counts": {str(k): int(v) for k, v in gained["fault_type"].value_counts().items()},
            "lost_fault_counts": {str(k): int(v) for k, v in lost["fault_type"].value_counts().items()},
        },
        "emission": {"baseline_episodes": len(baseline_episodes),
                     "candidate_episodes": len(candidate_episodes),
                     "positive_logit_rises": int(candidate_episodes["trigger_kind"].eq("positive_logit_rise").sum()),
                     "matched_by_trigger_kind": {str(k): int(v) for k, v in
                                                 kinds["trigger_kind"].value_counts().items()}},
        "score_sha256": expected["scores"], "observed_matching_sha256": expected["matching"],
        "baseline_exact_pair_replay": "PASS",
    }


def main() -> None:
    if OUTPUT.exists():
        raise ValueError("never overwrite a C0R2 development audit directory")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)).strip():
        raise ValueError("commit the development audit source before execution")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                     universal_newlines=True).strip()
    source_hashes = {
        name: sha256(ROOT / name) for name in (
            "scripts/p6/audit_c0r2_trainval_dev.py",
            "src/e2e/event_detection.py",
            "docs/P6_C0R2_TRAINVAL_DEV_PROTOCOL.md",
        )
    }
    results = {split: audit(split) for split in ("fit", "validation")}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / "results.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "p6_c0r2_trainval_dev_v1",
                   "status": "COMPLETE_DEVELOPMENT_ONLY",
                   "test_read": False, "test_selection": False,
                   "validation_inspected_before_protocol_freeze": True,
                   "historical_fit_validation_score_source": "UNVERIFIED",
                   "candidate": "positive run start plus strict within-run logit rise; fixed C0 threshold",
                   "source_commit": commit, "source_sha256": source_hashes,
                   "split_results": results}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("COMPLETE_DEVELOPMENT_ONLY", OUTPUT)


if __name__ == "__main__":
    main()
