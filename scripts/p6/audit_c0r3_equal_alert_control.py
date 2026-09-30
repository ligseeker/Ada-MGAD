#!/usr/bin/env python3
"""Fit/Validation-only equal-alert control for the C0R2 rising-logit decoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.p6.audit_c0r2_trainval_dev import (
    C0F, INPUTS, THRESHOLD, _candidate, _load, _pairs,
    match_by_time_component, sha256,
)
from src.e2e.event_detection import construct_predicted_episodes, event_metrics

OUTPUT = ROOT / "experiments/p6/c0r3_equal_alert_control/c0r3-v1"
PRIOR = ROOT / "experiments/p6/c0r2_trainval_dev/positive-logit-rise-v2/results.json"
PRIOR_SHA = "773f05832dcefb568cc867f3eb716c3db2be234cf57d539ede7ca6a59b5dc9f0"
SEEDS = tuple(range(20260930, 20260994))
TOLERANCE_MS = 60000
SOURCE_FILES = (
    "docs/P6_C0R3_EQUAL_ALERT_CONTROL_PROTOCOL.md",
    "scripts/p6/audit_c0r3_equal_alert_control.py",
    "scripts/p6/audit_c0r2_trainval_dev.py",
    "src/e2e/event_detection.py",
)


def positive_runs(scores: pd.DataFrame):
    positive = scores["score"].to_numpy(dtype=float) >= THRESHOLD
    if not np.isfinite(scores["score"].to_numpy(dtype=float)).all():
        raise ValueError("non-finite score")
    starts = np.flatnonzero(positive & ~np.r_[False, positive[:-1]])
    ends = np.flatnonzero(positive & ~np.r_[positive[1:], False]) + 1
    if len(starts) != len(ends):
        raise ValueError("positive-run geometry mismatch")
    return tuple((int(a), int(b)) for a, b in zip(starts, ends))


def candidate_extra_by_length(scores: pd.DataFrame):
    logits = scores["logit"].to_numpy(dtype=float)
    if not np.isfinite(logits).all():
        raise ValueError("non-finite logit")
    runs = positive_runs(scores)
    eligible = {}
    chosen = {}
    starts = []
    for a, b in runs:
        starts.append(a)
        if b - a <= 1:
            continue
        length = b - a
        eligible.setdefault(length, []).extend(range(a + 1, b))
        chosen.setdefault(length, []).extend(
            position for position in range(a + 1, b)
            if logits[position] > logits[position - 1])
    return (np.asarray(starts, dtype=np.int64),
            {k: np.asarray(v, dtype=np.int64) for k, v in eligible.items()},
            {k: np.asarray(v, dtype=np.int64) for k, v in chosen.items()})


def placebo_anchors(starts, eligible, chosen, seed: int):
    """Same extra-alert count in each run-length stratum, random positions."""
    rng = np.random.default_rng(int(seed))
    sampled = []
    for length in sorted(eligible):
        positions = eligible[length]
        count = len(chosen.get(length, ()))
        if count > len(positions):
            raise ValueError("alert budget exceeds eligible positive bins")
        if count:
            sampled.extend(rng.choice(positions, size=count, replace=False).tolist())
    anchors = np.sort(np.concatenate((starts, np.asarray(sampled, dtype=np.int64))))
    if len(anchors) != len(starts) + sum(len(v) for v in chosen.values()):
        raise ValueError("placebo alert-count drift")
    return anchors


def alerts(scores: pd.DataFrame, split: str, anchors, kind: str):
    times = scores["prediction_available_time"].to_numpy(dtype=np.int64)
    values = scores["score"].to_numpy(dtype=float)
    idx = np.asarray(anchors, dtype=np.int64)
    if (len(idx) == 0 or len(np.unique(idx)) != len(idx)
            or np.any(idx < 0) or np.any(idx >= len(scores))
            or np.any(values[idx] < THRESHOLD)):
        raise ValueError("invalid positive alert anchors")
    return pd.DataFrame({
        "prediction_id": ["{}-{}-pred-{:06d}".format(split, kind, i) for i in range(len(idx))],
        "split": split, "t_hat": times[idx],
        "episode_end_time": times[idx] + 30000,
        "positive_bins": 1, "system_score": values[idx], "threshold": THRESHOLD,
    })


def context_flags(gt: pd.DataFrame):
    starts = gt["start_ms"].to_numpy(dtype=np.int64)
    ends = gt["end_ms"].to_numpy(dtype=np.int64)
    sorted_starts = np.sort(starts)
    sorted_ends = np.sort(ends)
    onset_neighbors = (np.searchsorted(sorted_starts, starts + TOLERANCE_MS, side="right")
                       - np.searchsorted(sorted_starts, starts - TOLERANCE_MS, side="left") - 1)
    interval_neighbors = (np.searchsorted(sorted_starts, starts + TOLERANCE_MS, side="right")
                          - np.searchsorted(sorted_ends, starts - TOLERANCE_MS, side="right") - 1)
    if np.any(onset_neighbors < 0) or np.any(interval_neighbors < 0):
        raise ValueError("GT context counters invalid")
    return {"all": np.ones(len(gt), dtype=bool),
            "isolated_onset": onset_neighbors == 0,
            "clean_context": interval_neighbors == 0}


def matched_cases(matching: pd.DataFrame):
    return set(matching.loc[matching.match_status.eq("matched"), "case_id"].astype(str))


def group_recall(gt: pd.DataFrame, flags, cases):
    ids = gt.case_id.astype(str).to_numpy()
    return {group: {"n": int(mask.sum()),
                    "matched": int(sum(case_id in cases for case_id in ids[mask])),
                    "recall": (float(sum(case_id in cases for case_id in ids[mask]) / mask.sum())
                               if mask.sum() else None)}
            for group, mask in flags.items()}


def score_response(scores: pd.DataFrame, gt: pd.DataFrame, flags, base_cases):
    """Current-score response within each causal 60s interval; no model inference."""
    times = scores.prediction_available_time.to_numpy(dtype=np.int64)
    values = scores.score.to_numpy(dtype=float)
    starts = gt.start_ms.to_numpy(dtype=np.int64)
    left = np.searchsorted(times, starts, side="left")
    right = np.searchsorted(times, starts + TOLERANCE_MS, side="right")
    max_score = np.asarray([np.max(values[a:b]) if a < b else np.nan
                            for a, b in zip(left, right)], dtype=float)
    eligible = right > left
    positive = eligible & (max_score >= THRESHOLD)
    aligned = gt[["case_id", "fault_type"]].copy()
    duration_seconds = ((gt.end_ms.to_numpy(dtype=np.int64) - starts) / 1000.0)
    aligned["duration_stratum"] = np.select(
        (duration_seconds <= 15, duration_seconds <= 30, duration_seconds <= 60,
         duration_seconds <= 300),
        ("le_15s", "15_30s", "30_60s", "60_300s"), default="gt_300s")
    aligned["baseline_matched"] = aligned.case_id.astype(str).isin(base_cases)
    aligned["eligible"] = eligible
    aligned["positive"] = positive
    aligned["max_score"] = max_score
    aligned["clean_context"] = flags["clean_context"]
    summary = {}
    for column in ("fault_type", "duration_stratum", "clean_context"):
        rows = {}
        for key, subset in aligned.groupby(column, dropna=False, sort=True):
            finite = subset.max_score.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            rows[str(key)] = {
                "n": int(len(subset)), "eligible": int(subset.eligible.sum()),
                "causal_positive": int(subset.positive.sum()),
                "baseline_matched": int(subset.baseline_matched.sum()),
                "max_score_q10_q50_q90": ([float(v) for v in np.quantile(finite, (0.1, 0.5, 0.9))]
                                         if len(finite) else None),
            }
        summary[column] = rows
    for fault in ("login_failure", "memory_anomalies"):
        subset = aligned.loc[aligned.fault_type.eq(fault) & aligned.clean_context]
        finite = subset.max_score.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        summary.setdefault("clean_fault", {})[fault] = {
            "n": int(len(subset)), "eligible": int(subset.eligible.sum()),
            "causal_positive": int(subset.positive.sum()),
            "baseline_matched": int(subset.baseline_matched.sum()),
            "max_score_q10_q50_q90": ([float(v) for v in np.quantile(finite, (0.1, 0.5, 0.9))]
                                     if len(finite) else None),
        }
    return summary


def distribution(values):
    arr = np.asarray(values, dtype=float)
    return {"min": float(np.min(arr)), "p05": float(np.quantile(arr, 0.05)),
            "median": float(np.median(arr)), "p95": float(np.quantile(arr, 0.95)),
            "max": float(np.max(arr))}


def audit_split(split: str, prior):
    scores, observed, gt = _load(split)
    gt = gt.reset_index(drop=True)
    flags = context_flags(gt)
    baseline_episodes = construct_predicted_episodes(
        scores.rename(columns={"score": "system_score"}), THRESHOLD)
    baseline = match_by_time_component(baseline_episodes, gt)
    base_metrics = event_metrics(baseline)
    if (_pairs(baseline) != _pairs(observed)
            or [base_metrics[k] for k in ("true_positive_events", "false_positive_events", "false_negative_events")]
            != [INPUTS[split][k] for k in ("baseline_tp", "baseline_fp", "baseline_fn")]):
        raise ValueError("archived baseline exact replay failed")
    candidate_episodes = _candidate(scores, split)
    candidate = match_by_time_component(candidate_episodes, gt)
    candidate_metrics = event_metrics(candidate)
    prior_candidate = prior["split_results"][split]
    for key in ("true_positive_events", "false_positive_events", "false_negative_events"):
        if candidate_metrics[key] != prior_candidate["candidate"][key]:
            raise ValueError("C0R2 candidate metric replay failed")
    if len(candidate_episodes) != prior_candidate["emission"]["candidate_episodes"]:
        raise ValueError("C0R2 candidate alert-count replay failed")
    starts, eligible, chosen = candidate_extra_by_length(scores)
    if (len(starts) != len(baseline_episodes)
            or sum(len(v) for v in chosen.values()) + len(starts) != len(candidate_episodes)):
        raise ValueError("candidate run-length alert budget failed")
    base_cases = matched_cases(baseline)
    candidate_cases = matched_cases(candidate)
    control_rows = []
    for seed in SEEDS:
        anchors = placebo_anchors(starts, eligible, chosen, seed)
        control_alerts = alerts(scores, split, anchors, "placebo" + str(seed))
        matching = match_by_time_component(control_alerts, gt)
        metrics = event_metrics(matching)
        cases = matched_cases(matching)
        control_rows.append({
            "seed": seed,
            "tp": int(metrics["true_positive_events"]),
            "fp": int(metrics["false_positive_events"]),
            "fn": int(metrics["false_negative_events"]),
            "precision": float(metrics["event_precision"]),
            "recall": float(metrics["event_recall"]),
            "f1": float(metrics["event_f1"]),
            "group_recall": group_recall(gt, flags, cases),
            "gained_vs_baseline": len(cases - base_cases),
            "lost_vs_baseline": len(base_cases - cases),
        })
    summary = {}
    for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
        summary[key] = distribution([row[key] for row in control_rows])
    for group in flags:
        summary[group + "_recall"] = distribution([
            row["group_recall"][group]["recall"] for row in control_rows])
    candidate_groups = group_recall(gt, flags, candidate_cases)
    go = (candidate_metrics["event_f1"] > summary["f1"]["p95"]
          and candidate_groups["clean_context"]["recall"] > summary["clean_context_recall"]["p95"])
    return {
        "split": split, "gt_n": len(gt),
        "baseline": {"metrics": base_metrics, "group_recall": group_recall(gt, flags, base_cases)},
        "candidate": {"metrics": candidate_metrics, "group_recall": candidate_groups,
                      "gained_vs_baseline": len(candidate_cases - base_cases),
                      "lost_vs_baseline": len(base_cases - candidate_cases)},
        "run_length_budget": {str(k): {"eligible_extra_bins": len(v),
                                         "candidate_extra_alerts": len(chosen.get(k, ())) }
                              for k, v in sorted(eligible.items())},
        "candidate_extra_alerts": sum(len(v) for v in chosen.values()),
        "placebo": {"replicates": control_rows, "distribution": summary},
        "development_gate_pass": bool(go),
        "score_response": score_response(scores, gt, flags, base_cases),
        "baseline_exact_pair_replay": "PASS",
        "candidate_metric_and_alert_count_replay": "PASS",
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError("never overwrite a C0R3 audit directory")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)).strip():
        raise ValueError("commit protocol/source before running")
    if sha256(PRIOR) != PRIOR_SHA:
        raise ValueError("C0R2 result changed")
    prior = json.loads(PRIOR.read_text())
    result = {"schema_version": "p6_c0r3_equal_alert_control_v1",
              "status": "COMPLETE_DEVELOPMENT_ONLY", "test_read": False,
              "historical_fit_validation_score_source": "UNVERIFIED",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=str(ROOT), universal_newlines=True).strip(),
              "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
              "input_sha256": {"c0r2_result": PRIOR_SHA,
                               **{split + "_scores": INPUTS[split]["scores"] for split in INPUTS},
                               **{split + "_observed_matching": INPUTS[split]["matching"] for split in INPUTS}},
              "environment": {"python": platform.python_version(),
                              "numpy": np.__version__, "pandas": pd.__version__},
              "placebo_seeds": list(SEEDS),
              "split_results": {split: audit_split(split, prior)
                                for split in ("fit", "validation")}}
    result["development_gate_pass"] = all(
        item["development_gate_pass"] for item in result["split_results"].values())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / "results.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print("COMPLETE_DEVELOPMENT_ONLY", "GO" if result["development_gate_pass"] else "NO_GO")


if __name__ == "__main__":
    main()
