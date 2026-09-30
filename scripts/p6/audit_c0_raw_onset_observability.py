#!/usr/bin/env python3
"""Descriptive, no-training raw-input changes at Train event onsets."""

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
import scipy
import sklearn
from sklearn.metrics import roc_auc_score

from scripts.p6.audit_c0r2_trainval_dev import INPUTS, _load, sha256
from scripts.p6.audit_c0r3_equal_alert_control import context_flags

SOURCE = Path("/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2")
ARRAY_DIR = SOURCE / "data/p5/v3_preprocessing_v2/ad/train"
OUTPUT = ROOT / "experiments/p6/c0_raw_onset_observability/raw-onset-v1"
ARRAY_SHA = {
    "timestamps.npy": "ed06402501a887099df16b0353ef4825beda2f24a74d6dae6c38b7d34de91735",
    "metric.npy": "8fb92322f56007f59007b102f6eb07b0cad6496b44b4292f63ced0d797956ee0",
    "log.npy": "02af8ad63423cfaebadc336f122c984a0a6b7c718db4674c62a6ddd40f9ff9dc",
    "trace.npy": "fd49c42223e5dd99420bb5e13432b72c801e6c86e5562620e938599f74d628f2",
}
BLOCKS = {"fit": (1625133600000, 1626440400000),
          "validation": (1626440400000, 1626963120000)}
GRID_MS = 30000
PRE_BINS = 10
POST_MS = 60000
PSEUDO_SEED = 20260930
PSEUDO_N = 512
SOURCE_FILES = (
    "docs/P6_C0_RAW_ONSET_OBSERVABILITY_PROTOCOL.md",
    "scripts/p6/audit_c0_raw_onset_observability.py",
    "scripts/p6/audit_c0r2_trainval_dev.py",
    "scripts/p6/audit_c0r3_equal_alert_control.py",
    "src/e2e/event_detection.py",
)
STAT_NAMES = tuple(name + "_" + statistic
                   for name in ("metric", "log", "trace")
                   for statistic in ("mean_abs", "max_abs"))


def sha_files():
    actual = {name: sha256(ARRAY_DIR / name) for name in ARRAY_SHA}
    if actual != ARRAY_SHA:
        raise ValueError("frozen Train raw-input array drift")
    return actual


def open_arrays():
    arrays = {name: np.load(ARRAY_DIR / (name + ".npy"), mmap_mode="r", allow_pickle=False)
              for name in ("timestamps", "metric", "log", "trace")}
    times = arrays["timestamps"]
    if (len(times) != len(arrays["metric"]) or len(times) != len(arrays["log"])
            or len(times) != len(arrays["trace"])
            or not np.all(np.diff(times) == GRID_MS)
            or arrays["metric"].shape[1:] != (10, 48)
            or arrays["log"].shape[1:] != (10, 32)
            or arrays["trace"].shape[1:] != (10, 10, 8)):
        raise ValueError("frozen raw-input arrays are misaligned")
    return arrays


def slot_indices(bin_end: np.ndarray, onset: int, block):
    """Return ten pre-event bins and legal post-event bins, or None."""
    low, high = block
    pre_right = int(np.searchsorted(bin_end, onset, side="right"))
    pre_left = pre_right - PRE_BINS
    post_left = int(np.searchsorted(bin_end, onset, side="right"))
    post_right = int(np.searchsorted(bin_end, onset + POST_MS, side="right"))
    if (pre_left < 0 or post_left >= post_right or post_right > len(bin_end)
            or onset < low or onset >= high
            or int(bin_end[pre_left] - GRID_MS) < low
            or int(bin_end[post_right - 1]) > high):
        return None
    return pre_left, pre_right, post_left, post_right


def raw_change(arrays, slot):
    a, b, c, d = slot
    result = {}
    for name in ("metric", "log", "trace"):
        values = arrays[name]
        prior = np.asarray(values[a:b], dtype=np.float64)
        after = np.asarray(values[c:d], dtype=np.float64)
        if not np.isfinite(prior).all() or not np.isfinite(after).all():
            raise ValueError("non-finite frozen Train raw-input tensor")
        baseline = np.median(prior, axis=0)
        difference = np.abs(after - baseline)
        flat = difference.reshape(len(after), -1)
        result[name + "_mean_abs"] = float(np.max(flat.mean(axis=1)))
        result[name + "_max_abs"] = float(np.max(flat.max(axis=1)))
    return result


def no_event_intersects(starts, ends, left, right):
    """Count interval intersections with [left,right] from independently sorted ends."""
    return (np.searchsorted(starts, right, side="right")
            - np.searchsorted(ends, left, side="right")) == 0


def quantiles(values):
    arr = np.asarray(values, dtype=float)
    if not len(arr) or not np.isfinite(arr).all():
        return None
    return {"p10": float(np.quantile(arr, 0.1)),
            "median": float(np.median(arr)),
            "p90": float(np.quantile(arr, 0.9))}


def group_summary(events: pd.DataFrame, pseudo: pd.DataFrame):
    groups = {"login_failure": events.loc[events.fault_type.eq("login_failure")],
              "memory_anomalies": events.loc[events.fault_type.eq("memory_anomalies")],
              "other_faults": events.loc[~events.fault_type.isin(("login_failure", "memory_anomalies"))]}
    controls = pseudo
    result = {}
    for name, frame in groups.items():
        valid = frame.loc[frame.supported]
        item = {"clean_cases": int(len(frame)), "raw_supported": int(len(valid)),
                "pseudo_count": int(len(controls)), "statistics": {}}
        for statistic in STAT_NAMES:
            observed = valid[statistic].to_numpy(dtype=float)
            negative = controls[statistic].to_numpy(dtype=float)
            item["statistics"][statistic] = {
                "event_quantiles": quantiles(observed),
                "pseudo_quantiles": quantiles(negative),
                "descriptive_auc": (float(roc_auc_score(
                    np.r_[np.ones(len(observed)), np.zeros(len(negative))],
                    np.r_[observed, negative])) if len(observed) and len(negative) else None),
            }
        result[name] = item
    return result


def analyze_split(split: str, arrays):
    scores, observed, gt = _load(split)
    gt = gt.reset_index(drop=True)
    flags = context_flags(gt)
    block = BLOCKS[split]
    bin_end = np.asarray(arrays["timestamps"], dtype=np.int64) + GRID_MS
    if not (int(scores.prediction_available_time.min()) >= block[0]
            and int(scores.prediction_available_time.max()) <= block[1]):
        raise ValueError("score timestamps exceed frozen detector block")
    event_rows = []
    for row, clean in zip(gt.itertuples(index=False), flags["clean_context"]):
        if not clean:
            continue
        slot = slot_indices(bin_end, int(row.start_ms), block)
        stats = raw_change(arrays, slot) if slot is not None else {name: None for name in STAT_NAMES}
        event_rows.append({"case_id": str(row.case_id), "split": split,
                           "fault_type": str(row.fault_type),
                           "start_ms": int(row.start_ms),
                           "duration_seconds": float((int(row.end_ms) - int(row.start_ms)) / 1000),
                           "supported": slot is not None, **stats})
    events = pd.DataFrame(event_rows)
    if events.case_id.duplicated().any():
        raise ValueError("duplicate clean GT case")
    starts = np.sort(gt.start_ms.to_numpy(dtype=np.int64))
    ends = np.sort(gt.end_ms.to_numpy(dtype=np.int64))
    candidate_times = scores.prediction_available_time.to_numpy(dtype=np.int64) - 15000
    pool = []
    for t in candidate_times:
        t = int(t)
        if (slot_indices(bin_end, t, block) is not None
                and no_event_intersects(starts, ends, t - 300000, t + POST_MS)):
            pool.append(t)
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) < PSEUDO_N:
        raise ValueError("too few clean pseudo-onset times")
    rng = np.random.default_rng(PSEUDO_SEED)
    selected = np.sort(rng.choice(pool, size=PSEUDO_N, replace=False))
    pseudo = pd.DataFrame([{"split": split, "pseudo_onset_ms": int(t),
                            **raw_change(arrays, slot_indices(bin_end, int(t), block))}
                           for t in selected])
    if len(events) != int(flags["clean_context"].sum()) or len(pseudo) != PSEUDO_N:
        raise ValueError("clean case or pseudo-onset denominator changed")
    summary = {"split": split, "all_gt": len(gt),
               "clean_context_gt": len(events), "pseudo_pool": len(pool),
               "pseudo_selected": len(pseudo), "group_summary": group_summary(events, pseudo),
               "source_scores_sha256": INPUTS[split]["scores"],
               "source_observed_matching_sha256": INPUTS[split]["matching"]}
    return events, pseudo, summary


def main():
    if OUTPUT.exists():
        raise FileExistsError("never overwrite a raw onset audit directory")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)).strip():
        raise ValueError("commit protocol and source before raw audit")
    inputs = sha_files()
    arrays = open_arrays()
    outputs = {split: analyze_split(split, arrays) for split in ("fit", "validation")}
    result = {"schema_version": "p6_c0_raw_onset_observability_v1",
              "status": "COMPLETE_DEVELOPMENT_ONLY", "test_read": False,
              "historical_fit_validation_score_source": "UNVERIFIED",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=str(ROOT), universal_newlines=True).strip(),
              "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
              "input_sha256": {"train_arrays": inputs,
                               **{split + "_scores": INPUTS[split]["scores"] for split in INPUTS},
                               **{split + "_observed_matching": INPUTS[split]["matching"] for split in INPUTS}},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "scipy": scipy.__version__,
                              "scikit_learn": sklearn.__version__},
              "split_results": {split: outputs[split][2] for split in outputs}}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for split in ("fit", "validation"):
        outputs[split][0].to_csv(OUTPUT / (split + "_clean_events.csv"), index=False)
        outputs[split][1].to_csv(OUTPUT / (split + "_pseudo_onsets.csv"), index=False)
    result["output_sha256"] = {path.name: sha256(path) for path in sorted(OUTPUT.glob("*.csv"))}
    with (OUTPUT / "results.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
    print("COMPLETE_DEVELOPMENT_ONLY", OUTPUT)


if __name__ == "__main__":
    main()
