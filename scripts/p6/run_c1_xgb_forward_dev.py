#!/usr/bin/env python3
"""Train-only chronological C1 scorer sensitivity; no Test inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.e2e.c1_xgb_adapter import MODEL_PARAMETERS, grouped_training_data
from src.e2e.rca_model import (FEATURE_DIMENSION, GAIA_SERVICES, L2_LAMBDA,
                                fit_conditional_logit, rank_candidates,
                                save_conditional_logit)


SOURCE = Path("/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2")
COHORT = SOURCE / "experiments/p6/c1_supervision_oos/c1-supervision-oos-v1-seed42/train_cohort"
OUTPUT = ROOT / "experiments/p6/c1_xgb_forward_dev/c1-xgb-forward-v1"
SOURCE_SHA = {
    "cases.csv": "83ee698981272bfb1f89d1c2d9b1c2c622b63c419a08c251abde2bf7965821dd",
    "gt_features.npy": "fdfbdb12eb862bbfa3b93ea6d6e69c503e89b7fec0075577885a3ee7cda9ec68",
    "detected_features.npy": "b1c043b3ba1480b24a40ddb10646e4f9c7f8cd669de61498be03b7c124f6b4ed",
    "root_indices.npy": "c764da9739c7b80a38ba377462eae489ec43984c53c1034da33feb06a5d2283f",
    "completion_manifest.json": "b36dfaa5ba38bc741e318d8cb9e59e5f225324ff9cd3422b6040faff04634550",
}
SOURCE_FILES = (
    "docs/P6_C1_XGB_FORWARD_DEV_PROTOCOL.md",
    "scripts/p6/run_c1_xgb_forward_dev.py",
    "src/e2e/c1_xgb_adapter.py",
    "src/e2e/rca_model.py",
)
EXPECTED_ENV = {"python": "3.8.20", "xgboost": "2.1.4", "numpy": "1.24.1",
                "pandas": "1.5.3", "scipy": "1.10.1", "scikit_learn": "1.2.1"}
FOLDS = (("f1_to_f2", (1,), 2, 935, 1168),
         ("f12_to_f3", (1, 2), 3, 2103, 1122))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")


def environment():
    actual = {"python": platform.python_version(), "xgboost": xgb.__version__,
              "numpy": np.__version__, "pandas": pd.__version__,
              "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    if actual != EXPECTED_ENV:
        raise ValueError("pinned environment mismatch: " + repr(actual))
    return actual


def source_hashes():
    return {relative: sha256(ROOT / relative) for relative in SOURCE_FILES}


def input_hashes():
    actual = {name: sha256(COHORT / name) for name in SOURCE_SHA}
    if actual != SOURCE_SHA:
        raise ValueError("sealed Train cohort changed")
    manifest = json.loads((COHORT / "completion_manifest.json").read_text())
    if manifest.get("status") != "COMPLETE" or not manifest["details"].get("floors_pass"):
        raise ValueError("source cohort incomplete")
    for name in ("cases.csv", "gt_features.npy", "detected_features.npy", "root_indices.npy"):
        if manifest["files"][name]["sha256"] != actual[name]:
            raise ValueError("source cohort manifest mismatch: " + name)
    return actual


def public_cases():
    frame = pd.read_csv(COHORT / "cases.csv",
                        usecols=["case_id", "fold", "detected_anchor_ms"])
    if (len(frame) != 3225 or frame.case_id.isna().any() or frame.case_id.duplicated().any()
            or frame.fold.value_counts().to_dict() != {1: 935, 2: 1168, 3: 1122}):
        raise ValueError("cohort case/fold alignment failed")
    if frame.detected_anchor_ms.isna().any():
        raise ValueError("missing detected anchor")
    for _, earlier, later, train_n, eval_n in FOLDS:
        train = frame.fold.isin(earlier)
        evaluate = frame.fold.eq(later)
        if (int(train.sum()) != train_n or int(evaluate.sum()) != eval_n
                or int(frame.loc[train, "detected_anchor_ms"].max())
                >= int(frame.loc[evaluate, "detected_anchor_ms"].min())):
            raise ValueError("forward fold chronology/count failed")
    return frame


def preflight():
    environment()
    input_hashes()
    frame = public_cases()
    return {"train_cases": len(frame), "fold_counts": {str(k): int(v) for k, v in
                                               frame.fold.value_counts().sort_index().items()},
            "chronology": "strict earlier-fold max anchor < later-fold min anchor"}


def rank_rows(scores: np.ndarray):
    if (scores.ndim != 2 or scores.shape[1] != len(GAIA_SERVICES)
            or not np.isfinite(scores).all()):
        raise ValueError("invalid scorer output")
    return [json.dumps(rank_candidates(GAIA_SERVICES, row), separators=(",", ":"))
            for row in scores]


def check_lock():
    lock = json.loads((OUTPUT / "run_lock.json").read_text())
    if (lock.get("status") != "FIT_PREDICT_COMPLETE"
            or lock.get("input_sha256") != input_hashes()
            or lock.get("source_sha256") != source_hashes()
            or lock.get("environment") != environment()
            or lock.get("eval_labels_decoded_or_used_in_fit_predict") is not False):
        raise ValueError("forward prediction lock drift")
    if sha256(OUTPUT / "rankings.csv") != lock["rankings_sha256"]:
        raise ValueError("forward rankings changed after prediction lock")
    for name, expected in lock["model_sha256"].items():
        if sha256(OUTPUT / name) != expected:
            raise ValueError("forward model state changed: " + name)
    return lock


def fit_predict():
    audit = preflight()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)).strip():
        raise ValueError("commit protocol and code before fit-predict")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    cases = public_cases()
    # The mmap is indexed only at earlier-fold positions. Later-fold roots and
    # case.csv root_service/fault_type fields are not read until evaluate.
    roots = np.load(COHORT / "root_indices.npy", mmap_mode="r", allow_pickle=False)
    gt = np.load(COHORT / "gt_features.npy", mmap_mode="r", allow_pickle=False)
    detected = np.load(COHORT / "detected_features.npy", mmap_mode="r", allow_pickle=False)
    if (roots.shape != (len(cases),) or gt.shape != (len(cases), 10, FEATURE_DIMENSION)
            or detected.shape != gt.shape):
        raise ValueError("sealed Train tensor shapes changed")
    rows = []
    model_sha = {}
    fit_summaries = {}
    for name, earlier, later, train_n, eval_n in FOLDS:
        train_idx = np.flatnonzero(cases.fold.isin(earlier).to_numpy())
        eval_idx = np.flatnonzero(cases.fold.eq(later).to_numpy())
        train_roots = np.asarray(roots[train_idx], dtype=np.int64)
        train_gt = np.asarray(gt[train_idx], dtype=np.float64)
        train_detected = np.asarray(detected[train_idx], dtype=np.float64)
        eval_detected = np.asarray(detected[eval_idx], dtype=np.float64)
        if (len(train_idx) != train_n or len(eval_idx) != eval_n
                or np.any(train_roots < 0) or np.any(train_roots >= 10)
                or not all(np.isfinite(x).all() for x in (train_gt, train_detected, eval_detected))):
            raise ValueError("invalid forward feature/label slice")
        scaler = StandardScaler().fit(train_gt.reshape(-1, FEATURE_DIMENSION))
        cl = fit_conditional_logit(train_detected, train_roots,
                                   scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
                                   scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
                                   l2_lambda=L2_LAMBDA)
        if not cl.converged:
            raise ValueError("Conditional Logit did not converge: " + name)
        cl_path = OUTPUT / (name + "_cl.npz")
        save_conditional_logit(cl_path, cl)
        xgb_model = xgb.XGBRanker(**MODEL_PARAMETERS)
        matrix, target, group = grouped_training_data(train_detected, train_roots)
        xgb_model.fit(matrix, target, group=group, verbose=False)
        xgb_path = OUTPUT / (name + "_xgb.ubj")
        xgb_model.save_model(str(xgb_path))
        cl_scores = cl.scores(eval_detected)
        xgb_scores = np.asarray(xgb_model.predict(
            np.ascontiguousarray(eval_detected.reshape(-1, FEATURE_DIMENSION), dtype=np.float32)
        ), dtype=np.float64).reshape(-1, 10)
        cl_rankings, xgb_rankings = rank_rows(cl_scores), rank_rows(xgb_scores)
        for idx, cl_rank, xgb_rank in zip(eval_idx, cl_rankings, xgb_rankings):
            rows.append({"case_id": str(cases.iloc[idx].case_id), "fold": later,
                         "detected_anchor_ms": int(cases.iloc[idx].detected_anchor_ms),
                         "ranking_cl": cl_rank, "ranking_xgb": xgb_rank})
        for path in (cl_path, cl_path.with_suffix(".json"), xgb_path):
            model_sha[path.name] = sha256(path)
        fit_summaries[name] = {"train_n": train_n, "eval_n": eval_n,
                               "train_folds": list(earlier), "eval_fold": later,
                               "cl_converged": cl.converged,
                               "cl_gradient_norm": cl.gradient_norm,
                               "model_parameters": MODEL_PARAMETERS}
    rankings = pd.DataFrame(rows)
    rankings.to_csv(OUTPUT / "rankings.csv", index=False)
    write_new(OUTPUT / "run_lock.json", {
        "schema": "p6_c1_xgb_forward_dev_v1", "status": "FIT_PREDICT_COMPLETE",
        "evidence_grade": "Train-only temporal development; post hoc to reused Test",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                            universal_newlines=True).strip(),
        "source_sha256": source_hashes(), "input_sha256": input_hashes(),
        "environment": environment(), "preflight": audit,
        "fit_summaries": fit_summaries, "model_sha256": model_sha,
        "rankings_sha256": sha256(OUTPUT / "rankings.csv"),
        "eval_labels_decoded_or_used_in_fit_predict": False, "test_read": False,
    })
    check_lock()
    print("FIT_PREDICT_COMPLETE", OUTPUT)


def summarize(frame: pd.DataFrame):
    n = len(frame)
    out = {"n": n, "arms": {}}
    for arm in ("cl", "xgb"):
        ranks = frame["rank_" + arm].to_numpy(dtype=np.int64)
        out["arms"][arm] = {"AC@1": float(np.sum(ranks == 1) / n),
                            "AC@3": float(np.sum(ranks <= 3) / n),
                            "AC@5": float(np.sum(ranks <= 5) / n),
                            "MRR": float(np.sum(1.0 / ranks) / n),
                            "correct_at_1": int(np.sum(ranks == 1))}
    cl = frame.rank_cl.eq(1)
    xgb_ = frame.rank_xgb.eq(1)
    out["paired_top1"] = {"both_correct": int((cl & xgb_).sum()),
                          "cl_only": int((cl & ~xgb_).sum()),
                          "xgb_only": int((~cl & xgb_).sum()),
                          "both_incorrect": int((~cl & ~xgb_).sum())}
    out["delta_ac1_xgb_minus_cl"] = (out["paired_top1"]["xgb_only"]
                                      - out["paired_top1"]["cl_only"]) / n
    return out


def evaluate():
    lock = check_lock()
    if (OUTPUT / "results.json").exists():
        raise FileExistsError("evaluation output already exists")
    pred = pd.read_csv(OUTPUT / "rankings.csv", keep_default_na=False)
    cases = pd.read_csv(COHORT / "cases.csv")
    roots = np.load(COHORT / "root_indices.npy", allow_pickle=False)
    expected = cases.loc[cases.fold.isin((2, 3))].copy()
    if (len(pred) != 2290 or len(expected) != len(pred)
            or pred.case_id.tolist() != expected.case_id.tolist()
            or pred.fold.tolist() != expected.fold.tolist()
            or pred.detected_anchor_ms.tolist() != expected.detected_anchor_ms.tolist()
            or not np.array_equal(roots, cases.root_service.map(
                {service: idx for idx, service in enumerate(GAIA_SERVICES)}).to_numpy(dtype=np.int64))):
        raise ValueError("forward prediction universe or root alignment changed")
    expected = expected.reset_index(drop=True)
    for arm in ("cl", "xgb"):
        ranks = []
        for ranking, root in zip(pred["ranking_" + arm], expected.root_service):
            decoded = json.loads(ranking)
            if (not isinstance(decoded, list) or len(decoded) != 10
                    or set(decoded) != set(GAIA_SERVICES)):
                raise ValueError("missing or invalid complete ranking")
            ranks.append(decoded.index(root) + 1)
        expected["rank_" + arm] = ranks
    result = {"status": "COMPLETE_DEVELOPMENT_ONLY", "prediction_lock_sha256": sha256(OUTPUT / "run_lock.json"),
              "source_git_head": lock["git_head"], "evidence_grade": lock["evidence_grade"],
              "folds": {}, "pooled": summarize(expected), "by_fault": {}, "by_service": {},
              "test_read": False}
    for fold in (2, 3):
        result["folds"][str(fold)] = summarize(expected.loc[expected.fold.eq(fold)])
    for column, label in (("fault_type", "by_fault"), ("root_service", "by_service")):
        for key, group in expected.groupby(column, sort=True):
            result[label][str(key)] = summarize(group)
    write_new(OUTPUT / "results.json", result)
    print(json.dumps({"folds": result["folds"], "pooled": result["pooled"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "fit-predict", "evaluate"))
    args = parser.parse_args()
    if args.action == "preflight":
        print(json.dumps(preflight(), indent=2))
    elif args.action == "fit-predict":
        fit_predict()
    else:
        evaluate()


if __name__ == "__main__":
    main()
