#!/usr/bin/env python3
"""Train-only forward ablation of anchor-sensitive Z2 feature groups."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from src.e2e.c1_xgb_adapter import MODEL_PARAMETERS
from src.e2e.rca_model import GAIA_SERVICES, rank_candidates

SOURCE = Path("/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2")
COHORT = SOURCE / "experiments/p6/c1_supervision_oos/c1-supervision-oos-v1-seed42/train_cohort"
BASELINE = ROOT / "experiments/p6/c1_xgb_forward_dev/c1-xgb-forward-v1/rankings.csv"
OUTPUT = ROOT / "experiments/p6/c1_feature_ablation_dev/c1-feature-ablation-forward-v1"
FOLDS = (("f1_to_f2", (1,), 2), ("f12_to_f3", (1, 2), 3))


def masks() -> dict:
    all_ = np.arange(68)
    groups = {"metric": all_[:17], "log": all_[17:34],
              "trace_error": all_[34:51], "trace_latency": all_[51:68]}
    result = {"original": all_}
    result.update({"no_" + name: np.setdiff1d(all_, group) for name, group in groups.items()})
    result["base_only"] = np.asarray([17 * c + j for c in range(4) for j in range(8)])
    result["morphology_only"] = np.asarray([17 * c + j for c in range(4) for j in range(8, 17)])
    result["no_metric_magnitude"] = np.setdiff1d(all_, np.arange(3))
    result["no_all_magnitude"] = np.setdiff1d(all_, np.asarray(
        [17 * c + j for c in range(4) for j in range(3)]))
    result["metric_morphology_only"] = np.setdiff1d(all_, np.arange(8))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(ranks):
    return {"n": int(len(ranks)), "correct_at_1": int(np.sum(ranks == 1)),
            "AC@1": float(np.mean(ranks == 1)), "AC@3": float(np.mean(ranks <= 3)),
            "AC@5": float(np.mean(ranks <= 5)), "MRR": float(np.mean(1.0 / ranks))}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    cases = pd.read_csv(COHORT / "cases.csv", keep_default_na=False)
    x = np.load(COHORT / "detected_features.npy", allow_pickle=False)
    y = np.load(COHORT / "root_indices.npy", allow_pickle=False)
    baseline = pd.read_csv(BASELINE, keep_default_na=False)
    expected = cases.loc[cases.fold.isin((2, 3))].reset_index(drop=True)
    if (len(cases) != 3225 or x.shape != (3225, 10, 68) or y.shape != (3225,)
            or baseline.case_id.tolist() != expected.case_id.tolist()
            or baseline.fold.tolist() != expected.fold.tolist()
            or [GAIA_SERVICES[int(root)] for root in y] != cases.root_service.tolist()):
        raise ValueError("Train cohort/baseline drift")
    OUTPUT.mkdir(parents=True)
    results = {}
    for name, train_folds, eval_fold in FOLDS:
        train = np.flatnonzero(cases.fold.isin(train_folds).to_numpy())
        evaluate = np.flatnonzero(cases.fold.eq(eval_fold).to_numpy())
        if cases.iloc[train].detected_anchor_ms.max() >= cases.iloc[evaluate].detected_anchor_ms.min():
            raise ValueError("chronology drift")
        eval_cases = cases.iloc[evaluate].reset_index(drop=True)
        eval_base = baseline.loc[baseline.fold.eq(eval_fold)].reset_index(drop=True)
        base_ranks = np.asarray([json.loads(row).index(root) + 1 for row, root in
                                 zip(eval_base.ranking_xgb, eval_cases.root_service)])
        fold_results = {"train_n": len(train), "eval_n": len(evaluate),
                        "baseline": metrics(base_ranks), "variants": {}}
        for variant, keep in masks().items():
            train_values = np.ascontiguousarray(x[train][:, :, keep].reshape(-1, len(keep)))
            eval_values = np.ascontiguousarray(x[evaluate][:, :, keep].reshape(-1, len(keep)))
            target = np.zeros((len(train), 10), dtype=np.float32)
            target[np.arange(len(train)), y[train]] = 1.0
            model = xgb.XGBRanker(**MODEL_PARAMETERS)
            model.fit(train_values, target.reshape(-1), group=np.full(len(train), 10), verbose=False)
            scores = model.predict(eval_values).reshape(-1, 10)
            rankings = [rank_candidates(GAIA_SERVICES, row) for row in scores]
            ranks = np.asarray([row.index(root) + 1 for row, root in
                                zip(rankings, eval_cases.root_service)])
            if variant == "original" and not np.array_equal(ranks, base_ranks):
                raise ValueError("original ranker failed baseline replay")
            fold_results["variants"][variant] = {
                "features": keep.tolist(), "all": metrics(ranks),
                "fixed_errors": int(np.sum((ranks == 1) & (base_ranks != 1))),
                "new_errors": int(np.sum((ranks != 1) & (base_ranks == 1))),
            }
        results[name] = fold_results
    report = {"status": "TRAIN_ONLY_FORWARD_DEVELOPMENT", "test_read": False,
              "folds": results, "model_parameters": MODEL_PARAMETERS,
              "input_sha256": {str(path): sha256(path) for path in
                               (COHORT / "cases.csv", COHORT / "detected_features.npy",
                                COHORT / "root_indices.npy", BASELINE)},
              "source_sha256": sha256(Path(__file__))}
    with (OUTPUT / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({fold: {variant: item["all"]["AC@1"] for variant, item in
                             row["variants"].items()} for fold, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
