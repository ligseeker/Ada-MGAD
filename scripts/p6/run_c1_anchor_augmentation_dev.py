#!/usr/bin/env python3
"""Train-only forward check of GT/detected anchor augmentation for XGBRanker."""

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

from src.e2e.c1_xgb_adapter import MODEL_PARAMETERS, grouped_training_data
from src.e2e.rca_model import GAIA_SERVICES, rank_candidates

SOURCE = Path("/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2")
COHORT = SOURCE / "experiments/p6/c1_supervision_oos/c1-supervision-oos-v1-seed42/train_cohort"
BASELINE = ROOT / "experiments/p6/c1_xgb_forward_dev/c1-xgb-forward-v1/rankings.csv"
OUTPUT = ROOT / "experiments/p6/c1_anchor_augmentation_dev/c1-anchor-augmentation-forward-v1"
FOLDS = (("f1_to_f2", (1,), 2), ("f12_to_f3", (1, 2), 3))
VARIANTS = ("detected_only", "gt_only", "detected_plus_gt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(ranks):
    return {"n": len(ranks), "correct_at_1": int(np.sum(ranks == 1)),
            "AC@1": float(np.mean(ranks == 1)), "AC@3": float(np.mean(ranks <= 3)),
            "AC@5": float(np.mean(ranks <= 5)), "MRR": float(np.mean(1.0 / ranks))}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    cases = pd.read_csv(COHORT / "cases.csv", keep_default_na=False)
    gt = np.load(COHORT / "gt_features.npy", allow_pickle=False)
    detected = np.load(COHORT / "detected_features.npy", allow_pickle=False)
    roots = np.load(COHORT / "root_indices.npy", allow_pickle=False)
    baseline = pd.read_csv(BASELINE, keep_default_na=False)
    expected = cases.loc[cases.fold.isin((2, 3))].reset_index(drop=True)
    if (len(cases) != 3225 or gt.shape != (3225, 10, 68) or detected.shape != gt.shape
            or roots.shape != (3225,) or baseline.case_id.tolist() != expected.case_id.tolist()
            or baseline.fold.tolist() != expected.fold.tolist()):
        raise ValueError("cohort drift")
    if [GAIA_SERVICES[int(root)] for root in roots] != cases.root_service.tolist():
        raise ValueError("root index drift")
    OUTPUT.mkdir(parents=True)
    results = {}
    predictions = []
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
        for variant in VARIANTS:
            if variant == "detected_only":
                train_features, train_roots = detected[train], roots[train]
            elif variant == "gt_only":
                train_features, train_roots = gt[train], roots[train]
            else:
                train_features = np.concatenate((detected[train], gt[train]))
                train_roots = np.concatenate((roots[train], roots[train]))
            matrix, target, group = grouped_training_data(train_features, train_roots)
            model = xgb.XGBRanker(**MODEL_PARAMETERS)
            model.fit(matrix, target, group=group, verbose=False)
            scores = model.predict(np.ascontiguousarray(detected[evaluate].reshape(-1, 68))).reshape(-1, 10)
            rankings = [rank_candidates(GAIA_SERVICES, row) for row in scores]
            ranks = np.asarray([row.index(root) + 1 for row, root in
                                zip(rankings, eval_cases.root_service)])
            if variant == "detected_only" and not np.array_equal(ranks, base_ranks):
                raise ValueError("baseline failed exact replay")
            fold_results["variants"][variant] = {
                "all": metrics(ranks),
                "fixed_errors": int(np.sum((ranks == 1) & (base_ranks != 1))),
                "new_errors": int(np.sum((ranks != 1) & (base_ranks == 1))),
            }
            for case_id, row, rank, base_rank in zip(eval_cases.case_id, rankings, ranks, base_ranks):
                predictions.append({"case_id": case_id, "fold": eval_fold,
                                    "variant": variant, "ranking": json.dumps(row),
                                    "rank": int(rank), "baseline_rank": int(base_rank)})
        results[name] = fold_results
    pd.DataFrame(predictions).to_csv(OUTPUT / "predictions.csv", index=False)
    report = {"status": "TRAIN_ONLY_FORWARD_DEVELOPMENT", "test_read": False,
              "model_parameters": MODEL_PARAMETERS, "folds": results,
              "input_sha256": {str(path): sha256(path) for path in
                               (COHORT / "cases.csv", COHORT / "gt_features.npy",
                                COHORT / "detected_features.npy", COHORT / "root_indices.npy",
                                BASELINE)}, "source_sha256": sha256(Path(__file__))}
    with (OUTPUT / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({fold: {key: item["all"]["AC@1"] for key, item in
                             row["variants"].items()} for fold, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
