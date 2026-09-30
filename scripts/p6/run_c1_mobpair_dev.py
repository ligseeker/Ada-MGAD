#!/usr/bin/env python3
"""Train-only forward check of within-event mobservice1/2 RCA comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2")
COHORT = SOURCE / "experiments/p6/c1_supervision_oos/c1-supervision-oos-v1-seed42/train_cohort"
BASELINE = ROOT / "experiments/p6/c1_xgb_forward_dev/c1-xgb-forward-v1/rankings.csv"
OUTPUT = ROOT / "experiments/p6/c1_mobpair_dev/c1-mobpair-forward-v1"
MOB1, MOB2 = "mobservice1", "mobservice2"
FOLDS = (("f1_to_f2", (1,), 2), ("f12_to_f3", (1, 2), 3))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def features(values: np.ndarray, kind: str) -> np.ndarray:
    m1, m2 = values[:, 4, :], values[:, 5, :]
    difference = m1 - m2
    if kind == "difference":
        return np.asarray(difference, dtype=np.float32)
    if kind == "difference_sum":
        return np.asarray(np.concatenate((difference, m1 + m2), axis=1), dtype=np.float32)
    if kind == "full":
        return np.asarray(values.reshape(len(values), -1), dtype=np.float32)
    raise ValueError(kind)


def candidate(name: str):
    if name == "logistic_difference":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)), "difference"
    kind = name[4:] if name.startswith("xgb_") else name
    if kind not in {"difference", "difference_sum", "full"}:
        raise ValueError(name)
    return XGBClassifier(objective="binary:logistic", n_estimators=200, max_depth=3,
                         learning_rate=0.05, subsample=1.0, colsample_bytree=1.0,
                         reg_lambda=1.0, random_state=20260826, n_jobs=1,
                         tree_method="hist"), kind


def corrected(ranking: list[str], winner: str) -> list[str]:
    result = list(ranking)
    a, b = result.index(MOB1), result.index(MOB2)
    if result[min(a, b)] != winner:
        result[a], result[b] = result[b], result[a]
    return result


def summary(ranks: np.ndarray) -> dict:
    return {"n": int(len(ranks)), "correct_at_1": int(np.sum(ranks == 1)),
            "AC@1": float(np.mean(ranks == 1)), "AC@3": float(np.mean(ranks <= 3)),
            "AC@5": float(np.mean(ranks <= 5)), "MRR": float(np.mean(1.0 / ranks))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    cases = pd.read_csv(COHORT / "cases.csv", keep_default_na=False)
    values = np.load(COHORT / "detected_features.npy", allow_pickle=False)
    roots = np.load(COHORT / "root_indices.npy", allow_pickle=False)
    baseline = pd.read_csv(BASELINE, keep_default_na=False)
    if (len(cases) != 3225 or values.shape != (len(cases), 10, 68)
            or roots.shape != (len(cases),) or not np.isfinite(values).all()
            or cases.case_id.duplicated().any()):
        raise ValueError("Train cohort invalid")
    expected = cases.loc[cases.fold.isin((2, 3))].reset_index(drop=True)
    if (len(baseline) != len(expected) or baseline.case_id.tolist() != expected.case_id.tolist()
            or baseline.fold.tolist() != expected.fold.tolist()
            or baseline.detected_anchor_ms.tolist() != expected.detected_anchor_ms.tolist()):
        raise ValueError("XGB forward baseline differs from candidate cohort")
    if not np.array_equal((roots == 4), (cases.root_service == MOB1).to_numpy()) or not np.array_equal(
            (roots == 5), (cases.root_service == MOB2).to_numpy()):
        raise ValueError("Root index/service mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    models = ("logistic_difference", "xgb_difference", "xgb_difference_sum", "xgb_full")
    rankings = [json.loads(value) for value in baseline.ranking_xgb]
    if any(len(row) != 10 or set(row) != set(("dbservice1", "dbservice2", "logservice1",
                                          "logservice2", "mobservice1", "mobservice2",
                                          "redisservice1", "redisservice2", "webservice1",
                                          "webservice2")) for row in rankings):
        raise ValueError("Incomplete baseline ranking")
    baseline_ranks = np.asarray([row.index(root) + 1 for row, root in
                                 zip(rankings, expected.root_service)], dtype=int)
    fold_results = {}
    prediction_rows = []
    for fold_name, train_folds, eval_fold in FOLDS:
        train_idx = np.flatnonzero(cases.fold.isin(train_folds).to_numpy() & np.isin(roots, (4, 5)))
        eval_idx = np.flatnonzero(cases.fold.eq(eval_fold).to_numpy())
        if not (cases.iloc[train_idx].detected_anchor_ms.max() <
                cases.iloc[eval_idx].detected_anchor_ms.min()):
            raise ValueError("Forward chronology failed")
        eval_cases = cases.iloc[eval_idx].reset_index(drop=True)
        eval_baseline = baseline.loc[baseline.fold.eq(eval_fold)].reset_index(drop=True)
        eval_ranks = baseline_ranks[baseline.fold.eq(eval_fold).to_numpy()]
        fold_output = {"train_n": int(len(train_idx)), "eval_n": int(len(eval_idx)),
                       "baseline": summary(eval_ranks), "candidates": {}}
        y_train = (roots[train_idx] == 4).astype(int)
        for name in models:
            model, kind = candidate(name)
            x_train = features(values[train_idx], kind)
            x_eval = features(values[eval_idx], kind)
            model.fit(x_train, y_train)
            prob = model.predict_proba(x_eval)[:, 1]
            winner = np.where(prob >= 0.5, MOB1, MOB2)
            changed = [corrected(row, chosen) for row, chosen in
                       zip([json.loads(value) for value in eval_baseline.ranking_xgb], winner)]
            ranks = np.asarray([row.index(root) + 1 for row, root in
                                zip(changed, eval_cases.root_service)], dtype=int)
            mob = eval_cases.root_service.isin((MOB1, MOB2)).to_numpy()
            fold_output["candidates"][name] = {
                "all": summary(ranks), "mob_only": summary(ranks[mob]),
                "delta_correct_at_1": int(np.sum(ranks == 1) - np.sum(eval_ranks == 1)),
                "fixed_errors": int(np.sum((ranks == 1) & (eval_ranks != 1))),
                "new_errors": int(np.sum((ranks != 1) & (eval_ranks == 1))),
            }
            for case_id, p1, rank, base_rank in zip(eval_cases.case_id, prob, ranks, eval_ranks):
                prediction_rows.append({"case_id": case_id, "fold": eval_fold,
                                        "model": name, "p_mobservice1": float(p1),
                                        "rank": int(rank), "baseline_rank": int(base_rank)})
        fold_results[fold_name] = fold_output
    pd.DataFrame(prediction_rows).to_csv(args.output / "predictions.csv", index=False)
    report = {"status": "TRAIN_ONLY_FORWARD_DEVELOPMENT", "folds": fold_results,
              "input_sha256": {str(path): sha256(path) for path in
                               (COHORT / "cases.csv", COHORT / "detected_features.npy",
                                COHORT / "root_indices.npy", BASELINE)},
              "source_sha256": sha256(Path(__file__)),
              "test_read": False,
              "notes": "Post hoc Train development; shared preprocessing schema; Test already reused by predecessor."}
    with (args.output / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps({name: {"baseline": row["baseline"]["AC@1"],
                            "candidates": {model: val["all"]["AC@1"] for model, val in
                                           row["candidates"].items()}}
                      for name, row in fold_results.items()}, indent=2))


if __name__ == "__main__":
    main()
