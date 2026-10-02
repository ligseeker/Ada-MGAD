#!/usr/bin/env python3
"""Train-only forward check of event-relative and identity RCA features."""

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
OUTPUT = ROOT / "experiments/p6/c1_relative_ranker_dev/c1-relative-ranker-forward-v1"
FOLDS = (("f1_to_f2", (1,), 2), ("f12_to_f3", (1, 2), 3))
VARIANTS = ("original", "service_identity", "event_centered", "centered_identity",
            "mob_contrast")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def representation(x: np.ndarray, variant: str) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    parts = [values]
    if variant in ("event_centered", "centered_identity"):
        parts.append(values - values.mean(axis=1, keepdims=True))
    if variant in ("service_identity", "centered_identity"):
        one_hot = np.broadcast_to(np.eye(10, dtype=np.float32), (len(values), 10, 10))
        parts.append(one_hot)
    if variant == "mob_contrast":
        contrast = np.zeros_like(values)
        contrast[:, 4] = values[:, 4] - values[:, 5]
        contrast[:, 5] = -contrast[:, 4]
        parts.append(contrast)
    if variant not in VARIANTS:
        raise ValueError(variant)
    result = np.ascontiguousarray(np.concatenate(parts, axis=2))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite representation")
    return result


def summary(ranks: np.ndarray) -> dict:
    return {"n": len(ranks), "correct_at_1": int(np.sum(ranks == 1)),
            "AC@1": float(np.mean(ranks == 1)), "AC@3": float(np.mean(ranks <= 3)),
            "AC@5": float(np.mean(ranks <= 5)), "MRR": float(np.mean(1.0 / ranks))}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    cases = pd.read_csv(COHORT / "cases.csv", keep_default_na=False)
    x = np.load(COHORT / "detected_features.npy", allow_pickle=False)
    y = np.load(COHORT / "root_indices.npy", allow_pickle=False)
    baseline = pd.read_csv(BASELINE, keep_default_na=False)
    if len(cases) != 3225 or x.shape != (3225, 10, 68) or y.shape != (3225,):
        raise ValueError("Train cohort shape drift")
    expected = cases.loc[cases.fold.isin((2, 3))].reset_index(drop=True)
    if (baseline.case_id.tolist() != expected.case_id.tolist()
            or baseline.fold.tolist() != expected.fold.tolist()
            or baseline.detected_anchor_ms.tolist() != expected.detected_anchor_ms.tolist()):
        raise ValueError("baseline cohort drift")
    if [GAIA_SERVICES[int(root)] for root in y] != cases.root_service.tolist():
        raise ValueError("root index/service drift")
    OUTPUT.mkdir(parents=True)
    results = {}
    prediction_rows = []
    for fold_name, train_folds, eval_fold in FOLDS:
        train = np.flatnonzero(cases.fold.isin(train_folds).to_numpy())
        evaluate = np.flatnonzero(cases.fold.eq(eval_fold).to_numpy())
        if cases.iloc[train].detected_anchor_ms.max() >= cases.iloc[evaluate].detected_anchor_ms.min():
            raise ValueError("chronology drift")
        eval_cases = cases.iloc[evaluate].reset_index(drop=True)
        eval_baseline = baseline.loc[baseline.fold.eq(eval_fold)].reset_index(drop=True)
        base_ranks = np.asarray([json.loads(row).index(root) + 1 for row, root in
                                 zip(eval_baseline.ranking_xgb, eval_cases.root_service)])
        fold_results = {"baseline": summary(base_ranks), "train_n": len(train),
                        "eval_n": len(evaluate), "variants": {}}
        for variant in VARIANTS:
            a, b = representation(x[train], variant), representation(x[evaluate], variant)
            matrix = np.ascontiguousarray(a.reshape(-1, a.shape[2]))
            target = np.zeros((len(train), 10), dtype=np.float32)
            target[np.arange(len(train)), y[train]] = 1.0
            model = xgb.XGBRanker(**MODEL_PARAMETERS)
            model.fit(matrix, target.reshape(-1), group=np.full(len(train), 10), verbose=False)
            scores = model.predict(np.ascontiguousarray(b.reshape(-1, b.shape[2]))).reshape(-1, 10)
            rankings = [rank_candidates(GAIA_SERVICES, row) for row in scores]
            ranks = np.asarray([row.index(root) + 1 for row, root in
                                zip(rankings, eval_cases.root_service)])
            if variant == "original":
                if not np.array_equal(ranks, base_ranks):
                    raise ValueError("original ranker did not replay baseline")
                if [json.dumps(row, separators=(",", ":")) for row in rankings] != eval_baseline.ranking_xgb.tolist():
                    raise ValueError("original ranker rankings differ")
            mob = eval_cases.root_service.isin(("mobservice1", "mobservice2")).to_numpy()
            fold_results["variants"][variant] = {
                "all": summary(ranks), "mob_only": summary(ranks[mob]),
                "fixed_errors": int(np.sum((ranks == 1) & (base_ranks != 1))),
                "new_errors": int(np.sum((ranks != 1) & (base_ranks == 1))),
            }
            for case_id, row, rank, base_rank in zip(eval_cases.case_id, rankings, ranks, base_ranks):
                prediction_rows.append({"case_id": case_id, "fold": eval_fold,
                                        "variant": variant, "ranking": json.dumps(row),
                                        "rank": int(rank), "baseline_rank": int(base_rank)})
        results[fold_name] = fold_results
    pd.DataFrame(prediction_rows).to_csv(OUTPUT / "predictions.csv", index=False)
    report = {"status": "TRAIN_ONLY_FORWARD_DEVELOPMENT", "folds": results,
              "model_parameters": MODEL_PARAMETERS,
              "input_sha256": {str(path): sha256(path) for path in
                               (COHORT / "cases.csv", COHORT / "detected_features.npy",
                                COHORT / "root_indices.npy", BASELINE)},
              "source_sha256": sha256(Path(__file__)), "test_read": False}
    with (OUTPUT / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({fold: {name: item["all"]["AC@1"] for name, item in
                             row["variants"].items()} for fold, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
