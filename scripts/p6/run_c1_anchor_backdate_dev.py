#!/usr/bin/env python3
"""Train-only forward check of a fixed, fold-1-derived RCA anchor backdate."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from src.e2e.c1_common_cohort import extract_c1_anchor_features
from src.e2e.c1_xgb_adapter import MODEL_PARAMETERS, grouped_training_data
from src.e2e.gaia_rca_adapter import GaiaRcaRawIndex
from src.e2e.rca_model import GAIA_SERVICES, rank_candidates

SOURCE = Path("/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2")
COHORT = SOURCE / "experiments/p6/c1_supervision_oos/c1-supervision-oos-v1-seed42/train_cohort"
INDEX_MANIFEST = SOURCE / "data/p5/v3/rca_raw_index/index_manifest.json"
BASELINE = ROOT / "experiments/p6/c1_xgb_forward_dev/c1-xgb-forward-v1/rankings.csv"
OUTPUT = ROOT / "experiments/p6/c1_anchor_backdate_dev/c1-anchor-backdate-forward-v1"
FOLDS = (("f1_to_f2", (1,), 2), ("f12_to_f3", (1, 2), 3))
SHARD_SIZE = 50
RAW_INDEX = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def inputs() -> dict:
    return {str(path): sha256(path) for path in
            (COHORT / "cases.csv", COHORT / "detected_features.npy",
             COHORT / "gt_features.npy", COHORT / "root_indices.npy",
             BASELINE, INDEX_MANIFEST)}


def cases_and_offset():
    cases = pd.read_csv(COHORT / "cases.csv", keep_default_na=False)
    if len(cases) != 3225 or cases.case_id.duplicated().any() or cases.fold.value_counts().to_dict() != {
            1: 935, 2: 1168, 3: 1122}:
        raise ValueError("sealed Train cohort changed")
    fold1 = cases.loc[cases.fold.eq(1)]
    offset_ms = int(np.median(fold1.detected_anchor_ms.to_numpy(dtype=np.int64)
                              - fold1.gt_anchor_ms.to_numpy(dtype=np.int64)))
    if offset_ms != 25621:
        raise ValueError("fold-1 fixed offset differs from expected derivation")
    for _, earlier, later in FOLDS:
        train = cases.fold.isin(earlier)
        evaluate = cases.fold.eq(later)
        if cases.loc[train, "detected_anchor_ms"].max() >= cases.loc[evaluate, "detected_anchor_ms"].min():
            raise ValueError("forward Train chronology failed")
    return cases, offset_ms


def shard_path(index: int) -> Path:
    return OUTPUT / "features" / "shard_{:03d}.npy".format(index)


def extract_shard(task):
    index, records, offset_ms = task
    if RAW_INDEX is None:
        raise RuntimeError("raw index not initialized before fork")
    values = np.stack([extract_c1_anchor_features(
        RAW_INDEX, case_id=case_id, anchor_ms=int(anchor_ms) - offset_ms)
                       for case_id, anchor_ms in records])
    return index, values


def materialize(workers: int) -> None:
    global RAW_INDEX
    cases, offset_ms = cases_and_offset()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "features").mkdir(exist_ok=True)
    lock_path = OUTPUT / "run_lock.json"
    lock = {"status": "TRAIN_SHIFTED_FEATURES_RESERVED", "offset_ms": offset_ms,
            "offset_rule": "median detected minus GT anchor in fold 1 only",
            "input_sha256": inputs(), "source_sha256": sha256(Path(__file__)),
            "test_read": False, "shard_size": SHARD_SIZE, "n_cases": len(cases)}
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise ValueError("backdate run lock drift; use a new run directory")
    else:
        write_new(lock_path, lock)
    n_shards = (len(cases) + SHARD_SIZE - 1) // SHARD_SIZE
    jobs = []
    for shard in range(n_shards):
        start, stop = shard * SHARD_SIZE, min((shard + 1) * SHARD_SIZE, len(cases))
        path = shard_path(shard)
        if path.exists():
            old = np.load(path, allow_pickle=False)
            if old.shape != (stop - start, 10, 68) or not np.isfinite(old).all():
                raise ValueError("incomplete or invalid existing feature shard: " + str(path))
            continue
        subset = cases.iloc[start:stop]
        records = list(zip(subset.case_id.tolist(), subset.detected_anchor_ms.tolist()))
        jobs.append((shard, records, offset_ms))
    if jobs:
        RAW_INDEX = GaiaRcaRawIndex.from_manifest(INDEX_MANIFEST)
        # Replay a frozen original anchor before using this raw index for new features.
        original = np.load(COHORT / "detected_features.npy", mmap_mode="r", allow_pickle=False)
        for pos in (0, 934, 2102, 3224):
            row = cases.iloc[pos]
            replay = extract_c1_anchor_features(RAW_INDEX, case_id=row.case_id,
                                                anchor_ms=int(row.detected_anchor_ms))
            if not np.array_equal(replay, original[pos]):
                raise ValueError("raw index cannot replay sealed detected feature at " + str(pos))
        if workers == 1:
            iterator = map(extract_shard, jobs)
            for shard, values in iterator:
                with shard_path(shard).open("xb") as handle:
                    np.save(handle, values, allow_pickle=False)
                print("completed shard", shard + 1, "of", n_shards, flush=True)
        else:
            with mp.get_context("fork").Pool(processes=workers) as pool:
                for shard, values in pool.imap_unordered(extract_shard, jobs, chunksize=1):
                    with shard_path(shard).open("xb") as handle:
                        np.save(handle, values, allow_pickle=False)
                    print("completed shard", shard + 1, "of", n_shards, flush=True)
    parts = [np.load(shard_path(shard), allow_pickle=False) for shard in range(n_shards)]
    shifted = np.concatenate(parts)
    if shifted.shape != (3225, 10, 68) or not np.isfinite(shifted).all():
        raise ValueError("materialized feature tensor invalid")
    with (OUTPUT / "shifted_features.npy").open("xb") as handle:
        np.save(handle, shifted, allow_pickle=False)
    write_new(OUTPUT / "feature_manifest.json", {
        "status": "TRAIN_SHIFTED_FEATURES_COMPLETE", "offset_ms": offset_ms,
        "n_cases": len(cases), "shape": list(shifted.shape),
        "shard_sha256": {shard_path(shard).name: sha256(shard_path(shard))
                         for shard in range(n_shards)},
        "shifted_features_sha256": sha256(OUTPUT / "shifted_features.npy"),
        "run_lock_sha256": sha256(lock_path), "test_read": False})
    print("TRAIN_SHIFTED_FEATURES_COMPLETE", OUTPUT, flush=True)


def metrics(ranks):
    return {"n": int(len(ranks)), "correct_at_1": int(np.sum(ranks == 1)),
            "AC@1": float(np.mean(ranks == 1)), "AC@3": float(np.mean(ranks <= 3)),
            "AC@5": float(np.mean(ranks <= 5)), "MRR": float(np.mean(1.0 / ranks))}


def forward() -> None:
    cases, offset_ms = cases_and_offset()
    lock = json.loads((OUTPUT / "run_lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUTPUT / "feature_manifest.json").read_text(encoding="utf-8"))
    if (lock["input_sha256"] != inputs() or lock["source_sha256"] != sha256(Path(__file__))
            or manifest["run_lock_sha256"] != sha256(OUTPUT / "run_lock.json")
            or manifest["shifted_features_sha256"] != sha256(OUTPUT / "shifted_features.npy")
            or manifest["offset_ms"] != offset_ms):
        raise ValueError("backdate feature binding drift")
    if (OUTPUT / "results.json").exists():
        raise FileExistsError(OUTPUT / "results.json")
    detected = np.load(COHORT / "detected_features.npy", allow_pickle=False)
    gt = np.load(COHORT / "gt_features.npy", allow_pickle=False)
    shifted = np.load(OUTPUT / "shifted_features.npy", allow_pickle=False)
    roots = np.load(COHORT / "root_indices.npy", allow_pickle=False)
    baseline = pd.read_csv(BASELINE, keep_default_na=False)
    expected = cases.loc[cases.fold.isin((2, 3))].reset_index(drop=True)
    if (baseline.case_id.tolist() != expected.case_id.tolist()
            or baseline.fold.tolist() != expected.fold.tolist()
            or [GAIA_SERVICES[int(root)] for root in roots] != cases.root_service.tolist()):
        raise ValueError("baseline/candidate/label cohort mismatch")
    results, predictions = {}, []
    for name, train_folds, eval_fold in FOLDS:
        train = np.flatnonzero(cases.fold.isin(train_folds).to_numpy())
        evaluate = np.flatnonzero(cases.fold.eq(eval_fold).to_numpy())
        eval_cases = cases.iloc[evaluate].reset_index(drop=True)
        eval_baseline = baseline.loc[baseline.fold.eq(eval_fold)].reset_index(drop=True)
        base_ranks = np.asarray([json.loads(row).index(root) + 1 for row, root in
                                 zip(eval_baseline.ranking_xgb, eval_cases.root_service)])
        fold_results = {"train_n": len(train), "eval_n": len(evaluate),
                        "baseline": metrics(base_ranks), "variants": {}}
        for variant, train_values in (("detected_train_shifted_eval", detected),
                                      ("gt_train_shifted_eval", gt),
                                      ("shifted_train_shifted_eval", shifted)):
            matrix, target, group = grouped_training_data(train_values[train], roots[train])
            model = xgb.XGBRanker(**MODEL_PARAMETERS)
            model.fit(matrix, target, group=group, verbose=False)
            scores = model.predict(np.ascontiguousarray(
                shifted[evaluate].reshape(-1, 68))).reshape(-1, 10)
            rankings = [rank_candidates(GAIA_SERVICES, row) for row in scores]
            ranks = np.asarray([row.index(root) + 1 for row, root in
                                zip(rankings, eval_cases.root_service)])
            fold_results["variants"][variant] = {
                "all": metrics(ranks),
                "fixed_errors": int(np.sum((ranks == 1) & (base_ranks != 1))),
                "new_errors": int(np.sum((ranks != 1) & (base_ranks == 1))),
            }
            for case_id, ranking, rank, base_rank in zip(eval_cases.case_id, rankings, ranks,
                                                          base_ranks):
                predictions.append({"case_id": case_id, "fold": eval_fold,
                                    "variant": variant, "ranking": json.dumps(ranking),
                                    "rank": int(rank), "baseline_rank": int(base_rank)})
        results[name] = fold_results
    pd.DataFrame(predictions).to_csv(OUTPUT / "predictions.csv", index=False)
    write_new(OUTPUT / "results.json", {
        "status": "TRAIN_ONLY_FORWARD_DEVELOPMENT", "test_read": False,
        "offset_ms": offset_ms, "model_parameters": MODEL_PARAMETERS, "folds": results,
        "feature_manifest_sha256": sha256(OUTPUT / "feature_manifest.json"),
        "predictions_sha256": sha256(OUTPUT / "predictions.csv")})
    print(json.dumps({fold: {variant: item["all"]["AC@1"] for variant, item in
                             row["variants"].items()} for fold, row in results.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("materialize", "forward"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.action == "materialize":
        if not 1 <= args.workers <= 16:
            raise ValueError("workers must be 1..16")
        materialize(args.workers)
    else:
        forward()
