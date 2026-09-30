#!/usr/bin/env python3
"""Single fixed-offset Test comparison after Train-only forward development.

Train, label-free Test prediction, and Test evaluation are separate commands.
No command overwrites a completed stage or changes the frozen 25.621s offset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
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
from src.e2e.system_trigger import to_builtin

SOURCE = Path("/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2")
OLD_RUN = SOURCE / "experiments/p6/c1_supervision_oos/c1-supervision-oos-v1-seed42"
XGB_RUN = ROOT / "experiments/p6/c1_z2_xgb/c1-z2-xgb-v1-seed20260826"
DEV_RUN = ROOT / "experiments/p6/c1_anchor_backdate_dev/c1-anchor-backdate-forward-v1"
RAW_MANIFEST = SOURCE / "data/p5/v3/rca_raw_index/index_manifest.json"
RUN = ROOT / "experiments/p6/c1_anchor_backdate/c1-anchor-backdate-v1-delta25621"
OFFSET_MS = 25621
SHARD_SIZE = 50
RAW_INDEX = None
SOURCE_FILES = (
    "scripts/p6/run_c1_anchor_backdate_test.py", "src/e2e/c1_common_cohort.py",
    "src/e2e/c1_xgb_adapter.py", "src/e2e/gaia_rca_adapter.py",
    "src/e2e/rca_features.py", "src/e2e/rca_model.py",
    "src/e2e/c1_evaluation.py", "src/e2e/c1_v2_c2.py",
    "src/e2e/protocol.py", "src/e2e/event_detection.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, sort_keys=True, indent=2, ensure_ascii=False)
        handle.write("\n")


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "xgboost": xgb.__version__}


def train_input_hashes():
    names = (OLD_RUN / "train_cohort/cases.csv",
             OLD_RUN / "train_cohort/root_indices.npy",
             DEV_RUN / "run_lock.json", DEV_RUN / "feature_manifest.json",
             DEV_RUN / "shifted_features.npy", DEV_RUN / "results.json")
    return {str(path): sha256(path) for path in names}


def predict_input_hashes():
    names = (SOURCE / "experiments/p6/system_event_trigger/test_episodes.csv",
             OLD_RUN / "predictions/feature_inputs.npy",
             OLD_RUN / "predictions/feature_valid.npy",
             XGB_RUN / "predictions/scope_rankings.csv",
             XGB_RUN / "predictions/prediction_lock.json",
             SOURCE / "configs/e2e/gaia_p5_v3_preprocessing_v2.json", RAW_MANIFEST)
    return {str(path): sha256(path) for path in names}


def check_run():
    lock = json.loads((RUN / "run_lock.json").read_text(encoding="utf-8"))
    if (lock["status"] != "RUN_RESERVED" or lock["offset_ms"] != OFFSET_MS
            or lock["source_sha256"] != source_hashes()
            or lock["train_input_sha256"] != train_input_hashes()
            or lock["environment"] != environment()
            or lock["test_gt_or_matching_read"] is not False):
        raise ValueError("anchor backdate run provenance drift")
    return lock


def train() -> None:
    cases = pd.read_csv(OLD_RUN / "train_cohort/cases.csv", keep_default_na=False)
    roots = np.load(OLD_RUN / "train_cohort/root_indices.npy", allow_pickle=False)
    features = np.load(DEV_RUN / "shifted_features.npy", allow_pickle=False)
    dev_lock = json.loads((DEV_RUN / "run_lock.json").read_text(encoding="utf-8"))
    dev_manifest = json.loads((DEV_RUN / "feature_manifest.json").read_text(encoding="utf-8"))
    dev = json.loads((DEV_RUN / "results.json").read_text(encoding="utf-8"))
    if (len(cases) != 3225 or roots.shape != (3225,) or features.shape != (3225, 10, 68)
            or not np.isfinite(features).all()
            or [GAIA_SERVICES[int(i)] for i in roots] != cases.root_service.tolist()
            or dev_lock["offset_ms"] != OFFSET_MS or dev["offset_ms"] != OFFSET_MS
            or dev_lock["test_read"] is not False or dev["test_read"] is not False
            or dev_lock["source_sha256"] != sha256(ROOT / "scripts/p6/run_c1_anchor_backdate_dev.py")
            or dev_manifest["run_lock_sha256"] != sha256(DEV_RUN / "run_lock.json")
            or dev_manifest["shifted_features_sha256"] != sha256(DEV_RUN / "shifted_features.npy")
            or dev["feature_manifest_sha256"] != sha256(DEV_RUN / "feature_manifest.json")
            or dev["predictions_sha256"] != sha256(DEV_RUN / "predictions.csv")):
        raise ValueError("fixed Train candidate is invalid")
    for row in dev["folds"].values():
        if row["variants"]["shifted_train_shifted_eval"]["all"]["AC@1"] <= row["baseline"]["AC@1"]:
            raise ValueError("Train-only forward GO check failed")
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=str(ROOT)).strip():
        raise ValueError("commit Test execution code before training")
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "train").mkdir(exist_ok=False)
    write_new(RUN / "run_lock.json", {
        "status": "RUN_RESERVED", "git_head": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=str(ROOT), universal_newlines=True).strip(),
        "offset_ms": OFFSET_MS, "offset_rule": "fold-1 OOS delay median",
        "source_sha256": source_hashes(), "train_input_sha256": train_input_hashes(),
        "environment": environment(),
        "test_gt_or_matching_read": False,
        "evidence_grade": "exploratory reused Test after Train-only forward development",
    })
    matrix, target, group = grouped_training_data(features, roots)
    model = xgb.XGBRanker(**MODEL_PARAMETERS)
    model.fit(matrix, target, group=group, verbose=False)
    model.save_model(str(RUN / "train/model_state.ubj"))
    write_new(RUN / "train/completion_manifest.json", {
        "status": "COMPLETE", "stage": "train", "cases": len(cases),
        "rows": len(matrix), "model_parameters": MODEL_PARAMETERS,
        "model_sha256": sha256(RUN / "train/model_state.ubj"),
        "run_lock_sha256": sha256(RUN / "run_lock.json"),
        "test_read": False,
    })
    print("TRAIN_COMPLETE", RUN)


def shard_path(number):
    return RUN / "predictions/features/shard_{:03d}.npy".format(number)


def extract_shard(task):
    number, records = task
    if RAW_INDEX is None:
        raise RuntimeError("raw index missing before fork")
    features = np.stack([extract_c1_anchor_features(
        RAW_INDEX, case_id=prediction_id, anchor_ms=int(t_hat) - OFFSET_MS)
                         for prediction_id, t_hat in records])
    return number, features


def predict(workers: int) -> None:
    global RAW_INDEX
    check_run()
    train_manifest = json.loads((RUN / "train/completion_manifest.json").read_text())
    if (train_manifest["status"] != "COMPLETE" or train_manifest["model_sha256"] !=
            sha256(RUN / "train/model_state.ubj")):
        raise ValueError("Train model drift")
    episodes = pd.read_csv(SOURCE / "experiments/p6/system_event_trigger/test_episodes.csv",
                           keep_default_na=False)
    baseline = pd.read_csv(XGB_RUN / "predictions/scope_rankings.csv", keep_default_na=False)
    old_lock = json.loads((XGB_RUN / "predictions/prediction_lock.json").read_text())
    valid = np.load(OLD_RUN / "predictions/feature_valid.npy", allow_pickle=False)
    original = np.load(OLD_RUN / "predictions/feature_inputs.npy", mmap_mode="r", allow_pickle=False)
    base_config = json.loads((SOURCE / "configs/e2e/gaia_p5_v3_preprocessing_v2.json").read_text())
    start, end = int(base_config["split"]["boundary_ms"]), int(base_config["split"]["absolute_end_ms"])
    anchors = episodes.t_hat.to_numpy(dtype=np.int64)
    shifted = anchors - OFFSET_MS
    shifted_legal = (shifted - 300000 >= start) & (shifted + 300000 <= end)
    if (len(episodes) != 4214 or len(baseline) != len(episodes)
            or old_lock["scope_rankings_sha256"] != sha256(XGB_RUN / "predictions/scope_rankings.csv")
            or old_lock["test_gt_or_matching_read"] is not False
            or valid.shape != (4214,) or int(valid.sum()) != 4213
            or original.shape != (4214, 10, 68)
            or not np.array_equal(valid, shifted_legal)
            or baseline.prediction_id.tolist() != episodes.prediction_id.tolist()
            or baseline.t_hat.tolist() != episodes.t_hat.tolist()
            or not baseline.loc[valid, "scope_status"].eq("legal").all()
            or not baseline.loc[valid, "ranking_status"].eq("complete").all()
            or not baseline.loc[~valid, "ranking_status"].eq("not_applicable").all()):
        raise ValueError("fixed label-free Test cohort or shifted window legality drift")
    if (RUN / "predictions/prediction_lock.json").exists():
        raise FileExistsError("Test prediction is already locked")
    (RUN / "predictions/features").mkdir(parents=True, exist_ok=True)
    legal_positions = np.flatnonzero(valid)
    n_shards = (len(legal_positions) + SHARD_SIZE - 1) // SHARD_SIZE
    jobs = []
    for number in range(n_shards):
        positions = legal_positions[number * SHARD_SIZE:(number + 1) * SHARD_SIZE]
        path = shard_path(number)
        if path.exists():
            values = np.load(path, allow_pickle=False)
            if values.shape != (len(positions), 10, 68) or not np.isfinite(values).all():
                raise ValueError("existing Test feature shard invalid: " + str(path))
            continue
        records = list(zip(episodes.iloc[positions].prediction_id.tolist(),
                           episodes.iloc[positions].t_hat.tolist()))
        jobs.append((number, records))
    if jobs:
        RAW_INDEX = GaiaRcaRawIndex.from_manifest(RAW_MANIFEST)
        for position in (0, len(episodes) // 2, len(episodes) - 2):
            if not valid[position]:
                continue
            episode = episodes.iloc[position]
            replay = extract_c1_anchor_features(RAW_INDEX,
                case_id=str(episode.prediction_id), anchor_ms=int(episode.t_hat))
            if not np.array_equal(replay, original[position]):
                raise ValueError("raw index cannot replay frozen Test feature")
        if workers == 1:
            iterator = map(extract_shard, jobs)
            for number, values in iterator:
                with shard_path(number).open("xb") as handle:
                    np.save(handle, values, allow_pickle=False)
                print("completed Test shard", number + 1, "of", n_shards, flush=True)
        else:
            with mp.get_context("fork").Pool(processes=workers) as pool:
                for number, values in pool.imap_unordered(extract_shard, jobs, chunksize=1):
                    with shard_path(number).open("xb") as handle:
                        np.save(handle, values, allow_pickle=False)
                    print("completed Test shard", number + 1, "of", n_shards, flush=True)
    legal_features = np.concatenate([np.load(shard_path(number), allow_pickle=False)
                                     for number in range(n_shards)])
    if legal_features.shape != (4213, 10, 68) or not np.isfinite(legal_features).all():
        raise ValueError("shifted Test feature tensor invalid")
    all_features = np.zeros((len(episodes), 10, 68), dtype=np.float32)
    all_features[legal_positions] = legal_features
    with (RUN / "predictions/shifted_feature_inputs.npy").open("xb") as handle:
        np.save(handle, all_features, allow_pickle=False)
    model = xgb.XGBRanker(**MODEL_PARAMETERS)
    model.load_model(str(RUN / "train/model_state.ubj"))
    scores = model.predict(np.ascontiguousarray(legal_features.reshape(-1, 68))).reshape(-1, 10)
    new_rankings = [json.dumps(rank_candidates(GAIA_SERVICES, row), separators=(",", ":"))
                    for row in scores]
    scope = baseline[["prediction_id", "t_hat", "scope_status", "ranking_status",
                      "failure_reason"]].copy()
    scope["ranking_b"] = baseline.ranking_xgb.to_numpy()
    scope["ranking_c"] = ""
    scope.loc[valid, "ranking_c"] = new_rankings
    if (not scope.loc[valid, ["ranking_b", "ranking_c"]].ne("").all().all()
            or not scope.loc[~valid, ["ranking_b", "ranking_c"]].eq("").all().all()):
        raise ValueError("incomplete paired Test rankings")
    scope.to_csv(RUN / "predictions/scope_rankings.csv", index=False)
    write_new(RUN / "predictions/prediction_lock.json", {
        "status": "PREDICTION_LOCKED", "offset_ms": OFFSET_MS,
        "all_episodes": len(episodes), "legal_ranked": len(legal_positions),
        "illegal_context": int((~valid).sum()), "ranking_failure": 0,
        "train_model_sha256": sha256(RUN / "train/model_state.ubj"),
        "test_label_free_input_sha256": predict_input_hashes(),
        "shifted_features_sha256": sha256(RUN / "predictions/shifted_feature_inputs.npy"),
        "scope_rankings_sha256": sha256(RUN / "predictions/scope_rankings.csv"),
        "feature_shards_sha256": {shard_path(number).name: sha256(shard_path(number))
                                  for number in range(n_shards)},
        "test_gt_or_matching_read": False,
    })
    print("PREDICTION_LOCKED", RUN)


def evaluate() -> None:
    from src.e2e.c1_evaluation import evaluate_c1_prediction_lock
    from src.e2e.c1_v2_c2 import evaluate_c2_full_diagnosis
    from src.e2e.protocol import load_registry

    check_run()
    pred_lock = json.loads((RUN / "predictions/prediction_lock.json").read_text())
    if (pred_lock["status"] != "PREDICTION_LOCKED"
            or pred_lock["test_gt_or_matching_read"] is not False
            or pred_lock["scope_rankings_sha256"] != sha256(RUN / "predictions/scope_rankings.csv")
            or pred_lock["shifted_features_sha256"] != sha256(RUN / "predictions/shifted_feature_inputs.npy")
            or pred_lock["test_label_free_input_sha256"] != predict_input_hashes()):
        raise ValueError("label-free Test prediction lock drift")
    if (RUN / "evaluation").exists():
        raise FileExistsError("evaluation already exists")
    scope = pd.read_csv(RUN / "predictions/scope_rankings.csv", keep_default_na=False)
    matching_path = SOURCE / "experiments/p6/system_event_trigger/test_matching.csv"
    matching = pd.read_csv(matching_path, keep_default_na=False)
    config_path = SOURCE / "configs/e2e/gaia_p5_v3_preprocessing_v2.json"
    config = json.loads(config_path.read_text())
    registry_path = SOURCE / "artifacts/p5/v3/protocol/gt_event_registry.csv"
    registry = load_registry(config, SOURCE)
    interval = (int(config["split"]["boundary_ms"]), int(config["split"]["absolute_end_ms"]))
    c1 = evaluate_c1_prediction_lock(scope=scope, matching=matching)
    c2 = evaluate_c2_full_diagnosis(scope=scope, registry=registry,
                                    matching=matching, test_interval_ms=interval)
    old_c1 = json.loads((XGB_RUN / "evaluation/c1_paired_results.json").read_text())
    old_c2 = json.loads((XGB_RUN / "evaluation/c2_full_diagnosis.json").read_text())
    if (c1["primary"]["n"] != 4197 or c1["primary"]["correct_b"] !=
            old_c1["primary"]["correct_c"] or c2["summary"]["gt_population"] !=
            old_c2["gt_population"] or c2["summary"]["predicted_episodes"] !=
            old_c2["predicted_episodes"]):
        raise ValueError("baseline/cohort Test replay failed")
    for k in (1, 3, 5):
        if (c2["summary"]["arms"]["b"]["metrics"]["@{}".format(k)] !=
                old_c2["arms"]["c"]["metrics"]["@{}".format(k)]):
            raise ValueError("baseline E2E replay failed at Top-{}".format(k))
    (RUN / "evaluation").mkdir(exist_ok=False)
    write_new(RUN / "evaluation/c1_paired_results.json", c1)
    write_new(RUN / "evaluation/c2_full_diagnosis.json", c2["summary"])
    c2["ledger"].to_csv(RUN / "evaluation/c2_failure_ledger.csv", index=False)
    write_new(RUN / "evaluation/evaluation_provenance.json", {
        "status": "COMPLETE", "evidence_grade": "exploratory reused Test",
        "baseline_replay_gate": "PASS", "offset_ms": OFFSET_MS,
        "prediction_lock_sha256": sha256(RUN / "predictions/prediction_lock.json"),
        "test_gt_inputs_sha256": {str(path): sha256(path) for path in
                                  (matching_path, config_path, registry_path)},
        "independent_test_confirmation": False,
    })
    write_new(RUN / "evaluation/completion_manifest.json", {
        "status": "COMPLETE", "c1_paired_results_sha256": sha256(
            RUN / "evaluation/c1_paired_results.json"),
        "c2_full_diagnosis_sha256": sha256(RUN / "evaluation/c2_full_diagnosis.json"),
        "c2_failure_ledger_sha256": sha256(RUN / "evaluation/c2_failure_ledger.csv"),
        "evaluation_provenance_sha256": sha256(RUN / "evaluation/evaluation_provenance.json"),
    })
    print("EVALUATION_COMPLETE", c1["primary"],
          c2["summary"]["arms"]["c"]["metrics"]["@1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "predict", "evaluate"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be 1..16")
    if args.action == "train":
        train()
    elif args.action == "predict":
        predict(args.workers)
    else:
        evaluate()
