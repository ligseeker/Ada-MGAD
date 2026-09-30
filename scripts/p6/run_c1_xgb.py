#!/usr/bin/env python3
"""Run one frozen, isolated GAIA C1 68D + XGBRanker scorer comparison.

Invoke init, train, lock-test, evaluate separately. A stage never overwrites
an existing directory. Test GT and matching are opened only in evaluate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb

from src.e2e.c1_xgb_adapter import (
    MODEL_PARAMETERS, evaluator_scope, grouped_training_data, paired_scope,
)
from src.e2e.rca_model import GAIA_SERVICES
from src.e2e.system_trigger import to_builtin

CONFIG = ROOT / "configs/e2e/gaia_p6_c1_z2_xgb_v1.json"
RUN_REL = "experiments/p6/c1_z2_xgb/c1-z2-xgb-v1-seed20260826"
SOURCE_FILES = (
    "configs/e2e/gaia_p6_c1_z2_xgb_v1.json",
    "scripts/p6/run_c1_xgb.py",
    "src/e2e/c1_xgb_adapter.py",
    "src/e2e/c1_evaluation.py",
    "src/e2e/c1_v2_c2.py",
    "src/e2e/rca_model.py",
    "src/e2e/protocol.py",
    "src/e2e/system_trigger.py",
    "src/e2e/event_detection.py",
)
TRAIN_FILES = (
    "run_lock.json", "train_cohort/completion_manifest.json",
    "train_cohort/cases.csv", "train_cohort/detected_features.npy",
    "train_cohort/root_indices.npy",
)
TEST_LABEL_FREE_FILES = (
    "predictions/completion_manifest.json", "predictions/prediction_lock.json",
    "predictions/feature_inputs.npy", "predictions/feature_valid.npy",
    "predictions/scope_rankings.csv",
)
TEST_EVAL_FILES = ("evaluation/c1_results.json", "evaluation/c2_full_diagnosis.json")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def source_path(config: Mapping[str, object], relative: str) -> Path:
    return Path(config["source_root"]) / relative


def old_run_path(config: Mapping[str, object], relative: str) -> Path:
    return source_path(config, str(config["source_run"]) + "/" + relative)


def check_source_files(config: Mapping[str, object]) -> Mapping[str, str]:
    actual = {relative: digest(ROOT / relative) for relative in SOURCE_FILES}
    if config["output_root"] != RUN_REL or config["model"]["learner"] != "xgboost.XGBRanker":
        raise ValueError("XGB protocol or output path drift")
    params = {key: config["model"][key] for key in MODEL_PARAMETERS}
    if params != MODEL_PARAMETERS or config["model"]["normalization"] != "none":
        raise ValueError("XGB frozen scorer parameters drift")
    for relative, expected in config["reference_sha256"].items():
        if digest(Path(config["reference_root"]) / relative) != expected:
            raise ValueError("Ada-RCA frozen scorer source drift: " + relative)
    return actual


def check_environment(config: Mapping[str, object]) -> Mapping[str, str]:
    actual = {"python": platform.python_version(), "xgboost": xgb.__version__,
              "numpy": np.__version__, "pandas": pd.__version__,
              "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    if actual != config["expected_environment"]:
        raise ValueError("pinned XGBoost environment drift: " + repr(actual))
    return actual


def check_bound(config: Mapping[str, object], relatives: Sequence[str]) -> Mapping[str, str]:
    observed = {}
    for relative in relatives:
        path = source_path(config, relative)
        expected = config["source_sha256"][relative]
        actual = digest(path)
        if actual != expected:
            raise ValueError("sealed C1 input drift: " + relative)
        observed[relative] = actual
    return observed


def source_rel(config: Mapping[str, object], filename: str) -> str:
    return str(config["source_run"]) + "/" + filename


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                   universal_newlines=True).strip()


def _git_clean() -> bool:
    return not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT),
                                       universal_newlines=True).strip()


def _git_descends_from(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          cwd=str(ROOT), check=False).returncode == 0


def _lock(root: Path, config: Mapping[str, object]) -> Mapping[str, object]:
    lock = read_json(root / "run_lock.json")
    if (lock.get("status") != "RUN_RESERVED"
            or lock.get("protocol_id") != config["protocol_id"]
            or lock.get("config_sha256") != digest(CONFIG)
            or lock.get("source_sha256") != check_source_files(config)
            or lock.get("environment") != check_environment(config)
            or not _git_descends_from(str(lock.get("git_head")))):
        raise ValueError("XGB run lock, code, environment or commit drift")
    return lock


def _completed(root: Path, name: str) -> Mapping[str, object]:
    stage = root / name
    manifest = read_json(stage / "completion_manifest.json")
    if (manifest.get("status") != "COMPLETE" or manifest.get("stage") != name
            or manifest.get("run_lock_sha256") != digest(root / "run_lock.json")):
        raise ValueError("XGB stage is not sealed: " + name)
    for relative, expected in manifest["files"].items():
        if digest(stage / relative) != expected:
            raise ValueError("XGB stage output drift: " + name + "/" + relative)
    return manifest


def _seal(root: Path, stage: str, files: Sequence[str], details: Mapping[str, object]) -> None:
    directory = root / stage
    write_json_new(directory / "completion_manifest.json", {
        "schema_version": "p6_c1_xgb_stage_manifest_v1",
        "stage": stage, "status": "COMPLETE",
        "run_lock_sha256": digest(root / "run_lock.json"),
        "files": {name: digest(directory / name) for name in files},
        "details": details,
    })


def preflight(config: Mapping[str, object]) -> None:
    check_source_files(config)
    check_environment(config)
    check_bound(config, [source_rel(config, name) for name in TRAIN_FILES])
    old = read_json(old_run_path(config, "run_lock.json"))
    cohort = read_json(old_run_path(config, "train_cohort/completion_manifest.json"))
    if (old.get("git_head") != config["source_run_execution_commit"]
            or cohort.get("status") != "COMPLETE"
            or not cohort["details"].get("floors_pass")
            or {str(k): int(v) for k, v in cohort["details"]["common_case_coverage"].items()}
            != config["expected_cohort"]["fold_counts"]):
        raise ValueError("C1 formal Train cohort provenance or gates failed")


def init(config: Mapping[str, object], root: Path) -> None:
    preflight(config)
    if not _git_clean():
        raise ValueError("commit the frozen XGB protocol and code before reserving a formal run")
    root.mkdir(parents=True, exist_ok=False)
    write_json_new(root / "run_lock.json", {
        "schema_version": "p6_c1_xgb_run_lock_v1", "status": "RUN_RESERVED",
        "protocol_id": config["protocol_id"], "run_id": config["run_id"],
        "git_head": _git_head(), "config_sha256": digest(CONFIG),
        "source_sha256": check_source_files(config),
        "environment": check_environment(config),
        "source_run_lock_sha256": config["source_sha256"][source_rel(config, "run_lock.json")],
        "test_ground_truth_read": False,
    })


def train(config: Mapping[str, object], root: Path) -> None:
    _lock(root, config)
    bound = check_bound(config, [source_rel(config, name) for name in TRAIN_FILES])
    cases = pd.read_csv(old_run_path(config, "train_cohort/cases.csv"))
    features = np.load(old_run_path(config, "train_cohort/detected_features.npy"), allow_pickle=False)
    roots = np.load(old_run_path(config, "train_cohort/root_indices.npy"), allow_pickle=False)
    expected = config["expected_cohort"]
    counts = {str(k): int(v) for k, v in cases["fold"].value_counts().sort_index().items()}
    if (len(cases) != expected["train_cases"] or counts != expected["fold_counts"]
            or cases["case_id"].duplicated().any()
            or not cases["root_service"].isin(GAIA_SERVICES).all()
            or [GAIA_SERVICES[int(i)] for i in roots] != cases["root_service"].tolist()):
        raise ValueError("C1 Train cohort case, fold or root alignment failed")
    matrix, target, group = grouped_training_data(features, roots)
    stage = root / "train"
    stage.mkdir(exist_ok=False)
    model = xgb.XGBRanker(**MODEL_PARAMETERS)
    model.fit(matrix, target, group=group, verbose=False)
    model.save_model(str(stage / "model_state.ubj"))
    write_json_new(stage / "train_summary.json", {
        "train_cases": int(len(cases)), "train_rows": int(len(matrix)),
        "groups": int(len(group)), "positives": int(target.sum()),
        "fold_counts": counts, "root_counts": {str(k): int(v) for k, v in
                                            cases["root_service"].value_counts().sort_index().items()},
        "model_parameters": MODEL_PARAMETERS, "normalization": "none",
        "source_inputs_sha256": bound,
        "test_ground_truth_read": False, "test_features_read": False,
    })
    _seal(root, "train", ("model_state.ubj", "train_summary.json"),
          {"train_cases": len(cases), "group_size": 10})


def lock_test(config: Mapping[str, object], root: Path) -> None:
    _lock(root, config)
    _completed(root, "train")
    relatives = [source_rel(config, name) for name in TEST_LABEL_FREE_FILES]
    relatives.append("experiments/p6/system_event_trigger/test_episodes.csv")
    bound = check_bound(config, relatives)
    old_lock = read_json(old_run_path(config, "predictions/prediction_lock.json"))
    old_manifest = read_json(old_run_path(config, "predictions/completion_manifest.json"))
    if (old_lock.get("test_matching_read") is not False
            or old_lock.get("test_ground_truth_read") is not False
            or old_manifest.get("status") != "COMPLETE"):
        raise ValueError("source label-free prediction lock invalid")
    old_scope = pd.read_csv(old_run_path(config, "predictions/scope_rankings.csv"),
                            keep_default_na=False)
    features = np.load(old_run_path(config, "predictions/feature_inputs.npy"), allow_pickle=False)
    valid = np.load(old_run_path(config, "predictions/feature_valid.npy"), allow_pickle=False)
    episodes = pd.read_csv(source_path(config,
        "experiments/p6/system_event_trigger/test_episodes.csv"))
    expected = config["expected_cohort"]
    if (len(episodes) != expected["test_episodes"]
            or int(valid.sum()) != expected["test_legal"]
            or int((~valid).sum()) != expected["test_illegal_context"]):
        raise ValueError("frozen Test episode scope drift")
    model = xgb.XGBRanker(**MODEL_PARAMETERS)
    model.load_model(str(root / "train/model_state.ubj"))
    scope = paired_scope(old_scope=old_scope, episodes=episodes, features=features,
                         valid=valid, model=model)
    stage = root / "predictions"
    stage.mkdir(exist_ok=False)
    scope.to_csv(stage / "scope_rankings.csv", index=False)
    write_json_new(stage / "prediction_lock.json", {
        "schema_version": "p6_c1_xgb_prediction_lock_v1",
        "protocol_id": config["protocol_id"], "run_id": config["run_id"],
        "comparator": "C1-v2 Arm C Conditional Logit on detected Test anchor",
        "new_model": "detected Train anchor 68D + frozen XGBRanker on detected Test anchor",
        "all_episodes": len(scope), "legal_ranked": int(valid.sum()),
        "illegal_context": int((~valid).sum()),
        "model_sha256": digest(root / "train/model_state.ubj"),
        "label_free_inputs_sha256": bound,
        "scope_rankings_sha256": digest(stage / "scope_rankings.csv"),
        "test_gt_or_matching_read": False,
        "missing_ranking_policy": "zero correctness; never remove from denominator",
    })
    _seal(root, "predictions", ("scope_rankings.csv", "prediction_lock.json"),
          {"all_episodes": len(scope), "legal_ranked": int(valid.sum()),
           "test_gt_or_matching_read": False})


def _baseline_replay_gate(c1: Mapping[str, object], c2: Mapping[str, object],
                          old_c1: Mapping[str, object], old_c2: Mapping[str, object]) -> None:
    if (c1["primary"]["n"] != old_c1["primary"]["n"]
            or c1["primary"]["correct_b"] != old_c1["primary"]["correct_c"]
            or c2["gt_population"] != old_c2["gt_population"]
            or c2["predicted_episodes"] != old_c2["predicted_episodes"]):
        raise ValueError("C1 Conditional Logit comparator cohort replay failed")
    for k in (1, 3, 5):
        new = c2["arms"]["b"]["metrics"]["@{}".format(k)]
        old = old_c2["arms"]["c"]["metrics"]["@{}".format(k)]
        if new != old:
            raise ValueError("C1 Conditional Logit E2E replay failed at Top-{}".format(k))


def evaluate(config: Mapping[str, object], root: Path) -> None:
    from src.e2e.c1_evaluation import evaluate_c1_prediction_lock
    from src.e2e.c1_v2_c2 import evaluate_c2_full_diagnosis
    from src.e2e.protocol import load_registry

    _lock(root, config)
    _completed(root, "train")
    _completed(root, "predictions")
    prediction_lock = read_json(root / "predictions/prediction_lock.json")
    if prediction_lock.get("test_gt_or_matching_read") is not False:
        raise ValueError("Test prediction lock is not label-free")
    relatives = [source_rel(config, name) for name in TEST_EVAL_FILES]
    relatives += ["experiments/p6/system_event_trigger/test_matching.csv",
                  "artifacts/p5/v3/protocol/gt_event_registry.csv",
                  "configs/e2e/gaia_p5_v3_preprocessing_v2.json"]
    bound = check_bound(config, relatives)
    scope = pd.read_csv(root / "predictions/scope_rankings.csv", keep_default_na=False)
    mapped = evaluator_scope(scope)
    matching = pd.read_csv(source_path(config,
        "experiments/p6/system_event_trigger/test_matching.csv"))
    base = read_json(source_path(config, "configs/e2e/gaia_p5_v3_preprocessing_v2.json"))
    registry = load_registry(base, Path(config["source_root"]))
    interval = (base["split"]["boundary_ms"], base["split"]["absolute_end_ms"])
    c1 = evaluate_c1_prediction_lock(scope=mapped, matching=matching)
    c2 = evaluate_c2_full_diagnosis(scope=mapped, registry=registry,
                                    matching=matching, test_interval_ms=interval)
    old_c1 = read_json(old_run_path(config, "evaluation/c1_results.json"))
    old_c2 = read_json(old_run_path(config, "evaluation/c2_full_diagnosis.json"))
    _baseline_replay_gate(c1, c2["summary"], old_c1, old_c2)
    expected = config["expected_cohort"]
    if (c1["primary"]["n"] != expected["test_matched_legal"]
            or c1["scope"]["false_alarms"] != expected["test_false_alarms"]
            or c1["scope"]["misses"] != expected["test_misses"]
            or c2["summary"]["gt_population"] != expected["test_gt_events"]):
        raise ValueError("frozen C1/Test denominator drift")
    stage = root / "evaluation"
    stage.mkdir(exist_ok=False)
    with (stage / "c1_paired_results.json").open("x", encoding="utf-8") as stream:
        json.dump(to_builtin(c1), stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    with (stage / "c2_full_diagnosis.json").open("x", encoding="utf-8") as stream:
        json.dump(to_builtin(c2["summary"]), stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    c2["ledger"].to_csv(stage / "c2_failure_ledger.csv", index=False)
    write_json_new(stage / "evaluation_provenance.json", {
        "evidence_grade": config["evidence_grade"],
        "evaluator_slot_b": "original C1-v2 Arm C Conditional Logit",
        "evaluator_slot_c": "new 68D XGBRanker",
        "baseline_replay_gate": "PASS",
        "test_gt_inputs_sha256": bound,
        "prediction_lock_sha256": digest(root / "predictions/prediction_lock.json"),
        "independent_test_confirmation": False,
    })
    _seal(root, "evaluation", ("c1_paired_results.json", "c2_full_diagnosis.json",
                               "c2_failure_ledger.csv", "evaluation_provenance.json"),
          {"paired_matched_legal": c1["primary"]["n"],
           "xgb_minus_cl_ac_at_1": c1["primary"]["delta_c_minus_b"],
           "independent_test_confirmation": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "init", "train", "lock-test", "evaluate"))
    args = parser.parse_args()
    config = read_json(CONFIG)
    root = ROOT / RUN_REL
    if args.action == "preflight":
        preflight(config)
    elif args.action == "init":
        init(config, root)
    elif args.action == "train":
        train(config, root)
    elif args.action == "lock-test":
        lock_test(config, root)
    elif args.action == "evaluate":
        evaluate(config, root)
    print("PASS", args.action, config["protocol_id"])


if __name__ == "__main__":
    main()
