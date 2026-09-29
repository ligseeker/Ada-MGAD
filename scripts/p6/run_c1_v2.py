#!/usr/bin/env python3
"""Stage-by-stage P6-C1-v2 runner; full stages must be invoked manually.

Stages are exclusive and sealed. A failed stage stays INCOMPLETE and cannot be
resumed or overwritten; a formal rerun needs a new protocol version and run ID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.e2e.c1_v2_shared_data import (
    CONFIG, CONFIG_SHA256, PROTOCOL_ID, RUN_ID, SEGMENTS, bound_path, load_protocol,
    materialize_c1_v2_fold, read_json, run_root, sha256, source_hashes,
    validate_c1_v2_fold, validate_g1_stage, write_json_new,
)
from src.e2e.gaia_rca_adapter import GaiaRcaRawIndex, validate_raw_index_manifest
from src.e2e.protocol import load_registry
from src.e2e.system_trigger import to_builtin


def _protocol() -> Mapping[str, object]:
    if sha256(CONFIG) != CONFIG_SHA256:
        raise ValueError("P6-C1-v2 protocol config drift")
    return load_protocol()


def _base(protocol: Mapping[str, object]) -> Mapping[str, object]:
    return read_json(bound_path(protocol["bindings"]["base_config"]))


def _index(protocol: Mapping[str, object]):
    path = bound_path(protocol["bindings"]["rca_index_manifest"])
    validate_raw_index_manifest(path)
    return GaiaRcaRawIndex.from_manifest(path), path


def _save_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n")


def _save_npy(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()


def _stage(root: Path, name: str, work):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=False)
    write_json_new(directory / "INCOMPLETE.json", {"status": "INCOMPLETE", "stage": name})
    try:
        status, files, inputs, details = work(directory)
        if status not in ("COMPLETE", "NO_GO"):
            raise ValueError("P6-C1-v2 stage returned an invalid completion status")
        run_lock = read_json(root / "run_lock.json")
        if source_hashes() != run_lock["source_sha256"]:
            raise ValueError("P6-C1-v2 source changed during a formal stage")
        manifest = {"schema_version": "p6_c1_v2_stage_manifest_v1", "stage": name,
                    "status": status, "run_lock_sha256": sha256(root / "run_lock.json"),
                    "source_sha256": source_hashes(), "inputs": inputs,
                    "details": to_builtin(details),
                    "files": {filename: {"bytes": (directory / filename).stat().st_size,
                                         "sha256": sha256(directory / filename)}
                              for filename in files}}
        write_json_new(directory / "completion_manifest.json", manifest)
        (directory / "INCOMPLETE.json").unlink()
        return manifest
    except Exception as exc:
        try:
            write_json_new(directory / "failure.json",
                           {"status": "INCOMPLETE", "error_type": type(exc).__name__,
                            "message": str(exc)})
        except OSError:
            pass
        raise


def _validate_stage(root: Path, name: str, *, require_complete: bool = True):
    directory = root / name
    if (directory / "INCOMPLETE.json").exists():
        raise ValueError("P6-C1-v2 {} stage incomplete".format(name))
    manifest = read_json(directory / "completion_manifest.json")
    allowed = ("COMPLETE",) if require_complete else ("COMPLETE", "NO_GO")
    if (manifest.get("schema_version") != "p6_c1_v2_stage_manifest_v1"
            or manifest.get("stage") != name or manifest.get("status") not in allowed
            or manifest.get("run_lock_sha256") != sha256(root / "run_lock.json")
            or manifest.get("source_sha256") != source_hashes()):
        raise ValueError("P6-C1-v2 {} completion/source drift or NO_GO".format(name))
    for filename, entry in manifest["files"].items():
        path = directory / filename
        if (filename != Path(filename).name or path.is_symlink() or not path.is_file()
                or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]):
            raise ValueError("P6-C1-v2 {} output drift: {}".format(name, filename))
    return manifest


def _run_lock(root: Path, protocol: Mapping[str, object]) -> Mapping[str, object]:
    if root != run_root(protocol):
        raise ValueError("P6-C1-v2 run root drift")
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("P6-C1-v2 run initialization is incomplete")
    payload = read_json(root / "run_lock.json")
    g1_manifest = validate_g1_stage(protocol)
    if (payload.get("schema_version") != "p6_c1_v2_run_lock_v1"
            or payload.get("status") != "RUN_RESERVED"
            or payload.get("run_id") != RUN_ID
            or payload.get("protocol_id") != PROTOCOL_ID
            or payload.get("protocol_sha256") != sha256(CONFIG)
            or payload.get("g1_completion_manifest_sha256") != sha256(
                root / "g1_static/completion_manifest.json")
            or payload.get("g1_source_sha256") != g1_manifest["source"]["sha256"]
            or payload.get("source_sha256") != source_hashes()):
        raise ValueError("P6-C1-v2 run lock or source snapshot drift")
    return payload


def _bound_inputs(protocol: Mapping[str, object]) -> Mapping[str, object]:
    return {name: {"path": str(bound_path(binding).relative_to(ROOT)),
                   "sha256": str(binding["sha256"])}
            for name, binding in protocol["bindings"].items()}


def _preflight(protocol: Mapping[str, object]) -> None:
    root = run_root(protocol)
    if root.exists() and (root / "run_lock.json").exists():
        raise ValueError("P6-C1-v2 run root is already initialized")
    registry_path = bound_path(protocol["bindings"]["gt_registry"])
    base = _base(protocol)
    if sha256(registry_path) != base["event_registry"]["sha256"]:
        raise ValueError("P6-C1-v2 GT registry binding drift")
    for name, binding in protocol["bindings"].items():
        bound_path(binding)
    validate_g1_stage(protocol)
    validate_raw_index_manifest(bound_path(protocol["bindings"]["rca_index_manifest"]))
    print("PASS P6-C1-v2 protocol, bound inputs, G1 stage and RCA raw index")


def _init(protocol: Mapping[str, object], *, gpu: bool) -> None:
    root = run_root(protocol)
    if root.exists() and (root / "run_lock.json").exists():
        raise ValueError("P6-C1-v2 run root already exists; no overwrite")
    if gpu:
        import torch
        if not torch.cuda.is_available():
            raise ValueError("P6-C1-v2 CUDA was requested for the run but is unavailable")
    _preflight(protocol)
    g1_manifest = validate_g1_stage(protocol)
    write_json_new(root / "INCOMPLETE.json", {"status": "INCOMPLETE", "stage": "init"})
    write_json_new(root / "run_lock.json", {
        "schema_version": "p6_c1_v2_run_lock_v1",
        "protocol_id": PROTOCOL_ID, "run_id": RUN_ID,
        "protocol_sha256": sha256(CONFIG),
        "g1_completion_manifest_sha256": sha256(root / "g1_static/completion_manifest.json"),
        "g1_source_sha256": g1_manifest["source"]["sha256"],
        "source_sha256": source_hashes(),
        "git_head": _git_head(),
        "seed": int(protocol["seed"]),
        "status": "RUN_RESERVED",
        "device": "cuda" if gpu else "cpu",
        "bound_inputs": _bound_inputs(protocol)})
    (root / "folds").mkdir(exist_ok=False)
    (root / "INCOMPLETE.json").unlink()
    print("RESERVED", root)


def _fold(root: Path, protocol: Mapping[str, object], fold_number: int, *,
          action: str, gpu: bool) -> None:
    fold_root = root / "folds/fold_{:02d}".format(fold_number)
    if action == "fold-input":
        manifest = materialize_c1_v2_fold(protocol=protocol, fold_number=fold_number,
                                          fold_root=fold_root)
        print("COMPLETE P6-C1-v2 fold {} shared-array slices: {} / {} / {} windows".format(
            fold_number, manifest["segments"]["fit"]["legal_windows"],
            manifest["segments"]["selection"]["legal_windows"],
            manifest["segments"]["generation"]["legal_windows"]))
        return
    from src.e2e.c1_v2_detector import train_c1_v2_fold_detector
    run_lock = read_json(root / "run_lock.json")
    if gpu != (run_lock["device"] == "cuda"):
        raise ValueError("P6-C1-v2 fold detector device differs from the reserved run lock")
    detector = train_c1_v2_fold_detector(fold_root=fold_root, gpu=gpu,
                                         threshold_workers=int(
                                             protocol["resource_budget"]["threshold_workers"]),
                                         protocol=protocol)
    print("COMPLETE P6-C1-v2 fold {} detector: epoch {} threshold {} Selection {} episodes {}".format(
        fold_number, detector["selected_epoch"], detector["threshold"],
        detector["selection_metrics"], detector["generation_episodes"]))


def _cohort(root: Path, protocol: Mapping[str, object]) -> None:
    from src.e2e.c1_common_cohort import G2_MINIMUM_BY_FOLD, build_c1_common_cohort
    from src.e2e.c1_oos_matching import match_c1_oos_episodes
    from src.e2e.c1_v2_detector import validate_c1_v2_fold_detector
    locked_minimum = {int(fold["fold"]): int(fold["minimum_actual_common_cases"])
                      for fold in protocol["folds"]}
    if locked_minimum != G2_MINIMUM_BY_FOLD:
        raise ValueError("P6-C1-v2 cohort floors differ from the frozen protocol")

    def work(directory):
        sealed = []
        input_hashes = {}
        for fold_number in (1, 2, 3):
            fold_root = root / "folds/fold_{:02d}".format(fold_number)
            validate_c1_v2_fold_detector(fold_root=fold_root, protocol=protocol)
            manifest = validate_c1_v2_fold(fold_root, protocol=protocol)
            episodes = pd.read_csv(fold_root / "detector/generation_episodes.csv")
            interval = tuple(int(value) for value in
                             manifest["segments"]["generation"]["interval_ms"])
            sealed.append((fold_number, interval, episodes))
            input_hashes["fold_{:02d}_detector".format(fold_number)] = sha256(
                fold_root / "detector/completion_manifest.json")
        registry = load_registry(_base(protocol), ROOT)
        index, index_path = _index(protocol)
        input_hashes["rca_index_manifest"] = sha256(index_path)
        matched = [match_c1_oos_episodes(fold=fold_number, interval_ms=interval,
                                         episodes=episodes, registry=registry)
                   for fold_number, interval, episodes in sealed]
        cohort = build_c1_common_cohort(fold_matchings=matched, index=index)
        matching = pd.concat([item.matching for item in matched], ignore_index=True)
        _save_csv(directory / "matching.csv", matching)
        _save_csv(directory / "cases.csv", cohort.cases)
        _save_csv(directory / "exclusions.csv", cohort.exclusion_ledger)
        _save_npy(directory / "gt_features.npy", cohort.gt_features)
        _save_npy(directory / "detected_features.npy", cohort.detected_features)
        _save_npy(directory / "root_indices.npy", cohort.root_indices)
        details = {"fold_oos": [item.audit for item in matched],
                   "common_case_coverage": dict(cohort.coverage_by_fold),
                   "required_minimum": G2_MINIMUM_BY_FOLD,
                   "floors_pass": cohort.floors_pass,
                   "case_selection_uses_ranking_correctness": False}
        return ("COMPLETE" if cohort.floors_pass else "NO_GO",
                ("matching.csv", "cases.csv", "exclusions.csv", "gt_features.npy",
                 "detected_features.npy", "root_indices.npy"), input_hashes, details)

    result = _stage(root, "train_cohort", work)
    print(result["status"], "P6-C1-v2 common Train coverage",
          result["details"]["common_case_coverage"])


def _fit_rca(root: Path, protocol: Mapping[str, object]) -> None:
    from src.e2e.c1_shared_rca import fit_c1_shared_rca_arms
    from src.e2e.c1_v2_detector import validate_c1_v2_fold_detector
    from src.e2e.rca_model import save_conditional_logit
    cohort = _validate_stage(root, "train_cohort")
    for fold_number in (1, 2, 3):
        fold_root = root / "folds/fold_{:02d}".format(fold_number)
        validate_c1_v2_fold_detector(fold_root=fold_root, protocol=protocol)
        name = "fold_{:02d}_detector".format(fold_number)
        if cohort["inputs"].get(name) != sha256(fold_root / "detector/completion_manifest.json"):
            raise ValueError("P6-C1-v2 common cohort input drift: {}".format(name))
    if cohort["inputs"].get("rca_index_manifest") != sha256(
            bound_path(protocol["bindings"]["rca_index_manifest"])):
        raise ValueError("P6-C1-v2 common cohort raw-index identity drift")
    directory = root / "train_cohort"
    cases = pd.read_csv(directory / "cases.csv")
    gt = np.load(directory / "gt_features.npy", allow_pickle=False)
    detected = np.load(directory / "detected_features.npy", allow_pickle=False)
    roots = np.load(directory / "root_indices.npy", allow_pickle=False)
    if len(cases) == 0:
        raise ValueError("P6-C1-v2 common Train cohort is empty")

    def work(stage):
        arms = fit_c1_shared_rca_arms(case_ids=cases["case_id"].tolist(),
                                      gt_features=gt, detected_features=detected,
                                      root_indices=roots)
        save_conditional_logit(stage / "arm_b.npz", arms.gt_arm_b)
        save_conditional_logit(stage / "arm_c.npz", arms.detected_arm_c)
        with (stage / "shared_scaler.npz").open("xb") as stream:
            np.savez(stream, mean=arms.scaler_mean, scale=arms.scaler_scale)
        if not np.isfinite(arms.gt_arm_b.weights).all() or not np.isfinite(arms.detected_arm_c.weights).all():
            raise ValueError("P6-C1-v2 RCA arm weights are non-finite")
        return ("COMPLETE", ("arm_b.npz", "arm_b.json", "arm_c.npz", "arm_c.json",
                             "shared_scaler.npz"),
                {"train_cohort_manifest": sha256(directory / "completion_manifest.json")},
                {"common_cases": len(arms.case_ids), "same_scaler": True, "lambda": 1.0,
                 "unit_case_weights": True,
                 "arm_b_converged": bool(arms.gt_arm_b.converged),
                 "arm_c_converged": bool(arms.detected_arm_c.converged),
                 "arm_b_gradient_norm": float(arms.gt_arm_b.gradient_norm),
                 "arm_c_gradient_norm": float(arms.detected_arm_c.gradient_norm),
                 "arm_b_iterations": int(arms.gt_arm_b.iterations),
                 "arm_c_iterations": int(arms.detected_arm_c.iterations)})

    result = _stage(root, "rca", work)
    print("PASS P6-C1-v2 shared-scaler RCA arms B/C sealed",
          result["details"]["common_cases"])


def _lock_test(root: Path, protocol: Mapping[str, object]) -> None:
    from src.e2e.c1_test_scoring import score_c1_test_episodes_with_features
    from src.e2e.rca_model import load_conditional_logit
    rca = _validate_stage(root, "rca")
    _validate_stage(root, "train_cohort")
    if rca["inputs"].get("train_cohort_manifest") != sha256(
            root / "train_cohort/completion_manifest.json"):
        raise ValueError("P6-C1-v2 RCA common Train cohort input drift")
    index, index_path = _index(protocol)
    episode_path = bound_path(protocol["bindings"]["c0_test_episodes"])
    prediction_hash_only = bound_path(protocol["bindings"]["c0_test_predictions"])
    base = _base(protocol)
    episodes = pd.read_csv(episode_path)
    arm_b = load_conditional_logit(root / "rca/arm_b.npz")
    arm_c = load_conditional_logit(root / "rca/arm_c.npz")
    lock_scope = {"scope": "all_legal_c0_test_detected_episodes",
                  "label_free_columns_only": True,
                  "test_matching_read": False,
                  "test_ground_truth_read": False,
                  "missing_ranking": "counted_as_zero_correctness_stays_in_denominator",
                  "hash_only_inputs": {"c0_test_predictions": sha256(prediction_hash_only)}}

    def work(directory):
        scope, features, valid = score_c1_test_episodes_with_features(
            episodes=episodes, index=index, arm_b=arm_b, arm_c=arm_c,
            test_interval_ms=(base["split"]["boundary_ms"], base["split"]["absolute_end_ms"]))
        if (len(scope) != len(episodes)
                or scope["prediction_id"].tolist() != episodes["prediction_id"].tolist()
                or not scope.loc[valid, "scope_status"].eq("legal").all()
                or not scope.loc[scope["ranking_status"] == "complete"].index.isin(np.flatnonzero(valid)).all()
                or not scope.loc[scope["ranking_status"] == "failed", "scope_status"].eq("legal").all()
                or not scope.loc[scope["ranking_status"] == "not_applicable", "scope_status"].eq("illegal_context").all()
                or not scope.loc[scope["ranking_status"] == "complete", ["ranking_b", "ranking_c"]].ne("").all().all()
                or not scope.loc[scope["ranking_status"] == "failed", ["ranking_b", "ranking_c"]].eq("").any(axis=1).all()):
            raise ValueError("P6-C1-v2 prediction lock did not preserve every Test episode")
        _save_csv(directory / "scope_rankings.csv", scope)
        _save_npy(directory / "feature_inputs.npy", features)
        _save_npy(directory / "feature_valid.npy", valid)
        scaler = np.load(root / "rca/shared_scaler.npz", allow_pickle=False)
        lock = dict(lock_scope)
        lock.update({
            "schema_version": "p6_c1_v2_prediction_lock_v1",
            "protocol_id": PROTOCOL_ID,
            "run_id": RUN_ID,
            "scored_episodes": int(len(scope)),
            "legal_contexts": int(scope["scope_status"].eq("legal").sum()),
            "legal_ranked": int(scope["ranking_status"].eq("complete").sum()),
            "ranking_failures": int(scope["ranking_status"].eq("failed").sum()),
            "illegal_contexts": int(scope["scope_status"].eq("illegal_context").sum()),
            "ordered_episode_universe_sha256": sha256(directory / "scope_rankings.csv"),
            "ranking_b_sha256": sha256(directory / "scope_rankings.csv"),
            "feature_inputs_sha256": sha256(directory / "feature_inputs.npy"),
            "feature_valid_sha256": sha256(directory / "feature_valid.npy"),
            "arm_b_sha256": sha256(root / "rca/arm_b.npz"),
            "arm_c_sha256": sha256(root / "rca/arm_c.npz"),
            "shared_scaler_sha256": sha256(root / "rca/shared_scaler.npz"),
            "shared_scaler_mean": scaler["mean"].tolist(),
            "shared_scaler_scale": scaler["scale"].tolist(),
            "c0_test_episodes_sha256": sha256(episode_path),
            "source_sha256": source_hashes(),
        })
        write_json_new(directory / "prediction_lock.json", lock)
        return ("COMPLETE", ("scope_rankings.csv", "feature_inputs.npy", "feature_valid.npy",
                             "prediction_lock.json"),
                {"rca_manifest": sha256(root / "rca/completion_manifest.json"),
                 "raw_index_manifest": sha256(index_path),
                 "c0_test_episodes": sha256(episode_path),
                 "c0_test_predictions_hash_only": sha256(prediction_hash_only)},
                {"all_episodes": len(scope),
                 "legal_contexts": lock["legal_contexts"],
                 "feature_inputs_available": int(valid.sum()),
                 "legal_ranked": lock["legal_ranked"],
                 "ranking_failures": lock["ranking_failures"],
                 "illegal_contexts": lock["illegal_contexts"],
                 "test_matching_or_gt_read": False,
                 "all_episode_scope_before_gt_join": True})

    _stage(root, "predictions", work)
    print("LOCKED P6-C1-v2 label-free complete Test episode universe")


def _evaluate(root: Path, protocol: Mapping[str, object]) -> None:
    from src.e2e.c1_v2_c2 import evaluate_c2_full_diagnosis, quote_frozen_stage1
    from src.e2e.c1_evaluation import (
        evaluate_c1_oracle_a, evaluate_c1_prediction_lock, evaluate_c2_raw_failure,
    )
    from src.e2e.rca_model import load_conditional_logit
    lock = _validate_stage(root, "predictions")
    _validate_stage(root, "rca")
    if lock["inputs"].get("rca_manifest") != sha256(root / "rca/completion_manifest.json"):
        raise ValueError("P6-C1-v2 prediction lock RCA input drift")
    match_path = bound_path(protocol["bindings"]["c0_test_matching"])
    episode_path = bound_path(protocol["bindings"]["c0_test_episodes"])
    if (lock["inputs"].get("c0_test_episodes") != sha256(episode_path)
            or lock["inputs"].get("c0_test_predictions_hash_only") != sha256(
                bound_path(protocol["bindings"]["c0_test_predictions"]))
            or lock["inputs"].get("raw_index_manifest") != sha256(
                bound_path(protocol["bindings"]["rca_index_manifest"]))):
        raise ValueError("P6-C1-v2 prediction lock source input drift")
    scope = pd.read_csv(root / "predictions/scope_rankings.csv", keep_default_na=False)
    matching = pd.read_csv(match_path)
    episodes = pd.read_csv(episode_path)
    base = _base(protocol)
    registry = load_registry(base, ROOT)
    interval = (base["split"]["boundary_ms"], base["split"]["absolute_end_ms"])

    def work(directory):
        c1 = evaluate_c1_prediction_lock(scope=scope, matching=matching)
        index, index_path = _index(protocol)
        arm_b = load_conditional_logit(root / "rca/arm_b.npz")
        oracle_rows, oracle = evaluate_c1_oracle_a(
            locked_case_details=c1["case_details"], index=index, arm_b=arm_b,
            test_interval_ms=interval)
        c2_raw, raw_matching = evaluate_c2_raw_failure(
            episodes=episodes, scope=scope, registry=registry, test_interval_ms=interval)
        c2_full = evaluate_c2_full_diagnosis(
            scope=scope, registry=registry, matching=matching,
            test_interval_ms=interval)
        stage1 = quote_frozen_stage1(protocol)
        write_json_new(directory / "c1_results.json", to_builtin(c1))
        write_json_new(directory / "oracle_a_diagnostic.json", to_builtin(oracle))
        _save_csv(directory / "oracle_a_rankings.csv", oracle_rows)
        write_json_new(directory / "c2_stage1_reference.json", to_builtin(stage1))
        write_json_new(directory / "c2_failure_summary.json", to_builtin(c2_raw))
        _save_csv(directory / "c2_raw_matching.csv", raw_matching)
        write_json_new(directory / "c2_full_diagnosis.json", to_builtin(c2_full["summary"]))
        _save_csv(directory / "c2_failure_ledger.csv", c2_full["ledger"])
        return ("COMPLETE", ("c1_results.json", "oracle_a_diagnostic.json",
                             "oracle_a_rankings.csv", "c2_stage1_reference.json",
                             "c2_failure_summary.json", "c2_raw_matching.csv",
                             "c2_full_diagnosis.json", "c2_failure_ledger.csv"),
                {"prediction_lock": sha256(root / "predictions/completion_manifest.json"),
                 "c0_test_matching": sha256(match_path),
                 "rca_index_manifest": sha256(index_path),
                 "arm_b_model": sha256(root / "rca/arm_b.npz"),
                 "arm_c_model": sha256(root / "rca/arm_c.npz")},
                {"reused_test": True, "independent_confirmation": False,
                 "c1_n": c1["primary"]["n"], "c1_delta": c1["primary"]["delta_c_minus_b"],
                 "c2_raw_gt": c2_raw["raw_gt_complete"]})

    _stage(root, "evaluation", work)
    print("COMPLETE P6-C1-v2 C1/C2 reused-Test post-lock evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "init", "fold-input", "fold-detector",
                                           "cohort", "fit-rca", "lock-test", "evaluate"))
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    protocol = _protocol()
    if args.action == "preflight":
        _preflight(protocol)
        return
    if args.action == "init":
        _init(protocol, gpu=args.gpu)
        return
    root = run_root(protocol)
    _run_lock(root, protocol)
    if args.action in ("fold-input", "fold-detector"):
        if args.fold is None:
            parser.error("--fold is required for fold stages")
        _fold(root, protocol, args.fold, action=args.action, gpu=args.gpu)
    elif args.action == "cohort":
        _cohort(root, protocol)
    elif args.action == "fit-rca":
        _fit_rca(root, protocol)
    elif args.action == "lock-test":
        _lock_test(root, protocol)
    elif args.action == "evaluate":
        _evaluate(root, protocol)


if __name__ == "__main__":
    main()
