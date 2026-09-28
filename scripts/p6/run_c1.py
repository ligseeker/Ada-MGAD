#!/usr/bin/env python3
"""Stage-by-stage P6-C1 runner; full stages must be invoked manually.

Stages are exclusive and sealed. A failed stage remains INCOMPLETE; it cannot
be resumed or overwritten. A formal rerun requires a new protocol/run ID.
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

from src.e2e.c1_fold_preprocessing import (
    G2_CONFIG, G2_CONFIG_SHA256, _bound_path, _sha256, _write_json_new,
    materialize_c1_fold, validate_c1_fold, verify_raw_content,
)
from src.e2e.gaia_rca_adapter import GaiaRcaRawIndex, validate_raw_index_manifest
from src.e2e.protocol import load_registry
from src.e2e.system_trigger import to_builtin


SMOKE_REPORT = ROOT / "docs/P6_C1_G3_RAW_SMOKE_20260928.json"
SOURCE_PATHS = (
    "scripts/p6/run_c1.py", "scripts/p6/smoke_c1_raw_adapters.py",
    "src/e2e/c1_fold_preprocessing.py", "src/e2e/c1_fold_detector_data.py",
    "src/e2e/c1_fold_supervision.py", "src/e2e/c1_fold_generation.py",
    "src/e2e/c1_fold_detector.py", "src/e2e/c1_oos_matching.py",
    "src/e2e/c1_common_cohort.py", "src/e2e/c1_shared_rca.py",
    "src/e2e/c1_test_scoring.py", "src/e2e/c1_evaluation.py",
    "src/e2e/rca_model.py", "src/e2e/rca_features.py",
    "src/e2e/gaia_rca_adapter.py", "src/e2e/system_trigger.py",
    "src/e2e/system_trigger_model.py", "src/e2e/event_detection.py",
    "src/e2e/system_trigger_data.py", "src/e2e/ad_data.py",
    "src/e2e/protocol.py", "src/e2e/parallel.py",
    "src/e2e/gaia_preprocessing/raw.py", "src/e2e/gaia_preprocessing/metric.py",
    "src/e2e/gaia_preprocessing/logs.py", "src/e2e/gaia_preprocessing/traces.py",
    "src/e2e/gaia_preprocessing/schema.py", "src/e2e/gaia_preprocessing/materialize.py",
    "src/model_util.py", "util/util.py", "util/GAIA/gaia.ini",
)


def _protocol():
    if _sha256(G2_CONFIG) != G2_CONFIG_SHA256:
        raise ValueError("C1 corrected G2 design lock drift")
    return json.loads(G2_CONFIG.read_text(encoding="utf-8"))


def _run_root(protocol) -> Path:
    root = (ROOT / protocol["output_root"]).resolve()
    if root != (ROOT / "experiments/p6/c1_detector_aligned/c1-prefix-oos-v1-seed42").resolve():
        raise ValueError("C1 output root differs from G2 lock")
    return root


def _sources():
    return {name: _sha256(ROOT / name) for name in SOURCE_PATHS}


def _smoke():
    report = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
    if (report.get("status") != "PASS" or report.get("workers_compared") != [1, 24]
            or report.get("start_method") != "spawn"
            or report.get("formal_gaia_prefix_tested") is not False
            or report.get("source_sha256") != _sha256(ROOT / "scripts/p6/smoke_c1_raw_adapters.py")):
        raise ValueError("C1 G3 bounded raw smoke report/source drift")
    return report


def _run_lock(root: Path, protocol):
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("C1 formal run initialization is incomplete")
    payload = json.loads((root / "run_lock.json").read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    if (payload.get("schema_version") != "p6_c1_run_lock_v1"
            or payload.get("status") != "RUN_RESERVED"
            or payload.get("run_id") != protocol["run_id"]
            or payload.get("git_head") != head
            or payload.get("device") not in ("cpu", "cuda")
            or payload.get("protocol_sha256") != _sha256(G2_CONFIG)
            or payload.get("source_sha256") != _sources()
            or payload.get("smoke_report_sha256") != _sha256(SMOKE_REPORT)):
        raise ValueError("C1 run lock or source snapshot drift")
    if root != _run_root(protocol):
        raise ValueError("C1 run root drift")
    return payload


def _save_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n")


def _save_npy(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def _stage(root: Path, name: str, work):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=False)
    _write_json_new(directory / "INCOMPLETE.json", {"status": "INCOMPLETE", "stage": name})
    try:
        status, files, inputs, details = work(directory)
        if status not in ("COMPLETE", "NO_GO"):
            raise ValueError("C1 stage returned an invalid completion status")
        run_lock = json.loads((root / "run_lock.json").read_text(encoding="utf-8"))
        if _sources() != run_lock["source_sha256"]:
            raise ValueError("C1 source changed during formal stage")
        manifest = {"schema_version": "p6_c1_stage_manifest_v1", "stage": name,
                    "status": status, "run_lock_sha256": _sha256(root / "run_lock.json"),
                    "source_sha256": _sources(), "inputs": inputs, "details": to_builtin(details),
                    "files": {filename: {"bytes": (directory / filename).stat().st_size,
                                         "sha256": _sha256(directory / filename)}
                              for filename in files}}
        _write_json_new(directory / "completion_manifest.json", manifest)
        (directory / "INCOMPLETE.json").unlink()
        return manifest
    except Exception as exc:
        try:
            _write_json_new(directory / "failure.json", {"status": "INCOMPLETE",
                                                       "error_type": type(exc).__name__,
                                                       "message": str(exc)})
        except OSError:
            pass
        raise


def _validate_stage(root: Path, name: str, *, require_complete: bool = True):
    directory = root / name
    if (directory / "INCOMPLETE.json").exists():
        raise ValueError("C1 {} stage incomplete".format(name))
    manifest = json.loads((directory / "completion_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "p6_c1_stage_manifest_v1"
            or manifest.get("stage") != name
            or manifest.get("status") not in (("COMPLETE",) if require_complete else ("COMPLETE", "NO_GO"))
            or manifest.get("run_lock_sha256") != _sha256(root / "run_lock.json")
            or manifest.get("source_sha256") != _sources()):
        raise ValueError("C1 {} completion/source drift or NO_GO".format(name))
    for filename, record in manifest["files"].items():
        path = directory / filename
        if (filename != Path(filename).name or path.is_symlink() or not path.is_file()
                or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]):
            raise ValueError("C1 {} output drift: {}".format(name, filename))
    return manifest


def _base(protocol):
    return json.loads(_bound_path(protocol["bindings"]["base_config"]).read_text(encoding="utf-8"))


def _index(protocol):
    path = _bound_path(protocol["bindings"]["rca_index_manifest"])
    validate_raw_index_manifest(path)
    return GaiaRcaRawIndex.from_manifest(path), path


def _preflight(protocol):
    if _run_root(protocol).exists():
        raise ValueError("C1 formal run root already exists; no overwrite")
    _smoke()
    for binding in protocol["bindings"].values():
        _bound_path(binding)
    base = _base(protocol)
    manifest_path = _bound_path(protocol["bindings"]["raw_content_manifest"])
    verify_raw_content(manifest_path, Path(base["gaia_raw_root"]).resolve())
    validate_raw_index_manifest(_bound_path(protocol["bindings"]["rca_index_manifest"]))
    _bound_path(protocol["bindings"]["c0_test_episodes"])
    _bound_path(protocol["bindings"]["c0_test_predictions"])
    print("PASS C1 G3 source, smoke, bound raw bytes and index; run root new")


def _init(protocol, *, gpu: bool):
    root = _run_root(protocol)
    if root.exists():
        raise ValueError("C1 formal run root already exists; no overwrite")
    if gpu:
        import torch
        if not torch.cuda.is_available():
            raise ValueError("C1 CUDA was requested for the run but is unavailable")
    _preflight(protocol)
    root.mkdir(parents=True, exist_ok=False)
    _write_json_new(root / "INCOMPLETE.json", {"status": "INCOMPLETE", "stage": "init"})
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    _write_json_new(root / "run_lock.json", {
        "schema_version": "p6_c1_run_lock_v1", "protocol_sha256": _sha256(G2_CONFIG),
        "source_sha256": _sources(), "smoke_report_sha256": _sha256(SMOKE_REPORT),
        "git_head": head, "run_id": protocol["run_id"], "status": "RUN_RESERVED",
        "device": "cuda" if gpu else "cpu"})
    (root / "folds").mkdir()
    (root / "INCOMPLETE.json").unlink()
    print("RESERVED", root)


def _cohort(root: Path, protocol):
    from src.e2e.c1_oos_matching import match_c1_oos_episodes
    from src.e2e.c1_common_cohort import build_c1_common_cohort, G2_MINIMUM_BY_FOLD
    from src.e2e.c1_fold_detector import validate_c1_fold_detector
    locked_minimum = {int(fold["fold"]): int(fold["minimum_actual_common_cases"])
                      for fold in protocol["folds"]}
    if locked_minimum != G2_MINIMUM_BY_FOLD:
        raise ValueError("C1 cohort minimum differs from frozen G2 folds")
    def work(directory):
        sealed = []
        input_hashes = {}
        for fold_number in (1, 2, 3):
            fold_root = root / "folds/fold_{:02d}".format(fold_number)
            validate_c1_fold_detector(fold_root=fold_root, protocol_path=G2_CONFIG)
            fold = validate_c1_fold(fold_root, protocol_path=G2_CONFIG)
            episodes = pd.read_csv(fold_root / "detector/generation_episodes.csv")
            interval = tuple(fold["segments"]["generation"]["interval_ms"])
            sealed.append((fold_number, interval, episodes))
            input_hashes["fold_{:02d}_detector".format(fold_number)] = _sha256(
                fold_root / "detector/completion_manifest.json")
        base = _base(protocol)
        registry = load_registry(base, ROOT)
        index, index_path = _index(protocol)
        input_hashes["rca_index_manifest"] = _sha256(index_path)
        matched = [match_c1_oos_episodes(
            fold=fold_number, interval_ms=interval, episodes=episodes, registry=registry)
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
    print(result["status"], "C1 common Train coverage", result["details"]["common_case_coverage"])


def _fit_rca(root: Path, protocol):
    from src.e2e.c1_shared_rca import fit_c1_shared_rca_arms
    from src.e2e.c1_fold_detector import validate_c1_fold_detector
    from src.e2e.rca_model import save_conditional_logit
    cohort = _validate_stage(root, "train_cohort")
    for fold_number in (1, 2, 3):
        fold_root = root / "folds/fold_{:02d}".format(fold_number)
        validate_c1_fold_detector(fold_root=fold_root, protocol_path=G2_CONFIG)
        name = "fold_{:02d}_detector".format(fold_number)
        if cohort["inputs"].get(name) != _sha256(fold_root / "detector/completion_manifest.json"):
            raise ValueError("C1 common cohort input drift: {}".format(name))
    if cohort["inputs"].get("rca_index_manifest") != _sha256(
            _bound_path(protocol["bindings"]["rca_index_manifest"])):
        raise ValueError("C1 common cohort raw-index identity drift")
    directory = root / "train_cohort"
    cases = pd.read_csv(directory / "cases.csv")
    gt = np.load(directory / "gt_features.npy", allow_pickle=False)
    detected = np.load(directory / "detected_features.npy", allow_pickle=False)
    roots = np.load(directory / "root_indices.npy", allow_pickle=False)

    def work(stage):
        arms = fit_c1_shared_rca_arms(case_ids=cases["case_id"].tolist(),
                                      gt_features=gt, detected_features=detected,
                                      root_indices=roots)
        save_conditional_logit(stage / "arm_b.npz", arms.gt_arm_b)
        save_conditional_logit(stage / "arm_c.npz", arms.detected_arm_c)
        with (stage / "shared_scaler.npz").open("xb") as stream:
            np.savez(stream, mean=arms.scaler_mean, scale=arms.scaler_scale)
        return ("COMPLETE", ("arm_b.npz", "arm_b.json", "arm_c.npz", "arm_c.json",
                             "shared_scaler.npz"),
                {"train_cohort_manifest": _sha256(directory / "completion_manifest.json")},
                {"common_cases": len(arms.case_ids), "same_scaler": True,
                 "lambda": 1.0, "unit_case_weights": True})

    _stage(root, "rca", work)
    print("PASS C1 shared-scaler RCA arms B/C sealed")


def _lock_test(root: Path, protocol):
    from src.e2e.c1_test_scoring import score_c1_test_episodes_with_features
    from src.e2e.rca_model import load_conditional_logit
    rca = _validate_stage(root, "rca")
    _validate_stage(root, "train_cohort")
    if rca["inputs"].get("train_cohort_manifest") != _sha256(
            root / "train_cohort/completion_manifest.json"):
        raise ValueError("C1 RCA common Train cohort input drift")
    index, index_path = _index(protocol)
    episode_path = _bound_path(protocol["bindings"]["c0_test_episodes"])
    prediction_hash_only = _bound_path(protocol["bindings"]["c0_test_predictions"])
    base = _base(protocol)
    episodes = pd.read_csv(episode_path)
    arm_b = load_conditional_logit(root / "rca/arm_b.npz")
    arm_c = load_conditional_logit(root / "rca/arm_c.npz")

    def work(directory):
        scope, features, valid = score_c1_test_episodes_with_features(
            episodes=episodes, index=index, arm_b=arm_b, arm_c=arm_c,
            test_interval_ms=(base["split"]["boundary_ms"], base["split"]["absolute_end_ms"]))
        if (len(scope) != len(episodes) or scope["prediction_id"].tolist() != episodes["prediction_id"].tolist()
                or not scope.loc[valid, "scope_status"].eq("legal").all()
                or not scope.loc[scope["ranking_status"] == "complete"].index.isin(np.flatnonzero(valid)).all()
                or not scope.loc[scope["ranking_status"] == "failed", "scope_status"].eq("legal").all()
                or not scope.loc[scope["ranking_status"] == "not_applicable", "scope_status"].eq("illegal_context").all()
                or not scope.loc[scope["ranking_status"] == "complete", ["ranking_b", "ranking_c"]].ne("").all().all()
                or not scope.loc[scope["ranking_status"] == "failed", ["ranking_b", "ranking_c"]].eq("").any(axis=1).all()):
            raise ValueError("C1 prediction lock did not preserve every Test episode")
        _save_csv(directory / "scope_rankings.csv", scope)
        _save_npy(directory / "feature_inputs.npy", features)
        _save_npy(directory / "feature_valid.npy", valid)
        return ("COMPLETE", ("scope_rankings.csv", "feature_inputs.npy", "feature_valid.npy"),
                {"rca_manifest": _sha256(root / "rca/completion_manifest.json"),
                 "raw_index_manifest": _sha256(index_path),
                 "c0_test_episodes": _sha256(episode_path),
                 "c0_test_predictions_hash_only": _sha256(prediction_hash_only)},
                {"all_episodes": len(scope), "legal_contexts": int(scope["scope_status"].eq("legal").sum()),
                 "feature_inputs_available": int(valid.sum()),
                 "legal_ranked": int(scope["ranking_status"].eq("complete").sum()),
                 "ranking_failures": int(scope["ranking_status"].eq("failed").sum()),
                 "illegal_contexts": int(scope["scope_status"].eq("illegal_context").sum()),
                 "test_matching_or_gt_read": False,
                 "all_episode_scope_before_gt_join": True})

    _stage(root, "predictions", work)
    print("LOCKED C1 label-free complete Test episode universe")


def _evaluate(root: Path, protocol):
    from src.e2e.c1_evaluation import (
        evaluate_c1_prediction_lock, evaluate_c1_oracle_a, evaluate_c2_raw_failure,
    )
    from src.e2e.rca_model import load_conditional_logit
    lock = _validate_stage(root, "predictions")
    _validate_stage(root, "rca")
    if lock["inputs"].get("rca_manifest") != _sha256(root / "rca/completion_manifest.json"):
        raise ValueError("C1 prediction lock RCA input drift")
    match_path = _bound_path(protocol["bindings"]["c0_test_matching"])
    episode_path = _bound_path(protocol["bindings"]["c0_test_episodes"])
    if (lock["inputs"].get("c0_test_episodes") != _sha256(episode_path)
            or lock["inputs"].get("c0_test_predictions_hash_only") != _sha256(
                _bound_path(protocol["bindings"]["c0_test_predictions"]))
            or lock["inputs"].get("raw_index_manifest") != _sha256(
                _bound_path(protocol["bindings"]["rca_index_manifest"]))):
        raise ValueError("C1 prediction lock source input drift")
    scope = pd.read_csv(root / "predictions/scope_rankings.csv", keep_default_na=False)
    matching = pd.read_csv(match_path)
    episodes = pd.read_csv(episode_path)
    base = _base(protocol)
    registry = load_registry(base, ROOT)

    def work(directory):
        c1 = evaluate_c1_prediction_lock(scope=scope, matching=matching)
        index, index_path = _index(protocol)
        arm_b = load_conditional_logit(root / "rca/arm_b.npz")
        oracle_rows, oracle = evaluate_c1_oracle_a(
            locked_case_details=c1["case_details"], index=index, arm_b=arm_b,
            test_interval_ms=(base["split"]["boundary_ms"], base["split"]["absolute_end_ms"]))
        c2, raw_matching = evaluate_c2_raw_failure(
            episodes=episodes, scope=scope, registry=registry,
            test_interval_ms=(base["split"]["boundary_ms"], base["split"]["absolute_end_ms"]))
        _write_json_new(directory / "c1_results.json", to_builtin(c1))
        _write_json_new(directory / "oracle_a_diagnostic.json", to_builtin(oracle))
        _save_csv(directory / "oracle_a_rankings.csv", oracle_rows)
        _write_json_new(directory / "c2_failure_summary.json", to_builtin(c2))
        _save_csv(directory / "c2_raw_matching.csv", raw_matching)
        return ("COMPLETE", ("c1_results.json", "oracle_a_diagnostic.json",
                             "oracle_a_rankings.csv", "c2_failure_summary.json", "c2_raw_matching.csv"),
                {"prediction_lock": _sha256(root / "predictions/completion_manifest.json"),
                 "c0_test_matching": _sha256(match_path),
                 "rca_index_manifest": _sha256(index_path),
                 "arm_b_model": _sha256(root / "rca/arm_b.npz")},
                {"reused_test": True, "independent_confirmation": False,
                 "c1_n": c1["primary"]["n"], "c1_delta": c1["primary"]["delta_c_minus_b"],
                 "c2_raw_gt": c2["raw_gt_complete"]})

    _stage(root, "evaluation", work)
    print("COMPLETE C1/C2 reused-Test post-lock evaluation")


def main():
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
    root = _run_root(protocol)
    run_lock = _run_lock(root, protocol)
    if args.action in ("fold-input", "fold-detector"):
        if args.fold is None:
            parser.error("--fold is required for fold stages")
        fold_root = root / "folds/fold_{:02d}".format(args.fold)
        if args.action == "fold-input":
            materialize_c1_fold(protocol_path=G2_CONFIG, fold_number=args.fold,
                                fold_root=fold_root, runtime={"workers": 24,
                                                              "start_method": "spawn",
                                                              "chunk_rows": 100000})
            print("COMPLETE C1 fold {} prefix-fitted inputs".format(args.fold))
        else:
            from src.e2e.c1_fold_detector import train_c1_fold_detector
            if args.gpu != (run_lock["device"] == "cuda"):
                raise ValueError("C1 fold detector device differs from reserved run lock")
            train_c1_fold_detector(fold_root=fold_root, protocol_path=G2_CONFIG,
                                   gpu=args.gpu, threshold_workers=8)
            print("COMPLETE C1 fold {} Fit/Selection/Generation".format(args.fold))
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
