#!/usr/bin/env python3
"""Evaluate the already locked backdated Test prediction with CSV dtype correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.e2e.c1_evaluation import evaluate_c1_prediction_lock
from src.e2e.c1_v2_c2 import evaluate_c2_full_diagnosis
from src.e2e.protocol import load_registry
from src.e2e.system_trigger import to_builtin

SOURCE = Path("/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2")
PRED_RUN = ROOT / "experiments/p6/c1_anchor_backdate/c1-anchor-backdate-v1-delta25621"
BASE_RUN = ROOT / "experiments/p6/c1_z2_xgb/c1-z2-xgb-v1-seed20260826"
OUTPUT = ROOT / "experiments/p6/c1_anchor_backdate_eval_correction/c1-anchor-backdate-eval-v1"
SOURCE_FILES = (
    "scripts/p6/evaluate_c1_anchor_backdate_correction.py",
    "scripts/p6/run_c1_anchor_backdate_test.py", "src/e2e/c1_evaluation.py",
    "src/e2e/c1_v2_c2.py", "src/e2e/system_trigger.py",
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


def check_locked_prediction():
    run_lock_path = PRED_RUN / "run_lock.json"
    pred_lock_path = PRED_RUN / "predictions/prediction_lock.json"
    run_lock = json.loads(run_lock_path.read_text(encoding="utf-8"))
    pred_lock = json.loads(pred_lock_path.read_text(encoding="utf-8"))
    if (run_lock["status"] != "RUN_RESERVED" or run_lock["offset_ms"] != 25621
            or run_lock["test_gt_or_matching_read"] is not False
            or pred_lock["status"] != "PREDICTION_LOCKED"
            or pred_lock["test_gt_or_matching_read"] is not False
            or pred_lock["all_episodes"] != 4214 or pred_lock["legal_ranked"] != 4213
            or pred_lock["illegal_context"] != 1 or pred_lock["ranking_failure"] != 0
            or run_lock["source_sha256"]["scripts/p6/run_c1_anchor_backdate_test.py"] !=
            sha256(ROOT / "scripts/p6/run_c1_anchor_backdate_test.py")
            or pred_lock["train_model_sha256"] != sha256(PRED_RUN / "train/model_state.ubj")
            or pred_lock["scope_rankings_sha256"] != sha256(PRED_RUN / "predictions/scope_rankings.csv")
            or pred_lock["shifted_features_sha256"] != sha256(PRED_RUN / "predictions/shifted_feature_inputs.npy")):
        raise ValueError("locked Test prediction changed or is incomplete")
    for name, expected in pred_lock["feature_shards_sha256"].items():
        if sha256(PRED_RUN / "predictions/features" / name) != expected:
            raise ValueError("locked Test feature shard changed: " + name)
    if (PRED_RUN / "evaluation").exists():
        raise ValueError("original evaluator unexpectedly produced an evaluation directory")
    return run_lock, pred_lock


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=str(ROOT)).strip():
        raise ValueError("commit correction code before evaluation")
    run_lock, pred_lock = check_locked_prediction()
    scope = pd.read_csv(PRED_RUN / "predictions/scope_rankings.csv", keep_default_na=False)
    old_scope = pd.read_csv(BASE_RUN / "predictions/scope_rankings.csv", keep_default_na=False)
    if (len(scope) != 4214 or scope.prediction_id.tolist() != old_scope.prediction_id.tolist()
            or scope.t_hat.tolist() != old_scope.t_hat.tolist()
            or scope.ranking_b.tolist() != old_scope.ranking_xgb.tolist()
            or scope.scope_status.tolist() != old_scope.scope_status.tolist()
            or scope.ranking_status.tolist() != old_scope.ranking_status.tolist()):
        raise ValueError("baseline XGB ranking or Test episode cohort differs case by case")
    matching_path = SOURCE / "experiments/p6/system_event_trigger/test_matching.csv"
    # The original XGB evaluator uses pandas' default NA handling. The failed
    # v1 evaluator used keep_default_na=False, coercing nullable t_hat to str.
    matching = pd.read_csv(matching_path)
    config_path = SOURCE / "configs/e2e/gaia_p5_v3_preprocessing_v2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    registry_path = SOURCE / "artifacts/p5/v3/protocol/gt_event_registry.csv"
    registry = load_registry(config, SOURCE)
    interval = (int(config["split"]["boundary_ms"]),
                int(config["split"]["absolute_end_ms"]))
    c1 = evaluate_c1_prediction_lock(scope=scope, matching=matching)
    c2 = evaluate_c2_full_diagnosis(scope=scope, registry=registry,
                                    matching=matching, test_interval_ms=interval)
    old_c1 = json.loads((BASE_RUN / "evaluation/c1_paired_results.json").read_text())
    old_c2 = json.loads((BASE_RUN / "evaluation/c2_full_diagnosis.json").read_text())
    if (c1["primary"]["n"] != 4197
            or c1["primary"]["correct_b"] != old_c1["primary"]["correct_c"]
            or c2["summary"]["gt_population"] != old_c2["gt_population"]
            or c2["summary"]["predicted_episodes"] != old_c2["predicted_episodes"]):
        raise ValueError("C1/C2 comparator cohort replay failed")
    for top_k in (1, 3, 5):
        key = "@{}".format(top_k)
        if c2["summary"]["arms"]["b"]["metrics"][key] != old_c2["arms"]["c"]["metrics"][key]:
            raise ValueError("XGB full E2E baseline failed exact replay at " + key)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_new(OUTPUT / "c1_paired_results.json", c1)
    write_new(OUTPUT / "c2_full_diagnosis.json", c2["summary"])
    c2["ledger"].to_csv(OUTPUT / "c2_failure_ledger.csv", index=False)
    write_new(OUTPUT / "correction_report.json", {
        "status": "COMPLETE_CORRECTED_EVALUATION",
        "error_in_original_evaluate": "Test matching read with keep_default_na=False made nullable t_hat strings such as 1626963480000.0; evaluator int64 conversion failed before writing output",
        "correction": "Read test_matching.csv with pandas default NA handling, identical to prior XGB evaluator; reuse exactly the locked v1 Test predictions",
        "original_prediction_run": str(PRED_RUN),
        "original_run_lock_sha256": sha256(PRED_RUN / "run_lock.json"),
        "prediction_lock_sha256": sha256(PRED_RUN / "predictions/prediction_lock.json"),
        "prediction_scope_sha256": pred_lock["scope_rankings_sha256"],
        "baseline_case_ranking_replay": "PASS",
        "baseline_metric_replay": "PASS",
        "evidence_grade": "exploratory reused Test; not independent confirmation",
        "git_head": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=str(ROOT),
                                            universal_newlines=True).strip(),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "test_gt_input_sha256": {str(path): sha256(path) for path in
                                 (matching_path, config_path, registry_path)},
        "independent_test_confirmation": False,
    })
    write_new(OUTPUT / "completion_manifest.json", {
        "status": "COMPLETE", "original_prediction_lock_sha256": sha256(
            PRED_RUN / "predictions/prediction_lock.json"),
        "c1_paired_results_sha256": sha256(OUTPUT / "c1_paired_results.json"),
        "c2_full_diagnosis_sha256": sha256(OUTPUT / "c2_full_diagnosis.json"),
        "c2_failure_ledger_sha256": sha256(OUTPUT / "c2_failure_ledger.csv"),
        "correction_report_sha256": sha256(OUTPUT / "correction_report.json"),
    })
    print("COMPLETE_CORRECTED_EVALUATION", c1["primary"],
          c2["summary"]["arms"]["c"]["metrics"]["@1"])


if __name__ == "__main__":
    main()
