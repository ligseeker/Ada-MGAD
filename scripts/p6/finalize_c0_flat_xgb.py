#!/usr/bin/env python3
"""Finalize a sealed XGB worker fit in a new run without fitting again."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.p6.run_c0_trigger import git_head, write_predictions
from scripts.p6.run_c0_window_causal import check_development_cohort
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import evaluate_system_threshold, select_system_threshold, system_score_frame, to_builtin
from src.e2e.trigger_development import TriggerDevelopmentState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitted-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--threshold-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    input_lock_path = args.fitted_run / "development_input_lock.json"
    lock = json.loads(input_lock_path.read_text())
    # The original worktree remains intact: validate its actual locked files.
    for path, sha in lock["source_sha256"].items():
        if sha256_file(Path(path)) != sha:
            raise ValueError("original execution source/input drift: " + path)
    if sha256_file(args.config) != lock["source_sha256"][str(args.config.resolve())]:
        raise ValueError("original method config identity drift")
    feature_lock = json.loads((args.fitted_run / "feature_lock.json").read_text())
    for name, sha in feature_lock["files"].items():
        if sha256_file(args.fitted_run / name) != sha:
            raise ValueError("original feature input drift")
    fit_report_path = args.fitted_run / "fit_report.json"
    report = json.loads(fit_report_path.read_text())
    for name, key in (("trigger.ubj", "model_sha256"), ("validation_scores.npy", "prediction_sha256")):
        if sha256_file(args.fitted_run / name) != report[key]:
            raise ValueError("worker output digest drift")
    config = json.loads(args.config.read_text())
    base = load_config(ROOT / config["base_config"])
    if sha256_file(args.registry) != base["event_registry"]["sha256"]:
        raise ValueError("canonical registry drift")
    state = TriggerDevelopmentState(base, args.data_root, args.registry)
    identity = check_development_cohort(state, config)
    args.output_dir.mkdir(parents=True)
    sources = dict(lock["source_sha256"])
    for name in ("scripts/p6/finalize_c0_flat_xgb.py", "scripts/p6/export_c0_xgb_logits.py",
                 "scripts/p6/run_c0_trigger.py", "src/e2e/event_detection.py", "src/e2e/system_trigger.py",
                 "src/e2e/trigger_development.py", "src/e2e/system_trigger_data.py"):
        sources[str((ROOT / name).resolve())] = sha256_file(ROOT / name)
    execution_commit = git_head()
    write_json(args.output_dir / "finalization_input_lock.json", {
        "execution_commit": execution_commit, "fit_execution_commit": lock["execution_commit"],
        "fitted_run": str(args.fitted_run.resolve()), "original_input_lock_sha256": sha256_file(input_lock_path),
        "fit_report_sha256": sha256_file(fit_report_path), "feature_lock_sha256": sha256_file(args.fitted_run / "feature_lock.json"),
        "source_sha256": sources, "cohort": identity, "refit_run": False, "test_read": False,
        "correction": "initial driver omitted mandatory raw-logit CSV field; export-only correction"})
    for name in ("trigger.ubj", "validation_scores.npy", "fit_report.json"):
        shutil.copyfile(args.fitted_run / name, args.output_dir / name)
    subprocess.run([config["xgb_python"], str(ROOT / "scripts/p6/export_c0_xgb_logits.py"),
                    "--model", str(args.output_dir / "trigger.ubj"),
                    "--features", str(args.fitted_run / "validation_features.npy"),
                    "--scores", str(args.output_dir / "validation_scores.npy"),
                    "--output", str(args.output_dir / "validation_logits.npy")],
                   env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="42"), check=True)
    write_json(args.output_dir / "validation_prediction_lock.json", {
        "model_sha256": report["model_sha256"], "score_sha256": report["prediction_sha256"],
        "logit_sha256": sha256_file(args.output_dir / "validation_logits.npy"),
        "threshold_selected": False, "refit_run": False, "test_prediction_run": False})
    dataset = state.build_dataset("validation")
    scores = np.load(args.output_dir / "validation_scores.npy", allow_pickle=False)
    logits = np.load(args.output_dir / "validation_logits.npy", allow_pickle=False)
    frame = system_score_frame("validation", dataset.prediction_times(), scores)
    gt = state.gt_events("validation")
    selection = select_system_threshold(frame, gt, workers=args.threshold_workers, start_method="spawn")
    episodes, matching, metrics = evaluate_system_threshold(frame, gt, selection.threshold)
    write_predictions(args.output_dir / "validation_predictions.csv", dataset,
                       {"system_score": scores, "logits": logits}, selection.threshold)
    gt.to_csv(args.output_dir / "validation_gt.csv", index=False)
    episodes.to_csv(args.output_dir / "validation_episodes.csv", index=False)
    matching.to_csv(args.output_dir / "validation_matching.csv", index=False)
    write_json(args.output_dir / "validation_selection.json", to_builtin({
        "formal_result": False, "stage": "detector_development", "execution_commit": execution_commit,
        "fit_execution_commit": lock["execution_commit"], "refit_run": False,
        "selected_validation_threshold": selection.threshold, "selected_validation_metrics": metrics,
        "fixed_boosting_rounds": 200, "test_used_for_selection": False, "model": report, "cohort": identity}))
    if git_head() != execution_commit or any(sha256_file(Path(path)) != sha for path, sha in sources.items()):
        raise ValueError("source/input changed during finalization")
    write_json(args.output_dir / "completion_manifest.json", {
        "status": "COMPLETE_DEVELOPMENT_ONLY", "execution_commit": execution_commit,
        "fit_execution_commit": lock["execution_commit"], "refit_run": False, "test_inference_run": False,
        "files": {str(path.relative_to(args.output_dir)): sha256_file(path)
                  for path in args.output_dir.rglob("*") if path.is_file()}})
    print(json.dumps(to_builtin(metrics), sort_keys=True))


if __name__ == "__main__":
    main()
