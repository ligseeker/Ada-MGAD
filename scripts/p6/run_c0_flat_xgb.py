#!/usr/bin/env python3
"""Single fixed XGB detector experiment on the frozen Fit/Validation cohort."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.p6.run_c0_trigger import git_head, write_predictions
from scripts.p6.run_c0_window_causal import check_development_cohort, source_bindings
from src.e2e.flat_trigger_features import flatten_trigger_windows
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import (TRIGGER_IGNORE, TRIGGER_POSITIVE, evaluate_system_threshold,
                                    select_system_threshold, system_score_frame, to_builtin)
from src.e2e.trigger_development import TriggerDevelopmentState


def write_features(dataset, path, positions):
    width = dataset.window_bins * (np.prod(dataset.metric.shape[1:])
                                  + np.prod(dataset.log.shape[1:]) + np.prod(dataset.trace.shape[1:]))
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                      shape=(len(positions), int(width)))
    for start in range(0, len(positions), 256):
        chosen = positions[start:start + 256]
        rows = flatten_trigger_windows(dataset.metric, dataset.log, dataset.trace,
                                       dataset.sample_indices[chosen], dataset.window_bins)
        if not np.isfinite(rows).all():
            raise ValueError("nonfinite detector features")
        matrix[start:start + len(rows)] = rows
    matrix.flush()
    return {"rows": len(positions), "columns": int(width), "float32_bytes": int(matrix.nbytes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--threshold-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = json.loads(args.config.read_text())
    base_path = ROOT / config["base_config"]
    base = load_config(base_path)
    if sha256_file(args.registry) != base["event_registry"]["sha256"]:
        raise ValueError("canonical registry drift")
    state = TriggerDevelopmentState(base, args.data_root, args.registry)
    identity = check_development_cohort(state, config)
    args.output_dir.mkdir(parents=True)
    bindings = source_bindings(args.config, args.data_root, args.artifact_root, args.registry)
    for name in ("scripts/p6/run_c0_flat_xgb.py", "scripts/p6/fit_c0_flat_xgb.py",
                 "src/e2e/flat_trigger_features.py", config["base_config"]):
        bindings[str((ROOT / name).resolve())] = sha256_file(ROOT / name)
    execution_commit = git_head()
    write_json(args.output_dir / "development_input_lock.json", {
        "execution_commit": execution_commit, "source_sha256": bindings, "cohort": identity,
        "test_arrays_read": False, "test_labels_built": False,
        "preprocessing_grade": "frozen 70 percent Train includes Detector-Validation"})
    fit, validation = state.build_dataset("fit"), state.build_dataset("validation")
    fit_labels = fit.labels_at()
    positions = np.flatnonzero(fit_labels != TRIGGER_IGNORE)
    np.save(args.output_dir / "fit_targets.npy", (fit_labels[positions] == TRIGGER_POSITIVE).astype(np.int8))
    print("Materializing lossless causal windows; no new preprocessing fit", flush=True)
    geometry = {"fit": write_features(fit, args.output_dir / "fit_features.npy", positions),
                "validation": write_features(validation, args.output_dir / "validation_features.npy",
                                               np.arange(len(validation)))}
    feature_files = ("fit_features.npy", "fit_targets.npy", "validation_features.npy")
    write_json(args.output_dir / "feature_lock.json", {
        "files": {name: sha256_file(args.output_dir / name) for name in feature_files},
        "geometry": geometry, "layout": "metric[L,N,48],log[L,N,32],trace[L,N,N,8]; C order",
        "validation_labels_available_to_fit_worker": False, "test_read": False})
    environment = dict(os.environ, PYTHONHASHSEED=str(config["seed"]), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([config["xgb_python"], str(ROOT / "scripts/p6/fit_c0_flat_xgb.py"),
                    "--run-dir", str(args.output_dir.resolve()), "--config", str(args.config.resolve())],
                   env=environment, check=True)
    report = json.loads((args.output_dir / "fit_report.json").read_text())
    for name, key in (("trigger.ubj", "model_sha256"), ("validation_scores.npy", "prediction_sha256"),
                      ("validation_logits.npy", "logit_sha256")):
        if sha256_file(args.output_dir / name) != report[key]:
            raise ValueError("worker model/prediction digest drift")
    write_json(args.output_dir / "validation_prediction_lock.json", {
        "execution_commit": execution_commit, "config_sha256": sha256_file(args.config),
        "model_sha256": report["model_sha256"], "score_sha256": report["prediction_sha256"],
        "logit_sha256": report["logit_sha256"],
        "validation_threshold_selected": False, "test_prediction_run": False})
    scores = np.load(args.output_dir / "validation_scores.npy", allow_pickle=False)
    logits = np.load(args.output_dir / "validation_logits.npy", allow_pickle=False)
    times = validation.prediction_times()
    ground_truth = state.gt_events("validation")
    ground_truth.to_csv(args.output_dir / "validation_gt.csv", index=False)
    frame = system_score_frame("validation", times, scores)
    selected = select_system_threshold(frame, ground_truth, workers=args.threshold_workers, start_method="spawn")
    episodes, matching, metrics = evaluate_system_threshold(frame, ground_truth, selected.threshold)
    write_predictions(args.output_dir / "validation_predictions.csv", validation,
                       {"sample_index": validation.sample_indices, "prediction_available_time": times,
                        "system_score": scores, "logits": logits}, selected.threshold)
    episodes.to_csv(args.output_dir / "validation_episodes.csv", index=False)
    matching.to_csv(args.output_dir / "validation_matching.csv", index=False)
    write_json(args.output_dir / "validation_selection.json", to_builtin({
        "formal_result": False, "stage": "detector_development", "execution_commit": execution_commit,
        "selected_validation_threshold": selected.threshold, "selected_validation_metrics": metrics,
        "fixed_boosting_rounds": config["xgb_parameters"]["n_estimators"], "test_used_for_selection": False,
        "model": report, "cohort": identity}))
    if git_head() != execution_commit or any(sha256_file(Path(path)) != sha for path, sha in bindings.items()):
        raise ValueError("source/input changed during experiment")
    write_json(args.output_dir / "completion_manifest.json", {
        "status": "COMPLETE_DEVELOPMENT_ONLY", "execution_commit": execution_commit,
        "test_inference_run": False,
        "files": {str(path.relative_to(args.output_dir)): sha256_file(path)
                  for path in args.output_dir.rglob("*") if path.is_file()}})
    print(json.dumps(to_builtin(metrics), sort_keys=True))


if __name__ == "__main__":
    main()
