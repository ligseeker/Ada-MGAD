#!/usr/bin/env python3
"""Fit the fixed XGB trigger; this worker has no event/Validation-label input."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import sklearn
import xgboost as xgb


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if xgb.__version__ != config["expected_xgboost_version"]:
        raise ValueError("XGBoost environment drift")
    lock = json.loads((args.run_dir / "feature_lock.json").read_text())
    for name, sha in lock["files"].items():
        if digest(args.run_dir / name) != sha:
            raise ValueError("feature input drift: " + name)
    train = np.load(args.run_dir / "fit_features.npy", mmap_mode="r")
    labels = np.load(args.run_dir / "fit_targets.npy", allow_pickle=False)
    validation = np.load(args.run_dir / "validation_features.npy", mmap_mode="r")
    if len(train) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("invalid Fit labels")
    weight = float((labels == 0).sum() / (labels == 1).sum())
    parameters = dict(config["xgb_parameters"], scale_pos_weight=weight)
    model = xgb.XGBClassifier(**parameters)
    started = time.monotonic()
    print("Fitting fixed {}-tree trigger: {} rows x {} columns, CPU threads={}".format(
        parameters["n_estimators"], *train.shape, parameters["n_jobs"]), flush=True)
    model.fit(train, labels)
    model_path = args.run_dir / "trigger.ubj"
    model.save_model(model_path)
    scores = model.predict_proba(validation)[:, 1]
    if len(scores) != len(validation) or not np.isfinite(scores).all():
        raise ValueError("invalid Validation scores")
    np.save(args.run_dir / "validation_scores.npy", scores, allow_pickle=False)
    logits = model.predict(validation, output_margin=True)
    if len(logits) != len(scores) or not np.isfinite(logits).all():
        raise ValueError("invalid Validation logits")
    np.save(args.run_dir / "validation_logits.npy", logits, allow_pickle=False)
    positions = np.linspace(0, len(validation) - 1, 32, dtype=int)
    rows = np.asarray(validation[positions]).copy()
    original = model.predict_proba(rows)[:, 1]
    singleton = np.concatenate([model.predict_proba(row[None])[:, 1] for row in rows])
    reordered = model.predict_proba(rows[::-1])[:, 1][::-1]
    changed = rows.copy()
    changed[1:] = changed[1:] * 20 + 30
    modified = model.predict_proba(changed)[:, 1]
    errors = {"singletons": float(abs(original - singleton).max()),
              "reordered": float(abs(original - reordered).max()),
              "companion_perturbation": float(abs(original[0] - modified[0]))}
    if any(value != 0 for value in errors.values()):
        raise ValueError("XGB row independence gate failed")
    result = {"elapsed_seconds": time.monotonic() - started, "parameters": parameters,
              "fit_rows": len(train), "validation_rows": len(validation), "features": train.shape[1],
              "validation_labels_read": False, "test_read": False, "eval_set_used": False,
              "early_stopping_used": False, "causality_gate": {"atol_probability": 0, "errors": errors},
              "environment": {"python": platform.python_version(), "executable": sys.executable,
                              "xgboost": xgb.__version__, "numpy": np.__version__,
                              "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                              "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                              "xgboost_build": xgb.build_info()},
              "prediction_sha256": digest(args.run_dir / "validation_scores.npy"),
              "logit_sha256": digest(args.run_dir / "validation_logits.npy"),
              "model_sha256": digest(model_path)}
    (args.run_dir / "fit_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "FIT_COMPLETE", "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
