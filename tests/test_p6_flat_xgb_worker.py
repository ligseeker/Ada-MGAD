"""A real small XGB fit verifies the isolated worker's lock and output format."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def test_fixed_worker_fits_without_validation_labels_and_locks_scores(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/e2e/gaia_p6_c0_flat_xgb_v1.json"
    parameters = json.loads(config.read_text())
    rng = np.random.RandomState(42)
    train, validation = rng.rand(50, 20).astype(np.float32), rng.rand(35, 20).astype(np.float32)
    target = (train[:, 0] > 0.5).astype(np.int8)
    for name, array in (("fit_features", train), ("fit_targets", target), ("validation_features", validation)):
        np.save(tmp_path / (name + ".npy"), array)
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.glob("*.npy")}
    (tmp_path / "feature_lock.json").write_text(json.dumps({"files": files}))
    subprocess.run([parameters["xgb_python"], str(root / "scripts/p6/fit_c0_flat_xgb.py"),
                    "--config", str(config), "--run-dir", str(tmp_path)], check=True, capture_output=True)
    report = json.loads((tmp_path / "fit_report.json").read_text())
    scores = np.load(tmp_path / "validation_scores.npy")
    assert len(scores) == 35 and np.isfinite(scores).all()
    assert report["parameters"]["n_estimators"] == 200
    assert report["parameters"]["scale_pos_weight"] == float((target == 0).sum() / (target == 1).sum())
    assert not report["validation_labels_read"] and not report["eval_set_used"]
    assert not report["test_read"] and not report["early_stopping_used"]
    assert report["prediction_sha256"] == hashlib.sha256((tmp_path / "validation_scores.npy").read_bytes()).hexdigest()
