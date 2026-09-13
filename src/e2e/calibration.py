"""Train-only reconstruction-score calibration for the V3 detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ReconstructionCalibration:
    median: float
    mad: float
    scale: float
    train_count: int
    train_scores_sha256: str
    fit_split: str = "train"
    transform: str = "sigmoid((score-median)/(1.4826*max(MAD,1e-6)))"

    def transform_scores(self, scores) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("reconstruction scores contain non-finite values")
        normalized = (values - float(self.median)) / float(self.scale)
        normalized = np.clip(normalized, -60.0, 60.0)
        return (1.0 / (1.0 + np.exp(-normalized))).astype(np.float64)


def _score_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def fit_reconstruction_calibration(train_scores) -> ReconstructionCalibration:
    """Fit median/MAD using Train scores only."""

    values = np.asarray(train_scores, dtype=np.float64).reshape(-1)
    if not len(values):
        raise ValueError("Train reconstruction calibration requires at least one score")
    if not np.isfinite(values).all():
        raise ValueError("Train reconstruction scores contain non-finite values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * max(mad, 1e-6)
    return ReconstructionCalibration(
        median=median,
        mad=mad,
        scale=float(scale),
        train_count=int(len(values)),
        train_scores_sha256=_score_digest(values),
    )


def save_reconstruction_calibration(path: Path, calibration: ReconstructionCalibration) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(asdict(calibration), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_reconstruction_calibration(path: Path) -> ReconstructionCalibration:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"median", "mad", "scale", "train_count", "train_scores_sha256"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError("calibration artifact is missing fields: {}".format(missing))
    if str(data.get("fit_split")) != "train":
        raise ValueError("reconstruction calibration is not Train-fitted")
    return ReconstructionCalibration(
        median=float(data["median"]), mad=float(data["mad"]),
        scale=float(data["scale"]), train_count=int(data["train_count"]),
        train_scores_sha256=str(data["train_scores_sha256"]),
        fit_split=str(data.get("fit_split", "train")),
        transform=str(data.get("transform", "")),
    )
