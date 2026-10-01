#!/usr/bin/env python3
"""Export frozen XGB margins, verifying previously locked probabilities."""

import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if xgb.__version__ != "2.1.4":
        raise ValueError("XGBoost version drift")
    model = xgb.XGBClassifier(n_jobs=1)
    model.load_model(args.model)
    if model.get_booster().num_boosted_rounds() != 200:
        raise ValueError("fixed 200-round model drift")
    features = np.load(args.features, mmap_mode="r")
    frozen = np.load(args.scores, allow_pickle=False)
    if not np.array_equal(model.predict_proba(features)[:, 1], frozen):
        raise ValueError("loaded frozen model does not exactly replay probabilities")
    logits = model.predict(features, output_margin=True)
    if len(logits) != len(frozen) or not np.isfinite(logits).all():
        raise ValueError("invalid XGB margins")
    np.save(args.output, logits, allow_pickle=False)


if __name__ == "__main__":
    main()
