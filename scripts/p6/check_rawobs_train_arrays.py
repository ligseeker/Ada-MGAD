#!/usr/bin/env python3
"""Verify a raw-mask candidate changed only Train observation fractions."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.e2e.protocol import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-data", type=Path, required=True)
    parser.add_argument("--candidate-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    report = {"status": "INCOMPLETE", "test_arrays_read": False, "bindings": {}}
    for name in ("timestamps", "log", "trace"):
        paths = [root / "train" / (name + ".npy")
                 for root in (args.baseline_data, args.candidate_data)]
        hashes = [sha256_file(path) for path in paths]
        if hashes[0] != hashes[1]:
            raise ValueError(name + " changed; this is not a one-variable mask comparison")
        report["bindings"][name] = hashes
    graphs = [sha256_file(root / "graph.npy") for root in (args.baseline_data, args.candidate_data)]
    if graphs[0] != graphs[1]:
        raise ValueError("graph changed")
    report["bindings"]["graph"] = graphs
    arrays = [np.load(root / "train/metric.npy", mmap_mode="r")
              for root in (args.baseline_data, args.candidate_data)]
    if arrays[0].shape != arrays[1].shape or arrays[0].shape[1:] != (10, 48):
        raise ValueError("Metric schema dimensions changed")
    changed = np.zeros((10, 2), dtype=np.int64)
    decreases = np.zeros((10, 2), dtype=np.float64)
    for start in range(0, len(arrays[0]), 2048):
        old, new = [np.asarray(array[start:start + 2048]) for array in arrays]
        if not np.isfinite(new).all():
            raise ValueError("nonfinite candidate Metric values")
        if not np.array_equal(old[..., :45], new[..., :45]):
            raise ValueError("base Metric numeric values changed")
        if not np.array_equal(old[..., 46], new[..., 46]):
            raise ValueError("host applicability changed")
        original, corrected = old[..., [45, 47]], new[..., [45, 47]]
        if np.any(corrected > original) or np.any(corrected < 0) or np.any(corrected > 1):
            raise ValueError("raw observation fractions violate expected bounds")
        changed += (original != corrected).sum(axis=0)
        decreases += (original - corrected).sum(axis=0)
    report.update({"status": "PASS_TRAIN_ONLY_SINGLE_MASK_CHANGE", "train_bins": len(arrays[0]),
                   "changed_bins_by_service_and_mask": changed.tolist(),
                   "mean_fraction_decrease_by_service_and_mask": (decreases / len(arrays[0])).tolist(),
                   "metric_sha256": [sha256_file(root / "train/metric.npy")
                                     for root in (args.baseline_data, args.candidate_data)]})
    write_json(args.output_dir / "train_mask_validation.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
