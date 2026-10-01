#!/usr/bin/env python3
"""Fit annotation-decoding references, explicitly neither predictions nor ceilings."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e2e.bin_trigger_decoder import evaluate_bin_threshold
from src.e2e.onset_trigger import OnsetDevelopmentState
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import evaluate_system_threshold, prediction_time_grid, system_score_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = load_config(args.base_config)
    if sha256_file(args.registry) != config["event_registry"]["sha256"]:
        raise ValueError("registry drift")
    state = OnsetDevelopmentState(config, args.data_root, args.registry)
    indices = state.sample_indices("fit") + state.window_bins - 1
    times = prediction_time_grid(state.timestamps["train"])[indices]
    gt = state.gt_events("fit")
    bindings = {str(path.resolve()): sha256_file(path) for path in
                (Path(__file__), args.base_config, args.registry, args.data_root / "train/timestamps.npy",
                 ROOT / "src/e2e/onset_trigger.py", ROOT / "src/e2e/trigger_development.py",
                 ROOT / "src/e2e/bin_trigger_decoder.py", ROOT / "src/e2e/event_detection.py",
                 ROOT / "src/e2e/system_trigger.py")}
    args.output_dir.mkdir(parents=True)
    disclaimer = "GT annotation decoding reference only; not model inference or an achievable upper bound"
    write_json(args.output_dir / "scope_lock.json", {"source_sha256": bindings,
               "analysis_split": "fit", "gt_events": len(gt), "fit_windows": len(indices),
               "test_read": False, "disclaimer": disclaimer})
    results = {}
    for name, labels, bins in (("recent60_merged", state.baseline_labels, False),
                               ("onset30_merged", state.labels["train"], False),
                               ("onset30_independent_bins", state.labels["train"], True)):
        scores = system_score_frame("fit", times, (labels[indices] == 1).astype(float))
        evaluator = evaluate_bin_threshold if bins else evaluate_system_threshold
        predictions, matching, metrics = evaluator(scores, gt, .5)
        cases = matching.loc[matching.case_id.notna()]
        if not cases.case_id.is_unique or set(cases.case_id) != set(gt.case_id):
            raise ValueError("reference omitted a GT case")
        results[name] = metrics
    write_json(args.output_dir / "reference.json", {"results": results, "disclaimer": disclaimer,
               "fit_only_analysis": True, "test_read": False, "source_sha256": bindings})
    if any(sha256_file(Path(path)) != sha for path, sha in bindings.items()):
        raise ValueError("source/input drift")
    write_json(args.output_dir / "completion_manifest.json", {
        "status": "COMPLETE_DEVELOPMENT_ONLY", "test_inference_run": False,
        "files": {str(path.relative_to(args.output_dir)): sha256_file(path)
                  for path in args.output_dir.rglob("*") if path.is_file()}})
    print(json.dumps({"results": results, "disclaimer": disclaimer}, sort_keys=True))


if __name__ == "__main__":
    main()
