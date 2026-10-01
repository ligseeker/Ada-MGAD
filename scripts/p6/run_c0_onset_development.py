#!/usr/bin/env python3
"""Train/Validation-only onset ablation; no preprocessing or Test entry point."""

import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.p6.analyze_c0_causal_development import verify_completion
from scripts.p6.run_c0_trigger import build_model_args, git_head, run_training, setup_logging
from scripts.p6.run_c0_window_causal import (check_development_cohort,
                                            real_window_causality_gate, source_bindings)
from src.e2e.onset_trigger import OnsetDevelopmentState
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import to_builtin, trigger_label_counts


def check_ablation(state, config, baseline_dir):
    """Verify the old targets first, then allow only the specified target change."""
    new_labels = state.labels["train"]
    state.labels["train"] = state.baseline_labels
    try:
        identity = check_development_cohort(state, config)
    finally:
        state.labels["train"] = new_labels
    selection = json.loads((baseline_dir / "validation_selection.json").read_text())
    if selection["model_args"].get("graph_batch_scope") != "window":
        raise ValueError("baseline must use a window-independent graph")
    if config["training"] != selection["training"]:
        raise ValueError("training settings differ from the repaired baseline")
    if int(config["seed"]) != int(selection["model_args"]["random_seed"]):
        raise ValueError("seed differs from the repaired baseline")
    if config["trigger_label"]["target"] != "one_bin_onset_frozen_ignore":
        raise ValueError("wrong target protocol")
    if config["trigger_label"]["positive_window_seconds"] != 30:
        raise ValueError("onset bin width must remain 30 seconds")
    for split in ("fit", "validation"):
        dataset = state.build_dataset(split)
        frozen_gt = pd.read_csv(baseline_dir / (split + "_gt.csv"))
        current_gt = state.gt_events(split)
        columns = ["case_id", "source_index", "service", "fault_type", "start_ms", "end_ms"]
        pd.testing.assert_frame_equal(current_gt[columns].reset_index(drop=True),
                                      frozen_gt[columns].reset_index(drop=True), check_dtype=False)
        new = dataset.labels_at()
        old = state.baseline_labels[dataset.sample_indices + state.window_bins - 1]
        if not np.array_equal(new == 2, old == 2):
            raise ValueError("IGNORE loss mask changed")
        identity[split].update({"baseline_labels": identity[split]["labels"],
                               "labels": trigger_label_counts(new),
                               "positive_to_negative": int(((old == 1) & (new == 0)).sum()),
                               "other_transitions": int(((old != new) & ~((old == 1) & (new == 0))).sum()),
                               "ignore_mask_exact": True,
                               "gt_identity_exact": True})
        if identity[split]["other_transitions"]:
            raise ValueError("unexpected label transition")
    return identity, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "train"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--threshold-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = json.loads(args.config.read_text())
    if config["model"].get("graph_batch_scope") != "window":
        raise ValueError("onset ablation requires one graph per window")
    if args.gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    baseline_dir = Path(config["baseline_run"])
    verify_completion(baseline_dir)
    base_path = ROOT / config["base_config"]
    base = load_config(base_path)
    if sha256_file(args.registry) != base["event_registry"]["sha256"]:
        raise ValueError("registry digest drift")
    state = OnsetDevelopmentState(base, args.data_root, args.registry)
    identity, baseline_selection = check_ablation(state, config, baseline_dir)
    manifest_path = args.artifact_root / "ad_data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model_args = build_model_args(config, base, manifest, int(config["seed"]), args.gpu)
    expected_model = dict(baseline_selection["model_args"], gpu=bool(args.gpu))
    if model_args != expected_model:
        raise ValueError("model arguments differ from the repaired baseline")
    bindings = source_bindings(args.config, args.data_root, args.artifact_root, args.registry)
    for path in (base_path, Path(__file__), ROOT / "src/e2e/onset_trigger.py",
                 ROOT / "scripts/p6/analyze_c0_causal_development.py",
                 ROOT / "docs/P6_C0_ONSET_DEVELOPMENT_PROTOCOL.md",
                 Path(config["cohort_reference"]["fit"]), Path(config["cohort_reference"]["validation"]),
                 baseline_dir / "completion_manifest.json", baseline_dir / "validation_selection.json",
                 baseline_dir / "fit_gt.csv", baseline_dir / "validation_gt.csv"):
        bindings[str(path.resolve())] = sha256_file(path)
    # The tensors must be byte-identical to the repaired baseline's inputs.
    baseline_lock = json.loads((baseline_dir / "development_input_lock.json").read_text())
    for split in ("fit", "validation"):
        if identity[split]["reference_sha256"] != baseline_lock["cohort"][split]["reference_sha256"]:
            raise ValueError("archived cohort reference changed: " + split)
    for path in (manifest_path, args.data_root / "graph.npy") + tuple(
            args.data_root / "train" / (name + ".npy") for name in ("timestamps", "metric", "log", "trace")):
        if bindings[str(path.resolve())] != baseline_lock["source_sha256"].get(str(path.resolve())):
            raise ValueError("input differs from the repaired baseline: " + str(path))
    # No encoder, graph, scoring, or optimization code may drift in this ablation.
    for name in ("src/model_util.py", "src/e2e/window_dynamic_graph.py",
                 "src/e2e/system_trigger_model.py", "src/e2e/system_trigger_data.py",
                 "src/e2e/system_trigger.py", "src/e2e/event_detection.py", "scripts/p6/run_c0_trigger.py"):
        matches = [sha for path, sha in baseline_lock["source_sha256"].items()
                   if path.endswith("/" + name)]
        if len(matches) != 1 or sha256_file(ROOT / name) != matches[0]:
            raise ValueError("baseline training/model/evaluator source differs: " + name)
    execution_commit = git_head()
    torch.set_num_threads(args.threads)
    setup_logging(args.output_dir)
    write_json(args.output_dir / "development_input_lock.json", {
        "execution_commit": execution_commit, "source_sha256": bindings, "cohort": identity,
        "protocol_id": config["protocol_id"], "test_arrays_read": False,
        "test_trigger_labels_built": False, "registry_interval_read": "routing only; Test annotations skipped",
        "preprocessing_grade": "frozen 70 percent Train includes Detector-Validation",
        "environment": {"python": platform.python_version(), "executable": sys.executable,
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0) if args.gpu else "cpu"},
    })
    for split in ("fit", "validation"):
        state.gt_events(split).to_csv(args.output_dir / (split + "_gt.csv"), index=False)
    result = {"cohort": identity}
    if args.action == "train":
        result = run_training(state, args.output_dir, model_args, config["training"],
                              manifest_path, sha256_file(manifest_path), args.threshold_workers, "spawn")
        # Correct the legacy selector's metadata before this new run is sealed.
        result.update({"formal_result": False, "development_only": True,
                       "protocol_id": config["protocol_id"], "target": config["trigger_label"],
                       "baseline_run": str(baseline_dir)})
        write_json(args.output_dir / "validation_selection.json", to_builtin(result))
        device = torch.device("cuda" if args.gpu else "cpu")
        weights = torch.load(result["checkpoint"]["path"], map_location=device)
        write_json(args.output_dir / "real_window_causality_gate.json",
                   real_window_causality_gate(state, model_args, weights, device))
    if git_head() != execution_commit or any(sha256_file(Path(path)) != sha for path, sha in bindings.items()):
        raise ValueError("source/input changed during execution; keep run incomplete")
    files = {str(path.relative_to(args.output_dir)): sha256_file(path)
             for path in args.output_dir.rglob("*") if path.is_file()}
    write_json(args.output_dir / "completion_manifest.json", {
        "status": "COMPLETE_DEVELOPMENT_ONLY", "execution_commit": execution_commit,
        "protocol_id": config["protocol_id"], "action": args.action,
        "test_inference_run": False, "files": files,
    })
    print(json.dumps(to_builtin({"action": args.action, "cohort": identity,
                                "status": "COMPLETE_DEVELOPMENT_ONLY"}), sort_keys=True))


if __name__ == "__main__":
    main()
