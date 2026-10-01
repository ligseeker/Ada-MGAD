#!/usr/bin/env python3
"""Audit an archived C0 checkpoint or train a window-independent successor."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.p6.run_c0_trigger import (build_model_args, git_head, run_training,
                                      score_dataset, setup_logging, write_predictions)
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import (evaluate_system_threshold, select_system_threshold,
                                    system_score_frame, to_builtin, trigger_label_counts)
from src.e2e.system_trigger_model import SystemEventTrigger
from src.e2e.trigger_development import TriggerDevelopmentState


def real_window_causality_gate(state, model_args, weights, device):
    """Check real Validation inputs before accepting the new inference contract."""
    dataset = state.build_dataset("validation")
    positions = np.linspace(0, len(dataset) - 1, 32, dtype=int)
    batch = {key: torch.as_tensor(np.stack([dataset[int(i)][key] for i in positions]),
                                  device=device)
             for key in ("data_node", "data_log", "data_edge")}
    args = dict(model_args, graph_batch_scope="window", batch_size=32)
    graph = np.load(state.data_root / "graph.npy", allow_pickle=False)
    model = SystemEventTrigger(graph, **args).to(device).eval()
    model.load_state_dict(weights)
    changed = {key: value.clone() for key, value in batch.items()}
    for value in changed.values():
        value[1:] = value[1:] * 20 + 30
    permutation = torch.arange(31, -1, -1, device=device)
    errors = {}
    with torch.no_grad():
        reference = model(batch)[0]
        errors["future_companion_perturbation"] = float(abs(reference[0] - model(changed)[0][0]))
        reordered = model({key: value[permutation] for key, value in batch.items()})[0]
        errors["reordered_batch"] = float((reference - reordered[permutation]).abs().max())
        for size in (1, 2):
            local = SystemEventTrigger(graph, **dict(args, batch_size=size)).to(device).eval()
            local.load_state_dict(weights)
            outputs = [local({key: value[start:start + size] for key, value in batch.items()})[0]
                       for start in range(0, 32, size)]
            errors["batch_size_" + str(size)] = float((reference - torch.cat(outputs)).abs().max())
    # Fixed in the protocol before any checkpoint/metric audit.
    tolerance = 1e-5
    report = {"atol_logits": tolerance, "rtol_logits": 0, "max_absolute_errors": errors,
              "validation_sample_indices": dataset.sample_indices[positions].tolist(),
              "passed": all(value <= tolerance for value in errors.values())}
    if not report["passed"]:
        raise ValueError("real-window causality gate failed: " + json.dumps(report))
    return report


def source_bindings(config_path, data_root, artifact_root, registry_path):
    paths = [Path(config_path), Path(registry_path),
             Path(artifact_root) / "ad_data_manifest.json", Path(data_root) / "graph.npy"]
    paths += [Path(data_root) / "train" / (name + ".npy")
              for name in ("timestamps", "metric", "log", "trace")]
    paths += [ROOT / name for name in (
        "src/model_util.py", "src/e2e/window_dynamic_graph.py",
        "src/e2e/system_trigger_model.py", "src/e2e/trigger_development.py",
        "src/e2e/system_trigger.py", "src/e2e/system_trigger_data.py",
        "src/e2e/event_detection.py", "scripts/p6/run_c0_trigger.py",
        "scripts/p6/run_c0_window_causal.py")]
    return {str(path.resolve()): sha256_file(path) for path in paths}


def check_development_cohort(state, config):
    report = {}
    for split in ("fit", "validation"):
        dataset = state.build_dataset(split)
        reference = Path(config["cohort_reference"][split])
        frozen = pd.read_csv(reference, usecols=["sample_index", "prediction_available_time", "trigger_label"])
        if (not np.array_equal(dataset.sample_indices, frozen["sample_index"].to_numpy())
                or not np.array_equal(dataset.prediction_times(), frozen["prediction_available_time"].to_numpy())
                or not np.array_equal(dataset.labels_at(), frozen["trigger_label"].to_numpy())):
            raise ValueError(split + " window/time/label identity differs from archived C0")
        events = state.gt_events(split)
        if len(events) != config["expected_cohort"][split + "_events"]:
            raise ValueError(split + " event cohort count differs")
        report[split] = {"windows": len(dataset), "gt_events": len(events),
                         "labels": trigger_label_counts(dataset.labels_at()),
                         "reference": str(reference), "reference_sha256": sha256_file(reference)}
    return report


def checkpoint_audit(state, config, output, gpu, workers):
    selection_path = Path(config["legacy_selection"])
    selection = json.loads(selection_path.read_text())
    checkpoint = Path(selection["checkpoint"]["path"])
    if sha256_file(checkpoint) != selection["checkpoint"]["sha256"]:
        raise ValueError("legacy checkpoint digest drift")
    model_args = dict(selection["model_args"])
    model_args["gpu"] = bool(gpu)
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    graph = np.load(state.data_root / "graph.npy", allow_pickle=False)
    dataset = state.build_dataset("validation")
    ground_truth = state.gt_events("validation")
    weights = torch.load(checkpoint, map_location=device)
    result = {"legacy_selection_sha256": sha256_file(selection_path),
              "checkpoint_sha256": sha256_file(checkpoint), "scopes": {}}
    for scope in ("batch", "window"):
        model_args["graph_batch_scope"] = scope
        model = SystemEventTrigger(graph, **model_args).to(device)
        model.load_state_dict(weights)
        predictions = score_dataset(model, dataset, int(model_args["batch_size"]), 2,
                                    device, "validation")
        frame = system_score_frame("validation", predictions["prediction_available_time"],
                                   predictions["system_score"])
        threshold = float(selection["selected_validation_threshold"])
        _, matching, metrics = evaluate_system_threshold(frame, ground_truth, threshold)
        write_predictions(output / (scope + "_predictions.csv"), dataset, predictions, threshold)
        matching.to_csv(output / (scope + "_fixed_threshold_matching.csv"), index=False)
        entry = {"legacy_threshold": threshold, "fixed_threshold_metrics": metrics}
        if scope == "batch":
            expected = selection["selected_validation_metrics"]
            for key in ("true_positive_events", "false_positive_events", "false_negative_events"):
                if int(metrics[key]) != int(expected[key]):
                    raise ValueError("legacy checkpoint does not replay its selected metrics")
        else:
            chosen = select_system_threshold(frame, ground_truth, workers=workers)
            _, matched, selected_metrics = evaluate_system_threshold(frame, ground_truth, chosen.threshold)
            matched.to_csv(output / "window_validation_selected_matching.csv", index=False)
            entry.update({"validation_selected_threshold": chosen.threshold,
                          "validation_selected_metrics": selected_metrics})
        result["scopes"][scope] = entry
    write_json(output / "checkpoint_audit.json", to_builtin(result))
    write_json(output / "real_window_causality_gate.json",
               real_window_causality_gate(state, model_args, weights, device))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "train"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--threads", default=8, type=int)
    parser.add_argument("--threshold-workers", default=8, type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["model"].get("graph_batch_scope") != "window":
        raise ValueError("this successor requires one graph per window")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA is unavailable; no silent CPU fallback")
    torch.set_num_threads(args.threads)
    setup_logging(args.output_dir)
    base_path = ROOT / config["base_config"]
    base = load_config(base_path)
    if sha256_file(args.registry) != base["event_registry"]["sha256"]:
        raise ValueError("canonical registry digest drift")
    state = TriggerDevelopmentState(base, args.data_root, args.registry)
    identity = check_development_cohort(state, config)
    state.registry.to_csv(args.output_dir / "development_registry.csv", index=False)
    for split in ("fit", "validation"):
        state.gt_events(split).to_csv(args.output_dir / (split + "_gt.csv"), index=False)
    bindings = source_bindings(args.config, args.data_root, args.artifact_root, args.registry)
    bindings[str(base_path)] = sha256_file(base_path)
    execution_commit = git_head()
    write_json(args.output_dir / "development_input_lock.json", {
        "execution_commit": execution_commit, "source_sha256": bindings, "cohort": identity,
        "test_arrays_read": False, "test_trigger_labels_built": False,
        "registry_interval_read": "split routing only; Test annotation rows skipped",
        "preprocessing_grade": "frozen 70 percent Train includes Detector-Validation",
    })
    if args.action == "audit":
        result = checkpoint_audit(state, config, args.output_dir, args.gpu, args.threshold_workers)
    else:
        manifest_path = args.artifact_root / "ad_data_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        model_args = build_model_args(config, base, manifest, int(config["seed"]), args.gpu)
        result = run_training(state, args.output_dir, model_args, config["training"],
                              manifest_path, sha256_file(manifest_path), args.threshold_workers, "spawn")
        device = torch.device("cuda" if args.gpu else "cpu")
        weights = torch.load(result["checkpoint"]["path"], map_location=device)
        write_json(args.output_dir / "real_window_causality_gate.json",
                   real_window_causality_gate(state, model_args, weights, device))
    if git_head() != execution_commit or any(sha256_file(Path(path)) != digest
                                             for path, digest in bindings.items()):
        raise ValueError("source/input changed during execution; run remains incomplete")
    files = {str(path.relative_to(args.output_dir)): sha256_file(path)
             for path in args.output_dir.rglob("*") if path.is_file()}
    write_json(args.output_dir / "completion_manifest.json", {
        "status": "COMPLETE_DEVELOPMENT_ONLY", "execution_commit": execution_commit,
        "action": args.action, "test_inference_run": False, "files": files,
    })
    print(json.dumps(to_builtin({"action": args.action, "status": "COMPLETE_DEVELOPMENT_ONLY",
                                "result": result}), sort_keys=True))


if __name__ == "__main__":
    main()
