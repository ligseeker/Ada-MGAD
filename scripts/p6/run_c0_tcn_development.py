#!/usr/bin/env python3
"""Conditional encoder-only TCN experiment after the onset/decoder ablations."""

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.p6.analyze_c0_causal_development import verify_completion
from scripts.p6.run_c0_onset_development import check_ablation
from scripts.p6.run_c0_trigger import build_model_args, git_head, run_training, setup_logging
from scripts.p6.run_c0_window_causal import real_window_causality_gate, source_bindings
from scripts.p6.tcn_replication import replication_bindings
from src.e2e.onset_trigger import OnsetDevelopmentState
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import to_builtin
from src.e2e.system_trigger_tcn import TCN_SPEC, WindowCausalTCNTrigger


def verify_training_factory_only(current_path, frozen_path, frozen_sha):
    """The optimizer/loss/selection loop must differ only in model construction."""
    if sha256_file(frozen_path) != frozen_sha:
        raise ValueError("frozen training source drift")
    original = ast.parse(frozen_path.read_text())
    successor = ast.parse(current_path.read_text())
    function = next(node for node in successor.body if isinstance(node, ast.FunctionDef)
                    and node.name == "run_training")
    if function.args.args[-1].arg != "model_class":
        raise ValueError("unexpected training factory interface")
    function.args.args.pop()
    function.args.defaults.pop()
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "model_class"]
    if len(calls) != 1:
        raise ValueError("unexpected model construction changes")
    calls[0].func.id = "SystemEventTrigger"
    if ast.dump(original) != ast.dump(successor):
        raise ValueError("training engine changed beyond the model factory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--threshold-workers", type=int, default=8)
    parser.add_argument("--replication-seed", type=int, choices=(17, 2026))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = json.loads(args.config.read_text())
    effective_seed, replication_sources = replication_bindings(config, args.replication_seed, ROOT)
    if config["tcn_spec"] != TCN_SPEC:
        raise ValueError("fixed TCN architecture differs from its protocol")
    baseline_dir = Path(config["baseline_run"])
    reference_dir = Path(config["encoder_reference_run"])
    decoder_dir = Path(config["decoder_reference_run"])
    onset_report_path = Path(config["onset_reference_report"])
    for directory in (baseline_dir, reference_dir, decoder_dir):
        verify_completion(directory)
    e1 = json.loads((reference_dir / "validation_selection.json").read_text())
    e2 = json.loads((decoder_dir / "decoder_comparison.json").read_text())
    onset_report = json.loads(onset_report_path.read_text())
    baseline = json.loads((baseline_dir / "validation_selection.json").read_text())
    margins = config["development_gate"]
    gate_names = {"precision_floor", "recall_gain", "f1_gain", "single_onset_recall_decline_limit"}
    if (set(onset_report["gates"]) != gate_names
            or set(e2["gates"]) != {"onset_bins_fixed_threshold", "onset_bins_validation_selected"}
            or any(set(gates) != gate_names for gates in e2["gates"].values())
            or e2["test_read"] or e2["gt_denominator"] != 2901
            or onset_report["status"] != "NO_GO_DEVELOPMENT"):
        raise ValueError("incomplete or invalid E1/E2 decision")
    if (onset_report["test_read"] or onset_report["gt_denominator"] != 2901
            or onset_report["gate_margins"] != margins
            or onset_report["results"]["candidate"]["completion_sha256"] != sha256_file(reference_dir / "completion_manifest.json")
            or onset_report["results"]["baseline"]["completion_sha256"] != sha256_file(baseline_dir / "completion_manifest.json")):
        raise ValueError("E1 decision source/denominator/gate mismatch")
    if all(onset_report["gates"].values()) or any(all(gates.values()) for gates in e2["gates"].values()):
        raise ValueError("an onset/decoder candidate already passed; validate it before another encoder")
    if e1["training"] != config["training"]:
        raise ValueError("TCN experiment changes the training budget")
    base_path = ROOT / config["base_config"]
    base = load_config(base_path)
    if sha256_file(args.registry) != base["event_registry"]["sha256"]:
        raise ValueError("registry drift")
    state = OnsetDevelopmentState(base, args.data_root, args.registry)
    identity, baseline_selection = check_ablation(state, config, baseline_dir)
    manifest_path = args.artifact_root / "ad_data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model_args = build_model_args(config, base, manifest, effective_seed, args.gpu)
    if model_args != dict(e1["model_args"], gpu=bool(args.gpu), random_seed=effective_seed):
        raise ValueError("embedding, graph, head or input arguments changed")
    old_lock = json.loads((baseline_dir / "development_input_lock.json").read_text())
    for split in ("fit", "validation"):
        if identity[split]["reference_sha256"] != old_lock["cohort"][split]["reference_sha256"]:
            raise ValueError("archived cohort reference drift: " + split)
    frozen_path, frozen_sha = next((Path(path), sha) for path, sha in old_lock["source_sha256"].items()
                                  if path.endswith("/scripts/p6/run_c0_trigger.py"))
    verify_training_factory_only(ROOT / "scripts/p6/run_c0_trigger.py", frozen_path, frozen_sha)
    bindings = source_bindings(args.config, args.data_root, args.artifact_root, args.registry)
    bindings.update(replication_sources)
    bindings[str((ROOT / "scripts/p6/tcn_replication.py").resolve())] = sha256_file(ROOT / "scripts/p6/tcn_replication.py")
    for path in (Path(__file__), base_path, ROOT / "src/e2e/system_trigger_tcn.py",
                 ROOT / "src/e2e/onset_trigger.py", ROOT / "scripts/p6/run_c0_onset_development.py",
                 ROOT / "docs/P6_C0_TCN_DEVELOPMENT_PROTOCOL.md", reference_dir / "completion_manifest.json",
                 onset_report_path,
                 reference_dir / "validation_selection.json", decoder_dir / "completion_manifest.json",
                 decoder_dir / "decoder_comparison.json", baseline_dir / "completion_manifest.json",
                 frozen_path, Path(config["cohort_reference"]["fit"]), Path(config["cohort_reference"]["validation"])):
        bindings[str(path.resolve())] = sha256_file(path)
    for path, old_sha in old_lock["source_sha256"].items():
        if "/train/" in path or path.endswith("/graph.npy") or path.endswith("/ad_data_manifest.json"):
            if bindings.get(str(Path(path).resolve())) != old_sha:
                raise ValueError("frozen input drift: " + path)
    for name in ("src/model_util.py", "src/e2e/window_dynamic_graph.py", "src/e2e/system_trigger_model.py",
                 "src/e2e/system_trigger_data.py", "src/e2e/system_trigger.py", "src/e2e/event_detection.py"):
        expected = [sha for path, sha in old_lock["source_sha256"].items() if path.endswith("/" + name)]
        if len(expected) != 1 or sha256_file(ROOT / name) != expected[0]:
            raise ValueError("non-encoder source drift: " + name)
    execution_commit = git_head()
    torch.set_num_threads(args.threads)
    setup_logging(args.output_dir)
    write_json(args.output_dir / "development_input_lock.json", {
        "execution_commit": execution_commit, "source_sha256": bindings, "cohort": identity,
        "protocol_id": config["protocol_id"], "tcn_spec": TCN_SPEC,
        "model_class": "WindowCausalTCNTrigger", "training_factory_only_verified": True,
        "test_arrays_read": False, "test_trigger_labels_built": False,
        "effective_seed": effective_seed, "reference_seed": 42,
        "replication_changes": ["random_seed"] if replication_sources else [],
        "registry_interval_read": "global interval/domain columns for split routing; no Test service/fault annotations",
        "preprocessing_grade": "frozen 70 percent Train includes Detector-Validation",
    })
    for split in ("fit", "validation"):
        state.gt_events(split).to_csv(args.output_dir / (split + "_gt.csv"), index=False)
    result = run_training(state, args.output_dir, model_args, config["training"], manifest_path,
                          sha256_file(manifest_path), args.threshold_workers, "spawn",
                          model_class=WindowCausalTCNTrigger)
    result.update({"formal_result": False, "development_only": True, "protocol_id": config["protocol_id"],
                   "target": config["trigger_label"], "model_class": "WindowCausalTCNTrigger", "tcn_spec": TCN_SPEC,
                   "effective_seed": effective_seed})
    write_json(args.output_dir / "validation_selection.json", to_builtin(result))
    device = torch.device("cuda" if args.gpu else "cpu")
    weights = torch.load(result["checkpoint"]["path"], map_location=device)
    real_gate = real_window_causality_gate(state, model_args, weights, device, WindowCausalTCNTrigger)
    write_json(args.output_dir / "real_window_causality_gate.json", real_gate)
    if real_gate.get("passed") is not True:
        raise ValueError("actual-window independence gate failed; keep run incomplete")
    if git_head() != execution_commit or any(sha256_file(Path(path)) != sha for path, sha in bindings.items()):
        raise ValueError("source/input drift; run remains incomplete")
    write_json(args.output_dir / "completion_manifest.json", {
        "status": "COMPLETE_DEVELOPMENT_ONLY", "execution_commit": execution_commit,
        "test_inference_run": False, "protocol_id": config["protocol_id"],
        "files": {str(path.relative_to(args.output_dir)): sha256_file(path)
                  for path in args.output_dir.rglob("*") if path.is_file()},
    })
    print(json.dumps({"status": "COMPLETE_DEVELOPMENT_ONLY", "metrics": result["selected_validation_metrics"]}))


if __name__ == "__main__":
    main()
