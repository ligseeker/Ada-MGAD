#!/usr/bin/env python3
"""One-variable patience diagnostic; consumes frozen Fit/Validation only."""

import argparse
import ast
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch

from scripts.p6.analyze_c0_causal_development import verify_completion
from scripts.p6.run_c0_onset_development import check_ablation
from scripts.p6.run_c0_trigger import build_model_args, git_head, run_training, setup_logging
from scripts.p6.run_c0_window_causal import real_window_causality_gate, source_bindings
from src.e2e.onset_trigger import OnsetDevelopmentState
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import to_builtin
from src.e2e.system_trigger_tcn import TCN_SPEC, WindowCausalTCNTrigger
from src.e2e.training_budget_audit import EpochAudit, padded_fit_counts


def verify_observer_only(current_path, frozen_path, frozen_sha):
    """Require the complete source AST to match after removing exact audit additions."""
    if sha256_file(frozen_path) != frozen_sha:
        raise ValueError("archived training source changed")
    current = ast.parse(current_path.read_text())
    function = next(node for node in current.body if isinstance(node, ast.FunctionDef)
                    and node.name == "run_training")
    if function.args.args[-1].arg != "epoch_observer":
        raise ValueError("observer interface drift")
    function.args.args.pop()
    default = function.args.defaults.pop()
    if not isinstance(default, ast.Constant) or default.value is not None:
        raise ValueError("observer must be optional")
    loop = next(node for node in function.body if isinstance(node, ast.For))
    lr = ast.parse('learning_rates = [float(group["lr"]) for group in optimizer.param_groups]').body[0]
    hook = ast.parse('if epoch_observer is not None:\n    epoch_observer(model, validation_output, entry, learning_rates, [float(group["lr"]) for group in optimizer.param_groups])').body[0]
    if ast.dump(loop.body[0]) != ast.dump(lr):
        raise ValueError("learning rate audit drift")
    loop.body.pop(0)
    hooks = [node for node in loop.body if ast.dump(node) == ast.dump(hook)]
    if len(hooks) != 1:
        raise ValueError("observer call drift")
    loop.body.remove(hooks[0])
    if ast.dump(current) != ast.dump(ast.parse(frozen_path.read_text())):
        raise ValueError("training source changed beyond observer additions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "train"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(42, 17, 2026))
    parser.add_argument("--arm", choices=("legacy_replay", "control", "budget"), default="legacy_replay")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise ValueError("launch with the fixed PYTHONHASHSEED")
    if args.gpu and not torch.cuda.is_available():
        raise RuntimeError("requested GPU is unavailable")
    config = json.loads(args.config.read_text())
    if args.arm != "legacy_replay":
        if (config.get("paired_budget", {}).get("deterministic_algorithms") is not True
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
            raise ValueError("matched deterministic protocol/workspace missing")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    if config["tcn_spec"] != TCN_SPEC or config["training"]["patience"] != 30:
        raise ValueError("budget/TCN specification drift")
    old_run = Path(config["budget"]["old_runs"][str(args.seed)])
    baseline_dir = Path(config["baseline_run"])
    verify_completion(old_run)
    verify_completion(baseline_dir)
    old_selection = json.loads((old_run / "validation_selection.json").read_text())
    old_lock = json.loads((old_run / "development_input_lock.json").read_text())
    normalized_training = dict(config["training"], patience=8)
    if (normalized_training != old_selection["training"]
            or old_selection["training"]["max_epochs"] != 30
            or old_selection["model_args"]["random_seed"] != args.seed
            or old_selection["target"] != config["trigger_label"]):
        raise ValueError("recipe changed beyond patience")
    base_path = ROOT / config["base_config"]
    base = load_config(base_path)
    data_root = Path(config["budget"]["data_root"])
    artifact_root = Path(config["budget"]["artifact_root"])
    registry = Path(config["budget"]["registry"])
    if sha256_file(registry) != base["event_registry"]["sha256"]:
        raise ValueError("registry digest drift")
    state = OnsetDevelopmentState(base, data_root, registry)
    cohort_config = dict(config, training=normalized_training)
    identity, _ = check_ablation(state, cohort_config, baseline_dir)
    manifest_path = artifact_root / "ad_data_manifest.json"
    model_args = build_model_args(config, base, json.loads(manifest_path.read_text()), args.seed, args.gpu)
    if model_args != dict(old_selection["model_args"], gpu=bool(args.gpu)):
        raise ValueError("model/input argument drift")
    for split in ("fit", "validation"):
        columns = ["case_id", "source_index", "service", "fault_type", "start_ms", "end_ms"]
        pd.testing.assert_frame_equal(pd.read_csv(old_run / (split + "_gt.csv"))[columns],
                                      state.gt_events(split)[columns].reset_index(drop=True), check_dtype=False)
        if identity[split] != old_lock["cohort"][split]:
            raise ValueError("cohort identity drift: " + split)
    bindings = source_bindings(args.config, data_root, artifact_root, registry)
    for path, expected in old_lock["source_sha256"].items():
        if "/train/" in path or path.endswith(("/graph.npy", "/ad_data_manifest.json")):
            if bindings.get(str(Path(path).resolve())) != expected:
                raise ValueError("frozen input drift: " + path)
    # Old locks also bind the earlier non-TCN baseline. Select the actual old
    # TCN run's source, rather than the first suffix match in a nested ledger.
    frozen_path = old_run.resolve().parents[3] / "scripts/p6/run_c0_trigger.py"
    frozen_sha = old_lock["source_sha256"][str(frozen_path)]
    verify_observer_only(ROOT / "scripts/p6/run_c0_trigger.py", frozen_path, frozen_sha)
    for name in ("src/model_util.py", "src/e2e/window_dynamic_graph.py", "src/e2e/system_trigger_model.py",
                 "src/e2e/system_trigger_data.py", "src/e2e/system_trigger.py", "src/e2e/event_detection.py",
                 "src/e2e/system_trigger_tcn.py", "src/e2e/onset_trigger.py", "src/e2e/trigger_development.py"):
        hashes = {sha for path, sha in old_lock["source_sha256"].items() if path.endswith("/" + name)}
        if hashes != {sha256_file(ROOT / name)}:
            raise ValueError("frozen numeric dependency drift: " + name)
    paths = [Path(__file__), base_path, frozen_path, ROOT / "util/util.py", ROOT / "src/e2e/ad_data.py",
             ROOT / "src/e2e/training_budget_audit.py", ROOT / "src/e2e/bin_trigger_decoder.py",
             ROOT / "docs/P6_TRAIN_BUDGET_DIAGNOSTIC_PLAN_20261002.md",
             ROOT / "docs/GAIA_P5_CURRENT_CONTEXT.md"]
    if args.arm != "legacy_replay":
        paths.append(ROOT / "docs/P6_DETERMINISTIC_PAIRED_BUDGET_PROTOCOL.md")
    paths += [old_run / name for name in ("completion_manifest.json", "development_input_lock.json",
                                         "training_log.json", "validation_selection.json", "validation_predictions.csv")]
    paths += [Path(old_selection["checkpoint"]["path"])]
    bindings.update({str(path.resolve()): sha256_file(path) for path in paths})
    # The wrapper has no Test action; only global interval/domain routing is shared.
    execution_commit = git_head()
    threads = int(config["budget"]["threads_by_seed"][str(args.seed)])
    workers = int(config["budget"]["threshold_workers_by_seed"][str(args.seed)])
    if args.arm != "legacy_replay" and any(os.environ.get(name) != str(threads)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("matched OMP/MKL thread environment missing")
    torch.set_num_threads(threads)
    setup_logging(args.output_dir)
    write_json(args.output_dir / "development_input_lock.json", {
        "execution_commit": execution_commit, "source_sha256": bindings, "cohort": identity,
        "protocol_id": config["protocol_id"], "effective_seed": args.seed,
        "changes": ["patience:8->30"] if args.arm != "control" else [], "observer_ast_gate": True,
        "arm": args.arm, "historical_replay_required": args.arm == "legacy_replay",
        "test_arrays_read": False, "test_trigger_labels_built": False,
        "registry_interval_read": "global interval/domain routing only; Test annotations excluded",
        "preprocessing_grade": "frozen 70 percent Train includes Detector-Validation",
        "environment": {"executable": sys.executable, "python": platform.python_version(),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "argv": sys.argv,
                        "device": torch.cuda.get_device_name(0) if args.gpu else "cpu",
                        "torch_num_threads": threads, "threshold_workers": workers,
                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS")},
    })
    for split in ("fit", "validation"):
        state.gt_events(split).to_csv(args.output_dir / (split + "_gt.csv"), index=False)
    if args.action == "train":
        observer = EpochAudit(args.output_dir, old_run, state.gt_events("validation"),
                              padded_fit_counts(state.build_dataset("fit"), 32),
                              enforce_historical_replay=args.arm == "legacy_replay")
        training = dict(config["training"], patience=8 if args.arm == "control" else 30)
        result = run_training(state, args.output_dir, model_args, training, manifest_path,
                              sha256_file(manifest_path), workers, "spawn", model_class=WindowCausalTCNTrigger,
                              epoch_observer=observer)
        if args.arm != "control" and (result["epochs_completed"] != 30 or result["stop_reason"] != "max_epochs"):
            raise ValueError("full frozen 30 epoch trajectory was not completed")
        result.update({"formal_result": False, "development_only": True,
                       "protocol_id": config["protocol_id"], "target": config["trigger_label"],
                       "model_class": "WindowCausalTCNTrigger", "tcn_spec": TCN_SPEC,
                       "effective_seed": args.seed, "arm": args.arm})
        write_json(args.output_dir / "validation_selection.json", to_builtin(result))
        device = torch.device("cuda" if args.gpu else "cpu")
        weights = torch.load(result["checkpoint"]["path"], map_location=device)
        write_json(args.output_dir / "real_window_causality_gate.json",
                   real_window_causality_gate(state, model_args, weights, device, WindowCausalTCNTrigger))
    if git_head() != execution_commit or any(sha256_file(Path(path)) != sha for path, sha in bindings.items()):
        raise ValueError("source/input changed during execution; run remains incomplete")
    write_json(args.output_dir / "completion_manifest.json", {
        "status": "COMPLETE_DEVELOPMENT_ONLY", "execution_commit": execution_commit,
        "protocol_id": config["protocol_id"], "action": args.action, "test_inference_run": False,
        "files": {str(path.relative_to(args.output_dir)): sha256_file(path)
                  for path in args.output_dir.rglob("*") if path.is_file()}})
    print(json.dumps({"status": "COMPLETE_DEVELOPMENT_ONLY", "action": args.action, "seed": args.seed}))


if __name__ == "__main__":
    main()
