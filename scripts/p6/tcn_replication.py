"""Fail-closed registration of the two initialization stability replications."""

import json
import os
from pathlib import Path

from scripts.p6.analyze_c0_causal_development import verify_completion
from src.e2e.protocol import sha256_file


REPLICATION_SPEC = {
    "seeds": [17, 2026],
    "primary_seed": 42,
    "threshold_rule": "each seed's unchanged merged selector, frozen before bin decoding",
    "time_strata": "two equal duration halves of the frozen Validation block",
    "minimum_half_recall_gain": 0.03,
    "all_seeds_must_pass": True,
    "best_seed_selection": False,
}


def check_replication_config(config, reference, seed):
    spec = config.get("replication")
    if spec is None:
        if seed is not None:
            raise ValueError("replication seed requires a registered replication protocol")
        return int(config["seed"])
    if spec["spec"] != REPLICATION_SPEC or seed not in REPLICATION_SPEC["seeds"]:
        raise ValueError("unregistered replication seed or margins")
    allowed = {"protocol_id", "output_root", "start_condition", "replication"}
    if ({key: value for key, value in config.items() if key not in allowed}
            != {key: value for key, value in reference.items() if key not in allowed}):
        raise ValueError("replication changes the candidate recipe beyond initialization")
    if config["protocol_id"] != "P6-C0-ONSET30-TCN-REPLICATION-V1":
        raise ValueError("wrong replication protocol")
    return int(seed)


def replication_bindings(config, seed, root):
    if "replication" not in config:
        return check_replication_config(config, {}, seed), {}
    spec = config["replication"]
    reference_config_path = Path(spec["reference_config"])
    reference = json.loads(reference_config_path.read_text())
    effective_seed = check_replication_config(config, reference, seed)
    if os.environ.get("PYTHONHASHSEED") != str(effective_seed):
        raise ValueError("PYTHONHASHSEED must be set before launching each seed process")
    model_dir = Path(spec["reference_model_run"])
    decoder_dir = Path(spec["reference_decoder_run"])
    for directory in (model_dir, decoder_dir):
        verify_completion(directory)
    selected = json.loads((model_dir / "validation_selection.json").read_text())
    report = json.loads((decoder_dir / "decoder_comparison.json").read_text())
    if (selected["protocol_id"] != reference["protocol_id"]
            or selected["model_args"]["random_seed"] != 42
            or selected["tcn_spec"] != reference["tcn_spec"]
            or selected["training"] != reference["training"]
            or selected["target"] != reference["trigger_label"]
            or report["protocol_id"] != reference["protocol_id"]
            or report["gt_denominator"] != 2901 or report["full_e2e_run"]
            or report["model_completion_sha256"] != sha256_file(model_dir / "completion_manifest.json")
            or report["gate_margins"] != reference["development_gate"]
            or not all(report["gates"]["onset_bins_fixed_threshold"].values())
            or json.loads((model_dir / "real_window_causality_gate.json").read_text()).get("passed") is not True):
        raise ValueError("primary candidate has no completed, bound development GO")
    paths = [reference_config_path, root / "scripts/p6/tcn_replication.py",
             root / "docs/P6_TCN_REPLICATION_PROTOCOL_20261001.md",
             root / "scripts/p6/analyze_c0_tcn_replication.py"]
    paths.extend(directory / name for directory in (model_dir, decoder_dir)
                 for name in ("completion_manifest.json",))
    paths.extend([model_dir / "validation_selection.json", model_dir / "development_input_lock.json",
                  model_dir / "real_window_causality_gate.json", decoder_dir / "decoder_comparison.json"])
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    reference_lock = json.loads((model_dir / "development_input_lock.json").read_text())
    reference_root = reference_config_path.resolve().parents[2]
    for name in ("src/e2e/system_trigger_tcn.py", "src/e2e/onset_trigger.py",
                 "src/e2e/trigger_development.py", "scripts/p6/run_c0_trigger.py"):
        expected = reference_lock["source_sha256"].get(str(reference_root / name))
        if expected is None or sha256_file(root / name) != expected:
            raise ValueError("reference candidate source drift: " + name)
        bindings[str((root / name).resolve())] = expected
    # Preserve the reference GO source identity. Global registry interval routing
    # is disclosed separately; the legacy test_read flag means no Test inference.
    for path, expected in report["source_sha256"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("reference GO binding drift: " + path)
        bindings[str(Path(path).resolve())] = expected
    return effective_seed, bindings
