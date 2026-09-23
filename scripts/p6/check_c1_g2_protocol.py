#!/usr/bin/env python3
"""Read-only integrity and geometry gate for the P6-C1 G2 design lock."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "configs/e2e/gaia_p6_c1_g2_v1.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_binding(binding):
    path = (ROOT / binding["path"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("bound input missing or outside repository: {}".format(path))
    if path.stat().st_size != binding["bytes"] or sha256(path) != binding["sha256"]:
        raise ValueError("bound input drift: {}".format(path))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--require-new-run", action="store_true",
                        help="Also refuse an already existing formal C1 run directory.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (config["schema_version"] != "p6_c1_g2_design_lock_v1"
            or config["status"] != "DESIGN_LOCKED_EXECUTION_BLOCKED_G3"
            or config["evidence_grade"] != "prefix_fit_detector_fixed_transductive_raw_catalog"):
        raise ValueError("G2 design identity/status drift")
    if config["run_id"] != "c1-prefix-oos-v1-seed42" or config["seed"] != 42:
        raise ValueError("G2 run identity drift")
    if config["rca"]["shared_scaler_fit"] != "common_train_gt_candidate_rows":
        raise ValueError("shared RCA scaler decision drift")
    if config["detector"]["source"] != "p6_c0_frozen_architecture_and_training":
        raise ValueError("fold detector decision drift")
    budget = config["resource_budget"]
    expected_budget = {"ad_preprocessing_workers": 24, "detector_fits": 3,
                       "detector_loader_workers": 2, "fold_preprocessing_fits": 3,
                       "formal_run_repair": "new_protocol_version_and_new_run_id",
                       "hyperparameter_search": False, "max_epochs_per_fold": 30,
                       "preprocessing_start_method": "spawn", "rca_fits": 2,
                       "torch_threads": 8}
    if budget != expected_budget:
        raise ValueError("G2 compute budget drift")

    for binding in config["bindings"].values():
        checked_binding(binding)
    g1_path = checked_binding(config["bindings"]["g1_ledger"])
    g1 = json.loads(g1_path.read_text(encoding="utf-8"))
    if g1["status"] != "STATIC_INVENTORY_COMPLETE_EXECUTION_NO_GO_AS_IS":
        raise ValueError("G1 status drift")
    if g1["preferred_static_candidate"] != "fit_only_3fold":
        raise ValueError("G1 preferred candidate drift")
    checked_binding(g1["audit_source"])
    for binding in g1["input_files"].values():
        checked_binding(binding)
    folds = g1["designs"]["fit_only_3fold"]["folds"]
    if len(folds) != 3 or len(config["folds"]) != 3:
        raise ValueError("fold count drift")
    for expected, fixed in zip(folds, config["folds"]):
        upper = expected["generation_gt_feature_and_rca_context_eligible"]
        if (fixed["fold"] != expected["fold"]
                or fixed["intervals_ms"] != expected["intervals_ms"]
                or fixed["static_gt_context_upper_bound"] != upper
                or fixed["minimum_actual_common_cases"] != (upper + 1) // 2):
            raise ValueError("fold design differs from G1 static ledger")
    if sum(fold["static_gt_context_upper_bound"] for fold in config["folds"]) != 4254:
        raise ValueError("static GT support drift")

    raw = json.loads(checked_binding(config["bindings"]["raw_content_manifest"]).read_text(encoding="utf-8"))
    base = json.loads(checked_binding(config["bindings"]["base_config"]).read_text(encoding="utf-8"))
    if base["source"]["ada_rca_commit"] != config["rca"]["source_commit"]:
        raise ValueError("Ada-RCA source commit drift")
    for key in ("window_seconds", "bin_seconds", "feature_dimension", "l2_lambda"):
        if base["rca"][key] != config["rca"][key]:
            raise ValueError("RCA representation/model policy drift: {}".format(key))
    trigger = json.loads(checked_binding(config["bindings"]["c0_trigger_config"]).read_text(encoding="utf-8"))
    for key in ("grid_seconds", "history_seconds", "tolerance_seconds"):
        c1_key = "matching_tolerance_seconds" if key == "tolerance_seconds" else key
        if trigger["evaluation"][key] != config["detector"][c1_key]:
            raise ValueError("detector evaluation policy drift: {}".format(key))
    for key in ("max_epochs", "patience"):
        if trigger["training"][key] != config["detector"][key]:
            raise ValueError("detector training policy drift: {}".format(key))
    if (raw["schema_version"] != "p6_c1_raw_content_inventory_v1"
            or raw["config_sha256"] != config["bindings"]["base_config"]["sha256"]
            or raw["raw_root"] != str(Path(base["gaia_raw_root"]).resolve())
            or raw["file_count"] != len(raw["files"])
            or raw["total_bytes"] != sum(row["bytes"] for row in raw["files"])):
        raise ValueError("raw content inventory/header drift")
    names = [row["path"] for row in raw["files"]]
    if names != sorted(set(names)):
        raise ValueError("raw content inventory names are not sorted and unique")
    canonical = json.dumps(raw["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != raw["records_sha256"]:
        raise ValueError("raw content inventory record digest drift")

    c0 = json.loads(checked_binding(config["bindings"]["c0_manifest"]).read_text(encoding="utf-8"))
    if c0["checkpoint_sha256"] != config["bindings"]["c0_checkpoint"]["sha256"]:
        raise ValueError("C0 checkpoint binding drift")
    if c0["source_artifacts"]["base_config"]["sha256"] != config["bindings"]["base_config"]["sha256"]:
        raise ValueError("C0 base-config binding drift")
    if config["test_policy"] != "read_frozen_c0_episodes_only_no_test_model_inference":
        raise ValueError("Test source policy drift")
    if (config["bindings"]["c0_test_predictions"]["role"]
            != "provenance_hash_only_contains_trigger_label"
            or config["bindings"]["c0_test_episodes"]["role"] != "test_scoring_source"
            or config["bindings"]["c0_test_matching"]["role"] != "evaluator_only"):
        raise ValueError("Test label firewall roles drift")
    episodes_path = checked_binding(config["bindings"]["c0_test_episodes"])
    with episodes_path.open(newline="", encoding="utf-8") as stream:
        fields = csv.DictReader(stream).fieldnames
    if fields != ["prediction_id", "split", "t_hat", "episode_end_time",
                  "positive_bins", "system_score", "threshold"]:
        raise ValueError("Test episode label-free schema drift")
    predictions_path = checked_binding(config["bindings"]["c0_test_predictions"])
    with predictions_path.open(newline="", encoding="utf-8") as stream:
        fields = csv.DictReader(stream).fieldnames
    if "trigger_label" not in fields:
        raise ValueError("Test prediction file label-firewall premise drift")
    if args.require_new_run:
        run_root = (ROOT / config["output_root"]).resolve()
        if (not run_root.is_relative_to(ROOT / "experiments/p6/c1_detector_aligned")
                or run_root.name != config["run_id"] or run_root.exists()):
            raise ValueError("formal C1 run directory is invalid or already exists")
    print("PASS G2 design and bound inputs; C1 execution remains blocked pending G3")


if __name__ == "__main__":
    main()
