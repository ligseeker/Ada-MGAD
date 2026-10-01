#!/usr/bin/env python3
"""One preregistered normal forecast candidate, screened solely inside Fit."""

import argparse
import json
import logging
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.p6.analyze_c0_causal_development import verify_completion
from src.e2e.normal_forecast import (FORECAST_SPEC, MetricForecastDataset,
                                    NormalForecastState, NormalMetricForecaster, residual_score)
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import evaluate_system_threshold, system_score_frame, to_builtin
from src.e2e.system_trigger_failure_audit import build_failure_ledger, stratified_summary


def current_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def validate_config(config):
    training = {"epochs": 20, "batch_size": 256, "num_workers": 2,
                "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .0001,
                "huber_delta": 1., "grad_clip_norm": 10., "checkpoint": "last_epoch",
                "shuffle_normal_training": True}
    gates = {"precision_floor": .9, "recall_floor": .6, "f1_floor": .72,
             "f1_gain_vs_persistence": .02, "normal_holdout_positive_fraction_limit": .01}
    if (config["protocol_id"] != "P6-NORMAL-FORECAST-FIT-SCREEN-V1"
            or config["base_config"] != "configs/e2e/gaia_p5_v3_preprocessing_v2.json"
            or config["seed"] != 42 or config["model_spec"] != FORECAST_SPEC
            or config["training"] != training or config["screen_gate"] != gates
            or config["normal_quantile"] != .995 or config["min_normal_windows"] != 512
            or config["min_clean_memory_cases"] != 30):
        raise ValueError("configuration differs from the frozen Fit screen protocol")


def input_bindings(args, config):
    files = [Path(__file__), args.config, ROOT / config["base_config"], args.registry,
             ROOT / "docs/P6_NORMAL_FORECAST_FIT_SCREEN_PROTOCOL.md",
             ROOT / "src/e2e/normal_forecast.py", ROOT / "src/e2e/protocol.py",
             ROOT / "src/e2e/system_trigger.py", ROOT / "src/e2e/event_detection.py",
             ROOT / "src/e2e/system_trigger_failure_audit.py",
             ROOT / "scripts/p6/analyze_c0_causal_development.py",
             args.data_root / "train/timestamps.npy", args.data_root / "train/metric.npy",
             args.artifact_root / "ad_data_manifest.json"]
    return {str(path.resolve()): sha256_file(path) for path in files}


def validate_start_condition(config):
    refs = {name: Path(path) for name, path in config["references"].items()}
    for directory in refs.values():
        verify_completion(directory)
    tcn = json.loads((refs["tcn_run"] / "validation_selection.json").read_text())
    tcn_lock = json.loads((refs["tcn_run"] / "development_input_lock.json").read_text())
    total = json.loads((refs["tcn_total_comparison"] / "comparison.json").read_text())
    decoder = json.loads((refs["tcn_decoder_comparison"] / "decoder_comparison.json").read_text())
    names = {"precision_floor", "recall_gain", "f1_gain", "single_onset_recall_decline_limit"}
    margins = {"precision_floor": .98, "recall_gain": .05, "f1_gain": .02,
               "single_onset_recall_decline_limit": .03, "full_e2e_f1_at_1_gain": .02}
    if (total["status"] != "NO_GO_DEVELOPMENT" or set(total["gates"]) != names
            or total["comparison_role"] != "development_gate" or total["gate_margins"] != margins
            or total["baseline_trigger_target"] != "recent_onset60" or total["candidate_trigger_target"] != "onset30"
            or all(total["gates"].values()) or total["test_read"]
            or total["gt_denominator"] != 2901
            or tcn_lock["test_arrays_read"] is not False or tcn_lock["test_trigger_labels_built"] is not False
            or total["results"]["candidate"]["completion_sha256"] != sha256_file(refs["tcn_run"] / "completion_manifest.json")
            or tcn.get("model_class") != "WindowCausalTCNTrigger"
            or set(decoder["gates"]) != {"onset_bins_fixed_threshold", "onset_bins_validation_selected"}
            or any(set(gates) != names or all(gates.values()) for gates in decoder["gates"].values())
            or decoder["gate_margins"] != margins or decoder["test_read"] or decoder["gt_denominator"] != 2901
            or decoder["protocol_id"] != tcn["protocol_id"]
            or decoder["model_completion_sha256"] != sha256_file(refs["tcn_run"] / "completion_manifest.json")):
        raise ValueError("TCN/decoder has not completed a valid NO-GO decision")
    candidate_scores = refs["tcn_run"] / "validation_predictions.csv"
    if decoder["source_sha256"].get(str(candidate_scores.resolve())) != sha256_file(candidate_scores):
        raise ValueError("decoder decision belongs to another prediction run")
    files = []
    for directory in refs.values():
        files.extend(directory / name for name in ("completion_manifest.json",))
    files.extend([refs["tcn_total_comparison"] / "comparison.json",
                  refs["tcn_decoder_comparison"] / "decoder_comparison.json", candidate_scores,
                  refs["tcn_run"] / "development_input_lock.json"])
    return {str(path.resolve()): sha256_file(path) for path in files}


def cohort_summary(state, config):
    counts = {}
    for block in state.blocks:
        counts[block.name] = {"windows": len(state.indices(block.name)),
            "normal_windows": len(state.indices(block.name, True)), "complete_gt": len(state.gt_events(block.name))}
    clean = state.clean_memory_ids()
    enough = all(item["normal_windows"] >= config["min_normal_windows"] for item in counts.values())
    return {"blocks": {block.name: {"start_ms": block.start_ms, "end_ms": block.end_ms} for block in state.blocks},
            "counts": counts, "legal_context_events": len(state.legal_events),
            "purged_events": len(state.purged_events), "normal_support_passed": enough,
            "clean_memory_ids": clean, "clean_memory_n": len(clean),
            "clean_memory_support_passed": len(clean) >= config["min_clean_memory_cases"],
            "fit_boundary_prediction_excluded": True}


def loader(state, indices, config, gpu=False, shuffle=False):
    return DataLoader(MetricForecastDataset(state, indices), batch_size=config["training"]["batch_size"],
                      shuffle=shuffle, num_workers=config["training"]["num_workers"], pin_memory=gpu)


def export_scores(model, batches, state, device):
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            history = batch["history"].to(device)
            target = batch["target"].to(device)
            predicted = model(history)
            neural = residual_score(predicted, target).cpu().numpy()
            persistence = residual_score(history[:, -1, :, :45], target).cpu().numpy()
            indices = batch["sample_index"].numpy()
            observed = state.metric[indices + 9, :, 45].mean(axis=1)
            for index, score, reference, fraction in zip(indices, neural, persistence, observed):
                rows.append({"sample_index": int(index),
                    "prediction_available_time": int(state.timestamps[int(index) + 9]) + 30000,
                    "forecast_score": float(score), "persistence_score": float(reference),
                    "filled_observed_fraction": float(fraction)})
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[["forecast_score", "persistence_score", "filled_observed_fraction"]]).all().all():
        raise ValueError("nonfinite forecast output")
    return frame


def real_prediction_gate(model, state, device):
    batch = next(iter(DataLoader(MetricForecastDataset(state, state.indices("event_holdout")[:32]), batch_size=32)))
    history = batch["history"].to(device)
    model.eval()
    with torch.no_grad():
        full = model(history)
        singleton = torch.cat([model(row[None]) for row in history])
        changed = history.clone()
        changed[1:] += 5
        peer_error = float((full[:1] - model(changed)[:1]).abs().max())
        batch_error = float((full - singleton).abs().max())
    result = {"passed": max(batch_error, peer_error) <= 1e-5,
              "batch_singleton_max_error": batch_error, "peer_perturbation_max_error": peer_error, "atol": 1e-5}
    if not result["passed"]:
        raise ValueError("actual-input forecast independence gate failed")
    return result


def screen_results(output, state, calibration, holdout, config, summary):
    thresholds = {name: float(np.quantile(calibration[name + "_score"], config["normal_quantile"], interpolation="linear"))
                  for name in ("forecast", "persistence")}
    write_json(output / "prediction_lock.json", {"execution_commit": current_head(),
               "scope_lock_sha256": sha256_file(output / "scope_lock.json"),
               "checkpoint_sha256": sha256_file(output / "checkpoint/final.pt"),
               "thresholds": thresholds, "normal_quantile": config["normal_quantile"],
               "calibration_scores_sha256": sha256_file(output / "calibration_scores.csv"),
               "holdout_scores_sha256": sha256_file(output / "holdout_scores.csv"), "test_read": False})
    # Holdout labels have already served the preregistered cohort/normal-scope
    # eligibility audit. No metric or score-to-label join precedes this lock.
    gt = state.gt_events("event_holdout")
    gt.to_csv(output / "event_holdout_gt.csv", index=False)
    normal_indices = set(state.indices("event_holdout", True).tolist())
    holdout = holdout.copy()
    holdout["normal_window"] = holdout.sample_index.isin(normal_indices)
    results = {}
    for name, threshold in thresholds.items():
        frame = system_score_frame("event_holdout", holdout.prediction_available_time, holdout[name + "_score"])
        episodes, matching, metrics = evaluate_system_threshold(frame, gt, threshold)
        ledger, invariants = build_failure_ledger(gt, split="event_holdout",
            slot_times_ms=holdout.prediction_available_time.to_numpy(dtype=np.int64),
            slot_scores=holdout[name + "_score"].to_numpy(), threshold=threshold,
            episode_anchors_ms=episodes.t_hat.to_numpy(dtype=np.int64),
            episode_end_times_ms=episodes.episode_end_time.to_numpy(dtype=np.int64), matching=matching,
            origin_ms=state.origin_ms, split_end_ms=state.fit_end_ms, context_events=state.legal_events)
        if not all(invariants[k] for k in ("categories_sum_to_total", "matched_equals_tp",
                                           "unmatched_events_equal_fn", "unmatched_episodes_equal_fp")):
            raise ValueError("Fit holdout denominator/failure ledger does not close")
        episodes.to_csv(output / (name + "_episodes.csv"), index=False)
        matching.to_csv(output / (name + "_matching.csv"), index=False)
        ledger.to_csv(output / (name + "_failure_ledger.csv"), index=False)
        normal = holdout.loc[holdout.normal_window, name + "_score"]
        strata = []
        for lo, hi in ((0., .5), (.5, .9), (.9, 1.000001)):
            subset = holdout.loc[holdout.normal_window & holdout.filled_observed_fraction.ge(lo)
                                 & holdout.filled_observed_fraction.lt(hi), name + "_score"]
            strata.append({"observed_fraction_lower": lo, "upper": hi, "n": len(subset),
                           "positive_fraction": float((subset >= threshold).mean()) if len(subset) else None})
        results[name] = {"threshold": threshold, "metrics": metrics, "failure": invariants,
            "stratified": stratified_summary(ledger, split="event_holdout"),
            "normal_holdout_n": len(normal), "normal_holdout_positive_fraction": float((normal >= threshold).mean()),
            "filled_observation_strata": strata}
    neural, reference = results["forecast"], results["persistence"]
    margins = config["screen_gate"]
    gates = {"precision_floor": neural["metrics"]["event_precision"] >= margins["precision_floor"],
             "recall_floor": neural["metrics"]["event_recall"] >= margins["recall_floor"],
             "f1_floor": neural["metrics"]["event_f1"] >= margins["f1_floor"],
             "f1_gain_vs_persistence": neural["metrics"]["event_f1"] - reference["metrics"]["event_f1"] >= margins["f1_gain_vs_persistence"],
             "normal_holdout_positive_fraction_limit": neural["normal_holdout_positive_fraction"] <= margins["normal_holdout_positive_fraction_limit"]}
    return {"status": "GO_TO_VALIDATION_DEVELOPMENT" if all(gates.values()) else "NO_GO_FIT_SCREEN",
            "test_read": False, "validation_inference_run": False,
            "prior_validation_decisions_used_for_start": True, "full_e2e_run": False,
            "gt_denominator": len(gt), "gates": gates, "gate_margins": margins,
            "results": results, "cohort": summary,
            "limitations": ["shared 70 percent Train preprocessing", "filled-series residual with aggregate masks",
                            "normal denotes no registered event", "single fixed candidate and seed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "train"))
    for name in ("config", "output-dir", "data-root", "artifact-root", "registry"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = json.loads(args.config.read_text())
    validate_config(config)
    base = load_config(ROOT / config["base_config"])
    if sha256_file(args.registry) != base["event_registry"]["sha256"]:
        raise ValueError("registry drift")
    bindings = input_bindings(args, config)
    if args.action == "train":
        bindings.update(validate_start_condition(config))
        frozen_inputs = json.loads((Path(config["references"]["tcn_run"]) / "development_input_lock.json").read_text())["source_sha256"]
        for path in (args.data_root / "train/timestamps.npy", args.data_root / "train/metric.npy",
                     args.artifact_root / "ad_data_manifest.json"):
            name = str(path.resolve())
            if frozen_inputs.get(name) != bindings[name]:
                raise ValueError("forecast input differs from frozen detector: " + name)
    state = NormalForecastState(base, args.data_root, args.registry)
    summary = cohort_summary(state, config)
    if not np.isfinite(state.metric[state.timestamps < state.fit_end_ms]).all():
        raise ValueError("nonfinite Fit Metric array")
    execution_commit = current_head()
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "scope_lock.json", {"execution_commit": execution_commit,
        "protocol_id": config["protocol_id"], "source_sha256": bindings,
        "test_arrays_read": False, "validation_arrays_read": False,
        "registry_interval_columns_scanned_for_routing": True, "test_fault_annotations_parsed": False})
    state.windows.to_csv(args.output_dir / "window_cohort.csv", index=False)
    state.purged_events.to_csv(args.output_dir / "purged_events.csv", index=False)
    for block in state.blocks:
        if args.action == "check" or block.name != "event_holdout":
            state.gt_events(block.name).to_csv(args.output_dir / (block.name + "_gt.csv"), index=False)
    write_json(args.output_dir / "cohort_summary.json", to_builtin(summary))
    if args.action == "train":
        if not summary["normal_support_passed"]:
            raise ValueError("insufficient normal support; run remains incomplete")
        if args.gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_num_threads(args.threads)
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
            handlers=[logging.FileHandler(args.output_dir / "normal_forecast.log"), logging.StreamHandler()])
        device = torch.device("cuda" if args.gpu else "cpu")
        model = NormalMetricForecaster().to(device)
        training = config["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
        batches = loader(state, state.indices("normal_training", True), config, args.gpu, True)
        history = []
        for epoch in range(training["epochs"]):
            model.train()
            total, count = 0., 0
            for batch in batches:
                optimizer.zero_grad()
                prediction = model(batch["history"].to(device))
                loss = F.huber_loss(prediction, batch["target"].to(device), delta=training["huber_delta"])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite normal training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip_norm"])
                optimizer.step()
                size = len(batch["sample_index"])
                total += float(loss.detach()) * size
                count += size
            history.append({"epoch": epoch, "normal_training_huber": total / count})
            logging.info("epoch %d: normal_training_huber=%.6f", epoch, total / count)
        (args.output_dir / "checkpoint").mkdir()
        torch.save(model.state_dict(), args.output_dir / "checkpoint/final.pt")
        write_json(args.output_dir / "training_log.json", history)
        write_json(args.output_dir / "real_prediction_gate.json", real_prediction_gate(model, state, device))
        calibration = export_scores(model, loader(state, state.indices("normal_calibration", True), config, args.gpu), state, device)
        holdout = export_scores(model, loader(state, state.indices("event_holdout"), config, args.gpu), state, device)
        calibration.to_csv(args.output_dir / "calibration_scores.csv", index=False)
        holdout.to_csv(args.output_dir / "holdout_scores.csv", index=False)
        report = screen_results(args.output_dir, state, calibration, holdout, config, summary)
        write_json(args.output_dir / "screen_result.json", to_builtin(report))
    else:
        report = {"status": "PREFLIGHT_PASS" if summary["normal_support_passed"] else "INSUFFICIENT_NORMAL_SUPPORT", "cohort": summary}
    if current_head() != execution_commit or any(sha256_file(Path(path)) != sha for path, sha in bindings.items()):
        raise ValueError("source/input drift; screen remains incomplete")
    write_json(args.output_dir / "completion_manifest.json", {"status": "COMPLETE_DEVELOPMENT_ONLY",
        "execution_commit": execution_commit, "action": args.action, "test_inference_run": False,
        "files": {str(path.relative_to(args.output_dir)): sha256_file(path)
                  for path in args.output_dir.rglob("*") if path.is_file()}})
    print(json.dumps(to_builtin(report)))


if __name__ == "__main__":
    main()
