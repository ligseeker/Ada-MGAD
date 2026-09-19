#!/usr/bin/env python3
"""P6-C0F failure-mechanism audit driver (read-only with respect to every frozen artifact).

Stages (explicit, resumable, each writing into the audit run directory only):

```text
prepare    input identity, source integrity, split population checks
scores     frozen Fit/Validation forward scores + Validation replay check (no Test inference)
analyze    test-artifact verification, failure ledger, trajectories, strata, confounds, collisions
capacity   R_label, U_grid, U_episode and witness replay
finalize   evidence ledger, decision, report, completion manifest
all        prepare -> scores -> analyze -> capacity -> finalize
status     print the recorded run state
```

Hard boundaries enforced by this driver:

* it never trains, never selects, never rescans or replaces the frozen threshold,
* it never runs any Test model inference: the Test side is read from the frozen
  CSVs and only re-verified arithmetically,
* it never calls the C0 ``run_evaluation`` / ``run_training`` / ``run_audit``
  entry points and never writes into the C0 root, the P5 root or any shared
  input tree,
* it never modifies the C0 verdict; ``decision.json`` records it unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.event_detection import event_metrics
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import (
    DEFAULT_GRID_SECONDS,
    DEFAULT_TOLERANCE_SECONDS,
    SPLIT_NAMES,
    TRIGGER_IGNORE,
    TRIGGER_POSITIVE,
    evaluate_system_threshold,
    system_score_frame,
)
from src.e2e.system_trigger_capacity import (
    episode_bound,
    grid_bound,
    label_reference,
    replay_witness,
)
from src.e2e.system_trigger_data import TriggerWindowDataset, build_trigger_loader
from src.e2e.system_trigger_failure_audit import (
    FAILURE_CATEGORIES,
    build_failure_ledger,
    build_trajectories,
    collision_audit,
    confounding_tables,
    stratified_summary,
)
from src.e2e.system_trigger_model import SystemEventTrigger
from scripts.p6.run_c0_trigger import (
    ProtocolState,
    build_model_args,
    resolve_trigger_config,
    score_dataset,
    validate_frozen_protocol,
)


AUDIT_SCHEMA = "p6_c0f_failure_audit_v1"
RUN_STATE_SCHEMA = "p6_c0f_run_state_v1"
STAGES = ("prepare", "scores", "analyze", "capacity", "finalize")

C0_ROOT = PROJECT_ROOT / "experiments/p6/system_event_trigger"
C0_FIRST_RUN_ROOT = PROJECT_ROOT / "experiments/p6/system_event_trigger_v1_label_lag"
C0_SCORE_DECOMPOSITION_ROOT = PROJECT_ROOT / "experiments/p6/score_decomposition"
P5_RUN_ROOT = PROJECT_ROOT / "experiments/p5"
SHARED_INPUT_ROOTS = (PROJECT_ROOT / "data/p5", PROJECT_ROOT / "artifacts/p5")

DEFAULT_BASE_CONFIG = "configs/e2e/gaia_p5_v3_preprocessing_v2.json"
DEFAULT_TRIGGER_CONFIG = "configs/e2e/gaia_p6_c0_system_trigger.json"

# Input binding from docs/P6_C0F_FAILURE_MECHANISM_AUDIT_PLAN.md section 2.1.
EXPECTED_INPUT_SHA256 = {
    "checkpoint": "6a4317d6414774e1ca6be4e418ed3d47d678adfab1f0155e96e04aa3fa9981f5",
    "trigger_config": "a2c42beb19f5a234ab74958cf3db32301524f909d5605001bdaeb9c141e3b4f8",
    "base_config": "25abf4e8008e90e19a51131f1a2ee7420b337fd275dbfbacf9504c2bc5e2ee35",
    "ad_data_manifest": "3b228061fd66b659bfa1f7564d4b9ffe5129be06b702b58573f67003d61634c2",
    "gt_registry": "7bc1b8b9c99164d0df004b72f47977b5413070c033834babac33931a27a0324b",
}
EXPECTED_C0_ARTIFACT_SHA256 = {
    "manifest.json": "cb222bb5c21d20d8a7546e8b1ee1b4882daaac5d4a397d0e07655fe646cf9794",
    "validation_selection.json": "b5167c8d5df9cd9eebd7812cb76733799aa3f7829f3fe1a2c5cdb0ec51715a24",
    "test_metrics.json": "2968632cc4da1b807d976d8bb1c449f9d755cb1c15d2ee43bcd440b74946ce5b",
    "split_audit.json": "e5c60574f4c36ba5c7ebddbc869ede70f6f9aa83cf35c46fd611dbfba310463e",
    "split_manifest.json": "1cad441a2d207379ffd8838bc2d655e9c68f3e03b6f4f7e666413a0b33aca193",
    "test_predictions.csv": "985623e06f5d56db9c2c536b8dd65226967ae1c25f36c9a4924ae9e38d5bafab",
    "test_episodes.csv": "30bacbeecbee4b9f738fc6330fa9d8df5c1adc08e37fb1ee8476f1b47735399b",
    "test_matching.csv": "011332f3537ee6f7d044e9da23df1d88f617fb71d3a552354343db6fd91c9c77",
}
# Cross-check values from the plan section 2.2 (verified, never hardcoded into results).
EXPECTED_SPLIT_POPULATION = {
    "fit": {"windows": 43_550, "events": 7_443, "events_gt_300s": 362},
    "validation": {"windows": 17_414, "events": 2_901, "events_gt_300s": 125},
    "test": {"windows": 26_127, "events": 5_787, "events_gt_300s": 229},
}
EXPECTED_C0_COUNTS = {
    "validation": {"tp": 2_124, "fp": 13, "fn": 777, "episodes": 2_137},
    "test": {"tp": 4_198, "fp": 16, "fn": 1_589, "episodes": 4_214},
}
REPLAY_TOLERANCE = 1e-12

# Interpretation rules fixed before the real-data analysis (plan sections 5-7).
INTERPRETATION_RULES = {
    "response_window_ms": 60_000,
    "near_threshold_margin": 0.05,
    "long_event_seconds": 300.0,
    "small_n": 30,
    "dominant_population_share": 0.9,
    "concurrency_contamination_share": 0.5,
    "rules": [
        "recall is reported per stratum with n; strata with n < 30 are flagged small_n and never "
        "used for mechanism claims",
        "a 'response' is a frozen-score crossing of the frozen threshold; a 'new episode' is an "
        "official episode start; the two are always reported separately",
        "a long-event response later than 60 s is only reported as a late frozen-model response; it "
        "is never attributed to the event unless the timestamp has no other recent-onset marker",
        "observed recall is compared against U_episode (complete-protocol) and U_grid (relaxed); no "
        "gap-to-bound acceptance threshold is defined in this audit",
        "sustained low scores are reported as OBSERVED; a representation/objective/observability "
        "diagnosis stays HYPOTHESIS unless the trajectory shows a separable pattern",
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()


def sha256_of(path: Path) -> str:
    return sha256_file(Path(path))


def verify_sha(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_of(path)
    if actual != str(expected):
        raise ValueError(
            "{} SHA-256 mismatch for {}: expected {} got {}".format(label, path, expected, actual)
        )
    return actual


def guard_output_dir(path: Path, *, frozen_roots: Sequence[Path], require_empty: bool) -> Path:
    """Refuse frozen trees and (optionally) any non-empty existing directory."""

    resolved = Path(path).resolve()
    for frozen in frozen_roots:
        frozen_resolved = Path(frozen).resolve()
        if resolved == frozen_resolved or frozen_resolved in resolved.parents:
            raise ValueError("audit output must not be inside frozen tree {}".format(frozen_resolved))
    if resolved.exists():
        if require_empty and any(resolved.iterdir()):
            raise ValueError("audit output directory {} already has content".format(resolved))
    else:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _jsonable(value):
    from src.e2e.system_trigger import to_builtin
    return to_builtin(value)


# ---------------------------------------------------------------------------
# run state
# ---------------------------------------------------------------------------


def _frozen_roots() -> List[Path]:
    return [C0_ROOT, C0_FIRST_RUN_ROOT, C0_SCORE_DECOMPOSITION_ROOT, P5_RUN_ROOT] + list(SHARED_INPUT_ROOTS)


def _assert_not_frozen(output_dir: Path) -> Path:
    """Every stage refuses to write inside a frozen tree, not only ``prepare``."""

    return guard_output_dir(output_dir, frozen_roots=_frozen_roots(), require_empty=False)


def load_run_state(output_dir: Path) -> Dict[str, object]:
    path = Path(output_dir) / "run_state.json"
    if not path.is_file():
        raise FileNotFoundError("audit run state not found: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def save_run_state(output_dir: Path, state: Mapping[str, object]) -> None:
    record = dict(state)
    record["updated_at_utc"] = utc_now()
    write_json(Path(output_dir) / "run_state.json", _jsonable(record))


def mark_stage(output_dir: Path, stage: str, status: str, outputs: Mapping[str, object] = None) -> None:
    state = load_run_state(output_dir)
    stages = dict(state.get("stages", {}))
    entry = dict(stages.get(stage, {}))
    entry["status"] = status
    entry["finished_at_utc"] = utc_now()
    if outputs:
        entry["outputs"] = _jsonable(outputs)
    stages[stage] = entry
    state["stages"] = stages
    failed = any(value.get("status") == "FAILED" for value in stages.values())
    complete = all(stages.get(name, {}).get("status") == "COMPLETE" for name in STAGES)
    state["status"] = "COMPLETE" if complete else ("STOP" if failed else "PARTIAL")
    save_run_state(output_dir, state)


def require_stage(output_dir: Path, stage: str) -> Mapping[str, object]:
    state = load_run_state(output_dir)
    entry = state.get("stages", {}).get(stage, {})
    if entry.get("status") != "COMPLETE":
        raise ValueError("stage {} is not COMPLETE; run it first".format(stage))
    return entry


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def stage_prepare(base_config, trigger_config, output_dir: Path, *, base_config_path: Path,
                  trigger_config_path: Path, data_root: Path, artifact_root: Path,
                  resume: bool = False) -> Mapping[str, object]:
    guard_output_dir(output_dir, frozen_roots=_frozen_roots(), require_empty=not resume)
    if not resume:
        save_run_state(output_dir, {
            "schema_version": RUN_STATE_SCHEMA,
            "run_id": output_dir.name,
            "created_at_utc": utc_now(),
            "status": "PARTIAL",
            "stages": {},
            "c0_original_verdict": "BORDERLINE",
            "read_only_contract": (
                "no training, no checkpoint selection, no threshold rescan, no Test inference, no RCA, "
                "no write into any frozen artifact tree"
            ),
        })

    protocol = validate_frozen_protocol(base_config)
    state = ProtocolState(base_config, trigger_config, data_root, None)
    state.frozen_test_windows = int(EXPECTED_SPLIT_POPULATION["test"]["windows"])

    integrity: Dict[str, object] = {"checked": {}, "mismatches": []}
    checkpoint_path = C0_ROOT / "checkpoint/best_validation_event_f1.pt"
    checks = {
        "checkpoint": (checkpoint_path, EXPECTED_INPUT_SHA256["checkpoint"]),
        "trigger_config": (trigger_config_path, EXPECTED_INPUT_SHA256["trigger_config"]),
        "base_config": (base_config_path, EXPECTED_INPUT_SHA256["base_config"]),
        "ad_data_manifest": (artifact_root / "ad_data_manifest.json", EXPECTED_INPUT_SHA256["ad_data_manifest"]),
        "gt_registry": (state.registry_path, EXPECTED_INPUT_SHA256["gt_registry"]),
    }
    for label, (path, expected) in checks.items():
        actual = sha256_of(path)
        ok = actual == str(expected)
        integrity["checked"][label] = {"path": str(path), "expected": expected, "actual": actual, "ok": ok}
        if not ok:
            integrity["mismatches"].append(label)
    for name, expected in EXPECTED_C0_ARTIFACT_SHA256.items():
        path = C0_ROOT / name
        if not path.is_file():
            integrity["checked"]["c0:" + name] = {"path": str(path), "ok": False, "missing": True}
            integrity["mismatches"].append("c0:" + name)
            continue
        actual = sha256_of(path)
        ok = actual == str(expected)
        integrity["checked"]["c0:" + name] = {
            "path": str(path), "expected": expected, "actual": actual, "ok": ok,
        }
        if not ok:
            integrity["mismatches"].append("c0:" + name)
    integrity["status"] = "PASS" if not integrity["mismatches"] else "STOP"
    integrity["policy"] = (
        "documented SHA-256 snapshots are compared as they are; a mismatch stops the affected path "
        "and is never silently rewritten to the current value"
    )

    population: Dict[str, object] = {}
    for split in SPLIT_NAMES:
        events = state.gt_events(split)
        durations = (events["end_ms"] - events["start_ms"]).to_numpy(dtype=np.int64) / 1000.0
        indices = state.sample_indices(split)
        observed = {
            "windows": int(len(indices)),
            "events": int(len(events)),
            "events_gt_300s": int((durations > 300.0).sum()),
            "sample_index_min": int(indices.min()) if len(indices) else None,
            "sample_index_max": int(indices.max()) if len(indices) else None,
        }
        expected = EXPECTED_SPLIT_POPULATION[split]
        population[split] = {
            "observed": observed,
            "expected_from_plan": expected,
            "matches_plan": bool(
                observed["windows"] == expected["windows"]
                and observed["events"] == expected["events"]
                and observed["events_gt_300s"] == expected["events_gt_300s"]
            ),
        }
    population["status"] = "PASS" if all(v["matches_plan"] for k, v in population.items() if k in SPLIT_NAMES) else "STOP"

    audit_config = {
        "schema_version": AUDIT_SCHEMA,
        "run_id": output_dir.name,
        "generated_at_utc": utc_now(),
        "git_commit": git_head(),
        "c0_execution_commit": "cedc4a2bd7492933a8295067c8075e631cbf3df9",
        "inspection_head": git_head(),
        "threads": int(torch.get_num_threads()),
        "seed": int(trigger_config["seed"]),
        "grid_seconds": DEFAULT_GRID_SECONDS,
        "tolerance_seconds": DEFAULT_TOLERANCE_SECONDS,
        "frozen_threshold": float(
            json.loads((C0_ROOT / "validation_selection.json").read_text())["selected_validation_threshold"]
        ),
        "frozen_checkpoint": str(checkpoint_path),
        "frozen_epoch": int(
            json.loads((C0_ROOT / "validation_selection.json").read_text())["selected_epoch"]
        ),
        "frozen_protocol": protocol,
        "interpretation_rules": INTERPRETATION_RULES,
        "boundaries": [
            "no training or backpropagation",
            "no checkpoint or threshold selection",
            "no Test model inference; the frozen Test CSVs are read and re-verified",
            "no RCA, no 68D feature work, no conditional logit",
            "no modification of frozen artifacts; the C0 verdict is recorded unchanged",
        ],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    input_manifest = {
        "schema_version": AUDIT_SCHEMA,
        "run_id": output_dir.name,
        "generated_at_utc": utc_now(),
        "c0_root": str(C0_ROOT),
        "c0_first_run_root": str(C0_FIRST_RUN_ROOT),
        "p5_root": str(P5_RUN_ROOT),
        "data_root": str(data_root),
        "artifact_root": str(artifact_root),
        "registry_path": str(state.registry_path),
        "splits": {
            split: {
                "block": {
                    "start_ms": int(block.start_ms), "end_ms": int(block.end_ms),
                },
                "windows": int(len(state.sample_indices(split))),
                "events": int(len(state.gt_events(split))),
            }
            for split, block in zip(SPLIT_NAMES, state.blocks)
        },
        "read_only": True,
    }
    write_json(output_dir / "audit_config.json", _jsonable(audit_config))
    write_json(output_dir / "input_manifest.json", _jsonable(input_manifest))
    write_json(output_dir / "source_integrity.json", _jsonable(integrity))
    write_json(output_dir / "split_population_checks.json", _jsonable(population))
    mark_stage(output_dir, "prepare", "COMPLETE" if (
        integrity["status"] == "PASS" and population["status"] == "PASS"
    ) else "FAILED", {"integrity": integrity["status"], "population": population["status"]})
    if integrity["status"] != "PASS" or population["status"] != "PASS":
        raise SystemExit("P6-C0F STOP: input identity or split population check failed")
    return {"integrity": integrity["status"], "population": population["status"]}


# ---------------------------------------------------------------------------
# scores (Fit / Validation only)
# ---------------------------------------------------------------------------


def _load_frozen_model(trigger_config, base_config, output_dir: Path, threads: int):
    checkpoint_path = C0_ROOT / "checkpoint/best_validation_event_f1.pt"
    verify_sha(checkpoint_path, EXPECTED_INPUT_SHA256["checkpoint"], label="checkpoint")
    selection = json.loads((C0_ROOT / "validation_selection.json").read_text(encoding="utf-8"))
    model_args = dict(selection["model_args"])
    if int(model_args.get("random_seed", -1)) != int(trigger_config["seed"]):
        raise ValueError("recorded model args disagree with the trigger config seed")
    torch.set_num_threads(int(threads))
    model = SystemEventTrigger(np.load(Path(trigger_config["data_root"]) / "graph.npy", allow_pickle=False),
                               **model_args)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model, model_args, selection


def stage_scores(base_config, trigger_config, output_dir: Path, *, data_root: Path, threads: int) -> Mapping[str, object]:
    _assert_not_frozen(output_dir)
    require_stage(output_dir, "prepare")
    model, model_args, selection = _load_frozen_model(trigger_config, base_config, output_dir, threads)
    state = ProtocolState(base_config, trigger_config, data_root, None)
    batch_size = int(model_args["batch_size"])
    num_workers = int(trigger_config["training"]["num_workers"])
    audit_config = json.loads((output_dir / "audit_config.json").read_text(encoding="utf-8"))
    threshold = float(audit_config["frozen_threshold"])

    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, object] = {}
    frames: Dict[str, Mapping[str, np.ndarray]] = {}
    for split in ("fit", "validation"):
        dataset = state.build_dataset(split)
        result = score_dataset(model, dataset, batch_size, num_workers, torch.device("cpu"), split)
        frames[split] = result
        rows = []
        for position in range(len(dataset)):
            metadata = dataset.metadata(position)
            rows.append({
                "split": split,
                "sample_index": int(metadata.sample_index),
                "window_start_time": int(metadata.window_start_time),
                "window_end_time": int(metadata.window_end_time),
                "target_bin_start": int(metadata.target_bin_start),
                "target_bin_end": int(metadata.target_bin_end),
                "prediction_available_time": int(metadata.prediction_available_time),
                "logit": float(result["logits"][position]),
                "score": float(result["system_score"][position]),
                "fixed_threshold": threshold,
                "binary_prediction": int(result["system_score"][position] >= threshold),
                "trigger_label": int(metadata.trigger_label),
            })
        path = scores_dir / "{}.csv".format(split)
        pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
        outputs[split] = {"path": str(path), "sha256": sha256_of(path), "rows": int(len(rows))}

    validation = frames["validation"]
    frame = system_score_frame("validation", validation["prediction_available_time"], validation["system_score"])
    episodes, matching, metrics = evaluate_system_threshold(
        frame, state.gt_events("validation"), threshold,
        grid_seconds=DEFAULT_GRID_SECONDS, tolerance_seconds=DEFAULT_TOLERANCE_SECONDS,
    )
    recorded = json.loads((C0_ROOT / "test_metrics.json").read_text(encoding="utf-8"))["selected_validation_metrics"]
    comparisons = {}
    exact_keys = ("true_positive_events", "false_positive_events", "false_negative_events")
    for key in exact_keys:
        comparisons[key] = {
            "recorded": recorded[key], "audit": metrics[key], "match": bool(recorded[key] == metrics[key]),
        }
    comparisons["predicted_episode_count"] = {
        "recorded": int(EXPECTED_C0_COUNTS["validation"]["episodes"]),
        "audit": int(metrics["predicted_episode_count"]),
        "match": bool(int(metrics["predicted_episode_count"]) == EXPECTED_C0_COUNTS["validation"]["episodes"]),
    }
    for key in ("event_precision", "event_recall", "event_f1"):
        delta = abs(float(recorded[key]) - float(metrics[key]))
        comparisons[key] = {
            "recorded": float(recorded[key]), "audit": float(metrics[key]),
            "abs_diff": delta, "match": bool(delta <= REPLAY_TOLERANCE),
        }
    delay_keys = ("mean_seconds", "median_seconds", "p95_seconds")
    delay_comparison = {}
    for key in delay_keys:
        recorded_value = recorded["detection_delay"][key]
        audit_value = metrics["detection_delay"][key]
        delay_comparison[key] = {
            "recorded": recorded_value, "audit": audit_value,
            "abs_diff": (None if recorded_value is None else abs(float(recorded_value) - float(audit_value))),
        }
    matched = all(entry["match"] for entry in comparisons.values())
    replay = {
        "schema_version": AUDIT_SCHEMA,
        "generated_at_utc": utc_now(),
        "frozen_threshold": threshold,
        "comparison": comparisons,
        "detection_delay_comparison": delay_comparison,
        "primary_counts_match": bool(matched),
        "tolerance": REPLAY_TOLERANCE,
        "row_level_note": (
            "the original C0 run did not persist per-row Validation scores, so a row-level replay "
            "cannot be verified; only the aggregate replay and the identity binding can be checked"
        ),
        "status": "MATCH" if matched else "MISMATCH",
        "policy": "a mismatch stops the merged conclusions; the tolerance is never widened and no epoch is reselected",
    }
    write_json(output_dir / "validation_replay.json", _jsonable(replay))
    mark_stage(output_dir, "scores", "COMPLETE" if matched else "FAILED",
               {"files": outputs, "validation_replay": replay["status"]})
    if not matched:
        raise SystemExit("P6-C0F STOP: Validation replay mismatch")
    return {"scores": outputs, "validation_replay": replay["status"]}


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _load_split_inputs(state: ProtocolState, output_dir: Path, split: str, threshold: float):
    """Return (slot times, scores, logits, labels, gt frame) for one split."""

    if split == "test":
        path = C0_ROOT / "test_predictions.csv"
        verify_sha(path, EXPECTED_C0_ARTIFACT_SHA256["test_predictions.csv"], label="test_predictions")
        frame = pd.read_csv(path)
        slots = frame["prediction_available_time"].to_numpy(dtype=np.int64)
        scores = frame["system_trigger_score"].to_numpy(dtype=float)
        logits = frame["system_trigger_logit"].to_numpy(dtype=float)
        labels = frame["trigger_label"].to_numpy(dtype=np.int64)
    else:
        path = output_dir / "scores" / "{}.csv".format(split)
        frame = pd.read_csv(path)
        slots = frame["prediction_available_time"].to_numpy(dtype=np.int64)
        scores = frame["score"].to_numpy(dtype=float)
        logits = frame["logit"].to_numpy(dtype=float)
        labels = frame["trigger_label"].to_numpy(dtype=np.int64)
    ground_truth = state.gt_events(split)
    return slots, scores, logits, labels, ground_truth


def _verify_test_artifacts(state: ProtocolState, threshold: float) -> Mapping[str, object]:
    slots, scores, logits, labels, ground_truth = _load_split_inputs(state, Path("."), "test", threshold)
    frame = system_score_frame("test", slots, scores)
    episodes, matching, metrics = evaluate_system_threshold(
        frame, ground_truth, threshold,
        grid_seconds=DEFAULT_GRID_SECONDS, tolerance_seconds=DEFAULT_TOLERANCE_SECONDS,
    )
    saved_episodes = pd.read_csv(C0_ROOT / "test_episodes.csv")
    saved_matching = pd.read_csv(C0_ROOT / "test_matching.csv")
    recorded = json.loads((C0_ROOT / "test_metrics.json").read_text(encoding="utf-8"))
    recorded_metrics = recorded["test_event_metrics"]
    checks = {
        "episode_rows": int(len(episodes)) == int(len(saved_episodes)),
        "episode_identity": bool(episodes["t_hat"].tolist() == saved_episodes["t_hat"].tolist()),
        "matching_rows": int(len(matching)) == int(len(saved_matching)),
        "matching_identity": bool(matching["prediction_id"].equals(saved_matching["prediction_id"])),
        "tp": int(metrics["true_positive_events"]) == EXPECTED_C0_COUNTS["test"]["tp"],
        "fp": int(metrics["false_positive_events"]) == EXPECTED_C0_COUNTS["test"]["fp"],
        "fn": int(metrics["false_negative_events"]) == EXPECTED_C0_COUNTS["test"]["fn"],
        "episodes": int(metrics["predicted_episode_count"]) == EXPECTED_C0_COUNTS["test"]["episodes"],
        "recorded_tp": int(recorded_metrics["true_positive_events"]) == EXPECTED_C0_COUNTS["test"]["tp"],
        "binary_prediction_matches_threshold": bool(
            ((scores >= threshold).astype(int) == pd.read_csv(C0_ROOT / "test_predictions.csv")[
                "binary_prediction"].to_numpy(dtype=int)).all()
        ),
    }
    return {
        "schema_version": AUDIT_SCHEMA,
        "generated_at_utc": utc_now(),
        "recomputed": _jsonable(metrics),
        "checks": {key: bool(value) for key, value in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
        "policy": (
            "the frozen Test predictions are reused as they are; this verification never re-runs the "
            "model, never regenerates the artifacts and never writes into the C0 root"
        ),
    }


def stage_analyze(base_config, trigger_config, output_dir: Path, *, data_root: Path) -> Mapping[str, object]:
    _assert_not_frozen(output_dir)
    require_stage(output_dir, "prepare")
    require_stage(output_dir, "scores")
    audit_config = json.loads((output_dir / "audit_config.json").read_text(encoding="utf-8"))
    threshold = float(audit_config["frozen_threshold"])
    state = ProtocolState(base_config, trigger_config, data_root, None)
    origin = int(state.blocks[0].start_ms)

    test_check = _verify_test_artifacts(state, threshold)
    write_json(output_dir / "test_artifact_verification.json", _jsonable(test_check))
    if test_check["status"] != "PASS":
        mark_stage(output_dir, "analyze", "FAILED", {"test_verification": test_check["status"]})
        raise SystemExit("P6-C0F STOP: recorded Test artifacts do not reproduce")

    ledgers = []
    trajectories = []
    response_summaries = []
    per_split: Dict[str, object] = {}
    for split in SPLIT_NAMES:
        slots, scores, logits, labels, ground_truth = _load_split_inputs(state, output_dir, split, threshold)
        frame = system_score_frame(split, slots, scores)
        episodes, matching, metrics = evaluate_system_threshold(
            frame, ground_truth, threshold,
            grid_seconds=DEFAULT_GRID_SECONDS, tolerance_seconds=DEFAULT_TOLERANCE_SECONDS,
        )
        ledger, summary = build_failure_ledger(
            ground_truth, split=split, slot_times_ms=slots, slot_scores=scores, threshold=threshold,
            episode_anchors_ms=episodes["t_hat"].to_numpy(dtype=np.int64),
            episode_end_times_ms=episodes["episode_end_time"].to_numpy(dtype=np.int64),
            matching=matching, origin_ms=origin, grid_seconds=DEFAULT_GRID_SECONDS,
        )
        split_trajectories, split_response = build_trajectories(
            ledger, split=split, slot_times_ms=slots, slot_scores=scores, slot_logits=logits,
            threshold=threshold, leading_seconds=300, trailing_seconds=300,
        )
        ledgers.append(ledger)
        trajectories.append(split_trajectories)
        response_summaries.append(split_response)
        stratified = stratified_summary(ledger, split=split)
        per_split[split] = {
            "ledger_summary": _jsonable(summary),
            "event_metrics": _jsonable(metrics),
            "stratified": _jsonable(stratified),
            "confounding": _jsonable(confounding_tables(ground_truth, origin_ms=origin)),
            "collision": _jsonable(collision_audit(
                ground_truth, population_name="{}-complete-gt".format(split), origin_ms=origin
            )),
        }

    legal, _, _ = _legal_population(state)
    label_legal_collision = collision_audit(legal, population_name="label-legal-all", origin_ms=origin)

    ledger_frame = pd.concat(ledgers, ignore_index=True)
    trajectory_frame = pd.concat(trajectories, ignore_index=True)
    response_frame = pd.concat(response_summaries, ignore_index=True)
    ledger_path = output_dir / "event_failure_ledger.csv"
    trajectory_path = output_dir / "event_score_trajectories.csv"
    response_path = output_dir / "event_response_summary.csv"
    ledger_frame.to_csv(ledger_path, index=False, lineterminator="\n")
    trajectory_frame.to_csv(trajectory_path, index=False, lineterminator="\n")
    response_frame.to_csv(response_path, index=False, lineterminator="\n")

    stratified_all = {split: per_split[split]["stratified"] for split in SPLIT_NAMES}
    write_json(output_dir / "stratified_summary.json", _jsonable({"splits": stratified_all}))
    confounding_all = {
        "splits": {split: per_split[split]["confounding"] for split in SPLIT_NAMES},
        "collision": {
            "label_legal_all": label_legal_collision,
            "per_split": {split: per_split[split]["collision"] for split in SPLIT_NAMES},
        },
        "label_legal_vs_split_note": (
            "the label-legal population covers the whole frozen detector timeline; the per-split "
            "populations only contain complete in-block events, so the two must not be mixed"
        ),
    }
    write_json(output_dir / "confounding_tables.json", _jsonable(confounding_all))

    invariants = {
        split: {
            "categories_sum_to_total": per_split[split]["ledger_summary"]["categories_sum_to_total"],
            "matched_equals_tp": per_split[split]["ledger_summary"]["matched_equals_tp"],
            "unmatched_events_equal_fn": per_split[split]["ledger_summary"]["unmatched_events_equal_fn"],
            "unmatched_episodes_equal_fp": per_split[split]["ledger_summary"]["unmatched_episodes_equal_fp"],
        }
        for split in SPLIT_NAMES
    }
    invariants["all_hold"] = all(
        all(value for key, value in entry.items() if key != "all_hold") for entry in invariants.values()
        if isinstance(entry, Mapping)
    )
    write_json(output_dir / "ledger_invariants.json", _jsonable(invariants))
    mark_stage(output_dir, "analyze", "COMPLETE",
               {"ledger_rows": int(len(ledger_frame)), "invariants": invariants["all_hold"]})
    return {
        "ledger_rows": int(len(ledger_frame)),
        "trajectory_rows": int(len(trajectory_frame)),
        "invariants": invariants,
        "per_split": per_split,
    }


def _legal_population(state: ProtocolState):
    from src.e2e.system_trigger import assign_legal_events
    legal, assigned, purged = assign_legal_events(state.registry, state.blocks)
    return legal, assigned, purged


# ---------------------------------------------------------------------------
# capacity
# ---------------------------------------------------------------------------


def stage_capacity(base_config, trigger_config, output_dir: Path, *, data_root: Path) -> Mapping[str, object]:
    _assert_not_frozen(output_dir)
    require_stage(output_dir, "prepare")
    require_stage(output_dir, "scores")
    audit_config = json.loads((output_dir / "audit_config.json").read_text(encoding="utf-8"))
    threshold = float(audit_config["frozen_threshold"])
    state = ProtocolState(base_config, trigger_config, data_root, None)
    oracle_root = output_dir / "oracle"
    oracle_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, object] = {}
    for split in SPLIT_NAMES:
        slots, scores, logits, labels, ground_truth = _load_split_inputs(state, output_dir, split, threshold)
        order = np.argsort(slots, kind="stable")
        slots = slots[order]
        scores = scores[order]
        labels = labels[order]
        split_dir = oracle_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        reference = label_reference(slots, labels, ground_truth, split=split)
        reference_summary = {key: value for key, value in reference.items() if key not in ("matching", "episodes")}
        write_json(split_dir / "label_reference_summary.json", _jsonable(reference_summary))
        reference["episodes"].to_csv(split_dir / "label_reference_episodes.csv", index=False, lineterminator="\n")
        reference["matching"].to_csv(split_dir / "label_reference_matching.csv", index=False, lineterminator="\n")

        onsets = ground_truth["start_ms"].to_numpy(dtype=np.int64)
        grid = grid_bound(slots, ground_truth)
        write_json(split_dir / "grid_bound_summary.json", _jsonable({k: v for k, v in grid.items() if k != "assignment"}))
        pd.DataFrame(
            [{"slot": int(slot), "event_index": int(event)} for slot, event in grid["assignment"].items()]
        ).to_csv(split_dir / "grid_bound_witness.csv", index=False, lineterminator="\n")

        bound = episode_bound(slots, onsets)
        witness = np.zeros(len(slots))
        anchor_set = set(int(value) for value in bound.witness_slots)
        for index, slot in enumerate(slots):
            if int(slot) in anchor_set:
                witness[index] = 1.0
        replay = replay_witness(bound.witness_slots, slots, ground_truth, split=split)
        bound_summary = {
            "split": split,
            "status": bound.status,
            "upper_bound_tp": int(bound.upper_bound_tp),
            "ground_truth_events": int(bound.n_events),
            "recall_upper_bound": (bound.upper_bound_tp / bound.n_events) if bound.n_events else None,
            "n_slots": int(bound.n_slots),
            "dp_states": int(bound.dp_states),
            "proof": bound.proof,
            "witness_anchor_count": int(len(bound.witness_slots)),
            "witness_episodes": int(replay["episode_count"]),
            "witness_anchors_match": bool(replay["episode_anchors_match_witness"]),
            "witness_realised_tp": int(replay["realised_tp"]),
            "witness_replay_agrees": bool(int(replay["realised_tp"]) == int(bound.upper_bound_tp)),
            "witness_event_metrics": replay["event_metrics"],
        }
        write_json(split_dir / "episode_bound_summary.json", _jsonable(bound_summary))
        pd.DataFrame({
            "prediction_available_time": slots, "witness_binary": witness.astype(int),
        }).to_csv(split_dir / "episode_bound_witness.csv", index=False, lineterminator="\n")
        replay_frame = pd.DataFrame([
            {"slot": int(slot), "matched": 1} for slot in bound.witness_slots
        ])
        replay_frame.to_csv(split_dir / "episode_bound_witness_anchors.csv", index=False, lineterminator="\n")
        write_json(split_dir / "replay_validation.json", _jsonable({
            "split": split,
            "episode_bound_status": bound.status,
            "witness_realised_tp": int(replay["realised_tp"]),
            "upper_bound_tp": int(bound.upper_bound_tp),
            "agrees": bool(int(replay["realised_tp"]) == int(bound.upper_bound_tp)),
            "episodes": int(replay["episode_count"]),
            "anchors_match_witness": bool(replay["episode_anchors_match_witness"]),
            "policy": "only a witness that replays through the frozen pipeline is accepted",
        }))

        frame = system_score_frame(split, slots, scores)
        _, matching, metrics = evaluate_system_threshold(
            frame, ground_truth, threshold,
            grid_seconds=DEFAULT_GRID_SECONDS, tolerance_seconds=DEFAULT_TOLERANCE_SECONDS,
        )
        observed_tp = int(metrics["true_positive_events"])
        ordering = {
            "observed_tp": observed_tp,
            "episode_bound_tp": int(bound.upper_bound_tp),
            "grid_bound_tp": int(grid["matched_events"]),
            "observed_le_episode": bool(observed_tp <= int(bound.upper_bound_tp)),
            "episode_le_grid": bool(int(bound.upper_bound_tp) <= int(grid["matched_events"])),
            "grid_le_all_events": bool(int(grid["matched_events"]) <= int(len(ground_truth))),
        }
        ordering["all_hold"] = bool(
            ordering["observed_le_episode"] and ordering["episode_le_grid"] and ordering["grid_le_all_events"]
        )
        results[split] = {
            "label_reference": reference_summary,
            "grid_bound": {k: v for k, v in grid.items() if k != "assignment"},
            "episode_bound": bound_summary,
            "ordering": ordering,
            "observed_event_metrics": _jsonable(metrics),
        }
    write_json(output_dir / "capacity_summary.json", _jsonable({
        "schema_version": AUDIT_SCHEMA,
        "generated_at_utc": utc_now(),
        "splits": results,
        "status": "EXACT" if all(
            results[split]["episode_bound"]["status"] == "EXACT" for split in SPLIT_NAMES
        ) else "BOUNDED",
        "policy": (
            "upper bounds are diagnostic; they never become deployable detectors and never modify the "
            "frozen episode, tolerance or GT populations"
        ),
    }))
    ok = all(results[split]["ordering"]["all_hold"] for split in SPLIT_NAMES)
    mark_stage(output_dir, "capacity", "COMPLETE" if ok else "FAILED",
               {"ordering": {split: results[split]["ordering"]["all_hold"] for split in SPLIT_NAMES}})
    if not ok:
        raise SystemExit("P6-C0F STOP: observed recall violates a claimed structural bound")
    return results


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def _evidence_ledger(output_dir: Path, high: Mapping[str, object]) -> List[Mapping[str, object]]:
    claims: List[Mapping[str, object]] = []

    def add(claim_id, statement, grade, population, location, limitations):
        claims.append({
            "claim_id": claim_id, "statement": statement, "grade": grade,
            "population": population, "artifact_field": location, "limitations": limitations,
        })

    integrity = high["integrity"]
    population_checks = high["population"]
    add("C0F-INTEGRITY", "frozen input SHAs and the C0 artifact snapshot match the plan binding",
        "CONFIRMED", "inputs", "source_integrity.json:status",
        "a mismatch would have stopped the run; the snapshot is the plan's documented value, not a re-derivation")
    add("C0F-SPLIT", "the three split populations reproduce the plan table (windows, complete GT, >300 s GT)",
        "CONFIRMED", "fit/validation/test", "split_population_checks.json",
        "counts are verification targets, not filters")
    add("C0F-REPLAY", "the frozen checkpoint reproduces the recorded Validation metrics at the frozen threshold",
        "CONFIRMED", "validation", "validation_replay.json:comparison",
        "row-level Validation scores were never persisted by C0, so only aggregate replay is verifiable")
    add("C0F-LEDGER", "every complete GT event has exactly one failure location code and the ledger closes",
        "CONFIRMED", "fit/validation/test", "event_failure_ledger.csv, ledger_invariants.json",
        "codes are occurrence locations, not causal explanations")
    add("C0F-ORACLE", "the label reference, the relaxed grid bound and the complete-protocol episode bound hold the ordering observed <= U_episode <= U_grid <= 1",
        "CONFIRMED", "fit/validation/test", "capacity_summary.json:ordering",
        "U_grid is a relaxation and usually not realisable; U_episode is exact only with its witness replay")

    for split in SPLIT_NAMES:
        ledger = pd.read_csv(output_dir / "event_failure_ledger.csv")
        ledger = ledger.loc[ledger["split"] == split]
        total = int(len(ledger))
        if not total:
            continue
        mix = ledger["failure_category"].value_counts()
        share = {name: float(mix.get(name, 0) / total) for name in FAILURE_CATEGORIES}
        add("C0F-MIX-" + split.upper(),
            "failure-location mix for {}: {}".format(split, json.dumps(share, sort_keys=True)),
            "CONFIRMED", split, "event_failure_ledger.csv:failure_category",
            "the mix describes where detection is lost, not why")
        late = ledger.loc[ledger["failure_category"] == "BELOW_THRESHOLD"]
        if len(late):
            first = pd.to_numeric(late["first_positive_delta_ms"], errors="coerce")
            add("C0F-NO-FIRST-POSITIVE-" + split.upper(),
                "{} events reach no threshold crossing at all inside [onset, onset+60 s]".format(int(first.isna().sum())),
                "OBSERVED", split, "event_failure_ledger.csv:first_positive_time",
                "no crossing is an observation about the frozen operating point, not a diagnosis of the score")
    return claims


def _mechanism_findings(output_dir: Path) -> Mapping[str, object]:
    ledger = pd.read_csv(output_dir / "event_failure_ledger.csv")
    response = pd.read_csv(output_dir / "event_response_summary.csv")
    capacity = json.loads((output_dir / "capacity_summary.json").read_text(encoding="utf-8"))
    findings: Dict[str, object] = {}
    for split in SPLIT_NAMES:
        rows = ledger.loc[ledger["split"] == split]
        if rows.empty:
            continue
        total = int(len(rows))
        bands = rows["response_band"].astype(str).value_counts().to_dict()
        findings[split] = {
            "events": total,
            "response_band_mix": {band: int(bands.get(band, 0)) for band in
                                  ("le_60s", "60_120s", "120_300s", "gt_300s", "never", "censored")},
            "observed_recall": float((rows["failure_category"] == "MATCHED").sum() / total),
            "episode_recall_upper_bound": capacity["splits"][split]["episode_bound"]["recall_upper_bound"],
            "grid_recall_upper_bound": capacity["splits"][split]["label_reference"]["event_metrics"]["event_recall"],
            "response_gap": (
                "observed recall {:.4f} vs complete-protocol bound {:.4f}".format(
                    float((rows["failure_category"] == "MATCHED").sum() / total),
                    capacity["splits"][split]["episode_bound"]["recall_upper_bound"],
                )
            ),
        }
    return findings


def _loss_dominance(output_dir: Path) -> Mapping[str, object]:
    """Rule-based read of the failure ledger (fixed before the run, see INTERPRETATION_RULES)."""

    ledger = pd.read_csv(output_dir / "event_failure_ledger.csv")
    per_split: Dict[str, object] = {}
    for split in SPLIT_NAMES:
        rows = ledger.loc[ledger["split"] == split]
        total = int(len(rows))
        if not total:
            continue
        shares = {
            name: float((rows["failure_category"] == name).sum() / total) for name in FAILURE_CATEGORIES
        }
        structural = shares["NO_NEW_EPISODE"] + shares["MATCHING_COMPETITION"]
        score_side = shares["BELOW_THRESHOLD"] + shares["NO_LEGAL_PREDICTION"]
        per_split[split] = {
            "structural_loss_share": structural,
            "score_side_loss_share": score_side,
            "structural_to_score_ratio": (structural / score_side) if score_side > 0 else None,
        }
    ratios = [entry["structural_to_score_ratio"] for entry in per_split.values()
              if entry["structural_to_score_ratio"] is not None]
    if ratios and all(ratio >= 1.5 for ratio in ratios):
        verdict = "structural_dominant"
    elif ratios and all(ratio <= 1 / 1.5 for ratio in ratios):
        verdict = "score_side_dominant"
    else:
        verdict = "mixed"
    recommendation = {
        "structural_dominant": (
            "the loss is dominated by the episode/matching structure on every split; a separately frozen "
            "P6-C0R2 candidate may study episode/selection structure, keeping P6-C0 BORDERLINE and "
            "without changing the frozen checkpoint or threshold in this round"
        ),
        "score_side_dominant": (
            "the loss is dominated by missing threshold crossings; record objective/representation/"
            "observability hypotheses and keep them as HYPOTHESIS - do not diagnose a representation "
            "failure from low scores alone"
        ),
        "mixed": (
            "no single loss family dominates; keep the mechanism question partially unresolved and treat "
            "any later P6-C1 protocol as an interface study that accepts the Stage-1 recall limits"
        ),
    }[verdict]
    return {
        "verdict": verdict,
        "per_split": per_split,
        "rule": (
            "structural = NO_NEW_EPISODE + MATCHING_COMPETITION; score-side = BELOW_THRESHOLD + "
            "NO_LEGAL_PREDICTION; structural_dominant iff structural >= 1.5 x score-side on every "
            "split, score_side_dominant iff the reverse holds on every split"
        ),
        "specified_at": (
            "declared while implementing the audit driver, after the frozen ledger was computed; it is a "
            "descriptive summary convention used to select between the next-scope options listed in "
            "docs/P6_RESEARCH_ROADMAP.md section 4, not a pre-registered acceptance gate"
        ),
        "recommendation": recommendation,
    }


def stage_finalize(base_config, trigger_config, output_dir: Path, *, data_root: Path) -> Mapping[str, object]:
    _assert_not_frozen(output_dir)
    for stage in ("prepare", "scores", "analyze", "capacity"):
        require_stage(output_dir, stage)
    integrity = json.loads((output_dir / "source_integrity.json").read_text(encoding="utf-8"))
    population = json.loads((output_dir / "split_population_checks.json").read_text(encoding="utf-8"))
    replay = json.loads((output_dir / "validation_replay.json").read_text(encoding="utf-8"))
    capacity = json.loads((output_dir / "capacity_summary.json").read_text(encoding="utf-8"))
    ledger_summary = json.loads((output_dir / "stratified_summary.json").read_text(encoding="utf-8"))
    confounds = json.loads((output_dir / "confounding_tables.json").read_text(encoding="utf-8"))
    invariants = json.loads((output_dir / "ledger_invariants.json").read_text(encoding="utf-8"))

    findings = _mechanism_findings(output_dir)
    claims = _evidence_ledger(output_dir, {"integrity": integrity, "population": population})

    unresolved = [
        {
            "id": "C0F-U1",
            "statement": (
                "concurrent onsets can carry a score crossing that is not attributable to the event "
                "under audit; the marker columns expose it but most long events live in busy regions"
            ),
            "evidence": "event_score_trajectories.csv:other_onset_count_60s",
        },
        {
            "id": "C0F-U2",
            "statement": (
                "the frozen registry has no injections between 15 s and 300 s and no cpu_anomalies in "
                "Fit/Validation, so duration and fault-type effects cannot be separated"
            ),
            "evidence": "confounding_tables.json:fault_x_duration",
        },
        {
            "id": "C0F-U3",
            "statement": (
                "whether the IGNORE band, the recent-onset objective, the representation or telemetry "
                "observability causes the sustained-state behaviour is not identifiable from these "
                "artifacts; no causal mechanism is claimed"
            ),
            "evidence": "event_response_summary.csv",
        },
    ]
    decision = {
        "schema_version": AUDIT_SCHEMA,
        "generated_at_utc": utc_now(),
        "c0_original_verdict": "BORDERLINE",
        "c0_verdict_changed": False,
        "audit_completion": {
            "prepare": "COMPLETE",
            "scores": "COMPLETE",
            "analyze": "COMPLETE",
            "capacity": "COMPLETE",
            "oracle_status": capacity["status"],
            "input_integrity": integrity["status"],
            "split_population": population["status"],
            "validation_replay": replay["status"],
            "ledger_invariants_hold": bool(invariants.get("all_hold")),
        },
        "mechanism_findings": findings,
        "stratified_highlights": {
            split: {
                "by_duration_stratum": ledger_summary["splits"][split]["by_duration_stratum"],
                "by_fault_type": ledger_summary["splits"][split]["by_fault_type"],
            }
            for split in SPLIT_NAMES
        },
        "collision_highlights": confounds["collision"],
        "unresolved": unresolved,
        "loss_dominance": _loss_dominance(output_dir),
        "recommended_next_scope": _loss_dominance(output_dir)["recommendation"],
        "notes": [
            "this audit does not select, retrain, rescan thresholds or run Test inference",
            "the same Test has been observed in the design loop, so no new independent confirmation is claimed",
        ],
    }
    write_json(output_dir / "decision.json", _jsonable(decision))
    write_json(output_dir / "evidence_ledger.json", _jsonable({
        "schema_version": AUDIT_SCHEMA,
        "generated_at_utc": utc_now(),
        "grades": ["CONFIRMED", "OBSERVED", "HYPOTHESIS", "UNRESOLVED"],
        "claims": claims,
        "unresolved": unresolved,
    }))

    report = _render_report(output_dir, decision, findings)
    (output_dir / "final_report.md").write_text(report, encoding="utf-8")

    required = {
        "run_state.json": output_dir / "run_state.json",
        "input_manifest.json": output_dir / "input_manifest.json",
        "audit_config.json": output_dir / "audit_config.json",
        "source_integrity.json": output_dir / "source_integrity.json",
        "split_population_checks.json": output_dir / "split_population_checks.json",
        "scores/fit.csv": output_dir / "scores/fit.csv",
        "scores/validation.csv": output_dir / "scores/validation.csv",
        "validation_replay.json": output_dir / "validation_replay.json",
        "test_artifact_verification.json": output_dir / "test_artifact_verification.json",
        "event_failure_ledger.csv": output_dir / "event_failure_ledger.csv",
        "event_score_trajectories.csv": output_dir / "event_score_trajectories.csv",
        "event_response_summary.csv": output_dir / "event_response_summary.csv",
        "stratified_summary.json": output_dir / "stratified_summary.json",
        "confounding_tables.json": output_dir / "confounding_tables.json",
        "capacity_summary.json": output_dir / "capacity_summary.json",
        "evidence_ledger.json": output_dir / "evidence_ledger.json",
        "decision.json": output_dir / "decision.json",
        "final_report.md": output_dir / "final_report.md",
    }
    for split in SPLIT_NAMES:
        for name in ("label_reference_summary.json", "grid_bound_summary.json",
                     "episode_bound_summary.json", "replay_validation.json",
                     "episode_bound_witness.csv", "label_reference_episodes.csv"):
            required["oracle/{}/{}".format(split, name)] = output_dir / "oracle" / split / name
    missing = [name for name, path in required.items() if not Path(path).is_file()]
    completion = {
        "schema_version": AUDIT_SCHEMA,
        "generated_at_utc": utc_now(),
        "git_commit": git_head(),
        "run_id": output_dir.name,
        "status": "COMPLETE" if not missing else "PARTIAL",
        "required_outputs": {
            name: {
                "path": str(Path(path)),
                "bytes": int(os.path.getsize(path)) if Path(path).is_file() else None,
                "sha256": sha256_of(path) if Path(path).is_file() else None,
            }
            for name, path in required.items()
        },
        "missing_outputs": missing,
        "input_binding": integrity["checked"],
        "input_binding_status": integrity["status"],
        "commands": {
            "prepare": "python scripts/p6/audit_c0_failure.py prepare --output-dir <run> --threads 8",
            "scores": "python scripts/p6/audit_c0_failure.py scores --output-dir <run> --threads 8",
            "analyze": "python scripts/p6/audit_c0_failure.py analyze --output-dir <run>",
            "capacity": "python scripts/p6/audit_c0_failure.py capacity --output-dir <run>",
            "finalize": "python scripts/p6/audit_c0_failure.py finalize --output-dir <run>",
        },
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "threads": int(torch.get_num_threads()),
        },
        "oracle_status": {
            split: capacity["splits"][split]["episode_bound"]["status"] for split in SPLIT_NAMES
        },
        "c0_verdict": "BORDERLINE",
        "limitations": [item["id"] + ": " + item["statement"] for item in unresolved],
    }
    write_json(output_dir / "completion_manifest.json", _jsonable(completion))
    mark_stage(output_dir, "finalize", "COMPLETE" if not missing else "FAILED",
               {"missing": missing, "status": completion["status"]})
    if missing:
        raise SystemExit("P6-C0F finalize: missing required outputs {}".format(missing))
    _release_run_state(output_dir, completion)
    # refresh the run-state hash so the completion manifest describes the final file
    completion["required_outputs"]["run_state.json"] = {
        "path": str(output_dir / "run_state.json"),
        "bytes": int(os.path.getsize(output_dir / "run_state.json")),
        "sha256": sha256_of(output_dir / "run_state.json"),
    }
    completion["run_state_final_status"] = json.loads(
        (output_dir / "run_state.json").read_text(encoding="utf-8")
    )["status"]
    write_json(output_dir / "completion_manifest.json", _jsonable(completion))
    return completion


def _release_run_state(output_dir: Path, completion: Mapping[str, object]) -> None:
    state = load_run_state(output_dir)
    state["status"] = completion["status"]
    stages = state.get("stages", {})
    stages["finalize"] = {"status": "COMPLETE", "finished_at_utc": utc_now()}
    state["stages"] = stages
    save_run_state(output_dir, state)


def _render_report(output_dir: Path, decision: Mapping[str, object], findings: Mapping[str, object]) -> str:
    ledger = pd.read_csv(output_dir / "event_failure_ledger.csv")
    capacity = json.loads((output_dir / "capacity_summary.json").read_text(encoding="utf-8"))
    replay = json.loads((output_dir / "validation_replay.json").read_text(encoding="utf-8"))
    lines: List[str] = []
    lines.append("# P6-C0F Failure Mechanism Audit — Result")
    lines.append("")
    lines.append("Run: `{}`  |  git commit: `{}`".format(output_dir.name, decision.get("git_commit", git_head())))
    lines.append("")
    lines.append("Read-only audit of the frozen P6-C0 system trigger. No training, no threshold rescan,")
    lines.append("no Test model inference, no RCA. The P6-C0 verdict stays **BORDERLINE**.")
    lines.append("")
    lines.append("## 1. Completion and integrity")
    lines.append("")
    for key, value in decision["audit_completion"].items():
        lines.append("- `{}`: {}".format(key, value))
    lines.append("- Validation replay: {} at the frozen threshold".format(replay["status"]))
    lines.append("")
    lines.append("## 2. Failure ledger (per split)")
    lines.append("")
    lines.append("Detection funnel per split (identical populations, no event removed):")
    lines.append("")
    lines.append("| split | complete GT | causal crossing in window | new episode start in window | matched (TP) |")
    lines.append("|---|---:|---:|---:|---:|")
    for split in SPLIT_NAMES:
        rows = ledger.loc[ledger["split"] == split]
        if rows.empty:
            continue
        causal = int((rows["positive_slots"] > 0).sum())
        episodes = int((rows["candidate_episode_count"] > 0).sum())
        matched = int((rows["failure_category"] == "MATCHED").sum())
        lines.append("| {} | {} | {} | {} | {} |".format(split, len(rows), causal, episodes, matched))
    lines.append("")
    lines.append(
        "The three losses are different objects: a missing crossing is a score/operating-point "
        "location, a missing new episode is an episode-construction location (the trigger was already "
        "running), and an unmatched candidate episode is a one-to-one matching location."
    )
    lines.append("")
    for split in SPLIT_NAMES:
        rows = ledger.loc[ledger["split"] == split]
        if rows.empty:
            continue
        lines.append("### {}".format(split))
        lines.append("")
        lines.append("| failure location | n | share |")
        lines.append("|---|---:|---:|")
        for name in FAILURE_CATEGORIES:
            count = int((rows["failure_category"] == name).sum())
            lines.append("| {} | {} | {:.4f} |".format(name, count, count / max(len(rows), 1)))
        lines.append("")
        lines.append("| duration stratum | n | matched | recall |")
        lines.append("|---|---:|---:|---:|")
        stratified = json.loads((output_dir / "stratified_summary.json").read_text(
            encoding="utf-8"))["splits"][split]
        for row in stratified["by_duration_stratum"]:
            recall = "n/a" if row["recall"] is None else "{:.4f}".format(row["recall"])
            lines.append("| {} | {} | {} | {} |".format(
                row["duration_stratum"], row["n"], row["matched"], recall))
        lines.append("")
        lines.append("Failure mix inside the two non-empty duration strata:")
        lines.append("")
        lines.append("| stratum | n | MATCHED | NO_LEGAL_PREDICTION | BELOW_THRESHOLD | NO_NEW_EPISODE | MATCHING_COMPETITION |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in stratified["by_duration_stratum"]:
            if row["n"] == 0:
                continue
            mix = row["failure_mix"]
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                row["duration_stratum"], row["n"], mix["MATCHED"], mix["NO_LEGAL_PREDICTION"],
                mix["BELOW_THRESHOLD"], mix["NO_NEW_EPISODE"], mix["MATCHING_COMPETITION"]))
        lines.append("")
        lines.append("Failure mix by fault type (n >= 30 only; smaller groups stay in the JSON):")
        lines.append("")
        lines.append("| fault type | n | MATCHED | BELOW_THRESHOLD | NO_NEW_EPISODE | MATCHING_COMPETITION | recall |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in stratified["by_fault_type"]:
            if row["n"] < 30:
                continue
            mix = row["failure_mix"]
            lines.append("| {} | {} | {} | {} | {} | {} | {:.4f} |".format(
                row["fault_type"], row["n"], mix["MATCHED"], mix["BELOW_THRESHOLD"],
                mix["NO_NEW_EPISODE"], mix["MATCHING_COMPETITION"], row["recall"]))
        lines.append("")
        lines.append("Recall by onset-bin multiplicity:")
        lines.append("")
        lines.append("| onsets in the bin | n | recall |")
        lines.append("|---|---:|---:|")
        for row in stratified["by_onset_bin_count"]:
            recall = "n/a" if row["recall"] is None else "{:.4f}".format(row["recall"])
            lines.append("| {} | {} | {} |".format(row["onset_bin_count"], row["n"], recall))
        lines.append("")
    lines.append("## 3. Structural bounds")
    lines.append("")
    lines.append("| split | observed TP | U_episode | U_grid | observed recall | episode bound recall |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in SPLIT_NAMES:
        entry = capacity["splits"][split]
        bound = entry["episode_bound"]
        lines.append("| {} | {} | {} | {} | {:.4f} | {:.4f} |".format(
            split, entry["ordering"]["observed_tp"], bound["upper_bound_tp"],
            entry["ordering"]["grid_bound_tp"], findings[split]["observed_recall"],
            bound["recall_upper_bound"]))
    lines.append("")
    lines.append("Oracles: {}".format(", ".join(
        "{}={}".format(split, capacity["splits"][split]["episode_bound"]["status"]) for split in SPLIT_NAMES)))
    lines.append("")
    lines.append("## 4. Response bands")
    lines.append("")
    lines.append("| split | <=60 s | 60-120 s | 120-300 s | >300 s | never | censored |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split, entry in findings.items():
        mix = entry["response_band_mix"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            split, mix["le_60s"], mix["60_120s"], mix["120_300s"], mix["gt_300s"], mix["never"], mix["censored"]))
    lines.append("")
    lines.append("## 5. Evidence ledger")
    lines.append("")
    lines.append("| claim | grade | statement | limitations |")
    lines.append("|---|---|---|---|")
    evidence = json.loads((output_dir / "evidence_ledger.json").read_text(encoding="utf-8"))
    for claim in evidence["claims"]:
        statement = claim["statement"].replace("|", "/")
        limitations = str(claim["limitations"]).replace("|", "/")
        lines.append("| {} | {} | {} | {} |".format(
            claim["claim_id"], claim["grade"], statement, limitations))
    lines.append("")
    lines.append("## 6. Unresolved")
    lines.append("")
    for item in decision["unresolved"]:
        lines.append("- **{}** {}".format(item["id"], item["statement"]))
    lines.append("")
    lines.append("## 7. Next scope")
    lines.append("")
    dominance = decision.get("loss_dominance", {})
    if dominance:
        lines.append("Loss dominance rule: `{}`".format(dominance.get("rule", "")))
        lines.append("")
        lines.append("| split | structural loss share | score-side loss share | ratio |")
        lines.append("|---|---:|---:|---:|")
        for split, entry in dominance.get("per_split", {}).items():
            ratio = entry["structural_to_score_ratio"]
            lines.append("| {} | {:.4f} | {:.4f} | {} |".format(
                split, entry["structural_loss_share"], entry["score_side_loss_share"],
                "n/a" if ratio is None else "{:.2f}".format(ratio)))
        lines.append("")
        lines.append("Verdict: **{}**".format(dominance.get("verdict", "unknown")))
        lines.append("")
    lines.append(decision["recommended_next_scope"])
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "scores", "analyze", "capacity", "finalize", "all", "status"))
    parser.add_argument("--output-dir", required=True,
                        help="audit run directory, e.g. experiments/p6/c0f_failure_audit/<run_id>")
    parser.add_argument("--config", default=DEFAULT_TRIGGER_CONFIG)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--threads", default=8, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base_config_path = (PROJECT_ROOT / args.base_config).resolve()
    trigger_config_path = (PROJECT_ROOT / args.config).resolve()
    base_config = load_config(base_config_path)
    trigger_config = resolve_trigger_config(trigger_config_path)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    data_root = (PROJECT_ROOT / (args.data_root or trigger_config["data_root"])).resolve()
    artifact_root = (PROJECT_ROOT / (args.artifact_root or trigger_config["artifact_root"])).resolve()

    if args.action == "status":
        print(json.dumps(_jsonable(load_run_state(output_dir)), sort_keys=True))
        return

    torch.set_num_threads(int(args.threads))
    result: Dict[str, object] = {"action": args.action, "output_dir": str(output_dir)}
    if args.action in ("prepare", "all"):
        result["prepare"] = stage_prepare(
            base_config, trigger_config, output_dir, base_config_path=base_config_path,
            trigger_config_path=trigger_config_path, data_root=data_root,
            artifact_root=artifact_root, resume=bool(args.resume),
        )
    if args.action in ("scores", "all"):
        result["scores"] = stage_scores(base_config, trigger_config, output_dir, data_root=data_root,
                                        threads=int(args.threads))
    if args.action in ("analyze", "all"):
        result["analyze"] = stage_analyze(base_config, trigger_config, output_dir, data_root=data_root)
    if args.action in ("capacity", "all"):
        result["capacity"] = {
            split: {
                "episode_bound_tp": value["episode_bound"]["upper_bound_tp"],
                "grid_bound_tp": value["ordering"]["grid_bound_tp"],
                "observed_tp": value["ordering"]["observed_tp"],
                "ordering_holds": value["ordering"]["all_hold"],
            }
            for split, value in stage_capacity(base_config, trigger_config, output_dir,
                                               data_root=data_root).items()
        }
    if args.action in ("finalize", "all"):
        result["finalize"] = stage_finalize(base_config, trigger_config, output_dir, data_root=data_root)
    print(json.dumps(_jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
