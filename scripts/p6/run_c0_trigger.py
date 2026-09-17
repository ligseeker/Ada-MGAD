#!/usr/bin/env python3
"""P6-C0 root-agnostic system event trigger: split audit, training, frozen Test.

Stages
------

```text
audit     split feasibility + trigger label audit (no training)
train     Detector-Fit gradients, Detector-Validation checkpoint + threshold
evaluate  single frozen Test evaluation (observation only)
all       audit -> train -> evaluate
smoke     tiny synthetic end-to-end run outside the repository artifacts
```

Protocol constraints enforced by this driver:

* the frozen P5 preprocessing, event registry, 30 s grid, 300 s history, 60 s
  causal tolerance and causal max-cardinality/minimum-delay matching are reused
  unchanged; nothing is re-preprocessed,
* the detector is randomly initialised and never loads an Ada-MGAD checkpoint,
* the training loss consumes only the system-level trigger label (IGNORE bins are
  masked out); node anomaly labels and root-service labels are never read,
* the checkpoint and the threshold are chosen on Detector-Validation only,
* Test is evaluated once with the frozen checkpoint and the frozen threshold.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.ad_data import save_split_arrays
from src.e2e.protocol import (
    GAIA_SERVICES,
    load_config,
    load_registry,
    sha256_file,
    write_json,
)
from src.e2e.system_trigger import (
    DEFAULT_GRID_SECONDS,
    DEFAULT_POSITIVE_WINDOW_SECONDS,
    DEFAULT_TOLERANCE_SECONDS,
    DEFAULT_WINDOW_BINS,
    SPLIT_NAMES,
    TRIGGER_IGNORE,
    TRIGGER_POSITIVE,
    assign_legal_events,
    block_bounds,
    build_trigger_labels,
    duration_stratified_metrics,
    evaluate_system_threshold,
    field_stratified_metrics,
    onset_density_stratified_metrics,
    select_system_threshold,
    system_score_frame,
    to_builtin,
    trigger_binary_mask,
    trigger_label_counts,
    trigger_temporal_blocks,
    window_split_assignment,
)
from src.e2e.system_trigger_data import TriggerWindowDataset, build_trigger_loader
from src.e2e.system_trigger_model import SystemEventTrigger
from util.util import seed_everything


SCHEMA_AUDIT = "p6_c0_split_audit_v1"
SCHEMA_SELECTION = "p6_c0_validation_selection_v1"
SCHEMA_TEST = "p6_c0_test_evaluation_v1"
SCHEMA_MANIFEST = "p6_c0_trigger_manifest_v1"

DEFAULT_TRIGGER_CONFIG = "configs/e2e/gaia_p6_c0_system_trigger.json"
DEFAULT_BASE_CONFIG = "configs/e2e/gaia_p5_v3_preprocessing_v2.json"

MODEL_ARG_KEYS = (
    "feature_node", "feature_edge", "feature_log",
    "num_heads_edge", "num_heads_node", "num_heads_log",
    "num_heads_n2e", "num_heads_e2n", "num_layer", "dropout",
    "graph_hidden", "graph_sparse_weight", "graph_summary_mode",
)

REQUIRED_TRIGGER_CONFIG = {
    "schema_version": "p6_c0_trigger_v1",
    "split_mode": "chronological_50_20_30",
    "threshold_selection": "validation_only_exact_unique_scores",
    "matching_semantics": "causal_max_cardinality_minimum_delay",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "train", "evaluate", "all", "smoke"))
    parser.add_argument("--config", default=DEFAULT_TRIGGER_CONFIG)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--gpu", default=False, type=lambda value: value.lower() == "true")
    parser.add_argument("--threshold-workers", default=8, type=int)
    parser.add_argument("--start-method", choices=("spawn", "forkserver"), default="spawn")
    parser.add_argument("--threads", default=None, type=int)
    parser.add_argument("--smoke-root", default=None)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "p6_c0.log"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="a")]
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers,
    )


def resolve_trigger_config(path: Path) -> Mapping[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, expected in REQUIRED_TRIGGER_CONFIG.items():
        if str(data.get(key)) != expected:
            raise ValueError("trigger config drift on {}: {}".format(key, data.get(key)))
    for key in ("base_config", "data_root", "artifact_root", "output_root", "trigger_label",
                "model", "training", "threshold", "evaluation", "seed", "task"):
        if key not in data:
            raise ValueError("trigger config is missing {}".format(key))
    if data["threshold"]["objective"] != "validation_event_f1":
        raise ValueError("P6-C0 pre-registers a Validation event-F1 threshold objective")
    if str(data["evaluation"]["prediction_time"]) != "t_hat = prediction_available_time = target_bin_end":
        raise ValueError("P6-C0 must reuse the frozen prediction-available-time anchor")
    gate = data["evaluation"].get("gate", {})
    if set(gate) != {"precision", "recall", "f1"}:
        raise ValueError("pre-registered P6-C0 gate must define precision/recall/f1")
    return data


def guard_output_root(output_dir: Path, base_config: Mapping[str, object]) -> None:
    """Refuse to write P6-C0 outputs into a frozen artifact tree."""

    resolved = Path(output_dir).resolve()
    formal_root = (PROJECT_ROOT / "experiments/p5").resolve()
    if resolved == formal_root or formal_root in resolved.parents:
        raise ValueError("P6-C0 must not write inside experiments/p5")
    data_root = (PROJECT_ROOT / str(base_config.get("output_dir", ""))).resolve()
    artifacts = (PROJECT_ROOT / "artifacts").resolve()
    for frozen in (data_root, artifacts):
        if resolved == frozen or frozen in resolved.parents:
            raise ValueError("P6-C0 must not write inside frozen preprocessing artifacts")


def validate_frozen_protocol(base_config: Mapping[str, object]) -> Mapping[str, object]:
    """Assert the frozen P5 protocol values P6-C0 is required to reuse."""

    grid = int(base_config["ad"]["grid_seconds"])
    window_bins = int(base_config["ad"]["window_bins"])
    trigger = base_config["event_trigger"]
    if grid != DEFAULT_GRID_SECONDS or window_bins != DEFAULT_WINDOW_BINS:
        raise ValueError("frozen grid/window protocol drift")
    if int(trigger["matching_tolerance_seconds"]) != DEFAULT_TOLERANCE_SECONDS:
        raise ValueError("frozen causal tolerance drift")
    if str(trigger["matching_semantics"]) != "causal_max_cardinality_minimum_delay":
        raise ValueError("frozen event matching semantics drift")
    if str(trigger["threshold_selection"]) != "train_only_exact_unique_scores":
        raise ValueError("frozen threshold selection rule drift")
    if int(base_config["rca"]["window_seconds"]) != 300:
        raise ValueError("frozen RCA window drift")
    return {
        "grid_seconds": grid,
        "window_bins": window_bins,
        "history_seconds": int(base_config["rca"]["window_seconds"]),
        "tolerance_seconds": int(trigger["matching_tolerance_seconds"]),
        "matching_semantics": str(trigger["matching_semantics"]),
        "event_registry": dict(base_config["event_registry"]),
    }


def _frozen_array_paths(data_root: Path, split: str) -> Mapping[str, Path]:
    directory = Path(data_root) / split
    return {
        name: directory / (name + ".npy")
        for name in ("timestamps", "metric", "log", "trace")
    }


def source_artifact_hashes(data_root: Path, artifact_root: Path, registry_path: Path) -> Mapping[str, object]:
    records: Dict[str, object] = {}
    for split in ("train", "test"):
        for name, path in _frozen_array_paths(data_root, split).items():
            records["{}_{}.npy".format(split, name)] = {
                "path": str(path.resolve()), "sha256": sha256_file(path),
            }
    graph_path = Path(data_root) / "graph.npy"
    records["graph.npy"] = {"path": str(graph_path.resolve()), "sha256": sha256_file(graph_path)}
    manifest_path = Path(artifact_root) / "ad_data_manifest.json"
    records["ad_data_manifest.json"] = {
        "path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path),
    }
    records["event_registry"] = {
        "path": str(Path(registry_path).resolve()), "sha256": sha256_file(registry_path),
    }
    records["base_config"] = {
        "path": str((PROJECT_ROOT / DEFAULT_BASE_CONFIG).resolve()),
        "sha256": sha256_file(PROJECT_ROOT / DEFAULT_BASE_CONFIG),
    }
    return records


# ---------------------------------------------------------------------------
# protocol state
# ---------------------------------------------------------------------------


class ProtocolState:
    """Resolved frozen split, label and event populations for P6-C0."""

    def __init__(self, base_config, trigger_config, data_root: Path, registry_path: Path):
        self.base_config = base_config
        self.trigger_config = trigger_config
        self.data_root = Path(data_root)
        self.grid_seconds = DEFAULT_GRID_SECONDS
        self.window_bins = DEFAULT_WINDOW_BINS
        self.history_seconds = 300
        self.tolerance_seconds = DEFAULT_TOLERANCE_SECONDS
        self.positive_window_seconds = int(
            trigger_config["trigger_label"].get("positive_window_seconds", DEFAULT_POSITIVE_WINDOW_SECONDS)
        )
        if self.positive_window_seconds != DEFAULT_POSITIVE_WINDOW_SECONDS:
            raise ValueError("P6-C0 pre-registers a 60 s recent-onset trigger label")
        self.blocks = trigger_temporal_blocks(base_config)
        self.registry = load_registry(base_config, PROJECT_ROOT) if registry_path is None else pd.read_csv(registry_path)
        self.registry_path = Path(registry_path) if registry_path else (
            PROJECT_ROOT / str(base_config["event_registry"]["path"])
        )
        self.legal_events, self.assigned_events, self.purged_events = assign_legal_events(self.registry, self.blocks)
        self.timestamps: Dict[str, np.ndarray] = {}
        self.labels: Dict[str, np.ndarray] = {}
        self.assignments: Dict[str, pd.DataFrame] = {}
        self.frozen_test_windows: Optional[int] = None
        for source in ("train", "test"):
            timestamps = np.load(Path(data_root) / source / "timestamps.npy", mmap_mode="r")
            timestamps = np.asarray(timestamps, dtype=np.int64)
            self.timestamps[source] = timestamps
            self.labels[source] = build_trigger_labels(
                timestamps, self.legal_events,
                positive_window_seconds=self.positive_window_seconds,
            )
            assignment = window_split_assignment(
                timestamps, self.blocks,
                grid_seconds=self.grid_seconds, window_bins=self.window_bins,
            )
            assignment["source_array"] = source
            self.assignments[source] = assignment

    # -- population helpers ------------------------------------------------

    def gt_events(self, split: str) -> pd.DataFrame:
        frame = self.assigned_events.loc[self.assigned_events["split"].astype(str) == str(split)]
        if frame.empty:
            raise ValueError("no GT events assigned to split {}".format(split))
        return frame.sort_values(["start_ms", "source_index", "case_id"], kind="stable").reset_index(drop=True)

    def sample_indices(self, split: str) -> np.ndarray:
        source = "test" if split == "test" else "train"
        frame = self.assignments[source]
        selected = frame.loc[(frame["split"].astype(str) == str(split)) & frame["keep"]]
        return np.sort(selected["sample_index"].to_numpy(dtype=np.int64))

    def build_dataset(self, split: str) -> TriggerWindowDataset:
        source = "test" if split == "test" else "train"
        indices = self.sample_indices(split)
        return TriggerWindowDataset(
            self.data_root / source, self.window_bins, self.grid_seconds,
            indices, self.labels[source], split,
        )

    def window_report(self) -> Mapping[str, object]:
        report: Dict[str, object] = {}
        for source, frame in self.assignments.items():
            kept = frame.loc[frame["keep"]]
            purged = frame.loc[~frame["keep"]]
            report[source] = {
                "windows_in_array": int(len(frame)),
                "kept": int(len(kept)),
                "purged": int(len(purged)),
                "kept_by_split": {
                    name: int((kept["split"].astype(str) == name).sum()) for name in SPLIT_NAMES
                },
                "purged_by_reason": {
                    str(key): int(value) for key, value in purged["purge_reason"].value_counts().items()
                },
                "purged_by_owning_block": {
                    str(key): int(value) for key, value in purged["split"].value_counts().items()
                },
            }
        return report

    def purged_label_counts(self) -> Mapping[str, object]:
        counts: Dict[str, int] = {"positive": 0, "ignore": 0, "negative": 0}
        for source, frame in self.assignments.items():
            purged = frame.loc[~frame["keep"]]
            if purged.empty:
                continue
            target = purged["sample_index"].to_numpy(dtype=np.int64) + self.window_bins - 1
            values = self.labels[source][target]
            for value, name in ((TRIGGER_POSITIVE, "positive"), (TRIGGER_IGNORE, "ignore"), (0, "negative")):
                counts[name] += int((values == value).sum())
        return counts


def label_statistics(state: ProtocolState, split: str) -> Mapping[str, object]:
    source = "test" if split == "test" else "train"
    indices = state.sample_indices(split)
    target = indices + state.window_bins - 1
    labels = state.labels[source][target]
    return trigger_label_counts(labels)


def event_statistics(state: ProtocolState, split: str, grid_seconds: int) -> Mapping[str, object]:
    events = state.gt_events(split)
    durations = (events["end_ms"] - events["start_ms"]).to_numpy(dtype=np.int64) / 1000.0
    return {
        "ground_truth_events": int(len(events)),
        "fault_type_distribution": {
            str(key): int(value) for key, value in events["fault_type"].value_counts().sort_index().items()
        },
        "root_service_distribution": {
            str(key): int(value) for key, value in events["service"].value_counts().sort_index().items()
        },
        "duration_seconds": {
            "min": float(durations.min()), "max": float(durations.max()),
            "mean": float(durations.mean()), "median": float(np.median(durations)),
            "p95": float(np.percentile(durations, 95)),
        },
        "duration_stratum_counts": {
            str(key): int(value) for key, value in
            pd.Series(_duration_strata(durations)).value_counts().sort_index().items()
        },
    }


def _duration_strata(durations: np.ndarray) -> np.ndarray:
    from src.e2e.system_trigger import DURATION_STRATUM_LABELS
    return np.asarray(DURATION_STRATUM_LABELS, dtype=object)[
        np.searchsorted(np.array([15.0, 30.0, 60.0, 300.0]), durations, side="left")
    ]


def onset_statistics(state: ProtocolState, split: str, grid_seconds: int) -> Mapping[str, object]:
    from src.e2e.system_trigger import onset_density
    events = state.gt_events(split)
    origin = int(state.blocks[0].start_ms)
    density = onset_density(events, origin_ms=origin, grid_seconds=grid_seconds)
    return {
        "bins_with_onsets": int(len(density)),
        "bins_with_multiple_onsets": int((density["onset_count"] >= 2).sum()),
        "bins_with_multiple_root_services": int((density["root_service_count"] >= 2).sum()),
        "events_in_multi_onset_bins": int(
            density.loc[density["onset_count"] >= 2, "onset_count"].sum()
        ),
        "max_onsets_in_one_bin": int(density["onset_count"].max()),
    }


def structural_checks(state: ProtocolState) -> Mapping[str, object]:
    """Pre-registered split-feasibility checks; any failure is a STOP condition."""

    checks: List[Mapping[str, object]] = []
    populations: Dict[str, pd.DataFrame] = {}
    labels: Dict[str, Mapping[str, object]] = {}
    for split in SPLIT_NAMES:
        frame = state.assigned_events.loc[state.assigned_events["split"].astype(str) == str(split)]
        populations[split] = frame.reset_index(drop=True)
        labels[split] = label_statistics(state, split)
        checks.append({
            "name": "{}_has_gt_events".format(split),
            "ok": bool(len(frame) > 0),
            "detail": int(len(frame)),
        })
        checks.append({
            "name": "{}_has_positive_labels".format(split),
            "ok": bool(labels[split]["counts"]["positive"] > 0),
            "detail": int(labels[split]["counts"]["positive"]),
        })
    fit = populations["fit"]
    for field, split in (("fault_type", "test"), ("service", "test")):
        if populations[split].empty:
            continue
        test_counts = populations[split][field].value_counts(normalize=True)
        for value, share in test_counts.items():
            if share < 0.10:
                continue
            present = int((fit[field].astype(str) == str(value)).sum())
            checks.append({
                "name": "dominant_{}_{}_present_in_fit".format(field, value),
                "ok": bool(present > 0),
                "detail": {"test_share": float(share), "fit_events": present},
            })
    if state.frozen_test_windows is not None:
        indices = state.sample_indices("test")
        checks.append({
            "name": "test_population_matches_frozen_p5",
            "ok": bool(len(indices) == int(state.frozen_test_windows)),
            "detail": {"p6_c0_test_windows": int(len(indices)),
                       "frozen_p5_test_windows": int(state.frozen_test_windows)},
        })
        checks.append({
            "name": "test_window_identity_matches_frozen_p5",
            "ok": bool(np.array_equal(indices, np.arange(int(state.frozen_test_windows), dtype=np.int64))),
            "detail": "sample index 0..N-1 of the frozen Test array, unchanged",
        })
    return {
        "checks": checks,
        "passed": bool(all(check["ok"] for check in checks)),
    }


def purged_event_summary(purged: pd.DataFrame) -> Mapping[str, object]:
    if len(purged) == 0:
        return {"count": 0, "case_ids": [], "fault_type_distribution": {}}
    return {
        "count": int(len(purged)),
        "case_ids": [str(value) for value in purged["case_id"].tolist()],
        "fault_type_distribution": {
            str(key): int(value) for key, value in purged["fault_type"].value_counts().items()
        },
    }


def run_audit(state: ProtocolState, output_dir: Path, provenance: Mapping[str, object]) -> Mapping[str, object]:
    grid = state.grid_seconds
    audit = {
        "schema_version": SCHEMA_AUDIT,
        "generated_at_utc": utc_now(),
        "git_commit": git_head(),
        "formal_result": False,
        "stage": "split_and_label_audit",
        "split_mode": "chronological_50_20_30",
        "blocks": block_bounds(state.blocks),
        "purge_rule": (
            "a window is kept only when its full [t-history, t) input interval lies inside the "
            "block that owns t = prediction_available_time; a GT event is a metric case only when "
            "it is complete inside one block (frozen assign_event_blocks rule)"
        ),
        "positive_window_seconds": state.positive_window_seconds,
        "history_seconds": state.history_seconds,
        "window_bins": state.window_bins,
        "windows": state.window_report(),
        "purged_window_label_counts": state.purged_label_counts(),
        "labels": {split: label_statistics(state, split) for split in SPLIT_NAMES},
        "events": {split: event_statistics(state, split, grid) for split in SPLIT_NAMES},
        "onset_density": {split: onset_statistics(state, split, grid) for split in SPLIT_NAMES},
        "purged_events": purged_event_summary(state.purged_events),
        "label_legal_event_population": {
            "count": int(len(state.legal_events)),
            "note": (
                "labels are built from every GT event inside the frozen detector timeline, "
                "including the single event that crosses the frozen 70/30 boundary, so an "
                "ongoing fault is never labelled NEGATIVE; metric populations use only "
                "complete in-block events"
            ),
        },
        "provenance": provenance,
    }
    audit["structural_checks"] = structural_checks(state)
    split_manifest = {
        "schema_version": "p6_c0_split_manifest_v1",
        "generated_at_utc": utc_now(),
        "git_commit": audit["git_commit"],
        "split_mode": "chronological_50_20_30",
        "blocks": block_bounds(state.blocks),
        "window_bins": state.window_bins,
        "grid_seconds": grid,
        "history_seconds": state.history_seconds,
        "purge_rule": audit["purge_rule"],
        "windows": audit["windows"],
        "purged_window_label_counts": audit["purged_window_label_counts"],
        "purged_events": audit["purged_events"],
        "labels": audit["labels"],
        "events": audit["events"],
        "source_artifacts": provenance.get("source_artifacts", {}),
        "test_used_for_split_design": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "split_audit.json", to_builtin(audit))
    write_json(output_dir / "split_manifest.json", to_builtin(split_manifest))
    return audit


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------


def build_model_args(trigger_config, base_config, manifest, seed: int, gpu: bool):
    ad_model = base_config["ad_model"]
    args = {key: ad_model[key] for key in MODEL_ARG_KEYS}
    args.update({
        "main_model": "P6-C0-SystemEventTrigger",
        "random_seed": int(seed),
        "gpu": bool(gpu),
        "window": DEFAULT_WINDOW_BINS,
        "step": int(base_config["ad"]["window_step"]),
        "num_nodes": len(GAIA_SERVICES),
        "raw_node": int(manifest["dimensions"]["raw_node"]),
        "log_len": int(manifest["dimensions"]["log_len"]),
        "raw_edge": int(manifest["dimensions"]["raw_edge"]),
        "batch_size": int(trigger_config["training"]["batch_size"]),
        "head_hidden": int(trigger_config["model"].get("head_hidden", 32)),
    })
    return args


def move_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Mapping[str, torch.Tensor]:
    moved = {}
    for name, value in batch.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.is_floating_point():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            tensor = tensor.to(device=device, dtype=torch.float32)
        else:
            tensor = tensor.to(device=device)
        moved[name] = tensor
    return moved


@torch.no_grad()
def infer_trigger_logits(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    index_list: List[np.ndarray] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        logits, _ = model(batch)
        logits_list.append(logits.detach().cpu().numpy().reshape(-1))
        index_list.append(batch["sample_index"].detach().cpu().numpy().reshape(-1))
    if not logits_list:
        raise ValueError("trigger loader yielded no batches")
    logits = np.concatenate(logits_list)
    indices = np.concatenate(index_list).astype(np.int64)
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    keep = np.ones(len(sorted_indices), dtype=bool)
    keep[1:] = sorted_indices[1:] != sorted_indices[:-1]
    return sorted_indices[keep], logits[order][keep]


def score_dataset(model, dataset: TriggerWindowDataset, batch_size: int, num_workers: int,
                  device: torch.device, split: str) -> Mapping[str, np.ndarray]:
    loader = build_trigger_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    indices, logits = infer_trigger_logits(model, loader, device)
    if not np.array_equal(indices, dataset.sample_indices):
        raise ValueError("trigger inference did not cover every window exactly once")
    positions = np.searchsorted(dataset.sample_indices, indices)
    prediction_times = dataset.prediction_times(positions)
    labels = dataset.labels_at(positions)
    scores = 1.0 / (1.0 + np.exp(-logits))
    return {
        "split": str(split), "sample_index": indices,
        "prediction_available_time": prediction_times,
        "logits": logits, "system_score": scores, "trigger_label": labels,
    }


def bin_diagnostics(logits: np.ndarray, labels: np.ndarray) -> Mapping[str, object]:
    logits = np.asarray(logits, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    mask = labels != TRIGGER_IGNORE
    target = (labels[mask] == TRIGGER_POSITIVE).astype(np.float64)
    selected = logits[mask]
    counts = {
        "positive": int((labels == TRIGGER_POSITIVE).sum()),
        "ignore": int((labels == TRIGGER_IGNORE).sum()),
        "negative": int((labels == 0).sum()),
    }
    if len(selected) == 0 or len(np.unique(target)) < 2:
        auroc = None
        average_precision = None
    else:
        auroc = float(roc_auc_score(target, selected))
        average_precision = float(average_precision_score(target, selected))
    # numerically stable BCE on the raw logits over non-ignored bins
    bce = float(np.mean(np.logaddexp(0.0, selected) - target * selected)) if len(selected) else None
    return {
        "positive_bins": counts["positive"], "ignore_bins": counts["ignore"],
        "negative_bins": counts["negative"], "scored_bins": int(mask.sum()),
        "auroc": auroc, "average_precision": average_precision, "bce": bce,
        "ignore_bins_masked_from_loss_and_metrics": True,
    }


def write_predictions(path: Path, dataset: TriggerWindowDataset, output: Mapping[str, np.ndarray],
                      threshold: float) -> None:
    rows = []
    for position in range(len(dataset)):
        metadata = dataset.metadata(position)
        score = float(output["system_score"][position])
        rows.append({
            "split": metadata.split, "sample_index": metadata.sample_index,
            "window_start_time": metadata.window_start_time,
            "window_end_time": metadata.window_end_time,
            "target_bin_start": metadata.target_bin_start,
            "target_bin_end": metadata.target_bin_end,
            "prediction_available_time": metadata.prediction_available_time,
            "system_trigger_score": score,
            "system_trigger_logit": float(output["logits"][position]),
            "trigger_label": int(metadata.trigger_label),
            "binary_prediction": int(score >= float(threshold)),
        })
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _selection_key(metrics: Mapping[str, object], threshold: float) -> Tuple[float, float, float]:
    """Checkpoint tie-break: event F1, then recall, then threshold (pre-registered)."""

    return (
        float(metrics.get("event_f1") or 0.0),
        float(metrics.get("event_recall") or 0.0),
        float(threshold),
    )


def run_training(state: ProtocolState, output_dir: Path, model_args, training, manifest_path: Path,
                 manifest_sha: str, threshold_workers: int, start_method: str) -> Mapping[str, object]:
    device = torch.device("cuda" if model_args["gpu"] and torch.cuda.is_available() else "cpu")
    seed_everything(int(model_args["random_seed"]))
    torch.manual_seed(int(model_args["random_seed"]))
    graph = np.load(state.data_root / "graph.npy", allow_pickle=False)
    model = SystemEventTrigger(graph, **model_args).to(device)
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    logging.info("P6-C0 system trigger: %d parameters", parameter_count)

    fit_dataset = state.build_dataset("fit")
    validation_dataset = state.build_dataset("validation")
    fit_loader = build_trigger_loader(
        fit_dataset, batch_size=int(training["batch_size"]), num_workers=int(training["num_workers"])
    )
    validation_loader = build_trigger_loader(
        validation_dataset, batch_size=int(training["batch_size"]), num_workers=int(training["num_workers"])
    )
    validation_gt = state.gt_events("validation")

    fit_labels = fit_dataset.labels_at()
    target, mask = trigger_binary_mask(fit_labels)
    positives = float(target.sum())
    negatives = float(((1 - target) * mask).sum())
    if positives <= 0 or negatives <= 0:
        raise ValueError("Fit split must contain positive and negative trigger labels")
    pos_weight = negatives / positives
    logging.info(
        "Fit labels: positive=%d ignore=%d negative=%d pos_weight=%.6f",
        int(positives), int((fit_labels == TRIGGER_IGNORE).sum()), int(negatives), pos_weight,
    )

    from adabelief_pytorch import AdaBelief

    optimizer = AdaBelief(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, int(training["scheduler_step"]), float(training["scheduler_gamma"])
    )
    criterion = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_validation_event_f1.pt"
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    grad_clip = float(training["grad_clip_norm"])
    history: List[Mapping[str, object]] = []
    best: Dict[str, object] = {"key": None}
    worse_count = 0
    stop_reason = "max_epochs"

    for epoch in range(max_epochs):
        model.train()
        sums = {"total": 0.0, "bce": 0.0, "graph": 0.0}
        batches = 0
        for batch in fit_loader:
            batch = move_to_device(batch, device)
            logits, graph_reg = model(batch)
            labels = batch["trigger_label"]
            loss_mask = (labels != TRIGGER_IGNORE).to(dtype=torch.float32)
            loss_target = (labels == TRIGGER_POSITIVE).to(dtype=torch.float32)
            if float(loss_mask.sum()) <= 0:
                continue
            loss_vector = criterion(logits, loss_target)
            bce = (loss_vector * loss_mask).sum() / loss_mask.sum()
            loss = bce + graph_reg
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite P6-C0 loss at epoch {}".format(epoch))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
            optimizer.step()
            sums["total"] += float(loss.item())
            sums["bce"] += float(bce.item())
            sums["graph"] += float(graph_reg.item())
            batches += 1
        if batches == 0:
            raise ValueError("Fit loader yielded no usable batches")
        scheduler.step()
        epoch_loss = {key: value / batches for key, value in sums.items()}

        validation_output = score_dataset(
            model, validation_dataset, int(training["batch_size"]), int(training["num_workers"]),
            device, "validation",
        )
        validation_frame = system_score_frame(
            "validation", validation_output["prediction_available_time"], validation_output["system_score"]
        )
        selection = select_system_threshold(
            validation_frame, validation_gt,
            grid_seconds=state.grid_seconds, tolerance_seconds=state.tolerance_seconds,
            workers=int(threshold_workers), start_method=str(start_method),
        )
        metrics = selection.metrics
        key = _selection_key(metrics, selection.threshold)
        entry = {
            "epoch": int(epoch),
            "train_loss": float(epoch_loss["total"]),
            "train_bce": float(epoch_loss["bce"]),
            "train_graph_regularization": float(epoch_loss["graph"]),
            "validation_threshold": float(selection.threshold),
            "validation_candidate_count": int(selection.candidate_count),
            "validation_metrics": to_builtin(metrics),
            "selection_key": [float(value) for value in key],
        }
        history.append(entry)
        logging.info(
            "epoch %d: train_loss=%.6f val_threshold=%.6f val P=%.4f R=%.4f F1=%.4f",
            epoch, epoch_loss["total"], selection.threshold,
            float(metrics["event_precision"]), float(metrics["event_recall"]), float(metrics["event_f1"]),
        )
        if best["key"] is None or key > tuple(best["key"]):
            best = {
                "key": [float(value) for value in key],
                "epoch": int(epoch),
                "threshold": float(selection.threshold),
                "metrics": to_builtin(metrics),
                "state": {name: value.detach().clone() for name, value in model.state_dict().items()},
            }
            worse_count = 0
        else:
            worse_count += 1
        if patience > 0 and worse_count >= patience:
            stop_reason = "validation_event_f1_patience"
            break

    if best["key"] is None:
        raise RuntimeError("P6-C0 training produced no validation checkpoint")
    torch.save(best["state"], checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    selection_record = {
        "schema_version": SCHEMA_SELECTION,
        "generated_at_utc": utc_now(),
        "git_commit": git_head(),
        "formal_result": True,
        "stage": "detector_validation",
        "selected_epoch": int(best["epoch"]),
        "selected_validation_threshold": float(best["threshold"]),
        "selected_validation_metrics": best["metrics"],
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha},
        "parameter_count": parameter_count,
        "model_args": to_builtin(model_args),
        "training": dict(training),
        "fit_label_counts": {
            "positive": int(positives), "ignore": int((fit_labels == TRIGGER_IGNORE).sum()),
            "negative": int(negatives), "pos_weight": float(pos_weight),
        },
        "validation_gt_events": int(len(validation_gt)),
        "epochs_completed": int(len(history)),
        "stop_reason": stop_reason,
        "checkpoint_tie_break": "event_f1, then recall, then threshold",
        "threshold_tie_break": "highest threshold among equal Validation event F1",
        "test_used_for_selection": False,
        "history": history,
        "data_manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha},
    }
    write_json(output_dir / "validation_selection.json", to_builtin(selection_record))
    write_json(output_dir / "training_log.json", to_builtin({
        "schema_version": "p6_c0_training_log_v1",
        "generated_at_utc": utc_now(),
        "parameter_count": parameter_count,
        "fit_windows": int(len(fit_dataset)),
        "validation_windows": int(len(validation_dataset)),
        "history": history,
        "stop_reason": stop_reason,
    }))
    return selection_record


# ---------------------------------------------------------------------------
# frozen Test evaluation
# ---------------------------------------------------------------------------


def run_evaluation(state: ProtocolState, output_dir: Path, model_args, training,
                   manifest_path: Path, manifest_sha: str) -> Mapping[str, object]:
    selection_path = output_dir / "validation_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("P6-C0 evaluation requires a frozen validation_selection.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(selection["checkpoint"]["path"])
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != str(selection["checkpoint"]["sha256"]):
        raise ValueError("frozen P6-C0 checkpoint SHA-256 changed since validation selection")
    threshold = float(selection["selected_validation_threshold"])
    device = torch.device("cuda" if model_args["gpu"] and torch.cuda.is_available() else "cpu")

    graph = np.load(state.data_root / "graph.npy", allow_pickle=False)
    model = SystemEventTrigger(graph, **model_args).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    validation_dataset = state.build_dataset("validation")
    validation_output = score_dataset(
        model, validation_dataset, int(training["batch_size"]), int(training["num_workers"]),
        device, "validation",
    )
    validation_frame = system_score_frame(
        "validation", validation_output["prediction_available_time"], validation_output["system_score"]
    )
    _, _, replayed = evaluate_system_threshold(
        validation_frame, state.gt_events("validation"), threshold,
        grid_seconds=state.grid_seconds, tolerance_seconds=state.tolerance_seconds,
    )
    replay_keys = ("event_precision", "event_recall", "event_f1", "true_positive_events",
                   "false_positive_events", "false_negative_events")
    replay_match = all(
        _close(replayed.get(key), selection["selected_validation_metrics"].get(key)) for key in replay_keys
    )
    if not replay_match:
        raise ValueError("frozen checkpoint/threshold does not reproduce the recorded Validation metrics")

    test_dataset = state.build_dataset("test")
    test_output = score_dataset(
        model, test_dataset, int(training["batch_size"]), int(training["num_workers"]), device, "test",
    )
    test_gt = state.gt_events("test")
    test_frame = system_score_frame(
        "test", test_output["prediction_available_time"], test_output["system_score"]
    )
    episodes, matching, metrics = evaluate_system_threshold(
        test_frame, test_gt, threshold,
        grid_seconds=state.grid_seconds, tolerance_seconds=state.tolerance_seconds,
    )
    gate = state.trigger_config["evaluation"]["gate"]
    case_frame = test_gt[["case_id", "fault_type", "service", "start_ms", "end_ms"]].copy()
    case_frame["duration_seconds"] = (case_frame["end_ms"] - case_frame["start_ms"]) / 1000.0
    stratifications = {
        "duration": duration_stratified_metrics(matching, case_frame, grid_seconds=state.grid_seconds),
        "fault_type": field_stratified_metrics(matching, "fault_type"),
        "root_service": field_stratified_metrics(matching, "gt_service"),
        "onset_density": onset_density_stratified_metrics(
            matching, legal_events=state.legal_events,
            origin_ms=int(state.blocks[0].start_ms), grid_seconds=state.grid_seconds,
        ),
    }
    decision = decide_outcome(metrics, stratifications, gate)
    metrics_path = output_dir / "test_metrics.json"
    episodes_path = output_dir / "test_episodes.csv"
    matching_path = output_dir / "test_matching.csv"
    predictions_path = output_dir / "test_predictions.csv"
    write_predictions(predictions_path, test_dataset, test_output, threshold)
    episodes.to_csv(episodes_path, index=False, lineterminator="\n")
    matching.to_csv(matching_path, index=False, lineterminator="\n")
    result = {
        "schema_version": SCHEMA_TEST,
        "generated_at_utc": utc_now(),
        "git_commit": git_head(),
        "formal_result": True,
        "stage": "frozen_test_evaluation",
        "model": "P6-C0 root-agnostic system event trigger",
        "model_parameters": int(selection["parameter_count"]),
        "selected_epoch": int(selection["selected_epoch"]),
        "frozen_validation_threshold": threshold,
        "selected_validation_metrics": selection["selected_validation_metrics"],
        "validation_replay_at_frozen_threshold": {"metrics": to_builtin(replayed), "matched": replay_match},
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha},
        "test_windows": int(len(test_dataset)),
        "test_ground_truth_events": int(len(test_gt)),
        "test_event_metrics": to_builtin(metrics),
        "test_bin_diagnostics": to_builtin(
            bin_diagnostics(test_output["logits"], test_output["trigger_label"])
        ),
        "test_stratifications": to_builtin(stratifications),
        "pre_registered_gate": to_builtin(decision["gate"]),
        "p6_c0_decision": to_builtin(decision),
        "historical_node_supervised_reference": {
            "source": "P5 formal fused Ada-MGAD trigger (node-supervised)",
            "event_precision": 0.9827, "event_recall": 0.6399, "event_f1": 0.7751,
            "comparison_note": (
                "historical reference only; it is a node-supervised detector trained on the frozen "
                "70/30 split with a different label definition, so this is not a controlled "
                "equal-protocol SOTA comparison"
            ),
        },
        "artifact_files": {
            "test_predictions": {"path": str(predictions_path.resolve()), "sha256": sha256_file(predictions_path)},
            "test_episodes": {"path": str(episodes_path.resolve()), "sha256": sha256_file(episodes_path)},
            "test_matching": {"path": str(matching_path.resolve()), "sha256": sha256_file(matching_path)},
        },
        "test_used_for_selection": False,
        "rca_executed": False,
        "data_manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha},
    }
    write_json(metrics_path, to_builtin(result))
    return result


def _close(left, right, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def decide_outcome(metrics: Mapping[str, object], stratifications: Mapping[str, object],
                   gate: Mapping[str, object]) -> Mapping[str, object]:
    """Apply the pre-registered P6-C0 GO / BORDERLINE / NO-GO rule."""

    precision = float(metrics["event_precision"])
    recall = float(metrics["event_recall"])
    f1 = float(metrics["event_f1"])
    gate_result = {
        "precision": precision >= float(gate["precision"]),
        "recall": recall >= float(gate["recall"]),
        "f1": f1 >= float(gate["f1"]),
        "rule": "all three pre-registered engineering margins must hold for GO",
    }
    gate_result["passed"] = bool(all(gate_result[key] for key in ("precision", "recall", "f1")))
    failures: List[Mapping[str, object]] = []
    for row in stratifications["duration"]:
        if row["ground_truth_events"] >= 30 and row.get("recall") is not None and row["recall"] < 0.25:
            failures.append({"kind": "duration_recall", "stratum": row["duration_stratum"],
                             "n": row["ground_truth_events"], "recall": row["recall"]})
    for row in stratifications["fault_type"]:
        if row["ground_truth_events"] >= 30 and row.get("recall") is not None and row["recall"] < 0.25:
            failures.append({"kind": "fault_type_recall", "stratum": row["fault_type"],
                             "n": row["ground_truth_events"], "recall": row["recall"]})
    onset = stratifications["onset_density"]
    single, multi = onset["single_onset"], onset["multi_onset"]
    if (
        single["ground_truth_events"] >= 30 and multi["ground_truth_events"] >= 30
        and single.get("recall") is not None and multi.get("recall") is not None
        and multi["recall"] < 0.5 * single["recall"]
    ):
        failures.append({
            "kind": "onset_density_recall", "stratum": "multi_onset",
            "n": multi["ground_truth_events"], "recall": multi["recall"],
            "single_onset_recall": single["recall"],
        })
    if gate_result["passed"] and not failures:
        outcome = "GO"
    elif f1 < 0.60 or recall < 0.40:
        outcome = "NO-GO"
    else:
        outcome = "BORDERLINE"
    return {
        "outcome": outcome,
        "gate": gate_result,
        "stratification_failures": failures,
        "rule": "GO = gate holds and no stratification failure; NO-GO = F1 < 0.60 or Recall < 0.40; else BORDERLINE",
    }


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def smoke(trigger_config, base_config, root: Path) -> Mapping[str, object]:
    """Tiny synthetic end-to-end run; never a formal result."""

    root = Path(root)
    data_root = root / "data"
    output_dir = root / "output"
    seed = int(trigger_config["seed"])
    rng = np.random.RandomState(seed)
    grid_ms = DEFAULT_GRID_SECONDS * 1000
    origin = 1625133600000
    total_bins = 400
    fit_bins, validation_bins = 200, 280
    counts = {"train": 300, "test": 100}
    for split, count in counts.items():
        offset = 0 if split == "train" else counts["train"]
        timestamps = origin + (offset + np.arange(count, dtype=np.int64)) * grid_ms
        metric = rng.normal(size=(count, 10, 48)).astype(np.float32)
        logs = rng.uniform(size=(count, 10, 32)).astype(np.float32)
        trace = rng.uniform(size=(count, 10, 10, 8)).astype(np.float32)
        labels = np.zeros((count, 10), dtype=np.int8)
        save_split_arrays(data_root, split, {
            "timestamps": timestamps, "metric": metric, "log": logs, "trace": trace,
            "labels": labels, "label_mask": labels.copy(),
        })
    graph = np.zeros((10, 10), dtype=np.float32)
    for index in range(10):
        graph[index, (index + 1) % 10] = 1.0
        graph[(index + 1) % 10, index] = 1.0
    np.save(data_root / "graph.npy", graph, allow_pickle=False)
    manifest = {"dimensions": {"raw_node": 48, "log_len": 32, "raw_edge": 8}}
    events = pd.DataFrame([
        {"case_id": "smoke-{:03d}".format(bin_index), "source_index": bin_index,
         "service": GAIA_SERVICES[bin_index % len(GAIA_SERVICES)], "fault_type": "login_failure",
         "start_ms": int(origin + bin_index * grid_ms),
         "end_ms": int(origin + bin_index * grid_ms + 60_000),
         "detector_domain": True}
        for bin_index in (50, 120, 150, 250, 350, 360)
    ])
    registry_path = root / "registry.csv"
    events.to_csv(registry_path, index=False, lineterminator="\n")

    smoke_base = json.loads(json.dumps(base_config))
    smoke_base["split"] = {
        "mode": "chronological_metric_50_20_30_smoke",
        "absolute_start_ms": int(origin),
        "absolute_end_ms": int(origin + total_bins * grid_ms),
        "boundary_ms": int(origin + validation_bins * grid_ms),
    }
    smoke_trigger = json.loads(json.dumps(trigger_config))
    smoke_trigger["training"]["max_epochs"] = 1
    smoke_trigger["training"]["patience"] = 0

    state = ProtocolState(smoke_base, smoke_trigger, data_root, registry_path)
    if not all(len(state.gt_events(split)) > 0 for split in SPLIT_NAMES):
        raise ValueError("smoke split construction lost a GT population")
    model_args = build_model_args(smoke_trigger, smoke_base, manifest, seed, gpu=False)
    model_args.update({
        "feature_node": 8, "feature_edge": 4, "feature_log": 4,
        "graph_hidden": 8, "head_hidden": 8, "batch_size": 32,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = run_audit(state, output_dir, {"smoke": True})
    seed_everything(seed)
    torch.manual_seed(seed)
    model = SystemEventTrigger(graph, **model_args)
    batch = {
        "data_node": torch.randn(32, 10, 10, 48),
        "data_log": torch.rand(32, 10, 10, 32),
        "data_edge": torch.rand(32, 10, 10, 10, 8),
    }
    logits, graph_reg = model(batch)
    if logits.shape != (32,) or not torch.isfinite(logits).all() or not torch.isfinite(graph_reg):
        raise ValueError("smoke forward pass did not produce finite scalar system logits")
    # exercise the real training + frozen Test path on the synthetic fixture
    selection = run_training(
        state, output_dir, model_args, smoke_trigger["training"],
        registry_path, sha256_file(registry_path), 1, "spawn",
    )
    evaluation = run_evaluation(
        state, output_dir, model_args, smoke_trigger["training"],
        registry_path, sha256_file(registry_path),
    )
    _write_manifest(
        state, output_dir, {"trigger_config": {"path": str(registry_path), "sha256": ""},
                            "base_config": {"path": str(registry_path), "sha256": ""},
                            "source_artifacts": {}},
        evaluation, smoke_trigger, model_args, registry_path, sha256_file(registry_path),
    )
    summary = {
        "schema_version": "p6_c0_smoke_v1", "generated_at_utc": utc_now(),
        "git_commit": git_head(), "status": "PASS", "formal_result": False,
        "root": str(root),
        "structural_checks_passed": bool(audit["structural_checks"]["passed"]),
        "windows_per_split": {split: int(len(state.sample_indices(split))) for split in SPLIT_NAMES},
        "purged_windows": {
            source: int((~frame["keep"]).sum()) for source, frame in state.assignments.items()
        },
        "logits_shape": list(logits.shape),
        "graph_regularization": float(graph_reg.item()),
        "selected_epoch": int(selection["selected_epoch"]),
        "validation_threshold": float(selection["selected_validation_threshold"]),
        "validation_event_metrics": selection["selected_validation_metrics"],
        "test_event_metrics": evaluation["test_event_metrics"],
        "test_decision": evaluation["p6_c0_decision"]["outcome"],
    }
    write_json(root / "smoke_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    base_config_path = (PROJECT_ROOT / args.base_config).resolve()
    base_config = load_config(base_config_path)
    trigger_config_path = (PROJECT_ROOT / args.config).resolve()
    trigger_config = resolve_trigger_config(trigger_config_path)

    if args.threads:
        torch.set_num_threads(int(args.threads))

    if args.action == "smoke":
        root = Path(args.smoke_root).resolve() if args.smoke_root else Path(
            tempfile.mkdtemp(prefix="p6c0_smoke_")
        )
        summary = smoke(trigger_config, base_config, root)
        print(json.dumps(to_builtin(summary), sort_keys=True))
        return

    data_root = (PROJECT_ROOT / (args.data_root or trigger_config["data_root"])).resolve()
    artifact_root = (PROJECT_ROOT / (args.artifact_root or trigger_config["artifact_root"])).resolve()
    registry_path = Path(args.registry).resolve() if args.registry else None
    output_dir = (PROJECT_ROOT / (args.output_dir or trigger_config["output_root"])).resolve()
    guard_output_root(output_dir, base_config)
    setup_logging(output_dir)

    manifest_path = artifact_root / "ad_data_manifest.json"
    from scripts.p5.run_i1_ad import load_manifest

    manifest = load_manifest(artifact_root, base_config_path)
    if set(manifest.get("split_counts", {})) != {"train", "test"}:
        raise ValueError("frozen Ada-MGAD manifest must contain exactly Train/Test splits")
    manifest_sha = sha256_file(manifest_path)

    protocol = validate_frozen_protocol(base_config)
    state = ProtocolState(base_config, trigger_config, data_root, registry_path)
    state.frozen_test_windows = int(manifest["split_counts"]["test"]["windows"])
    provenance = {
        "trigger_config": {"path": str(trigger_config_path), "sha256": sha256_file(trigger_config_path)},
        "base_config": {"path": str(base_config_path), "sha256": sha256_file(base_config_path)},
        "source_artifacts": source_artifact_hashes(data_root, artifact_root, state.registry_path),
        "frozen_protocol": protocol,
        "no_retraining_of_ada_mgad": True,
        "ada_mgad_checkpoint_loaded": False,
        "node_anomaly_labels_read": False,
        "root_service_label_used": False,
        "random_seed": int(trigger_config["seed"]),
        "torch_threads": int(torch.get_num_threads()),
        "git_commit": git_head(),
    }
    logging.info("P6-C0 %s starting; output=%s", args.action, output_dir)

    result: Mapping[str, object] = {"action": args.action, "output_dir": str(output_dir)}
    if args.action in ("audit", "all"):
        audit = run_audit(state, output_dir, provenance)
        result["audit"] = {
            "structural_checks": audit["structural_checks"],
            "windows": audit["windows"],
            "labels": audit["labels"],
            "events": audit["events"],
        }
        if not audit["structural_checks"]["passed"]:
            logging.error("P6-C0 split feasibility STOP: %s", audit["structural_checks"])
            print(json.dumps(to_builtin({
                "status": "STOP_SPLIT_FEASIBILITY", "audit": audit["structural_checks"],
            }), sort_keys=True))
            sys.exit(2)
        logging.info("P6-C0 split audit PASSED: %s", audit["structural_checks"]["checks"])
    if args.action in ("train", "all"):
        model_args = build_model_args(
            trigger_config, base_config, manifest, int(trigger_config["seed"]), bool(args.gpu)
        )
        selection = run_training(
            state, output_dir, model_args, trigger_config["training"],
            manifest_path, manifest_sha, int(args.threshold_workers), str(args.start_method),
        )
        result["validation_selection"] = {
            "selected_epoch": selection["selected_epoch"],
            "threshold": selection["selected_validation_threshold"],
            "checkpoint": selection["checkpoint"],
            "metrics": selection["selected_validation_metrics"],
        }
    if args.action in ("evaluate", "all"):
        model_args = build_model_args(
            trigger_config, base_config, manifest, int(trigger_config["seed"]), bool(args.gpu)
        )
        evaluation = run_evaluation(
            state, output_dir, model_args, trigger_config["training"], manifest_path, manifest_sha
        )
        _write_manifest(
            state, output_dir, provenance, evaluation, trigger_config, model_args,
            manifest_path, manifest_sha,
        )
        result["test"] = {
            "event_metrics": evaluation["test_event_metrics"],
            "gate": evaluation["pre_registered_gate"],
            "outcome": evaluation["p6_c0_decision"]["outcome"],
        }
    print(json.dumps(to_builtin(result), sort_keys=True))


def _write_manifest(state, output_dir, provenance, evaluation, trigger_config, model_args,
                    manifest_path, manifest_sha):
    selection = json.loads((output_dir / "validation_selection.json").read_text(encoding="utf-8"))
    audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "generated_at_utc": utc_now(),
        "status": "COMPLETE",
        "formal_result": True,
        "experiment": "P6-C0 root-agnostic system event trigger",
        "git_commit": git_head(),
        "random_seed": int(trigger_config["seed"]),
        "config": provenance.get("trigger_config", {}),
        "base_config": provenance.get("base_config", {}),
        "source_artifacts": provenance.get("source_artifacts", {}),
        "frozen_protocol": provenance.get("frozen_protocol", {}),
        "split_mode": "chronological_50_20_30",
        "split_blocks": block_bounds(state.blocks),
        "split_manifest": {"path": str((output_dir / "split_manifest.json").resolve()),
                           "sha256": sha256_file(output_dir / "split_manifest.json")},
        "purged_windows": audit["windows"],
        "purged_window_label_counts": audit["purged_window_label_counts"],
        "purged_events": audit["purged_events"],
        "label_statistics": audit["labels"],
        "event_statistics": audit["events"],
        "windows_per_split": {split: int(len(state.sample_indices(split))) for split in SPLIT_NAMES},
        "checkpoint": selection["checkpoint"],
        "checkpoint_sha256": selection["checkpoint"]["sha256"],
        "selected_epoch": int(selection["selected_epoch"]),
        "selected_validation_threshold": float(selection["selected_validation_threshold"]),
        "selected_validation_metrics": selection["selected_validation_metrics"],
        "parameter_count": int(selection["parameter_count"]),
        "model_args": to_builtin(model_args),
        "training": dict(trigger_config["training"]),
        "pre_registered_gate": evaluation["pre_registered_gate"],
        "test_event_metrics": evaluation["test_event_metrics"],
        "policy_flags": {
            "no_retraining_of_ada_mgad": True,
            "ada_mgad_checkpoint_loaded": False,
            "node_anomaly_labels_read": False,
            "root_service_label_used": False,
            "rca_retrained": False,
            "conditional_logit_touched": False,
            "reintroduced_error_class": False,
            "test_used_for_checkpoint_selection": False,
            "test_used_for_threshold_selection": False,
            "test_used_for_model_or_gate_selection": False,
            "gate_values_changed_after_test": False,
            "writes_into_frozen_artifact_tree": False,
            "ada_mgad_checkpoint_dir_touched": False,
        },
        "data_manifest": {"path": str(Path(manifest_path).resolve()), "sha256": manifest_sha},
        "test_summary": evaluation,
    }
    write_json(output_dir / "manifest.json", to_builtin(manifest))


if __name__ == "__main__":
    main()
