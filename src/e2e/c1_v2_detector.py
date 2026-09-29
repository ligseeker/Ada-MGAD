"""P6-C1-v2 fold detector: shared-array dataset, label-free Generation, C0 training.

The detector copies the frozen P6-C0 architecture and training rule. It never
loads a checkpoint, never sees a node-anomaly label, a root label or a fault
label, and its Generation pass is label-free and sealed before any GT join.

This is a fold-stage API used by ``scripts/p6/run_c1_v2.py``; it is not a formal
all-stage runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from .c1_v2_shared_data import (
    ARRAY_NAMES, PROTOCOL_ID, ROOT, RUN_ID, SEGMENTS, bound_path, load_protocol,
    load_segment, read_json, record, sha256, source_hashes, validate_c1_v2_fold,
    within, write_json_new,
)
from .event_detection import construct_predicted_episodes
from .protocol import GAIA_SERVICES, load_registry
from .system_trigger import (
    TRIGGER_IGNORE, TRIGGER_POSITIVE, assert_prediction_time_grid, build_trigger_labels,
    prediction_time_grid, select_system_threshold, system_score_frame, to_builtin,
    trigger_binary_mask,
)
from .system_trigger_data import build_trigger_loader
from .system_trigger_model import SystemEventTrigger
from util.util import seed_everything


MODEL_ARG_KEYS = (
    "feature_node", "feature_edge", "feature_log", "num_heads_edge",
    "num_heads_node", "num_heads_log", "num_heads_n2e", "num_heads_e2n",
    "num_layer", "dropout", "graph_hidden", "graph_sparse_weight",
    "graph_summary_mode",
)
STAGE_FILES = ("checkpoint.pt", "selection.json", "training_log.json",
               "generation_scores.csv", "generation_episodes.csv")
DETECTOR_MANIFEST_SCHEMA = "p6_c1_v2_fold_detector_v1"
GT_FORBIDDEN_COLUMNS = {"case_id", "trigger_label", "service", "fault_type",
                        "root_service", "gt_start_ms", "gt_end_ms", "match_status"}


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


class C1V2SegmentWindowDataset(Dataset):
    """Legal windows of one sealed shared-array fold segment.

    Fit/Selection receive caller-built trigger labels. Generation has no label
    parameter and no label access path in samples or metadata.
    """

    def __init__(self, fold_root: Path, segment: str, *, manifest: Mapping[str, object],
                 trigger_labels: Optional[np.ndarray] = None):
        if segment not in SEGMENTS:
            raise ValueError("P6-C1-v2 segment must be fit, selection or generation")
        if segment == "generation" and trigger_labels is not None:
            raise ValueError("Generation input must remain label-free")
        if segment != "generation" and trigger_labels is None:
            raise ValueError("Fit/Selection require explicit system trigger labels")
        self.fold_root = Path(fold_root).resolve()
        self.segment = segment
        self.manifest = manifest
        self.record = manifest["segments"][segment]
        arrays = load_segment(self.fold_root, segment, manifest=manifest)
        self.timestamps = arrays["timestamps"]
        self.metric = arrays["metric"]
        self.log = arrays["log"]
        self.trace = arrays["trace"]
        self.grid_ms = (int(self.record["interval_ms"][1]) - int(self.record["interval_ms"][0])) \
            // int(self.record["time_bins"])
        self.window_bins = int(self.record["window_bins"])
        self.sample_indices = np.arange(len(self.timestamps) - self.window_bins, dtype=np.int64)
        if (int(self.record["legal_windows"]) != len(self.sample_indices)
                or len(self.timestamps) != int(self.record["time_bins"])):
            raise ValueError("P6-C1-v2 legal-window geometry drift")
        if trigger_labels is None:
            self._labels = None
        else:
            labels = np.asarray(trigger_labels, dtype=np.int64).reshape(-1)
            if len(labels) != len(self.timestamps):
                raise ValueError("P6-C1-v2 trigger labels must cover the segment grid")
            self._labels = labels.copy()

    def __len__(self) -> int:
        return int(len(self.sample_indices))

    def _sample_index(self, position: int) -> int:
        if position < 0 or position >= len(self):
            raise IndexError(position)
        return int(self.sample_indices[position])

    def __getitem__(self, position: int) -> Mapping[str, np.ndarray]:
        index = self._sample_index(position)
        end = index + self.window_bins
        result = {
            "data_node": np.array(self.metric[index:end], dtype=np.float32),
            "data_log": np.array(self.log[index:end], dtype=np.float32),
            "data_edge": np.array(self.trace[index:end], dtype=np.float32),
            "sample_index": np.asarray(index, dtype=np.int64),
        }
        if self._labels is not None:
            result["trigger_label"] = np.asarray(self._labels[end - 1], dtype=np.int64)
        if set(result) & {"case_id", "root_service", "fault_type", "service"}:
            raise ValueError("P6-C1-v2 dataset sample breached the GT firewall")
        return result

    def prediction_times(self) -> np.ndarray:
        target = np.asarray(self.sample_indices) + self.window_bins - 1
        return np.asarray(self.timestamps[target], dtype=np.int64) + self.grid_ms

    def labels_at(self) -> np.ndarray:
        if self._labels is None:
            raise ValueError("Generation labels are unavailable before the OOS episode seal")
        target = np.asarray(self.sample_indices) + self.window_bins - 1
        return np.asarray(self._labels[target], dtype=np.int64)


# ---------------------------------------------------------------------------
# supervision (Fit/Selection only; Generation labels are never built)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class C1V2Supervision:
    fit_labels: np.ndarray
    selection_labels: np.ndarray
    selection_gt: pd.DataFrame
    audit: Mapping[str, object]


def load_c1_v2_registry(protocol: Mapping[str, object]) -> pd.DataFrame:
    base = read_json(bound_path(protocol["bindings"]["base_config"]))
    registry_path = bound_path(protocol["bindings"]["gt_registry"])
    if sha256(registry_path) != base["event_registry"]["sha256"]:
        raise ValueError("P6-C1-v2 GT registry differs from the base-config binding")
    return load_registry(base, ROOT)


def build_c1_v2_supervision(*, protocol: Mapping[str, object],
                            manifest: Mapping[str, object],
                            registry: pd.DataFrame) -> C1V2Supervision:
    if "detector_domain" not in registry.columns:
        raise ValueError("P6-C1-v2 registry lacks detector_domain")
    legal = registry.loc[registry["detector_domain"].astype(bool)].copy()
    labels = {}
    counts = {}
    for name in ("fit", "selection"):
        start, end = manifest["segments"][name]["interval_ms"]
        timestamps = np.load(
            Path(manifest["fold_root"]) / manifest["segments"][name]["files"]["timestamps"]["path"],
            mmap_mode="r", allow_pickle=False)
        prediction_times = prediction_time_grid(timestamps, grid_seconds=30)
        assert_prediction_time_grid(prediction_times, timestamps, grid_seconds=30)
        overlapping = legal.loc[(legal["start_ms"] < end) & (legal["end_ms"] > start)]
        labels[name] = build_trigger_labels(
            prediction_times, overlapping[["start_ms", "end_ms"]],
            positive_window_seconds=60)
        counts[name] = {"label_source_events": int(len(overlapping)),
                        "positive_bins": int(np.sum(labels[name] == TRIGGER_POSITIVE)),
                        "ignore_bins": int(np.sum(labels[name] == TRIGGER_IGNORE))}
    selection_start, selection_end = manifest["segments"]["selection"]["interval_ms"]
    selection_gt = legal.loc[(legal["start_ms"] >= selection_start)
                             & (legal["end_ms"] <= selection_end)].copy()
    selection_gt = selection_gt.sort_values(
        ["start_ms", "source_index", "case_id"], kind="stable").reset_index(drop=True)
    if selection_gt.empty:
        raise ValueError("P6-C1-v2 Selection has no complete GT event")
    crossing = legal.loc[(legal["start_ms"] < selection_end)
                         & (legal["end_ms"] > selection_start)
                         & ~((legal["start_ms"] >= selection_start)
                             & (legal["end_ms"] <= selection_end))]
    audit = {"fold": int(manifest["fold"]), "fit": counts["fit"], "selection": counts["selection"],
             "selection_complete_gt": int(len(selection_gt)),
             "selection_cross_boundary_excluded_from_metrics": int(len(crossing)),
             "generation_labels_built": False}
    return C1V2Supervision(labels["fit"], labels["selection"], selection_gt, audit)


# ---------------------------------------------------------------------------
# label-free inference
# ---------------------------------------------------------------------------


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


@torch.no_grad()
def score_c1_v2_segment(model, dataset: C1V2SegmentWindowDataset, *, batch_size: int,
                        num_workers: int, device: torch.device) -> pd.DataFrame:
    """Return label-free scores; a Generation batch carrying a label fails closed."""
    if not isinstance(dataset, C1V2SegmentWindowDataset) \
            or dataset.segment not in ("selection", "generation"):
        raise ValueError("P6-C1-v2 scorer requires a Selection or Generation dataset")
    loader = build_trigger_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    indices, logits_parts = [], []
    for batch in loader:
        if dataset.segment == "generation" and "trigger_label" in batch:
            raise ValueError("P6-C1-v2 Generation batch contains a label")
        if any("label" in key and key != "trigger_label" for key in batch):
            raise ValueError("P6-C1-v2 scoring batch contains an unexpected label")
        batch.pop("trigger_label", None)
        moved = {}
        for key, value in batch.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if tensor.is_floating_point():
                if not torch.isfinite(tensor).all():
                    raise ValueError("P6-C1-v2 inference tensor contains non-finite values")
                tensor = tensor.to(device=device, dtype=torch.float32)
            else:
                tensor = tensor.to(device=device)
            moved[key] = tensor
        logits, _ = model(moved)
        values = logits.detach().cpu().numpy().reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("P6-C1-v2 model emitted non-finite logits")
        indices.append(moved["sample_index"].detach().cpu().numpy().reshape(-1))
        logits_parts.append(values)
    if not indices:
        raise ValueError("P6-C1-v2 loader yielded no samples")
    sample_indices = np.concatenate(indices).astype(np.int64)
    logits = np.concatenate(logits_parts).astype(np.float64)
    order = np.argsort(sample_indices, kind="stable")
    ordered_indices = sample_indices[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = ordered_indices[1:] != ordered_indices[:-1]
    ordered_indices = ordered_indices[keep]
    logits = logits[order][keep]
    if not np.array_equal(ordered_indices, dataset.sample_indices):
        raise ValueError("P6-C1-v2 inference did not cover every legal window")
    times = dataset.prediction_times()
    if len(times) != len(logits) or np.any(np.diff(times) <= 0):
        raise ValueError("P6-C1-v2 prediction times are not strictly ordered")
    frame = system_score_frame(dataset.segment, times, _sigmoid(logits))
    frame["sample_index"] = ordered_indices
    frame["system_trigger_logit"] = logits
    if any("label" in column or "root" in column for column in frame.columns):
        raise ValueError("P6-C1-v2 score frame breached the label firewall")
    return frame


def score_c1_v2_selection(model, dataset: C1V2SegmentWindowDataset, *, batch_size: int,
                          num_workers: int, device: torch.device) -> pd.DataFrame:
    if dataset.segment != "selection":
        raise ValueError("P6-C1-v2 Selection scorer requires Selection windows")
    return score_c1_v2_segment(model, dataset, batch_size=batch_size,
                               num_workers=num_workers, device=device)


def score_c1_v2_generation(model, dataset: C1V2SegmentWindowDataset, *, batch_size: int,
                           num_workers: int, device: torch.device,
                           threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every Generation window without accepting or returning GT labels."""
    if dataset.segment != "generation":
        raise ValueError("P6-C1-v2 Generation scorer requires the label-free dataset")
    if not np.isfinite(threshold):
        raise ValueError("P6-C1-v2 selected threshold must be finite")
    frame = score_c1_v2_segment(model, dataset, batch_size=batch_size,
                                num_workers=num_workers, device=device)
    frame["binary_prediction"] = (frame["system_score"].to_numpy() >= float(threshold)).astype(np.int8)
    episodes = construct_predicted_episodes(frame, float(threshold), grid_seconds=30)
    return frame, episodes


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _model_args(base: Mapping[str, object], c0: Mapping[str, object],
                manifest: Mapping[str, object], *, gpu: bool) -> Mapping[str, object]:
    args = {key: base["ad_model"][key] for key in MODEL_ARG_KEYS}
    args.update({
        "main_model": "P6-C0-SystemEventTrigger", "random_seed": int(c0["seed"]),
        "gpu": bool(gpu), "window": int(manifest["segments"]["fit"]["window_bins"]),
        "step": int(base["ad"]["window_step"]),
        "num_nodes": len(manifest["services"]),
        "batch_size": int(c0["training"]["batch_size"]),
        "head_hidden": int(c0["model"].get("head_hidden", 32)),
    })
    args.update({key: int(manifest["dimensions"][key]) for key in ("raw_node", "log_len", "raw_edge")})
    return args


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> Mapping[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.is_floating_point():
            if not torch.isfinite(tensor).all():
                raise ValueError("P6-C1-v2 detector input contains a non-finite value")
            tensor = tensor.to(device=device, dtype=torch.float32)
        else:
            tensor = tensor.to(device=device)
        moved[key] = tensor
    return moved


def _selection_key(metrics: Mapping[str, object], threshold: float):
    return (float(metrics["event_f1"]), float(metrics["event_recall"]), float(threshold))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def train_c1_v2_fold_detector(*, fold_root: Path, gpu: bool = False,
                              threshold_workers: int = 8,
                              epochs: Optional[int] = None,
                              patience: Optional[int] = None,
                              protocol: Optional[Mapping[str, object]] = None) -> Mapping[str, object]:
    """Run the frozen C0 training rule once on one sealed shared-array fold.

    ``epochs``/``patience`` exist only for the bounded smoke path and are refused
    inside the formal run root.
    """
    protocol = load_protocol() if protocol is None else protocol
    fold_root = Path(fold_root).resolve()
    if (epochs is not None or patience is not None) and within(
            fold_root, ROOT / "experiments/p6/c1_supervision_oos"):
        raise ValueError("epoch overrides are forbidden inside the formal P6-C1-v2 run root")
    manifest = validate_c1_v2_fold(fold_root, protocol=protocol)
    fold_manifest_sha256 = sha256(fold_root / "completion_manifest.json")
    base = read_json(bound_path(protocol["bindings"]["base_config"]))
    c0 = read_json(bound_path(protocol["bindings"]["c0_trigger_config"]))
    training = c0["training"]
    budget = protocol["resource_budget"]
    if (int(c0["seed"]) != int(protocol["seed"])
            or int(training["max_epochs"]) != int(protocol["detector"]["expected_frozen_values"]["max_epochs"])
            or int(training["patience"]) != int(protocol["detector"]["expected_frozen_values"]["patience"])
            or int(training["num_workers"]) != int(budget["detector_loader_workers"])
            or not 1 <= int(threshold_workers) <= int(budget["threshold_workers"])):
        raise ValueError("P6-C1-v2 detector training differs from the frozen C0 budget")
    if gpu and not torch.cuda.is_available():
        raise ValueError("P6-C1-v2 CUDA was requested but is unavailable")
    fit_epochs = int(training["max_epochs"]) if epochs is None else int(epochs)
    fit_patience = int(training["patience"]) if patience is None else int(patience)
    if not 1 <= fit_epochs <= int(budget["max_epochs_per_fold"]) or fit_patience < 0:
        raise ValueError("P6-C1-v2 epoch budget differs from the frozen budget")
    source_hashes_before = source_hashes()

    supervision = build_c1_v2_supervision(protocol=protocol, manifest=manifest,
                                          registry=load_c1_v2_registry(protocol))
    fit = C1V2SegmentWindowDataset(fold_root, "fit", manifest=manifest,
                                   trigger_labels=supervision.fit_labels)
    selection = C1V2SegmentWindowDataset(fold_root, "selection", manifest=manifest,
                                         trigger_labels=supervision.selection_labels)
    generation = C1V2SegmentWindowDataset(fold_root, "generation", manifest=manifest)
    fit_labels = fit.labels_at()
    target, mask = trigger_binary_mask(fit_labels)
    positives = float(target.sum())
    negatives = float(((1.0 - target) * mask).sum())
    if positives <= 0 or negatives <= 0:
        raise ValueError("P6-C1-v2 Fit must contain positive and negative trigger windows")

    stage = fold_root / "detector"
    stage.mkdir(exist_ok=False)
    write_json_new(stage / "INCOMPLETE.json", {"status": "INCOMPLETE", "fold": manifest["fold"]})
    try:
        torch.set_num_threads(int(budget["torch_threads"]))
        seed_everything(int(protocol["seed"]))
        device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        args = _model_args(base, c0, manifest, gpu=gpu)
        graph = np.load(fold_root / manifest["graph"]["path"], allow_pickle=False)
        model = SystemEventTrigger(graph, **args).to(device)
        from adabelief_pytorch import AdaBelief
        optimizer = AdaBelief(model.parameters(), lr=float(training["learning_rate"]),
                              weight_decay=float(training["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, int(training["scheduler_step"]), float(training["scheduler_gamma"]))
        criterion = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device))
        fit_loader = build_trigger_loader(fit, batch_size=int(training["batch_size"]),
                                          num_workers=int(training["num_workers"]))
        history = []
        best = None
        stale = 0
        stop_reason = "max_epochs"
        for epoch in range(fit_epochs):
            model.train()
            loss_sum = 0.0
            usable = 0
            for batch in fit_loader:
                moved = _move(batch, device)
                logits, graph_reg = model(moved)
                labels = moved["trigger_label"]
                loss_mask = (labels != TRIGGER_IGNORE).to(dtype=torch.float32)
                if not bool(loss_mask.any()):
                    continue
                loss_target = (labels == TRIGGER_POSITIVE).to(dtype=torch.float32)
                bce = (criterion(logits, loss_target) * loss_mask).sum() / loss_mask.sum()
                loss = bce + graph_reg
                if not torch.isfinite(loss):
                    raise FloatingPointError("P6-C1-v2 non-finite Fit loss at epoch {}".format(epoch))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),
                                         max_norm=float(training["grad_clip_norm"]), norm_type=2)
                optimizer.step()
                loss_sum += float(loss.item())
                usable += 1
            if not usable:
                raise ValueError("P6-C1-v2 Fit yielded no usable batches")
            scheduler.step()
            selected_scores = score_c1_v2_selection(
                model, selection, batch_size=int(training["batch_size"]),
                num_workers=int(training["num_workers"]), device=device)
            threshold_choice = select_system_threshold(
                selected_scores, supervision.selection_gt, grid_seconds=30,
                tolerance_seconds=int(protocol["detector"]["matching_tolerance_seconds"]),
                workers=int(threshold_workers), start_method="spawn")
            key = _selection_key(threshold_choice.metrics, threshold_choice.threshold)
            history.append({"epoch": epoch, "train_loss": loss_sum / usable,
                            "selection_threshold": float(threshold_choice.threshold),
                            "selection_candidate_count": int(threshold_choice.candidate_count),
                            "selection_metrics": to_builtin(threshold_choice.metrics),
                            "selection_key": list(key)})
            if best is None or key > best["key"]:
                best = {"key": key, "epoch": epoch,
                        "threshold": float(threshold_choice.threshold),
                        "metrics": to_builtin(threshold_choice.metrics),
                        "state": {name: value.detach().cpu().clone()
                                  for name, value in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
            if fit_patience > 0 and stale >= fit_patience:
                stop_reason = "selection_event_f1_patience"
                break
        if best is None:
            raise RuntimeError("P6-C1-v2 Fit produced no selected checkpoint")
        model.load_state_dict(best["state"])
        with (stage / "checkpoint.pt").open("xb") as stream:
            torch.save(best["state"], stream)
        scores, episodes = score_c1_v2_generation(
            model, generation, batch_size=int(training["batch_size"]),
            num_workers=int(training["num_workers"]), device=device,
            threshold=best["threshold"])
        scores.insert(0, "fold", int(manifest["fold"]))
        episodes.insert(0, "fold", int(manifest["fold"]))
        episodes["prediction_id"] = episodes["prediction_id"].map(
            lambda value: "fold_{:02d}-{}".format(int(manifest["fold"]), value))
        if set(scores.columns) & GT_FORBIDDEN_COLUMNS:
            raise ValueError("P6-C1-v2 Generation score schema contains GT")
        if set(episodes.columns) & GT_FORBIDDEN_COLUMNS:
            raise ValueError("P6-C1-v2 Generation episode schema contains GT")
        with (stage / "generation_scores.csv").open("x", encoding="utf-8", newline="") as stream:
            scores.to_csv(stream, index=False, lineterminator="\n")
        with (stage / "generation_episodes.csv").open("x", encoding="utf-8", newline="") as stream:
            episodes.to_csv(stream, index=False, lineterminator="\n")
        write_json_new(stage / "selection.json", {
            "selected_epoch": best["epoch"], "threshold": best["threshold"],
            "selection_metrics": best["metrics"],
            "tie_break": "event_f1_then_recall_then_threshold",
            "selection_gt_events": len(supervision.selection_gt),
            "test_used": False})
        write_json_new(stage / "training_log.json", {
            "history": history, "stop_reason": stop_reason,
            "fit_windows": len(fit), "selection_windows": len(selection),
            "generation_windows": len(generation), "fit_positive_windows": int(positives),
            "fit_negative_windows": int(negatives),
            "fit_ignore_windows": int((fit_labels == TRIGGER_IGNORE).sum()),
            "fit_pos_weight": negatives / positives,
            "supervision_audit": to_builtin(supervision.audit),
            "model_args": to_builtin(args), "training": to_builtin(training),
            "smoke_epoch_override": None if epochs is None else int(epochs)})
        if source_hashes() != source_hashes_before:
            raise ValueError("P6-C1-v2 detector source changed during Fit/Generation")
        if sha256(fold_root / "completion_manifest.json") != fold_manifest_sha256:
            raise ValueError("P6-C1-v2 fold manifest changed during detector Fit")
        detector_manifest = {
            "schema_version": DETECTOR_MANIFEST_SCHEMA,
            "status": "COMPLETE",
            "protocol_id": PROTOCOL_ID,
            "run_id": RUN_ID,
            "fold": int(manifest["fold"]),
            "created_utc": _now(),
            "fold_manifest_sha256": fold_manifest_sha256,
            "source_sha256": source_hashes_before,
            "generation_label_free": True,
            "generation_labels_built": False,
            "checkpoint_loaded": False,
            "per_fold_random_initialization": True,
            "selected_epoch": int(best["epoch"]),
            "threshold": float(best["threshold"]),
            "selection_metrics": best["metrics"],
            "generation_scores": int(len(scores)),
            "generation_episodes": int(len(episodes)),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_threads": int(budget["torch_threads"]),
                "dataloader_workers": int(training["num_workers"]),
                "device": str(device),
                "epochs": int(fit_epochs),
                "patience": int(fit_patience),
            },
            "files": {name: {"sha256": sha256(stage / name),
                             "bytes": (stage / name).stat().st_size} for name in STAGE_FILES},
        }
        write_json_new(stage / "completion_manifest.json", detector_manifest)
        (stage / "INCOMPLETE.json").unlink()
        return detector_manifest
    except Exception as exc:
        try:
            write_json_new(stage / "failure.json",
                           {"status": "INCOMPLETE", "error_type": type(exc).__name__,
                            "message": str(exc)})
        except OSError:
            pass
        raise


def validate_c1_v2_fold_detector(*, fold_root: Path,
                                 protocol: Optional[Mapping[str, object]] = None) -> Mapping[str, object]:
    """Validate the sealed label-free Generation outputs before GT matching."""
    protocol = load_protocol() if protocol is None else protocol
    fold_root = Path(fold_root).resolve()
    manifest = validate_c1_v2_fold(fold_root, protocol=protocol)
    stage = fold_root / "detector"
    if (stage / "INCOMPLETE.json").exists():
        raise ValueError("P6-C1-v2 detector stage incomplete")
    detector = read_json(stage / "completion_manifest.json")
    if (detector.get("schema_version") != DETECTOR_MANIFEST_SCHEMA
            or detector.get("status") != "COMPLETE"
            or int(detector.get("fold", -1)) != int(manifest["fold"])
            or detector.get("fold_manifest_sha256") != sha256(fold_root / "completion_manifest.json")
            or detector.get("source_sha256") != source_hashes()
            or set(detector.get("files", {})) != set(STAGE_FILES)):
        raise ValueError("P6-C1-v2 detector completion identity/source drift")
    for name, entry in detector["files"].items():
        path = stage / name
        if not path.is_file() or path.stat().st_size != entry["bytes"] \
                or sha256(path) != entry["sha256"]:
            raise ValueError("P6-C1-v2 detector output drift: {}".format(name))
    episodes = pd.read_csv(stage / "generation_episodes.csv")
    if set(episodes.columns) & GT_FORBIDDEN_COLUMNS:
        raise ValueError("P6-C1-v2 sealed Generation episodes contain GT fields")
    if len(episodes) != int(detector["generation_episodes"]):
        raise ValueError("P6-C1-v2 sealed Generation episode count drift")
    return detector
