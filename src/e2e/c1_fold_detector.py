"""One prefix-fitted C1 detector: Fit, Selection, then label-free Generation.

This is a fold-stage API, not a formal all-stage runner. A failed attempt keeps
its exclusive detector directory and cannot be resumed or silently replaced.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from .c1_fold_detector_data import C1SegmentWindowDataset
from .c1_fold_generation import score_c1_generation, score_c1_selection
from .c1_fold_preprocessing import ROOT, _bound_path, _sha256, _write_json_new, validate_c1_fold
from .c1_fold_supervision import build_c1_fit_selection_supervision
from .system_trigger import TRIGGER_IGNORE, TRIGGER_POSITIVE, select_system_threshold, to_builtin, trigger_binary_mask
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_args(base: Mapping[str, object], c0: Mapping[str, object],
                fold: Mapping[str, object], *, gpu: bool) -> Mapping[str, object]:
    args = {key: base["ad_model"][key] for key in MODEL_ARG_KEYS}
    args.update({
        "main_model": "P6-C0-SystemEventTrigger", "random_seed": int(c0["seed"]),
        "gpu": bool(gpu), "window": 10, "step": int(base["ad"]["window_step"]),
        "num_nodes": len(fold["services"]), "batch_size": int(c0["training"]["batch_size"]),
        "head_hidden": int(c0["model"].get("head_hidden", 32)),
    })
    args.update({key: int(fold["dimensions"][key]) for key in ("raw_node", "log_len", "raw_edge")})
    return args


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> Mapping[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.is_floating_point():
            if not torch.isfinite(tensor).all():
                raise ValueError("C1 detector input contains a non-finite value")
            tensor = tensor.to(device=device, dtype=torch.float32)
        else:
            tensor = tensor.to(device=device)
        moved[key] = tensor
    return moved


def _selection_key(metrics: Mapping[str, object], threshold: float):
    return (float(metrics["event_f1"]), float(metrics["event_recall"]), float(threshold))


def _sources(protocol: Mapping[str, object], protocol_path: Path) -> Mapping[str, str]:
    paths = {
        "c1_detector": Path(__file__),
        "c1_generation": ROOT / "src/e2e/c1_fold_generation.py",
        "c1_supervision": ROOT / "src/e2e/c1_fold_supervision.py",
        "c1_dataset": ROOT / "src/e2e/c1_fold_detector_data.py",
        "c0_model": ROOT / "src/e2e/system_trigger_model.py",
        "c0_trigger": ROOT / "src/e2e/system_trigger.py",
        "matching": ROOT / "src/e2e/event_detection.py",
        "g2_protocol": protocol_path,
        "c0_config": _bound_path(protocol["bindings"]["c0_trigger_config"]),
    }
    return {key: _sha256(path) for key, path in paths.items()}


def train_c1_fold_detector(*, fold_root: Path, protocol_path: Path,
                           gpu: bool = False, threshold_workers: int = 8) -> Mapping[str, object]:
    """Run the frozen C0 training rule once on one sealed C1 fold.

    The only GT read is Fit/Selection supervision. Generation scoring takes a
    label-free dataset and writes scores and episodes before any GT join.
    """
    fold_root = Path(fold_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    fold = validate_c1_fold(fold_root, protocol_path=protocol_path)
    fold_manifest_sha256 = _sha256(fold_root / "completion_manifest.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base = json.loads(_bound_path(protocol["bindings"]["base_config"]).read_text(encoding="utf-8"))
    c0 = json.loads(_bound_path(protocol["bindings"]["c0_trigger_config"]).read_text(encoding="utf-8"))
    training = c0["training"]
    budget = protocol["resource_budget"]
    if (int(c0["seed"]) != int(protocol["seed"]) or int(training["max_epochs"]) != int(protocol["detector"]["max_epochs"])
            or int(training["patience"]) != int(protocol["detector"]["patience"])
            or int(training["num_workers"]) != int(budget["detector_loader_workers"])
            or int(training["max_epochs"]) > int(budget["max_epochs_per_fold"])
            or int(threshold_workers) < 1):
        raise ValueError("C1 detector training differs from G2/C0 frozen budget")
    if gpu and not torch.cuda.is_available():
        raise ValueError("C1 CUDA was requested but is unavailable")
    source_hashes = _sources(protocol, protocol_path)
    supervision = build_c1_fit_selection_supervision(fold_root, protocol_path=protocol_path)
    fit = C1SegmentWindowDataset(fold_root, "fit", protocol_path=protocol_path,
                                 trigger_labels=supervision.fit_labels)
    selection = C1SegmentWindowDataset(fold_root, "selection", protocol_path=protocol_path,
                                       trigger_labels=supervision.selection_labels)
    generation = C1SegmentWindowDataset(fold_root, "generation", protocol_path=protocol_path)
    fit_labels = fit.labels_at()
    target, mask = trigger_binary_mask(fit_labels)
    positives = float(target.sum())
    negatives = float(((1.0 - target) * mask).sum())
    if positives <= 0 or negatives <= 0:
        raise ValueError("C1 Fit must contain positive and negative trigger windows")
    stage = fold_root / "detector"
    stage.mkdir(exist_ok=False)
    _write_json_new(stage / "INCOMPLETE.json", {"status": "INCOMPLETE", "fold": fold["fold"]})
    try:
        torch.set_num_threads(int(budget["torch_threads"]))
        seed_everything(int(protocol["seed"]))
        device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        args = _model_args(base, c0, fold, gpu=gpu)
        graph = np.load(fold_root / fold["graph"]["path"], allow_pickle=False)
        model = SystemEventTrigger(graph, **args).to(device)
        from adabelief_pytorch import AdaBelief
        optimizer = AdaBelief(model.parameters(), lr=float(training["learning_rate"]),
                              weight_decay=float(training["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, int(training["scheduler_step"]), float(training["scheduler_gamma"]))
        criterion = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.tensor([negatives / positives],
                                                       dtype=torch.float32, device=device))
        fit_loader = build_trigger_loader(
            fit, batch_size=int(training["batch_size"]), num_workers=int(training["num_workers"]))
        history = []
        best = None
        stale = 0
        stop_reason = "max_epochs"
        for epoch in range(int(training["max_epochs"])):
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
                    raise FloatingPointError("C1 non-finite Fit loss at epoch {}".format(epoch))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(training["grad_clip_norm"]), norm_type=2)
                optimizer.step()
                loss_sum += float(loss.item())
                usable += 1
            if not usable:
                raise ValueError("C1 Fit yielded no usable batches")
            scheduler.step()
            selected_scores = score_c1_selection(
                model, selection, batch_size=int(training["batch_size"]),
                num_workers=int(training["num_workers"]), device=device)
            threshold_choice = select_system_threshold(
                selected_scores, supervision.selection_gt, grid_seconds=30,
                tolerance_seconds=60, workers=int(threshold_workers), start_method="spawn")
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
            if int(training["patience"]) > 0 and stale >= int(training["patience"]):
                stop_reason = "selection_event_f1_patience"
                break
        if best is None:
            raise RuntimeError("C1 Fit produced no selected checkpoint")
        model.load_state_dict(best["state"])
        with (stage / "checkpoint.pt").open("xb") as stream:
            torch.save(best["state"], stream)
        scores, episodes = score_c1_generation(
            model, generation, batch_size=int(training["batch_size"]),
            num_workers=int(training["num_workers"]), device=device,
            threshold=best["threshold"])
        scores.insert(0, "fold", int(fold["fold"]))
        episodes.insert(0, "fold", int(fold["fold"]))
        episodes["prediction_id"] = episodes["prediction_id"].map(
            lambda value: "fold_{:02d}-{}".format(int(fold["fold"]), value))
        if set(scores.columns) & {"case_id", "trigger_label", "service", "fault_type", "root_service"}:
            raise ValueError("C1 Generation score schema contains GT")
        if set(episodes.columns) & {"case_id", "trigger_label", "service", "fault_type", "root_service"}:
            raise ValueError("C1 Generation episode schema contains GT")
        with (stage / "generation_scores.csv").open("x", encoding="utf-8", newline="") as stream:
            scores.to_csv(stream, index=False, lineterminator="\n")
        with (stage / "generation_episodes.csv").open("x", encoding="utf-8", newline="") as stream:
            episodes.to_csv(stream, index=False, lineterminator="\n")
        _write_json_new(stage / "selection.json", {
            "selected_epoch": best["epoch"], "threshold": best["threshold"],
            "selection_metrics": best["metrics"],
            "tie_break": "event_f1_then_recall_then_threshold",
            "selection_gt_events": len(supervision.selection_gt),
            "test_used": False})
        _write_json_new(stage / "training_log.json", {
            "history": history, "stop_reason": stop_reason,
            "fit_windows": len(fit), "selection_windows": len(selection),
            "generation_windows": len(generation), "fit_positive_windows": int(positives),
            "fit_negative_windows": int(negatives), "fit_ignore_windows": int((fit_labels == TRIGGER_IGNORE).sum()),
            "fit_pos_weight": negatives / positives,
            "supervision_audit": to_builtin(supervision.audit),
            "model_args": to_builtin(args), "training": to_builtin(training)})
        if _sources(protocol, protocol_path) != source_hashes:
            raise ValueError("C1 detector source changed during Fit/Generation")
        if _sha256(fold_root / "completion_manifest.json") != fold_manifest_sha256:
            raise ValueError("C1 fold manifest changed during detector Fit")
        if validate_c1_fold(fold_root, protocol_path=protocol_path)["fold"] != fold["fold"]:
            raise ValueError("C1 fold inputs changed during detector Fit")
        manifest = {"schema_version": "p6_c1_fold_detector_v1", "status": "COMPLETE",
                    "fold": fold["fold"], "created_utc": _now(),
                    "fold_manifest_sha256": fold_manifest_sha256,
                    "source_sha256": source_hashes,
                    "generation_scores": len(scores), "generation_episodes": len(episodes),
                    "files": {name: {"sha256": _sha256(stage / name),
                                     "bytes": (stage / name).stat().st_size} for name in STAGE_FILES}}
        _write_json_new(stage / "completion_manifest.json", manifest)
        (stage / "INCOMPLETE.json").unlink()
        return manifest
    except Exception as exc:
        try:
            _write_json_new(stage / "failure.json", {"status": "INCOMPLETE",
                                                     "error_type": type(exc).__name__,
                                                     "message": str(exc)})
        except OSError:
            pass
        raise


def validate_c1_fold_detector(*, fold_root: Path, protocol_path: Path) -> Mapping[str, object]:
    """Validate the sealed label-free Generation outputs before GT matching."""
    fold_root = Path(fold_root).resolve()
    fold = validate_c1_fold(fold_root, protocol_path=protocol_path)
    stage = fold_root / "detector"
    if (stage / "INCOMPLETE.json").exists():
        raise ValueError("C1 detector stage incomplete")
    manifest = json.loads((stage / "completion_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if (manifest.get("status") != "COMPLETE" or manifest.get("fold") != fold["fold"]
            or manifest.get("fold_manifest_sha256") != _sha256(fold_root / "completion_manifest.json")
            or manifest.get("source_sha256") != _sources(protocol, Path(protocol_path).resolve())
            or set(manifest.get("files", {})) != set(STAGE_FILES)):
        raise ValueError("C1 detector completion identity/source drift")
    for name, record in manifest["files"].items():
        path = stage / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError("C1 detector output drift: {}".format(name))
    return manifest
