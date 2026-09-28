"""Label-free C1 Generation inference and OOS episode construction."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .c1_fold_detector_data import C1SegmentWindowDataset
from .event_detection import construct_predicted_episodes
from .system_trigger import system_score_frame
from .system_trigger_data import build_trigger_loader


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


@torch.no_grad()
def _score_c1_segment(model, dataset: C1SegmentWindowDataset, *, batch_size: int,
                      num_workers: int, device: torch.device) -> pd.DataFrame:
    """Return label-free scores; Selection labels are removed before inference."""
    if not isinstance(dataset, C1SegmentWindowDataset) or dataset.segment not in ("selection", "generation"):
        raise ValueError("C1 scorer requires a Selection or Generation dataset")
    loader = build_trigger_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    indices, logits_parts = [], []
    for batch in loader:
        if dataset.segment == "generation" and "trigger_label" in batch:
            raise ValueError("Generation batch contains a label")
        if any("label" in key and key != "trigger_label" for key in batch):
            raise ValueError("C1 scoring batch contains an unexpected label")
        batch.pop("trigger_label", None)
        moved = {}
        for key, value in batch.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if tensor.is_floating_point():
                if not torch.isfinite(tensor).all():
                    raise ValueError("Generation tensor contains non-finite values")
                tensor = tensor.to(device=device, dtype=torch.float32)
            else:
                tensor = tensor.to(device=device)
            moved[key] = tensor
        logits, _ = model(moved)
        values = logits.detach().cpu().numpy().reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Generation model emitted non-finite logits")
        indices.append(moved["sample_index"].detach().cpu().numpy().reshape(-1))
        logits_parts.append(values)
    if not indices:
        raise ValueError("Generation loader yielded no samples")
    sample_indices = np.concatenate(indices).astype(np.int64)
    logits = np.concatenate(logits_parts).astype(np.float64)
    order = np.argsort(sample_indices, kind="stable")
    ordered_indices = sample_indices[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = ordered_indices[1:] != ordered_indices[:-1]
    ordered_indices = ordered_indices[keep]
    logits = logits[order][keep]
    if not np.array_equal(ordered_indices, dataset.sample_indices):
        raise ValueError("C1 inference did not cover every legal window")
    times = dataset.prediction_times()
    if len(times) != len(logits) or np.any(np.diff(times) <= 0):
        raise ValueError("C1 prediction times are not strictly ordered")
    scores = _sigmoid(logits)
    frame = system_score_frame(dataset.segment, times, scores)
    frame["sample_index"] = ordered_indices
    frame["system_trigger_logit"] = logits
    if any("label" in column or "root" in column for column in frame.columns):
        raise ValueError("C1 score frame breached the label firewall")
    return frame


def score_c1_selection(model, dataset: C1SegmentWindowDataset, *, batch_size: int,
                       num_workers: int, device: torch.device) -> pd.DataFrame:
    if dataset.segment != "selection":
        raise ValueError("C1 Selection scorer requires Selection windows")
    return _score_c1_segment(model, dataset, batch_size=batch_size,
                             num_workers=num_workers, device=device)


def score_c1_generation(model, dataset: C1SegmentWindowDataset, *, batch_size: int,
                        num_workers: int, device: torch.device,
                        threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every Generation window without accepting or returning GT labels."""
    if dataset.segment != "generation":
        raise ValueError("C1 Generation scorer requires the label-free Generation dataset")
    if not np.isfinite(threshold):
        raise ValueError("C1 selected threshold must be finite")
    frame = _score_c1_segment(model, dataset, batch_size=batch_size,
                              num_workers=num_workers, device=device)
    frame["binary_prediction"] = (frame["system_score"].to_numpy() >= float(threshold)).astype(np.int8)
    episodes = construct_predicted_episodes(frame, float(threshold), grid_seconds=30)
    return frame, episodes
