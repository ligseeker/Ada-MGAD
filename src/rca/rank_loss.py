"""Listwise ranking loss for single-root-cause supervision."""

import torch
import torch.nn.functional as F


def listwise_ce(scores, target, label_smoothing=0.0, sample_weight=None):
    """Softmax cross-entropy over the N node scores of each window.

    scores: [B, N] root-cause scores; target: [B] root-cause node index.
    sample_weight: optional [B] per-window weights (e.g. type balancing).
    Equivalent to a ListNet top-one objective.
    """
    loss = F.cross_entropy(scores, target, label_smoothing=label_smoothing,
                           reduction='none' if sample_weight is not None else 'mean')
    if sample_weight is not None:
        loss = (loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
    return loss
