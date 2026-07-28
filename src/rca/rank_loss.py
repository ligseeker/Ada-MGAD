"""Listwise ranking loss for single-root-cause supervision."""

import torch
import torch.nn.functional as F


def listwise_ce(scores, target, label_smoothing=0.0):
    """Softmax cross-entropy over the N node scores of each window.

    scores: [B, N] root-cause scores; target: [B] root-cause node index.
    Equivalent to a ListNet top-one objective.
    """
    return F.cross_entropy(scores, target, label_smoothing=label_smoothing)
