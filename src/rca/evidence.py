"""Three-evidence feature extraction for root-cause localization.

Extracts a per-(window, node) feature vector from a frozen Ada-MGAD
detector and the directed GAIA call graph:

  anomaly evidence (i)
    a_cls      classification probability of being anomalous (last step)
    a_rec      MAD-normalized reconstruction energy, squashed to (0, 1)
    a_fused    score-fusion of the two (train.py rule)

  propagation evidence (ii)  — directed call graph modulated per batch by
                               the detector's learned dynamic edge weights
    explained    anomaly of n's callees (n's symptoms may be inherited)
    unexplained  a_fused - explained (anomaly not explained by dependencies)
    blame        anomaly of n's callers (n's fault would explain their symptoms)

  temporal evidence (iii) — per-timestep reconstruction energy within the
                            10-step window
    com          center of mass of the anomaly series (earlier = smaller)
    lead         com minus the window's earliest com (relative onset delay)
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from util.GAIA.constant import GAIA_SERVICES

FEATURE_GROUPS = {
    'anomaly': ['a_cls', 'a_rec', 'a_fused'],
    'propagation': ['explained', 'unexplained', 'blame'],
    'temporal': ['com', 'lead'],
}
FEATURE_NAMES = [name for names in FEATURE_GROUPS.values() for name in names]


def feature_indices(groups):
    """Column indices of FEATURE_NAMES covered by the given group names."""
    wanted = [name for g in groups for name in FEATURE_GROUPS[g]]
    return [FEATURE_NAMES.index(name) for name in wanted]


def build_call_graph(data_path):
    """Directed caller -> callee weight matrix from trace.csv.

    W[u, v] = share of service u's calls that go to v (row-normalized).
    Anomalies propagate along the reverse direction (callee -> caller).
    """
    trace_file = os.path.join(data_path, 'trace.csv')
    df = pd.read_csv(trace_file, usecols=['src_service', 'dst_service', 'count'])
    index = {s: i for i, s in enumerate(GAIA_SERVICES)}
    W = np.zeros((len(GAIA_SERVICES), len(GAIA_SERVICES)), dtype=np.float64)
    for row in df.itertuples(index=False):
        if row.src_service in index and row.dst_service in index:
            W[index[row.src_service], index[row.dst_service]] += row.count
    row_sum = W.sum(axis=1, keepdims=True)
    return np.divide(W, row_sum, out=np.zeros_like(W), where=row_sum > 0)


def _pad_batch(batch, batch_size):
    cur_size = len(next(iter(batch.values())))
    if cur_size < batch_size:
        pad = batch_size - cur_size
        batch = {k: torch.cat([v] + [v[:1]] * pad, dim=0) for k, v in batch.items()}
    return batch, cur_size


@torch.no_grad()
def extract_features(system, dataset, indices, batch_size, call_graph):
    """Run the frozen detector and build the evidence feature tensor.

    Returns a dict with:
      feats   float32 [M, N, F]  evidence features (FEATURE_NAMES order)
      a_fused float32 [M, N]     fused anomaly score (also a feature column)
      dyn     float32 [M, N, N]  per-window learned dynamic graph (analysis)
    """
    model = system.model
    captured = []
    hook = model.dynamic_graph_learner.register_forward_hook(
        lambda module, inputs, output: captured.append(output[0].detach().cpu()))

    subset = [dataset[i] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0)

    cls_list, rec_list, rec_full_list = [], [], []
    try:
        for batch in loader:
            batch, cur_size = _pad_batch(batch, batch_size)
            batch = system.input2device(batch, system.use_gpu)
            cls_result, _, rec_score, rec_full = model(
                batch, evaluate=True, return_eval_aux=True, return_full_rec=True)
            cls_list.append(cls_result[:cur_size].cpu())
            rec_list.append(rec_score[:cur_size].cpu())
            rec_full_list.append(rec_full[:cur_size].cpu())
    finally:
        hook.remove()

    cls_prob = torch.cat(cls_list, dim=0)[..., 1].numpy()          # [M, N]
    rec = torch.cat(rec_list, dim=0).numpy()                       # [M, N]
    rec_full = torch.cat(rec_full_list, dim=0).numpy()             # [M, W, N, F]
    m = len(indices)
    dyn = torch.stack(captured, dim=0).numpy()                     # [B, N, N]
    dyn = np.repeat(dyn, batch_size, axis=0)[:m]                   # [M, N, N]

    median = np.median(rec)
    mad = max(np.median(np.abs(rec - median)), 1e-6)
    rec_prob = 1.0 / (1.0 + np.exp(-(rec - median) / (1.4826 * mad)))
    alpha = min(max(system.score_fusion_alpha, 0.0), 1.0)
    a_fused = alpha * cls_prob + (1 - alpha) * rec_prob            # [M, N]

    # propagation: modulate the static call graph with the learned weights
    w_mod = call_graph[None, :, :] * dyn                           # [M, N, N]
    explained = np.einsum('inm,im->in', w_mod, a_fused)            # callees
    blame = np.einsum('imn,im->in', w_mod, a_fused)                # callers
    unexplained = a_fused - explained

    # temporal onset from the per-timestep reconstruction energy
    series = rec_full.sum(axis=-1)                                 # [M, W, N]
    series = np.log1p(np.maximum(series, 0.0))
    weights = series - series.min(axis=1, keepdims=True)           # >= 0, per-step
    w_sum = weights.sum(axis=1)                                    # [M, N]
    steps = np.arange(series.shape[1], dtype=np.float64)[None, :, None]  # [1, W, 1]
    com = np.divide((weights * steps).sum(axis=1), w_sum,
                    out=np.full_like(w_sum, (series.shape[1] - 1) / 2.0),
                    where=w_sum > 1e-12)                           # [M, N]
    lead = com - com.min(axis=1, keepdims=True)

    feats = np.stack([
        cls_prob, rec_prob, a_fused,
        explained, unexplained, blame,
        com, lead,
    ], axis=-1).astype(np.float32)
    assert feats.shape[-1] == len(FEATURE_NAMES)
    return {'feats': feats, 'a_fused': a_fused.astype(np.float32),
            'dyn': dyn.astype(np.float32)}
