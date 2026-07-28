"""Root-cause-analysis ranking evaluation on GAIA anomalous windows.

Scores every single-root anomalous window in the chronological test split
(last 30% of windows) with per-node root-cause scores and reports
HR@1/3/5, MRR and NDCG@3/5 per fault type plus macro-averages.

Methods evaluated:
  * detector      fused anomaly scores of a frozen Ada-MGAD checkpoint
                  (cls probability + MAD-normalized reconstruction energy,
                  identical to util/train.py score fusion)
  * random        uniform random scores (mean over 5 seeds)
  * frequency     historical root-cause frequency of the train split
  * metric_dev    per-node |z-score| of the last timestep's metrics
                  against the window history
  * log_dev       per-node L1 deviation of the last timestep's log vector
  * trace_dev     per-edge |z-score| of the last timestep's trace features,
                  averaged over each node's incident edges

Window end timestamps are reconstructed from the label.csv grid with the
same segmentation rule as util/GAIA/data_GAIA.py (gap > 3 * 30s), and
joined with label_rca.csv built by util/GAIA/build_rca_labels.py.

Example:
  python util/eval_rca.py \
      --model_path /path/to/MSTGAD-GAIA-save-xxx \
      --data_path  /path/to/GAIA-pre \
      --dataset_path /path/to/GAIA-save
"""

import argparse
import importlib
import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util.runtime_config import DATASET_PROFILES
from util.GAIA.constant import GAIA_SERVICES
import util.util as util

logger = logging.getLogger(__name__)

INTERVAL_MS = 30 * 1000
GAP_THRESHOLD_MS = INTERVAL_MS * 3  # same as util/GAIA/data_GAIA.py
KS = (1, 3, 5)
RANDOM_SEEDS = (42, 43, 44, 45, 46)


# --------------------------------------------------------------------------
# window <-> timestamp mapping
# --------------------------------------------------------------------------

def reconstruct_end_timestamps(data_path, window):
    """Reconstruct the end timestamp of every dataset window, in order.

    Mirrors the segmentation in util/GAIA/data_GAIA.py: windows are built
    per continuous segment (gap > GAP_THRESHOLD_MS) with stride 1, and a
    window's label is taken at its last timestep.
    """
    grid = pd.read_csv(os.path.join(data_path, 'label.csv'),
                       usecols=['timestamp'])['timestamp'].values.astype(np.int64)
    grid.sort()
    segments = []
    seg_start = 0
    for i, d in enumerate(np.diff(grid)):
        if d > GAP_THRESHOLD_MS:
            segments.append((seg_start, i + 1))
            seg_start = i + 1
    segments.append((seg_start, len(grid)))

    end_ts = []
    for seg_start, seg_end in segments:
        for i in range(seg_end - seg_start - window + 1):
            end_ts.append(grid[seg_start + i + window - 1])
    return np.asarray(end_ts, dtype=np.int64)


def load_rca_labels(data_path):
    frame = pd.read_csv(os.path.join(data_path, 'label_rca.csv'))
    return {int(row.timestamp): (int(row.root_cause_id), str(row.fault_type))
            for row in frame.itertuples(index=False)}


# --------------------------------------------------------------------------
# scoring methods
# --------------------------------------------------------------------------

@torch.no_grad()
def detector_scores(system, dataset, indices, batch_size):
    """Fused per-node anomaly scores of the frozen detector.

    The model precomputes edge indices for a fixed batch_size, so the final
    partial batch is padded with repeated samples and truncated afterwards.
    """
    subset = [dataset[i] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0)
    cls_list, rec_list = [], []
    for batch in loader:
        cur_size = len(next(iter(batch.values())))
        if cur_size < batch_size:
            pad = batch_size - cur_size
            batch = {k: torch.cat([v] + [v[:1]] * pad, dim=0)
                     for k, v in batch.items()}
        batch = system.input2device(batch, system.use_gpu)
        cls_result, _, rec_score = system.model(batch, evaluate=True, return_eval_aux=True)
        cls_list.append(cls_result[:cur_size].cpu())
        rec_list.append(rec_score[:cur_size].cpu())
    cls_probs = torch.cat(cls_list, dim=0)[..., 1]          # [M, N]
    rec = torch.cat(rec_list, dim=0)                        # [M, N]

    median = torch.median(rec)
    mad = torch.median(torch.abs(rec - median)).clamp_min(1e-6)
    rec_prob = torch.sigmoid((rec - median) / (1.4826 * mad))

    alpha = min(max(system.score_fusion_alpha, 0.0), 1.0)
    return (alpha * cls_probs + (1 - alpha) * rec_prob).numpy()


def zscore_last(values):
    """|z| of the last timestep against the window history. (W, ...) -> (...)"""
    history, last = values[:-1], values[-1]
    mean = history.mean(axis=0)
    std = history.std(axis=0)
    return np.abs((last - mean) / (std + 1e-6))


def heuristic_scores(dataset, indices):
    """Model-free per-node evidence from the window tensors."""
    metric_dev = np.zeros((len(indices), len(GAIA_SERVICES)))
    log_dev = np.zeros_like(metric_dev)
    trace_dev = np.zeros_like(metric_dev)
    for row, i in enumerate(indices):
        sample = dataset[i]
        metric_dev[row] = zscore_last(sample['data_node']).mean(axis=-1)
        log_dev[row] = np.abs(
            sample['data_log'][-1] - np.median(sample['data_log'][:-1], axis=0)
        ).mean(axis=-1)
        edge_z = zscore_last(sample['data_edge']).mean(axis=-1)  # (N, N)
        incident = edge_z + edge_z.T
        trace_dev[row] = incident.sum(axis=1) / np.maximum(
            (incident > 0).sum(axis=1), 1)
    return {'metric_dev': metric_dev, 'log_dev': log_dev, 'trace_dev': trace_dev}


def frequency_scores(train_labels, n_eval):
    counts = np.zeros(len(GAIA_SERVICES))
    for node_id in train_labels:
        counts[node_id] += 1
    return np.tile(counts, (n_eval, 1))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def rca_metrics(scores, labels):
    ranks = np.empty(len(labels))
    order = np.argsort(-scores, axis=1, kind='stable')
    for i in range(len(labels)):
        ranks[i] = int(np.where(order[i] == labels[i])[0][0]) + 1
    result = {f'HR@{k}': float((ranks <= k).mean()) for k in KS}
    result['MRR'] = float((1.0 / ranks).mean())
    for k in KS:
        result[f'NDCG@{k}'] = float(
            np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0).mean())
    return result


def evaluate_method(name, scores, eval_labels, eval_types, fault_types):
    per_type = {}
    for fault_type in fault_types:
        mask = np.array([t == fault_type for t in eval_types])
        if mask.sum() == 0:
            continue
        per_type[fault_type] = rca_metrics(scores[mask], eval_labels[mask])
        per_type[fault_type]['n_windows'] = int(mask.sum())
    metric_names = [f'HR@{k}' for k in KS] + ['MRR'] + [f'NDCG@{k}' for k in KS]
    macro = {metric: float(np.mean([m[metric] for m in per_type.values()]))
             for metric in metric_names}
    overall = rca_metrics(scores, eval_labels)
    return {'per_type': per_type, 'macro': macro, 'micro': overall}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_cli_args():
    parser = argparse.ArgumentParser(description='GAIA RCA ranking evaluation.')
    parser.add_argument('--model_path', default=None,
                        help='Checkpoint dir with params.json and my_{f1,loss}_stage.ckpt. '
                             'Omit to run only model-free baselines.')
    parser.add_argument('--ckpt', default='f1', choices=['f1', 'loss'])
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--out', default=None, help='Output JSON path.')
    parser.add_argument('--max_windows', type=int, default=0,
                        help='Limit test windows for a smoke run (0 = all).')
    parser.add_argument('--localizer_path', action='append', default=[],
                        help='Dir with localizer.pt/config.json/features_test.npz '
                             '(from util/train_rca.py). Repeatable.')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cli = parse_cli_args()

    args = dict(DATASET_PROFILES['gaia'])
    if cli.model_path:
        with open(os.path.join(cli.model_path, 'params.json')) as f:
            args.update(json.load(f))
    args['data_path'] = cli.data_path
    args['dataset_path'] = cli.dataset_path
    util.seed_everything(args['random_seed'])

    logger.info('Loading dataset from %s ...', cli.dataset_path)
    module = importlib.import_module(args['data_module'])
    processed = module.Process(args)
    n_total = len(processed.dataset)

    end_ts = reconstruct_end_timestamps(cli.data_path, args['window'])
    assert len(end_ts) == n_total, \
        f'window/timestamp mismatch: {len(end_ts)} reconstructed vs {n_total} windows'

    rca_labels = load_rca_labels(cli.data_path)

    split_idx = int(n_total * 0.7)
    test_indices = list(range(split_idx, n_total))
    if cli.max_windows > 0:
        test_indices = test_indices[:cli.max_windows]

    eval_indices, eval_labels, eval_types = [], [], []
    train_root_causes = []
    n_mixed_type = 0
    test_set = set(test_indices)
    for i in range(n_total):
        hit = rca_labels.get(int(end_ts[i]))
        if hit is None or hit[0] < 0:
            continue
        node_id, type_str = hit
        primary_type = type_str.split('|')[0].strip('[]')
        if '|' in type_str:
            n_mixed_type += 1
        if i < split_idx:
            train_root_causes.append(node_id)
        elif i in test_set:
            eval_indices.append(i)
            eval_labels.append(node_id)
            eval_types.append(primary_type)
    eval_labels = np.asarray(eval_labels)
    fault_types = sorted(set(eval_types))
    logger.info('RCA eval windows: %d (single-root, test split); types: %s; '
                'mixed-type windows folded to primary: %d',
                len(eval_indices), {t: eval_types.count(t) for t in fault_types}, n_mixed_type)

    eval_indices = np.asarray(eval_indices)
    methods = {}

    logger.info('Computing heuristic scores ...')
    methods.update(heuristic_scores(processed.dataset, eval_indices))

    methods['frequency'] = frequency_scores(train_root_causes, len(eval_indices))

    rng_scores = np.zeros((len(eval_indices), len(GAIA_SERVICES)))
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        ranks = np.argsort(-rng.random(rng_scores.shape), axis=1, kind='stable')
        for i in range(len(eval_indices)):
            rng_scores[i, ranks[i]] += np.arange(len(GAIA_SERVICES), 0, -1)
    methods['random'] = rng_scores

    if cli.model_path:
        logger.info('Loading detector checkpoint %s (my_%s_stage.ckpt) ...',
                    cli.model_path, cli.ckpt)
        import src.model as model
        import util.train as train
        models = model.MyModel(processed.graph, **args)
        system = train.MY(models, **args)
        system.load_model(cli.model_path, name=cli.ckpt)
        methods['detector'] = detector_scores(system, processed.dataset,
                                              test_indices, args['batch_size'])
        test_pos = {idx: pos for pos, idx in enumerate(test_indices)}
        eval_pos = [test_pos[i] for i in eval_indices]
        methods['detector'] = methods['detector'][eval_pos]

    for loc_dir in cli.localizer_path:
        with open(os.path.join(loc_dir, 'config.json')) as f:
            loc_cfg = json.load(f)
        from src.rca.localize import Localizer
        localizer = Localizer(groups=loc_cfg['groups'], hidden=loc_cfg['hidden'])
        localizer.load_state_dict(torch.load(os.path.join(loc_dir, 'localizer.pt'),
                                             map_location='cpu'))
        localizer.eval()
        cache = np.load(os.path.join(loc_dir, 'features_test.npz'))
        mean = np.asarray(loc_cfg['feature_mean'], dtype=np.float32)
        std = np.asarray(loc_cfg['feature_std'], dtype=np.float32)
        norm = (cache['feats'] - mean) / std
        with torch.no_grad():
            all_scores = localizer(torch.tensor(norm)).numpy()
        win_pos = {int(w): pos for pos, w in enumerate(cache['windows'])}
        rows = [win_pos[i] for i in eval_indices]
        tag = ''.join(g[0].upper() for g in loc_cfg['groups'])
        methods[f'localizer[{tag}]'] = all_scores[rows]

    results = {name: evaluate_method(name, scores, eval_labels, eval_types, fault_types)
               for name, scores in methods.items()}

    metric_names = [f'HR@{k}' for k in KS] + ['MRR'] + [f'NDCG@{k}' for k in KS]
    header = f"{'method':<14}" + ''.join(f'{m:>10}' for m in metric_names)
    print('\n=== macro-averaged over fault types ===')
    print(header)
    for name, res in results.items():
        row = f'{name:<14}' + ''.join(f"{res['macro'][m]:>10.4f}" for m in metric_names)
        print(row)
    print('\n=== micro (all single-root windows) ===')
    print(header)
    for name, res in results.items():
        row = f'{name:<14}' + ''.join(f"{res['micro'][m]:>10.4f}" for m in metric_names)
        print(row)

    out = cli.out or (os.path.join(cli.model_path, 'rca_eval.json') if cli.model_path
                      else './rca_eval.json')
    payload = {
        'n_eval_windows': int(len(eval_indices)),
        'fault_type_counts': {t: eval_types.count(t) for t in fault_types},
        'results': results,
        'args': {'model_path': cli.model_path, 'ckpt': cli.ckpt,
                 'data_path': cli.data_path, 'dataset_path': cli.dataset_path},
    }
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    logger.info('Saved %s', out)


if __name__ == '__main__':
    main()
