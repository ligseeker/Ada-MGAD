"""Train the evidence-fusion root-cause localizer on GAIA.

Pipeline: frozen Ada-MGAD detector -> three-evidence features
(src/rca/evidence.py) -> Localizer MLP trained with a listwise CE loss
against the single-root-cause labels of the train split (label_rca.csv).

Also extracts and caches test-split features so util/eval_rca.py can
score the trained localizer without reloading the detector.

Example:
  python util/train_rca.py \
      --model_path /path/to/MSTGAD-GAIA-save-xxx \
      --data_path /path/to/GAIA-pre --dataset_path /path/to/GAIA-save \
      --out_dir ./result/rca-full --groups anomaly,propagation,temporal \
      --label_budget 0.5 --seed 42
"""

import argparse
import importlib
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util.runtime_config import DATASET_PROFILES
from util.eval_rca import reconstruct_end_timestamps, load_rca_labels
from src.rca.evidence import build_call_graph, extract_features
from src.rca.localize import build_localizer
from src.rca.rank_loss import listwise_ce
import util.util as util

logger = logging.getLogger(__name__)


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Train the GAIA RCA localizer.')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--ckpt', default='f1', choices=['f1', 'loss'])
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--groups', default='anomaly,propagation,temporal')
    parser.add_argument('--label_budget', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--arch', default='mlp', choices=['mlp', 'setattn'])
    parser.add_argument('--residual', action='store_true',
                        help='Anchor scores to the a_fused anomaly evidence (residual ranking).')
    parser.add_argument('--balanced', action='store_true',
                        help='Weight the loss of each window inversely to its fault-type frequency.')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--reuse_features', default=None,
                        help='Dir with cached features_train.npz/features_test.npz; '
                             'skips detector loading and feature extraction.')
    return parser.parse_args()


def stratified_subsample(rng, types, budget):
    """Per-fault-type subsample of window indices to the given fraction."""
    keep = []
    types = np.asarray(types)
    for fault_type in sorted(set(types)):
        idx = np.where(types == fault_type)[0]
        n = max(1, int(round(len(idx) * budget)))
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def hr_at_1(scores, labels):
    return float((scores.argmax(axis=1) == labels).mean())


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cli = parse_cli_args()
    os.makedirs(cli.out_dir, exist_ok=True)
    groups = tuple(g.strip() for g in cli.groups.split(',') if g.strip())

    args = dict(DATASET_PROFILES['gaia'])
    with open(os.path.join(cli.model_path, 'params.json')) as f:
        args.update(json.load(f))
    args['data_path'] = cli.data_path
    args['dataset_path'] = cli.dataset_path
    util.seed_everything(cli.seed)

    logger.info('Loading dataset ...')
    module = importlib.import_module(args['data_module'])
    processed = module.Process(args)
    n_total = len(processed.dataset)
    end_ts = reconstruct_end_timestamps(cli.data_path, args['window'])
    assert len(end_ts) == n_total
    rca_labels = load_rca_labels(cli.data_path)
    split_idx = int(n_total * 0.7)

    train_windows, train_labels, train_types = [], [], []
    test_windows = list(range(split_idx, n_total))
    for i in range(n_total):
        hit = rca_labels.get(int(end_ts[i]))
        if hit is None or hit[0] < 0 or i >= split_idx:
            continue
        train_windows.append(i)
        train_labels.append(hit[0])
        train_types.append(hit[1].split('|')[0].strip('[]'))
    logger.info('Train-split single-root windows: %d; test windows: %d',
                len(train_windows), len(test_windows))

    if cli.reuse_features:
        logger.info('Reusing cached features from %s', cli.reuse_features)
        cached_tr = np.load(os.path.join(cli.reuse_features, 'features_train.npz'))
        cached_te = np.load(os.path.join(cli.reuse_features, 'features_test.npz'))
        assert (cached_tr['windows'] == np.asarray(train_windows)).all()
        assert (cached_te['windows'] == np.asarray(test_windows)).all()
        train_feats = cached_tr['feats']
    else:
        logger.info('Loading detector checkpoint ...')
        import src.model as model
        import util.train as train
        models = model.MyModel(processed.graph, **args)
        system = train.MY(models, **args)
        system.load_model(cli.model_path, name=cli.ckpt)

        call_graph = build_call_graph(cli.data_path)
        logger.info('Extracting train features (%d windows) ...', len(train_windows))
        train_feat = extract_features(system, processed.dataset, train_windows,
                                      args['batch_size'], call_graph)
        logger.info('Extracting test features (%d windows) ...', len(test_windows))
        test_feat = extract_features(system, processed.dataset, test_windows,
                                     args['batch_size'], call_graph)
        np.savez_compressed(os.path.join(cli.out_dir, 'features_train.npz'),
                            feats=train_feat['feats'], labels=np.asarray(train_labels),
                            types=np.asarray(train_types), windows=np.asarray(train_windows))
        np.savez_compressed(os.path.join(cli.out_dir, 'features_test.npz'),
                            feats=test_feat['feats'], windows=np.asarray(test_windows))
        train_feats = train_feat['feats']

    rng = np.random.default_rng(cli.seed)
    labeled = stratified_subsample(rng, train_types, cli.label_budget)
    n_val = max(1, int(round(0.1 * len(labeled))))
    perm = rng.permutation(labeled)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    logger.info('Label budget %.2f -> %d labeled windows (%d train / %d val)',
                cli.label_budget, len(labeled), len(tr_idx), len(val_idx))

    if cli.reuse_features:
        import shutil
        shutil.copy(os.path.join(cli.reuse_features, 'features_test.npz'),
                    os.path.join(cli.out_dir, 'features_test.npz'))

    feats = train_feats
    labels = np.asarray(train_labels)
    mean = feats[tr_idx].reshape(-1, feats.shape[-1]).mean(axis=0)
    std = feats[tr_idx].reshape(-1, feats.shape[-1]).std(axis=0) + 1e-6
    norm = (feats - mean) / std

    x_tr = torch.tensor(norm[tr_idx], dtype=torch.float32)
    y_tr = torch.tensor(labels[tr_idx], dtype=torch.long)
    x_val = torch.tensor(norm[val_idx], dtype=torch.float32)
    y_val = torch.tensor(labels[val_idx], dtype=torch.long)

    if cli.balanced:
        tr_types = np.asarray(train_types)[tr_idx]
        freq = pd.Series(tr_types).value_counts()
        w_tr = torch.tensor((1.0 / freq[tr_types].values).astype(np.float32))
        w_tr = w_tr * (len(w_tr) / w_tr.sum())
        logger.info('Type-balanced loss; type frequencies: %s', dict(freq))
    else:
        w_tr = torch.ones(len(tr_idx), dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    localizer = build_localizer(cli.arch, groups, hidden=cli.hidden,
                                d_model=cli.d_model, num_layers=cli.num_layers,
                                num_nodes=args['num_nodes'], residual=cli.residual).to(device)
    optimizer = torch.optim.Adam(localizer.parameters(), lr=cli.lr)
    loader = DataLoader(TensorDataset(x_tr, y_tr, w_tr), batch_size=cli.batch_size,
                        shuffle=True, drop_last=False)

    best = {'hr1': -1.0, 'epoch': -1}
    bad_epochs = 0
    for epoch in range(cli.epochs):
        localizer.train()
        total_loss = 0.0
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            loss = listwise_ce(localizer(xb), yb, sample_weight=wb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(xb)
        localizer.eval()
        with torch.no_grad():
            val_hr1 = hr_at_1(localizer(x_val.to(device)).cpu().numpy(), y_val.numpy())
        if val_hr1 > best['hr1']:
            best = {'hr1': val_hr1, 'epoch': epoch,
                    'state': {k: v.detach().cpu().clone()
                              for k, v in localizer.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch % 10 == 0 or bad_epochs == 0:
            logger.info('epoch %d  loss %.4f  val HR@1 %.4f', epoch,
                        total_loss / len(tr_idx), val_hr1)
        if bad_epochs >= cli.patience:
            logger.info('Early stop at epoch %d (best %d, HR@1 %.4f)',
                        epoch, best['epoch'], best['hr1'])
            break

    torch.save(best['state'], os.path.join(cli.out_dir, 'localizer.pt'))
    config = {'groups': list(groups), 'arch': cli.arch, 'hidden': cli.hidden,
              'd_model': cli.d_model, 'num_layers': cli.num_layers,
              'residual': cli.residual, 'balanced': cli.balanced,
              'label_budget': cli.label_budget, 'seed': cli.seed,
              'feature_mean': mean.tolist(), 'feature_std': std.tolist(),
              'val_hr1': best['hr1'], 'best_epoch': best['epoch'],
              'model_path': cli.model_path, 'ckpt': cli.ckpt,
              'n_labeled': int(len(labeled))}
    with open(os.path.join(cli.out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    logger.info('Saved localizer to %s (val HR@1 %.4f)', cli.out_dir, best['hr1'])


if __name__ == '__main__':
    main()
