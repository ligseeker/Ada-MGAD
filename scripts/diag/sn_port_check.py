"""One-off porting check for the SN dataset adaptation.

Loads util.SN.data_SN.Process and asserts sample shapes, the experiment-level
train/test split (disjoint, covering all windows, test = held-out experiment
only), and label-feature visibility on the held-out experiment.

Run from the repo root:
    python3 scripts/diag/sn_port_check.py [--test_experiment -1]
"""
import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util.runtime_config import DATASET_PROFILES
from util.SN.data_SN import Process


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_experiment', type=int, default=-1)
    args = parser.parse_args()

    run_args = dict(DATASET_PROFILES['sn'])
    run_args['test_experiment'] = args.test_experiment
    proc = Process(run_args)

    num_node = run_args['num_nodes']
    window = run_args['window']
    raw_edge = run_args['raw_edge']
    vocab = pickle.load(open(os.path.join(run_args['data_path'], 'stats_vocab.pkl'), 'rb'))
    assert raw_edge == len(vocab), f"raw_edge {raw_edge} != vocab {len(vocab)}"
    logging.info("raw_edge auto-filled = %d", raw_edge)

    for idx in (0, len(proc.dataset) // 2, len(proc.dataset) - 1):
        sample = proc.dataset[idx]
        assert sample['data_node'].shape == (window, num_node, run_args['raw_node']), sample['data_node'].shape
        assert sample['data_log'].shape == (window, num_node, run_args['log_len']), sample['data_log'].shape
        assert sample['data_edge'].shape == (window, num_node, num_node, raw_edge), sample['data_edge'].shape
        assert sample['groundtruth_cls'].shape == (num_node, 3)
        assert sample['groundtruth_real'].shape == (num_node, 2)
        assert isinstance(sample['exp_id'], int)
        for key in ('data_node', 'data_log', 'data_edge'):
            assert np.isfinite(sample[key]).all(), f"non-finite values in {key}"
        assert np.allclose(sample['groundtruth_real'].sum(axis=-1), 1.0)
        assert np.allclose(sample['groundtruth_cls'].sum(axis=-1), 1.0)
        assert sample['groundtruth_cls'][:, 2].sum() == 0, "unknown class must stay 0 (SN fully labeled)"
        assert np.array_equal(sample['groundtruth_cls'][:, :2], sample['groundtruth_real'])
    logging.info("sample shapes OK (checked first/middle/last)")

    total = len(proc.dataset)
    train_set, test_set = set(proc.train_indices), set(proc.test_indices)
    assert train_set.isdisjoint(test_set), "train/test overlap"
    assert train_set | test_set == set(range(total)), "split does not cover all windows"

    fault_ids = [i for i, bound in enumerate(proc.bounds) if bound['has_fault']]
    held_out = fault_ids[args.test_experiment]
    exp_ids = np.array([sample['exp_id'] for sample in proc.dataset])
    assert set(np.unique(exp_ids[proc.test_indices])) == {held_out}, "test set contains other experiments"
    assert set(np.unique(exp_ids[proc.train_indices])) == set(range(len(proc.bounds))) - {held_out}

    bound = proc.bounds[held_out]
    span = (window - 1) * run_args['step']
    expected_test = max(0, len(range(bound['start'], bound['end'] - span, run_args['step'])))
    assert len(proc.test_indices) == expected_test, (len(proc.test_indices), expected_test)
    per_exp = {i: int((exp_ids == i).sum()) for i in range(len(proc.bounds))}
    logging.info("split OK: held-out exp %d (%s), windows per experiment: %s", held_out, bound['name'], per_exp)

    # label-feature visibility on the held-out experiment (metric is per-experiment robust-z)
    label = pickle.load(open(os.path.join(run_args['data_path'], 'label.pkl'), 'rb'))
    metric = pd.read_csv(os.path.join(run_args['data_path'], 'metric.csv'), sep=',')
    metric_values = metric.drop(columns=['now']).values
    seg_label = label[bound['start']:bound['end']]
    seg_metric = metric_values[bound['start']:bound['end']].reshape(-1, num_node, run_args['raw_node'])
    if seg_label.sum() > 0:
        pos = np.abs(seg_metric[seg_label == 1]).mean()
        neg = np.abs(seg_metric[seg_label == 0]).mean()
        logging.info("held-out robust-|z|: abnormal=%.3f normal=%.3f", pos, neg)
        assert pos > neg, "faults should be visible in the metric features"
    else:
        logging.info("held-out experiment has no labeled faults, skipping visibility check")

    logging.info("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
