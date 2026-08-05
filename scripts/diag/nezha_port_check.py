"""One-off porting check for the Nezha OnlineBoutique dataset adaptation.

Loads util.Nezha.data_Nezha.Process once per fold (test_experiment=1 and 0)
and asserts sample shapes, the day-level train/test split (disjoint, covering
all windows, test = held-out day only, no cross-day windows), and
label-feature visibility on faulted minutes (e.g. CpuUsageRate spikes for
cpu_contention/cpu_consumed faults).

Run from the repo root:
    python3 scripts/diag/nezha_port_check.py
"""
import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util.runtime_config import DATASET_PROFILES
from util.Nezha.data_Nezha import Process
from util.Nezha.pre_Nezha import DEFAULT_RAW_DIR
from util.Nezha.constant import NEZHA_SERVICE2NID, pod_to_service


def check_samples(proc, run_args):
    num_node = run_args['num_nodes']
    window = run_args['window']
    raw_edge = run_args['raw_edge']
    vocab = pickle.load(open(os.path.join(run_args['data_path'], 'stats_vocab.pkl'), 'rb'))
    assert raw_edge == len(vocab), f"raw_edge {raw_edge} != vocab {len(vocab)}"

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
        assert sample['groundtruth_cls'][:, 2].sum() == 0, "unknown class must stay 0 (Nezha fully labeled)"
        assert np.array_equal(sample['groundtruth_cls'][:, :2], sample['groundtruth_real'])
    logging.info("sample shapes OK (checked first/middle/last), raw_edge=%d", raw_edge)


def check_split(proc, run_args, test_experiment):
    window, step = run_args['window'], run_args['step']
    span = (window - 1) * step
    total = len(proc.dataset)
    train_set, test_set = set(proc.train_indices), set(proc.test_indices)
    assert train_set.isdisjoint(test_set), "train/test overlap"
    assert train_set | test_set == set(range(total)), "split does not cover all windows"

    fault_ids = [i for i, bound in enumerate(proc.bounds) if bound['has_fault']]
    held_out = fault_ids[test_experiment]
    exp_ids = np.array([sample['exp_id'] for sample in proc.dataset])
    assert set(np.unique(exp_ids[proc.test_indices])) == {held_out}, "test set contains other days"
    assert set(np.unique(exp_ids[proc.train_indices])) == set(range(len(proc.bounds))) - {held_out}

    per_day = {i: int((exp_ids == i).sum()) for i in range(len(proc.bounds))}
    for day_id, bound in enumerate(proc.bounds):
        expected = max(0, len(range(bound['start'], bound['end'] - span, step)))
        assert per_day[day_id] == expected, (day_id, per_day[day_id], expected)
    expected_test = max(0, len(range(proc.bounds[held_out]['start'], proc.bounds[held_out]['end'] - span, step)))
    assert len(proc.test_indices) == expected_test, (len(proc.test_indices), expected_test)
    logging.info("split OK: held-out day %d (%s), windows per day: %s", held_out, proc.bounds[held_out]['name'], per_day)


def check_visibility(run_args):
    label = pickle.load(open(os.path.join(run_args['data_path'], 'label.pkl'), 'rb'))
    metric = pd.read_csv(os.path.join(run_args['data_path'], 'metric.csv'), sep=',')
    metric_values = metric.drop(columns=['now']).values
    bounds = pickle.load(open(os.path.join(run_args['data_path'], 'bounds.pkl'), 'rb'))
    num_node = run_args['num_nodes']
    raw_node = run_args['raw_node']

    seg_metric = metric_values.reshape(-1, num_node, raw_node)
    pos = np.abs(seg_metric[label == 1]).mean()
    neg = np.abs(seg_metric[label == 0]).mean()
    logging.info("labeled robust-|z|: abnormal=%.3f normal=%.3f", pos, neg)
    assert pos > neg, "faults should be visible in the metric features"

    cpu_hits, cpu_total = 0, 0
    for bound in bounds:
        faults = json.load(open(os.path.join(DEFAULT_RAW_DIR, 'rca_data', bound['name'],
                                             f"{bound['name']}-fault_list.json")))
        seg_label = label[bound['start']:bound['end']]
        for fault in [item for group in faults.values() for item in group]:
            if fault['inject_type'] not in ('cpu_contention', 'cpu_consumed'):
                continue
            service = pod_to_service(fault['inject_pod'])
            # feature 0 of each service is CpuUsageRate(%) by KPI ordering
            col = seg_metric[:, NEZHA_SERVICE2NID[service], 0]
            rows = np.flatnonzero(seg_label[:, NEZHA_SERVICE2NID[service]] == 1)
            assert len(rows) > 0, f"no labeled minutes for fault {fault}"
            if col[rows].mean() > col.mean() + 0.5:
                cpu_hits += 1
            cpu_total += 1
    logging.info("cpu-fault CpuUsageRate visibility: %d/%d faults show elevated z", cpu_hits, cpu_total)
    assert cpu_hits > cpu_total / 2, "cpu faults should raise CpuUsageRate"


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for test_experiment in (1, 0):
        logging.info("=== fold test_experiment=%d ===", test_experiment)
        run_args = dict(DATASET_PROFILES['nezha'])
        run_args['test_experiment'] = test_experiment
        proc = Process(run_args)
        check_samples(proc, run_args)
        check_split(proc, run_args, test_experiment)
    check_visibility(DATASET_PROFILES['nezha'])
    logging.info("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
