"""One-off porting check for the TT dataset adaptation.

Loads util.TT.data_TT.Process and asserts sample shapes, window containment
within single experiments, the experiment-level train/test split (disjoint,
covering all windows, test = held-out experiment only), and CPU-z visibility
inside the cpu_load fault window of the held-out experiment.

Run from the repo root:
    python3 scripts/diag/tt_port_check.py [--test_experiment 0] [--rebuild_cache]
"""
import argparse
import json
import logging
import os
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util.runtime_config import DATASET_PROFILES
from util.TT.constant import DEFAULT_RAW_DIR
from util.TT.data_TT import Process


def fault_json_path(exp_name):
    fault_name = exp_name.replace('TT.', 'TT.fault-') + '.json'
    return os.path.join(DEFAULT_RAW_DIR, 'data', fault_name)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_experiment', type=int, default=0)
    parser.add_argument('--rebuild_cache', action='store_true',
                        help='Delete the TT-save cache first so the rebuild path is exercised.')
    args = parser.parse_args()

    run_args = dict(DATASET_PROFILES['tt'])
    run_args['test_experiment'] = args.test_experiment
    if args.rebuild_cache:
        shutil.rmtree(run_args['dataset_path'], ignore_errors=True)
        logging.info("deleted cache dir %s", run_args['dataset_path'])
    proc = Process(run_args)

    num_node = run_args['num_nodes']
    window = run_args['window']
    step = run_args['step']
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
            assert sample[key].dtype == np.float32
        assert np.allclose(sample['groundtruth_real'].sum(axis=-1), 1.0)
        assert np.allclose(sample['groundtruth_cls'].sum(axis=-1), 1.0)
        assert sample['groundtruth_cls'][:, 2].sum() == 0, "unknown class must stay 0 (TT fully labeled)"
        assert np.array_equal(sample['groundtruth_cls'][:, :2], sample['groundtruth_real'])
    logging.info("sample shapes OK (checked first/middle/last)")

    span = (window - 1) * step
    for exp_id, t0 in proc.window_index:
        bound = proc.bounds[exp_id]
        assert bound['start'] <= t0 and t0 + span < bound['end'], \
            f"window ({exp_id}, {t0}) crosses experiment bounds {bound}"
    logging.info("all %d windows stay inside their experiment bounds", len(proc.window_index))

    total = len(proc.dataset)
    train_set, test_set = set(proc.train_indices), set(proc.test_indices)
    assert train_set.isdisjoint(test_set), "train/test overlap"
    assert train_set | test_set == set(range(total)), "split does not cover all windows"

    fault_ids = [i for i, bound in enumerate(proc.bounds) if bound['has_fault']]
    held_out = fault_ids[args.test_experiment]
    exp_ids = np.array([exp_id for exp_id, _ in proc.window_index])
    assert set(np.unique(exp_ids[proc.test_indices])) == {held_out}, "test set contains other experiments"
    assert set(np.unique(exp_ids[proc.train_indices])) == set(range(len(proc.bounds))) - {held_out}

    bound = proc.bounds[held_out]
    expected_test = max(0, len(range(bound['start'], bound['end'] - span, step)))
    assert len(proc.test_indices) == expected_test, (len(proc.test_indices), expected_test)
    per_exp = {i: int((exp_ids == i).sum()) for i in range(len(proc.bounds))}
    logging.info("split OK: held-out exp %d (%s), windows per experiment: %s", held_out, bound['name'], per_exp)

    seg_label = proc.set['label'][bound['start']:bound['end'], :, 1]
    seg_metric = proc.set['metric'][bound['start']:bound['end']]
    fault_path = fault_json_path(bound['name'])
    assert os.path.exists(fault_path), f"fault JSON not found: {fault_path}"
    fault_data = json.load(open(fault_path))
    t0_epoch = bound.get('t0', int(fault_data['start']))
    cpu_checks = 0
    for fault in fault_data.get('faults', []):
        service = fault['name'].replace('dockercomposemanifests_', '').rsplit('_', 1)[0]
        from util.TT.constant import TT_SERVICE2NID
        if service not in TT_SERVICE2NID:
            continue
        nid = TT_SERVICE2NID[service]
        rel_start = max(0, int(fault['start'] - t0_epoch))
        rel_end = min(seg_label.shape[0], rel_start + int(fault['duration']))
        in_win = np.zeros(seg_label.shape[0], dtype=bool)
        in_win[rel_start:rel_end] = True
        assert seg_label[in_win, nid].sum() > 0, f"label missing for {service} in window {fault}"
        channels = slice(0, 3) if fault['fault'] == 'cpu_load' else slice(None)
        series = np.abs(seg_metric[:, nid, channels])
        pos = series[in_win].mean()
        neg = series[~in_win].mean()
        logging.info("fault %-14s %-28s window [%d,%d): |z| in=%.3f out=%.3f ratio=%.2f",
                     fault['fault'], service, rel_start, rel_end, pos, neg, pos / max(neg, 1e-9))
        if fault['fault'] == 'cpu_load':
            cpu_checks += 1
            assert pos > 2.0 * neg, f"cpu_load fault on {service} not visible in CPU z"
    assert cpu_checks > 0, "no cpu_load fault found in the held-out experiment"
    logging.info("cpu_load CPU-z visibility OK (%d windows)", cpu_checks)

    logging.info("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
