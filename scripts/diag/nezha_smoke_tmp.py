import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
sys.path.insert(0, '.')

from util.runtime_config import DATASET_PROFILES

args = dict(DATASET_PROFILES['nezha'])
args['dataset'] = 'nezha'

from util.runtime import load_process

processed = load_process(args)

sample = processed.dataset[0]
print('=== smoke test ===')
print('num samples:', len(processed.dataset))
print('data_node :', sample['data_node'].shape)
print('data_log  :', sample['data_log'].shape)
print('data_edge :', sample['data_edge'].shape)
print('gt_cls    :', sample['groundtruth_cls'].shape, ' gt_real:', sample['groundtruth_real'].shape)
print('resolved  : raw_node=%d log_len=%d raw_edge=%d num_nodes=%d' % (
    args['raw_node'], args['log_len'], args['raw_edge'], args['num_nodes']))
print('graph     :', processed.graph.shape, 'edges:', int(processed.graph.sum() // 2))
print('train=%d test=%d' % (len(processed.train_indices), len(processed.test_indices)))

import numpy as np

train_abn = sum(1 for i in processed.train_indices if processed.dataset[i]['groundtruth_real'][:, 1].sum() > 0)
test_abn = sum(1 for i in processed.test_indices if processed.dataset[i]['groundtruth_real'][:, 1].sum() > 0)
print('train abnormal windows: %d (%.1f%%)' % (train_abn, 100 * train_abn / len(processed.train_indices)))
print('test  abnormal windows: %d (%.1f%%)' % (test_abn, 100 * test_abn / len(processed.test_indices)))

exp_ids = np.array([s['exp_id'] for s in processed.dataset])
test_exp = np.unique(exp_ids[processed.test_indices])
train_exp = np.unique(exp_ids[processed.train_indices])
print('train exp_ids:', train_exp.tolist(), '| test exp_ids:', test_exp.tolist())

# model-side dimension check
import torch
import src.model as model

m = model.MyModel(processed.graph, **args)
n_params = sum(p.numel() for p in m.parameters())
print('model built OK, params=%d' % n_params)
