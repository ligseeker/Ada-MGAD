#!/usr/bin/env python3
"""Score the existing fixed flat XGB detector, with no fitting or label API."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import xgboost as xgb
from src.e2e.flat_trigger_features import flatten_trigger_windows


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scope', type=Path, required=True)
    p.add_argument('--model-id', required=True); p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(); scope = json.loads(args.scope.read_text()); config = scope['config']
    record = next(m for m in config['models'] if m['id'] == args.model_id)
    if xgb.__version__ != record['xgboost_version']: raise ValueError('XGBoost version mismatch')
    model = xgb.XGBClassifier(); model.load_model(record['checkpoint']); model.set_params(n_jobs=8)
    run = Path(record['run']); reference = pd.read_csv(run / 'validation_predictions.csv', float_precision='round_trip',
        usecols=['sample_index', 'system_trigger_score', 'system_trigger_logit'])
    selected = np.r_[np.arange(32), np.arange(len(reference) // 2, len(reference) // 2 + 32),
                     np.arange(len(reference) - 32, len(reference))]
    train = {name: np.load(Path(config['data_root']) / 'train' / (name + '.npy'), mmap_mode='r')
             for name in ('metric', 'log', 'trace')}
    rows = flatten_trigger_windows(train['metric'], train['log'], train['trace'],
        reference.sample_index.to_numpy()[selected])
    actual = model.predict_proba(rows)[:, 1]
    error = float(np.max(abs(actual.astype(float) - reference.system_trigger_score.to_numpy()[selected])))
    if error != 0: raise ValueError('frozen flat feature/model Validation replay failed')
    arrays = {name: np.load(Path(config['data_root']) / 'test' / (name + '.npy'), mmap_mode='r')
              for name in ('metric', 'log', 'trace', 'timestamps')}
    indices = np.arange(len(arrays['timestamps']) - 9); scores = []; logits = []
    for start in range(0, len(indices), 256):
        rows = flatten_trigger_windows(arrays['metric'], arrays['log'], arrays['trace'], indices[start:start + 256])
        scores.append(model.predict_proba(rows)[:, 1]); logits.append(model.predict(rows, output_margin=True))
    pd.DataFrame({'sample_index': indices,
        'prediction_available_time': np.asarray(arrays['timestamps'][indices + 9], dtype=np.int64) + 30000,
        'system_score': np.concatenate(scores), 'logit': np.concatenate(logits)}).to_csv(args.output_dir / 'scores.csv', index=False)
    (args.output_dir / 'replay.json').write_text(json.dumps({'passed': True, 'n': len(selected),
        'probability_error': error, 'xgboost_version': xgb.__version__, 'labels_consumed': False}) + '\n')


if __name__ == '__main__': main()
