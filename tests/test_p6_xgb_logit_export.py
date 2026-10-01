import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.p6.run_c0_trigger import write_predictions


def test_exported_margin_fulfills_shared_csv_interface(tmp_path):
    root = Path(__file__).resolve().parents[1]
    original = root
    config = original / 'configs/e2e/gaia_p6_c0_flat_xgb_v1.json'
    python = json.loads(config.read_text())['xgb_python']
    rng = np.random.RandomState(42)
    train, validation = rng.rand(40, 12).astype('float32'), rng.rand(35, 12).astype('float32')
    for name, array in [('fit_features', train), ('fit_targets', (train[:, 0] > .5).astype('int8')),
                        ('validation_features', validation)]:
        np.save(tmp_path / (name + '.npy'), array)
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.glob('*.npy')}
    (tmp_path / 'feature_lock.json').write_text(json.dumps({'files': files}))
    subprocess.run([python, str(original / 'scripts/p6/fit_c0_flat_xgb.py'), '--config', str(config),
                    '--run-dir', str(tmp_path)], check=True, capture_output=True)
    subprocess.run([python, str(root / 'scripts/p6/export_c0_xgb_logits.py'), '--model', str(tmp_path / 'trigger.ubj'),
                    '--features', str(tmp_path / 'validation_features.npy'),
                    '--scores', str(tmp_path / 'validation_scores.npy'), '--output', str(tmp_path / 'logits.npy')],
                   check=True, capture_output=True)
    scores, logits = np.load(tmp_path / 'validation_scores.npy'), np.load(tmp_path / 'logits.npy')
    assert np.isfinite(logits).all()

    class Dataset:
        def __len__(self):
            return len(scores)

        def metadata(self, position):
            end = (position + 10) * 30000
            return SimpleNamespace(split='validation', sample_index=position, window_start_time=position * 30000,
                                   window_end_time=end, target_bin_start=end-30000, target_bin_end=end,
                                   prediction_available_time=end, trigger_label=0)

    csv = tmp_path / 'predictions.csv'
    write_predictions(csv, Dataset(), {'system_score': scores, 'logits': logits}, .5)
    frame = pd.read_csv(csv, float_precision='round_trip')
    np.testing.assert_array_equal(frame.system_trigger_score, scores)
    np.testing.assert_array_equal(frame.system_trigger_logit, logits)
