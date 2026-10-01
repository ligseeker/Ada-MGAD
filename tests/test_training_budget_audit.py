"""Integrity gates protecting a one-variable numerical training contrast."""

import ast
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.p6.run_c0_training_budget import verify_observer_only
from src.e2e.protocol import sha256_file, write_json
from src.e2e.training_budget_audit import EpochAudit, assert_same_rng, rng_snapshot

ROOT = Path(__file__).resolve().parents[1]


def test_observer_ast_gate_accepts_audit_and_rejects_training_change(tmp_path):
    previous = subprocess.check_output(['git', 'show', '2e16c8e:scripts/p6/run_c0_trigger.py'],
                                       cwd=ROOT, text=True)
    frozen = tmp_path / 'frozen.py'
    frozen.write_text(previous)
    current = ROOT / 'scripts/p6/run_c0_trigger.py'
    verify_observer_only(current, frozen, sha256_file(frozen))
    changed = tmp_path / 'changed.py'
    changed.write_text(current.read_text().replace('pos_weight = negatives / positives',
                                                 'pos_weight = 2 * negatives / positives'))
    with pytest.raises(ValueError, match='beyond observer'):
        verify_observer_only(changed, frozen, sha256_file(frozen))


def test_rng_gate_rejects_changed_random_state():
    before = rng_snapshot()
    random.random()
    with pytest.raises(ValueError, match='RNG state'):
        assert_same_rng(before, rng_snapshot())


def test_epoch_capture_is_read_only_and_preserves_exact_outputs(tmp_path):
    old = tmp_path / 'old'
    old.mkdir()
    output = tmp_path / 'new'
    output.mkdir()
    model = torch.nn.Linear(2, 1)
    weights = old / 'weights.pt'
    torch.save(model.state_dict(), weights)
    entry = {'epoch': 0, 'train_loss': 1.0, 'train_bce': .9, 'train_graph_regularization': .1,
             'validation_threshold': .5, 'selection_key': [1.0, 1.0, .5],
             'validation_metrics': {'event_f1': 1.0}}
    write_json(old / 'training_log.json', {'history': [entry]})
    write_json(old / 'validation_selection.json', {'checkpoint': {'path': str(weights)},
                                                  'selected_epoch': 0})
    logits = np.array([-1., 1.], dtype=np.float32)
    pd.DataFrame({'system_trigger_logit': logits}).to_csv(old / 'validation_predictions.csv', index=False)
    val = {'split': 'validation', 'logits': logits, 'system_score': np.array([.2, .8], dtype=np.float32),
           'sample_index': np.array([0, 1]), 'prediction_available_time': np.array([30000, 60000]),
           'trigger_label': np.array([0, 1])}
    audit = EpochAudit(output, old, pd.DataFrame({'start_ms': [50000]}), {'effective_bins': 2})
    before = rng_snapshot()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    audit(model, val, entry, [.001], [.001])
    assert_same_rng(before, rng_snapshot())
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
    stored = np.load(output / 'epochs/epoch-00/validation_outputs.npz')
    assert np.array_equal(stored['logits'], logits)
    record = json.loads((output / 'epochs/epoch-00/epoch_audit.json').read_text())
    assert record['early_trajectory_replay']['passed']
    assert record['observer_rng_unchanged']
    assert record['diagnostics']['bin_metrics_at_merged_threshold']['true_positive_events'] == 1
