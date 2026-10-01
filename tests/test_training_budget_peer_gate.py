import copy

import numpy as np
import pytest
import torch

from scripts.p6.analyze_c0_training_budget import paired_trajectory_gate, choose_bin_epoch
from src.e2e.protocol import write_json


def test_fresh_peer_gate_checks_actual_weights_and_arrays(tmp_path):
    row = {'epoch': 0, 'train_loss': 1., 'train_bce': 1., 'train_graph_regularization': 0.,
           'validation_threshold': .5, 'validation_candidate_count': 2, 'validation_metrics': {},
           'selection_key': [.8, .8, .5], 'learning_rates_used': [.001], 'learning_rates_next_epoch': [.001],
           'fit_consumed_labels': {'positive': 1}, 'diagnostics': {}, 'observer_rng_unchanged': True}
    left, right = tmp_path / 'control', tmp_path / 'budget'
    for directory, epochs in ((left, 1), (right, 30)):
        directory.mkdir()
        history = []
        for epoch in range(epochs):
            entry = dict(row, epoch=epoch)
            history.append(entry)
            folder = directory / 'epochs' / 'epoch-{:02d}'.format(epoch)
            folder.mkdir(parents=True)
            np.savez(folder / 'validation_outputs.npz', logits=np.array([.5], dtype=np.float32))
            torch.save({'weight': torch.tensor([1.])}, folder / 'model_state.pt')
        write_json(directory / 'epoch_progress.json', {'history': history})
    assert len(paired_trajectory_gate(left, right)) == 1
    torch.save({'weight': torch.tensor([1.0000001])}, right / 'epochs/epoch-00/model_state.pt')
    with pytest.raises(ValueError, match='peer trajectory mismatch'):
        paired_trajectory_gate(left, right)


def test_selector_uses_bin_metrics_with_existing_threshold_and_earliest_tie():
    history = [dict(epoch=epoch, validation_threshold=.4,
                    diagnostics={'bin_metrics_at_merged_threshold': {'event_f1': f1, 'event_recall': recall}})
               for epoch, f1, recall in ((0, .8, .7), (1, .9, .8), (2, .9, .8))]
    unchanged = copy.deepcopy(history)
    assert choose_bin_epoch(history) == 1
    assert history == unchanged
