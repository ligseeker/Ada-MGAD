"""Boundary, label isolation and denominator checks for the Test wrapper."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.p6.run_frozen_test_review import InputOnlyWindows, make_episodes
from scripts.p6.run_c0_trigger import infer_trigger_logits
from src.e2e.system_trigger_data import build_trigger_loader
from src.e2e.event_detection import match_events, event_metrics


class IdentityModel(torch.nn.Module):
    def forward(self, batch):
        return batch['sample_index'].float(), None


class FrozenTestReviewTests(unittest.TestCase):
    def test_no_label_files_last_complete_window_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name, array in {
                'timestamps': np.arange(44, dtype=np.int64) * 30000,
                'metric': np.zeros((44, 10, 48), np.float32),
                'log': np.zeros((44, 10, 32), np.float32),
                'trace': np.zeros((44, 10, 10, 8), np.float32),
            }.items():
                np.save(directory / (name + '.npy'), array)
            dataset = InputOnlyWindows(directory)
            self.assertEqual(len(dataset), 35)
            self.assertEqual(set(dataset[34]), {'data_node', 'data_log', 'data_edge', 'sample_index'})
            loader = build_trigger_loader(dataset, batch_size=32, num_workers=0)
            indices, logits = infer_trigger_logits(IdentityModel(), loader, torch.device('cpu'))
            np.testing.assert_array_equal(indices, np.arange(35))
            np.testing.assert_array_equal(logits, np.arange(35))
            self.assertEqual(int(dataset.timestamps[indices[-1] + 9]) + 30000, 44 * 30000)

    def test_rise_requires_adjacent_positives_and_strict_logit_increase(self):
        scores = pd.DataFrame({'prediction_available_time': np.arange(6) * 30000 + 30000,
            'system_score': [.1, .9, .9, .95, .1, .9], 'logit': [-2., 2., 2., 3., -2., 2.]})
        expected = {'merged': [60000, 180000], 'rise': [60000, 120000, 180000],
                    'bin': [60000, 90000, 120000, 180000]}
        for decoder, anchors in expected.items():
            episodes = make_episodes(scores, {'threshold': .9, 'decoder': decoder})
            self.assertEqual(episodes.t_hat.tolist(), anchors)
            self.assertTrue(episodes.prediction_id.is_unique)

    def test_extra_alarm_and_missed_event_remain_in_denominators(self):
        scores = pd.DataFrame({'prediction_available_time': [30000, 60000, 90000],
            'system_score': [1., 1., 1.], 'logit': [2., 2., 2.]})
        episodes = make_episodes(scores, {'threshold': .9, 'decoder': 'bin'})
        gt = pd.DataFrame({'case_id': ['first', 'second', 'missed'], 'source_index': [0, 1, 2],
            'service': ['mobservice1'] * 3, 'fault_type': ['login_failure'] * 3,
            'start_ms': [10000, 35000, 200000], 'end_ms': [11000, 36000, 201000]})
        metrics = event_metrics(match_events(episodes, gt, tolerance_seconds=60))
        self.assertEqual(metrics['ground_truth_event_count'], 3)
        self.assertEqual(metrics['predicted_episode_count'], 3)
        self.assertEqual(metrics['true_positive_events'], 2)
        self.assertEqual(metrics['false_positive_events'], 1)
        self.assertEqual(metrics['false_negative_events'], 1)


if __name__ == '__main__':
    unittest.main()
