"""Causal Metric drift and exact zero-residual model compatibility."""

import unittest

import numpy as np
import torch

from src.e2e.system_trigger_model import SystemEventTrigger, metric_temporal_drift


class MetricDriftTests(unittest.TestCase):
    def test_fixed_statistic_and_service_invariance(self):
        window = torch.zeros(2, 10, 10, 48)
        window[0, 8] = 1.0
        window[0, 9] = 2.0
        window[1, 9, 3, 4] = 480.0
        actual = metric_temporal_drift(window)
        self.assertTrue(torch.allclose(actual, torch.tensor([2.0, 1.0])))
        self.assertTrue(torch.equal(actual, metric_temporal_drift(window[:, :, torch.randperm(10)])))
        with self.assertRaisesRegex(ValueError, "Metric drift expects"):
            metric_temporal_drift(window[:, :9])

    def test_zero_residual_preserves_initial_c0_logits(self):
        graph = np.zeros((10, 10), dtype=np.float32)
        for index in range(10):
            graph[index, (index + 1) % 10] = 1.0
            graph[(index + 1) % 10, index] = 1.0
        args = {
            "gpu": False, "window": 10, "batch_size": 2, "num_nodes": 10,
            "num_layer": 1, "feature_node": 8, "feature_edge": 4, "feature_log": 4,
            "num_heads_node": 2, "num_heads_log": 2, "num_heads_edge": 2,
            "num_heads_n2e": 2, "num_heads_e2n": 1, "dropout": 0.0,
            "graph_hidden": 8, "graph_sparse_weight": 0.001,
            "graph_summary_mode": "last", "head_hidden": 8,
            "raw_node": 48, "log_len": 8, "raw_edge": 4,
        }
        torch.manual_seed(19)
        original = SystemEventTrigger(graph, **args).eval()
        torch.manual_seed(19)
        variant = SystemEventTrigger(graph, **dict(args, metric_drift_residual=True)).eval()
        variant.set_metric_drift_stats(0.02, 0.01)
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, variant.state_dict()[name]), name)
        self.assertEqual(float(variant.metric_drift_alpha), 0.0)
        torch.manual_seed(29)
        batch = {
            "data_node": torch.randn(2, 10, 10, 48),
            "data_log": torch.randn(2, 10, 10, 8),
            "data_edge": torch.randn(2, 10, 10, 10, 4),
        }
        with torch.no_grad():
            old_logits, old_reg = original(batch)
            new_logits, new_reg = variant(batch)
        self.assertTrue(torch.equal(old_logits, new_logits))
        self.assertTrue(torch.equal(old_reg, new_reg))


if __name__ == "__main__":
    unittest.main()
