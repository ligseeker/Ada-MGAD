"""C1 Test ranking covers legal and failed episodes without GT input."""

import json
import unittest

import numpy as np
import pandas as pd

from src.e2e.c1_test_scoring import score_c1_test_episodes
from src.e2e.rca_model import ConditionalLogitFit, GAIA_SERVICES


class _EmptyIndex:
    def case_indicators(self, anchor_ms, spec):
        return {}


class _FailingIndex:
    def case_indicators(self, anchor_ms, spec):
        raise ValueError("synthetic feature failure")


def _model():
    return ConditionalLogitFit(
        weights=np.zeros(68), scaler_mean=np.zeros(68), scaler_scale=np.ones(68),
        train_case_indices=(0,), initial_loss=0.0, final_loss=0.0,
        gradient_norm=0.0, iterations=0, converged=True, message="test")


class C1TestScoringTest(unittest.TestCase):
    def test_all_episode_scope_and_label_schema_firewall(self):
        episodes = pd.DataFrame([
            {"prediction_id": "test-pred-000000", "split": "test", "t_hat": 600000,
             "episode_end_time": 630000, "positive_bins": 1, "system_score": 0.9, "threshold": 0.5},
            {"prediction_id": "test-pred-000001", "split": "test", "t_hat": 990000,
             "episode_end_time": 1020000, "positive_bins": 1, "system_score": 0.8, "threshold": 0.5},
        ])
        scores = score_c1_test_episodes(
            episodes=episodes, index=_EmptyIndex(), arm_b=_model(), arm_c=_model(),
            test_interval_ms=(0, 1200000))
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores["scope_status"].tolist(), ["legal", "illegal_context"])
        self.assertEqual(scores["ranking_status"].tolist(), ["complete", "not_applicable"])
        self.assertEqual(tuple(json.loads(scores["ranking_b"].iloc[0])), tuple(sorted(GAIA_SERVICES)))
        self.assertEqual(scores["ranking_b"].iloc[1], "")
        failed = score_c1_test_episodes(
            episodes=episodes, index=_FailingIndex(), arm_b=_model(), arm_c=_model(),
            test_interval_ms=(0, 1200000))
        self.assertEqual(failed["scope_status"].iloc[0], "legal")
        self.assertEqual(failed["ranking_status"].iloc[0], "failed")
        self.assertEqual(failed["ranking_b"].iloc[0], "")
        with self.assertRaisesRegex(ValueError, "label-free"):
            score_c1_test_episodes(
                episodes=episodes.assign(trigger_label=1), index=_EmptyIndex(),
                arm_b=_model(), arm_c=_model(), test_interval_ms=(0, 1200000))


if __name__ == "__main__":
    unittest.main()
