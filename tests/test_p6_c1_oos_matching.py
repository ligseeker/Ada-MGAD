"""C1 OOS matching keeps causal misses, false alarms, and context exclusions."""

import unittest

import pandas as pd

from src.e2e.c1_oos_matching import match_c1_oos_episodes


class C1OOSMatchingTest(unittest.TestCase):
    def test_complete_generation_gt_and_dual_context_filter(self):
        episodes = pd.DataFrame([
            {"fold": 1, "prediction_id": "fold_01-generation-pred-000000",
             "split": "generation", "t_hat": 300000, "episode_end_time": 330000},
            {"fold": 1, "prediction_id": "fold_01-generation-pred-000001",
             "split": "generation", "t_hat": 450000, "episode_end_time": 480000},
            {"fold": 1, "prediction_id": "fold_01-generation-pred-000002",
             "split": "generation", "t_hat": 630000, "episode_end_time": 660000},
        ])
        registry = pd.DataFrame([
            {"case_id": "early", "source_index": 1, "service": "dbservice1",
             "fault_type": "x", "start_ms": 270000, "end_ms": 330000, "detector_domain": True},
            {"case_id": "legal", "source_index": 2, "service": "dbservice1",
             "fault_type": "x", "start_ms": 600000, "end_ms": 660000, "detector_domain": True},
            {"case_id": "miss", "source_index": 3, "service": "dbservice1",
             "fault_type": "x", "start_ms": 900000, "end_ms": 960000, "detector_domain": True},
            {"case_id": "boundary", "source_index": 4, "service": "dbservice1",
             "fault_type": "x", "start_ms": 1140000, "end_ms": 1230000, "detector_domain": True},
        ])
        result = match_c1_oos_episodes(
            fold=1, interval_ms=(0, 1200000), episodes=episodes, registry=registry)
        self.assertEqual(result.audit["matched"], 2)
        self.assertEqual(result.audit["false_alarms"], 1)
        self.assertEqual(result.audit["misses"], 1)
        self.assertEqual(result.audit["cross_boundary_gt"], 1)
        self.assertEqual(result.cohort_candidates["case_id"].tolist(), ["legal"])
        self.assertEqual(set(result.exclusion_ledger["case_id"]), {"early", "boundary"})
        self.assertEqual(len(result.matching), 4)
        with self.assertRaisesRegex(ValueError, "GT fields"):
            match_c1_oos_episodes(fold=1, interval_ms=(0, 1200000),
                                  episodes=episodes.assign(trigger_label=0), registry=registry)


if __name__ == "__main__":
    unittest.main()
