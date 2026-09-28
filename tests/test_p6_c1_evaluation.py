"""C1 post-lock evaluator preserves paired denominator and false alarms."""

import json
import unittest

import numpy as np
import pandas as pd

from src.e2e.c1_evaluation import evaluate_c1_prediction_lock, evaluate_c1_oracle_a, evaluate_c2_raw_failure
from src.e2e.rca_model import ConditionalLogitFit, GAIA_SERVICES


class _EmptyIndex:
    def case_indicators(self, anchor_ms, spec):
        return {}


class C1EvaluationTest(unittest.TestCase):
    def test_paired_transitions_include_missing_ranking_as_failure(self):
        names = list(GAIA_SERVICES)
        root = names[0]
        correct = json.dumps(names)
        wrong = json.dumps(names[1:] + names[:1])
        scope = pd.DataFrame([
            {"prediction_id": "p0", "t_hat": 600000, "scope_status": "legal",
             "ranking_status": "complete", "ranking_b": correct, "ranking_c": wrong},
            {"prediction_id": "p1", "t_hat": 660000, "scope_status": "legal",
             "ranking_status": "complete", "ranking_b": wrong, "ranking_c": correct},
            {"prediction_id": "p2", "t_hat": 720000, "scope_status": "legal",
             "ranking_status": "failed", "ranking_b": "", "ranking_c": correct},
            {"prediction_id": "p3", "t_hat": 780000, "scope_status": "legal",
             "ranking_status": "failed", "ranking_b": "", "ranking_c": ""},
        ])
        matching = pd.DataFrame([
            {"prediction_id": "p0", "t_hat": 600000, "match_status": "matched",
             "case_id": "c0", "gt_service": root, "fault_type": "login_failure", "gt_start_ms": 570000},
            {"prediction_id": "p1", "t_hat": 660000, "match_status": "matched",
             "case_id": "c1", "gt_service": root, "fault_type": "login_failure", "gt_start_ms": 630000},
            {"prediction_id": "p2", "t_hat": 720000, "match_status": "matched",
             "case_id": "c2", "gt_service": root, "fault_type": "login_failure", "gt_start_ms": 690000},
            {"prediction_id": "p3", "t_hat": 780000, "match_status": "false_alarm",
             "case_id": None, "gt_service": None, "fault_type": None, "gt_start_ms": None},
        ])
        result = evaluate_c1_prediction_lock(scope=scope, matching=matching)
        self.assertEqual(result["primary"]["n"], 3)
        self.assertEqual(result["primary"]["correct_b"], 1)
        self.assertEqual(result["primary"]["correct_c"], 2)
        self.assertEqual(result["primary"]["delta_c_minus_b"], 1 / 3)
        self.assertEqual(result["primary"]["paired_transitions"], {
            "both_correct": 0, "b_only_correct": 1,
            "c_only_correct": 2, "both_incorrect": 0})
        self.assertEqual(result["scope"]["false_alarms"], 1)
        self.assertIn(root, result["root_strata"]["small_descriptive_groups"])

    def test_c2_raw_gt_includes_misses_and_unmeasured_final_latency(self):
        episodes = pd.DataFrame([
            {"prediction_id": "p0", "split": "test", "t_hat": 600000,
             "episode_end_time": 630000, "positive_bins": 1,
             "system_score": 0.9, "threshold": 0.5},
            {"prediction_id": "p1", "split": "test", "t_hat": 900000,
             "episode_end_time": 930000, "positive_bins": 1,
             "system_score": 0.8, "threshold": 0.5},
        ])
        scope = pd.DataFrame([
            {"prediction_id": "p0", "t_hat": 600000, "scope_status": "legal",
             "ranking_status": "complete", "ranking_c": json.dumps(list(GAIA_SERVICES))},
            {"prediction_id": "p1", "t_hat": 900000, "scope_status": "legal",
             "ranking_status": "complete", "ranking_c": json.dumps(list(GAIA_SERVICES))},
        ])
        registry = pd.DataFrame([
            {"case_id": "c0", "source_index": 0, "service": GAIA_SERVICES[0],
             "fault_type": "x", "start_ms": 570000, "end_ms": 630000},
            {"case_id": "c1", "source_index": 1, "service": GAIA_SERVICES[1],
             "fault_type": "x", "start_ms": 750000, "end_ms": 810000},
            {"case_id": "boundary", "source_index": 2, "service": GAIA_SERVICES[1],
             "fault_type": "x", "start_ms": -30000, "end_ms": 30000},
        ])
        summary, raw_matching = evaluate_c2_raw_failure(
            episodes=episodes, scope=scope, registry=registry,
            test_interval_ms=(0, 1200000))
        self.assertEqual(summary["raw_gt_complete"], 2)
        self.assertEqual(summary["raw_gt_cross_boundary"], 1)
        self.assertEqual(summary["diagnosis_at_1_all_raw_gt_denominator"], 3)
        self.assertEqual(summary["misses"], 1)
        self.assertEqual(summary["false_alarms"], 1)
        self.assertEqual(summary["matched_diagnosis"]["correct"], 1)
        self.assertIsNone(summary["final_diagnosis_wall_time"])
        self.assertEqual(len(raw_matching), 4)
        self.assertEqual(set(raw_matching["diagnosis_status"]),
                         {"correct", "detector_miss", "false_alarm", "boundary_gt_excluded"})

    def test_oracle_a_is_post_lock_same_denominator_diagnostic(self):
        model = ConditionalLogitFit(
            weights=np.zeros(68), scaler_mean=np.zeros(68), scaler_scale=np.ones(68),
            train_case_indices=(0,), initial_loss=0, final_loss=0,
            gradient_norm=0, iterations=0, converged=True, message="test")
        rows, summary = evaluate_c1_oracle_a(
            locked_case_details=[{"prediction_id": "p0", "case_id": "c0",
                                  "gt_start_ms": 600000, "gt_service": GAIA_SERVICES[0]}],
            index=_EmptyIndex(), arm_b=model, test_interval_ms=(0, 1200000))
        self.assertEqual(summary["n_same_as_bc"], 1)
        self.assertEqual(summary["AC@1"], 1.0)
        self.assertTrue(summary["diagnostic_only"])
        self.assertEqual(rows["status"].iloc[0], "complete")


if __name__ == "__main__":
    unittest.main()
