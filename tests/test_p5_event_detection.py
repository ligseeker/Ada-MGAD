import unittest

import numpy as np
import pandas as pd

from src.e2e.event_detection import (
    aggregate_system_scores,
    construct_predicted_episodes,
    event_metrics,
    match_events,
    run_event_detection,
    select_validation_threshold,
)
from src.e2e.protocol import GAIA_SERVICES


def prediction_rows(timestamps, scores, split="test"):
    rows = []
    for timestamp, score in zip(timestamps, scores):
        for service_index, service in enumerate(GAIA_SERVICES):
            value = float(score) if service_index == 0 else 0.0
            rows.append({
                "split": split,
                "prediction_timestamp": int(timestamp),
                "service": service,
                "anomaly_score": value,
            })
    return pd.DataFrame(rows)


def gt_rows(records):
    return pd.DataFrame(records, columns=[
        "case_id", "source_index", "service", "fault_type", "start_ms", "end_ms", "split"
    ])


class TriggerTests(unittest.TestCase):
    def test_system_score_is_max_and_episode_uses_first_bin(self):
        timestamps = np.arange(6, dtype=np.int64) * 30_000
        predictions = prediction_rows(timestamps, [0.1, 0.8, 0.9, 0.7, 0.1, 0.8])
        scores = aggregate_system_scores(predictions, split="test")
        self.assertEqual(scores["system_score"].tolist(), [0.1, 0.8, 0.9, 0.7, 0.1, 0.8])
        episodes = construct_predicted_episodes(scores, 0.5)
        self.assertEqual(episodes["t_hat"].tolist(), [30_000, 150_000])
        self.assertEqual(episodes["positive_bins"].tolist(), [3, 1])
        self.assertEqual(episodes["episode_end_time"].tolist(), [120_000, 180_000])

    def test_system_score_requires_complete_service_registry(self):
        incomplete = prediction_rows([0], [0.8]).iloc[:-1]
        with self.assertRaisesRegex(ValueError, "all canonical GAIA services"):
            aggregate_system_scores(incomplete)

    def test_gap_breaks_episode_even_if_rows_are_positive(self):
        predictions = prediction_rows([0, 30_000, 90_000], [0.8, 0.8, 0.8])
        episodes = construct_predicted_episodes(aggregate_system_scores(predictions), 0.5)
        self.assertEqual(episodes["t_hat"].tolist(), [0, 90_000])


class MatchingTests(unittest.TestCase):
    def test_unique_match_false_alarm_miss_and_one_to_one(self):
        episodes = pd.DataFrame([
            {"prediction_id": "p0", "t_hat": 100_000, "split": "test"},
            {"prediction_id": "p1", "t_hat": 250_000, "split": "test"},
            {"prediction_id": "p2", "t_hat": 500_000, "split": "test"},
        ])
        ground_truth = gt_rows([
            {"case_id": "early", "source_index": 1, "service": "dbservice1", "fault_type": "login failure", "start_ms": 40_000, "end_ms": 50_000, "split": "test"},
            {"case_id": "late", "source_index": 0, "service": "dbservice2", "fault_type": "memory_anomalies", "start_ms": 160_000, "end_ms": 170_000, "split": "test"},
            {"case_id": "miss", "source_index": 2, "service": "webservice1", "fault_type": "cpu_anomalies", "start_ms": 900_000, "end_ms": 910_000, "split": "test"},
        ])
        matching = match_events(episodes, ground_truth, tolerance_seconds=60)
        self.assertEqual(matching["match_status"].tolist(), ["matched", "false_alarm", "false_alarm", "miss", "miss"])
        self.assertEqual(matching.iloc[0]["case_id"], "early")
        self.assertAlmostEqual(float(matching.iloc[0]["detection_delay_seconds"]), 60.0)
        self.assertEqual(matching["case_id"].tolist(), ["early", None, None, "late", "miss"])

    def test_equal_distance_tie_uses_earlier_start_then_source_then_case(self):
        episodes = pd.DataFrame([{"prediction_id": "p0", "t_hat": 100_000}])
        ground_truth = gt_rows([
            {"case_id": "later", "source_index": 0, "service": "dbservice1", "fault_type": "login failure", "start_ms": 160_000, "end_ms": 170_000, "split": "test"},
            {"case_id": "earlier", "source_index": 9, "service": "dbservice2", "fault_type": "login failure", "start_ms": 40_000, "end_ms": 50_000, "split": "test"},
        ])
        matching = match_events(episodes, ground_truth, tolerance_seconds=60)
        self.assertEqual(matching.iloc[0]["case_id"], "earlier")

    def test_event_delay_percentiles(self):
        matching = pd.DataFrame({
            "match_status": ["matched", "matched", "matched", "false_alarm", "miss"],
            "detection_delay_seconds": [-30.0, 0.0, 60.0, np.nan, np.nan],
        })
        metrics = event_metrics(matching)
        self.assertEqual(metrics["true_positive_events"], 3)
        self.assertEqual(metrics["false_positive_events"], 1)
        self.assertEqual(metrics["false_negative_events"], 1)
        self.assertAlmostEqual(metrics["event_precision"], 0.75)
        self.assertAlmostEqual(metrics["event_recall"], 0.75)
        self.assertAlmostEqual(metrics["detection_delay"]["mean_seconds"], 10.0)
        self.assertEqual(metrics["detection_delay"]["median_seconds"], 0.0)


class ThresholdTests(unittest.TestCase):
    def test_threshold_is_selected_from_validation_only(self):
        timestamps = np.arange(5, dtype=np.int64) * 30_000
        validation = prediction_rows(timestamps, [0.1, 0.8, 0.8, 0.1, 0.2], split="validation")
        validation_gt = gt_rows([
            {"case_id": "v", "source_index": 0, "service": "dbservice1", "fault_type": "login failure", "start_ms": 30_000, "end_ms": 60_000, "split": "validation"},
        ])
        selection = select_validation_threshold(validation, validation_gt)
        self.assertAlmostEqual(selection.threshold, 0.8)
        self.assertEqual(selection.validation_metrics["event_f1"], 1.0)

    def test_run_uses_split_gt_and_never_test_scores_for_threshold(self):
        timestamps = np.arange(5, dtype=np.int64) * 30_000
        validation = prediction_rows(timestamps, [0.1, 0.8, 0.8, 0.1, 0.2], split="validation")
        test_a = prediction_rows(timestamps, [0.1, 0.8, 0.8, 0.1, 0.2], split="test")
        test_b = prediction_rows(timestamps, [0.99, 0.99, 0.99, 0.99, 0.99], split="test")
        registry = gt_rows([
            {"case_id": "v", "source_index": 0, "service": "dbservice1", "fault_type": "login failure", "start_ms": 30_000, "end_ms": 60_000, "split": "validation"},
            {"case_id": "t", "source_index": 1, "service": "dbservice2", "fault_type": "login failure", "start_ms": 90_000, "end_ms": 120_000, "split": "test"},
        ])
        first = run_event_detection(validation, test_a, registry, ())
        second = run_event_detection(validation, test_b, registry, ())
        self.assertEqual(first["threshold_selection"].threshold, second["threshold_selection"].threshold)


if __name__ == "__main__":
    unittest.main()
