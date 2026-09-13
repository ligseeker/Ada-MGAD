import unittest

import pandas as pd

from src.e2e.event_detection import match_events, select_train_threshold
from src.e2e.protocol import GAIA_SERVICES


def scores(times, values):
    rows = []
    for timestamp, value in zip(times, values):
        for index, service in enumerate(GAIA_SERVICES):
            rows.append({
                "split": "train", "prediction_available_time": timestamp,
                "service": service, "anomaly_score": value if index == 0 else 0.0,
            })
    return pd.DataFrame(rows)


def ground_truth(rows):
    return pd.DataFrame(rows, columns=[
        "case_id", "source_index", "service", "fault_type", "start_ms", "end_ms", "split",
    ])


class V3EventProtocolTests(unittest.TestCase):
    def test_bin_end_is_used_as_t_hat(self):
        from src.e2e.event_detection import aggregate_system_scores, construct_predicted_episodes
        predictions = scores([30_000, 60_000], [0.1, 0.9])
        episodes = construct_predicted_episodes(aggregate_system_scores(predictions), 0.5)
        self.assertEqual(episodes.iloc[0]["t_hat"], 60_000)

    def test_future_gt_is_not_a_causal_match(self):
        result = match_events(
            pd.DataFrame([{"prediction_id": "p", "t_hat": 60_000}]),
            ground_truth([{
                "case_id": "g", "source_index": 0, "service": "dbservice1",
                "fault_type": "login_failure", "start_ms": 60_001, "end_ms": 60_002,
                "split": "test",
            }]),
        )
        self.assertEqual(result.iloc[0]["match_status"], "false_alarm")
        self.assertEqual(result.iloc[1]["match_status"], "miss")

    def test_maximum_cardinality_beats_greedy_nearest(self):
        result = match_events(
            pd.DataFrame([
                {"prediction_id": "p0", "t_hat": 60_000},
                {"prediction_id": "p1", "t_hat": 100_000},
            ]),
            ground_truth([
                {"case_id": "g0", "source_index": 0, "service": "dbservice1", "fault_type": "login_failure", "start_ms": 0, "end_ms": 1, "split": "test"},
                {"case_id": "g1", "source_index": 1, "service": "dbservice2", "fault_type": "login_failure", "start_ms": 50_000, "end_ms": 50_001, "split": "test"},
            ]),
        )
        self.assertEqual(result["match_status"].tolist(), ["matched", "matched"])
        self.assertEqual(result["case_id"].tolist(), ["g0", "g1"])

    def test_train_threshold_ignores_test_scores(self):
        train = scores([30_000, 60_000, 90_000], [0.1, 0.8, 0.1])
        gt = ground_truth([{
            "case_id": "g", "source_index": 0, "service": "dbservice1",
            "fault_type": "login_failure", "start_ms": 60_000, "end_ms": 60_001,
            "split": "train",
        }])
        first = select_train_threshold(train, gt)
        second = select_train_threshold(train, gt)
        self.assertEqual(first.threshold, second.threshold)
        self.assertEqual(first.train_metrics, second.train_metrics)
        self.assertEqual(first.validation_metrics, first.train_metrics)


if __name__ == "__main__":
    unittest.main()
