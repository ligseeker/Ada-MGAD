import unittest

import pandas as pd

from src.e2e.e2e_evaluation import (
    anchor_metric_delta,
    delay_ranking_relationship,
    detector_only_rankings,
    diagnosis_metrics,
)
from src.e2e.protocol import GAIA_SERVICES


class DetectorOnlyTests(unittest.TestCase):
    def test_scores_descending_and_ties_use_registry_order(self):
        matching = pd.DataFrame([{
            "prediction_id": "p0", "t_hat": 30_000, "match_status": "matched",
            "case_id": "c0", "gt_service": "dbservice1",
        }])
        predictions = pd.DataFrame([
            {
                "split": "test", "prediction_timestamp": 30_000,
                "service": service, "anomaly_score": 0.9 if index in (1, 2) else 0.1,
            }
            for index, service in enumerate(GAIA_SERVICES)
        ])
        output = detector_only_rankings(matching, predictions)["p0"]
        self.assertEqual(output["ranking"][:2], ("dbservice2", "logservice1"))


class DiagnosisMetricTests(unittest.TestCase):
    def setUp(self):
        self.matching = pd.DataFrame([
            {"prediction_id": "p0", "match_status": "matched", "gt_service": "dbservice1",
             "detection_delay_seconds": -45.0},
            {"prediction_id": "p1", "match_status": "matched", "gt_service": "webservice2",
             "detection_delay_seconds": 15.0},
            {"prediction_id": "p2", "match_status": "false_alarm", "gt_service": None,
             "detection_delay_seconds": None},
            {"prediction_id": None, "match_status": "miss", "gt_service": "logservice1",
             "detection_delay_seconds": None},
        ])
        self.rankings = {
            "p0": GAIA_SERVICES,
            "p1": tuple(service for service in GAIA_SERVICES if service != "webservice2") + ("webservice2",),
        }

    def test_ranking_failure_counts_as_fp_and_fn(self):
        result = diagnosis_metrics(self.matching, self.rankings)
        # p0 succeeds @1; p1 is a ranking failure, plus one false alarm/miss.
        self.assertEqual(result["metrics"]["@1"]["diagnosis_true_positive"], 1)
        self.assertEqual(result["metrics"]["@1"]["diagnosis_false_positive"], 2)
        self.assertEqual(result["metrics"]["@1"]["diagnosis_false_negative"], 2)
        self.assertAlmostEqual(result["metrics"]["@1"]["f1"], 1.0 / 3.0)

    def test_delay_relationship_and_anchor_delta(self):
        relation = delay_ranking_relationship(self.matching, self.rankings)
        self.assertEqual(relation["matched_cases"], 2)
        self.assertEqual(relation["delay_bands"]["early_lt_-30s"]["case_count"], 1)
        delta = anchor_metric_delta(
            {"overall": {"AC@1": 1, "AC@3": 1, "AC@5": 1, "MRR": 1}},
            {"overall": {"AC@1": 0.5, "AC@3": 0.75, "AC@5": 1, "MRR": 0.6}},
        )
        self.assertEqual(delta["delta_detected_minus_oracle_AC@1"], -0.5)


if __name__ == "__main__":
    unittest.main()
