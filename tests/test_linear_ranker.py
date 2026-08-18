import unittest

import numpy as np

from src.data import RCACaseLabel
from src.models import (
    IndependentLinearRanker,
    LinearRankerError,
    case_balanced_targets,
    rank_service_scores,
    select_feature_columns,
)


class LinearRankerTest(unittest.TestCase):
    def setUp(self):
        self.index = tuple(
            {"case_id": case_id, "extractor": "toy", "service": service}
            for case_id in ("case-a", "case-b")
            for service in ("svc-a", "svc-b", "svc-c")
        )
        self.labels = (
            RCACaseLabel("case-a", "svc-a", "cpu"),
            RCACaseLabel("case-b", "svc-c", "memory"),
        )

    def test_case_balanced_targets(self):
        targets, weights = case_balanced_targets(self.index, self.labels)
        self.assertEqual(targets.tolist(), [1, 0, 0, 0, 0, 1])
        for start in (0, 3):
            self.assertAlmostEqual(weights[start : start + 3].sum(), 1.0)
            self.assertAlmostEqual(
                weights[start : start + 3][targets[start : start + 3] == 1].sum(),
                0.5,
            )

    def test_selection_ranking_and_tie_break(self):
        self.assertEqual(
            select_feature_columns(("whole.a", "stage60.a", "whole.b"), ("whole.",)),
            (0, 2),
        )
        rankings = rank_service_scores(self.index, (2, 1, 0, 0, 1, 1))
        self.assertEqual(rankings["case-a"], ("svc-a", "svc-b", "svc-c"))
        self.assertEqual(rankings["case-b"], ("svc-b", "svc-c", "svc-a"))
        with self.assertRaises(LinearRankerError):
            rank_service_scores(self.index, (float("nan"),) * 6)

    def test_fit_is_deterministic_and_auditable(self):
        targets, weights = case_balanced_targets(self.index, self.labels)
        features = np.asarray(
            [[2, 1], [0, 1], [-1, 1], [-1, 0], [0, 0], [2, 0]],
            dtype=np.float64,
        )
        first = IndependentLinearRanker(0.1).fit(features, targets, weights)
        second = IndependentLinearRanker(0.1).fit(features, targets, weights)
        np.testing.assert_allclose(first.score(features), second.score(features))
        self.assertEqual(first.audit(), second.audit())


if __name__ == "__main__":
    unittest.main()
