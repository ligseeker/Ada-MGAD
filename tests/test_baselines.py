import unittest

import numpy as np

from src.baselines import (
    deterministic_random_ranking,
    fit_root_frequency,
    metric_change_prediction,
    window_mean_shift_scores,
)
from src.data.schema import RCACaseInput, RCACaseLabel
from src.evaluation import evaluate_ranking_report, validate_ranking


class SanityBaselineTest(unittest.TestCase):
    def setUp(self):
        self.services = ("svc-a", "svc-b", "svc-c")
        self.cases = (
            RCACaseInput("case-1", "toy", 1, self.services),
            RCACaseInput("case-2", "toy", 2, self.services),
            RCACaseInput("case-3", "toy", 3, self.services),
        )

    def test_random_ranking_is_seeded_complete_and_label_free(self):
        first = deterministic_random_ranking(self.cases[0], seed=7)
        second = deterministic_random_ranking(self.cases[0], seed=7)
        self.assertEqual(first, second)
        validate_ranking(self.cases[0], first)
        seeds = {
            deterministic_random_ranking(self.cases[0], seed=seed)
            for seed in range(10)
        }
        self.assertGreater(len(seeds), 1)

    def test_frequency_uses_only_supplied_training_labels(self):
        model = fit_root_frequency(
            (
                RCACaseLabel("train-1", "svc-b"),
                RCACaseLabel("train-2", "svc-b"),
                RCACaseLabel("train-3", "svc-a"),
            )
        )
        self.assertEqual(model.training_case_count, 3)
        self.assertEqual(model.rank(self.cases[0]), ("svc-b", "svc-a", "svc-c"))

    def test_frequency_ties_are_lexicographic(self):
        model = fit_root_frequency((RCACaseLabel("train", "svc-c"),))
        self.assertEqual(model.rank(self.cases[0]), ("svc-c", "svc-a", "svc-b"))

    def test_reporting_includes_overall_and_unweighted_macros(self):
        labels = (
            RCACaseLabel("case-1", "svc-a", "cpu"),
            RCACaseLabel("case-2", "svc-a", "cpu"),
            RCACaseLabel("case-3", "svc-c", "mem"),
        )
        rankings = {case.case_id: self.services for case in self.cases}
        report = evaluate_ranking_report(self.cases, labels, rankings)
        self.assertAlmostEqual(report["overall"]["AC@1"], 2.0 / 3.0)
        self.assertAlmostEqual(report["fault_type"]["macro"]["AC@1"], 0.5)
        self.assertEqual(report["root_service"]["category_count"], 2)

    def test_metric_change_uses_half_open_windows_and_reduces_duplicates(self):
        timestamps = (0, 1000, 1000, 2000, 3000, 4000, 5000)
        values = (0.0, 0.0, 2.0, 0.0, 10.0, 10.0, 10.0)
        scores = window_mean_shift_scores(
            timestamps,
            values,
            anchors_ms=(3000,),
            radius_ms=3000,
            min_samples_per_side=2,
        )
        self.assertTrue(np.isfinite(scores[0]))
        self.assertGreater(scores[0], 1.0)

    def test_metric_change_missing_services_use_alphabetical_fallback(self):
        prediction = metric_change_prediction(
            self.cases[0],
            {"svc-a": 0.2, "svc-b": 3.0},
            {"svc-a": 1, "svc-b": 2, "svc-c": 0},
        )
        self.assertEqual(prediction.ranking, ("svc-b", "svc-a", "svc-c"))
        self.assertEqual(prediction.fallback_services, ("svc-c",))

    def test_metric_change_requires_coverage_on_both_sides(self):
        scores = window_mean_shift_scores(
            timestamps_ms=(0, 1000, 3000),
            values=(0.0, 0.0, 10.0),
            anchors_ms=(3000,),
            radius_ms=3000,
            min_samples_per_side=2,
        )
        self.assertTrue(np.isnan(scores[0]))


if __name__ == "__main__":
    unittest.main()
