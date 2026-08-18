import unittest

from src.data.schema import RCACaseInput, RCACaseLabel, SchemaValidationError
from src.evaluation.evaluator import (
    evaluate_rankings,
    predict_rankings,
    validate_ranking,
)
from src.evaluation.metrics import average_at_k, hit_at_k, reciprocal_rank


class RankingMetricsTest(unittest.TestCase):
    def setUp(self):
        self.services = ("svc-a", "svc-b", "svc-c", "svc-d", "svc-e", "svc-f")
        self.inputs = [
            RCACaseInput("case-1", "toy", 1, self.services),
            RCACaseInput("case-2", "toy", 2, self.services),
            RCACaseInput("case-3", "toy", 3, self.services),
        ]
        self.labels = [
            RCACaseLabel("case-1", "svc-a"),
            RCACaseLabel("case-2", "svc-b"),
            RCACaseLabel("case-3", "svc-d"),
        ]
        self.rankings = {
            "case-1": self.services,
            "case-2": self.services,
            "case-3": self.services,
        }

    def test_individual_metrics(self):
        ranking = ("a", "b", "c", "d")
        self.assertEqual(hit_at_k(ranking, "c", 2), 0.0)
        self.assertEqual(hit_at_k(ranking, "c", 3), 1.0)
        self.assertAlmostEqual(average_at_k(ranking, "c", 5), 3.0 / 5.0)
        self.assertAlmostEqual(reciprocal_rank(ranking, "c"), 1.0 / 3.0)

    def test_dataset_metrics_for_known_root_ranks(self):
        result = evaluate_rankings(self.inputs, self.labels, self.rankings)
        self.assertEqual(result.total_cases, 3)
        self.assertAlmostEqual(result.ac_at[1], 1.0 / 3.0)
        self.assertAlmostEqual(result.ac_at[3], 2.0 / 3.0)
        self.assertAlmostEqual(result.ac_at[5], 1.0)
        self.assertAlmostEqual(result.avg_at_5, 11.0 / 15.0)
        self.assertAlmostEqual(result.mrr, 7.0 / 12.0)
        self.assertEqual([row.root_rank for row in result.per_case], [1, 2, 4])

    def test_ranking_must_be_a_complete_permutation(self):
        case_input = self.inputs[0]
        with self.assertRaisesRegex(SchemaValidationError, "duplicate"):
            validate_ranking(
                case_input,
                ("svc-a", "svc-b", "svc-c", "svc-d", "svc-e", "svc-e"),
            )
        with self.assertRaisesRegex(SchemaValidationError, "missing"):
            validate_ranking(case_input, self.services[:-1])
        with self.assertRaisesRegex(SchemaValidationError, "extra"):
            validate_ranking(case_input, self.services + ("svc-x",))

    def test_missing_root_is_rejected_before_scoring(self):
        bad_label = list(self.labels)
        bad_label[0] = RCACaseLabel("case-1", "not-observed")
        with self.assertRaisesRegex(SchemaValidationError, "not in services"):
            evaluate_rankings(self.inputs, bad_label, self.rankings)

    def test_prediction_boundary_does_not_expose_labels(self):
        def leaky_predictor(case_input):
            # A model that attempts this shortcut must fail: labels are not a
            # property of RCACaseInput.
            return (case_input.root_service,)  # type: ignore[attr-defined]

        with self.assertRaises(AttributeError):
            predict_rankings([self.inputs[0]], leaky_predictor)

        predictions = predict_rankings(
            [self.inputs[0]], lambda case_input: case_input.services
        )
        self.assertEqual(predictions["case-1"], self.services)


if __name__ == "__main__":
    unittest.main()
