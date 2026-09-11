import unittest
from pathlib import Path
import tempfile

import numpy as np

from src.e2e.rca_model import (
    FEATURE_DIMENSION,
    N_CANDIDATES,
    fit_conditional_logit,
    fit_root_frequency,
    load_conditional_logit,
    oracle_rankings,
    rank_candidates,
    rca_metrics,
    save_conditional_logit,
)


SERVICES = tuple("s{}".format(i) for i in range(N_CANDIDATES))


class P5RcaModelTest(unittest.TestCase):
    def test_scaler_is_fit_on_train_candidate_rows_only(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(3, N_CANDIDATES, FEATURE_DIMENSION))
        # This validation-only shift must not change train-fitted scaler state.
        features[2] += 1e6
        roots = np.array([0, 1, 2])
        fit = fit_conditional_logit(features, roots, train_indices=[0, 1])
        train_rows = features[[0, 1]].reshape(-1, FEATURE_DIMENSION)
        np.testing.assert_allclose(fit.scaler_mean, train_rows.mean(axis=0))
        self.assertFalse(np.allclose(fit.scaler_mean, features.reshape(-1, FEATURE_DIMENSION).mean(axis=0)))

    def test_ranking_ties_use_service_name(self):
        self.assertEqual(
            rank_candidates(SERVICES, np.zeros(N_CANDIDATES)),
            tuple(sorted(SERVICES)),
        )

    def test_oracle_and_root_frequency_are_deterministic(self):
        roots = np.array([2, 2, 1, 2, 1, 9])
        oracle = oracle_rankings(roots, SERVICES)
        self.assertTrue(all(row[0] == SERVICES[root] for row, root in zip(oracle, roots)))
        baseline = fit_root_frequency(roots, train_indices=[0, 1, 2, 3], candidates=SERVICES)
        self.assertEqual(baseline.ranking[:3], ("s2", "s1", "s0"))

    def test_metrics_overall_root_macro_and_fault_macro(self):
        rankings = (
            ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"),
            ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"),
            ("s0", "s3", "s2", "s1", "s4", "s5", "s6", "s7", "s8", "s9"),
        )
        result = rca_metrics(rankings, [0, 0, 2], SERVICES, ["cpu", "cpu", "disk"])
        self.assertAlmostEqual(result["overall"]["AC@1"], 2.0 / 3.0)
        self.assertEqual(result["root_macro"]["group_count"], 2)
        self.assertEqual(result["fault_macro"]["group_count"], 2)
        self.assertAlmostEqual(result["case_metrics"][0]["Avg@5"], 1.0)
        self.assertAlmostEqual(result["case_metrics"][2]["MRR"], 1.0 / 3.0)

    def test_persisted_model_round_trip_checksums_and_scores(self):
        rng = np.random.default_rng(19)
        features = rng.normal(size=(3, N_CANDIDATES, FEATURE_DIMENSION))
        fit = fit_conditional_logit(features, [0, 1, 2], train_indices=[0, 1])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            save_conditional_logit(path, fit)
            loaded = load_conditional_logit(path)
            np.testing.assert_allclose(loaded.scores(features), fit.scores(features))
            self.assertEqual(loaded.train_case_indices, (0, 1))


if __name__ == "__main__":
    unittest.main()
