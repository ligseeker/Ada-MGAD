import unittest

from scripts.bootstrap_p2_m1_s import (
    COMPARATOR,
    METHOD,
    SECONDARY_GUARDRAIL,
    decide_gate,
)


def _dataset_result(primary_point, primary_lower, secondary_point):
    return {
        "metric_results": {
            "AC@1": {
                "ci95_lower": secondary_point - 0.05,
                "ci95_upper": secondary_point + 0.05,
                "point_delta": secondary_point,
            },
            "Avg@5": {
                "ci95_lower": primary_lower,
                "ci95_upper": primary_point + 0.05,
                "point_delta": primary_point,
            },
        }
    }


def _results(gaia, re2ob):
    return {"gaia_main": gaia, "re2ob": re2ob}


class StageBootstrapGateTest(unittest.TestCase):
    def test_targets_m1_s_against_c1_i(self):
        self.assertEqual(METHOD, "m1_s")
        self.assertEqual(COMPARATOR, "c1_i")
        self.assertEqual(SECONDARY_GUARDRAIL, -0.01)

    def test_positive_cis_and_intact_secondary_are_claim_ready(self):
        gate = decide_gate(
            _results(
                _dataset_result(0.03, 0.01, 0.02),
                _dataset_result(0.02, 0.005, 0.0),
            )
        )
        self.assertTrue(gate["exploratory_signal"])
        self.assertTrue(gate["claim_ready"])
        self.assertEqual(gate["p2_g4_decision"], "claim-ready")
        self.assertTrue(
            gate["claim_ready_checks"]["both_primary_ci_lower_bounds_positive"]
        )
        self.assertTrue(
            gate["claim_ready_checks"][
                "both_secondary_point_deltas_at_least_minus_0_01"
            ]
        )

    def test_one_ci_touching_zero_is_exploratory_only(self):
        gate = decide_gate(
            _results(
                _dataset_result(0.03, -0.004, 0.0),
                _dataset_result(0.02, 0.005, 0.0),
            )
        )
        self.assertTrue(gate["exploratory_signal"])
        self.assertFalse(gate["claim_ready"])
        self.assertEqual(gate["p2_g4_decision"], "exploratory-signal-only")
        self.assertFalse(
            gate["claim_ready_checks"]["both_primary_ci_lower_bounds_positive"]
        )

    def test_a_ci_lower_bound_of_exactly_zero_is_not_claim_ready(self):
        gate = decide_gate(
            _results(
                _dataset_result(0.03, 0.0, 0.0),
                _dataset_result(0.02, 0.005, 0.0),
            )
        )
        self.assertFalse(gate["claim_ready"])
        self.assertEqual(gate["p2_g4_decision"], "exploratory-signal-only")

    def test_a_nonpositive_primary_point_delta_is_no_go(self):
        gate = decide_gate(
            _results(
                _dataset_result(-0.01, -0.05, 0.0),
                _dataset_result(0.02, 0.005, 0.0),
            )
        )
        self.assertFalse(gate["exploratory_signal"])
        self.assertFalse(gate["claim_ready"])
        self.assertEqual(gate["p2_g4_decision"], "no-go")

    def test_a_zero_primary_point_delta_is_no_go(self):
        gate = decide_gate(
            _results(
                _dataset_result(0.0, -0.02, 0.0),
                _dataset_result(0.02, 0.005, 0.0),
            )
        )
        self.assertFalse(gate["exploratory_signal"])
        self.assertEqual(gate["p2_g4_decision"], "no-go")

    def test_secondary_regression_beyond_the_guardrail_blocks_claim_ready(self):
        gate = decide_gate(
            _results(
                _dataset_result(0.03, 0.01, -0.02),
                _dataset_result(0.02, 0.005, 0.0),
            )
        )
        self.assertTrue(gate["exploratory_signal"])
        self.assertFalse(gate["claim_ready"])
        self.assertEqual(gate["p2_g4_decision"], "exploratory-signal-only")
        self.assertTrue(
            gate["claim_ready_checks"]["both_primary_ci_lower_bounds_positive"]
        )
        self.assertFalse(
            gate["claim_ready_checks"][
                "both_secondary_point_deltas_at_least_minus_0_01"
            ]
        )

    def test_secondary_regression_exactly_at_the_guardrail_is_allowed(self):
        gate = decide_gate(
            _results(
                _dataset_result(0.03, 0.01, -0.01),
                _dataset_result(0.02, 0.005, 0.0),
            )
        )
        self.assertTrue(gate["claim_ready"])
        self.assertEqual(gate["p2_g4_decision"], "claim-ready")

    def test_gate_requires_at_least_one_dataset(self):
        with self.assertRaisesRegex(ValueError, "at least one dataset"):
            decide_gate({})


if __name__ == "__main__":
    unittest.main()
