"""Tests for the RE2-TT protocol-extension stages (E3-E5).

Two things need pinning here. First, the extension reuses P1 code paths, so the
defaults that reproduce frozen P1 behaviour must not drift: the metric-change config
literal and ``_extract_re2ob``'s progress label are both checked against the values
recorded in ``artifacts/p1/baseline_summary.json``, restated as golden literals so
the test survives a fresh clone (artifacts are gitignored).

Second, the headroom gate is a pre-registered decision rule. Its boundary behaviour
is tested directly, because a silently loosened comparison would turn a benchmark
finding into a false green light.
"""

import inspect
import unittest

from scripts.audit_ext_re2tt_gates import (
    SATURATION_CEILING,
    _audit_headroom,
    _total_variation,
)
from scripts.run_ext_re2tt_baselines import _headroom_gate, _metric_change_config
from scripts.run_p1_metric_change import _extract_re2ob


# Copied out of artifacts/p1/baseline_summary.json; the RE2-OB B2 row that the
# extension is compared against was produced with exactly these settings.
FROZEN_METRIC_CHANGE_CONFIG = {
    "duplicate_timestamp_reduce": "mean of finite values",
    "feature_aggregation": "mean of top-k standardized feature shifts",
    "minimum_samples_per_side": 2,
    "missing_service_fallback": (
        "observed services first; missing services alphabetical last"
    ),
    "score_cap": 20.0,
    "score_scale": (
        "pooled within-window population standard deviation + relative epsilon"
    ),
    "top_k_features": 5,
    "window_semantics": "half-open [t0-window,t0) vs [t0,t0+window)",
    "window_seconds": 300,
}


class _Args:
    window_seconds = 300
    min_samples_per_side = 2
    top_k_features = 5
    score_cap = 20.0


def _macro(ac1, ac3, ac5, avg5, mrr=0.5):
    return {"AC@1": ac1, "AC@3": ac3, "AC@5": ac5, "Avg@5": avg5, "MRR": mrr}


def _summary(b1, b2):
    return {
        "metric_change": {"root_service_macro": b2},
        "root_frequency": {"root_service_macro": b1},
    }


def _audit_input(b1, b2):
    return {
        "baselines": {
            "metric_change": {"recomputed_root_service_macro": b2},
            "root_frequency": {"recomputed_root_service_macro": b1},
        }
    }


class FrozenReuseTest(unittest.TestCase):
    def test_metric_change_config_matches_the_frozen_p1_config(self):
        self.assertEqual(_metric_change_config(_Args()), FROZEN_METRIC_CHANGE_CONFIG)

    def test_extract_re2ob_progress_label_default_is_unchanged(self):
        signature = inspect.signature(_extract_re2ob)
        self.assertEqual(
            signature.parameters["progress_label"].default, "RE2-OB cases"
        )

    def test_saturation_ceiling_is_the_pre_registered_value(self):
        self.assertEqual(SATURATION_CEILING, 0.90)


class HeadroomGateTest(unittest.TestCase):
    B1 = _macro(0.166667, 0.555556, 1.0, 0.566667)

    def test_primary_endpoint_exactly_at_the_ceiling_passes(self):
        gate = _headroom_gate(
            _summary(self.B1, _macro(0.80, 0.90, 0.95, 0.90)), {}
        )
        self.assertEqual(gate["decision"], "pass")
        self.assertEqual(gate["failed_gates"], [])

    def test_primary_endpoint_just_above_the_ceiling_fails(self):
        gate = _headroom_gate(
            _summary(self.B1, _macro(0.80, 0.90, 0.95, 0.900001)), {}
        )
        self.assertEqual(gate["decision"], "fail")
        self.assertEqual(gate["failed_gates"], ["H-1"])

    def test_key_secondary_endpoint_above_the_ceiling_fails(self):
        gate = _headroom_gate(
            _summary(self.B1, _macro(0.95, 0.96, 0.97, 0.88)), {}
        )
        self.assertEqual(gate["failed_gates"], ["H-2"])

    def test_b2_must_strictly_beat_b1_on_both_endpoints(self):
        # Avg@5 improves but AC@1 only ties, so H-3 must not pass.
        gate = _headroom_gate(
            _summary(self.B1, _macro(0.166667, 0.60, 0.90, 0.70)), {}
        )
        self.assertIn("H-3", gate["failed_gates"])

    def test_b1_ac_at_5_is_reported_and_never_gates(self):
        gate = _headroom_gate(
            _summary(self.B1, _macro(0.80, 0.90, 0.95, 0.85)), {}
        )
        h4 = next(check for check in gate["checks"] if check["id"] == "H-4")
        self.assertEqual(h4["kind"], "report_only")
        self.assertEqual(h4["observed"], 1.0)
        self.assertEqual(gate["gate_count"], 3)
        self.assertEqual(gate["decision"], "pass")

    def test_audit_reaches_the_same_verdict_as_the_run_script(self):
        b2 = _macro(0.822222, 0.922222, 0.944444, 0.904444)
        run_gate = _headroom_gate(_summary(self.B1, b2), {})
        audit_gate = _audit_headroom(_audit_input(self.B1, b2), _NoPath())
        self.assertEqual(run_gate["decision"], audit_gate["decision"])
        self.assertEqual(run_gate["failed_gates"], audit_gate["failed_gates"])
        self.assertAlmostEqual(
            audit_gate["headroom_above_b2_primary"], 1.0 - b2["Avg@5"]
        )


class _NoPath:
    def is_file(self):
        return False


class TotalVariationTest(unittest.TestCase):
    def test_identical_distributions_have_zero_distance(self):
        self.assertEqual(
            _total_variation({"a": 3, "b": 3}, {"a": 0.5, "b": 0.5}), 0.0
        )

    def test_missing_stratum_is_penalized(self):
        self.assertAlmostEqual(
            _total_variation({"a": 6}, {"a": 0.5, "b": 0.5}), 0.5
        )

    def test_empty_observation_is_maximally_distant(self):
        self.assertEqual(_total_variation({}, {"a": 1.0}), 1.0)


if __name__ == "__main__":
    unittest.main()
