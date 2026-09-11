import unittest

import numpy as np

from src.e2e.rca_features import (
    CHANNELS,
    TemporalSpec,
    extract_case_features_from_indicators,
    feature_health,
    flatten_features,
)


class TemporalRepresentationTests(unittest.TestCase):
    def test_default_is_w300_b15_and_z2_is_68d(self):
        spec = TemporalSpec()
        self.assertEqual(spec.window_seconds, 300)
        self.assertEqual(spec.bin_seconds, 15)
        self.assertEqual(spec.pre_bins, 20)
        self.assertEqual(spec.post_bins, 20)
        self.assertEqual(spec.n_bins, 40)
        self.assertEqual(spec.onset_sentinel, 300.0)
        self.assertEqual(spec.post_position_denominator, 19.0)

        pre = np.arange(20, dtype=float)
        post = np.zeros(20, dtype=float)
        post[2:4] = 100.0
        post[-1] = 200.0
        series = np.concatenate((pre, post))
        indicators = {
            channel: {"svc": series.copy()} for channel in CHANNELS
        }
        # One unobserved bin must remain represented by the mask, not as a
        # non-finite feature value.
        indicators["metric"]["svc"][25] = np.nan

        features = extract_case_features_from_indicators(
            "tiny-case", ("svc",), indicators, spec
        )
        self.assertEqual(features.base.shape, (1, 4, 8))
        self.assertEqual(features.z2.shape, (1, 4, 9))
        self.assertEqual(flatten_features(features, "z2").shape, (1, 68))
        self.assertTrue(all(np.all(np.isfinite(array)) for array in (
            features.a, features.base, features.z, features.q,
            features.q_mask, features.morphology_active, features.z2, features.z3,
        )))
        self.assertEqual(features.q_mask.shape, (1, 4, 40))
        self.assertEqual(features.q_mask[0, 0, 25], 0.0)

    def test_missing_onset_uses_300_and_post_peak_uses_denominator_19(self):
        pre = np.arange(20, dtype=float)
        post = np.zeros(20, dtype=float)
        series = np.concatenate((pre, post))
        indicators = {channel: {"svc": series.copy()} for channel in CHANNELS}
        features = extract_case_features_from_indicators(
            "sentinel-case", ("svc",), indicators
        )
        # All post values are observed but below the threshold, so this is an
        # available channel with a missing onset rather than an unavailable one.
        self.assertEqual(features.base[0, 0, 3], 300.0)
        self.assertEqual(features.base[0, 0, 4], 1.0)
        # Make the final post bin the unique peak.  Its post position is 19,
        # hence the frozen normalized peak-time fraction is exactly one.
        series[-1] = 200.0
        indicators = {channel: {"svc": series.copy()} for channel in CHANNELS}
        features = extract_case_features_from_indicators(
            "peak-case", ("svc",), indicators
        )
        self.assertAlmostEqual(features.z2[0, 0, 3], 19.0 / 19.0)

    def test_service_matrix_and_flat_mapping_are_equivalent(self):
        pre = np.arange(20, dtype=float)
        post = np.linspace(0.0, 50.0, 20)
        series = np.concatenate((pre, post))
        matrix_mapping = {
            channel: {"svc": np.asarray([series])} for channel in CHANNELS
        }
        flat_mapping = {
            channel: {"svc::indicator": series.copy()} for channel in CHANNELS
        }
        matrix = extract_case_features_from_indicators("matrix", ("svc",), matrix_mapping)
        flat = extract_case_features_from_indicators("flat", ("svc",), flat_mapping)
        np.testing.assert_array_equal(matrix.base, flat.base)
        np.testing.assert_array_equal(matrix.z2, flat.z2)
        np.testing.assert_array_equal(matrix.q_mask, flat.q_mask)

        health = feature_health((matrix, flat))
        self.assertEqual(health["feature_dimension"], 68)
        self.assertEqual(health["finite_ratio"], 1.0)
        self.assertEqual(health["candidate_rows"], 2)

    def test_temporal_constants_cannot_drift_independently(self):
        with self.assertRaisesRegex(ValueError, "post_bins - 1"):
            TemporalSpec(post_position_denominator=20)
        with self.assertRaisesRegex(ValueError, "post-window horizon"):
            TemporalSpec(onset_sentinel=600)


if __name__ == "__main__":
    unittest.main()
