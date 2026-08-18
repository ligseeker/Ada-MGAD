import math
import unittest

from src.features import (
    RE2_TRACE_SERVICE_ALIASES,
    log_stream_features,
    normalize_trace_service,
    trace_stream_features,
)


class EventFeatureTest(unittest.TestCase):
    def test_log_features_are_half_open_and_mask_empty_content(self):
        names, values, masks = log_stream_features(
            (-300000, -1000, 0, 1000, 30000, 299999, 300000),
            (0, 0, 1, 0, 1, 1, 0),
            (10, 10, 20, 20, 30, 30, 999),
            0,
        )
        self.assertEqual(len(names), 50)
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertTrue(masks[names.index("whole.error_fraction_shift")])
        self.assertNotEqual(values[names.index("whole.log_event_rate_ratio")], 0.0)

        empty_names, empty_values, empty_masks = log_stream_features(
            (), (), (), 0, entity_observed=True
        )
        self.assertTrue(empty_masks[empty_names.index("whole.log_event_rate_ratio")])
        self.assertFalse(empty_masks[empty_names.index("whole.error_fraction_shift")])
        self.assertEqual(empty_values[empty_names.index("whole.error_fraction_shift")], 0.0)

    def test_absent_entity_masks_every_log_and_trace_feature(self):
        _, log_values, log_masks = log_stream_features(
            (0,), (0,), (10,), 0, entity_observed=False
        )
        self.assertFalse(any(log_masks))
        self.assertTrue(all(value == 0.0 for value in log_values))
        _, trace_values, trace_masks = trace_stream_features(
            (0,), (0.1,), (0,), ("trace",), ("op",), (0,), 0,
            entity_observed=False,
        )
        self.assertFalse(any(trace_masks))
        self.assertTrue(all(value == 0.0 for value in trace_values))

    def test_trace_features_keep_status_duration_and_parent_masks_separate(self):
        names, values, masks = trace_stream_features(
            (-2000, -1000, 0, 1000, 2000),
            (0.1, 0.2, 0.3, 0.4, 0.5),
            (0, float("nan"), 1, 0, float("nan")),
            ("a", "b", "c", "c", "d"),
            ("get", "get", "post", "post", ""),
            (1, 0, 1, float("nan"), 0),
            0,
        )
        self.assertEqual(len(names), 80)
        self.assertTrue(masks[names.index("whole.duration_p90_ratio")])
        self.assertTrue(masks[names.index("whole.error_fraction_shift")])
        self.assertTrue(masks[names.index("whole.parent_fraction_shift")])
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_trace_alias_is_explicit_and_candidate_bounded(self):
        candidates = ("frontend", "checkoutservice")
        self.assertEqual(
            normalize_trace_service(
                "frontendservice", candidates, RE2_TRACE_SERVICE_ALIASES
            ),
            "frontend",
        )
        self.assertEqual(
            normalize_trace_service("checkoutservice", candidates, {}),
            "checkoutservice",
        )
        self.assertIsNone(
            normalize_trace_service("unknown", candidates, RE2_TRACE_SERVICE_ALIASES)
        )

    def test_invalid_binary_and_negative_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "0/1/NaN"):
            log_stream_features((0,), (2,), (10,), 0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            trace_stream_features(
                (0,), (-1,), (0,), (None,), (None,), (0,), 0
            )


if __name__ == "__main__":
    unittest.main()
