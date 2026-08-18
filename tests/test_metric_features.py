import math
import unittest

from src.features import (
    MetricFeatureAccumulator,
    aggregate_metric_contrasts,
    metric_series_contrasts,
    metric_series_contrasts_many,
)


class MetricFeatureTest(unittest.TestCase):
    def test_half_open_stages_and_duplicate_reduction(self):
        timestamps = (-300000, -1000, 0, 0, 30000, 60000, 299999, 300000)
        values = (0.0, 0.0, 1.0, 3.0, 2.0, 5.0, 5.0, 999.0)
        actual = metric_series_contrasts(timestamps, values, 0)
        reduced = metric_series_contrasts(
            (-300000, -1000, 0, 30000, 60000, 299999, 300000),
            (0.0, 0.0, 2.0, 2.0, 5.0, 5.0, -999.0),
            0,
        )
        self.assertEqual(actual, reduced)
        self.assertTrue(actual["whole"].observed)
        self.assertTrue(actual["stage60.pre_onset"].observed)
        self.assertTrue(actual["stage60.pre_impact"].observed)
        self.assertGreater(actual["stage60.pre_impact"].signed_mean_shift, 0.0)

    def test_undercovered_segments_are_masked_independently(self):
        actual = metric_series_contrasts(
            (-2000, -1000, 0, 1000, 2000),
            (0.0, 0.0, 1.0, 1.0, 1.0),
            0,
        )
        self.assertTrue(actual["whole"].observed)
        self.assertFalse(actual["stage30.pre_impact"].observed)
        self.assertFalse(actual["stage60.onset_impact"].observed)

    def test_aggregation_is_fixed_width_finite_and_mask_aware(self):
        observed = metric_series_contrasts(
            (-2000, -1000, 0, 1000, 60000, 61000, 120000, 121000),
            (0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            0,
        )
        missing = metric_series_contrasts((), (), 0)
        names, values, masks = aggregate_metric_contrasts(
            (observed, missing), total_series_count=3
        )
        self.assertEqual(len(names), len(values))
        self.assertEqual(len(values), len(masks))
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(math.isfinite(value) for value in values))
        whole_count = names.index("whole.valid_count")
        whole_ratio = names.index("whole.valid_ratio")
        self.assertEqual(values[whole_count], 1.0)
        self.assertAlmostEqual(values[whole_ratio], 1.0 / 3.0)
        missing_names, missing_values, missing_masks = aggregate_metric_contrasts(
            (missing,), total_series_count=3
        )
        missing_index = missing_names.index(
            "stage120.onset_impact.signed_mean_shift.abs_max"
        )
        self.assertFalse(missing_masks[missing_index])
        self.assertEqual(missing_values[missing_index], 0.0)

    def test_batch_and_streaming_accumulator_match_scalar_results(self):
        timestamps = (-2000, -1000, 0, 1000, 60000, 61000, 120000, 121000)
        first_values = (0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        second_values = (1.0, 1.0, 2.0, 4.0, 3.0, 5.0, 8.0, 9.0)
        anchors = (0, 1000)
        batches = metric_series_contrasts_many(
            timestamps, first_values, anchors
        )
        for index, anchor in enumerate(anchors):
            scalar = metric_series_contrasts(timestamps, first_values, anchor)
            for name, contrast in scalar.items():
                self.assertEqual(contrast.observed, bool(batches[name].observed[index]))
                self.assertAlmostEqual(
                    contrast.signed_mean_shift,
                    batches[name].signed_mean_shift[index],
                )

        accumulator = MetricFeatureAccumulator(2, ("svc",))
        accumulator.update(
            "svc", metric_series_contrasts_many(timestamps, first_values, anchors)
        )
        accumulator.update(
            "svc", metric_series_contrasts_many(timestamps, second_values, anchors)
        )
        for index, anchor in enumerate(anchors):
            scalar_rows = (
                metric_series_contrasts(timestamps, first_values, anchor),
                metric_series_contrasts(timestamps, second_values, anchor),
            )
            expected_names, expected_values, expected_masks = aggregate_metric_contrasts(
                scalar_rows, total_series_count=2
            )
            names, values, masks = accumulator.feature_vector(index, "svc")
            self.assertEqual(names, expected_names)
            self.assertEqual(masks, expected_masks)
            for actual, expected in zip(values, expected_values):
                self.assertAlmostEqual(actual, expected, places=6)

        case_accumulator = MetricFeatureAccumulator(2, ("svc",))
        for index, anchor in enumerate(anchors):
            case_accumulator.update_case(
                "svc",
                index,
                metric_series_contrasts(timestamps, first_values, anchor),
            )
            case_accumulator.update_case(
                "svc",
                index,
                metric_series_contrasts(timestamps, second_values, anchor),
            )
            expected = accumulator.feature_vector(index, "svc")
            actual = case_accumulator.feature_vector(index, "svc")
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[2], expected[2])
            for actual_value, expected_value in zip(actual[1], expected[1]):
                self.assertAlmostEqual(actual_value, expected_value, places=6)


if __name__ == "__main__":
    unittest.main()
