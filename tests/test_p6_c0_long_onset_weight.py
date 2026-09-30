"""Train-only long-onset weighting keeps the global positive loss mass fixed."""

import unittest

import numpy as np
import pandas as pd

from scripts.p6.run_c0_trigger import fit_long_onset_weights
from src.e2e.system_trigger import TRIGGER_POSITIVE, build_trigger_labels


class _Dataset:
    split = "fit"

    def __init__(self, times, labels):
        self.times = times
        self.labels = labels
        self.sample_indices = np.arange(len(times), dtype=np.int64)

    def __len__(self):
        return len(self.times)

    def labels_at(self):
        return self.labels

    def prediction_times(self):
        return self.times


class _State:
    def __init__(self, events):
        self.events = events

    def gt_events(self, split):
        if split != "fit":
            raise AssertionError("weight fitting accessed another split")
        return self.events


class LongOnsetWeightTests(unittest.TestCase):
    def test_relative_weights_preserve_positive_mass_and_ignore_other_splits(self):
        times = np.arange(1, 41, dtype=np.int64) * 30000
        events = pd.DataFrame([
            {"start_ms": 90000, "end_ms": 510000},
            {"start_ms": 570000, "end_ms": 581000},
            {"start_ms": 750000, "end_ms": 761000},
            {"start_ms": 930000, "end_ms": 941000},
        ])
        labels = build_trigger_labels(times, events)
        dataset = _Dataset(times, labels)
        lookup, stats = fit_long_onset_weights(_State(events), dataset)
        self.assertEqual(stats["long_events_gt300s"], 1)
        self.assertGreater(stats["long_positive_bins"], 0)
        self.assertGreater(stats["other_positive_bins"], 0)
        self.assertGreater(stats["long_positive_weight"], stats["other_positive_weight"])
        self.assertAlmostEqual(
            float(lookup[labels == TRIGGER_POSITIVE].sum(dtype=np.float64)),
            stats["positive_bins"], places=3,
        )
        self.assertTrue(np.all(lookup[labels != TRIGGER_POSITIVE] == 1))
        with self.assertRaisesRegex(ValueError, "population drift"):
            fit_long_onset_weights(_State(events), dataset, {"fit_events": -1})


if __name__ == "__main__":
    unittest.main()
