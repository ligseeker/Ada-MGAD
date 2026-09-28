"""C1 fold detector input keeps generation labels outside the data path."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.e2e.c1_fold_detector_data import C1SegmentWindowDataset


class C1SegmentWindowDatasetTest(unittest.TestCase):
    def _fixture(self, root, segment, start=0):
        directory = root / "ad_data" / segment
        directory.mkdir(parents=True)
        np.save(directory / "timestamps.npy", start + np.arange(20, dtype=np.int64) * 30000)
        np.save(directory / "metric.npy", np.zeros((20, 10, 48), dtype=np.float32))
        np.save(directory / "log.npy", np.zeros((20, 10, 32), dtype=np.float32))
        np.save(directory / "trace.npy", np.zeros((20, 10, 10, 8), dtype=np.float32))
        np.save(directory / "legal_window_indices.npy", np.arange(10, dtype=np.int64))
        return {"segments": {segment: {"interval_ms": [start, start + 600000],
                                       "time_bins": 20, "window_bins": 10}}}

    def test_generation_exposes_inputs_and_times_without_any_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fold_01"
            manifest = self._fixture(root, "generation", 1200000)
            with patch("src.e2e.c1_fold_detector_data.validate_c1_fold", return_value=manifest):
                dataset = C1SegmentWindowDataset(root, "generation", protocol_path=Path("unused"))
            self.assertEqual(len(dataset), 10)
            self.assertEqual(set(dataset[0]), {"data_node", "data_log", "data_edge", "sample_index"})
            self.assertEqual(dataset[0]["data_node"].shape, (10, 10, 48))
            self.assertEqual(int(dataset.prediction_times()[0]), 1500000)
            self.assertEqual(int(dataset.prediction_times()[-1]), 1770000)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                dataset.labels_at()
            with patch("src.e2e.c1_fold_detector_data.validate_c1_fold", return_value=manifest):
                with self.assertRaisesRegex(ValueError, "label-free"):
                    C1SegmentWindowDataset(root, "generation", protocol_path=Path("unused"),
                                           trigger_labels=np.zeros(20, dtype=np.int64))

    def test_fit_labels_are_explicit_and_boundary_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fold_01"
            manifest = self._fixture(root, "fit")
            labels = np.zeros(20, dtype=np.int64)
            labels[9] = 1
            with patch("src.e2e.c1_fold_detector_data.validate_c1_fold", return_value=manifest):
                dataset = C1SegmentWindowDataset(root, "fit", protocol_path=Path("unused"),
                                                 trigger_labels=labels)
                self.assertEqual(int(dataset[0]["trigger_label"]), 1)
                self.assertEqual(int(dataset.labels_at()[0]), 1)
                np.save(root / "ad_data/fit/legal_window_indices.npy", np.arange(11, dtype=np.int64))
                with self.assertRaisesRegex(ValueError, "legal-window indices"):
                    C1SegmentWindowDataset(root, "fit", protocol_path=Path("unused"),
                                           trigger_labels=labels)


if __name__ == "__main__":
    unittest.main()
