"""C1 Generation scoring remains label-free through OOS episode creation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.e2e.c1_fold_detector_data import C1SegmentWindowDataset
from src.e2e.c1_fold_generation import score_c1_generation, score_c1_selection


class _SyntheticModel:
    def eval(self):
        return self

    def __call__(self, batch):
        if "trigger_label" in batch:
            raise AssertionError("scorer passed Selection labels into the model")
        indices = batch["sample_index"]
        logits = torch.where(indices == 0, torch.tensor(2.0), torch.tensor(-2.0))
        return logits, torch.tensor(0.0)


class C1FoldGenerationTest(unittest.TestCase):
    def test_padded_batches_produce_complete_label_free_episode_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fold_01"
            directory = root / "ad_data/generation"
            directory.mkdir(parents=True)
            np.save(directory / "timestamps.npy", np.arange(20, dtype=np.int64) * 30000)
            np.save(directory / "metric.npy", np.zeros((20, 10, 48), dtype=np.float32))
            np.save(directory / "log.npy", np.zeros((20, 10, 32), dtype=np.float32))
            np.save(directory / "trace.npy", np.zeros((20, 10, 10, 8), dtype=np.float32))
            np.save(directory / "legal_window_indices.npy", np.arange(10, dtype=np.int64))
            manifest = {"segments": {"generation": {"interval_ms": [0, 600000],
                                                    "time_bins": 20, "window_bins": 10}}}
            with patch("src.e2e.c1_fold_detector_data.validate_c1_fold", return_value=manifest):
                dataset = C1SegmentWindowDataset(root, "generation", protocol_path=Path("unused"))
            frame, episodes = score_c1_generation(
                _SyntheticModel(), dataset, batch_size=4, num_workers=0,
                device=torch.device("cpu"), threshold=0.5)
            self.assertEqual(len(frame), 10)
            self.assertEqual(frame["sample_index"].tolist(), list(range(10)))
            self.assertEqual(frame["prediction_available_time"].iloc[0], 300000)
            self.assertEqual(frame["prediction_available_time"].iloc[-1], 570000)
            self.assertTrue(all("label" not in name and "root" not in name for name in frame.columns))
            self.assertEqual(len(episodes), 1)
            self.assertEqual(int(episodes["t_hat"].iloc[0]), 300000)
            self.assertNotIn("case_id", episodes.columns)

    def test_selection_scores_strip_supervision_before_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fold_01"
            directory = root / "ad_data/selection"
            directory.mkdir(parents=True)
            np.save(directory / "timestamps.npy", np.arange(20, dtype=np.int64) * 30000)
            np.save(directory / "metric.npy", np.zeros((20, 10, 48), dtype=np.float32))
            np.save(directory / "log.npy", np.zeros((20, 10, 32), dtype=np.float32))
            np.save(directory / "trace.npy", np.zeros((20, 10, 10, 8), dtype=np.float32))
            np.save(directory / "legal_window_indices.npy", np.arange(10, dtype=np.int64))
            manifest = {"segments": {"selection": {"interval_ms": [0, 600000],
                                                   "time_bins": 20, "window_bins": 10}}}
            with patch("src.e2e.c1_fold_detector_data.validate_c1_fold", return_value=manifest):
                dataset = C1SegmentWindowDataset(
                    root, "selection", protocol_path=Path("unused"),
                    trigger_labels=np.ones(20, dtype=np.int64))
            frame = score_c1_selection(_SyntheticModel(), dataset, batch_size=4,
                                       num_workers=0, device=torch.device("cpu"))
            self.assertEqual(len(frame), 10)
            self.assertTrue(all("label" not in name for name in frame.columns))
            self.assertNotIn("binary_prediction", frame.columns)


if __name__ == "__main__":
    unittest.main()
