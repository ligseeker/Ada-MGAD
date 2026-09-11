import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.e2e.ad_data import (
    PaddedSequentialSampler,
    TimestampedArrayDataset,
    build_registry_node_labels,
    build_semisupervised_mask,
    save_split_arrays,
)
from util.train import MY


class RegistryLabelTests(unittest.TestCase):
    def test_only_explicit_registry_intervals_become_positive(self):
        timestamps = np.asarray([0, 30000, 60000, 90000], dtype=np.int64)
        registry = pd.DataFrame([
            {"case_id": "supported", "service": "dbservice1", "start_ms": 31000, "end_ms": 61000},
        ])
        labels = build_registry_node_labels(timestamps, registry)
        self.assertEqual(labels[:, 0].tolist(), [0, 1, 1, 0])
        self.assertEqual(int(labels[:, 1:].sum()), 0)

    def test_semisupervised_mask_has_only_expected_states(self):
        labels = np.zeros((30, 10), dtype=np.int8)
        labels[10:20, 0] = 1
        masked = build_semisupervised_mask(labels, 0.5, 10)
        self.assertEqual(masked.shape, labels.shape)
        self.assertTrue(set(np.unique(masked)).issubset({0, 1, 2}))


class TimestampedDatasetTests(unittest.TestCase):
    def test_metadata_and_arrays_remain_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            count = 12
            arrays = {
                "timestamps": np.arange(count, dtype=np.int64) * 30000,
                "metric": np.zeros((count, 10, 2), dtype=np.float32),
                "log": np.zeros((count, 10, 3), dtype=np.float32),
                "trace": np.zeros((count, 10, 10, 4), dtype=np.float32),
                "labels": np.zeros((count, 10), dtype=np.int8),
                "label_mask": np.zeros((count, 10), dtype=np.int8),
            }
            arrays["labels"][9, 3] = 1
            save_split_arrays(root, "train", arrays)
            dataset = TimestampedArrayDataset(root / "train", 10, 30)
            sample = dataset[0]
            metadata = dataset.metadata(0)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(sample["data_node"].shape, (10, 10, 2))
            self.assertEqual(metadata.window_start_time, 0)
            self.assertEqual(metadata.window_end_time, 300000)
            self.assertEqual(metadata.prediction_timestamp, 270000)
            self.assertEqual(metadata.node_labels[3], 1)
            self.assertEqual(tuple(PaddedSequentialSampler(dataset, 2)), (0, 1, 2, 2))

    def test_score_fusion_preserves_timestamp_node_shape(self):
        trainer = MY.__new__(MY)
        trainer.score_fusion_alpha = 0.7
        probabilities = torch.full((2, 10, 2), 0.5)
        reconstruction = torch.arange(20, dtype=torch.float32)
        fused = trainer._fuse_predict_with_reconstruction(probabilities, reconstruction)
        self.assertEqual(tuple(fused.shape), (2, 10, 2))
        self.assertTrue(torch.allclose(fused.sum(dim=-1), torch.ones((2, 10))))


if __name__ == "__main__":
    unittest.main()
