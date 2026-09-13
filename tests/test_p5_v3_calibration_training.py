import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.e2e.calibration import fit_reconstruction_calibration, load_reconstruction_calibration, save_reconstruction_calibration
from util.train import MY


class DummyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "dummy"
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch, evaluate=False, **kwargs):
        batch_size = batch["groundtruth_real"].shape[0]
        windows = batch["groundtruth_real"].shape[1]
        if evaluate:
            score = torch.sigmoid(self.weight).expand(batch_size, windows, 1)
            output = torch.cat((1.0 - score, score), dim=-1)
            return output, batch["groundtruth_cls"]
        zero = self.weight * 0.0
        cls_result = torch.empty((0, 2), device=self.weight.device)
        cls_label = torch.empty((0, 2), device=self.weight.device)
        return [zero + 1.0], cls_result, cls_label, zero, zero


def train_batches():
    rows = []
    for _ in range(4):
        rows.append({
            "groundtruth_real": torch.zeros((3, 2), dtype=torch.float32),
            "groundtruth_cls": torch.zeros((3, 3), dtype=torch.float32),
            "data_node": torch.zeros((3, 1, 1), dtype=torch.float32),
        })
    return DataLoader(rows, batch_size=2)


class CalibrationTests(unittest.TestCase):
    def test_fit_save_load_is_train_only_and_frozen(self):
        calibration = fit_reconstruction_calibration(np.asarray([1.0, 2.0, 3.0]))
        self.assertEqual(calibration.fit_split, "train")
        self.assertEqual(calibration.train_count, 3)
        before = calibration.transform_scores(np.asarray([100.0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_reconstruction_calibration(path, calibration)
            loaded = load_reconstruction_calibration(path)
            np.testing.assert_array_equal(before, loaded.transform_scores(np.asarray([100.0])))


class TrainCheckpointTests(unittest.TestCase):
    def test_train_loss_stopping_and_all_v3_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            args = {
                "gpu": False, "epochs": 5, "patience": 1, "learning_rate": 0.001,
                "weight_decay": 0.0, "learning_change": 30, "learning_gamma": 0.5,
                "eval_interval": 1, "train_eval_interval": 1, "rec_down": 100,
                "para_low": 0.01, "abnormal_weight": 1, "evaluate": False,
                "result_dir": directory, "checkpoint_policy": "v3",
                "contrast_weight": 0.0, "batch_progress": False,
            }
            trainer = MY(DummyDetector(), **args)
            summary = trainer.fit(train_batches(), train_eval_loader=train_batches())
            self.assertEqual(summary["early_stopping_metric"], "train_total_loss")
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertFalse(summary["test_used_for_fit_or_selection"])
            self.assertEqual(
                sorted(path.name for path in Path(directory).glob("*.pt")),
                ["best_train_f1.pt", "best_train_loss.pt", "last.pt"],
            )

    def test_validation_and_test_loaders_are_rejected(self):
        args = {
            "gpu": False, "epochs": 1, "patience": 1, "learning_rate": 0.001,
            "weight_decay": 0.0, "learning_change": 30, "learning_gamma": 0.5,
            "eval_interval": 1, "rec_down": 1, "para_low": 0.01,
            "abnormal_weight": 1, "evaluate": False, "result_dir": tempfile.mkdtemp(),
        }
        trainer = MY(DummyDetector(), **args)
        with self.assertRaisesRegex(ValueError, "forbids"):
            trainer.fit(train_batches(), val_loader=train_batches())


if __name__ == "__main__":
    unittest.main()
