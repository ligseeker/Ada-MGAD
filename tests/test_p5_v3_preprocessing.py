import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.e2e.ad_data import TimestampedArrayDataset, save_split_arrays
from src.e2e.ad_preprocess import build_log_arrays, build_metric_arrays
from src.e2e.protocol import GAIA_SERVICES


def _write_log_fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for index, service in enumerate(GAIA_SERVICES):
        pd.DataFrame({
            "message": [
                "2021-07-01 08:00:00,000 | INFO | train template",
                "2021-07-01 08:00:30,000 | ERROR | unseen test template {}".format(index),
            ],
        }).to_csv(root / "business_table_{}_2021-07.csv".format(service), index=False)


class V3PreprocessingIsolationTests(unittest.TestCase):
    def test_metric_normalization_is_fit_on_train_not_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metric_root = root / "metric"
            metric_root.mkdir()
            timestamps = np.asarray([1625097600000 + 30_000 * i for i in range(4)])
            for service in GAIA_SERVICES:
                pd.DataFrame({
                    "timestamp": timestamps,
                    "value": [1.0, 2.0, 100.0, 200.0],
                }).to_csv(
                    metric_root / (
                        "{}_0.0.0.1_request_count_2021-07-01_2021-07-15.csv".format(service)
                    ),
                    index=False,
                )
            values, stats = build_metric_arrays(
                metric_root,
                timestamps,
                {"train": slice(0, 2), "test": slice(2, 4)},
                root / "cache",
                workers=1,
                chunk_rows=2,
            )
            normalization = stats["normalization"]["dbservice1::request_count"]
            self.assertEqual(normalization["train_min"], 1.0)
            self.assertEqual(normalization["train_max"], 2.0)
            np.testing.assert_array_equal(values[:2, 0, 0], np.asarray([0.0, 1.0], dtype=np.float32))
            self.assertGreater(float(values[2, 0, 0]), 1.0)

    def test_drain3_test_transform_is_frozen_and_unseen_maps_to_unk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_root = root / "logs"
            _write_log_fixture(log_root)
            grid = np.asarray([1625097600000, 1625097630000], dtype=np.int64)
            values, stats = build_log_arrays(
                log_root,
                grid,
                {"train": slice(0, 1), "test": slice(1, 2)},
                chunk_rows=1,
                workers=1,
                template_artifact_dir=root / "templates",
            )
            self.assertIn("template_UNK", stats["feature_names"])
            unknown = stats["feature_names"].index("template_UNK")
            self.assertGreater(float(values[1, :, unknown].sum()), 0.0)
            self.assertGreater(stats["unseen_template_rows"], 0)
            self.assertEqual(stats["drain3"]["fit_split"], "train")
            self.assertIn("no cluster creation", stats["drain3"]["test_transform"])

    def test_window_dataset_is_split_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, offset in (("train", 0), ("test", 1_000_000)):
                count = 12
                arrays = {
                    "timestamps": offset + np.arange(count, dtype=np.int64) * 30_000,
                    "metric": np.zeros((count, 10, 1), dtype=np.float32),
                    "log": np.zeros((count, 10, 1), dtype=np.float32),
                    "trace": np.zeros((count, 10, 10, 1), dtype=np.float32),
                    "labels": np.zeros((count, 10), dtype=np.int8),
                    "label_mask": np.zeros((count, 10), dtype=np.int8),
                }
                save_split_arrays(root, split, arrays)
            train = TimestampedArrayDataset(root / "train", 10, 30)
            test = TimestampedArrayDataset(root / "test", 10, 30)
            self.assertEqual(len(train), 3)
            self.assertEqual(len(test), 3)
            for dataset, split in ((train, "train"), (test, "test")):
                for index in range(len(dataset)):
                    metadata = dataset.metadata(index)
                    self.assertEqual(metadata.split, split)
                    self.assertLessEqual(metadata.window_start_time, metadata.target_bin_start)
                    self.assertEqual(
                        metadata.prediction_available_time,
                        metadata.target_bin_end,
                    )
                    self.assertEqual(
                        metadata.target_bin_end - metadata.window_start_time,
                        10 * 30_000,
                    )


if __name__ == "__main__":
    unittest.main()
