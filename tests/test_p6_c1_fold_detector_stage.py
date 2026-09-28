"""C1 detector stage seals a label-free Generation output after Fit/Selection."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.e2e.c1_fold_detector import train_c1_fold_detector
from src.e2e.c1_fold_supervision import C1FoldSupervision
from src.e2e.c1_fold_preprocessing import G2_CONFIG


class _TinyModel(torch.nn.Module):
    def __init__(self, graph, **kwargs):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        if any("label" in key for key in batch if key != "trigger_label"):
            raise AssertionError("unexpected label in detector input")
        logits = self.bias + batch["sample_index"].float() / 10.0
        return logits, self.bias * 0.0


class C1FoldDetectorStageTest(unittest.TestCase):
    def test_stage_seals_generation_before_any_gt_join(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fold_01"
            for segment, start in (("fit", 0), ("selection", 600000), ("generation", 1200000)):
                directory = root / "ad_data" / segment
                directory.mkdir(parents=True)
                np.save(directory / "timestamps.npy", start + np.arange(20, dtype=np.int64) * 30000)
                np.save(directory / "metric.npy", np.zeros((20, 10, 48), dtype=np.float32))
                np.save(directory / "log.npy", np.zeros((20, 10, 32), dtype=np.float32))
                np.save(directory / "trace.npy", np.zeros((20, 10, 10, 8), dtype=np.float32))
                np.save(directory / "legal_window_indices.npy", np.arange(10, dtype=np.int64))
            (root / "ad_data").mkdir(exist_ok=True)
            np.save(root / "ad_data/graph.npy", np.zeros((10, 10), dtype=np.float32))
            (root / "completion_manifest.json").write_text("{}\n", encoding="utf-8")
            segments = {name: {"interval_ms": [start, start + 600000],
                               "time_bins": 20, "window_bins": 10}
                        for name, start in (("fit", 0), ("selection", 600000),
                                            ("generation", 1200000))}
            fold = {"fold": 1, "services": list(("dbservice1", "dbservice2", "logservice1", "logservice2",
                                                  "mobservice1", "mobservice2", "redisservice1", "redisservice2",
                                                  "webservice1", "webservice2")),
                    "dimensions": {"raw_node": 48, "log_len": 32, "raw_edge": 8},
                    "graph": {"path": "ad_data/graph.npy"}, "segments": segments}
            fit_labels = np.zeros(20, dtype=np.int64)
            fit_labels[9] = 1
            supervision = C1FoldSupervision(
                fit_labels, np.zeros(20, dtype=np.int64),
                pd.DataFrame([{"start_ms": 900000}]), {"generation_labels_built": False})
            threshold = SimpleNamespace(
                threshold=0.5, candidate_count=1,
                metrics={"event_f1": 1.0, "event_recall": 1.0, "event_precision": 1.0})
            with patch("src.e2e.c1_fold_detector.validate_c1_fold", return_value=fold), \
                 patch("src.e2e.c1_fold_detector_data.validate_c1_fold", return_value=fold), \
                 patch("src.e2e.c1_fold_detector.build_c1_fit_selection_supervision", return_value=supervision), \
                 patch("src.e2e.c1_fold_detector.SystemEventTrigger", _TinyModel), \
                 patch("src.e2e.c1_fold_detector.select_system_threshold", return_value=threshold), \
                 patch("src.e2e.c1_fold_detector._sources", return_value={"test": "fixed"}):
                manifest = train_c1_fold_detector(
                    fold_root=root, protocol_path=G2_CONFIG, gpu=False, threshold_workers=1)
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertFalse((root / "detector/INCOMPLETE.json").exists())
            scores = pd.read_csv(root / "detector/generation_scores.csv")
            episodes = pd.read_csv(root / "detector/generation_episodes.csv")
            self.assertEqual(len(scores), 10)
            self.assertEqual(scores["fold"].unique().tolist(), [1])
            self.assertTrue(episodes["prediction_id"].str.startswith("fold_01-").all())
            self.assertNotIn("trigger_label", scores.columns)
            self.assertNotIn("case_id", episodes.columns)
            self.assertEqual(json.loads((root / "detector/selection.json").read_text())["selected_epoch"], 0)


if __name__ == "__main__":
    unittest.main()
