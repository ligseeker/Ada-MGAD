"""P6-C1-v2 detector stage: label firewall, Selection-only choice, sealed Generation."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.e2e import c1_v2_detector as detector_module
from src.e2e import c1_v2_shared_data as shared
from src.e2e.c1_v2_detector import (
    C1V2SegmentWindowDataset, C1V2Supervision, train_c1_v2_fold_detector,
    validate_c1_v2_fold_detector,
)

MODEL_ARG_KEYS = (
    "feature_node", "feature_edge", "feature_log", "num_heads_edge", "num_heads_node",
    "num_heads_log", "num_heads_n2e", "num_heads_e2n", "num_layer", "dropout",
    "graph_hidden", "graph_sparse_weight", "graph_summary_mode",
)
BASE_AD_MODEL = {
    "feature_node": 16, "feature_edge": 4, "feature_log": 8, "num_heads_edge": 4,
    "num_heads_node": 4, "num_heads_log": 4, "num_heads_n2e": 4, "num_heads_e2n": 2,
    "num_layer": 2, "dropout": 0.2, "graph_hidden": 16, "graph_sparse_weight": 0.001,
    "graph_summary_mode": "last",
}
C0_TRAINING = {"max_epochs": 30, "patience": 8, "batch_size": 32, "num_workers": 2,
               "learning_rate": 0.001, "weight_decay": 0.0005, "scheduler_step": 10,
               "scheduler_gamma": 0.5, "grad_clip_norm": 10.0}


class _TinyModel(torch.nn.Module):
    def __init__(self, graph, **kwargs):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        if any("label" in key for key in batch if key != "trigger_label"):
            raise AssertionError("unexpected label in detector input")
        logits = self.bias + batch["sample_index"].float() / 1000.0
        return logits, self.bias * 0.0


def _build_fold(root: Path, *, bins: int = 20):
    segments = {}
    for segment, offset in (("fit", 0), ("selection", 600000), ("generation", 1200000)):
        directory = root / "ad_data" / segment
        directory.mkdir(parents=True)
        start = 1625133600000 + offset
        np.save(directory / "timestamps.npy", start + np.arange(bins, dtype=np.int64) * 30000)
        np.save(directory / "metric.npy", np.zeros((bins, 10, 48), dtype=np.float32))
        np.save(directory / "log.npy", np.zeros((bins, 10, 32), dtype=np.float32))
        np.save(directory / "trace.npy", np.zeros((bins, 10, 10, 8), dtype=np.float32))
        np.save(directory / "legal_window_indices.npy", np.arange(bins - 10, dtype=np.int64))
        segments[segment] = {
            "interval_ms": [start, start + bins * 30000], "time_bins": bins,
            "window_bins": 10, "legal_windows": bins - 10,
            "first_prediction_available_ms": start + 300000,
            "last_prediction_available_ms": start + bins * 30000 - 30000,
            "label_fields_present": [],
            "files": {name: {"path": "ad_data/{}/{}.npy".format(segment, name)}
                      for name in ("timestamps", "metric", "log", "trace", "legal_window_indices")},
        }
    np.save(root / "ad_data/graph.npy", np.zeros((10, 10), dtype=np.float32))
    (root / "completion_manifest.json").write_text("{}\n", encoding="utf-8")
    return {
        "schema_version": shared.FOLD_MANIFEST_SCHEMA, "status": "COMPLETE",
        "protocol_id": shared.PROTOCOL_ID, "run_id": shared.RUN_ID, "fold": 1,
        "fold_root": str(Path(root).resolve()), "formal": True, "smoke": False,
        "shared_preprocessing_refit": False, "schema_refit": False, "scaler_refit": False,
        "graph_refit": False, "drain3_refit": False,
        "services": list(shared.GAIA_SERVICES),
        "dimensions": dict(shared.EXPECTED_DIMENSIONS),
        "graph": {"path": "ad_data/graph.npy"},
        "segments": segments,
    }


class _ProtocolFixture:
    def __init__(self, root: Path):
        (root / "base.json").write_text(json.dumps({
            "ad": {"grid_seconds": 30, "window_bins": 10, "window_step": 1},
            "ad_model": dict(BASE_AD_MODEL)}), encoding="utf-8")
        (root / "c0.json").write_text(json.dumps({
            "seed": 42, "model": {"head_hidden": 32}, "training": dict(C0_TRAINING)}),
            encoding="utf-8")
        self.protocol = {
            "protocol_id": shared.PROTOCOL_ID, "run_id": shared.RUN_ID, "seed": 42,
            "detector": {"matching_tolerance_seconds": 60,
                         "expected_frozen_values": {"max_epochs": 30, "patience": 8}},
            "resource_budget": {"detector_loader_workers": 2, "threshold_workers": 8,
                                "max_epochs_per_fold": 30, "torch_threads": 2},
            "bindings": {"base_config": {"path": str(root / "base.json")},
                         "c0_trigger_config": {"path": str(root / "c0.json")}},
        }


class C1V2DetectorStageTest(unittest.TestCase):
    def _run_stage(self, root: Path, *, metrics_by_epoch=None, epochs: int = 1):
        fixture = _ProtocolFixture(root)
        fold = _build_fold(root / "fold_01")
        fit_labels = np.zeros(20, dtype=np.int64)
        fit_labels[9] = 1
        supervision = C1V2Supervision(fit_labels, np.zeros(20, dtype=np.int64),
                                      pd.DataFrame([{"start_ms": 1625134200000}]),
                                      {"generation_labels_built": False})
        counter = {"epoch": -1}

        def _threshold(*args, **kwargs):
            counter["epoch"] += 1
            epoch = counter["epoch"]
            table = metrics_by_epoch or {0: (0.5, 0.5)}
            f1, recall = table.get(epoch, table[max(table)])
            return SimpleNamespace(threshold=0.5, candidate_count=1,
                                   metrics={"event_f1": f1, "event_recall": recall,
                                            "event_precision": 1.0})

        patches = [
            patch.object(detector_module, "validate_c1_v2_fold", return_value=fold),
            patch.object(detector_module, "bound_path",
                         side_effect=lambda binding: Path(binding["path"])),
            patch.object(detector_module, "source_hashes", return_value={"test": "fixed"}),
            patch.object(detector_module, "build_c1_v2_supervision", return_value=supervision),
            patch.object(detector_module, "load_c1_v2_registry", return_value=pd.DataFrame()),
            patch.object(detector_module, "SystemEventTrigger", _TinyModel),
            patch.object(detector_module, "select_system_threshold", side_effect=_threshold),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        manifest = train_c1_v2_fold_detector(
            fold_root=root / "fold_01", gpu=False, threshold_workers=1, epochs=epochs,
            patience=0, protocol=fixture.protocol)
        return manifest, fixture

    def test_generation_stays_label_free_and_sealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._run_stage(root)
            fold_root = root / "fold_01"
            episodes = pd.read_csv(fold_root / "detector/generation_episodes.csv")
            scores = pd.read_csv(fold_root / "detector/generation_scores.csv")
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertFalse((fold_root / "detector/INCOMPLETE.json").exists())
            self.assertEqual(len(scores), 10)
            self.assertEqual(scores["fold"].unique().tolist(), [1])
            self.assertTrue(episodes["prediction_id"].str.startswith("fold_01-").all())
            for frame in (scores, episodes):
                self.assertFalse({"trigger_label", "case_id", "root_service", "fault_type",
                                  "service", "match_status"} & set(frame.columns))
            self.assertTrue((fold_root / "detector/checkpoint.pt").is_file())
            selection = json.loads((fold_root / "detector/selection.json").read_text())
            self.assertFalse(selection["test_used"])

    def test_selection_only_checkpoint_choice_uses_f1_then_recall(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._run_stage(
                root, epochs=4,
                metrics_by_epoch={0: (0.5, 0.9), 1: (0.9, 0.6), 2: (0.9, 0.7), 3: (0.8, 0.8)})
            self.assertEqual(manifest["selected_epoch"], 2)
            self.assertEqual(manifest["threshold"], 0.5)

    def test_generation_dataset_refuses_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fold = _build_fold(root / "fold_01")
            with self.assertRaisesRegex(ValueError, "label-free"):
                C1V2SegmentWindowDataset(root / "fold_01", "generation",
                                         manifest=fold, trigger_labels=np.zeros(20, dtype=np.int64))
            dataset = C1V2SegmentWindowDataset(root / "fold_01", "generation", manifest=fold)
            self.assertNotIn("trigger_label", dataset[0])
            with self.assertRaisesRegex(ValueError, "Generation labels are unavailable"):
                dataset.labels_at()

    def test_epoch_override_is_refused_inside_the_formal_run_root(self):
        formal_fold = shared.ROOT / "experiments/p6/c1_supervision_oos/probe/folds/fold_01"
        with self.assertRaisesRegex(ValueError, "forbidden inside the formal"):
            train_c1_v2_fold_detector(fold_root=formal_fold, epochs=1, patience=0)


if __name__ == "__main__":
    unittest.main()
