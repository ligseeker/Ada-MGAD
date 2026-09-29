"""P6-C1-v2 shared-array slicing: frozen identity, purge geometry and sealing."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.e2e import c1_v2_shared_data as shared


SERVICES = list(shared.GAIA_SERVICES)
GRID_MS = 30000
TRAIN_START = 1625133600000
BINS = 240


def _sha(path: Path) -> str:
    return shared.sha256(path)


def _binding(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _base_config(path: Path) -> None:
    path.write_text(json.dumps({
        "services": SERVICES,
        "split": {"absolute_start_ms": TRAIN_START,
                  "absolute_end_ms": TRAIN_START + BINS * GRID_MS},
        "ad": {"grid_seconds": 30, "window_bins": 10, "window_step": 1},
    }), encoding="utf-8")


def _c0_config(path: Path) -> None:
    path.write_text(json.dumps({
        "seed": 42,
        "model": {"head_hidden": 32},
        "training": {"max_epochs": 30, "patience": 8, "batch_size": 32, "num_workers": 2,
                     "learning_rate": 0.001, "weight_decay": 0.0005, "scheduler_step": 10,
                     "scheduler_gamma": 0.5, "grad_clip_norm": 10.0},
    }), encoding="utf-8")


class _Fixture:
    """Small frozen-looking arrays plus the protocol bindings that point at them."""

    def __init__(self, root: Path, bins: int = BINS):
        self.root = root
        self.bins = bins
        root.mkdir(parents=True, exist_ok=True)
        self.timestamps = TRAIN_START + np.arange(bins, dtype=np.int64) * GRID_MS
        np.save(root / "timestamps.npy", self.timestamps)
        self.metric = (np.arange(bins * 10 * 48, dtype=np.float32).reshape(bins, 10, 48) / 1000.0)
        self.log = (np.arange(bins * 10 * 32, dtype=np.float32).reshape(bins, 10, 32) / 2000.0)
        self.trace = (np.arange(bins * 10 * 10 * 8, dtype=np.float32).reshape(bins, 10, 10, 8) / 3000.0)
        np.save(root / "metric.npy", self.metric)
        np.save(root / "log.npy", self.log)
        np.save(root / "trace.npy", self.trace)
        graph = np.zeros((10, 10), dtype=np.float32)
        graph[0, 1] = graph[1, 0] = 1.0
        np.save(root / "graph.npy", graph)
        np.save(root / "test_timestamps.npy", self.timestamps[-1:] + GRID_MS)
        (root / "schema.json").write_text("{}\n", encoding="utf-8")
        (root / "ad_manifest.json").write_text("{}\n", encoding="utf-8")
        _base_config(root / "base_config.json")
        _c0_config(root / "c0.json")
        self.protocol = {
            "protocol_id": shared.PROTOCOL_ID,
            "run_id": shared.RUN_ID,
            "output_root": str(shared.ROOT / "experiments/p6/c1_supervision_oos" / shared.RUN_ID),
            "seed": 42,
            "detector": {"history_seconds": 300, "window_bins": 10,
                         "matching_tolerance_seconds": 60},
            "resource_budget": {"detector_loader_workers": 2, "threshold_workers": 8,
                                "max_epochs_per_fold": 30, "torch_threads": 8},
            "bindings": {
                "base_config": _binding(root / "base_config.json"),
                "c0_trigger_config": _binding(root / "c0.json"),
                "shared_train_metric": _binding(root / "metric.npy"),
                "shared_train_log": _binding(root / "log.npy"),
                "shared_train_trace": _binding(root / "trace.npy"),
                "shared_train_timestamps": _binding(root / "timestamps.npy"),
                "shared_test_timestamps": _binding(root / "test_timestamps.npy"),
                "shared_ad_graph": _binding(root / "graph.npy"),
                "shared_frozen_schema": _binding(root / "schema.json"),
                "shared_ad_manifest": _binding(root / "ad_manifest.json"),
            },
        }
        self.intervals = {
            "fit": (TRAIN_START, TRAIN_START + 60 * GRID_MS),
            "selection": (TRAIN_START + 60 * GRID_MS, TRAIN_START + 90 * GRID_MS),
            "generation": (TRAIN_START + 90 * GRID_MS, TRAIN_START + 120 * GRID_MS),
        }

    def patches(self, run_dir: Path):
        return (
            patch.object(shared, "bound_path", side_effect=lambda binding: Path(binding["path"])),
            patch.object(shared, "validate_g1_stage", return_value={"status": "COMPLETE"}),
            patch.object(shared, "run_root", return_value=run_dir),
        )


class C1V2SharedDataTest(unittest.TestCase):
    def test_smoke_slices_are_exact_frozen_copies_with_legal_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            (run_dir / "g1_static").mkdir(parents=True)
            (run_dir / "g1_static/completion_manifest.json").write_text("{}\n", encoding="utf-8")
            fixture = _Fixture(root / "frozen")
            fold_root = root / "fold_01"
            first, second, third = fixture.patches(run_dir)
            with first, second, third:
                manifest = shared.materialize_c1_v2_fold(
                    protocol=fixture.protocol, fold_number=1, fold_root=fold_root,
                    intervals_ms=fixture.intervals, smoke=True)
            self.assertTrue(manifest["smoke"])
            self.assertFalse(manifest["formal"])
            self.assertFalse(manifest["shared_preprocessing_refit"])
            self.assertFalse(manifest["node_anomaly_labels_read"])
            for name, interval in fixture.intervals.items():
                segment = manifest["segments"][name]
                start, end = interval
                first_index = (start - TRAIN_START) // GRID_MS
                bins = (end - start) // GRID_MS
                timestamps = np.load(fold_root / segment["files"]["timestamps"]["path"])
                metric = np.load(fold_root / segment["files"]["metric"]["path"])
                legal = np.load(fold_root / segment["files"]["legal_window_indices"]["path"])
                self.assertTrue(np.array_equal(timestamps, fixture.timestamps[first_index:first_index + bins]))
                self.assertTrue(np.array_equal(metric, fixture.metric[first_index:first_index + bins]))
                self.assertEqual(segment["time_bins"], bins)
                self.assertEqual(segment["legal_windows"], bins - 10)
                self.assertTrue(np.array_equal(legal, np.arange(bins - 10, dtype=np.int64)))
                self.assertEqual(segment["first_prediction_available_ms"], start + 10 * GRID_MS)
                self.assertEqual(segment["last_prediction_available_ms"], end - GRID_MS)
                self.assertEqual(segment["label_fields_present"], [])
            graph = np.load(fold_root / manifest["graph"]["path"])
            self.assertTrue(np.array_equal(graph, np.load(root / "frozen/graph.npy")))
            first, second, third = fixture.patches(run_dir)
            with first, second, third:
                again = shared.validate_c1_v2_fold(fold_root, protocol=fixture.protocol)
            self.assertEqual(again["fold"], 1)

    def test_tampered_segment_fails_the_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            (run_dir / "g1_static").mkdir(parents=True)
            (run_dir / "g1_static/completion_manifest.json").write_text("{}\n", encoding="utf-8")
            fixture = _Fixture(root / "frozen")
            fold_root = root / "fold_01"
            first, second, third = fixture.patches(run_dir)
            with first, second, third:
                manifest = shared.materialize_c1_v2_fold(
                    protocol=fixture.protocol, fold_number=1, fold_root=fold_root,
                    intervals_ms=fixture.intervals, smoke=True)
                path = fold_root / manifest["segments"]["fit"]["files"]["metric"]["path"]
                values = np.load(path)
                values[0, 0, 0] += 1.0
                np.save(path, values)
                with self.assertRaisesRegex(ValueError, "output drift"):
                    shared.validate_c1_v2_fold(fold_root, protocol=fixture.protocol)

    def test_smoke_cannot_write_inside_the_formal_run_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root / "frozen")
            forbidden = shared.ROOT / "experiments/p6/c1_supervision_oos/probe_fold_01"
            first, second, third = fixture.patches(root / "run")
            with first, second, third:
                with self.assertRaisesRegex(ValueError, "formal run root"):
                    shared.materialize_c1_v2_fold(
                        protocol=fixture.protocol, fold_number=1, fold_root=forbidden,
                        intervals_ms=fixture.intervals, smoke=True)
            self.assertFalse(forbidden.exists())

    def test_formal_path_requires_the_locked_run_root_and_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root / "frozen")
            first, second, third = fixture.patches(root / "run")
            with first, second, third:
                with self.assertRaisesRegex(ValueError, "folds/fold_XX"):
                    shared.materialize_c1_v2_fold(protocol=fixture.protocol, fold_number=1,
                                                  fold_root=root / "elsewhere")
                with self.assertRaisesRegex(ValueError, "locked protocol intervals"):
                    shared.materialize_c1_v2_fold(
                        protocol=fixture.protocol, fold_number=1,
                        fold_root=root / "run/folds/fold_01",
                        intervals_ms=fixture.intervals)


if __name__ == "__main__":
    unittest.main()
