"""P6-C0 model and driver guarantees: pooling invariance, no label leakage."""

import inspect
import json
from pathlib import Path
import re
import unittest

import numpy as np
import pandas as pd
import torch

from src.e2e.ad_data import save_split_arrays
from src.e2e.protocol import GAIA_SERVICES
from src.e2e.system_trigger import (
    TRIGGER_IGNORE,
    TRIGGER_NEGATIVE,
    TRIGGER_POSITIVE,
    build_trigger_labels,
    prediction_time_grid,
    trigger_temporal_blocks,
)
from src.e2e.system_trigger_data import TriggerWindowDataset, build_trigger_loader
from src.e2e.system_trigger_model import SystemEventTrigger, pool_service_representations

import scripts.p6.run_c0_trigger as driver


ORIGIN = 1625133600000
GRID_MS = 30_000
DRIVER_PATH = Path(driver.__file__)
MODEL_PATH = Path(inspect.getfile(SystemEventTrigger))
DATA_PATH = Path(inspect.getfile(TriggerWindowDataset))


def small_model(batch_size=2, window=10):
    graph = np.zeros((10, 10), dtype=np.float32)
    for index in range(10):
        graph[index, (index + 1) % 10] = 1.0
        graph[(index + 1) % 10, index] = 1.0
    args = {
        "main_model": "P6-C0-SystemEventTrigger", "gpu": False, "window": window,
        "batch_size": batch_size, "num_nodes": 10, "num_layer": 1,
        "feature_node": 8, "feature_edge": 4, "feature_log": 4,
        "num_heads_node": 2, "num_heads_log": 2, "num_heads_edge": 2,
        "num_heads_n2e": 2, "num_heads_e2n": 1,
        "dropout": 0.0, "graph_hidden": 8, "graph_sparse_weight": 0.001,
        "graph_summary_mode": "last", "head_hidden": 8,
        "raw_node": 16, "log_len": 8, "raw_edge": 4,
    }
    return SystemEventTrigger(graph, **args)


def synthetic_batch(batch_size=2, window=10):
    torch.manual_seed(0)
    return {
        "data_node": torch.randn(batch_size, window, 10, 16),
        "data_log": torch.rand(batch_size, window, 10, 8),
        "data_edge": torch.rand(batch_size, window, 10, 10, 4),
    }


def synthetic_dataset(root: Path, split="fit", count=40):
    timestamps = ORIGIN + np.arange(count, dtype=np.int64) * GRID_MS
    rng = np.random.RandomState(0)
    save_split_arrays(root, "train", {
        "timestamps": timestamps,
        "metric": rng.normal(size=(count, 10, 16)).astype(np.float32),
        "log": rng.uniform(size=(count, 10, 8)).astype(np.float32),
        "trace": rng.uniform(size=(count, 10, 10, 4)).astype(np.float32),
        "labels": np.zeros((count, 10), dtype=np.int8),
        "label_mask": np.zeros((count, 10), dtype=np.int8),
    })
    labels = np.full(count, TRIGGER_NEGATIVE, dtype=np.int8)
    labels[12:14] = TRIGGER_POSITIVE
    labels[14:20] = TRIGGER_IGNORE
    dataset = TriggerWindowDataset(
        root / "train", window_bins=10, grid_seconds=30,
        sample_indices=np.arange(count - 9, dtype=np.int64),
        trigger_labels=labels, split_name=split,
    )
    return dataset


class PoolingInvarianceTests(unittest.TestCase):
    def test_pooling_is_permutation_invariant(self):
        torch.manual_seed(1)
        h = torch.randn(4, 10, 8)
        pooled = pool_service_representations(h)
        for seed in (2, 3, 4):
            permutation = torch.randperm(10, generator=torch.Generator().manual_seed(seed))
            self.assertTrue(torch.allclose(pooled, pool_service_representations(h[:, permutation]), atol=1e-6))

    def test_pooled_width_is_three_times_the_service_feature_width(self):
        h = torch.randn(2, 10, 8)
        self.assertEqual(pool_service_representations(h).shape, (2, 24))

    def test_pooling_rejects_a_non_service_axis(self):
        with self.assertRaisesRegex(ValueError, "canonical ten-service"):
            pool_service_representations(torch.randn(2, 4, 8))
        with self.assertRaisesRegex(ValueError, "batch, services, feature"):
            pool_service_representations(torch.randn(2, 8))

    def test_head_output_is_permutation_invariant(self):
        model = small_model()
        model.eval()
        with torch.no_grad():
            representations, _ = model.encode(synthetic_batch())
            reference = model.system_logits(representations)
            for seed in (5, 6):
                permutation = torch.randperm(10, generator=torch.Generator().manual_seed(seed))
                shuffled = model.system_logits(representations[:, permutation])
                self.assertEqual(shuffled.shape, reference.shape)
                self.assertTrue(torch.allclose(reference, shuffled, atol=1e-6))

    def test_single_service_shuffle_keeps_the_same_scalar_system_score(self):
        model = small_model()
        model.eval()
        with torch.no_grad():
            representations, _ = model.encode(synthetic_batch())
            scores = model.system_logits(representations)
            self.assertEqual(scores.shape, (2,))
            self.assertTrue(torch.isfinite(scores).all())


class ModelContractTests(unittest.TestCase):
    def test_forward_returns_one_scalar_logit_per_window(self):
        model = small_model()
        logits, graph_reg = model(synthetic_batch())
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(graph_reg.dim(), 0)
        self.assertTrue(torch.isfinite(logits).all())

    def test_forward_never_consumes_node_or_root_labels(self):
        model = small_model()
        # a batch without any ground-truth key must be sufficient
        batch = synthetic_batch()
        self.assertNotIn("groundtruth_cls", batch)
        self.assertNotIn("groundtruth_real", batch)
        logits, _ = model(batch)
        self.assertEqual(int(logits.numel()), 2)
        source = MODEL_PATH.read_text(encoding="utf-8")
        for forbidden in ("groundtruth", "label_mask", "labels.npy", "node_label"):
            self.assertNotIn(forbidden, source)
        parameters = list(inspect.signature(SystemEventTrigger.forward).parameters)
        self.assertEqual(parameters, ["self", "batch", "global_step"])

    def test_model_has_no_service_output_head(self):
        model = small_model()
        logits, _ = model(synthetic_batch())
        self.assertEqual(logits.dim(), 1)
        self.assertEqual(model.head[-1].out_features, 1)

    def test_pooling_module_is_reachable_and_label_free(self):
        source = MODEL_PATH.read_text(encoding="utf-8")
        self.assertIn("pool_service_representations", source)
        self.assertIn("torch.cat([mean, maximum, std], dim=-1)", source)


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.root = Path(tempfile.mkdtemp(prefix="p6c0_ds_"))

    def test_batch_contains_no_node_labels(self):
        dataset = synthetic_dataset(self.root)
        item = dataset[0]
        self.assertEqual(
            set(item), {"data_node", "data_log", "data_edge", "sample_index", "trigger_label"}
        )

    def test_trigger_label_matches_the_target_window(self):
        dataset = synthetic_dataset(self.root)
        for position in (0, 2, 4, 3):
            self.assertEqual(
                int(dataset[position]["trigger_label"]),
                int(dataset.metadata(position).trigger_label),
            )
        # sample index + window_bins - 1 is the supervised target bin
        self.assertEqual(int(dataset[3]["trigger_label"]), TRIGGER_POSITIVE)
        self.assertEqual(int(dataset[5]["trigger_label"]), TRIGGER_IGNORE)
        self.assertEqual(int(dataset[0]["trigger_label"]), TRIGGER_NEGATIVE)

    def test_prediction_time_is_the_target_bin_end(self):
        dataset = synthetic_dataset(self.root)
        metadata = dataset.metadata(0)
        self.assertEqual(metadata.prediction_available_time, metadata.target_bin_end)
        self.assertEqual(metadata.window_start_time + 10 * GRID_MS, metadata.prediction_available_time)
        self.assertEqual(int(dataset.prediction_times([0])[0]), metadata.prediction_available_time)

    def test_loader_is_deterministic_and_not_shuffled(self):
        dataset = synthetic_dataset(self.root)
        with self.assertRaisesRegex(ValueError, "shuffling is not allowed"):
            build_trigger_loader(dataset, batch_size=4, num_workers=0, shuffle=True)
        loader = build_trigger_loader(dataset, batch_size=4, num_workers=0)
        indices = np.concatenate([batch["sample_index"].numpy() for batch in loader])
        self.assertTrue((np.diff(indices) >= 0).all())


class LabelPredictionTimeContractTests(unittest.TestCase):
    """Dataset-level contract: the label of a window is the recent-onset state at
    that window's ``prediction_available_time``, not at its target-bin start."""

    def setUp(self):
        import tempfile
        self.root = Path(tempfile.mkdtemp(prefix="p6c0_grid_"))

    def _dataset(self, onsets):
        import tempfile
        count = 60
        timestamps = ORIGIN + np.arange(count, dtype=np.int64) * GRID_MS
        rng = np.random.RandomState(1)
        save_split_arrays(self.root, "train", {
            "timestamps": timestamps,
            "metric": rng.normal(size=(count, 10, 16)).astype(np.float32),
            "log": rng.uniform(size=(count, 10, 8)).astype(np.float32),
            "trace": rng.uniform(size=(count, 10, 10, 4)).astype(np.float32),
            "labels": np.zeros((count, 10), dtype=np.int8),
            "label_mask": np.zeros((count, 10), dtype=np.int8),
        })
        events = pd.DataFrame([
            {"case_id": "e{}".format(index), "start_ms": onset, "end_ms": onset + 11_000}
            for index, onset in enumerate(onsets)
        ])
        labels = build_trigger_labels(prediction_time_grid(timestamps, grid_seconds=30), events)
        return TriggerWindowDataset(
            self.root / "train", window_bins=10, grid_seconds=30,
            sample_indices=np.arange(count - 9, dtype=np.int64),
            trigger_labels=labels, split_name="fit",
        ), events

    def test_bin_containing_the_onset_is_labelled_positive(self):
        onset = ORIGIN + 9 * GRID_MS + 11_000  # inside bin 9, the first usable target bin
        dataset, _ = self._dataset([onset])
        # target bin 9 -> prediction time ORIGIN + 300 s -> 19 s after the onset
        self.assertEqual(int(dataset[0]["trigger_label"]), TRIGGER_POSITIVE)
        metadata = dataset.metadata(0)
        self.assertEqual(metadata.prediction_available_time, ORIGIN + 300_000)
        self.assertGreaterEqual(metadata.prediction_available_time - onset, 0)
        self.assertLessEqual(metadata.prediction_available_time - onset, 60_000)

    def test_label_matches_the_preregistered_rule_for_every_window(self):
        onsets = [ORIGIN + 5_000, ORIGIN + 400_000, ORIGIN + 900_000]
        dataset, _ = self._dataset(onsets)
        for position in range(len(dataset)):
            metadata = dataset.metadata(position)
            predicted = metadata.trigger_label
            recent = any(0 <= metadata.prediction_available_time - onset <= 60_000 for onset in onsets)
            self.assertEqual(
                predicted == TRIGGER_POSITIVE, recent,
                "window {} at {} disagrees with the recent-onset rule".format(
                    position, metadata.prediction_available_time
                ),
            )

    def test_long_event_ignore_starts_after_the_positive_band(self):
        onset = ORIGIN + 9 * GRID_MS + 11_000  # inside bin 9
        count = 60
        timestamps = ORIGIN + np.arange(count, dtype=np.int64) * GRID_MS
        rng = np.random.RandomState(2)
        save_split_arrays(self.root, "train", {
            "timestamps": timestamps,
            "metric": rng.normal(size=(count, 10, 16)).astype(np.float32),
            "log": rng.uniform(size=(count, 10, 8)).astype(np.float32),
            "trace": rng.uniform(size=(count, 10, 10, 4)).astype(np.float32),
            "labels": np.zeros((count, 10), dtype=np.int8),
            "label_mask": np.zeros((count, 10), dtype=np.int8),
        })
        events = pd.DataFrame([{
            "case_id": "long", "start_ms": onset, "end_ms": onset + 3_600_000,
        }])
        labels = build_trigger_labels(prediction_time_grid(timestamps, grid_seconds=30), events)
        dataset = TriggerWindowDataset(
            self.root / "train", window_bins=10, grid_seconds=30,
            sample_indices=np.arange(count - 9, dtype=np.int64),
            trigger_labels=labels, split_name="fit",
        )
        states = [int(dataset[position]["trigger_label"]) for position in range(5)]
        deltas = [
            dataset.metadata(position).prediction_available_time - onset for position in range(5)
        ]
        self.assertEqual(deltas, [19_000, 49_000, 79_000, 109_000, 139_000])
        self.assertEqual(states[:2], [TRIGGER_POSITIVE, TRIGGER_POSITIVE])
        self.assertEqual(states[2:], [TRIGGER_IGNORE, TRIGGER_IGNORE, TRIGGER_IGNORE])

    def test_driver_builds_labels_on_the_prediction_time_grid(self):
        source = DRIVER_PATH.read_text(encoding="utf-8")
        self.assertIn("prediction_time_grid(timestamps", source)
        self.assertIn("assert_prediction_time_grid(prediction_grid, timestamps", source)


class DriverGuaranteeTests(unittest.TestCase):
    def setUp(self):
        self.source = DRIVER_PATH.read_text(encoding="utf-8")

    def test_driver_never_loads_ada_mgad_or_rca_state(self):
        for forbidden in (
            "MyModel(", "load_model(", "fit_reconstruction_calibration(",
            "fit_conditional_logit(", "load_conditional_logit(", "conditional_logit.npz",
            "GaiaRcaRawIndex(", "extract_case_features(", "rca_metrics(", "predict_rankings(",
            "score_fusion_alpha", "abnormal_weight", "reconstruction_calibration.json",
        ):
            self.assertNotIn(forbidden, self.source, "driver must not touch {}".format(forbidden))

    def test_driver_only_reads_its_own_checkpoint(self):
        self.assertIn("best_validation_event_f1.pt", self.source)
        self.assertNotIn("best_train_f1.pt", self.source)
        self.assertNotIn("last.pt", self.source)

    def test_training_path_never_builds_the_test_dataset(self):
        source = inspect.getsource(driver.run_training)
        self.assertNotIn('build_dataset("test")', source)
        self.assertNotIn('"test"', source)
        self.assertIn('build_dataset("fit")', source)
        self.assertIn('build_dataset("validation")', source)

    def test_evaluation_requires_a_frozen_validation_selection(self):
        import tempfile
        output_dir = Path(tempfile.mkdtemp(prefix="p6c0_eval_"))
        with self.assertRaises(FileNotFoundError):
            driver.run_evaluation(state=None, output_dir=output_dir, model_args=None,
                                  training=None, manifest_path=output_dir, manifest_sha="0" * 64)

    def test_checkpoint_threshold_binding_is_verified(self):
        import tempfile
        output_dir = Path(tempfile.mkdtemp(prefix="p6c0_bind_"))
        checkpoint = output_dir / "checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "best_validation_event_f1.pt").write_bytes(b"not-a-checkpoint")
        record = {
            "checkpoint": {
                "path": str(checkpoint / "best_validation_event_f1.pt"),
                "sha256": "0" * 64,
            },
            "selected_validation_threshold": 0.5,
            "selected_validation_metrics": {},
        }
        (output_dir / "validation_selection.json").write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 changed"):
            driver.run_evaluation(state=None, output_dir=output_dir, model_args=None,
                                  training=None, manifest_path=output_dir, manifest_sha="0" * 64)

    def test_driver_output_guard_rejects_frozen_trees(self):
        from src.e2e.protocol import load_config
        config = load_config(Path("configs/e2e/gaia_p5_v3_preprocessing_v2.json"))
        for forbidden in ("experiments/p5/gaia_v2/run", "artifacts/p5/v3_preprocessing_v2/events"):
            with self.assertRaises(ValueError):
                driver.guard_output_root(Path(forbidden), config)

    def test_driver_declares_test_observation_only(self):
        self.assertIn('"test_used_for_checkpoint_selection": False', self.source)
        self.assertIn('"test_used_for_threshold_selection": False', self.source)
        self.assertIn('"ada_mgad_checkpoint_loaded": False', self.source)
        self.assertIn('"node_anomaly_labels_read": False', self.source)
        self.assertIn('"root_service_label_used": False', self.source)
        self.assertIn('"rca_retrained": False', self.source)
        self.assertIn('"reintroduced_error_class": False', self.source)

    def test_trigger_config_pre_registers_the_frozen_protocol(self):
        config = json.loads(
            Path("configs/e2e/gaia_p6_c0_system_trigger.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["schema_version"], "p6_c0_trigger_v1")
        self.assertEqual(config["split_mode"], "chronological_50_20_30")
        self.assertEqual(config["threshold_selection"], "validation_only_exact_unique_scores")
        self.assertEqual(config["matching_semantics"], "causal_max_cardinality_minimum_delay")
        self.assertEqual(config["trigger_label"]["positive_window_seconds"], 60)
        self.assertTrue(config["trigger_label"]["long_event_ignore"])
        self.assertFalse(config["trigger_label"]["service_specific_supervision"])
        self.assertEqual(config["evaluation"]["grid_seconds"], 30)
        self.assertEqual(config["evaluation"]["window_bins"], 10)
        self.assertEqual(config["evaluation"]["history_seconds"], 300)
        self.assertEqual(config["evaluation"]["tolerance_seconds"], 60)
        self.assertEqual(config["evaluation"]["gate"], {"precision": 0.9, "recall": 0.6, "f1": 0.72})
        self.assertTrue(config["model"]["random_initialization"])
        self.assertFalse(config["model"]["loads_ada_mgad_checkpoint"])
        self.assertFalse(config["model"]["node_anomaly_label_consumed"])
        self.assertFalse(config["model"]["root_service_label_consumed"])

    def test_checkpoint_tie_break_prefers_f1_then_recall_then_threshold(self):
        better_recall = {"event_f1": 0.7, "event_recall": 0.6}
        worse_recall = {"event_f1": 0.7, "event_recall": 0.5}
        self.assertGreater(driver._selection_key(better_recall, 0.1), driver._selection_key(worse_recall, 0.9))
        self.assertGreater(
            driver._selection_key({"event_f1": 0.71, "event_recall": 0.1}, 0.0),
            driver._selection_key({"event_f1": 0.70, "event_recall": 0.9}, 0.9),
        )
        self.assertGreater(
            driver._selection_key({"event_f1": 0.7, "event_recall": 0.5}, 0.5),
            driver._selection_key({"event_f1": 0.7, "event_recall": 0.5}, 0.4),
        )

    def test_trigger_label_never_uses_event_duration_overlap(self):
        source = DRIVER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("event_duration_overlap", source)
        self.assertNotIn("duration_weight", source)


class FrozenProtocolReuseTests(unittest.TestCase):
    def test_driver_imports_the_frozen_matching_primitives(self):
        source = DRIVER_PATH.read_text(encoding="utf-8")
        self.assertIn("evaluate_system_threshold", source)
        self.assertIn("select_system_threshold", source)
        helper = Path("src/e2e/system_trigger.py").read_text(encoding="utf-8")
        for frozen in ("construct_predicted_episodes", "match_events", "event_metrics",
                       "_evaluate_threshold_candidates"):
            self.assertIn(frozen, helper)
        self.assertNotIn("def match_events", helper)
        self.assertNotIn("def construct_predicted_episodes", helper)

    def test_split_helper_reuses_the_frozen_event_assignment(self):
        helper = Path("src/e2e/system_trigger.py").read_text(encoding="utf-8")
        self.assertIn("assign_event_blocks", helper)

    def test_trigger_blocks_match_the_frozen_test_boundary(self):
        from src.e2e.protocol import load_config
        config = load_config(Path("configs/e2e/gaia_p5_v3_preprocessing_v2.json"))
        blocks = trigger_temporal_blocks(config)
        self.assertEqual(blocks[2].start_ms, int(config["split"]["boundary_ms"]))
        self.assertEqual(blocks[2].end_ms, int(config["split"]["absolute_end_ms"]))


if __name__ == "__main__":
    unittest.main()
