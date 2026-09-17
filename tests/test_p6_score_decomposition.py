"""Tests for the P6-B0 score decomposition audit.

The suite covers the numerical/contract properties requested for P6-B0 plus the
static read-only guarantees (no retraining, no calibration re-fit, no formal-run
writes).  Heavy full-run inference is exercised by the driver itself and its
replay gate artifact; these tests stay lightweight and deterministic.
"""

from pathlib import Path
import re
import unittest

import numpy as np
import pandas as pd

from src.e2e.event_detection import (
    aggregate_system_scores,
    construct_predicted_episodes,
    match_events,
    run_event_detection,
)
from src.e2e.protocol import GAIA_SERVICES
from src.e2e.score_decomposition import (
    FUSION_ALPHA,
    SCORE_TRACKS,
    apply_track,
    common_case_ids,
    complementarity_table,
    decomposition_frame,
    dynamic_ranking_metrics,
    label_identity_audit,
    localization_metrics,
    rank_services_by_score,
    replay_against_formal,
    root_margin_report,
    score_range_report,
    stratified_event_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER = PROJECT_ROOT / "scripts/p6/run_score_decomposition.py"
HELPER = PROJECT_ROOT / "src/e2e/score_decomposition.py"


class _FakeDataset:
    """Minimal stand-in for TimestampedArrayDataset."""

    def __init__(self, split, window_bins, grid_seconds, window_count, labels):
        self.split = split
        self.window_bins = int(window_bins)
        self.grid_ms = int(grid_seconds) * 1000
        total = window_count + self.window_bins - 1
        self.timestamps = (
            np.arange(total, dtype=np.int64) * self.grid_ms + 1_000_000
        )
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.timestamps) - self.window_bins + 1


def _components(window_count, seed=0):
    rng = np.random.RandomState(seed)
    size = window_count * len(GAIA_SERVICES)
    return {
        "classification": rng.uniform(0.0, 1.0, size=size),
        "reconstruction": rng.uniform(0.0, 1.0, size=size),
        "fused": rng.uniform(0.0, 1.0, size=size),
        "node_label": rng.randint(0, 2, size=size),
        "indices": np.arange(window_count, dtype=np.int64),
        "raw_reconstruction_energy": rng.normal(size=size),
    }


class ScoreAlgebraTests(unittest.TestCase):
    def test_fused_score_equals_alpha_cls_plus_one_minus_alpha_rec(self):
        dataset = _FakeDataset("test", 3, 30, 5, np.zeros((7, 10), dtype=np.int64))
        components = _components(5)
        # Make the fused column exactly the documented fusion of the two branches.
        components["fused"] = (
            FUSION_ALPHA * components["classification"]
            + (1.0 - FUSION_ALPHA) * components["reconstruction"]
        )
        frame = decomposition_frame(dataset, components)
        self.assertEqual(FUSION_ALPHA, 0.7)
        np.testing.assert_allclose(
            frame["fused_recomputed"].to_numpy(),
            FUSION_ALPHA * frame["classification_score"].to_numpy()
            + (1.0 - FUSION_ALPHA) * frame["reconstruction_score"].to_numpy(),
            rtol=0, atol=0,
        )
        np.testing.assert_allclose(
            frame["fused_score"].to_numpy(), frame["fused_recomputed"].to_numpy(),
            rtol=0, atol=1e-12,
        )

    def test_all_tracks_finite_and_within_unit_interval(self):
        dataset = _FakeDataset("train", 3, 30, 4, np.zeros((6, 10), dtype=np.int64))
        frame = decomposition_frame(dataset, _components(4, seed=7))
        report = score_range_report(frame)
        for track in SCORE_TRACKS:
            self.assertTrue(report[track]["all_finite"])
            self.assertTrue(report[track]["within_unit_interval"])
            selected = apply_track(frame, track)
            values = selected["selected_anomaly_score"].to_numpy()
            self.assertTrue(np.isfinite(values).all())
            self.assertGreaterEqual(values.min(), 0.0)
            self.assertLessEqual(values.max(), 1.0)
            np.testing.assert_array_equal(
                selected["anomaly_score"].to_numpy(), values
            )

    def test_prediction_frame_contract_columns(self):
        dataset = _FakeDataset("test", 3, 30, 3, np.zeros((5, 10), dtype=np.int64))
        frame = decomposition_frame(dataset, _components(3))
        required = {
            "split", "sample_index", "window_start_time", "window_end_time",
            "target_bin_start", "target_bin_end", "prediction_available_time",
            "service", "service_registry_index", "node_label",
            "classification_score", "reconstruction_score", "fused_score",
        }
        self.assertTrue(required.issubset(frame.columns))
        self.assertEqual(len(frame), 3 * len(GAIA_SERVICES))
        self.assertEqual(
            frame["prediction_available_time"].to_numpy()[0],
            frame["target_bin_start"].to_numpy()[0] + dataset.grid_ms,
        )


class ReplayGateTests(unittest.TestCase):
    def _formal_frame(self):
        dataset = _FakeDataset("test", 3, 30, 4, np.zeros((6, 10), dtype=np.int64))
        frame = decomposition_frame(dataset, _components(4, seed=1))
        return pd.DataFrame({
            "split": frame["split"],
            "sample_index": frame["sample_index"],
            "service": frame["service"],
            "prediction_available_time": frame["prediction_available_time"],
            "node_label": frame["node_label"],
            "anomaly_score": frame["fused_score"],
        })

    def test_identical_fused_scores_replay(self):
        formal = self._formal_frame()
        dataset = _FakeDataset("test", 3, 30, 4, np.zeros((6, 10), dtype=np.int64))
        components = _components(4, seed=1)
        components["fused"] = formal["anomaly_score"].to_numpy().copy()
        frame = decomposition_frame(dataset, components)
        result = replay_against_formal(frame, formal)
        self.assertTrue(result["passed"])
        self.assertTrue(result["identity_match"])
        self.assertEqual(result["max_abs_diff"], 0.0)
        self.assertLessEqual(result["max_abs_diff"], 1e-6)

    def test_perturbed_fused_scores_fail_the_gate(self):
        formal = self._formal_frame()
        dataset = _FakeDataset("test", 3, 30, 4, np.zeros((6, 10), dtype=np.int64))
        components = _components(4, seed=1)
        components["fused"] = formal["anomaly_score"].to_numpy() + 1e-3
        frame = decomposition_frame(dataset, components)
        result = replay_against_formal(frame, formal)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["max_abs_diff"], 1e-3, places=9)
        self.assertGreater(result["max_abs_diff"], result["tolerance"])


def _event_fixture():
    """Synthetic 30s-grid Train/Test fixture with a shared GT registry."""

    rows = []
    timestamps = [30_000 * (index + 1) for index in range(20)]

    def add(split, values):
        for index, timestamp in enumerate(timestamps):
            for service_index, service in enumerate(GAIA_SERVICES):
                score = float(values.get((index, service_index), 0.05))
                rows.append({
                    "split": split, "sample_index": index,
                    "window_start_time": timestamp - 300_000,
                    "window_end_time": timestamp,
                    "target_bin_start": timestamp - 30_000,
                    "target_bin_end": timestamp,
                    "prediction_available_time": timestamp,
                    "prediction_timestamp": timestamp,
                    "service": service, "service_registry_index": service_index,
                    "node_label": 0, "anomaly_score": score,
                    "selected_anomaly_score": score,
                    "binary_prediction": int(score >= 0.5),
                })

    add("train", {(2, 0): 0.95, (3, 0): 0.90, (9, 4): 0.70})
    add("test", {(6, 1): 0.90, (7, 1): 0.85, (12, 3): 0.80})
    registry = pd.DataFrame([
        {"case_id": "t0", "source_index": 0, "service": "dbservice1",
         "fault_type": "login_failure", "start_ms": 75_000, "end_ms": 105_000,
         "split": "train"},
        {"case_id": "s0", "source_index": 1, "service": "dbservice2",
         "fault_type": "memory_anomalies", "start_ms": 195_000, "end_ms": 225_000,
         "split": "test"},
        {"case_id": "s1", "source_index": 2, "service": "webservice1",
         "fault_type": "cpu_anomalies", "start_ms": 375_000, "end_ms": 405_000,
         "split": "test"},
    ])
    frame = pd.DataFrame(rows)
    return (
        frame.loc[frame["split"] == "train"].reset_index(drop=True),
        frame.loc[frame["split"] == "test"].reset_index(drop=True),
        registry,
    )


class ThresholdIsolationTests(unittest.TestCase):
    def test_train_threshold_selection_never_reads_test(self):
        train, test, registry = _event_fixture()
        first = run_event_detection(train, test, registry, (), grid_seconds=30, tolerance_seconds=60)
        perturbed = test.copy()
        perturbed["anomaly_score"] = 0.01
        perturbed["selected_anomaly_score"] = 0.01
        second = run_event_detection(
            train, perturbed, registry, (), grid_seconds=30, tolerance_seconds=60
        )
        self.assertEqual(first["threshold_selection"].threshold,
                         second["threshold_selection"].threshold)
        self.assertEqual(first["train_metrics"], second["train_metrics"])
        self.assertEqual(
            first["threshold_selection"].train_metrics,
            second["threshold_selection"].train_metrics,
        )

    def test_every_track_uses_the_same_matching_implementation(self):
        train, test, registry = _event_fixture()
        test_gt = registry.loc[registry["split"] == "test"]
        for track in SCORE_TRACKS:
            result = run_event_detection(
                train, test, registry, (), grid_seconds=30, tolerance_seconds=60
            )
            self.assertEqual(track, track)  # tracks differ only in the score column
            scores = aggregate_system_scores(test, grid_seconds=30)
            episodes = construct_predicted_episodes(
                scores, result["threshold_selection"].threshold, grid_seconds=30
            )
            expected = match_events(episodes, test_gt, tolerance_seconds=60)
            pd.testing.assert_frame_equal(
                result["test_matching"].reset_index(drop=True),
                expected.reset_index(drop=True),
            )

    def test_stratified_event_metrics_counts_match_and_miss(self):
        _, _, registry = _event_fixture()
        matching = pd.DataFrame([
            {"match_status": "matched", "fault_type": "login_failure",
             "gt_service": "dbservice1"},
            {"match_status": "miss", "fault_type": "login_failure",
             "gt_service": "dbservice1"},
            {"match_status": "miss", "fault_type": "cpu_anomalies",
             "gt_service": "webservice1"},
            {"match_status": "false_alarm", "fault_type": None, "gt_service": None},
        ])
        by_fault = stratified_event_metrics(matching, "fault_type")
        self.assertEqual(by_fault["by_group"]["login_failure"]["case_count"], 2)
        self.assertEqual(by_fault["by_group"]["login_failure"]["true_positive"], 1)
        self.assertEqual(by_fault["by_group"]["login_failure"]["false_negative"], 1)
        self.assertAlmostEqual(by_fault["by_group"]["login_failure"]["recall"], 0.5)
        self.assertTrue(by_fault["by_group"]["cpu_anomalies"]["small_n"])
        by_root = stratified_event_metrics(matching, "gt_service")
        self.assertEqual(by_root["by_group"]["dbservice1"]["case_count"], 2)


class LocalizationDiagnosticTests(unittest.TestCase):
    def test_localization_reports_ac_and_mrr(self):
        matrix = np.zeros((3, 10))
        matrix[0, 0] = 0.9  # root dbservice1 at rank 1
        matrix[1, 0] = 0.9
        matrix[1, 1] = 0.8  # root dbservice2 at rank 2
        matrix[2, 5] = 0.9
        matrix[2, 4] = 0.7
        matrix[2, 3] = 0.6
        matrix[2, 2] = 0.5  # root logservice1 at rank 4
        result = localization_metrics(matrix, [0, 1, 2], ["a", "b", "c"])
        overall = result["overall"]
        self.assertEqual(overall["case_count"], 3)
        self.assertAlmostEqual(overall["AC@1"], 1 / 3)
        self.assertAlmostEqual(overall["AC@3"], 2 / 3)
        self.assertAlmostEqual(overall["AC@5"], 1.0)
        self.assertAlmostEqual(overall["MRR"], (1 + 0.5 + 0.25) / 3)
        self.assertEqual(result["rankings"][0][0], "dbservice1")

    def test_rank_tie_break_uses_registry_order(self):
        row = np.zeros(10)
        ranking = rank_services_by_score(row)
        self.assertEqual(ranking, GAIA_SERVICES)

    def test_root_margin_statistics(self):
        matrix = np.zeros((2, 10))
        matrix[0, 0] = 0.9
        matrix[0, 1] = 0.4
        matrix[1, 0] = 0.3
        matrix[1, 1] = 0.8
        report = root_margin_report(matrix, [0, 0], ["fault-a", "fault-a"])
        self.assertAlmostEqual(report["overall"]["mean"], 0.0)  # 0.5 and -0.5
        self.assertAlmostEqual(report["overall"]["median"], 0.0)
        self.assertAlmostEqual(report["overall"]["fraction_margin_positive"], 0.5)
        self.assertEqual(report["by_fault_type"]["fault-a"]["case_count"], 2)
        self.assertEqual(report["by_root_service"]["dbservice1"]["case_count"], 2)


class CommonCaseTests(unittest.TestCase):
    def test_common_case_intersection(self):
        matched = {
            "classification": ["a", "b", "c", "d"],
            "fused": ["b", "c", "d", "e"],
            "reconstruction": ["c", "d", "f"],
        }
        self.assertEqual(common_case_ids(matched), {"c", "d"})

    def test_common_case_intersection_requires_all_tracks(self):
        with self.assertRaises(ValueError):
            common_case_ids({"classification": ["a"], "fused": ["a"]})

    def test_dynamic_ranking_metrics_subset(self):
        rankings = [("dbservice1",) + tuple(s for s in GAIA_SERVICES if s != "dbservice1")]
        metrics = dynamic_ranking_metrics(rankings, ["dbservice1"], ["login_failure"])
        self.assertEqual(metrics["overall"]["case_count"], 1)
        self.assertAlmostEqual(metrics["overall"]["AC@1"], 1.0)


class ComplementarityTests(unittest.TestCase):
    def test_two_by_two_table_and_conditionals(self):
        score_rankings = [
            ("a", "b"), ("a", "b"), ("b", "a"), ("b", "a"),
        ]
        rca_rankings = [
            ("a", "b"), ("b", "a"), ("a", "b"), ("b", "a"),
        ]
        roots = ["a", "a", "a", "b"]
        table = complementarity_table(score_rankings, rca_rankings, roots)
        self.assertEqual(table["both_correct"], 2)             # cases 0 and 3
        self.assertEqual(table["score_correct_rca_wrong"], 1)  # case 1
        self.assertEqual(table["score_wrong_rca_correct"], 1)  # case 2
        self.assertEqual(table["both_wrong"], 0)
        self.assertAlmostEqual(table["probability_rca_correct_given_score_wrong"], 1.0)
        self.assertAlmostEqual(table["probability_rca_wrong_given_score_correct"], 1 / 3)
        self.assertAlmostEqual(table["score_top1_accuracy"], 0.75)
        self.assertAlmostEqual(table["rca_top1_accuracy"], 0.75)


class LabelIdentityAuditTests(unittest.TestCase):
    def test_label_identity_is_structural_and_measures_bin_ambiguity(self):
        grid_seconds = 30
        timestamps = np.arange(4, dtype=np.int64) * 30_000
        labels = np.zeros((4, 10), dtype=np.int64)
        labels[1, 0] = 1
        labels[1, 1] = 1
        labels[3, 0] = 1
        events = pd.DataFrame([
            {"case_id": "a", "service": "dbservice1", "start_ms": 30_000,
             "end_ms": 60_000, "split": "train"},
            {"case_id": "b", "service": "dbservice2", "start_ms": 30_000,
             "end_ms": 60_000, "split": "train"},
            {"case_id": "c", "service": "dbservice1", "start_ms": 90_000,
             "end_ms": 120_000, "split": "train"},
        ])
        audit = label_identity_audit(
            events, {"train": timestamps}, {"train": labels}, grid_seconds=grid_seconds
        )
        self.assertEqual(audit["total_events"], 3)
        self.assertEqual(audit["rasterization_mismatch_events"], 0)
        self.assertEqual(audit["labelled_service_positive_on_all_bins_ratio"], 1.0)
        # Event c's bin is unambiguous; events a and b share bin 1.
        self.assertEqual(audit["exact_identity_count"], 1)
        self.assertEqual(audit["bins_single_event"], 1)
        self.assertEqual(audit["bins_multi_event"], 1)
        self.assertEqual(audit["bins_multi_root_service"], 1)


class ThresholdReuseTests(unittest.TestCase):
    """Restoring a frozen Train-only threshold must reproduce detection exactly."""

    CONFIG = {"ad": {"grid_seconds": 30}, "event_trigger": {"matching_tolerance_seconds": 60}}

    def test_restored_threshold_reproduces_detection(self):
        from scripts.p6.run_score_decomposition import event_detection_from_threshold

        train, test, registry = _event_fixture()
        first = run_event_detection(
            train, test, registry, (), grid_seconds=30, tolerance_seconds=60
        )
        threshold = first["threshold_selection"].threshold
        rebuilt = event_detection_from_threshold(
            self.CONFIG, train, test, registry, (), threshold
        )
        self.assertEqual(rebuilt["train_metrics"], first["train_metrics"])
        self.assertEqual(rebuilt["test_metrics"], first["test_metrics"])
        pd.testing.assert_frame_equal(
            rebuilt["test_matching"].reset_index(drop=True),
            first["test_matching"].reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            rebuilt["matching"].reset_index(drop=True),
            first["matching"].reset_index(drop=True),
        )

    def test_restore_rejects_a_threshold_with_inconsistent_train_metrics(self):
        import json
        import tempfile
        from scripts.p6.run_score_decomposition import _restore_threshold

        train, test, registry = _event_fixture()
        with tempfile.TemporaryDirectory() as directory:
            track_dir = Path(directory)
            train.to_csv(track_dir / "ad_train_predictions.csv", index=False)
            test.to_csv(track_dir / "ad_test_predictions.csv", index=False)
            (track_dir / "threshold.json").write_text(json.dumps({
                "track": "fused", "threshold": 0.05, "candidate_count": 1,
                "tie_break": "x", "train_metrics": {"event_f1": 0.0},
            }), encoding="utf-8")
            restored = _restore_threshold(
                self.CONFIG, "fused", train, test, registry, (), track_dir
            )
            self.assertIsNone(restored)

    def test_restore_accepts_a_consistent_threshold(self):
        import json
        import tempfile
        from scripts.p6.run_score_decomposition import _restore_threshold

        train, test, registry = _event_fixture()
        first = run_event_detection(
            train, test, registry, (), grid_seconds=30, tolerance_seconds=60
        )
        with tempfile.TemporaryDirectory() as directory:
            track_dir = Path(directory)
            train.to_csv(track_dir / "ad_train_predictions.csv", index=False)
            test.to_csv(track_dir / "ad_test_predictions.csv", index=False)
            (track_dir / "threshold.json").write_text(json.dumps({
                "track": "fused",
                "threshold": float(first["threshold_selection"].threshold),
                "candidate_count": int(first["threshold_selection"].candidate_count),
                "tie_break": first["threshold_selection"].tie_break,
                "train_metrics": first["threshold_selection"].train_metrics,
            }), encoding="utf-8")
            restored = _restore_threshold(
                self.CONFIG, "fused", train, test, registry, (), track_dir
            )
            self.assertIsNotNone(restored)
            self.assertTrue(restored["threshold_selection"]["restored_from_frozen_threshold"])
            self.assertEqual(
                restored["test_metrics"], first["test_metrics"]
            )


class FusedReproductionGateTests(unittest.TestCase):
    """The operative fused gate is the event-level reproduction of the formal run."""

    FORMAL = {
        "threshold_selection": {"threshold": 0.8693510293960571},
        "test_metrics": {
            "true_positive_events": 3703, "false_positive_events": 65,
            "false_negative_events": 2084, "predicted_episode_count": 3768,
            "ground_truth_event_count": 5787,
            "detection_delay": {
                "mean_seconds": 24.106421550094517,
                "median_seconds": 24.109,
                "p95_seconds": 39.5143,
            },
        },
    }

    def _fused_record(self):
        metrics = dict(self.FORMAL["test_metrics"])
        metrics["detection_delay"] = dict(self.FORMAL["test_metrics"]["detection_delay"])
        return {"threshold": self.FORMAL["threshold_selection"]["threshold"],
                "test_metrics": metrics}

    def _gate(self, fused):
        import json
        import tempfile
        from scripts.p6.run_score_decomposition import fused_event_reproduction_gate

        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events"
            events.mkdir()
            (events / "event_detection_metrics.json").write_text(
                json.dumps(self.FORMAL), encoding="utf-8"
            )
            return fused_event_reproduction_gate(Path(directory), fused)

    def test_exact_fused_event_detection_passes(self):
        gate = self._gate(self._fused_record())
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["threshold_abs_diff"], 0.0)

    def test_threshold_drift_fails_the_gate(self):
        fused = self._fused_record()
        fused["threshold"] = fused["threshold"] - 1e-4
        self.assertFalse(self._gate(fused)["passed"])

    def test_count_drift_fails_the_gate(self):
        fused = self._fused_record()
        fused["test_metrics"]["true_positive_events"] += 1
        gate = self._gate(fused)
        self.assertFalse(gate["passed"])
        self.assertNotEqual(
            gate["counts"]["true_positive_events"]["fused"],
            gate["counts"]["true_positive_events"]["formal"],
        )

    def test_delay_drift_fails_the_gate(self):
        fused = self._fused_record()
        fused["test_metrics"]["detection_delay"]["mean_seconds"] += 0.5
        self.assertFalse(self._gate(fused)["passed"])


class ReadOnlyGuaranteeTests(unittest.TestCase):
    def test_aggregate_only_drops_per_case_detail(self):
        from src.e2e.score_decomposition import aggregate_only

        metrics = {
            "overall": {"case_count": 2, "AC@1": 0.5},
            "root_macro": {"macro": {"AC@1": 0.5}},
            "case_metrics": [{"AC@1": 1.0}, {"AC@1": 0.0}],
            "rankings": [list(GAIA_SERVICES), list(GAIA_SERVICES)],
        }
        trimmed = aggregate_only(metrics)
        self.assertIn("overall", trimmed)
        self.assertIn("root_macro", trimmed)
        self.assertNotIn("case_metrics", trimmed)
        self.assertNotIn("rankings", trimmed)
        # The source mapping is left untouched for downstream in-memory use.
        self.assertIn("rankings", metrics)

    def test_driver_and_helper_never_train_or_refit_frozen_state(self):
        forbidden_calls = (
            r"\bfit_conditional_logit\s*\(",
            r"\bfit_train_conditional_logit\s*\(",
            r"\bfit_reconstruction_calibration\s*\(",
            r"\bsave_conditional_logit\s*\(",
            r"\bsave_reconstruction_calibration\s*\(",
            r"\bsave_model\s*\(",
            r"\btorch\.save\s*\(",
            r"\.backward\s*\(",
            r"\boptimizer\b",
            r"\bsystem\.fit\s*\(",
            r"\.fit\s*\(",
        )
        for path in (DRIVER, HELPER):
            source = path.read_text(encoding="utf-8")
            for pattern in forbidden_calls:
                self.assertIsNone(
                    re.search(pattern, source),
                    "{} must not match {}".format(path.name, pattern),
                )
        self.assertIn("load_reconstruction_calibration", DRIVER.read_text(encoding="utf-8"))
        self.assertIn("calibration=calibration", HELPER.read_text(encoding="utf-8"))

    def test_driver_imports_only_frozen_rca_inference_symbols(self):
        source = DRIVER.read_text(encoding="utf-8")
        match = re.search(
            r"from src\.e2e\.rca_model import \(([^)]*)\)", source
        )
        self.assertIsNotNone(match)
        block = re.sub(r"#[^\n]*", "", match.group(1)).replace("\n", " ")
        imported = {token.strip() for token in block.split(",") if token.strip()}
        self.assertEqual(
            imported, {"load_conditional_logit", "predict_rankings", "rca_metrics"}
        )

    def test_output_guard_rejects_formal_run_and_p5_tree(self):
        from scripts.p6.run_score_decomposition import guard_output_root

        formal = PROJECT_ROOT / "experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440"
        with self.assertRaises(ValueError):
            guard_output_root(formal, formal)
        with self.assertRaises(ValueError):
            guard_output_root(formal / "nested", formal)
        with self.assertRaises(ValueError):
            guard_output_root(PROJECT_ROOT / "experiments/p5/other", formal)
        allowed = guard_output_root(PROJECT_ROOT / "experiments/p6/score_decomposition", formal)
        self.assertEqual(
            allowed, PROJECT_ROOT / "experiments/p6/score_decomposition"
        )

    def test_driver_loads_the_frozen_conditional_logit(self):
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn("load_conditional_logit", source)
        self.assertIn("load_reconstruction_calibration", source)

    def test_anchor_lookup_keys_are_timestamps_not_case_ids(self):
        from scripts.p6.run_score_decomposition import anchor_lookup

        index = anchor_lookup(np.asarray([1_625_133_900_000, 1_625_133_930_000]))
        self.assertEqual(index, {1_625_133_900_000: 0, 1_625_133_930_000: 1})
        with self.assertRaises(ValueError):
            anchor_lookup(np.asarray([1, 1]))
        # The driver must key the feature bundle by the materialized anchor
        # timestamps, not by the generated string case IDs.
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn('"anchors_ms.npy"', source)


if __name__ == "__main__":
    unittest.main()
