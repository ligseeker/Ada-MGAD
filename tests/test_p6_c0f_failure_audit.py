"""P6-C0F failure-accounting, trajectory and guard contract tests."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.e2e.event_detection import construct_predicted_episodes, match_events
from src.e2e.system_trigger import (
    TRIGGER_IGNORE,
    TRIGGER_NEGATIVE,
    TRIGGER_POSITIVE,
    system_score_frame,
)
from src.e2e.system_trigger_failure_audit import (
    FAILURE_CATEGORIES,
    build_failure_ledger,
    build_trajectories,
    collision_audit,
    confounding_tables,
    stratified_summary,
)

import scripts.p6.audit_c0_failure as driver


ORIGIN = 1625133600000
GRID_MS = 30_000


def slot_times(count):
    return ORIGIN + np.arange(count, dtype=np.int64) * GRID_MS


def events_frame(records):
    rows = []
    for index, record in enumerate(records):
        onset = ORIGIN + record["onset_ms"]
        rows.append({
            "case_id": record.get("case_id", "ev{:03d}".format(index)),
            "source_index": index,
            "service": record.get("service", "mobservice1"),
            "fault_type": record.get("fault_type", "login_failure"),
            "start_ms": int(onset),
            "end_ms": int(onset + record.get("duration_ms", 11_000)),
        })
    return pd.DataFrame(rows)


def pipeline(slots, binary, ground_truth, tolerance_seconds=60):
    frame = system_score_frame("audit", slots, np.asarray(binary, dtype=float))
    episodes = construct_predicted_episodes(frame, 0.5)
    matching = match_events(episodes, ground_truth, tolerance_seconds=tolerance_seconds)
    return episodes, matching


def ledger_for(slots, scores, ground_truth, threshold=0.5, tolerance_seconds=60):
    binary = (np.asarray(scores, dtype=float) >= float(threshold)).astype(float)
    episodes, matching = pipeline(slots, binary, ground_truth, tolerance_seconds)
    ledger, summary = build_failure_ledger(
        ground_truth, split="test", slot_times_ms=slots,
        slot_scores=np.asarray(scores, dtype=float), threshold=threshold,
        episode_anchors_ms=episodes["t_hat"].to_numpy(dtype=np.int64),
        episode_end_times_ms=episodes["episode_end_time"].to_numpy(dtype=np.int64),
        matching=matching, origin_ms=int(slots[0]),
    )
    return ledger, summary, episodes, matching


class FailureLedgerTests(unittest.TestCase):
    def test_categories_are_mutually_exclusive_and_complete(self):
        slots = slot_times(12)
        ground_truth = events_frame([
            {"onset_ms": 5_000},                              # matched by the slot at +30 s
            {"onset_ms": 400_000},                            # no legal prediction (past the grid)
            {"onset_ms": 95_000},                             # below threshold
        ])
        scores = np.zeros(len(slots))
        scores[1] = 0.9   # +30 s slot -> 25 s after the onset
        ledger, summary, episodes, matching = ledger_for(slots, scores, ground_truth)
        self.assertEqual(summary["category_counts"]["MATCHED"], 1)
        self.assertEqual(summary["category_counts"]["NO_LEGAL_PREDICTION"], 1)
        self.assertEqual(summary["category_counts"]["BELOW_THRESHOLD"], 1)
        self.assertEqual(sum(summary["category_counts"].values()), len(ground_truth))
        self.assertTrue(summary["categories_sum_to_total"])
        self.assertTrue(summary["matched_equals_tp"])
        self.assertTrue(summary["unmatched_events_equal_fn"])
        self.assertTrue(summary["unmatched_episodes_equal_fp"])
        self.assertTrue(set(ledger["failure_category"]).issubset(set(FAILURE_CATEGORIES)))

    def test_episode_started_before_the_onset_is_not_a_new_episode(self):
        slots = slot_times(14)
        ground_truth = events_frame([{"onset_ms": 155_000}])  # inside bin 5
        scores = np.zeros(len(slots))
        scores[4:9] = 0.9  # one long episode from +120 s to +240 s, started before the onset
        ledger, summary, episodes, matching = ledger_for(slots, scores, ground_truth)
        row = ledger.iloc[0]
        self.assertEqual(row["failure_category"], "NO_NEW_EPISODE")
        self.assertTrue(bool(row["episode_active_before_onset"]))
        self.assertGreater(int(row["positive_slots"]), 0)
        self.assertEqual(int(row["candidate_episode_count"]), 0)
        self.assertIsNone(row["matched_prediction_id"])
        self.assertIsNotNone(row["first_positive_time"])

    def test_matching_competition_is_auditable(self):
        slots = slot_times(12)
        ground_truth = events_frame([{"onset_ms": 5_000}, {"onset_ms": 25_000}])
        scores = np.zeros(len(slots))
        scores[1] = 0.9  # single anchor inside both windows
        ledger, summary, episodes, matching = ledger_for(slots, scores, ground_truth)
        self.assertEqual(int(summary["category_counts"]["MATCHED"]), 1)
        self.assertEqual(int(summary["category_counts"]["MATCHING_COMPETITION"]), 1)
        competitor = ledger.loc[ledger["failure_category"] == "MATCHING_COMPETITION"].iloc[0]
        self.assertGreaterEqual(int(competitor["candidate_episode_count"]), 1)
        self.assertGreater(int(competitor["positive_slots"]), 0)

    def test_short_event_without_grid_point_is_not_zero_scored(self):
        slots = slot_times(3)
        ground_truth = events_frame([{"onset_ms": 100, "duration_ms": 2_900}])
        # the only slots are at +0 s, +30 s and +60 s, all inside the 60 s window
        ledger, summary, _, _ = ledger_for(slots, np.array([0.0, 0.0, 0.0]), ground_truth)
        self.assertEqual(ledger.iloc[0]["failure_category"], "BELOW_THRESHOLD")
        self.assertIsNone(ledger.iloc[0]["first_positive_time"])
        self.assertIsNone(ledger.iloc[0]["first_positive_delta_ms"])
        far = events_frame([{"onset_ms": 400_000}])
        ledger_far, _, _, _ = ledger_for(slots, np.zeros(3), far)
        self.assertEqual(ledger_far.iloc[0]["failure_category"], "NO_LEGAL_PREDICTION")
        self.assertEqual(int(ledger_far.iloc[0]["eligible_slots"]), 0)

    def test_late_response_from_another_onset_does_not_become_a_tp(self):
        slots = slot_times(14)
        ground_truth = events_frame([{"onset_ms": 5_000}, {"onset_ms": 200_000}])
        scores = np.zeros(len(slots))
        scores[7] = 0.9  # +210 s: far outside onset A, 10 s after onset B
        ledger, summary, _, _ = ledger_for(slots, scores, ground_truth)
        by_id = ledger.set_index("case_id")
        self.assertEqual(by_id.loc["ev001", "failure_category"], "MATCHED")
        self.assertEqual(by_id.loc["ev000", "failure_category"], "BELOW_THRESHOLD")
        _, response = build_trajectories(
            ledger, split="test", slot_times_ms=slots, slot_scores=scores,
            slot_logits=np.log(np.clip(scores, 1e-6, 1.0)), threshold=0.5,
        )
        self.assertEqual(str(response.set_index("case_id").loc["ev001", "response_band"]), "le_60s")
        trajectories, _ = build_trajectories(
            ledger, split="test", slot_times_ms=slots, slot_scores=scores,
            slot_logits=np.log(np.clip(scores, 1e-6, 1.0)), threshold=0.5,
        )
        late = trajectories.loc[
            (trajectories["case_id"] == "ev000")
            & (trajectories["prediction_available_time"] == int(slots[7]))
        ]
        self.assertEqual(int(late["other_onset_count_60s"].iloc[0]), 1)
        self.assertTrue(bool(late["in_other_recent_onset_window"].iloc[0]))

    def test_censoring_is_distinguished_from_a_complete_no_response(self):
        slots = slot_times(10)
        covered = events_frame([{"onset_ms": 5_000, "duration_ms": 60_000}])
        cut = events_frame([{"onset_ms": 5_000, "duration_ms": 600_000}])
        ledger_complete, _, _, _ = ledger_for(slots, np.zeros(len(slots)), covered)
        ledger_cut, _, _, _ = ledger_for(slots, np.zeros(len(slots)), cut)
        self.assertEqual(ledger_complete.iloc[0]["response_band"], "never")
        self.assertFalse(bool(ledger_complete.iloc[0]["censored_by_score_coverage"]))
        self.assertEqual(ledger_cut.iloc[0]["response_band"], "censored")
        self.assertTrue(bool(ledger_cut.iloc[0]["censored_by_score_coverage"]))

    def test_boundary_onsets_and_bin_multiplicity_are_recorded(self):
        slots = slot_times(12)
        ground_truth = events_frame([
            {"onset_ms": 0}, {"onset_ms": 5_000}, {"onset_ms": 34_000},
        ])
        ledger, _, _, _ = ledger_for(slots, np.zeros(len(slots)), ground_truth)
        self.assertTrue(bool(ledger.iloc[0]["onset_on_grid_boundary"]))
        self.assertFalse(bool(ledger.iloc[1]["onset_on_grid_boundary"]))
        self.assertEqual(set(ledger["onset_bin"].tolist()), {0, 1})
        self.assertEqual(sorted(ledger["onset_bin_count"].tolist()), [1, 2, 2])


class TrajectoryTests(unittest.TestCase):
    def test_trajectory_windows_and_concurrency_markers(self):
        slots = slot_times(20)
        ground_truth = events_frame([{"onset_ms": 5_000}, {"onset_ms": 275_000}])
        scores = np.zeros(len(slots))
        scores[10] = 0.9  # +300 s: 295 s after onset A, 25 s after onset B
        ledger, _, _, _ = ledger_for(slots, scores, ground_truth)
        trajectories, summary = build_trajectories(
            ledger, split="test", slot_times_ms=slots, slot_scores=scores,
            slot_logits=np.log(np.clip(scores, 1e-6, 1.0)), threshold=0.5,
        )
        first = trajectories.loc[trajectories["case_id"] == "ev000"]
        self.assertTrue(set(first["window_tag"]).issubset({"pre_onset", "response", "early", "mid", "event_rest"}))
        self.assertEqual(int(first.loc[first["delta_ms"] < 0, "binary_prediction"].sum()), 0)
        marker = first.loc[first["prediction_available_time"] == int(slots[10])]
        self.assertEqual(int(marker["other_onset_count_60s"].iloc[0]), 1)
        self.assertTrue(bool(marker["in_other_recent_onset_window"].iloc[0]))
        row = summary.set_index("case_id").loc["ev000"]
        self.assertIn("response_max_score", row.index)
        self.assertEqual(row["response_band"], "120_300s")

    def test_empty_window_is_null_not_zero(self):
        slots = slot_times(4)  # +0 s .. +90 s
        ground_truth = events_frame([{"onset_ms": 5_000}])
        ledger, _, _, _ = ledger_for(slots, np.zeros(len(slots)), ground_truth)
        _, summary = build_trajectories(
            ledger, split="test", slot_times_ms=slots, slot_scores=np.zeros(len(slots)),
            slot_logits=np.zeros(len(slots)), threshold=0.5,
        )
        row = summary.iloc[0]
        self.assertEqual(int(row["mid_count"]), 0)
        self.assertIsNone(row["mid_max_score"])
        self.assertEqual(int(row["mid_positive"]), 0)
        self.assertGreater(int(row["response_count"]), 0)
        self.assertEqual(int(row["response_positive"]), 0)


class StratificationTests(unittest.TestCase):
    def _ledger(self):
        slots = slot_times(12)
        ground_truth = events_frame([
            {"onset_ms": 5_000, "service": "mobservice1"},
            {"onset_ms": 95_000, "service": "dbservice1", "fault_type": "memory_anomalies",
             "duration_ms": 600_000},
        ])
        scores = np.zeros(len(slots))
        scores[1] = 0.9  # +30 s: 25 s after the first onset
        return ledger_for(slots, scores, ground_truth)[0]

    def test_stratified_summary_keeps_empty_strata_and_category_mix(self):
        report = stratified_summary(self._ledger(), split="test")
        strata = {row["duration_stratum"]: row for row in report["by_duration_stratum"]}
        self.assertEqual(strata["le_15s"]["n"], 1)
        self.assertEqual(strata["gt_300s"]["n"], 1)
        self.assertEqual(strata["15_30s"]["n"], 0)
        self.assertIsNone(strata["15_30s"]["recall"])
        self.assertEqual(set(strata["le_15s"]["failure_mix"]), set(FAILURE_CATEGORIES))
        self.assertEqual(report["response_band_totals"]["le_60s"], 1)

    def test_confounding_tables_expose_duration_fault_service_links(self):
        ground_truth = events_frame([
            {"onset_ms": 5_000, "service": "mobservice1", "fault_type": "login_failure"},
            {"onset_ms": 100_000, "service": "dbservice1", "fault_type": "memory_anomalies",
             "duration_ms": 600_000},
        ])
        tables = confounding_tables(ground_truth, origin_ms=ORIGIN)
        pairs = {(row["fault_type"], row["duration_stratum"]) for row in tables["fault_x_duration"]}
        self.assertIn(("login_failure", "le_15s"), pairs)
        self.assertIn(("memory_anomalies", "gt_300s"), pairs)
        self.assertEqual(tables["observed_combination_count"], 2)
        # the full cartesian size is computed from the observed marginal cardinalities
        self.assertEqual(tables["cartesian_product_size"], 2 * 5 * 2)

    def test_collision_audit_counts_bins_and_boundary_onsets(self):
        ground_truth = events_frame([
            {"onset_ms": 0}, {"onset_ms": 5_000}, {"onset_ms": 30_000}, {"onset_ms": 65_000},
        ])
        report = collision_audit(ground_truth, population_name="test-complete", origin_ms=ORIGIN)
        self.assertEqual(report["events"], 4)
        self.assertEqual(report["onsets_on_grid_boundary"], 2)
        self.assertEqual(report["bins_with_exactly_two_onsets"], 1)
        self.assertEqual(report["events_in_two_onset_bins"], 2)
        self.assertEqual(report["events_in_multi_onset_bins"], 2)
        self.assertEqual(report["distinct_onset_bins"], 3)


class DriverGuardTests(unittest.TestCase):
    def test_guard_rejects_frozen_roots_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen = Path(tmp) / "frozen"
            frozen.mkdir()
            with self.assertRaises(ValueError):
                driver.guard_output_dir(frozen / "audit", frozen_roots=[frozen], require_empty=True)
            existing = Path(tmp) / "run"
            existing.mkdir()
            (existing / "run_state.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                driver.guard_output_dir(existing, frozen_roots=[frozen], require_empty=True)
            fresh = Path(tmp) / "fresh"
            driver.guard_output_dir(fresh, frozen_roots=[frozen], require_empty=True)

    def test_every_stage_refuses_a_frozen_output_root(self):
        for frozen in (driver.C0_ROOT, driver.P5_RUN_ROOT, driver.SHARED_INPUT_ROOTS[0],
                       driver.SHARED_INPUT_ROOTS[1], driver.C0_FIRST_RUN_ROOT):
            with self.assertRaisesRegex(ValueError, "must not be inside frozen tree"):
                driver._assert_not_frozen(frozen / "audit_run")
        with tempfile.TemporaryDirectory() as tmp:
            resolved = driver._assert_not_frozen(Path(tmp) / "audit_run")
            self.assertTrue(Path(resolved).is_dir())

    def test_verify_sha_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(json.dumps({"a": 1}), encoding="utf-8")
            actual = driver.sha256_of(path)
            driver.verify_sha(path, actual, label="ok")
            with self.assertRaises(ValueError):
                driver.verify_sha(path, "0" * 64, label="tampered")

    def test_driver_never_calls_the_c0_evaluate_or_training_entry_points(self):
        source = Path(driver.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "run_evaluation(", "run_training(", "run_audit(", "torch.save(",
            "AdaBelief", "backward()", "optimizer.step", "fit_reconstruction_calibration(",
            "load_conditional_logit(", "rca_metrics(", "construct_predicted_episodes(",
        ):
            self.assertNotIn(forbidden, source, "audit driver must not reference {}".format(forbidden))
        # the frozen single-window inference helper is reused on purpose
        self.assertIn("score_dataset", source)

    def test_driver_declares_the_read_only_contract(self):
        source = Path(driver.__file__).read_text(encoding="utf-8")
        for required in (
            "def stage_scores", "def stage_analyze", "def stage_capacity", "def stage_prepare",
            "def stage_finalize", "validation_replay.json", "event_failure_ledger.csv",
            "completion_manifest.json",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
