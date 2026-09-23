"""Regression cases from the C0F artifact review; synthetic data only."""

import json
from pathlib import Path
import tempfile
import unittest
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

import scripts.p6.audit_c0_failure as driver
from src.e2e.system_trigger import evaluate_system_threshold, system_score_frame
from src.e2e.system_trigger_failure_audit import (
    build_failure_ledger, build_trajectories, window_observation, isolation_summary, confounding_tables,
    _active_vector,
)
from src.e2e.system_trigger_audit_provenance import (
    assert_stage_results, capacity_checks, file_record, verify_records, source_digest, source_records,
    bind_tests,
)


def fixture(onsets=(5_000, 35_000), durations=(180_000, 10_000), count=12):
    origin = 1625133600000
    slots = origin + np.arange(count, dtype=np.int64) * 30_000
    events = pd.DataFrame([
        dict(case_id="e{}".format(i), source_index=i, start_ms=origin + onset,
             end_ms=origin + onset + duration, service="mobservice1", fault_type="login_failure")
        for i, (onset, duration) in enumerate(zip(onsets, durations))
    ])
    scores = np.zeros(count)
    episodes, matching, _ = evaluate_system_threshold(system_score_frame("test", slots, scores), events, 0.5)
    ledger, _ = build_failure_ledger(
        events, split="test", slot_times_ms=slots, slot_scores=scores, threshold=0.5,
        episode_anchors_ms=episodes.t_hat.to_numpy(),
        episode_end_times_ms=episodes.episode_end_time.to_numpy(), matching=matching, origin_ms=origin,
    )
    trajectories, summary = build_trajectories(
        ledger, split="test", slot_times_ms=slots, slot_scores=scores, slot_logits=scores, threshold=0.5,
    )
    return events, ledger, trajectories, summary


class CorrectionRegressionTests(unittest.TestCase):
    def test_source_snapshot_requires_the_correction_protocol(self):
        with patch("src.e2e.system_trigger_audit_provenance.subprocess.check_output",
                   return_value=b"scripts/p6/audit_c0_failure.py\0"):
            with self.assertRaisesRegex(ValueError, "correction protocol"):
                source_records(driver.PROJECT_ROOT)

    def test_reused_run_requires_the_pinned_completion_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "historical"
            (parent / "scores").mkdir(parents=True)
            records = {}
            for split in ("fit", "validation"):
                path = parent / "scores" / (split + ".csv")
                path.write_text("score\n0.5\n")
                records["scores/" + split + ".csv"] = file_record(path)
            completion = parent / "completion_manifest.json"
            completion.write_text(json.dumps({"status": "COMPLETE", "required_outputs": records}))
            with patch.object(driver, "HISTORICAL_C0F_RUN", parent), patch.object(
                    driver, "EXPECTED_HISTORICAL_C0F_COMPLETION_SHA256", file_record(completion)["sha256"]):
                self.assertEqual(driver.verify_reuse_run(parent)["status"], "COMPLETE")
                with self.assertRaisesRegex(ValueError, "pinned historical"):
                    driver.verify_reuse_run(Path(temp))
                completion.write_text(json.dumps({"status": "COMPLETE", "required_outputs": records, "changed": True}))
                with self.assertRaisesRegex(ValueError, "completion manifest"):
                    driver.verify_reuse_run(parent)

    def test_actual_input_roots_must_match_frozen_config_paths(self):
        trigger = {"data_root": "data/p5/v3_preprocessing_v2/ad",
                   "artifact_root": "artifacts/p5/v3_preprocessing_v2/ad"}
        base = driver.PROJECT_ROOT / driver.DEFAULT_BASE_CONFIG
        config = driver.PROJECT_ROOT / driver.DEFAULT_TRIGGER_CONFIG
        data = driver.PROJECT_ROOT / trigger["data_root"]
        artifacts = driver.PROJECT_ROOT / trigger["artifact_root"]
        driver.assert_frozen_bindings(base, config, data, artifacts, trigger)
        with self.assertRaisesRegex(ValueError, "frozen data root"):
            driver.assert_frozen_bindings(base, config, data.parent, artifacts, trigger)
        with self.assertRaisesRegex(ValueError, "frozen artifact root"):
            driver.assert_frozen_bindings(base, config, data, artifacts.parent, trigger)

    def test_overlapping_unequal_durations_have_exact_active_counts(self):
        events, ledger, trajectories, _ = fixture()
        for row in ledger.itertuples(index=False):
            expected_active = events.loc[
                (events.case_id != row.case_id) & (events.start_ms <= row.onset_ms)
                & (row.onset_ms < events.end_ms)
            ]
            expected_recent = events.loc[
                (events.case_id != row.case_id) & (events.start_ms >= row.onset_ms - 60_000)
                & (events.start_ms <= row.onset_ms)
            ]
            self.assertEqual(row.other_active_event_count, len(expected_active))
            self.assertEqual(row.other_onset_count_60s, len(expected_recent))
        for row in trajectories.itertuples(index=False):
            expected = events.loc[
                (events.case_id != row.case_id) & (events.start_ms <= row.prediction_available_time)
                & (row.prediction_available_time < events.end_ms)
            ]
            self.assertEqual(row.other_active_event_count, len(expected))

    def test_short_event_at_split_end_is_censored_for_300_second_followup(self):
        _, ledger, _, summary = fixture(onsets=(75_000,), durations=(10_000,), count=4)
        self.assertEqual(ledger.iloc[0].response_band, "censored")
        self.assertTrue(summary.iloc[0].response_censored)
        self.assertTrue(summary.iloc[0].early_censored)
        self.assertTrue(summary.iloc[0].mid_censored)

    def test_report_grid_bound_does_not_read_label_reference(self):
        _, ledger, _, response = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger.to_csv(root / "event_failure_ledger.csv", index=False)
            response.to_csv(root / "event_response_summary.csv", index=False)
            (root / "capacity_summary.json").write_text(json.dumps({"splits": {"test": {
                "episode_bound": {"recall_upper_bound": 0.8},
                "grid_bound": {"recall_upper_bound": 1.0},
                "label_reference": {"event_metrics": {"event_recall": 0.4}},
            }}}))
            self.assertEqual(driver._mechanism_findings(root)["test"]["grid_recall_upper_bound"], 1.0)

    def test_finalized_directory_refuses_further_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_state.json").write_text(json.dumps({"status": "COMPLETE"}))
            with self.assertRaisesRegex(ValueError, "complete|COMPLETE|sealed"):
                driver._assert_not_frozen(root)

    def test_stopped_directory_refuses_further_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_state.json").write_text(json.dumps({"status": "STOP"}))
            with self.assertRaisesRegex(ValueError, "STOP|stopped|sealed"):
                driver._assert_not_frozen(root)

    def test_external_validation_record_cannot_replace_own_code_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / "pytest.log"
            xml = root / "pytest.xml"
            log.write_text("fabricated passing output")
            xml.write_text('<testsuite tests="1" failures="0" errors="0"/>')
            record = root / "test_results.json"
            record.write_text(json.dumps({
                "exit_code": 0, "source_digest": source_digest(source_records(driver.PROJECT_ROOT)),
                "files": {"pytest.log": file_record(log), "pytest.xml": file_record(xml)},
            }))
            (root / "audit").mkdir()
            with patch("src.e2e.system_trigger_audit_provenance.run_code_checks",
                       side_effect=ValueError("own C0F code checks failed")):
                with self.assertRaisesRegex(ValueError, "own C0F code checks failed"):
                    bind_tests(driver.PROJECT_ROOT, root / "audit", record)

    def test_active_vector_accepts_unsorted_endpoints(self):
        np.testing.assert_array_equal(
            _active_vector(np.array([10, 20]), np.array([90, 30]), np.array([0, 25, 40, 95]), 1),
            [0, 1, 1, 0])

    def test_window_coverage_uses_lattice_and_open_endpoints(self):
        slots = np.array([0, 30000, 60000], dtype=np.int64)
        selection, coverage = window_observation(slots, 1000, 10000, right_closed=False)
        self.assertEqual(coverage["status"], "no_grid_observation")
        self.assertEqual(selection.stop - selection.start, 0)
        _, coverage = window_observation(slots, 60000, 90000, right_closed=False)
        self.assertEqual(coverage["status"], "complete")
        _, coverage = window_observation(slots, 60000, 90000)
        self.assertTrue(coverage["right_censored"])
        _, coverage = window_observation(np.array([0, 60000]), 0, 60000)
        self.assertTrue(coverage["interior_missing"])

    def test_long_event_end_is_excluded_from_first_response(self):
        origin = 1625133600000
        slots = origin + np.arange(15) * 30000
        events = pd.DataFrame([dict(case_id="long", source_index=0, start_ms=origin,
                                   end_ms=origin + 330000, service="mobservice1", fault_type="memory_anomalies")])
        scores = np.zeros(len(slots)); scores[11] = 1.0
        episodes, matching, _ = evaluate_system_threshold(system_score_frame("test", slots, scores), events, 0.5)
        ledger, _ = build_failure_ledger(events, split="test", slot_times_ms=slots, slot_scores=scores, threshold=0.5,
                                       episode_anchors_ms=episodes.t_hat.to_numpy(),
                                       episode_end_times_ms=episodes.episode_end_time.to_numpy(), matching=matching, origin_ms=origin)
        self.assertEqual(ledger.iloc[0].response_band, "never")
        self.assertIsNone(ledger.iloc[0].first_positive_time)

    def test_isolation_keeps_official_population_and_excludes_censored_cases(self):
        _, ledger, trajectories, response = fixture()
        cases, summary = isolation_summary(ledger, trajectories, response)
        self.assertEqual(len(cases), len(ledger) * 5)
        row = next(r for r in summary["groups"] if r["duration_stratum"] == "all" and r["window"] == "response")
        self.assertEqual(row["events"], 2)
        self.assertLess(row["clean_observations"], row["observations"])
        self.assertEqual(row["fully_clean_case_windows"], 0)
        _, ledger, trajectories, response = fixture(onsets=(75000,), durations=(10000,), count=4)
        cases, _ = isolation_summary(ledger, trajectories, response)
        self.assertFalse(cases.loc[cases.window == "response", "fully_clean_case_window"].iloc[0])

    def test_quantiles_and_empty_cross_cells_are_explicit(self):
        events, _, _, response = fixture()
        self.assertEqual(response.iloc[0].response_score_q50, 0.0)
        tables = confounding_tables(events, origin_ms=0)
        self.assertEqual(len(tables["fault_x_duration_x_service"]), 300)
        self.assertEqual(sum(row["n"] for row in tables["fault_x_duration_x_service"]), 2)
        self.assertTrue(any(row["n"] == 0 for row in tables["fault_x_duration_x_service"]))

    def test_failed_stage_records_stop_instead_of_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            driver.save_run_state(root, {"stages": {}, "status": "PARTIAL"})
            with self.assertRaisesRegex(ValueError, "validation failed"):
                driver.finish_checked_stage(root, "analyze", {"ledger_invariants": False})
            state = driver.load_run_state(root)
            self.assertEqual(state["status"], "STOP")
            self.assertEqual(state["stages"]["analyze"]["status"], "FAILED")
            self.assertFalse((root / "completion_manifest.json").exists())

    def test_good_ordering_cannot_hide_invalid_witness(self):
        results = {split: {
            "ordering": {"observed_tp": 1},
            "grid_bound": {"matched_events": 3, "cross_check_agrees": None},
            "episode_bound": {"upper_bound_tp": 2, "ground_truth_events": 4, "witness_replay_agrees": False,
                              "witness_anchors_match": True, "witness_realised_tp": 1, "status": "EXACT", "proof": "DP"},
        } for split in ("fit", "validation", "test")}
        with self.assertRaisesRegex(ValueError, "validation failed"):
            assert_stage_results("capacity", capacity_checks(results))

    def test_changed_or_missing_sealed_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input"
            path.write_text("original")
            records = {"input": file_record(path)}
            path.write_text("modified")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_records(records)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_records(records)

    def test_source_digest_detects_byte_changes_and_new_files(self):
        before = {"a.py": {"path": "/a.py", "sha256": "aa", "bytes": 2}}
        after = {"a.py": {"path": "/a.py", "sha256": "ab", "bytes": 2}}
        self.assertNotEqual(source_digest(before), source_digest(after))
        self.assertNotEqual(source_digest(before), source_digest(dict(before, **{"b.py": before["a.py"]})))

    def test_budget_timeout_cannot_publish_exact(self):
        with self.assertRaises(TimeoutError):
            with driver.capacity_budget(0.01):
                time.sleep(0.04)

    def test_descriptive_summary_does_not_select_route_from_test_mix(self):
        _, ledger, _, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger.to_csv(root / "event_failure_ledger.csv", index=False)
            first = driver._loss_dominance(root)
            ledger["failure_category"] = "MATCHING_COMPETITION"
            ledger.to_csv(root / "event_failure_ledger.csv", index=False)
            second = driver._loss_dominance(root)
            self.assertEqual(first["recommendation"], second["recommendation"])
            self.assertEqual(second["verdict"], "DESCRIPTIVE_ONLY")

    def test_sixty_second_evidence_counts_late_positives_as_misses(self):
        _, ledger, _, _ = fixture()
        ledger["first_positive_delta_ms"] = [90000, np.nan]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger.to_csv(root / "event_failure_ledger.csv", index=False)
            claims = driver._evidence_ledger(root, {"integrity": {}, "population": {}})
            claim = next(c for c in claims if c["claim_id"] == "C0F-NO-FIRST-POSITIVE-TEST")
            self.assertTrue(claim["statement"].startswith("2 events"))

    def test_analysis_gate_detects_corrupted_active_marker(self):
        events, ledger, trajectories, _ = fixture()
        self.assertTrue(all(driver.validate_analysis(ledger, trajectories, events).values()))
        trajectories.loc[0, "other_active_event_count"] = -1
        self.assertFalse(driver.validate_analysis(ledger, trajectories, events)["active_counts_independently_match"])


if __name__ == "__main__":
    unittest.main()
