"""P6-C0 protocol tests: trigger labels, split, purge, selection, strata."""

import unittest

import numpy as np
import pandas as pd

from src.e2e.event_detection import select_train_threshold
from src.e2e.protocol import GAIA_SERVICES, TemporalBlock
from src.e2e.system_trigger import (
    DEFAULT_TOLERANCE_SECONDS,
    TRIGGER_IGNORE,
    TRIGGER_NEGATIVE,
    TRIGGER_POSITIVE,
    assign_legal_events,
    block_bounds,
    build_trigger_labels,
    duration_stratified_metrics,
    duration_stratum,
    evaluate_system_threshold,
    onset_density,
    onset_density_stratified_metrics,
    select_system_threshold,
    system_score_frame,
    trigger_binary_mask,
    trigger_label_counts,
    trigger_temporal_blocks,
    window_split_assignment,
)


ORIGIN = 1625133600000
GRID_MS = 30_000


def grid_timestamps(count, origin=ORIGIN):
    return origin + np.arange(count, dtype=np.int64) * GRID_MS


def registry_rows(records):
    return pd.DataFrame(records, columns=[
        "case_id", "source_index", "service", "fault_type", "start_ms", "end_ms", "detector_domain",
    ])


def base_config(total_bins=400):
    grid_ms = GRID_MS
    end = ORIGIN + total_bins * grid_ms
    boundary = ORIGIN + (7 * total_bins // 10) * grid_ms
    return {
        "schema_version": "p5_v3_test_fixture",
        "services": list(GAIA_SERVICES),
        "ad": {"grid_seconds": 30, "window_bins": 10, "window_step": 1},
        "split": {
            "mode": "chronological_metric_70_30",
            "absolute_start_ms": ORIGIN,
            "absolute_end_ms": end,
            "boundary_ms": boundary,
        },
        "event_trigger": {
            "matching_tolerance_seconds": 60,
            "matching_semantics": "causal_max_cardinality_minimum_delay",
            "threshold_selection": "train_only_exact_unique_scores",
        },
        "rca": {"window_seconds": 300},
    }


class TriggerLabelTests(unittest.TestCase):
    def test_short_11s_event_marks_only_its_recent_onset_window(self):
        timestamps = grid_timestamps(10)
        events = pd.DataFrame([{
            "case_id": "short", "start_ms": ORIGIN + 5_000, "end_ms": ORIGIN + 16_000,
        }])
        labels = build_trigger_labels(timestamps, events)
        self.assertEqual(
            labels.tolist(),
            [TRIGGER_NEGATIVE, TRIGGER_POSITIVE, TRIGGER_POSITIVE] + [TRIGGER_NEGATIVE] * 7,
        )
        # an 11 s event is inside the 60 s recent-onset window from its onset only
        self.assertEqual(int((labels == TRIGGER_IGNORE).sum()), 0)

    def test_30s_on_grid_onset_crossing_bins(self):
        timestamps = grid_timestamps(8)
        events = pd.DataFrame([{
            "case_id": "on-grid", "start_ms": ORIGIN + 30_000, "end_ms": ORIGIN + 41_000,
        }])
        labels = build_trigger_labels(timestamps, events)
        self.assertEqual(labels[1], TRIGGER_POSITIVE)
        self.assertEqual(labels[2], TRIGGER_POSITIVE)
        self.assertEqual(labels[3], TRIGGER_POSITIVE)  # exactly +60 s is still positive
        self.assertEqual(labels[4], TRIGGER_NEGATIVE)
        self.assertEqual(labels[0], TRIGGER_NEGATIVE)

    def test_60s_boundary_is_positive_and_beyond_it_is_ignore(self):
        timestamps = grid_timestamps(150)
        events = pd.DataFrame([{
            "case_id": "long", "start_ms": ORIGIN, "end_ms": ORIGIN + 3_600_000,
        }])
        labels = build_trigger_labels(timestamps, events)
        self.assertEqual(labels[0], TRIGGER_POSITIVE)
        self.assertEqual(labels[1], TRIGGER_POSITIVE)
        self.assertEqual(labels[2], TRIGGER_POSITIVE)  # t - start == 60 s exactly
        self.assertEqual(labels[3], TRIGGER_IGNORE)    # t - start == 90 s, still ongoing
        self.assertEqual(labels[119], TRIGGER_IGNORE)
        # t == end is outside the half-open event interval
        self.assertEqual(labels[120], TRIGGER_NEGATIVE)

    def test_3600s_event_produces_bounded_positive_supervision(self):
        timestamps = grid_timestamps(150)
        short = pd.DataFrame([{"case_id": "s", "start_ms": ORIGIN, "end_ms": ORIGIN + 11_000}])
        long = pd.DataFrame([{"case_id": "l", "start_ms": ORIGIN, "end_ms": ORIGIN + 3_600_000}])
        short_positive = int((build_trigger_labels(timestamps, short) == TRIGGER_POSITIVE).sum())
        long_positive = int((build_trigger_labels(timestamps, long) == TRIGGER_POSITIVE).sum())
        self.assertEqual(short_positive, 3)
        self.assertEqual(long_positive, 3)
        # the positive contribution does not grow with duration
        self.assertEqual(short_positive, long_positive)

    def test_overlapping_events_collapse_to_a_single_system_positive(self):
        timestamps = grid_timestamps(12)
        single = pd.DataFrame([{"case_id": "a", "start_ms": ORIGIN, "end_ms": ORIGIN + 11_000}])
        overlapping = pd.DataFrame([
            {"case_id": "a", "start_ms": ORIGIN, "end_ms": ORIGIN + 11_000},
            {"case_id": "b", "start_ms": ORIGIN + 30_000, "end_ms": ORIGIN + 41_000},
        ])
        single_labels = build_trigger_labels(timestamps, single)
        overlapping_labels = build_trigger_labels(timestamps, overlapping)
        # system level: overlapping onsets add no second channel and no extra class
        self.assertEqual(overlapping_labels.ndim, 1)
        self.assertEqual(len(overlapping_labels), len(timestamps))
        self.assertTrue(
            (overlapping_labels[single_labels == TRIGGER_POSITIVE] == TRIGGER_POSITIVE).all()
        )
        self.assertGreater(
            int((overlapping_labels == TRIGGER_POSITIVE).sum()),
            int((single_labels == TRIGGER_POSITIVE).sum()),
        )

    def test_multiple_root_services_in_one_bin_are_one_system_positive(self):
        timestamps = grid_timestamps(12)
        events = pd.DataFrame([
            {"case_id": "a", "service": "mobservice1", "start_ms": ORIGIN, "end_ms": ORIGIN + 11_000},
            {"case_id": "b", "service": "dbservice2", "start_ms": ORIGIN, "end_ms": ORIGIN + 11_000},
        ])
        labels = build_trigger_labels(timestamps, events)
        self.assertEqual(labels.tolist(), build_trigger_labels(timestamps, events.iloc[:1]).tolist())

    def test_normal_region_is_negative(self):
        timestamps = grid_timestamps(6)
        events = pd.DataFrame([{"case_id": "a", "start_ms": ORIGIN + 90_000, "end_ms": ORIGIN + 101_000}])
        labels = build_trigger_labels(timestamps, events)
        self.assertEqual(labels[0], TRIGGER_NEGATIVE)
        self.assertEqual(labels[1], TRIGGER_NEGATIVE)

    def test_labels_are_strictly_causal(self):
        timestamps = grid_timestamps(8)
        events = pd.DataFrame([{"case_id": "future", "start_ms": ORIGIN + 300_000, "end_ms": ORIGIN + 311_000}])
        labels = build_trigger_labels(timestamps, events)
        self.assertEqual(int((labels == TRIGGER_POSITIVE).sum()), 0)
        self.assertEqual(int((labels == TRIGGER_IGNORE).sum()), 0)

    def test_ignore_bins_are_masked_out_of_the_loss(self):
        labels = np.array([TRIGGER_NEGATIVE, TRIGGER_POSITIVE, TRIGGER_IGNORE], dtype=np.int8)
        target, mask = trigger_binary_mask(labels)
        self.assertEqual(target.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(mask.tolist(), [1.0, 1.0, 0.0])

    def test_label_counts_report_all_three_states(self):
        counts = trigger_label_counts(np.array([0, 1, 1, 2, 2, 2]))
        self.assertEqual(counts["counts"], {"negative": 1, "positive": 2, "ignore": 3})
        self.assertEqual(counts["positive_region_count"], 1)

    def test_rejects_non_positive_intervals(self):
        with self.assertRaisesRegex(ValueError, "positive half-open spans"):
            build_trigger_labels(grid_timestamps(4), pd.DataFrame([
                {"case_id": "bad", "start_ms": 100, "end_ms": 100},
            ]))


class SplitTests(unittest.TestCase):
    def test_blocks_are_chronological_50_20_30_and_keep_the_frozen_test(self):
        config = base_config()
        blocks = trigger_temporal_blocks(config)
        self.assertEqual([block.name for block in blocks], ["fit", "validation", "test"])
        self.assertEqual(blocks[0].start_ms, config["split"]["absolute_start_ms"])
        self.assertEqual(blocks[1].start_ms, blocks[0].end_ms)
        self.assertEqual(blocks[2].start_ms, blocks[1].end_ms)
        self.assertEqual(blocks[2].end_ms, config["split"]["absolute_end_ms"])
        self.assertEqual(blocks[2].start_ms, config["split"]["boundary_ms"])
        self.assertEqual(blocks[0].end_ms - blocks[0].start_ms, 200 * GRID_MS)
        self.assertEqual(blocks[1].end_ms - blocks[1].start_ms, 80 * GRID_MS)

    def test_purge_history_is_exactly_window_bins_at_each_internal_boundary(self):
        blocks = trigger_temporal_blocks(base_config())
        # a 280-bin train array covering fit + validation, like the frozen Train array
        timestamps = grid_timestamps(280)
        assignment = window_split_assignment(timestamps, blocks, grid_seconds=30, window_bins=10)
        purged = assignment.loc[~assignment["keep"]]
        # ten windows whose 300 s history crosses the Fit/Validation boundary, plus
        # the last window of the array whose target bin belongs to the Test block
        self.assertEqual(int(len(purged)), 11)
        self.assertEqual(
            sorted(purged["purge_reason"].unique().tolist()),
            ["input_history_crosses_split_boundary"],
        )
        fit_end = int(blocks[0].end_ms)
        first_validation = assignment.loc[
            (assignment["split"] == "validation") & assignment["keep"]
        ].iloc[0]
        self.assertEqual(int(first_validation["window_start_time"]), fit_end)
        self.assertEqual(int(first_validation["prediction_available_time"]), fit_end + 10 * GRID_MS)

    def test_split_is_chronological_and_non_overlapping(self):
        blocks = trigger_temporal_blocks(base_config())
        assignment = window_split_assignment(grid_timestamps(280), blocks, grid_seconds=30, window_bins=10)
        kept = assignment.loc[assignment["keep"]]
        fit = kept.loc[kept["split"] == "fit", "prediction_available_time"]
        validation = kept.loc[kept["split"] == "validation", "prediction_available_time"]
        self.assertEqual(int(fit.max()), int(blocks[0].end_ms) - GRID_MS)
        self.assertEqual(int(validation.min()), int(blocks[1].start_ms) + 10 * GRID_MS)
        self.assertLess(int(fit.max()), int(validation.min()))

    def test_final_prediction_timestamp_is_inside_the_final_block(self):
        blocks = trigger_temporal_blocks(base_config())
        timestamps = grid_timestamps(400)
        assignment = window_split_assignment(timestamps, blocks, grid_seconds=30, window_bins=10)
        last = assignment.iloc[-1]
        self.assertTrue(bool(last["keep"]))
        self.assertEqual(str(last["split"]), "test")
        self.assertEqual(int(last["prediction_available_time"]), int(blocks[-1].end_ms))

    def test_crossing_event_is_purged_from_every_metric_population(self):
        config = base_config()
        blocks = trigger_temporal_blocks(config)
        boundary = int(blocks[1].start_ms)
        registry = registry_rows([
            {"case_id": "fit-only", "source_index": 0, "service": "mobservice1",
             "fault_type": "login_failure", "start_ms": ORIGIN + 60_000, "end_ms": ORIGIN + 71_000,
             "detector_domain": True},
            {"case_id": "crossing", "source_index": 1, "service": "mobservice2",
             "fault_type": "login_failure", "start_ms": boundary - 5_000, "end_ms": boundary + 6_000,
             "detector_domain": True},
            {"case_id": "validation-only", "source_index": 2, "service": "dbservice1",
             "fault_type": "memory_anomalies", "start_ms": boundary + 60_000,
             "end_ms": boundary + 71_000, "detector_domain": True},
            {"case_id": "outside-domain", "source_index": 3, "service": "dbservice2",
             "fault_type": "cpu_anomalies", "start_ms": ORIGIN - 600_000, "end_ms": ORIGIN - 590_000,
             "detector_domain": False},
        ])
        legal, assigned, purged = assign_legal_events(registry, blocks)
        self.assertEqual(sorted(assigned["case_id"]), ["fit-only", "validation-only"])
        self.assertEqual(list(purged["case_id"]), ["crossing"])
        # the crossing event still defines the ground-truth fault state for labels
        self.assertIn("crossing", set(legal["case_id"]))
        self.assertNotIn("outside-domain", set(legal["case_id"]))

    def test_event_ending_exactly_at_the_boundary_stays_in_the_earlier_block(self):
        blocks = trigger_temporal_blocks(base_config())
        boundary = int(blocks[1].start_ms)
        registry = registry_rows([
            {"case_id": "ends-at-boundary", "source_index": 0, "service": "mobservice1",
             "fault_type": "login_failure", "start_ms": boundary - 11_000, "end_ms": boundary,
             "detector_domain": True},
        ])
        _, assigned, purged = assign_legal_events(registry, blocks)
        self.assertEqual(list(assigned["case_id"]), ["ends-at-boundary"])
        self.assertEqual(list(assigned["split"]), ["fit"])
        self.assertTrue(purged.empty)

    def test_block_bounds_are_half_open_and_ordered(self):
        bounds = block_bounds(trigger_temporal_blocks(base_config()))
        self.assertEqual(list(bounds), ["fit", "validation", "test"])
        self.assertEqual(bounds["fit"]["end_ms"], bounds["validation"]["start_ms"])
        self.assertEqual(bounds["validation"]["end_ms"], bounds["test"]["start_ms"])


class ThresholdSelectionTests(unittest.TestCase):
    def _series(self):
        timestamps = grid_timestamps(12)
        scores = np.array(
            [0.05, 0.9, 0.9, 0.05, 0.05, 0.2, 0.85, 0.85, 0.05, 0.05, 0.05, 0.05]
        )
        gt = pd.DataFrame([{
            "case_id": "a", "source_index": 0, "service": "dbservice1",
            "fault_type": "login_failure", "start_ms": int(timestamps[1]),
            "end_ms": int(timestamps[1]) + 11_000,
        }])
        return timestamps, scores, gt

    def test_matches_the_frozen_train_only_threshold_rule_exactly(self):
        timestamps, scores, gt = self._series()
        mine = select_system_threshold(
            system_score_frame("validation", timestamps, scores), gt, workers=1
        )
        rows = [
            {"split": "train", "prediction_available_time": int(t),
             "prediction_timestamp": int(t), "service": service, "anomaly_score": float(score)}
            for t, score in zip(timestamps, scores) for service in GAIA_SERVICES
        ]
        frozen = select_train_threshold(pd.DataFrame(rows), gt)
        self.assertAlmostEqual(mine.threshold, frozen.threshold, places=12)
        self.assertAlmostEqual(mine.metrics["event_f1"], frozen.train_metrics["event_f1"], places=12)
        self.assertAlmostEqual(mine.metrics["event_precision"], frozen.train_metrics["event_precision"], places=12)
        self.assertAlmostEqual(mine.metrics["event_recall"], frozen.train_metrics["event_recall"], places=12)
        self.assertEqual(mine.metrics["true_positive_events"], frozen.train_metrics["true_positive_events"])
        self.assertEqual(mine.metrics["false_positive_events"], frozen.train_metrics["false_positive_events"])
        self.assertEqual(mine.metrics["false_negative_events"], frozen.train_metrics["false_negative_events"])

    def test_equal_f1_takes_the_highest_threshold(self):
        timestamps = grid_timestamps(3)
        scores = np.array([0.9, 0.9, 0.1])
        gt = pd.DataFrame([{
            "case_id": "a", "source_index": 0, "service": "dbservice1",
            "fault_type": "login_failure", "start_ms": int(timestamps[0]),
            "end_ms": int(timestamps[0]) + 11_000,
        }])
        selection = select_system_threshold(
            system_score_frame("validation", timestamps, scores), gt, workers=1
        )
        # 0.9 and 0.1 give the identical episode; the frozen rule takes the higher one
        self.assertEqual(selection.threshold, 0.9)
        self.assertEqual(selection.metrics["event_f1"], 1.0)

    def test_threshold_selection_never_reads_a_split_column(self):
        timestamps, scores, gt = self._series()
        frame = system_score_frame("validation", timestamps, scores)
        frame["split"] = "test"  # a wrong label must not change the decision
        first = select_system_threshold(frame, gt, workers=1).threshold
        frame["split"] = "validation"
        self.assertAlmostEqual(first, select_system_threshold(frame, gt, workers=1).threshold, places=12)

    def test_rejects_non_finite_scores(self):
        timestamps = grid_timestamps(3)
        frame = system_score_frame("validation", timestamps, np.array([0.1, np.nan, 0.2]))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            select_system_threshold(frame, pd.DataFrame([{"start_ms": 0}]), workers=1)


class EpisodeAndMatchingTests(unittest.TestCase):
    def test_episode_anchor_is_the_first_positive_prediction_time(self):
        timestamps = grid_timestamps(6)
        scores = np.array([0.1, 0.9, 0.8, 0.1, 0.95, 0.95])
        frame = system_score_frame("test", timestamps, scores)
        episodes, matching, metrics = evaluate_system_threshold(
            frame, pd.DataFrame(columns=["case_id", "source_index", "service", "fault_type", "start_ms", "end_ms"]),
            0.5,
        )
        self.assertEqual(episodes["t_hat"].tolist(), [int(timestamps[1]), int(timestamps[4])])
        self.assertEqual(episodes["positive_bins"].tolist(), [2, 2])
        self.assertEqual(int(len(matching)), 2)
        self.assertEqual(metrics["event_precision"], 0.0)

    def test_frozen_causal_60s_tolerance_is_reused(self):
        timestamps = grid_timestamps(8)
        scores = np.zeros(len(timestamps))
        scores[0] = 0.9   # episode 30 s before the onset: not causal
        scores[1] = 0.9   # same consecutive episode as index 0
        scores[3] = 0.9   # exactly +60 s after the onset: accepted
        scores[5] = 0.9   # exactly +120 s: outside tolerance
        gt = pd.DataFrame([{
            "case_id": "a", "source_index": 0, "service": "dbservice1",
            "fault_type": "login_failure", "start_ms": int(timestamps[1]),
            "end_ms": int(timestamps[1]) + 11_000,
        }])
        frame = system_score_frame("test", timestamps, scores)
        episodes, matching, metrics = evaluate_system_threshold(frame, gt, 0.5)
        self.assertEqual(episodes["t_hat"].tolist(),
                         [int(timestamps[0]), int(timestamps[3]), int(timestamps[5])])
        self.assertEqual(metrics["true_positive_events"], 1)
        self.assertEqual(metrics["false_positive_events"], 2)
        self.assertEqual(metrics["false_negative_events"], 0)
        matched = matching.loc[matching["match_status"] == "matched"]
        self.assertEqual(float(matched["detection_delay_seconds"].iloc[0]), 60.0)
        self.assertEqual(metrics["detection_delay"]["causal_rule"],
                         "0 <= t_hat - gt_start_ms <= tolerance_seconds")
        self.assertEqual(DEFAULT_TOLERANCE_SECONDS, 60)


class StratificationTests(unittest.TestCase):
    def test_duration_strata_boundaries(self):
        values = [11.0, 15.0, 15.1, 30.0, 30.1, 60.0, 60.1, 300.0, 300.1, 3600.0]
        self.assertEqual(
            duration_stratum(values).tolist(),
            ["le_15s", "le_15s", "15_30s", "15_30s", "30_60s",
             "30_60s", "60_300s", "60_300s", "gt_300s", "gt_300s"],
        )

    def test_duration_stratified_recall_uses_gt_events(self):
        matching = pd.DataFrame([
            {"match_status": "matched", "case_id": "a", "gt_start_ms": 0,
             "detection_delay_seconds": 10.0},
            {"match_status": "miss", "case_id": "b", "gt_start_ms": 0,
             "detection_delay_seconds": None},
            {"match_status": "matched", "case_id": "c", "gt_start_ms": 0,
             "detection_delay_seconds": 20.0},
            {"match_status": "false_alarm", "case_id": None, "gt_start_ms": None,
             "detection_delay_seconds": None},
        ])
        case_frame = pd.DataFrame([
            {"case_id": "a", "duration_seconds": 11.0},
            {"case_id": "b", "duration_seconds": 11.0},
            {"case_id": "c", "duration_seconds": 200.0},
        ])
        rows = {row["duration_stratum"]: row for row in duration_stratified_metrics(matching, case_frame)}
        self.assertEqual(rows["le_15s"]["ground_truth_events"], 2)
        self.assertEqual(rows["le_15s"]["recall"], 0.5)
        self.assertEqual(rows["60_300s"]["recall"], 1.0)
        self.assertTrue(rows["le_15s"]["small_n"])
        self.assertIsNone(rows["gt_300s"]["recall"])

    def test_onset_density_counts_multi_onset_bins(self):
        events = pd.DataFrame([
            {"case_id": "a", "service": "mobservice1", "start_ms": ORIGIN + 1_000, "end_ms": ORIGIN + 12_000},
            {"case_id": "b", "service": "dbservice1", "start_ms": ORIGIN + 2_000, "end_ms": ORIGIN + 13_000},
            {"case_id": "c", "service": "mobservice1", "start_ms": ORIGIN + 90_000, "end_ms": ORIGIN + 101_000},
        ])
        density = onset_density(events, origin_ms=ORIGIN, grid_seconds=30)
        self.assertEqual(density["onset_count"].tolist(), [2, 1])
        self.assertEqual(density["root_service_count"].tolist(), [2, 1])

    def test_single_and_multi_onset_recall_are_reported_separately(self):
        events = pd.DataFrame([
            {"case_id": "a", "service": "mobservice1", "start_ms": ORIGIN + 1_000, "end_ms": ORIGIN + 12_000},
            {"case_id": "b", "service": "dbservice1", "start_ms": ORIGIN + 2_000, "end_ms": ORIGIN + 13_000},
            {"case_id": "c", "service": "mobservice1", "start_ms": ORIGIN + 90_000, "end_ms": ORIGIN + 101_000},
        ])
        matching = pd.DataFrame([
            {"match_status": "matched", "case_id": "a", "gt_start_ms": ORIGIN + 1_000,
             "gt_service": "mobservice1", "fault_type": "login_failure",
             "detection_delay_seconds": 5.0},
            {"match_status": "miss", "case_id": "b", "gt_start_ms": ORIGIN + 2_000,
             "gt_service": "dbservice1", "fault_type": "login_failure",
             "detection_delay_seconds": None},
            {"match_status": "miss", "case_id": "c", "gt_start_ms": ORIGIN + 90_000,
             "gt_service": "mobservice1", "fault_type": "login_failure",
             "detection_delay_seconds": None},
        ])
        report = onset_density_stratified_metrics(
            matching, legal_events=events, origin_ms=ORIGIN, grid_seconds=30
        )
        self.assertEqual(report["single_onset"]["recall"], 0.0)
        self.assertEqual(report["multi_onset"]["ground_truth_events"], 2)
        self.assertEqual(report["multi_onset"]["recall"], 0.5)
        self.assertEqual(report["bins_with_multiple_root_services"], 1)
        self.assertEqual(report["same_bin_multi_root_events"], 2)


class AssignmentAuditTests(unittest.TestCase):
    def test_window_assignment_marks_outside_windows(self):
        blocks = (
            TemporalBlock("fit", ORIGIN, ORIGIN + 100 * GRID_MS),
            TemporalBlock("validation", ORIGIN + 100 * GRID_MS, ORIGIN + 140 * GRID_MS),
            TemporalBlock("test", ORIGIN + 140 * GRID_MS, ORIGIN + 200 * GRID_MS, is_final=True),
        )
        timestamps = grid_timestamps(200)
        assignment = window_split_assignment(timestamps, blocks, grid_seconds=30, window_bins=10)
        self.assertTrue(set(assignment["split"]).issubset({"fit", "validation", "test", "outside"}))
        # ten purged windows at each of the two internal boundaries
        self.assertEqual(int((~assignment["keep"]).sum()), 20)
        outside = assignment.loc[assignment["split"] == "outside"]
        self.assertTrue(outside.empty)


if __name__ == "__main__":
    unittest.main()
