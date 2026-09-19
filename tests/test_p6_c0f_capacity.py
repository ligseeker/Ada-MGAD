"""P6-C0F capacity tests: R_label counterexample, U_grid and exact U_episode.

The episode bound is cross-validated against exhaustive enumeration of every
binary sequence on small instances, using the frozen episode construction and the
frozen causal matching.
"""

import itertools
import unittest

import numpy as np
import pandas as pd

from src.e2e.event_detection import construct_predicted_episodes, event_metrics, match_events
from src.e2e.system_trigger import (
    TRIGGER_IGNORE,
    TRIGGER_NEGATIVE,
    TRIGGER_POSITIVE,
    system_score_frame,
)
from src.e2e.system_trigger_capacity import (
    adjacency_ranges,
    assert_contiguous_slots,
    edf_matching,
    episode_bound,
    grid_bound,
    label_reference,
    replay_witness,
)


ORIGIN = 1625133600000
GRID_MS = 30_000


def slots(count, origin=ORIGIN):
    return origin + np.arange(count, dtype=np.int64) * GRID_MS


def gt_frame(onsets, origin=ORIGIN):
    rows = []
    for index, (relative_ms, duration_ms) in enumerate(onsets):
        start = origin + relative_ms
        rows.append({
            "case_id": "e{:02d}".format(index), "source_index": index,
            "service": "mobservice1", "fault_type": "login_failure",
            "start_ms": int(start), "end_ms": int(start + duration_ms),
        })
    return pd.DataFrame(rows)


def brute_force_tp(slot_times, ground_truth, tolerance_seconds=60):
    """Maximum TP over every binary sequence, through the frozen pipeline."""

    best = 0
    count = len(slot_times)
    for bits in itertools.product((0.0, 1.0), repeat=count):
        frame = system_score_frame("test", slot_times, np.asarray(bits))
        episodes = construct_predicted_episodes(frame, 0.5)
        matching = match_events(episodes, ground_truth, tolerance_seconds=tolerance_seconds)
        best = max(best, int(event_metrics(matching)["true_positive_events"]))
    return best


class GeometryTests(unittest.TestCase):
    def test_adjacency_is_causal_and_inclusive(self):
        slot_times = slots(5)
        onset = ORIGIN + 11_000
        adjacency = adjacency_ranges(slot_times, np.array([onset]))
        left, right = int(adjacency[0][0]), int(adjacency[0][1])
        # relative offsets 19 s and 49 s are inside [0, 60 s]; -11 s and 79 s are not
        self.assertEqual(left, 1)
        self.assertEqual(right - left, 2)
        self.assertEqual([int(value) - onset for value in slot_times[left:right]], [19_000, 49_000])
        for index in range(left, right):
            self.assertGreaterEqual(int(slot_times[index]) - onset, 0)
            self.assertLessEqual(int(slot_times[index]) - onset, 60_000)

    def test_adjacency_excludes_non_causal_and_late_slots(self):
        slot_times = slots(6)
        onset = ORIGIN + 70_000
        adjacency = adjacency_ranges(slot_times, np.array([onset]))
        left, right = int(adjacency[0][0]), int(adjacency[0][1])
        self.assertGreaterEqual(int(slot_times[left]) - onset, 0)
        self.assertGreater(int(slot_times[right - 1]) - onset, 60_000 - 30_000)
        self.assertEqual(right - left, 2)  # 80 s and 110 s relative offsets

    def test_contiguous_guard_rejects_gaps(self):
        with self.assertRaisesRegex(ValueError, "contiguous legal slot grid"):
            assert_contiguous_slots(np.array([0, 30_000, 90_000], dtype=np.int64))


class GridBoundTests(unittest.TestCase):
    def test_grid_bound_matches_the_frozen_matching(self):
        slot_times = slots(6)
        ground_truth = gt_frame([(5_000, 11_000), (65_000, 11_000), (400_000, 11_000)])
        result = grid_bound(slot_times, ground_truth)
        self.assertTrue(result["cross_check_agrees"])
        self.assertEqual(result["matched_events"], 2)
        self.assertAlmostEqual(result["recall_upper_bound"], 2 / 3)

    def test_grid_bound_is_a_relaxation_of_the_episode_problem(self):
        slot_times = slots(2)
        ground_truth = gt_frame([(-50_000, 11_000), (20_000, 11_000)])
        grid = grid_bound(slot_times, ground_truth)
        episode = episode_bound(slot_times, ground_truth["start_ms"].to_numpy())
        self.assertEqual(grid["matched_events"], 2)
        self.assertEqual(episode.upper_bound_tp, 1)
        self.assertLessEqual(episode.upper_bound_tp, grid["matched_events"])

    def test_edf_matching_is_maximum_on_a_random_instance(self):
        rng = np.random.RandomState(0)
        for _ in range(20):
            slot_times = slots(7)
            ground_truth = gt_frame([(int(value), 11_000) for value in rng.randint(0, 200_000, size=4)])
            adjacency = adjacency_ranges(slot_times, ground_truth["start_ms"].to_numpy())
            count, _ = edf_matching(adjacency, len(slot_times))
            self.assertEqual(count, grid_bound(slot_times, ground_truth)["matched_events"])
            # the grid bound relaxes the episode constraint, so it can only be larger
            self.assertGreaterEqual(count, brute_force_tp(slot_times, ground_truth))


class EpisodeBoundTests(unittest.TestCase):
    def test_single_event(self):
        slot_times = slots(3)
        ground_truth = gt_frame([(10_000, 11_000)])
        self.assertEqual(episode_bound(slot_times, ground_truth["start_ms"].to_numpy()).upper_bound_tp, 1)

    def test_two_independent_events(self):
        slot_times = slots(10)
        ground_truth = gt_frame([(10_000, 11_000), (200_000, 11_000)])
        result = episode_bound(slot_times, ground_truth["start_ms"].to_numpy())
        self.assertEqual(result.upper_bound_tp, 2)
        self.assertEqual(brute_force_tp(slot_times, ground_truth), 2)

    def test_adjacent_slot_conflict_is_respected(self):
        slot_times = slots(2)
        ground_truth = gt_frame([(-50_000, 11_000), (20_000, 11_000)])
        result = episode_bound(slot_times, ground_truth["start_ms"].to_numpy())
        self.assertEqual(result.upper_bound_tp, 1)
        self.assertEqual(len(result.witness_slots), 1)
        self.assertEqual(brute_force_tp(slot_times, ground_truth), 1)

    def test_witness_realises_the_bound_through_the_frozen_pipeline(self):
        slot_times = slots(9)
        ground_truth = gt_frame([(10_000, 11_000), (70_000, 11_000), (400_000, 11_000)])
        result = episode_bound(slot_times, ground_truth["start_ms"].to_numpy())
        replay = replay_witness(result.witness_slots, slot_times, ground_truth, split="test")
        self.assertEqual(replay["realised_tp"], result.upper_bound_tp)
        self.assertTrue(replay["episode_anchors_match_witness"])

    def test_random_instances_agree_with_exhaustive_enumeration(self):
        rng = np.random.RandomState(7)
        for trial in range(30):
            count = int(rng.randint(4, 8))
            slot_times = slots(count)
            event_count = int(rng.randint(1, 5))
            ground_truth = gt_frame([
                (int(value), 11_000) for value in rng.randint(-20_000, (count - 1) * GRID_MS, size=event_count)
            ])
            result = episode_bound(slot_times, ground_truth["start_ms"].to_numpy())
            exhaustive = brute_force_tp(slot_times, ground_truth)
            self.assertEqual(result.upper_bound_tp, exhaustive, "trial {} mismatch".format(trial))

    def test_episode_bound_is_bounded_by_grid_bound(self):
        rng = np.random.RandomState(11)
        for _ in range(20):
            count = int(rng.randint(4, 9))
            slot_times = slots(count)
            ground_truth = gt_frame([
                (int(value), 11_000) for value in rng.randint(0, (count - 1) * GRID_MS, size=3)
            ])
            episode = episode_bound(slot_times, ground_truth["start_ms"].to_numpy()).upper_bound_tp
            grid = grid_bound(slot_times, ground_truth)["matched_events"]
            self.assertLessEqual(episode, grid)
            self.assertLessEqual(grid, len(ground_truth))

    def test_windows_longer_than_three_slots_are_rejected(self):
        slot_times = slots(4)
        with self.assertRaisesRegex(ValueError, "at most three legal slots"):
            episode_bound(slot_times, np.array([ORIGIN]), tolerance_seconds=90)

    def test_events_without_any_slot_are_ignored(self):
        slot_times = slots(4)
        ground_truth = gt_frame([(0, 11_000), (5_000_000, 11_000)])
        result = episode_bound(slot_times, ground_truth["start_ms"].to_numpy())
        self.assertEqual(result.upper_bound_tp, 1)


class LabelReferenceTests(unittest.TestCase):
    def test_label_decoding_counterexample_is_not_an_upper_bound(self):
        slot_times = slots(4)
        ground_truth = gt_frame([(-20_000, 11_000), (40_000, 11_000)])
        labels = np.array([TRIGGER_POSITIVE] * 4, dtype=np.int64)
        reference = label_reference(slot_times, labels, ground_truth, split="test")
        decoded_tp = int(reference["event_metrics"]["true_positive_events"])
        self.assertEqual(decoded_tp, 1)
        self.assertEqual(brute_force_tp(slot_times, ground_truth), 2)
        self.assertLess(decoded_tp, brute_force_tp(slot_times, ground_truth))

    def test_ignore_rows_stay_on_the_timeline(self):
        slot_times = slots(4)
        ground_truth = gt_frame([(0, 3_600_000)])
        labels = np.array(
            [TRIGGER_POSITIVE, TRIGGER_IGNORE, TRIGGER_IGNORE, TRIGGER_NEGATIVE], dtype=np.int64
        )
        reference = label_reference(slot_times, labels, ground_truth, split="test")
        self.assertEqual(reference["n_slots"], 4)
        self.assertEqual(reference["positive_slots"], 1)
        self.assertEqual(reference["episode_count"], 1)

    def test_label_reference_rejects_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "one label per legal prediction slot"):
            label_reference(slots(3), np.array([0, 1]), gt_frame([(0, 11_000)]))


if __name__ == "__main__":
    unittest.main()
