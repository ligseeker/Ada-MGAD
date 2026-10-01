"""Compare the sweep's counts to the official matching implementation."""

import numpy as np
import pandas as pd

from src.e2e.bin_trigger_decoder import (construct_bin_candidates, evaluate_bin_threshold,
                                        independent_bin_counts, select_bin_threshold)
from src.e2e.system_trigger import system_score_frame


def gt_frame(starts):
    return pd.DataFrame({"case_id": ["c{}".format(i) for i in range(len(starts))],
                         "source_index": np.arange(len(starts)),
                         "split": "validation", "service": "mobservice1", "fault_type": "test",
                         "start_ms": starts, "end_ms": np.asarray(starts) + 1000})


def test_adjacent_candidates_are_retained_and_excess_candidates_are_fp():
    frame = system_score_frame("validation", np.array([30_000, 60_000, 90_000]), np.array([.9, .9, .9]))
    candidates, matching, metrics = evaluate_bin_threshold(frame, gt_frame([1000]), .8)
    assert len(candidates) == 3
    assert metrics["true_positive_events"] == 1
    assert metrics["false_positive_events"] == 2
    assert matching.case_id.notna().sum() == 1


def test_fast_sweep_counts_and_selection_agree_with_official_matcher():
    rng = np.random.RandomState(42)
    times = np.arange(30_000, 300_001, 30_000)
    for _ in range(16):
        starts = np.sort(rng.randint(0, 300_001, 15))
        scores = rng.randint(0, 5, len(times)) / 4.0
        frame = system_score_frame("validation", times, scores)
        gt = gt_frame(starts)
        official = []
        thresholds = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
        for threshold in thresholds:
            _, _, metrics = evaluate_bin_threshold(frame, gt, threshold)
            counts = independent_bin_counts(times, scores, starts, threshold)
            assert counts == tuple(metrics[key] for key in
                                   ("true_positive_events", "false_positive_events", "false_negative_events"))
            official.append((metrics["event_f1"], threshold))
        chosen, count = select_bin_threshold(frame, gt)
        assert count == len(thresholds)
        assert chosen == max(official)[1]


def test_empty_candidate_schema_is_accepted_by_matcher():
    frame = system_score_frame("validation", np.array([30_000]), np.array([.1]))
    assert construct_bin_candidates(frame, .5).empty
    _, _, metrics = evaluate_bin_threshold(frame, gt_frame([1000]), .5)
    assert metrics["false_negative_events"] == 1
