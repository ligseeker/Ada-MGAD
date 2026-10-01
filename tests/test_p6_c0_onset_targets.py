"""Boundary and loss-mask guarantees for the onset-only controlled ablation."""

import numpy as np
import pandas as pd
import pytest

from src.e2e.onset_trigger import build_onset_bin_labels
from src.e2e.system_trigger import build_trigger_labels


def test_half_open_bin_boundaries_and_preserved_active_mask():
    times = np.arange(30_000, 240_001, 30_000)
    events = pd.DataFrame({"start_ms": [30_000], "end_ms": [230_000]})
    old = build_trigger_labels(times, events)
    new = build_onset_bin_labels(times, events, old)
    # At 30s the onset is not observed yet; [30s,60s) is observed at 60s.
    np.testing.assert_array_equal(new, [0, 1, 0, 2, 2, 2, 2, 0])
    np.testing.assert_array_equal(new == 2, old == 2)


def test_bin_start_inclusive_bin_end_exclusive_with_multiplicity():
    times = np.array([30_000, 60_000, 90_000, 120_000])
    events = pd.DataFrame({"start_ms": [0, 29_999, 30_000, 90_000],
                           "end_ms": [10, 30_001, 30_100, 90_100]})
    old = build_trigger_labels(times, events)
    np.testing.assert_array_equal(build_onset_bin_labels(times, events, old), [1, 1, 0, 1])


def test_new_onset_overrides_other_active_event_without_changing_mask():
    times = np.arange(30_000, 180_001, 30_000)
    events = pd.DataFrame({"start_ms": [0, 120_001], "end_ms": [300_000, 130_000]})
    old = build_trigger_labels(times, events)
    new = build_onset_bin_labels(times, events, old)
    assert new[4] == 1 and new[5] == 0
    np.testing.assert_array_equal(new == 2, old == 2)


def test_inconsistent_baseline_fails_closed():
    events = pd.DataFrame({"start_ms": [0], "end_ms": [100_000]})
    with pytest.raises(ValueError, match="IGNORE"):
        build_onset_bin_labels([30_000], events, [2])
    with pytest.raises(ValueError, match="subset"):
        build_onset_bin_labels([30_000], events, [0])


def test_empty_events_and_invalid_time_grid():
    events = pd.DataFrame({"start_ms": [], "end_ms": []})
    np.testing.assert_array_equal(build_onset_bin_labels([30_000, 60_000], events, [0, 0]), [0, 0])
    with pytest.raises(ValueError, match="strictly increasing"):
        build_onset_bin_labels([30_000, 30_000], events, [0, 0])
