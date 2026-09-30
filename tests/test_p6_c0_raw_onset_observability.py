import numpy as np

from scripts.p6.audit_c0_raw_onset_observability import (
    no_event_intersects, raw_change, slot_indices,
)


def test_slot_geometry_uses_ten_pre_bins_and_first_two_post_bins():
    ends = np.arange(1, 21, dtype=np.int64) * 30000
    slot = slot_indices(ends, 315000, (0, 600000))
    assert slot == (0, 10, 10, 12)
    assert slot_indices(ends, 150000, (0, 600000)) is None


def test_raw_change_is_calculated_from_pre_event_median():
    values = np.zeros((20, 1, 2), dtype=float)
    values[10, 0] = [2.0, 4.0]
    values[11, 0] = [4.0, 0.0]
    arrays = {name: values for name in ("metric", "log", "trace")}
    result = raw_change(arrays, (0, 10, 10, 12))
    assert result["metric_mean_abs"] == 3.0
    assert result["metric_max_abs"] == 4.0


def test_negative_pool_rejects_intersecting_events():
    starts = np.array([100, 300], dtype=np.int64)
    ends = np.array([200, 400], dtype=np.int64)
    assert no_event_intersects(starts, ends, 201, 299)
    assert not no_event_intersects(starts, ends, 200, 300)
