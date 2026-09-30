import numpy as np
import pandas as pd

from scripts.p6.audit_c0r3_equal_alert_control import (
    THRESHOLD, candidate_extra_by_length, context_flags, placebo_anchors,
)


def test_placebo_preserves_alert_budget_by_run_length_and_is_deterministic():
    scores = pd.DataFrame({
        "score": [THRESHOLD + 0.00001] * 3 + [0.0]
                 + [THRESHOLD + 0.00001] * 2 + [0.0]
                 + [THRESHOLD + 0.00001],
        "logit": [1.0, 2.0, 1.0, -1.0, 1.0, 2.0, -1.0, 1.0],
    })
    starts, eligible, chosen = candidate_extra_by_length(scores)
    assert starts.tolist() == [0, 4, 7]
    assert {k: len(v) for k, v in chosen.items()} == {3: 1, 2: 1}
    a = placebo_anchors(starts, eligible, chosen, 42)
    b = placebo_anchors(starts, eligible, chosen, 42)
    assert np.array_equal(a, b)
    assert len(a) == 5 and set(starts).issubset(a)
    assert sum(x in (1, 2) for x in a) == 1
    assert 5 in a


def test_clean_context_is_stricter_than_isolated_onset():
    gt = pd.DataFrame({"start_ms": [0, 120000, 500000],
                       "end_ms": [300000, 131000, 511000]})
    flags = context_flags(gt)
    assert flags["isolated_onset"].tolist() == [True, True, True]
    assert flags["clean_context"].tolist() == [True, False, True]
