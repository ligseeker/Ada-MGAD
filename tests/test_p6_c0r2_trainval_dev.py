import numpy as np
import pandas as pd

from scripts.p6.audit_c0r2_trainval_dev import _candidate, match_by_time_component
from src.e2e.event_detection import match_events


def test_rise_decoder_adds_only_strict_positive_logit_rises():
    frame = pd.DataFrame({
        "prediction_available_time": np.arange(6, dtype=np.int64) * 30000,
        "score": [0.5, 0.9999, 0.9999, 0.9999, 0.1, 0.9999],
        "logit": [0.0, 9.0, 8.0, 8.5, -1.0, 9.0],
    })
    result = _candidate(frame, "fit")
    assert result["t_hat"].tolist() == [30000, 90000, 150000]
    assert result["trigger_kind"].tolist() == ["run_start", "positive_logit_rise", "run_start"]


def test_component_matcher_agrees_with_exact_matcher_across_time_gap():
    episodes = pd.DataFrame({"prediction_id": ["a", "b", "c"],
                             "t_hat": [30000, 90000, 300000]})
    gt = pd.DataFrame({"case_id": ["g1", "g2", "g3"], "source_index": [1, 2, 3],
                       "service": ["s"] * 3, "fault_type": ["f"] * 3,
                       "start_ms": [1000, 50000, 280000],
                       "end_ms": [11000, 60000, 290000]})
    full = match_events(episodes, gt)
    parts = match_by_time_component(episodes, gt)
    key = lambda frame: sorted((str(row.prediction_id), str(row.case_id), row.match_status)
                               for row in frame.itertuples(index=False))
    assert key(parts) == key(full)
