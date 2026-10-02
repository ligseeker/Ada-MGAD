import json

import numpy as np
import pandas as pd

from scripts.p6.run_c1_xgb_forward_dev import rank_rows, summarize
from src.e2e.rca_model import GAIA_SERVICES


def test_rank_rows_complete_and_canonical_tie_break():
    rankings = rank_rows(np.zeros((2, 10)))
    assert len(rankings) == 2
    assert all(json.loads(value) == list(GAIA_SERVICES) for value in rankings)


def test_paired_summary_preserves_all_cases():
    cases = pd.DataFrame({"rank_cl": [1, 1, 2, 4],
                          "rank_xgb": [1, 2, 1, 3]})
    summary = summarize(cases)
    assert summary["n"] == 4
    assert summary["paired_top1"] == {
        "both_correct": 1, "cl_only": 1, "xgb_only": 1, "both_incorrect": 1}
    assert summary["delta_ac1_xgb_minus_cl"] == 0
    assert summary["arms"]["cl"]["AC@3"] == 0.75
    assert summary["arms"]["xgb"]["AC@3"] == 1.0
