"""Contract checks for GAIA event-group XGBRanker transport."""

import json

import numpy as np
import pandas as pd
import pytest

from src.e2e.c1_xgb_adapter import (
    evaluator_scope, grouped_training_data, paired_scope,
)
from src.e2e.rca_model import GAIA_SERVICES


class _FirstFeatureScorer:
    def predict(self, rows):
        return rows[:, 0]


def test_grouped_training_keeps_one_root_and_ten_contiguous_candidates():
    features = np.arange(2 * 10 * 68, dtype=np.float32).reshape(2, 10, 68)
    rows, target, group = grouped_training_data(features, np.array([4, 7]))
    assert rows.shape == (20, 68)
    assert np.array_equal(rows[10], features[1, 0])
    assert np.array_equal(group, [10, 10])
    assert np.array_equal(np.flatnonzero(target), [4, 17])
    with pytest.raises(ValueError):
        grouped_training_data(features, np.array([4, 10]))


def test_paired_test_scope_preserves_all_episodes_and_old_c_rankings():
    ranking = json.dumps(GAIA_SERVICES)
    old = pd.DataFrame({
        "prediction_id": ["p1", "p2"], "t_hat": [100, 200],
        "scope_status": ["legal", "illegal_context"],
        "ranking_status": ["complete", "not_applicable"],
        "failure_reason": ["", "detected_context_outside_test"],
        "ranking_c": [ranking, ""],
    })
    episodes = pd.DataFrame({"prediction_id": ["p1", "p2"], "t_hat": [100, 200]})
    features = np.zeros((2, 10, 68), dtype=np.float32)
    features[0, :, 0] = np.arange(10)
    mask = np.array([True, False])
    scope = paired_scope(old_scope=old, episodes=episodes, features=features,
                         valid=mask, model=_FirstFeatureScorer())
    assert len(scope) == 2
    assert scope.loc[0, "ranking_cl"] == ranking
    assert json.loads(scope.loc[0, "ranking_xgb"])[0] == GAIA_SERVICES[-1]
    assert scope.loc[1, "ranking_xgb"] == ""
    mapped = evaluator_scope(scope)
    assert mapped.loc[0, "ranking_b"] == ranking
    assert mapped.loc[0, "ranking_c"] == scope.loc[0, "ranking_xgb"]
    with pytest.raises(ValueError):
        paired_scope(old_scope=old, episodes=episodes.iloc[::-1].reset_index(drop=True),
                     features=features, valid=mask, model=_FirstFeatureScorer())
    with pytest.raises(ValueError):
        paired_scope(old_scope=old, episodes=episodes, features=features,
                     valid=np.array([False, True]), model=_FirstFeatureScorer())
