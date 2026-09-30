"""GAIA event-group adapter for the frozen Ada-RCA Z2 XGBRanker scorer.

This module accepts already sealed C1-v2 feature tensors. It never extracts
features, chooses an anchor, fits a scaler, or reads Test labels.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .rca_model import FEATURE_DIMENSION, GAIA_SERVICES, N_CANDIDATES, rank_candidates


MODEL_PARAMETERS = {
    "objective": "rank:pairwise",
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "random_state": 20260826,
    "n_jobs": 1,
    "tree_method": "hist",
}


def grouped_training_data(features: np.ndarray, roots: np.ndarray):
    """Make one contiguous ten-service query with one positive per Train case."""
    values = np.asarray(features)
    indices = np.asarray(roots)
    if (values.ndim != 3 or values.shape[1:] != (N_CANDIDATES, FEATURE_DIMENSION)
            or values.shape[0] < 1 or not np.issubdtype(values.dtype, np.number)
            or not np.isfinite(values).all()):
        raise ValueError("Train features must be finite [case,10,68]")
    if (indices.shape != (len(values),) or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= N_CANDIDATES)):
        raise ValueError("Train roots must be canonical candidate indices")
    matrix = np.ascontiguousarray(values.reshape(-1, FEATURE_DIMENSION), dtype=np.float32)
    target = np.zeros((len(values), N_CANDIDATES), dtype=np.float32)
    target[np.arange(len(values)), indices] = 1.0
    group = np.full(len(values), N_CANDIDATES, dtype=np.int32)
    if int(target.sum()) != len(values) or int(group.sum()) != len(matrix):
        raise ValueError("Train event-group boundary or relevance mismatch")
    return matrix, target.reshape(-1), group


def rank_label_free_features(model, features: np.ndarray, valid: np.ndarray):
    """Score only sealed legal Test episode features, retaining their order."""
    values = np.asarray(features)
    mask = np.asarray(valid)
    if (values.ndim != 3 or values.shape[1:] != (N_CANDIDATES, FEATURE_DIMENSION)
            or mask.shape != (len(values),) or mask.dtype != np.dtype(bool)):
        raise ValueError("Test feature/mask shape mismatch")
    selected = values[mask]
    if not np.isfinite(selected).all():
        raise ValueError("legal Test features are non-finite")
    flat = np.ascontiguousarray(selected.reshape(-1, FEATURE_DIMENSION), dtype=np.float32)
    scores = np.asarray(model.predict(flat), dtype=np.float64)
    if scores.shape != (len(flat),) or not np.isfinite(scores).all():
        raise ValueError("XGBRanker returned invalid Test scores")
    rankings = [""] * len(values)
    for position, score_row in zip(np.flatnonzero(mask), scores.reshape(-1, N_CANDIDATES)):
        ranked = rank_candidates(GAIA_SERVICES, score_row)
        rankings[int(position)] = json.dumps(ranked, separators=(",", ":"))
    return rankings


def paired_scope(*, old_scope: pd.DataFrame, episodes: pd.DataFrame,
                 features: np.ndarray, valid: np.ndarray, model) -> pd.DataFrame:
    """Reuse sealed C1 Arm C as comparator; XGB sees no Test labels."""
    if (len(old_scope) != len(episodes) or len(features) != len(episodes)
            or old_scope["prediction_id"].duplicated().any()
            or old_scope["prediction_id"].tolist() != episodes["prediction_id"].tolist()
            or not np.array_equal(old_scope["t_hat"].to_numpy(dtype=np.int64),
                                  episodes["t_hat"].to_numpy(dtype=np.int64))):
        raise ValueError("sealed Test episode order or anchors differ")
    legal = old_scope["scope_status"].eq("legal").to_numpy()
    if (not np.array_equal(legal, valid)
            or not old_scope.loc[legal, "ranking_status"].eq("complete").all()
            or not old_scope.loc[~legal, "ranking_status"].eq("not_applicable").all()):
        raise ValueError("sealed Test feature mask or context statuses differ")
    for ranking in old_scope.loc[legal, "ranking_c"]:
        try:
            decoded = json.loads(ranking)
        except (TypeError, ValueError) as exc:
            raise ValueError("sealed Conditional Logit ranking is invalid") from exc
        if (not isinstance(decoded, list) or len(decoded) != N_CANDIDATES
                or len(set(decoded)) != N_CANDIDATES or set(decoded) != set(GAIA_SERVICES)):
            raise ValueError("sealed Conditional Logit ranking is incomplete")
    rankings = rank_label_free_features(model, features, valid)
    frame = old_scope[["prediction_id", "t_hat", "scope_status", "ranking_status",
                       "failure_reason"]].copy()
    frame["ranking_cl"] = old_scope["ranking_c"].to_numpy()
    frame["ranking_xgb"] = rankings
    if (not frame.loc[legal, ["ranking_cl", "ranking_xgb"]].ne("").all().all()
            or not frame.loc[~legal, ["ranking_cl", "ranking_xgb"]].eq("").all().all()):
        raise ValueError("paired Test scope contains a missing legal ranking")
    return frame


def evaluator_scope(scope: pd.DataFrame) -> pd.DataFrame:
    """Map CL/XGB to evaluator slots b/c without changing evaluator arithmetic."""
    required = {"prediction_id", "t_hat", "scope_status", "ranking_status",
                "failure_reason", "ranking_cl", "ranking_xgb"}
    if required - set(scope.columns):
        raise ValueError("paired scope lacks required fields")
    return scope.rename(columns={"ranking_cl": "ranking_b", "ranking_xgb": "ranking_c"})
