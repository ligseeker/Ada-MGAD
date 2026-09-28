"""Label-free B/C rankings for every frozen C0 Test detected episode."""

from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import pandas as pd

from .c1_common_cohort import extract_c1_anchor_features
from .rca_model import ConditionalLogitFit, GAIA_SERVICES, rank_candidates


TEST_EPISODE_COLUMNS = (
    "prediction_id", "split", "t_hat", "episode_end_time", "positive_bins",
    "system_score", "threshold",
)


def score_c1_test_episodes_with_features(*, episodes: pd.DataFrame, index,
                                         arm_b: ConditionalLogitFit, arm_c: ConditionalLogitFit,
                                         test_interval_ms: tuple) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Rank the complete episode universe before any Test GT/matching read."""
    if tuple(episodes.columns) != TEST_EPISODE_COLUMNS:
        raise ValueError("C1 Test scorer accepts only the frozen label-free episode schema")
    if episodes.empty or episodes["prediction_id"].duplicated().any() or not episodes["split"].eq("test").all():
        raise ValueError("C1 Test episode universe must be nonempty, unique and Test-only")
    start, end = (int(value) for value in test_interval_ms)
    if start >= end:
        raise ValueError("C1 Test interval is invalid")
    times = pd.to_numeric(episodes["t_hat"], errors="raise").to_numpy(dtype=np.int64)
    episode_ends = pd.to_numeric(episodes["episode_end_time"], errors="raise").to_numpy(dtype=np.int64)
    strengths = pd.to_numeric(episodes["system_score"], errors="raise").to_numpy(dtype=float)
    thresholds = pd.to_numeric(episodes["threshold"], errors="raise").to_numpy(dtype=float)
    lengths = pd.to_numeric(episodes["positive_bins"], errors="raise").to_numpy(dtype=np.int64)
    if (np.any(times < start) or np.any(times >= end) or np.any(episode_ends <= times)
            or np.any(episode_ends > end) or np.any(np.diff(times) <= 0)
            or not np.isfinite(strengths).all() or not np.isfinite(thresholds).all()
            or np.any(strengths < 0) or np.any(strengths > 1)
            or np.any(thresholds < 0) or np.any(thresholds > 1)
            or np.any(lengths < 1)):
        raise ValueError("C1 Test episodes leave the frozen chronological interval")
    if (not np.array_equal(arm_b.scaler_mean, arm_c.scaler_mean)
            or not np.array_equal(arm_b.scaler_scale, arm_c.scaler_scale)):
        raise ValueError("C1 B/C models do not share the GT-fitted scaler")
    rows = []
    features_by_episode = np.zeros((len(episodes), 10, 68), dtype=np.float32)
    feature_valid = np.zeros(len(episodes), dtype=bool)
    for position, episode in enumerate(episodes.itertuples(index=False)):
        anchor = int(episode.t_hat)
        row = {"prediction_id": str(episode.prediction_id), "t_hat": anchor,
               "scope_status": "legal", "ranking_status": "complete", "failure_reason": "",
               "ranking_b": "", "ranking_c": ""}
        if anchor - 300000 < start or anchor + 300000 > end:
            row["scope_status"] = "illegal_context"
            row["ranking_status"] = "not_applicable"
            row["failure_reason"] = "detected_context_outside_test"
        else:
            try:
                features = extract_c1_anchor_features(
                    index, case_id=str(episode.prediction_id), anchor_ms=anchor)
                features_by_episode[position] = features
                feature_valid[position] = True
            except ValueError as exc:
                row["ranking_status"] = "failed"
                row["failure_reason"] = "invalid_feature: {}".format(exc)
            if feature_valid[position]:
                failures = []
                for name, model in (("b", arm_b), ("c", arm_c)):
                    try:
                        ranked = rank_candidates(GAIA_SERVICES, model.scores(features))
                        row["ranking_" + name] = json.dumps(ranked, separators=(",", ":"))
                    except ValueError as exc:
                        failures.append("{}:{}".format(name, exc))
                if failures:
                    row["ranking_status"] = "failed"
                    row["failure_reason"] = "invalid_score:" + ";".join(failures)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=("prediction_id", "t_hat", "scope_status", "ranking_status",
                                        "failure_reason", "ranking_b", "ranking_c"))
    return frame, features_by_episode, feature_valid


def score_c1_test_episodes(*, episodes: pd.DataFrame, index,
                           arm_b: ConditionalLogitFit, arm_c: ConditionalLogitFit,
                           test_interval_ms: tuple) -> pd.DataFrame:
    frame, _, _ = score_c1_test_episodes_with_features(
        episodes=episodes, index=index, arm_b=arm_b, arm_c=arm_c,
        test_interval_ms=test_interval_ms)
    return frame
