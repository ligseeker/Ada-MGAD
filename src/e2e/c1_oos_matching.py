"""Train-only C1 OOS episode matching after a fold's prediction seal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .event_detection import match_events


@dataclass(frozen=True)
class C1OOSMatching:
    matching: pd.DataFrame
    cohort_candidates: pd.DataFrame
    exclusion_ledger: pd.DataFrame
    audit: Mapping[str, int]


def match_c1_oos_episodes(*, fold: int, interval_ms: tuple,
                          episodes: pd.DataFrame, registry: pd.DataFrame,
                          context_seconds: int = 300) -> C1OOSMatching:
    """Join Generation GT only after episodes exist; preserve all misses/FPs.

    This does not extract features or apply the G2 per-fold common-case floor.
    It only identifies matched events for which both anchor windows fit within
    the same Generation interval. No ranking or correctness enters selection.
    """
    start, end = (int(value) for value in interval_ms)
    if fold not in (1, 2, 3) or start >= end or context_seconds != 300:
        raise ValueError("C1 matching fold/context differs from G2")
    required = {"prediction_id", "t_hat", "episode_end_time", "fold"}
    if required - set(episodes.columns):
        raise ValueError("C1 episodes lack sealed Generation identity or time")
    forbidden = {"case_id", "gt_service", "fault_type", "trigger_label", "root_service", "match_status"}
    if forbidden & set(episodes.columns):
        raise ValueError("C1 Generation episodes contain GT fields")
    predictions = episodes.copy()
    if (predictions["prediction_id"].duplicated().any()
            or not predictions["fold"].eq(fold).all()
            or not predictions["prediction_id"].astype(str).str.startswith("fold_{:02d}-".format(fold)).all()):
        raise ValueError("C1 Generation prediction identities drift")
    times = pd.to_numeric(predictions["t_hat"], errors="raise").to_numpy(dtype=np.int64)
    ends = pd.to_numeric(predictions["episode_end_time"], errors="raise").to_numpy(dtype=np.int64)
    if np.any(times < start) or np.any(times >= end) or np.any(ends <= times) or np.any(ends > end):
        raise ValueError("C1 Generation episode leaves its half-open interval")
    gt_columns = {"case_id", "source_index", "service", "fault_type", "start_ms", "end_ms", "detector_domain"}
    if gt_columns - set(registry.columns):
        raise ValueError("C1 GT registry lacks required event fields")
    domain = registry.loc[registry["detector_domain"].astype(bool)].copy()
    if domain["case_id"].duplicated().any():
        raise ValueError("C1 GT case_id must be unique")
    onset = pd.to_numeric(domain["start_ms"], errors="raise")
    event_end = pd.to_numeric(domain["end_ms"], errors="raise")
    if (event_end <= onset).any():
        raise ValueError("C1 GT events must have positive duration")
    intersecting = domain.loc[(onset < end) & (event_end > start)].copy()
    complete = intersecting.loc[(intersecting["start_ms"] >= start)
                                & (intersecting["end_ms"] <= end)].copy()
    boundary = intersecting.loc[~intersecting["case_id"].isin(complete["case_id"])].copy()
    matching = match_events(predictions, complete, tolerance_seconds=60)
    matching.insert(0, "fold", int(fold))
    matched = matching.loc[matching["match_status"] == "matched"].copy()
    history_ms = int(context_seconds) * 1000
    gt_time = matched["gt_start_ms"].to_numpy(dtype=np.int64)
    detected_time = matched["t_hat"].to_numpy(dtype=np.int64)
    legal_gt = (gt_time - history_ms >= start) & (gt_time + history_ms <= end)
    legal_detected = (detected_time - history_ms >= start) & (detected_time + history_ms <= end)
    matched["gt_context_legal"] = legal_gt
    matched["detected_context_legal"] = legal_detected
    matched["context_legal"] = legal_gt & legal_detected
    matched["exclusion_reason"] = np.where(
        ~legal_gt, "gt_context_outside_generation",
        np.where(~legal_detected, "detected_context_outside_generation", ""))
    eligible = matched.loc[matched["context_legal"]].copy()
    exclusions = matched.loc[~matched["context_legal"]].copy()
    if not boundary.empty:
        excluded_boundary = pd.DataFrame({
            "fold": int(fold), "case_id": boundary["case_id"].astype(str),
            "prediction_id": None, "exclusion_reason": "gt_event_crosses_generation_boundary"})
        exclusions = pd.concat([exclusions, excluded_boundary], ignore_index=True, sort=False)
    audit = {"fold": int(fold), "complete_gt": int(len(complete)),
             "cross_boundary_gt": int(len(boundary)),
             "episodes": int(len(predictions)),
             "matched": int((matching["match_status"] == "matched").sum()),
             "false_alarms": int((matching["match_status"] == "false_alarm").sum()),
             "misses": int((matching["match_status"] == "miss").sum()),
             "dual_context_legal": int(len(eligible)),
             "matched_context_illegal": int(len(matched) - len(eligible))}
    return C1OOSMatching(matching, eligible, exclusions, audit)
