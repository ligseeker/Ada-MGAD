"""C1 same-case Train cohort and paired GT/detected W300-B15 Z2 features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .c1_oos_matching import C1OOSMatching
from .rca_features import TemporalSpec, extract_case_features_from_indicators, flatten_features
from .rca_model import GAIA_SERVICES


G2_MINIMUM_BY_FOLD = {1: 596, 2: 765, 3: 767}


@dataclass(frozen=True)
class C1CommonCohort:
    cases: pd.DataFrame
    gt_features: np.ndarray
    detected_features: np.ndarray
    root_indices: np.ndarray
    exclusion_ledger: pd.DataFrame
    coverage_by_fold: Mapping[int, int]
    floors_pass: bool


def extract_c1_anchor_features(index, *, case_id: str, anchor_ms: int) -> np.ndarray:
    """Extract label-free 10x68 features; no GT/root argument enters the index."""
    spec = TemporalSpec.w300_b15()
    indicators = index.case_indicators(int(anchor_ms), spec)
    features = extract_case_features_from_indicators(str(case_id), GAIA_SERVICES, indicators, spec)
    if tuple(features.candidates) != GAIA_SERVICES:
        raise ValueError("C1 feature extractor changed candidate order")
    values = flatten_features(features, "z2")
    if values.shape != (10, 68) or not np.isfinite(values).all():
        raise ValueError("C1 feature extractor emitted invalid 10x68 values")
    return values.astype(np.float32)


def build_c1_common_cohort(*, fold_matchings: Sequence[C1OOSMatching], index) -> C1CommonCohort:
    """Select only legal paired Train cases, preserving exclusions and floors."""
    if len(fold_matchings) != 3 or [int(item.audit["fold"]) for item in fold_matchings] != [1, 2, 3]:
        raise ValueError("C1 cohort requires exactly the three locked OOS folds in order")
    candidates = pd.concat([item.cohort_candidates for item in fold_matchings], ignore_index=True)
    excluded = pd.concat([item.exclusion_ledger for item in fold_matchings], ignore_index=True, sort=False)
    if candidates["case_id"].duplicated().any() or candidates["prediction_id"].duplicated().any():
        raise ValueError("C1 common Train identities must be unique across all folds")
    rows, gt_arrays, detected_arrays, roots = [], [], [], []
    for record in candidates.itertuples(index=False):
        case_id = str(record.case_id)
        try:
            gt = extract_c1_anchor_features(index, case_id=case_id, anchor_ms=int(record.gt_start_ms))
            detected = extract_c1_anchor_features(index, case_id=case_id, anchor_ms=int(record.t_hat))
        except ValueError as exc:
            excluded = pd.concat([excluded, pd.DataFrame([{
                "fold": int(record.fold), "case_id": case_id,
                "prediction_id": str(record.prediction_id),
                "exclusion_reason": "invalid_paired_features",
                "detail": str(exc)}])], ignore_index=True, sort=False)
            continue
        root = str(record.gt_service)
        if root not in GAIA_SERVICES:
            raise ValueError("C1 matched GT root outside canonical candidate order")
        rows.append({"case_id": case_id, "fold": int(record.fold),
                     "prediction_id": str(record.prediction_id),
                     "gt_anchor_ms": int(record.gt_start_ms),
                     "detected_anchor_ms": int(record.t_hat),
                     "root_service": root, "fault_type": str(record.fault_type)})
        gt_arrays.append(gt)
        detected_arrays.append(detected)
        roots.append(GAIA_SERVICES.index(root))
    cases = pd.DataFrame(rows, columns=("case_id", "fold", "prediction_id", "gt_anchor_ms",
                                        "detected_anchor_ms", "root_service", "fault_type"))
    gt_features = np.stack(gt_arrays) if gt_arrays else np.empty((0, 10, 68), dtype=np.float32)
    detected_features = np.stack(detected_arrays) if detected_arrays else np.empty((0, 10, 68), dtype=np.float32)
    coverage = {fold: int((cases["fold"] == fold).sum()) for fold in (1, 2, 3)}
    return C1CommonCohort(cases, gt_features, detected_features,
                          np.asarray(roots, dtype=np.int64), excluded,
                          coverage, all(coverage[fold] >= floor for fold, floor in G2_MINIMUM_BY_FOLD.items()))


def require_c1_common_cohort_floor(cohort: C1CommonCohort) -> None:
    if not cohort.floors_pass:
        raise ValueError("C1 common Train floor NO_GO: observed={} required={}".format(
            dict(cohort.coverage_by_fold), G2_MINIMUM_BY_FOLD))
