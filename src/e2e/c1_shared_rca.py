"""C1 same-case RCA arms with one GT-anchor Train scaler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from .rca_model import (
    ConditionalLogitFit, FEATURE_DIMENSION, GAIA_SERVICES, L2_LAMBDA,
    N_CANDIDATES, fit_conditional_logit,
)


@dataclass(frozen=True)
class C1SharedRcaArms:
    case_ids: Tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    gt_arm_b: ConditionalLogitFit
    detected_arm_c: ConditionalLogitFit


def fit_c1_shared_rca_arms(*, case_ids: Sequence[str], gt_features: np.ndarray,
                           detected_features: np.ndarray,
                           root_indices: Sequence[int],
                           candidates: Sequence[str] = GAIA_SERVICES) -> C1SharedRcaArms:
    """Fit B/C on identical cases and labels; only anchor features differ."""
    ids = tuple(str(value) for value in case_ids)
    gt = np.asarray(gt_features, dtype=np.float64)
    detected = np.asarray(detected_features, dtype=np.float64)
    roots = np.asarray(root_indices)
    if (not ids or len(set(ids)) != len(ids) or any(not value for value in ids)
            or tuple(candidates) != GAIA_SERVICES):
        raise ValueError("C1 common Train cases/candidate order invalid")
    if (gt.shape != detected.shape or gt.shape != (len(ids), N_CANDIDATES, FEATURE_DIMENSION)
            or not np.isfinite(gt).all() or not np.isfinite(detected).all()):
        raise ValueError("C1 paired GT/detected features must be finite aligned 10x68 tensors")
    if roots.shape != (len(ids),) or not np.issubdtype(roots.dtype, np.integer):
        raise ValueError("C1 roots must be one integer candidate index per common case")
    if np.any(roots < 0) or np.any(roots >= N_CANDIDATES):
        raise ValueError("C1 root index outside canonical candidates")
    scaler = StandardScaler().fit(gt.reshape(-1, FEATURE_DIMENSION))
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    arm_b = fit_conditional_logit(gt, roots, scaler_mean=mean, scaler_scale=scale,
                                  l2_lambda=L2_LAMBDA)
    arm_c = fit_conditional_logit(detected, roots, scaler_mean=mean, scaler_scale=scale,
                                  l2_lambda=L2_LAMBDA)
    for fitted in (arm_b, arm_c):
        if not np.array_equal(fitted.scaler_mean, mean) or not np.array_equal(fitted.scaler_scale, scale):
            raise ValueError("C1 arm did not preserve the single shared scaler")
    if arm_b.train_case_indices != arm_c.train_case_indices:
        raise ValueError("C1 RCA arms used different common Train cases")
    return C1SharedRcaArms(ids, mean, scale, arm_b, arm_c)
