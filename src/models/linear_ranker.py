"""Independent, train-fold-only linear service scorer for P2."""

import hashlib
import json
from typing import Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.schema import RCACaseLabel


class LinearRankerError(ValueError):
    """Raised when feature rows, labels, or rankings do not align."""


def select_feature_columns(
    feature_names: Sequence[str], prefixes: Sequence[str]
) -> Tuple[int, ...]:
    """Select a non-empty, deterministic subset by explicit name prefix."""

    prefixes = tuple(prefixes)
    if not prefixes or any(not prefix for prefix in prefixes):
        raise LinearRankerError("feature prefixes must be non-empty")
    selected = tuple(
        index
        for index, name in enumerate(feature_names)
        if any(name.startswith(prefix) for prefix in prefixes)
    )
    if not selected:
        raise LinearRankerError("feature selection produced no columns")
    return selected


def case_balanced_targets(index, labels: Sequence[RCACaseLabel]) -> tuple:
    """Build row targets with total root/non-root weight 0.5 per case."""

    label_by_case = {label.case_id: label for label in labels}
    if len(label_by_case) != len(labels):
        raise LinearRankerError("training labels contain duplicate case IDs")
    rows_by_case = {}
    for row_index, record in enumerate(index):
        rows_by_case.setdefault(record["case_id"], []).append(
            (row_index, record["service"])
        )
    if set(rows_by_case) != set(label_by_case):
        raise LinearRankerError("feature index and training labels do not align")
    targets = np.zeros(len(index), dtype=np.int8)
    weights = np.zeros(len(index), dtype=np.float64)
    for case_id, rows in rows_by_case.items():
        root_service = label_by_case[case_id].root_service
        root_rows = [row_index for row_index, service in rows if service == root_service]
        if len(root_rows) != 1 or len(rows) < 2:
            raise LinearRankerError("each case requires one root and one non-root")
        targets[root_rows[0]] = 1
        weights[root_rows[0]] = 0.5
        non_root_weight = 0.5 / (len(rows) - 1)
        for row_index, service in rows:
            if service != root_service:
                weights[row_index] = non_root_weight
    return targets, weights


def rank_service_scores(index, scores: Sequence[float]) -> Mapping[str, tuple]:
    """Project finite row scores to complete per-case service rankings."""

    numeric = np.asarray(scores, dtype=np.float64)
    if numeric.shape != (len(index),) or not np.isfinite(numeric).all():
        raise LinearRankerError("service scores must be finite and row-aligned")
    by_case = {}
    for record, score in zip(index, numeric):
        by_case.setdefault(record["case_id"], []).append(
            (record["service"], float(score))
        )
    rankings = {}
    for case_id, rows in by_case.items():
        services = [service for service, _ in rows]
        if len(set(services)) != len(services):
            raise LinearRankerError("candidate services must be unique per case")
        rankings[case_id] = tuple(
            service
            for service, _ in sorted(rows, key=lambda row: (-row[1], row[0]))
        )
    return rankings


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + contiguous.tobytes()).hexdigest()


class IndependentLinearRanker:
    """Standardized binary logistic scorer fitted only on supplied rows."""

    def __init__(self, regularization_c: float, random_seed: int = 20260819):
        if regularization_c <= 0:
            raise LinearRankerError("regularization C must be positive")
        self.regularization_c = float(regularization_c)
        self.random_seed = int(random_seed)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.regularization_c,
            max_iter=1000,
            random_state=self.random_seed,
            solver="liblinear",
            tol=1e-6,
        )
        self._fitted = False

    def fit(self, features, targets, sample_weights):
        matrix = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.int8)
        weights = np.asarray(sample_weights, dtype=np.float64)
        if matrix.ndim != 2 or targets.shape != (len(matrix),):
            raise LinearRankerError("training arrays do not align")
        if weights.shape != targets.shape or not np.isfinite(matrix).all():
            raise LinearRankerError("training weights or values are invalid")
        if set(np.unique(targets)) != {0, 1} or np.any(weights <= 0):
            raise LinearRankerError("training requires weighted binary targets")
        standardized = self.scaler.fit_transform(matrix)
        self.model.fit(standardized, targets, sample_weight=weights)
        self._fitted = True
        return self

    def score(self, features) -> np.ndarray:
        if not self._fitted:
            raise LinearRankerError("ranker must be fitted before scoring")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise LinearRankerError("scoring features must be a finite matrix")
        return self.model.decision_function(self.scaler.transform(matrix))

    def audit(self) -> Mapping[str, object]:
        if not self._fitted:
            raise LinearRankerError("ranker must be fitted before audit")
        return {
            "coefficient_sha256": _array_digest(self.model.coef_),
            "intercept": [float(value) for value in self.model.intercept_],
            "n_iter": [int(value) for value in self.model.n_iter_],
            "scaler_mean_sha256": _array_digest(self.scaler.mean_),
            "scaler_scale_sha256": _array_digest(self.scaler.scale_),
        }
