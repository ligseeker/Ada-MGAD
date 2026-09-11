"""Frozen P5-I1 event-level RCA model and label-only baselines.

The numerical core is copied minimally from Ada-RCA-cleanup's frozen P4
implementation at commit ``a2c620922e7c0ab3615d34654d4a3690d1b22c8e``.
This module deliberately accepts an event tensor (cases, candidates, 68),
where the labelled fault service is represented separately by an integer
candidate index.  Candidate rows are observations within an event, not
independent training examples.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


SOURCE_COMMIT = "a2c620922e7c0ab3615d34654d4a3690d1b22c8e"
FEATURE_DIMENSION = 68
N_CANDIDATES = 10
L2_LAMBDA = 1.0
MAX_ITER = 1000
GRADIENT_TOLERANCE = 1e-8

# The protocol's deterministic candidate order.  Callers may provide another
# explicit order in tests, but production GAIA code should use this tuple.
GAIA_SERVICES = (
    "dbservice1", "dbservice2", "logservice1", "logservice2",
    "mobservice1", "mobservice2", "redisservice1", "redisservice2",
    "webservice1", "webservice2",
)


def _as_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (N_CANDIDATES, FEATURE_DIMENSION):
        raise ValueError("features must have shape (cases, 10, 68)")
    if values.shape[0] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("features must be non-empty and finite")
    return values


def _as_root_indices(root_indices: Sequence[int], n_cases: int) -> np.ndarray:
    roots = np.asarray(root_indices)
    if roots.ndim != 1 or len(roots) != n_cases:
        raise ValueError("root_indices must have one index per case")
    if not np.issubdtype(roots.dtype, np.integer):
        if not np.all(np.isfinite(roots)) or not np.all(roots == roots.astype(np.int64)):
            raise ValueError("root_indices must be integers")
    roots = roots.astype(np.int64, copy=False)
    if np.any(roots < 0) or np.any(roots >= N_CANDIDATES):
        raise ValueError("root index outside the ten-candidate event")
    return roots


def _indices(indices: Optional[Sequence[int]], n_cases: int, name: str) -> np.ndarray:
    if indices is None:
        return np.arange(n_cases, dtype=np.int64)
    values = np.asarray(indices)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("{} must be a one-dimensional integer index sequence".format(name))
    values = values.astype(np.int64, copy=False)
    if len(values) == 0 or np.any(values < 0) or np.any(values >= n_cases):
        raise ValueError("{} contains an invalid case index".format(name))
    if len(np.unique(values)) != len(values):
        raise ValueError("{} must not contain duplicate case indices".format(name))
    return values


def _candidate_names(candidates: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if candidates is None:
        values = GAIA_SERVICES
    else:
        values = tuple(str(x) for x in candidates)
    if len(values) != N_CANDIDATES or len(set(values)) != N_CANDIDATES:
        raise ValueError("candidates must contain ten unique services")
    return values


def _event_loss_gradient(
    weights: np.ndarray,
    matrices: Sequence[np.ndarray],
    root_indices: Sequence[int],
    l2_lambda: float,
) -> Tuple[float, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    loss = 0.5 * float(l2_lambda) * float(weights.dot(weights))
    gradient = float(l2_lambda) * weights.copy()
    for matrix, root_index in zip(matrices, root_indices):
        scores = np.asarray(matrix, dtype=np.float64).dot(weights)
        maximum = float(np.max(scores))
        exp_scores = np.exp(scores - maximum)
        probabilities = exp_scores / np.sum(exp_scores)
        loss += maximum + float(np.log(np.sum(exp_scores))) - float(scores[int(root_index)])
        gradient += np.asarray(matrix, dtype=np.float64).T.dot(probabilities)
        gradient -= np.asarray(matrix, dtype=np.float64)[int(root_index)]
    return float(loss), gradient


def _event_hessian(weights: np.ndarray, matrices: Sequence[np.ndarray], l2_lambda: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    hessian = float(l2_lambda) * np.eye(weights.size, dtype=np.float64)
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=np.float64)
        scores = matrix.dot(weights)
        shifted = scores - np.max(scores)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
        centered = matrix - probabilities.dot(matrix)[None, :]
        hessian += centered.T.dot(probabilities[:, None] * centered)
    return hessian


@dataclass(frozen=True)
class ConditionalLogitFit:
    """Persistable frozen ranker state, including the train-only scaler."""

    weights: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    train_case_indices: Tuple[int, ...]
    initial_loss: float
    final_loss: float
    gradient_norm: float
    iterations: int
    converged: bool
    message: str
    source_commit: str = SOURCE_COMMIT

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim not in (2, 3) or values.shape[-1] != FEATURE_DIMENSION:
            raise ValueError("features must end in 68 dimensions")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        return (values - self.scaler_mean) / self.scaler_scale

    def scores(self, features: np.ndarray) -> np.ndarray:
        values = self.transform(features)
        return np.dot(values, self.weights)

    # Alias useful to callers that distinguish model inference from raw scores.
    predict_scores = scores


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_conditional_logit(path: Path, fit: ConditionalLogitFit) -> Mapping[str, object]:
    """Persist the frozen model and train-only scaler with a checksum."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weights=np.asarray(fit.weights, dtype=np.float64),
        scaler_mean=np.asarray(fit.scaler_mean, dtype=np.float64),
        scaler_scale=np.asarray(fit.scaler_scale, dtype=np.float64),
        train_case_indices=np.asarray(fit.train_case_indices, dtype=np.int64),
    )
    metadata = {
        "schema_version": "p5_i1_conditional_logit_state_v1",
        "source_commit": fit.source_commit,
        "feature_dimension": FEATURE_DIMENSION,
        "candidate_count": N_CANDIDATES,
        "lambda": L2_LAMBDA,
        "max_iter": MAX_ITER,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "optimizer": "scipy.optimize.minimize:L-BFGS-B with deterministic Newton polish",
        "standard_scaler_fit": "Train candidate rows only",
        "initial_loss": float(fit.initial_loss),
        "final_loss": float(fit.final_loss),
        "gradient_norm": float(fit.gradient_norm),
        "iterations": int(fit.iterations),
        "converged": bool(fit.converged),
        "message": str(fit.message),
        "array_file": path.name,
        "array_sha256": _sha256_file(path),
    }
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_conditional_logit(path: Path) -> ConditionalLogitFit:
    """Load a persisted model, rejecting checksum or protocol drift."""

    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("array_sha256") != _sha256_file(path):
        raise ValueError("conditional-logit model checksum mismatch")
    expected = {
        "source_commit": SOURCE_COMMIT,
        "feature_dimension": FEATURE_DIMENSION,
        "candidate_count": N_CANDIDATES,
        "lambda": L2_LAMBDA,
        "max_iter": MAX_ITER,
        "gradient_tolerance": GRADIENT_TOLERANCE,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError("conditional-logit model protocol mismatch: {}".format(key))
    with np.load(path, allow_pickle=False) as arrays:
        fit = ConditionalLogitFit(
            weights=np.asarray(arrays["weights"], dtype=np.float64),
            scaler_mean=np.asarray(arrays["scaler_mean"], dtype=np.float64),
            scaler_scale=np.asarray(arrays["scaler_scale"], dtype=np.float64),
            train_case_indices=tuple(
                int(value) for value in np.asarray(arrays["train_case_indices"], dtype=np.int64)
            ),
            initial_loss=float(metadata["initial_loss"]),
            final_loss=float(metadata["final_loss"]),
            gradient_norm=float(metadata["gradient_norm"]),
            iterations=int(metadata["iterations"]),
            converged=bool(metadata["converged"]),
            message=str(metadata["message"]),
            source_commit=str(metadata["source_commit"]),
        )
    if any(array.shape != (FEATURE_DIMENSION,) for array in (
        fit.weights, fit.scaler_mean, fit.scaler_scale
    )):
        raise ValueError("conditional-logit model array shape mismatch")
    if not all(np.all(np.isfinite(array)) for array in (
        fit.weights, fit.scaler_mean, fit.scaler_scale
    )):
        raise ValueError("conditional-logit model contains non-finite values")
    return fit


def fit_conditional_logit(
    features: np.ndarray,
    root_indices: Sequence[int],
    train_indices: Optional[Sequence[int]] = None,
    *,
    l2_lambda: float = L2_LAMBDA,
    max_iter: int = MAX_ITER,
    gradient_tolerance: float = GRADIENT_TOLERANCE,
) -> ConditionalLogitFit:
    """Fit frozen event-level Conditional Logit on selected training cases.

    ``StandardScaler`` is fitted strictly on candidate rows from
    ``train_indices``.  Validation and test rows must only be passed to
    :meth:`ConditionalLogitFit.transform`/``scores`` after fitting.
    """

    values = _as_features(features)
    roots = _as_root_indices(root_indices, len(values))
    train = _indices(train_indices, len(values), "train_indices")
    train_values = values[train]
    scaler = StandardScaler().fit(train_values.reshape(-1, FEATURE_DIMENSION))
    matrices = [scaler.transform(row).astype(np.float64) for row in train_values]
    train_roots = roots[train]
    initial = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    initial_loss, _ = _event_loss_gradient(initial, matrices, train_roots, l2_lambda)

    def objective(weights):
        return _event_loss_gradient(weights, matrices, train_roots, l2_lambda)

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iter),
            "gtol": float(gradient_tolerance),
            "ftol": 1e-15,
            "maxls": 50,
        },
    )

    # Canonical deterministic Newton polish from frozen P4.  The loss slack is
    # bounded to eight ULPs to avoid accepting a numerically upward step.
    polished = np.asarray(result.x, dtype=np.float64).copy()
    for _ in range(32):
        polish_loss, polish_gradient = objective(polished)
        polish_norm = float(np.linalg.norm(polish_gradient, ord=np.inf))
        if polish_norm <= float(gradient_tolerance):
            break
        hessian = _event_hessian(polished, matrices, l2_lambda)
        try:
            step = np.linalg.solve(hessian, polish_gradient)
        except np.linalg.LinAlgError:
            break
        step_size = 1.0
        while step_size >= 2.0 ** -30:
            candidate = polished - step_size * step
            candidate_loss, candidate_gradient = objective(candidate)
            candidate_norm = float(np.linalg.norm(candidate_gradient, ord=np.inf))
            loss_slack = 8.0 * float(np.spacing(abs(polish_loss)))
            if candidate_loss < polish_loss or (
                candidate_loss <= polish_loss + loss_slack and candidate_norm < polish_norm
            ):
                polished = candidate
                break
            step_size *= 0.5
        else:
            break

    final_loss, final_gradient = objective(polished)
    final_norm = float(np.linalg.norm(final_gradient, ord=np.inf))
    return ConditionalLogitFit(
        weights=polished,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        train_case_indices=tuple(int(i) for i in train),
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        gradient_norm=final_norm,
        iterations=int(result.nit),
        converged=bool(result.success) and final_norm <= float(gradient_tolerance),
        message=str(result.message) + "; Newton polish gradient_inf={:.3e}".format(final_norm),
    )


# Explicit name for callers that prefer the protocol's train/validation/test
# vocabulary.
fit_train_conditional_logit = fit_conditional_logit


def rank_candidates(candidates: Sequence[str], scores: Sequence[float]) -> Tuple[str, ...]:
    names = tuple(str(x) for x in candidates)
    values = np.asarray(scores, dtype=np.float64)
    if len(names) != N_CANDIDATES or len(set(names)) != len(names):
        raise ValueError("candidates must contain ten unique services")
    if values.ndim != 1 or len(values) != len(names) or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite score for every candidate")
    order = sorted(range(len(names)), key=lambda i: (-float(values[i]), names[i]))
    return tuple(names[i] for i in order)


def predict_rankings(
    features: np.ndarray,
    model: ConditionalLogitFit,
    candidates: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[str, ...], ...]:
    values = _as_features(features)
    names = _candidate_names(candidates)
    scores = model.scores(values)
    return tuple(rank_candidates(names, row) for row in scores)


@dataclass(frozen=True)
class RootFrequencyBaseline:
    counts: Mapping[str, int]
    ranking: Tuple[str, ...]
    candidates: Tuple[str, ...]
    train_case_indices: Tuple[int, ...]

    def predict(self, n_cases: int) -> Tuple[Tuple[str, ...], ...]:
        if isinstance(n_cases, bool) or int(n_cases) != n_cases or int(n_cases) < 1:
            raise ValueError("n_cases must be a positive integer")
        return tuple(self.ranking for _ in range(int(n_cases)))


def fit_root_frequency(
    root_indices: Sequence[int],
    train_indices: Optional[Sequence[int]] = None,
    candidates: Optional[Sequence[str]] = None,
) -> RootFrequencyBaseline:
    """Fit the frequency shortcut on labelled fault services from Train only."""

    names = _candidate_names(candidates)
    roots = _as_root_indices(root_indices, len(root_indices))
    train = _indices(train_indices, len(roots), "train_indices")
    counts = {name: 0 for name in names}
    for index in roots[train]:
        counts[names[int(index)]] += 1
    ranking = tuple(sorted(names, key=lambda name: (-counts[name], name)))
    return RootFrequencyBaseline(counts, ranking, names, tuple(int(i) for i in train))


def root_frequency_rankings(
    root_indices: Sequence[int],
    train_indices: Optional[Sequence[int]] = None,
    candidates: Optional[Sequence[str]] = None,
    n_cases: Optional[int] = None,
) -> Tuple[Tuple[str, ...], ...]:
    baseline = fit_root_frequency(root_indices, train_indices, candidates)
    if n_cases is None:
        n_cases = len(root_indices)
    return baseline.predict(n_cases)


def _case_metrics(ranking: Sequence[str], root: str) -> Mapping[str, float]:
    ranking = tuple(ranking)
    rank = ranking.index(root) + 1
    return {
        "AC@1": float(rank <= 1),
        "AC@3": float(rank <= 3),
        "AC@5": float(rank <= 5),
        "Avg@5": sum(float(rank <= k) for k in range(1, 6)) / 5.0,
        "MRR": 1.0 / float(rank),
    }


def _aggregate(rows: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    if not rows:
        raise ValueError("cannot aggregate zero cases")
    keys = ("AC@1", "AC@3", "AC@5", "Avg@5", "MRR")
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def _stratified(
    case_rows: Sequence[Mapping[str, object]], field: str
) -> Mapping[str, object]:
    groups: Dict[str, list] = {}
    for row in case_rows:
        groups.setdefault(str(row[field]), []).append(row["metrics"])
    by_group = {
        group: {"case_count": len(rows), **_aggregate(rows)}
        for group, rows in sorted(groups.items())
    }
    metric_names = ("AC@1", "AC@3", "AC@5", "Avg@5", "MRR")
    macro = {
        metric: sum(float(row[metric]) for row in by_group.values()) / len(by_group)
        for metric in metric_names
    }
    return {"by_group": by_group, "group_count": len(by_group), "macro": macro}


def rca_metrics(
    rankings: Sequence[Sequence[str]],
    root_indices: Sequence[int],
    candidates: Optional[Sequence[str]] = None,
    fault_types: Optional[Sequence[str]] = None,
) -> Mapping[str, object]:
    """Compute overall, root-macro, and fault-macro ranking metrics.

    ``root_indices`` is the only root label consumed by model-independent
    metrics.  ``fault_types`` is evaluation metadata and is never used by the
    ranker; omitting it returns an empty fault stratum rather than inventing a
    fault label.
    """

    names = _candidate_names(candidates)
    roots = _as_root_indices(root_indices, len(root_indices))
    if len(rankings) != len(roots):
        raise ValueError("rankings and root_indices must align")
    if fault_types is not None and len(fault_types) != len(roots):
        raise ValueError("fault_types and root_indices must align")
    rows = []
    for i, ranking in enumerate(rankings):
        ranking = tuple(str(x) for x in ranking)
        if len(ranking) != N_CANDIDATES or set(ranking) != set(names):
            raise ValueError("each ranking must contain every candidate exactly once")
        root_name = names[int(roots[i])]
        rows.append({
            "root_service": root_name,
            "fault_type": "<none>" if fault_types is None else str(fault_types[i]),
            "metrics": _case_metrics(ranking, root_name),
        })
    overall = _aggregate([row["metrics"] for row in rows])
    fault = _stratified(rows, "fault_type") if fault_types is not None else {
        "by_group": {}, "group_count": 0,
        "macro": {key: float("nan") for key in ("AC@1", "AC@3", "AC@5", "Avg@5", "MRR")},
    }
    return {
        "overall": {"case_count": len(rows), **overall},
        "root_macro": _stratified(rows, "root_service"),
        "fault_macro": fault,
        "case_metrics": tuple(row["metrics"] for row in rows),
    }


# Readable aliases for orchestration code and tests.
compute_rca_metrics = rca_metrics
evaluate_rankings = rca_metrics
