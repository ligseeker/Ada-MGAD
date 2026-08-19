"""Deterministic paired bootstrap for root-service macro metric deltas."""

from typing import Mapping, Sequence

import numpy as np


def paired_root_macro_bootstrap(
    case_ids: Sequence[str],
    root_services: Sequence[str],
    resampling_units: Sequence[str],
    actual_metrics: Mapping[str, Sequence[float]],
    reference_metrics: Mapping[str, Sequence[float]],
    iterations: int = 10000,
    random_seed: int = 20260819,
    batch_size: int = 128,
):
    """Resample units and return root-macro paired-delta intervals."""

    case_ids = tuple(case_ids)
    roots = tuple(root_services)
    units = tuple(resampling_units)
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("bootstrap case IDs must be non-empty and unique")
    if len(roots) != len(case_ids) or len(units) != len(case_ids):
        raise ValueError("bootstrap roots and units must align with cases")
    if iterations <= 0 or batch_size <= 0:
        raise ValueError("bootstrap iterations and batch size must be positive")
    metric_names = tuple(sorted(actual_metrics))
    if not metric_names or set(metric_names) != set(reference_metrics):
        raise ValueError("paired bootstrap metric names must align")
    actual = np.column_stack(
        [np.asarray(actual_metrics[name], dtype=np.float64) for name in metric_names]
    )
    reference = np.column_stack(
        [np.asarray(reference_metrics[name], dtype=np.float64) for name in metric_names]
    )
    if actual.shape != (len(case_ids), len(metric_names)) or reference.shape != actual.shape:
        raise ValueError("paired bootstrap metric arrays must align")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("paired bootstrap metrics must be finite")

    unit_names = tuple(sorted(set(units)))
    service_names = tuple(sorted(set(roots)))
    unit_position = {value: index for index, value in enumerate(unit_names)}
    service_position = {value: index for index, value in enumerate(service_names)}
    unit_indices = np.asarray([unit_position[value] for value in units], dtype=np.int64)
    service_indices = np.asarray(
        [service_position[value] for value in roots], dtype=np.int64
    )
    deltas = actual - reference
    unit_sums = np.zeros(
        (len(unit_names), len(service_names), len(metric_names)), dtype=np.float64
    )
    unit_counts = np.zeros((len(unit_names), len(service_names)), dtype=np.int64)
    np.add.at(unit_sums, (unit_indices, service_indices), deltas)
    np.add.at(unit_counts, (unit_indices, service_indices), 1)

    service_sums = unit_sums.sum(axis=0)
    service_counts = unit_counts.sum(axis=0)
    if np.any(service_counts == 0):
        raise ValueError("every root service must have at least one case")
    point = (service_sums / service_counts[:, None]).mean(axis=0)

    rng = np.random.default_rng(int(random_seed))
    samples = np.empty((iterations, len(metric_names)), dtype=np.float64)
    probabilities = np.full(len(unit_names), 1.0 / len(unit_names))
    for start in range(0, iterations, batch_size):
        size = min(batch_size, iterations - start)
        weights = rng.multinomial(
            len(unit_names), probabilities, size=size
        ).astype(np.float64, copy=False)
        sums = np.einsum("bu,usm->bsm", weights, unit_sums, optimize=True)
        counts = weights @ unit_counts
        service_means = np.full_like(sums, np.nan)
        np.divide(
            sums,
            counts[:, :, None],
            out=service_means,
            where=counts[:, :, None] > 0,
        )
        samples[start : start + size] = np.nanmean(service_means, axis=1)
    result = {}
    for column, name in enumerate(metric_names):
        distribution = samples[:, column]
        lower, upper = np.quantile(distribution, (0.025, 0.975), method="linear")
        result[name] = {
            "bootstrap_mean_delta": float(distribution.mean()),
            "bootstrap_std": float(distribution.std(ddof=1)),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "point_delta": float(point[column]),
            "probability_delta_nonpositive": float((distribution <= 0.0).mean()),
        }
    return {
        "iterations": int(iterations),
        "metric_results": result,
        "random_generator": "numpy.default_rng/PCG64",
        "random_seed": int(random_seed),
        "resampling_unit_count": len(unit_names),
        "root_service_count": len(service_names),
    }
