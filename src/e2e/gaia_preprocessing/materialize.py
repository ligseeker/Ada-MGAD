"""Publication boundary for transformed GAIA Ada-MGAD inputs."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .schema import FrozenPreprocessingSchema


def validate_transformed_modalities(
    *,
    schema: FrozenPreprocessingSchema,
    metric: np.ndarray,
    logs: np.ndarray,
    trace: np.ndarray,
    service_count: int,
) -> Mapping[str, object]:
    """Validate shapes and finiteness before any completion manifest is written."""

    if not isinstance(schema, FrozenPreprocessingSchema):
        raise TypeError("schema must be FrozenPreprocessingSchema")
    if service_count < 1:
        raise ValueError("service_count must be positive")
    metric_values = np.asarray(metric)
    log_values = np.asarray(logs)
    trace_values = np.asarray(trace)
    if metric_values.ndim != 3 or log_values.ndim != 3 or trace_values.ndim != 4:
        raise ValueError("modalities must have Metric/Log/Trace ranks 3/3/4")
    dimensions = schema.dimensions
    expected_metric = (metric_values.shape[0], service_count, dimensions["raw_node"])
    expected_logs = (metric_values.shape[0], service_count, dimensions["log_len"])
    expected_trace = (
        metric_values.shape[0],
        service_count,
        service_count,
        dimensions["raw_edge"],
    )
    if metric_values.shape != expected_metric:
        raise ValueError(
            "metric shape {} does not match {}".format(metric_values.shape, expected_metric)
        )
    if log_values.shape != expected_logs:
        raise ValueError("log shape {} does not match {}".format(log_values.shape, expected_logs))
    if trace_values.shape != expected_trace:
        raise ValueError(
            "trace shape {} does not match {}".format(trace_values.shape, expected_trace)
        )
    for name, values in (
        ("metric", metric_values),
        ("logs", log_values),
        ("trace", trace_values),
    ):
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError("{} must be a finite numeric array".format(name))
    return {
        "schema_sha256": schema.sha256,
        "dimensions": dict(dimensions),
        "time_bins": int(metric_values.shape[0]),
        "service_count": int(service_count),
        "shapes": {
            "metric": list(metric_values.shape),
            "logs": list(log_values.shape),
            "trace": list(trace_values.shape),
        },
        "finite": True,
    }
