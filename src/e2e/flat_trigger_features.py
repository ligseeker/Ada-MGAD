"""Lossless, row-local flattening of the existing causal detector inputs."""

import numpy as np


def flatten_trigger_windows(metric, logs, trace, sample_indices, window_bins=10):
    indices = np.asarray(sample_indices, dtype=np.int64)
    if indices.ndim != 1 or (indices.size and
                            (indices.min() < 0 or indices.max() + window_bins > len(metric))):
        raise ValueError("window indices leave the input timeline")
    if not len(metric) == len(logs) == len(trace):
        raise ValueError("modality timelines differ")
    positions = indices[:, None] + np.arange(window_bins, dtype=np.int64)[None, :]
    parts = [np.asarray(array[positions], dtype=np.float32).reshape(len(indices), -1)
             for array in (metric, logs, trace)]
    return np.concatenate(parts, axis=1)
