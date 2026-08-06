"""Shared GAIA window/label IO for RCA baseline adapters (pure numpy/pandas).

The baseline adapters (baselines/diagfusion-main/build_gaia_data.py,
baselines/Eadro/build_gaia_chunks.py) must stay free of torch/dgl imports,
so the window-reconstruction and RCA-label logic that util/eval_rca.py
embeds is factored out here. util/eval_rca.py keeps its own copies to avoid
refactoring risk on the frozen evaluation path.
"""

import os

import numpy as np
import pandas as pd

GAIA_WINDOW = 10          # window size in 30s timesteps
SPLIT_RATIO = 0.7         # chronological train fraction (same as train_rca/eval_rca)
STEP_MS = 30 * 1000
GAP_THRESHOLD_MS = STEP_MS * 3  # same segmentation rule as util/GAIA/data_GAIA.py


def reconstruct_end_timestamps(data_path, window=GAIA_WINDOW):
    """End timestamp of every dataset window, in chronological order.

    Mirrors the segmentation in util/GAIA/data_GAIA.py: windows are built
    per continuous segment (gap > GAP_THRESHOLD_MS) with stride 1, and a
    window's label is taken at its last timestep.
    """
    grid = pd.read_csv(os.path.join(data_path, 'label.csv'),
                       usecols=['timestamp'])['timestamp'].values.astype(np.int64)
    grid.sort()
    segments = []
    seg_start = 0
    for i, d in enumerate(np.diff(grid)):
        if d > GAP_THRESHOLD_MS:
            segments.append((seg_start, i + 1))
            seg_start = i + 1
    segments.append((seg_start, len(grid)))

    end_ts = []
    for seg_start, seg_end in segments:
        for i in range(seg_end - seg_start - window + 1):
            end_ts.append(grid[seg_start + i + window - 1])
    return np.asarray(end_ts, dtype=np.int64)


def load_rca_labels(data_path):
    """timestamp -> (root_cause_id, fault_type_str) from label_rca.csv."""
    frame = pd.read_csv(os.path.join(data_path, 'label_rca.csv'))
    return {int(row.timestamp): (int(row.root_cause_id), str(row.fault_type))
            for row in frame.itertuples(index=False)}


def window_span(end_ts, window=GAIA_WINDOW):
    """(start, end) inclusive timestamps covered by each window."""
    return end_ts - (window - 1) * STEP_MS, end_ts


def is_single_root(type_str):
    """True if the fault_type string of a window has no multi-root separator."""
    return '|' not in type_str


def primary_type(type_str):
    """First fault type of the string, brackets stripped."""
    return type_str.split('|')[0].strip('[]')


def primary_node(labels_row):
    """root_cause_id (primary node) of a label_rca row."""
    return labels_row[0]
