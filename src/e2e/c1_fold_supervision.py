"""Fit/Selection supervision for P6-C1, outside Generation scoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .c1_fold_preprocessing import ROOT, validate_c1_fold
from .protocol import load_registry
from .system_trigger import (
    assert_prediction_time_grid, build_trigger_labels, prediction_time_grid,
)


@dataclass(frozen=True)
class C1FoldSupervision:
    fit_labels: np.ndarray
    selection_labels: np.ndarray
    selection_gt: pd.DataFrame
    audit: Mapping[str, object]


def _build_from_registry(fold_root: Path, manifest: Mapping[str, object],
                         registry: pd.DataFrame) -> C1FoldSupervision:
    """Build labels only for segments allowed to supervise detector decisions."""
    if "detector_domain" not in registry:
        raise ValueError("C1 registry lacks detector_domain")
    legal = registry.loc[registry["detector_domain"].astype(bool)].copy()
    labels = {}
    counts = {}
    for name in ("fit", "selection"):
        start, end = manifest["segments"][name]["interval_ms"]
        timestamps = np.load(Path(fold_root) / "ad_data" / name / "timestamps.npy",
                             mmap_mode="r", allow_pickle=False)
        prediction_times = prediction_time_grid(timestamps, grid_seconds=30)
        assert_prediction_time_grid(prediction_times, timestamps, grid_seconds=30)
        # Only events already begun before this segment ends can affect its
        # trigger state. Generation onsets never enter Fit/Selection labels.
        overlapping = legal.loc[(legal["start_ms"] < end) & (legal["end_ms"] > start)]
        labels[name] = build_trigger_labels(
            prediction_times, overlapping[["start_ms", "end_ms"]],
            positive_window_seconds=60)
        counts[name] = {"label_source_events": int(len(overlapping)),
                        "positive_bins": int(np.sum(labels[name] == 1)),
                        "ignore_bins": int(np.sum(labels[name] == 2))}
    selection_start, selection_end = manifest["segments"]["selection"]["interval_ms"]
    selection_gt = legal.loc[(legal["start_ms"] >= selection_start)
                             & (legal["end_ms"] <= selection_end)].copy()
    selection_gt = selection_gt.sort_values(
        ["start_ms", "source_index", "case_id"], kind="stable").reset_index(drop=True)
    if selection_gt.empty:
        raise ValueError("C1 Selection has no complete GT event for threshold choice")
    crossing = legal.loc[(legal["start_ms"] < selection_end)
                         & (legal["end_ms"] > selection_start)
                         & ~((legal["start_ms"] >= selection_start)
                             & (legal["end_ms"] <= selection_end))]
    audit = {"fold": manifest["fold"], "fit": counts["fit"],
             "selection": counts["selection"],
             "selection_complete_gt": int(len(selection_gt)),
             "selection_cross_boundary_excluded_from_metrics": int(len(crossing)),
             "generation_labels_built": False}
    return C1FoldSupervision(labels["fit"], labels["selection"], selection_gt, audit)


def build_c1_fit_selection_supervision(fold_root: Path, *, protocol_path: Path) -> C1FoldSupervision:
    """Load the frozen registry, never deriving or returning Generation labels."""
    manifest = validate_c1_fold(fold_root, protocol_path=protocol_path)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    base_path = ROOT / protocol["bindings"]["base_config"]["path"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    registry = load_registry(base, ROOT)
    return _build_from_registry(Path(fold_root), manifest, registry)
