"""Detector development using Train arrays and Fit/Validation GT only."""

from pathlib import Path

import numpy as np
import pandas as pd

from .system_trigger import (assign_legal_events, build_trigger_labels,
                             prediction_time_grid, trigger_temporal_blocks,
                             window_split_assignment, TRIGGER_IGNORE)
from .system_trigger_data import TriggerWindowDataset


class TriggerDevelopmentState:
    """Retain the frozen C0 geometry without opening any Test array."""

    def __init__(self, base_config, data_root, registry_path):
        self.base_config = base_config
        self.data_root = Path(data_root)
        self.registry_path = Path(registry_path)
        self.grid_seconds, self.window_bins = 30, 10
        self.history_seconds, self.tolerance_seconds = 300, 60
        self.blocks = trigger_temporal_blocks(base_config)
        boundary = int(self.blocks[1].end_ms)
        # Inspect interval columns solely to route rows by the frozen split.
        # Skip Test rows before parsing their service/fault annotation fields.
        intervals = pd.read_csv(self.registry_path,
                                usecols=["start_ms", "end_ms", "detector_domain"])
        keep = set(np.flatnonzero(intervals["detector_domain"].astype(bool)
                                  & (intervals["start_ms"] < boundary)).tolist())
        self.registry = pd.read_csv(
            self.registry_path,
            usecols=["case_id", "source_index", "service", "fault_type",
                     "start_ms", "end_ms", "detector_domain"],
            skiprows=lambda row: row > 0 and row - 1 not in keep,
        )
        self.legal_events, self.assigned_events, self.purged_events = assign_legal_events(
            self.registry, self.blocks)
        if not set(self.assigned_events["split"]).issubset({"fit", "validation"}):
            raise ValueError("development registry contains a Test metric case")
        timestamps = np.load(self.data_root / "train" / "timestamps.npy", mmap_mode="r")
        self.timestamps = {"train": np.asarray(timestamps, dtype=np.int64)}
        grid = prediction_time_grid(self.timestamps["train"])
        # The final Train-array bin predicts exactly at the Test boundary.
        # It is purged by window assignment and must not be rasterized as a
        # Test-owned target. Retain an unconsumed sentinel for array alignment.
        owned = grid < boundary
        labels = np.full(grid.shape, TRIGGER_IGNORE, dtype=np.int8)
        labels[owned] = build_trigger_labels(grid[owned], self.legal_events)
        self.labels = {"train": labels}
        self.assignments = {"train": window_split_assignment(self.timestamps["train"], self.blocks)}

    def gt_events(self, split):
        if split not in ("fit", "validation"):
            raise ValueError("development state excludes Test")
        return self.assigned_events.loc[self.assigned_events["split"].eq(split)].sort_values(
            ["start_ms", "source_index", "case_id"], kind="stable").reset_index(drop=True)

    def sample_indices(self, split):
        if split not in ("fit", "validation"):
            raise ValueError("development state excludes Test")
        frame = self.assignments["train"]
        return np.sort(frame.loc[frame["split"].eq(split) & frame["keep"],
                                 "sample_index"].to_numpy(dtype=np.int64))

    def build_dataset(self, split):
        return TriggerWindowDataset(self.data_root / "train", self.window_bins,
                                    self.grid_seconds, self.sample_indices(split),
                                    self.labels["train"], split)
