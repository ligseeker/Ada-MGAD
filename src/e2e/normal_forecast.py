"""Fit-only normal-window prediction on the frozen filled Metric series."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .protocol import TemporalBlock, assign_event_blocks
from .system_trigger import trigger_temporal_blocks, window_split_assignment


FORECAST_SPEC = {"hidden_channels": 32, "kernel_size": 3, "dilations": [1, 2],
                 "convolutions_per_block": 2, "dropout": .2,
                 "history_bins": 9, "target_bins": 1, "numeric_slots": 45,
                 "mask_slots": 3, "services": 10}


def forecast_blocks(base_config):
    original = trigger_temporal_blocks(base_config)[0]
    count = (original.end_ms - original.start_ms) // 30000
    first = original.start_ms + (count * 6 // 10) * 30000
    second = original.start_ms + (count * 8 // 10) * 30000
    return (TemporalBlock("normal_training", original.start_ms, first),
            TemporalBlock("normal_calibration", first, second),
            TemporalBlock("event_holdout", second, original.end_ms))


def event_free_windows(times_ms, events, history_ms=300000):
    """Half-open windows must avoid every interval, including crossing events."""
    times = np.asarray(times_ms, dtype=np.int64)
    if events.empty:
        return np.ones(times.shape, dtype=bool)
    ordered = events.sort_values("start_ms", kind="stable")
    starts = ordered.start_ms.to_numpy(dtype=np.int64)
    ends = np.maximum(ordered.end_ms.to_numpy(dtype=np.int64), starts + 1)
    cumulative_ends = np.maximum.accumulate(ends)
    index = np.searchsorted(starts, times, side="left") - 1
    overlaps = np.zeros(times.shape, dtype=bool)
    valid = index >= 0
    overlaps[valid] = cumulative_ends[index[valid]] > times[valid] - int(history_ms)
    return ~overlaps


class NormalForecastState:
    """Only Fit annotations are parsed; only Train timestamps/Metric are mapped."""

    def __init__(self, base_config, data_root, registry_path):
        self.blocks = forecast_blocks(base_config)
        self.origin_ms = int(self.blocks[0].start_ms)
        self.fit_end_ms = int(self.blocks[-1].end_ms)
        route = pd.read_csv(registry_path, usecols=["start_ms", "end_ms", "detector_domain"])
        keep = set(np.flatnonzero(route.detector_domain.astype(bool)
                                 & (route.start_ms < self.fit_end_ms)
                                 & (route.end_ms >= self.origin_ms)).tolist())
        self.legal_events = pd.read_csv(registry_path,
            usecols=["case_id", "source_index", "service", "fault_type", "start_ms", "end_ms", "detector_domain"],
            skiprows=lambda row: row > 0 and row - 1 not in keep)
        if not self.legal_events.case_id.is_unique:
            raise ValueError("Fit event identity is not unique")
        self.assigned_events, self.purged_events = assign_event_blocks(self.legal_events, self.blocks)
        directory = Path(data_root) / "train"
        self.timestamps = np.load(directory / "timestamps.npy", mmap_mode="r")
        self.metric = np.load(directory / "metric.npy", mmap_mode="r")
        if (self.metric.shape != (len(self.timestamps), 10, 48)
                or not np.all(np.diff(self.timestamps) == 30000)):
            raise ValueError("frozen Train Metric geometry drift")
        # Geometry stops at Fit. The last prediction at fit_end belongs to
        # Validation in the original experiment and is never a screen sample.
        fit_timestamps = self.timestamps[self.timestamps < self.fit_end_ms]
        windows = window_split_assignment(fit_timestamps, self.blocks)
        windows = windows.loc[windows.prediction_available_time < self.fit_end_ms].copy()
        windows["normal"] = event_free_windows(windows.prediction_available_time, self.legal_events)
        self.windows = windows

    def indices(self, block, normal_only=False):
        if block not in {item.name for item in self.blocks}:
            raise ValueError("normal forecast excludes Validation/Test")
        rows = self.windows.loc[self.windows.split.eq(block) & self.windows.keep]
        if normal_only:
            rows = rows.loc[rows.normal]
        return rows.sample_index.to_numpy(dtype=np.int64)

    def gt_events(self, block):
        if block not in {item.name for item in self.blocks}:
            raise ValueError("normal forecast excludes Validation/Test")
        return self.assigned_events.loc[self.assigned_events.split.eq(block)].sort_values(
            ["start_ms", "source_index", "case_id"], kind="stable").reset_index(drop=True)

    def clean_memory_ids(self):
        memory = self.gt_events("event_holdout")
        memory = memory.loc[memory.fault_type.eq("memory_anomalies")]
        clean = []
        available = set(self.timestamps[self.indices("event_holdout") + 9] + 30000)
        for row in memory.itertuples():
            first_t = self.origin_ms + ((int(row.start_ms) - self.origin_ms) // 30000 + 1) * 30000
            others = self.legal_events.loc[self.legal_events.case_id.ne(row.case_id)]
            if first_t in available and event_free_windows([first_t], others)[0]:
                clean.append(row.case_id)
        return sorted(clean)


class MetricForecastDataset(Dataset):
    def __init__(self, state, indices):
        self.state = state
        self.sample_indices = np.asarray(indices, dtype=np.int64)
        if self.sample_indices.size and (self.sample_indices.min() < 0
                or self.sample_indices.max() + 10 > len(state.timestamps)):
            raise ValueError("forecast window outside Train")
        if hasattr(state, "fit_end_ms") and self.sample_indices.size:
            times = state.timestamps[self.sample_indices + 9] + 30000
            legal_indices = state.windows.loc[state.windows.keep, "sample_index"].to_numpy()
            if np.any(times >= state.fit_end_ms) or not np.isin(self.sample_indices, legal_indices).all():
                raise ValueError("forecast sample crosses a Fit screen boundary")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, position):
        index = int(self.sample_indices[position])
        return {"history": np.array(self.state.metric[index:index + 9], dtype=np.float32),
                "target": np.array(self.state.metric[index + 9, :, :45], dtype=np.float32),
                "sample_index": np.int64(index)}


class ForecastResidualBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.padding = 2 * int(dilation)
        self.convs = nn.ModuleList([nn.Conv1d(32, 32, 3, dilation=dilation) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(32) for _ in range(2)])
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        value = x
        for conv, norm in zip(self.convs, self.norms):
            value = conv(F.pad(value, (self.padding, 0)))
            value = self.dropout(F.relu(norm(value.transpose(1, 2)).transpose(1, 2)))
        return x + value


class NormalMetricForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(48, 32, 1)
        self.blocks = nn.Sequential(ForecastResidualBlock(1), ForecastResidualBlock(2))
        self.output = nn.Conv1d(32, 45, 1)

    def encode(self, history):
        if history.ndim != 4 or history.shape[1:] != (9, 10, 48):
            raise ValueError("forecast model accepts only 9 historical bins")
        batch = history.shape[0]
        value = history.permute(0, 2, 3, 1).reshape(batch * 10, 48, 9)
        value = self.output(self.blocks(self.input(value)))
        return value.reshape(batch, 10, 45, 9).permute(0, 3, 1, 2)

    def forward(self, history):
        return self.encode(history)[:, -1]


def residual_score(prediction, target):
    return (prediction - target).abs().mean(dim=-1).max(dim=-1).values
