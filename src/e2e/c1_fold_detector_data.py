"""Segment-local P6-C1 detector windows with a Generation label firewall."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
from torch.utils.data import Dataset

from .c1_fold_preprocessing import SEGMENTS, validate_c1_fold
from .protocol import GAIA_SERVICES
from .system_trigger import TRIGGER_LABEL_VALUES


class C1SegmentWindowDataset(Dataset):
    """Use only the legal windows in one sealed fold segment.

    Fit and Selection receive caller-built system trigger labels. Generation
    has no label parameter or label access path in returned samples/metadata.
    """

    def __init__(self, fold_root: Path, segment: str, *, protocol_path: Path,
                 trigger_labels: Optional[np.ndarray] = None):
        if segment not in SEGMENTS:
            raise ValueError("C1 segment must be fit, selection or generation")
        if segment == "generation" and trigger_labels is not None:
            raise ValueError("Generation input must remain label-free")
        if segment != "generation" and trigger_labels is None:
            raise ValueError("Fit/Selection require explicit system trigger labels")
        self.fold_root = Path(fold_root).resolve()
        self.segment = segment
        manifest = validate_c1_fold(self.fold_root, protocol_path=protocol_path)
        record = manifest["segments"][segment]
        self.grid_ms = (record["interval_ms"][1] - record["interval_ms"][0]) // record["time_bins"]
        self.window_bins = int(record["window_bins"])
        directory = self.fold_root / "ad_data" / segment
        self.timestamps = np.load(directory / "timestamps.npy", mmap_mode="r", allow_pickle=False)
        self.metric = np.load(directory / "metric.npy", mmap_mode="r", allow_pickle=False)
        self.log = np.load(directory / "log.npy", mmap_mode="r", allow_pickle=False)
        self.trace = np.load(directory / "trace.npy", mmap_mode="r", allow_pickle=False)
        self.sample_indices = np.load(directory / "legal_window_indices.npy", mmap_mode="r", allow_pickle=False)
        if (self.metric.shape != (len(self.timestamps), len(GAIA_SERVICES), 48)
                or self.log.shape != (len(self.timestamps), len(GAIA_SERVICES), 32)
                or self.trace.shape != (len(self.timestamps), len(GAIA_SERVICES), len(GAIA_SERVICES), 8)
                or not np.array_equal(self.sample_indices,
                                      np.arange(len(self.timestamps) - self.window_bins, dtype=np.int64))):
            raise ValueError("C1 detector arrays or legal-window indices drift")
        if trigger_labels is None:
            self._labels = None
        else:
            labels = np.asarray(trigger_labels, dtype=np.int64).reshape(-1)
            if len(labels) != len(self.timestamps) or not np.isin(labels, TRIGGER_LABEL_VALUES).all():
                raise ValueError("C1 trigger labels must cover the segment grid with frozen values")
            self._labels = labels.copy()

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _sample_index(self, position: int) -> int:
        if position < 0 or position >= len(self):
            raise IndexError(position)
        return int(self.sample_indices[position])

    def __getitem__(self, position: int) -> Mapping[str, np.ndarray]:
        index = self._sample_index(position)
        end = index + self.window_bins
        result = {
            "data_node": np.array(self.metric[index:end], dtype=np.float32),
            "data_log": np.array(self.log[index:end], dtype=np.float32),
            "data_edge": np.array(self.trace[index:end], dtype=np.float32),
            "sample_index": np.asarray(index, dtype=np.int64),
        }
        if self._labels is not None:
            result["trigger_label"] = np.asarray(self._labels[end - 1], dtype=np.int64)
        return result

    def prediction_times(self) -> np.ndarray:
        target = np.asarray(self.sample_indices) + self.window_bins - 1
        return np.asarray(self.timestamps[target], dtype=np.int64) + self.grid_ms

    def labels_at(self) -> np.ndarray:
        if self._labels is None:
            raise ValueError("Generation labels are unavailable before OOS episode lock")
        target = np.asarray(self.sample_indices) + self.window_bins - 1
        return np.asarray(self._labels[target], dtype=np.int64)
