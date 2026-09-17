"""Read-only window dataset for the P6-C0 system event trigger.

The dataset re-slices the frozen V2 Ada-MGAD arrays.  It returns **only** the
three multimodal inputs, the underlying sample index and the system-level
trigger label.  Node anomaly labels (`labels.npy` / `label_mask.npy`) and the
labelled root service are never read, so no service-level supervision can leak
into the detector through the data path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from torch.utils.data import DataLoader, Dataset

from .ad_data import PaddedSequentialSampler
from .protocol import GAIA_SERVICES
from .system_trigger import SPLIT_NAMES, TRIGGER_LABEL_VALUES


@dataclass(frozen=True)
class TriggerWindowMetadata:
    split: str
    sample_index: int
    window_start_time: int
    window_end_time: int
    target_bin_start: int
    target_bin_end: int
    prediction_available_time: int
    trigger_label: int


class TriggerWindowDataset(Dataset):
    """Chronological sliding windows restricted to an explicit sample index list."""

    def __init__(
        self,
        directory: Path,
        window_bins: int,
        grid_seconds: int,
        sample_indices: Sequence[int],
        trigger_labels: np.ndarray,
        split_name: str,
    ):
        self.directory = Path(directory)
        if str(split_name) not in SPLIT_NAMES:
            raise ValueError("trigger dataset split must be fit/validation/test")
        self.split = str(split_name)
        self.window_bins = int(window_bins)
        self.grid_ms = int(grid_seconds) * 1000
        if self.window_bins < 1 or self.grid_ms < 1:
            raise ValueError("window geometry must be positive")
        self.timestamps = np.load(self.directory / "timestamps.npy", mmap_mode="r")
        self.metric = np.load(self.directory / "metric.npy", mmap_mode="r")
        self.log = np.load(self.directory / "log.npy", mmap_mode="r")
        self.trace = np.load(self.directory / "trace.npy", mmap_mode="r")
        lengths = {len(self.timestamps), len(self.metric), len(self.log), len(self.trace)}
        if len(lengths) != 1:
            raise ValueError("frozen trigger input arrays are not aligned")
        if len(self.timestamps) > 1 and not np.all(np.diff(self.timestamps) == self.grid_ms):
            raise ValueError("a frozen split array must be a contiguous detector grid")
        if self.metric.shape[1] != len(GAIA_SERVICES) or self.log.shape[1] != len(GAIA_SERVICES):
            raise ValueError("frozen trigger inputs do not follow the canonical service order")
        if self.trace.shape[1:3] != (len(GAIA_SERVICES), len(GAIA_SERVICES)):
            raise ValueError("frozen trace tensor does not follow the canonical service order")
        self.trigger_labels = np.asarray(trigger_labels, dtype=np.int64).reshape(-1)
        if len(self.trigger_labels) != len(self.timestamps):
            raise ValueError("trigger labels must be defined on the whole split timeline")
        if not np.isin(self.trigger_labels, TRIGGER_LABEL_VALUES).all():
            raise ValueError("trigger labels must be NEGATIVE/POSITIVE/IGNORE")
        indices = np.asarray(list(sample_indices), dtype=np.int64)
        if indices.size and (indices.min() < 0 or indices.max() + self.window_bins > len(self.timestamps)):
            raise ValueError("sample index list leaves the frozen split timeline")
        self.sample_indices = indices

    def __len__(self) -> int:
        return int(len(self.sample_indices))

    def _target_index(self, position: int) -> int:
        if position < 0 or position >= len(self):
            raise IndexError(position)
        return int(self.sample_indices[position]) + self.window_bins - 1

    def __getitem__(self, position: int) -> Mapping[str, np.ndarray]:
        index = int(self.sample_indices[position])
        end = index + self.window_bins
        target = self._target_index(position)
        # ``np.array`` copies out of the read-only memory map so collation never
        # hands a non-writable buffer to the model.
        return {
            "data_node": np.array(self.metric[index:end], dtype=np.float32),
            "data_log": np.array(self.log[index:end], dtype=np.float32),
            "data_edge": np.array(self.trace[index:end], dtype=np.float32),
            "sample_index": np.asarray(index, dtype=np.int64),
            "trigger_label": np.asarray(self.trigger_labels[target], dtype=np.int64),
        }

    def metadata(self, position: int) -> TriggerWindowMetadata:
        index = int(self.sample_indices[position])
        target = self._target_index(position)
        start = int(self.timestamps[index])
        prediction_time = int(self.timestamps[target]) + self.grid_ms
        return TriggerWindowMetadata(
            split=self.split,
            sample_index=index,
            window_start_time=start,
            window_end_time=prediction_time,
            target_bin_start=int(self.timestamps[target]),
            target_bin_end=prediction_time,
            prediction_available_time=prediction_time,
            trigger_label=int(self.trigger_labels[target]),
        )

    def prediction_times(self, positions: Sequence[int] = None) -> np.ndarray:
        positions = range(len(self)) if positions is None else positions
        return np.asarray([
            int(self.timestamps[int(self.sample_indices[position]) + self.window_bins - 1]) + self.grid_ms
            for position in positions
        ], dtype=np.int64)

    def labels_at(self, positions: Sequence[int] = None) -> np.ndarray:
        positions = range(len(self)) if positions is None else positions
        return np.asarray([
            int(self.trigger_labels[self._target_index(position)]) for position in positions
        ], dtype=np.int64)


def build_trigger_loader(
    dataset: TriggerWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    """Loader with the frozen padded-sampler geometry required by the encoder."""

    if shuffle:
        raise ValueError("P6-C0 loaders are deterministic; shuffling is not allowed")
    if len(dataset) == 0:
        raise ValueError("cannot build a trigger loader over an empty dataset")
    batch_size = int(batch_size)
    sampler = PaddedSequentialSampler(dataset, batch_size)
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
