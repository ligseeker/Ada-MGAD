"""Timestamp-preserving Ada-MGAD-G arrays and lazy window datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Sampler

from .protocol import GAIA_SERVICES, TemporalBlock, sha256_file


ARRAY_NAMES = ("timestamps", "metric", "log", "trace", "labels", "label_mask")


def grid_for_block(block: TemporalBlock, grid_seconds: int) -> np.ndarray:
    """Return bin starts whose full 30-second bins are inside the block."""

    step = int(grid_seconds) * 1000
    first = ((int(block.start_ms) + step - 1) // step) * step
    last = (int(block.end_ms) // step) * step
    if last == block.end_ms:
        last -= step
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, step, dtype=np.int64)


def build_registry_node_labels(
    timestamps_ms: np.ndarray,
    registry: pd.DataFrame,
    services: Sequence[str] = GAIA_SERVICES,
    grid_seconds: int = 30,
) -> np.ndarray:
    """Rasterize only supported registry intervals onto the detector grid."""

    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    labels = np.zeros((len(timestamps), len(services)), dtype=np.int8)
    if timestamps.size == 0:
        return labels
    service_index = {service: index for index, service in enumerate(services)}
    step_ms = int(grid_seconds) * 1000
    for row in registry.itertuples(index=False):
        service = str(row.service)
        if service not in service_index:
            raise ValueError("registry service is outside the canonical service registry")
        first_bin = (int(row.start_ms) // step_ms) * step_ms
        last_bin = (int(row.end_ms) // step_ms) * step_ms
        left = int(np.searchsorted(timestamps, first_bin, side="left"))
        right = int(np.searchsorted(timestamps, last_bin, side="right"))
        if right > left:
            labels[left:right, service_index[service]] = 1
    return labels


def build_semisupervised_mask(
    labels: np.ndarray, label_percent: float, window_bins: int
) -> np.ndarray:
    """Preserve the original Ada-MGAD deterministic partial-label schedule."""

    binary = np.asarray(labels, dtype=np.int64)
    if binary.ndim != 2:
        raise ValueError("labels must be timestamp by service")
    masked = binary.copy()
    one_hot = np.eye(2, dtype=np.float32)[binary]
    counts = np.zeros((binary.shape[1], 2), dtype=np.float64)
    threshold = 10.0 * float(label_percent)
    for timestamp_index in range(binary.shape[0]):
        if timestamp_index < int(window_bins):
            continue
        counts += one_hot[timestamp_index]
        current_class = binary[timestamp_index]
        selected_counts = counts[np.arange(binary.shape[1]), current_class]
        unknown = np.mod(selected_counts, 10.0) >= threshold
        masked[timestamp_index, unknown] = 2
    return masked.astype(np.int8)


@dataclass(frozen=True)
class WindowMetadata:
    split: str
    window_start_time: int
    window_end_time: int
    prediction_timestamp: int
    service_names: Tuple[str, ...]
    node_labels: Tuple[int, ...]


class TimestampedArrayDataset(Dataset):
    """Lazy windows over split-local arrays; timestamps never enter the model."""

    def __init__(self, directory: Path, window_bins: int, grid_seconds: int):
        self.directory = Path(directory)
        self.split = self.directory.name
        self.window_bins = int(window_bins)
        self.grid_ms = int(grid_seconds) * 1000
        self.timestamps = np.load(self.directory / "timestamps.npy", mmap_mode="r")
        self.metric = np.load(self.directory / "metric.npy", mmap_mode="r")
        self.log = np.load(self.directory / "log.npy", mmap_mode="r")
        self.trace = np.load(self.directory / "trace.npy", mmap_mode="r")
        self.labels = np.load(self.directory / "labels.npy", mmap_mode="r")
        self.label_mask = np.load(self.directory / "label_mask.npy", mmap_mode="r")
        lengths = {len(array) for array in (
            self.timestamps, self.metric, self.log, self.trace, self.labels, self.label_mask
        )}
        if len(lengths) != 1:
            raise ValueError("timestamped Ada-MGAD arrays are not aligned")
        if self.metric.shape[1] != len(GAIA_SERVICES):
            raise ValueError("metric node order does not match canonical services")
        if self.log.shape[1] != len(GAIA_SERVICES):
            raise ValueError("log node order does not match canonical services")
        if self.trace.shape[1:3] != (len(GAIA_SERVICES), len(GAIA_SERVICES)):
            raise ValueError("trace node axes do not match canonical services")
        if len(self.timestamps) > 1 and not np.all(np.diff(self.timestamps) == self.grid_ms):
            raise ValueError("a split array must be a contiguous detector grid")

    def __len__(self) -> int:
        return max(0, len(self.timestamps) - self.window_bins + 1)

    def __getitem__(self, index: int) -> Mapping[str, np.ndarray]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        end = index + self.window_bins
        target = end - 1
        return {
            "data_node": np.asarray(self.metric[index:end], dtype=np.float32),
            "data_log": np.asarray(self.log[index:end], dtype=np.float32),
            "data_edge": np.asarray(self.trace[index:end], dtype=np.float32),
            "groundtruth_cls": np.eye(3, dtype=np.float32)[np.asarray(self.label_mask[target], dtype=np.int64)],
            "groundtruth_real": np.eye(2, dtype=np.float32)[np.asarray(self.labels[target], dtype=np.int64)],
            "sample_index": np.asarray(index, dtype=np.int64),
        }

    def metadata(self, index: int) -> WindowMetadata:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        end = index + self.window_bins
        target = end - 1
        return WindowMetadata(
            split=self.split,
            window_start_time=int(self.timestamps[index]),
            window_end_time=int(self.timestamps[target]) + self.grid_ms,
            prediction_timestamp=int(self.timestamps[target]),
            service_names=GAIA_SERVICES,
            node_labels=tuple(int(value) for value in self.labels[target]),
        )


class PaddedSequentialSampler(Sampler[int]):
    """Pad evaluation batches to the model's frozen fixed batch size."""

    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if len(dataset) == 0 or self.batch_size <= 0:
            raise ValueError("padded sampler requires positive dataset and batch size")

    def __iter__(self) -> Iterable[int]:
        size = len(self.dataset)
        padded = ((size + self.batch_size - 1) // self.batch_size) * self.batch_size
        for index in range(padded):
            yield index if index < size else size - 1

    def __len__(self) -> int:
        size = len(self.dataset)
        return ((size + self.batch_size - 1) // self.batch_size) * self.batch_size


def save_split_arrays(
    root: Path,
    split: str,
    arrays: Mapping[str, np.ndarray],
) -> Mapping[str, object]:
    missing = set(ARRAY_NAMES) - set(arrays)
    if missing:
        raise ValueError("missing split arrays: {}".format(sorted(missing)))
    directory = Path(root) / split
    directory.mkdir(parents=True, exist_ok=True)
    records: Dict[str, object] = {}
    expected_length = None
    for name in ARRAY_NAMES:
        values = np.asarray(arrays[name])
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError("split arrays must have identical timestamp length")
        path = directory / (name + ".npy")
        np.save(path, values, allow_pickle=False)
        records[name] = {
            "path": str(path.resolve()),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def load_timestamped_datasets(
    data_root: Path, window_bins: int, grid_seconds: int
) -> Mapping[str, TimestampedArrayDataset]:
    return {
        split: TimestampedArrayDataset(Path(data_root) / split, window_bins, grid_seconds)
        for split in ("train", "validation", "test")
    }
