"""One-bin onset supervision with the archived C0 loss mask preserved."""

import numpy as np

from .system_trigger import TRIGGER_IGNORE, TRIGGER_NEGATIVE, TRIGGER_POSITIVE, prediction_time_grid
from .trigger_development import TriggerDevelopmentState


def build_onset_bin_labels(prediction_times_ms, events, baseline_labels, grid_seconds=30):
    """Label an onset in [t-grid, t); never change the baseline IGNORE mask.

    An onset exactly at t is first supervised at t+grid, when its bin has
    completed. Multiple onsets in one bin remain one positive in this ablation.
    Existing recent-onset positives outside that bin become negatives. In
    particular, the 60-second loss-mask policy is not shortened to 30 seconds.
    """
    times = np.asarray(prediction_times_ms, dtype=np.int64)
    baseline = np.asarray(baseline_labels)
    if times.ndim != 1 or baseline.shape != times.shape:
        raise ValueError("onset labels require aligned one-dimensional arrays")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("prediction times must be strictly increasing")
    if int(grid_seconds) != 30:
        raise ValueError("the onset ablation freezes the 30-second grid")
    if not np.isin(baseline, [TRIGGER_NEGATIVE, TRIGGER_POSITIVE, TRIGGER_IGNORE]).all():
        raise ValueError("invalid baseline trigger label")
    starts = np.sort(events["start_ms"].to_numpy(dtype=np.int64), kind="stable")
    # Counts are audit metadata only; they are never used by inference.
    left = np.searchsorted(starts, times - int(grid_seconds) * 1000, side="left")
    right = np.searchsorted(starts, times, side="left")
    positive = right > left
    if np.any(positive & (baseline == TRIGGER_IGNORE)):
        raise ValueError("new onset overlaps the baseline IGNORE mask")
    labels = np.full(times.shape, TRIGGER_NEGATIVE, dtype=np.int8)
    labels[baseline == TRIGGER_IGNORE] = TRIGGER_IGNORE
    labels[positive] = TRIGGER_POSITIVE
    if np.any((labels == TRIGGER_POSITIVE) & (baseline != TRIGGER_POSITIVE)):
        raise ValueError("one-bin positives must be a subset of baseline positives")
    return labels


class OnsetDevelopmentState(TriggerDevelopmentState):
    """The same Fit/Validation windows and GT, with one-bin onset targets."""

    def __init__(self, base_config, data_root, registry_path):
        super().__init__(base_config, data_root, registry_path)
        self.baseline_labels = self.labels["train"].copy()
        times = prediction_time_grid(self.timestamps["train"])
        owned = times < int(self.blocks[1].end_ms)
        labels = np.full(times.shape, TRIGGER_IGNORE, dtype=np.int8)
        labels[owned] = build_onset_bin_labels(times[owned], self.legal_events,
                                              self.baseline_labels[owned], self.grid_seconds)
        self.labels["train"] = labels
