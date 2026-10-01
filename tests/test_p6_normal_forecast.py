import numpy as np
import copy
import json
from pathlib import Path
import pandas as pd
import pytest
import torch

from src.e2e.normal_forecast import (MetricForecastDataset, NormalMetricForecaster,
                                     NormalForecastState, event_free_windows)
from scripts.p6.analyze_c0_causal_development import verify_gt_identity, verify_target
from scripts.p6.run_normal_forecast_fit_screen import validate_config


def test_normal_windows_keep_crossing_events_and_half_open_edges():
    events = pd.DataFrame({"start_ms": [100, 600], "end_ms": [500, 600]})
    # [500,600) excludes an event ending at500 and an onset at600.
    # [501,601) includes the instantaneous event at600.
    assert event_free_windows([500, 600, 601, 701], events, 100).tolist() == [False, True, False, True]
    # A very old but still active event must not be hidden by later short ones.
    events = pd.DataFrame({"start_ms": [0, 200], "end_ms": [1000, 210]})
    assert not event_free_windows([900], events, 100)[0]


def test_dataset_never_hands_target_bin_or_target_mask_to_forecaster():
    class State:
        timestamps = np.arange(10) * 30000
        metric = np.ones((10, 10, 48), dtype=np.float32)
    State.metric[-1] = 999
    item = MetricForecastDataset(State(), [0])[0]
    assert item["history"].shape == (9, 10, 48)
    assert item["target"].shape == (10, 45)
    assert np.all(item["history"] == 1) and np.all(item["target"] == 999)


def test_forecast_has_no_temporal_or_peer_lookahead():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = NormalMetricForecaster().eval()
    history = torch.randn(2, 9, 10, 48)
    with torch.no_grad():
        encoded = model.encode(history)
        altered = history.clone()
        altered[:, 5:] += 100
        assert torch.allclose(encoded[:, :5], model.encode(altered)[:, :5], atol=1e-6)
        assert torch.allclose(model(history)[:1], model(history[:1]), atol=1e-6)
    with pytest.raises(ValueError):
        model(torch.randn(1, 10, 10, 48))


def test_target_missing_or_wrong_cannot_default_to_a_pass():
    with pytest.raises(ValueError):
        verify_target({}, "recent_onset60")
    with pytest.raises(ValueError):
        verify_target({"target": {"target": "one_bin_onset_frozen_ignore"}}, "recent_onset60")


def test_same_gt_ids_with_changed_annotations_are_rejected():
    gt = pd.DataFrame({"case_id": ["a"], "source_index": [1], "service": ["s"],
                       "fault_type": ["f"], "start_ms": [10], "end_ms": [20]})
    other = gt.copy()
    other.loc[0, "end_ms"] = 21
    with pytest.raises(ValueError):
        verify_gt_identity(other, gt)


def test_fit_state_excludes_boundary_targets_and_later_annotations(tmp_path):
    base = {"ad": {"grid_seconds": 30}, "split": {"absolute_start_ms": 0,
            "absolute_end_ms": 9000000, "boundary_ms": 6300000}}
    directory = tmp_path / "train"
    directory.mkdir()
    np.save(directory / "timestamps.npy", np.arange(210, dtype=np.int64) * 30000)
    np.save(directory / "metric.npy", np.ones((210, 10, 48), dtype=np.float32))
    registry = tmp_path / "registry.csv"
    pd.DataFrame({"case_id": ["fit-crossing", "later"], "source_index": [0, 1],
        "service": ["s", "must-not-parse"], "fault_type": ["memory_anomalies", "must-not-parse"],
        "start_ms": [2600000, 4600000], "end_ms": [3000000, 4700000],
        "detector_domain": [True, True]}).to_csv(registry, index=False)
    state = NormalForecastState(base, tmp_path, registry)
    assert state.legal_events.case_id.tolist() == ["fit-crossing"]
    assert np.all(state.windows.prediction_available_time < state.fit_end_ms)
    for block in state.blocks:
        indices = state.indices(block.name)
        assert np.all(state.timestamps[indices] >= block.start_ms)
        assert np.all(state.timestamps[indices + 9] + 30000 < block.end_ms)
    with pytest.raises(ValueError):
        state.indices("validation")
    with pytest.raises(ValueError):
        MetricForecastDataset(state, [140])  # its target is exactly original FitEnd


def test_frozen_training_and_calibration_parameters_cannot_silently_drift():
    path = Path(__file__).resolve().parents[1] / "configs/e2e/gaia_p6_normal_forecast_fit_screen_v1.json"
    config = json.loads(path.read_text())
    validate_config(config)
    for key, value in (("seed", 43), ("normal_quantile", .99)):
        altered = copy.deepcopy(config)
        altered[key] = value
        with pytest.raises(ValueError):
            validate_config(altered)
    altered = copy.deepcopy(config)
    altered["training"]["epochs"] = 10
    with pytest.raises(ValueError):
        validate_config(altered)
