"""Keep crossing-event state while never opening Test arrays/annotations."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.e2e.system_trigger import build_trigger_labels, prediction_time_grid
from src.e2e.trigger_development import TriggerDevelopmentState


def test_development_loader_preserves_state_across_boundaries(tmp_path, monkeypatch):
    # The complete timeline is 100 bins, with Fit/Validation ending at 50/70.
    config = {"split": {"absolute_start_ms": 0, "absolute_end_ms": 3_000_000,
                        "boundary_ms": 2_100_000}, "ad": {"grid_seconds": 30}}
    train = tmp_path / "ad" / "train"
    train.mkdir(parents=True)
    timestamps = np.arange(70, dtype=np.int64) * 30_000
    np.save(train / "timestamps.npy", timestamps)
    for name, shape in (("metric", (70, 10, 48)), ("log", (70, 10, 32)),
                        ("trace", (70, 10, 10, 8))):
        np.save(train / (name + ".npy"), np.zeros(shape, dtype=np.float32))
    rows = [
        ("fit", 0, "mobservice1", "login_failure", 900_000, 910_000, True),
        ("cross-fit-val", 1, "mobservice1", "memory", 1_470_000, 1_800_000, True),
        ("validation", 2, "mobservice2", "login_failure", 1_950_000, 1_960_000, True),
        ("cross-val-test", 3, "mobservice2", "memory", 2_040_000, 2_190_000, True),
        ("test", 4, "FORBIDDEN_TEST_ROOT", "FORBIDDEN_TEST_FAULT", 2_400_000, 2_410_000, True),
        ("out-of-domain", 5, "FORBIDDEN_ROOT", "FORBIDDEN_FAULT", 800_000, 810_000, False),
    ]
    registry = pd.DataFrame(rows, columns=["case_id", "source_index", "service", "fault_type",
                                           "start_ms", "end_ms", "detector_domain"])
    registry_path = tmp_path / "registry.csv"
    registry.to_csv(registry_path, index=False)
    original_load, original_read = np.load, pd.read_csv
    opened, parsed = [], []

    def guarded_load(path, *args, **kwargs):
        opened.append(Path(path))
        assert "test" not in Path(path).parts
        assert Path(path).name not in ("labels.npy", "label_mask.npy")
        return original_load(path, *args, **kwargs)

    def guarded_read(*args, **kwargs):
        frame = original_read(*args, **kwargs)
        parsed.append(frame.copy())
        assert not any(frame.astype(str).eq("FORBIDDEN_TEST_ROOT").any())
        assert not any(frame.astype(str).eq("FORBIDDEN_TEST_FAULT").any())
        return frame

    monkeypatch.setattr(np, "load", guarded_load)
    monkeypatch.setattr(pd, "read_csv", guarded_read)
    state = TriggerDevelopmentState(config, train.parent, registry_path)
    expected = build_trigger_labels(prediction_time_grid(timestamps), registry.loc[registry.detector_domain])
    np.testing.assert_array_equal(state.labels["train"], expected)
    assert set(state.gt_events("fit").case_id) == {"fit"}
    assert set(state.gt_events("validation").case_id) == {"validation"}
    assert set(state.purged_events.case_id) == {"cross-fit-val", "cross-val-test"}
    assert len(state.build_dataset("fit")) == 40
    assert len(state.build_dataset("validation")) == 10
    assert len(parsed) == 2 and len(opened) == 9
    for method in (state.gt_events, state.sample_indices, state.build_dataset):
        with pytest.raises(ValueError, match="excludes Test"):
            method("test")
