"""Raw observation survives finite forward fill; sealed transforms fail closed."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.e2e.gaia_preprocessing.metric import fit_metric_scaler
from src.e2e.gaia_preprocessing.raw import (MetricFile, MetricFit, MetricSlot,
                                           transform_metric_with_observability)
from src.e2e.gaia_preprocessing.materialize import (
    _raw_train_binding, materialize_ad_inputs, validate_fitted_schema_identity,
)
from src.e2e.protocol import GAIA_SERVICES


def test_raw_mask_does_not_count_forward_fill_or_change_numeric_values(tmp_path):
    path = tmp_path / "signal.csv"
    pd.DataFrame({"timestamp": [0, 120_000], "value": [0.0, 10.0]}).to_csv(path, index=False)
    records = (MetricFile(str(path), "system", "0.0.0.1", "signal", None, GAIA_SERVICES),)
    slots = tuple(MetricSlot("slot" + str(i), "global" if i < 30 else "host", "signal",
                             "value", "gauge", GAIA_SERVICES, ("signal",)) for i in range(45))
    scaler = fit_metric_scaler(np.array([0.0, 10.0]))
    fit = MetricFit(30_000, 0, 240_000, GAIA_SERVICES, slots,
                    {slot.name: scaler for slot in slots}, {"signal": records}, 60_000)
    legacy, old = transform_metric_with_observability(fit, tmp_path, 0, 240_000)
    fixed, new = transform_metric_with_observability(fit, tmp_path, 0, 240_000,
                                                    observation_mode="raw")
    assert fixed.shape == (8, 10, 48) and np.isfinite(fixed).all()
    np.testing.assert_array_equal(fixed[..., :45], legacy[..., :45])
    np.testing.assert_array_equal(new["host_applicable"], old["host_applicable"])
    np.testing.assert_array_equal(new["observed"][:, 0, 0],
                                  [True, False, False, False, True, False, False, False])
    np.testing.assert_array_equal(old["observed"][:, 0, 0],
                                  [True, True, True, False, True, True, True, False])
    assert new["global_observed_fraction"][0, 0, 0] == 1  # real normalized zero
    assert new["global_observed_fraction"][1, 0, 0] == 0  # finite fill is unobserved
    assert (new["global_observed_fraction"] <= old["global_observed_fraction"]).all()
    with pytest.raises(ValueError, match="observation_mode"):
        transform_metric_with_observability(fit, tmp_path, 0, 240_000, observation_mode="invalid")


@pytest.mark.parametrize("modality", ["metric", "logs", "traces"])
def test_same_dimension_changed_scaler_or_vocab_is_rejected(modality):
    payload = {"metric": {"ordered_slots": ["a"], "observation_mode": "filled",
                           "scalers": {"a": {"q01": 0, "q99": 10}}},
               "logs": {"ordered_slots": ["stable_template_1"], "drain_state_path": "old",
                        "drain_state": "sealed", "scalers": {"x": 10}},
               "traces": {"directed_edges": [["a", "b"]], "scalers": {"x": 10}}}
    equal = copy.deepcopy(payload)
    equal["logs"]["drain_state_path"] = "new"
    validate_fitted_schema_identity(payload, equal)
    changed = copy.deepcopy(equal)
    changed[modality]["scalers"] = {"x": 11}
    with pytest.raises(ValueError, match="refitted " + modality):
        validate_fitted_schema_identity(payload, changed)


def test_content_binding_detects_same_size_mutation(tmp_path):
    for name in ("metric/metric_split/metric", "business/business_split/business", "trace/trace_split/trace"):
        (tmp_path / name).mkdir(parents=True)
    path = tmp_path / "metric/metric_split/metric/signal.csv"
    path.write_text("a,1\n")
    legacy = _raw_train_binding(tmp_path, 0, 30_000)
    strict = _raw_train_binding(tmp_path, 0, 30_000, content_sha256=True)
    path.write_text("a,2\n")
    assert _raw_train_binding(tmp_path, 0, 30_000) == legacy
    assert _raw_train_binding(tmp_path, 0, 30_000, content_sha256=True) != strict


def test_existing_artifact_is_rejected_before_config_read_or_fit(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    state = artifacts / "drain3_train_state.json"
    state.write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="existing V2 output root"):
        materialize_ad_inputs(schema_path=tmp_path / "schema.json",
                              config_path=tmp_path / "absent-config.json",
                              raw_root=tmp_path / "absent-raw", data_root=tmp_path / "new-data",
                              artifact_root=artifacts, runtime={})
    assert state.read_text() == "do not overwrite"


def test_preprocess_cli_does_not_precreate_the_protected_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.p5 import run_i1_ad as driver
    artifact = tmp_path / "new-artifacts"
    config = {"preprocessing": {"cpu_budget": 2, "metric_workers": 1,
                                 "log_workers": 1, "trace_workers": 1}}
    args = SimpleNamespace(action="preprocess", config="fake.json", data_root=str(tmp_path / "data"),
                           artifact_root=str(artifact), preprocess_artifact_root=None,
                           checkpoint_dir=str(tmp_path / "checkpoints"), raw_root=str(tmp_path / "raw"),
                           workers=1, chunk_rows=10, start_method="spawn", metric_workers=1,
                           log_workers=1, trace_workers=1)
    calls = []
    monkeypatch.setattr(driver, "parse_args", lambda: args)
    monkeypatch.setattr(driver, "load_config", lambda path: config)
    monkeypatch.setattr(driver, "preprocessing_runtime",
                        lambda *a, **k: {"chunk_rows": 10, "workers": 1, "start_method": "spawn"})

    def fake_build(*positional, **kwargs):
        assert not Path(positional[3]).exists()
        calls.append(True)
        return {"status": "COMPLETE"}

    monkeypatch.setattr(driver, "build_ad_data", fake_build)
    driver.main()
    assert calls == [True]
