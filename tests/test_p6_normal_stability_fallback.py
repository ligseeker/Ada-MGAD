import copy
import json
from pathlib import Path

import pytest

from scripts.p6.normal_forecast_stability_fallback import (validate_fallback_config,
                                                          verify_forecast_kernel)
from scripts.p6.run_normal_forecast_fit_screen import validate_config


def test_fallback_cannot_change_calibration_training_or_fit_screen():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/e2e/gaia_p6_normal_forecast_stability_v2.json").read_text())
    validate_config(config)
    for key, value in (("normal_quantile", .99), ("seed", 17)):
        altered = copy.deepcopy(config)
        altered[key] = value
        with pytest.raises(ValueError):
            validate_fallback_config(altered)
    for key in ("training", "screen_gate", "model_spec"):
        altered = copy.deepcopy(config)
        altered[key]["unregistered_change"] = True
        with pytest.raises(ValueError):
            validate_fallback_config(altered)


def test_forecast_train_score_and_metric_kernel_exactly_preserved():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/e2e/gaia_p6_normal_forecast_stability_v2.json").read_text())
    verify_forecast_kernel(root / "scripts/p6/run_normal_forecast_fit_screen.py",
                           config["stability_fallback"]["reference_runner"])
