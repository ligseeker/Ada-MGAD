"""Prevent another window in a batch from changing a causal detector output."""

import runpy
from pathlib import Path

import pytest
import torch

from src.e2e.window_dynamic_graph import WindowDynamicGraphLearner
from src.model_util import DynamicGraphLearner

HELPERS = runpy.run_path(str(Path(__file__).with_name("test_p6_c0_trigger_model.py")))


def model_for_scope(scope, batch_size=2):
    model = HELPERS["small_model"](batch_size=batch_size)
    model.graph_batch_scope = scope
    if scope == "window":
        old = model.dynamic_graph_learner
        learner = WindowDynamicGraphLearner(8, 4, hidden_dim=8, num_nodes=10,
                                           summary_mode=old.summary_mode)
        learner.load_state_dict(old.state_dict())
        model.dynamic_graph_learner = learner
    return model


def perturb_companion(batch):
    changed = {key: value.clone() for key, value in batch.items()}
    for value in changed.values():
        value[1] = value[1] * 20 + 30
    return changed


def test_legacy_batch_graph_reproduces_companion_dependency():
    torch.manual_seed(42)
    model = model_for_scope("batch").eval()
    batch = HELPERS["synthetic_batch"]()
    changed = perturb_companion(batch)
    with torch.no_grad():
        first = model(batch)[0][0]
        second = model(changed)[0][0]
    assert all(torch.equal(batch[key][0], changed[key][0]) for key in batch)
    assert abs(float(first - second)) > 1e-6


@pytest.mark.parametrize("summary", ["last", "mean"])
def test_window_graph_and_full_logit_ignore_companion(summary):
    torch.manual_seed(42)
    model = model_for_scope("window").eval()
    model.dynamic_graph_learner.summary_mode = summary
    batch = HELPERS["synthetic_batch"]()
    changed = perturb_companion(batch)
    with torch.no_grad():
        base = model(batch)[0]
        altered = model(changed)[0]
    torch.testing.assert_close(base[0], altered[0], rtol=0, atol=1e-7)
    assert abs(float(base[1] - altered[1])) > 1e-6


@pytest.mark.parametrize("summary", ["last", "mean"])
def test_single_window_matches_legacy_graph_and_regularizer(summary):
    torch.manual_seed(8)
    legacy = DynamicGraphLearner(8, 4, hidden_dim=8, num_nodes=10, summary_mode=summary).eval()
    fixed = WindowDynamicGraphLearner(8, 4, hidden_dim=8, num_nodes=10, summary_mode=summary).eval()
    fixed.load_state_dict(legacy.state_dict())
    node, log = torch.randn(1, 10, 10, 8), torch.randn(1, 10, 10, 4)
    static = torch.ones(10, 10) - torch.eye(10)
    a, ra = legacy(node, log, static)
    b, rb = fixed(node, log, static)
    torch.testing.assert_close(a, b[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(ra, rb, rtol=1e-6, atol=1e-7)


def test_corrected_output_matches_singleton_batch():
    torch.manual_seed(42)
    full = model_for_scope("window", 2).eval()
    singleton = model_for_scope("window", 1).eval()
    singleton.load_state_dict(full.state_dict())
    batch = HELPERS["synthetic_batch"]()
    with torch.no_grad():
        a = full(batch)[0][0]
        b = singleton({key: value[:1] for key, value in batch.items()})[0][0]
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_corrected_batch_32_is_invariant_to_companions_and_order():
    torch.manual_seed(42)
    model = model_for_scope("window", 32).eval()
    batch = HELPERS["synthetic_batch"](batch_size=32)
    changed = {key: value.clone() for key, value in batch.items()}
    for value in changed.values():
        value[1:] = value[1:] * 20 + 30
    permutation = torch.arange(31, -1, -1)
    with torch.no_grad():
        original = model(batch)[0]
        perturbed = model(changed)[0]
        reordered = model({key: value[permutation] for key, value in batch.items()})[0]
    torch.testing.assert_close(original[0], perturbed[0], rtol=0, atol=1e-7)
    torch.testing.assert_close(original, reordered[permutation], rtol=1e-6, atol=1e-7)


def test_window_model_backpropagates_finite_loss_with_unchanged_parameters():
    model = model_for_scope("window")
    legacy = model_for_scope("batch")
    assert model.state_dict().keys() == legacy.state_dict().keys()
    logits, regularizer = model(HELPERS["synthetic_batch"]())
    (logits.square().mean() + regularizer).backward()
    assert regularizer.ndim == 0 and torch.isfinite(regularizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
