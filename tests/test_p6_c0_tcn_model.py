"""Causal convolution and preservation of the existing non-encoder modules."""

import numpy as np
import torch

from src.e2e.system_trigger_model import SystemEventTrigger
from src.e2e.system_trigger_tcn import GraphCausalTCNEncoder, WindowCausalTCNTrigger


def model_args():
    return {"gpu": False, "num_nodes": 10, "raw_node": 48, "feature_node": 16,
            "log_len": 32, "feature_log": 8, "raw_edge": 8, "feature_edge": 4,
            "num_heads_node": 4, "num_heads_log": 4, "num_heads_edge": 4,
            "num_heads_n2e": 4, "num_heads_e2n": 2, "dropout": .2, "batch_size": 2,
            "window": 10, "num_layer": 2, "graph_batch_scope": "window"}


def test_encoder_never_reads_future_steps():
    torch.manual_seed(42)
    encoder = GraphCausalTCNEncoder(torch.ones(10, 10), 16, 8, 4).eval()
    node = torch.randn(2, 10, 10, 16)
    log = torch.randn(2, 10, 10, 8)
    edge = torch.randn(2, 10, 10, 10, 4)
    reference = encoder(node, edge, log)[0]
    changed = [x.clone() for x in (node, edge, log)]
    for x in changed:
        x[:, 5:] += 100
    torch.testing.assert_close(reference[:, :5], encoder(*changed)[0][:, :5], rtol=0, atol=0)


def test_same_nonencoder_initialization_loss_and_batch_independence():
    graph = np.ones((10, 10), dtype=np.float32)
    torch.manual_seed(42)
    original = SystemEventTrigger(graph, **model_args()).eval()
    torch.manual_seed(42)
    candidate = WindowCausalTCNTrigger(graph, **model_args()).eval()
    for key, value in original.state_dict().items():
        if not key.startswith("encoder."):
            torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    batch = {"data_node": torch.randn(2, 10, 10, 48), "data_log": torch.randn(2, 10, 10, 32),
             "data_edge": torch.randn(2, 10, 10, 10, 8)}
    with torch.no_grad():
        logits, reg = candidate(batch)
        assert logits.shape == (2,) and reg.ndim == 0
        torch.testing.assert_close(reg, original(batch)[1], rtol=0, atol=0)
        changed = {key: value.clone() for key, value in batch.items()}
        for value in changed.values():
            value[1] += 100
        torch.testing.assert_close(logits[0], candidate(changed)[0][0], rtol=0, atol=1e-5)
        torch.testing.assert_close(logits[:1], candidate({key: value[:1] for key, value in batch.items()})[0],
                                   rtol=0, atol=1e-5)
