"""P6-C0 root-agnostic system event trigger model.

The detector answers exactly one question: *is the system in the first 60 s of a
fault event right now?*  It emits one scalar system logit per prediction
timestamp — no root service, no service ranking, no ten-way service output.

Design constraints (P6-C0 protocol):

* random initialisation; no Ada-MGAD checkpoint is ever loaded,
* no node-classification loss, no node anomaly label, no root-service label,
* no label-conditioned reconstruction objective and no reconstruction branch,
* no fusion of node anomaly scores,
* permutation-invariant pooling over the service axis before the head.

The multimodal encoder and the dynamic graph learner are reused from
``src.model_util`` unchanged; only the head is new.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from src.model_util import DynamicGraphLearner, Embed, Encoder

from .protocol import GAIA_SERVICES


DEFAULT_HEAD_HIDDEN = 32


def pool_service_representations(h: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant pooling over the service axis.

    ``h`` is ``(batch, num_services, feature_node)``.  ``mean``, ``max`` and
    ``std`` are each invariant to any permutation of the service axis, so the
    pooled system representation — and therefore the trigger logit — is invariant
    too.

    Scope note: the graph encoder upstream still consumes the canonical GAIA
    service order, because the static/fused trace graph is defined by it.  The
    invariance guaranteed here is exactly the interface-level invariance the
    P6-C0 protocol requires (no fixed service position is exposed to the head);
    it is not a claim that the whole graph network is a permutation-invariant
    function of the raw node order.
    """

    if h.dim() != 3:
        raise ValueError("service representations must be (batch, services, feature)")
    if h.shape[1] != len(GAIA_SERVICES):
        raise ValueError("service pooling expects the canonical ten-service axis")
    mean = h.mean(dim=1)
    maximum = h.amax(dim=1)
    std = h.std(dim=1, unbiased=False)
    return torch.cat([mean, maximum, std], dim=-1)


class SystemEventTrigger(nn.Module):
    """Random-init multimodal encoder + permutation-invariant system pooling head."""

    def __init__(self, graph, **args):
        super(SystemEventTrigger, self).__init__()
        self.name = args.get("main_model", "P6-C0-SystemEventTrigger")
        self.device = torch.device(
            "cuda" if args.get("gpu", False) and torch.cuda.is_available() else "cpu"
        )
        self.graph = torch.as_tensor(graph).to(self.device)
        self.num_nodes = int(args.get("num_nodes", graph.shape[0]))
        if self.num_nodes != len(GAIA_SERVICES):
            raise ValueError("system trigger expects the canonical ten GAIA services")

        adj = dense_to_sparse(self.graph)[0]
        trace2pod = torch.nn.functional.one_hot(adj[0], num_classes=self.graph.shape[0]) \
            + torch.nn.functional.one_hot(adj[1], num_classes=self.graph.shape[0])
        trace2pod = trace2pod / trace2pod.sum(axis=0, keepdim=True)
        trace2pod = torch.where(
            torch.isnan(trace2pod), torch.full_like(trace2pod, 0), trace2pod
        )

        self.node_emb = Embed(args["raw_node"], args["feature_node"], dim=4)
        self.log_emb = Embed(args["log_len"], args["feature_log"], dim=4)
        self.egde_emb = Embed(args["raw_edge"], args["feature_edge"], dim=5)

        self.dynamic_graph_learner = DynamicGraphLearner(
            node_dim=args["feature_node"],
            log_dim=args["feature_log"],
            hidden_dim=args.get("graph_hidden", 16),
            num_nodes=self.num_nodes,
            summary_mode=args.get("graph_summary_mode", "last"),
        )
        self.graph_sparse_weight = float(args.get("graph_sparse_weight", 1e-3))

        self.encoder = Encoder(
            graph=self.graph, node_embedding=args["feature_node"],
            edge_embedding=args["feature_edge"], log_embedding=args["feature_log"],
            node_heads=args["num_heads_node"], log_heads=args["num_heads_log"],
            edge_heads=args["num_heads_edge"], n2e_heads=args["num_heads_n2e"],
            e2n_heads=args["num_heads_e2n"], dropout=args["dropout"],
            batch_size=args["batch_size"], window_size=args["window"],
            num_layer=args["num_layer"], trace2pod=trace2pod,
            graph_hidden=args.get("graph_hidden", 16), num_nodes=self.num_nodes,
        )

        head_hidden = int(args.get("head_hidden", DEFAULT_HEAD_HIDDEN))
        self.head = nn.Sequential(
            nn.Linear(3 * int(args["feature_node"]), head_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(head_hidden, 1),
        )

    # -- inference interface -------------------------------------------------

    def encode(self, batch: Mapping[str, torch.Tensor]):
        """Return ``(service_representations, graph_reg_loss)`` for the target bin."""

        x_node, _ = self.node_emb(batch["data_node"])
        x_edge, _ = self.egde_emb(batch["data_edge"])
        x_log, _ = self.log_emb(batch["data_log"])

        edge_weights, graph_reg_loss = self.dynamic_graph_learner(x_node, x_log, self.graph)
        weight_mask = edge_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        z_node, _, _ = self.encoder(x_node, x_edge * weight_mask, x_log)
        # The target bin is the last window step, exactly like the frozen
        # Ada-MGAD evaluation path.
        return z_node[:, -1], graph_reg_loss * self.graph_sparse_weight

    def system_logits(self, service_representations: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant pooling followed by the scalar trigger head."""

        return self.head(pool_service_representations(service_representations)).reshape(-1)

    def forward(self, batch: Mapping[str, torch.Tensor], global_step=None):
        representations, graph_reg_loss = self.encode(batch)
        return self.system_logits(representations), graph_reg_loss

    @torch.no_grad()
    def system_scores(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.forward(batch)[0])
