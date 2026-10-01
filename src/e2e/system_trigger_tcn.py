"""Fixed small TCN encoder with the existing embeddings, graph loss and head."""

import torch
from torch import nn
from torch.nn import functional as F

from .system_trigger_model import SystemEventTrigger


TCN_SPEC = {"hidden_channels": 32, "kernel_size": 3, "dilations": [1, 2, 4],
            "convolutions_per_block": 2, "dropout": 0.2,
            "normalization": "per-time channel LayerNorm", "padding": "left only"}


class CausalResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.left_padding = 2 * int(dilation)
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, 3, dilation=dilation) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(2)])
        self.dropout = nn.Dropout(TCN_SPEC["dropout"])

    def forward(self, x):
        result = x
        for conv, norm in zip(self.convs, self.norms):
            result = conv(F.pad(result, (self.left_padding, 0)))
            result = norm(result.transpose(1, 2)).transpose(1, 2)
            result = self.dropout(F.relu(result))
        return x + result


class GraphCausalTCNEncoder(nn.Module):
    """Aggregate preweighted incoming/outgoing traces, then encode each service.

    The service axis is folded into the batch only for shared convolutions.
    Neither normalization nor convolution reduces over other samples/services.
    """

    def __init__(self, graph, node_dim, log_dim, edge_dim):
        super().__init__()
        graph = torch.as_tensor(graph)
        self.register_buffer("out_degree", (graph > 0).sum(dim=1).clamp_min(1).float())
        self.register_buffer("in_degree", (graph > 0).sum(dim=0).clamp_min(1).float())
        input_dim = int(node_dim) + int(log_dim) + 2 * int(edge_dim)
        hidden = TCN_SPEC["hidden_channels"]
        self.input_projection = nn.Conv1d(input_dim, hidden, 1)
        self.blocks = nn.Sequential(*[CausalResidualBlock(hidden, d) for d in TCN_SPEC["dilations"]])
        self.output_projection = nn.Conv1d(hidden, int(node_dim), 1)

    def forward(self, node, edge, log):
        outgoing = edge.sum(dim=3) / self.out_degree.view(1, 1, -1, 1)
        incoming = edge.sum(dim=2) / self.in_degree.view(1, 1, -1, 1)
        joined = torch.cat([node, log, outgoing, incoming], dim=-1)
        batch, time, services, channels = joined.shape
        sequences = joined.permute(0, 2, 3, 1).reshape(batch * services, channels, time)
        result = self.output_projection(self.blocks(self.input_projection(sequences)))
        result = result.reshape(batch, services, -1, time).permute(0, 3, 1, 2)
        return result, edge, log


class WindowCausalTCNTrigger(SystemEventTrigger):
    """Replace only the encoder; all other C0 modules/losses remain inherited."""

    def __init__(self, graph, **args):
        if args.get("graph_batch_scope") != "window":
            raise ValueError("TCN successor requires a window-independent graph")
        super().__init__(graph, **args)
        # Keeping the inherited initialization preserves the initial embeddings,
        # graph learner and head. The discarded old encoder is not optimized.
        self.encoder = GraphCausalTCNEncoder(self.graph, args["feature_node"], args["feature_log"],
                                            args["feature_edge"]).to(self.device)
