"""A dynamic graph for each window, with no aggregation across samples."""

import torch
import torch.nn.functional as F

from src.model_util import DynamicGraphLearner


class WindowDynamicGraphLearner(DynamicGraphLearner):
    """Keep the legacy parameters and singleton loss, retaining the batch axis."""

    def _summarize_inputs(self, x_node, x_log):
        h = torch.cat([x_node, x_log], dim=-1)
        if self.summary_mode == "mean":
            return h.mean(dim=1)
        return h[:, -1]

    def forward(self, x_node, x_log, static_graph):
        h = self._summarize_inputs(x_node, x_log)
        src, tgt = self.proj_src(h), self.proj_tgt(h)
        sim = torch.bmm(src, tgt.transpose(1, 2)) / (src.shape[-1] ** 0.5)
        if self.training:
            noise = torch.zeros_like(sim).uniform_(1e-6, 1 - 1e-6)
            noise = torch.log(noise) - torch.log(1 - noise)
            dynamic = torch.sigmoid((sim + noise) / self.temperature)
        else:
            dynamic = torch.sigmoid(sim / self.temperature)
        dynamic = dynamic * (1 - torch.eye(self.num_nodes, device=sim.device)).unsqueeze(0)
        static = static_graph.float().unsqueeze(0).expand_as(dynamic)
        weights = dynamic * static
        lam = torch.sigmoid(self.graph_lambda)
        fused = lam * weights + (1 - lam) * static
        # The old flattened batchmean divides a singleton's KL sum by N*N.
        # Average that same quantity over independent windows.
        kl = F.kl_div(weights.clamp(1e-6, 1 - 1e-6).log(),
                      static.clamp(1e-6, 1 - 1e-6), reduction="none").mean()
        return fused, weights.mean() + 0.1 * kl
