"""Evidence-fusion localizers: per-node root-cause scoring."""

import torch
import torch.nn as nn

from src.rca.evidence import FEATURE_NAMES, feature_indices


class Localizer(nn.Module):
    """Maps per-node evidence features to a root-cause score.

    `groups` selects the evidence groups (anomaly / propagation / temporal)
    used as input; dropping groups yields the ablation variants.
    """

    def __init__(self, groups=('anomaly', 'propagation', 'temporal'), hidden=32):
        super().__init__()
        self.groups = tuple(groups)
        self.idx = feature_indices(self.groups)
        in_dim = len(self.idx)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):
        """feats: [B, N, len(FEATURE_NAMES)] -> scores [B, N]."""
        x = feats[..., self.idx]
        return self.mlp(x).squeeze(-1)

    def config(self):
        return {'groups': list(self.groups), 'hidden': self.mlp[0].out_features,
                'feature_names': FEATURE_NAMES}


class SetAttentionLocalizer(nn.Module):
    """Cross-node comparative scorer.

    The per-node MLP scores every node independently and therefore cannot
    express relative judgments ("which twin looks more anomalous in THIS
    window"). Here node feature vectors attend to each other within the
    window before scoring, so a node's score can depend on its peers.

    Learnable node embeddings preserve node identity (e.g. mob1 vs mob2).
    """

    def __init__(self, groups=('anomaly', 'propagation', 'temporal'),
                 d_model=64, nhead=4, num_layers=2, num_nodes=10, dropout=0.1):
        super().__init__()
        self.groups = tuple(groups)
        self.idx = feature_indices(self.groups)
        self.in_proj = nn.Linear(len(self.idx), d_model)
        self.node_emb = nn.Parameter(torch.randn(num_nodes, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, feats):
        """feats: [B, N, len(FEATURE_NAMES)] -> scores [B, N]."""
        h = self.in_proj(feats[..., self.idx]) + self.node_emb[None, :, :]
        h = self.encoder(h)
        return self.head(h).squeeze(-1)

    def config(self):
        return {'groups': list(self.groups), 'd_model': self.in_proj.out_features,
                'feature_names': FEATURE_NAMES}


def build_localizer(arch, groups, **kwargs):
    if arch == 'mlp':
        return Localizer(groups=groups, hidden=kwargs.get('hidden', 32))
    if arch == 'setattn':
        return SetAttentionLocalizer(
            groups=groups, d_model=kwargs.get('d_model', 64),
            nhead=kwargs.get('nhead', 4), num_layers=kwargs.get('num_layers', 2),
            num_nodes=kwargs.get('num_nodes', 10))
    raise ValueError(f'unknown arch: {arch}')
