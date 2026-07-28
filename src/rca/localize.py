"""Evidence-fusion localizer: per-node root-cause scoring MLP."""

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
