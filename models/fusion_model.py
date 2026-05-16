import torch
from torch import nn


class FusionClassifier(nn.Module):
    def __init__(self, rppg_dim: int = 64, artifact_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(rppg_dim + artifact_dim + 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
        )
        self.fake_head = nn.Linear(hidden // 2, 1)
        self.live_head = nn.Linear(hidden // 2, 1)
        self.conf_head = nn.Linear(hidden // 2, 1)

    def forward(self, rppg_feature, artifact_feature, quality_features):
        x = torch.cat([rppg_feature, artifact_feature, quality_features], dim=1)
        h = self.mlp(x)
        return {
            "fake_probability": torch.sigmoid(self.fake_head(h)).squeeze(-1),
            "liveness_score": torch.sigmoid(self.live_head(h)).squeeze(-1),
            "confidence_score": torch.sigmoid(self.conf_head(h)).squeeze(-1),
        }

