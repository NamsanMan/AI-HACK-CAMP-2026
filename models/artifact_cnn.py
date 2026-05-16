import torch
from torch import nn


class ArtifactCNN(nn.Module):
    """Small CPU-friendly CNN for aligned face crops, input B x 3 x H x W."""

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feature = nn.Linear(128, feature_dim)
        self.fake_head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.features(x).flatten(1)
        feat = torch.relu(self.feature(h))
        fake = torch.sigmoid(self.fake_head(feat))
        return {"feature": feat, "artifact_fake": fake.squeeze(-1)}

