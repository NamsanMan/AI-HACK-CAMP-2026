import torch
from torch import nn


class RPPGTCN(nn.Module):
    """Lightweight temporal model over raw cheek/forehead RGB means, shape B x T x 9."""

    def __init__(self, in_channels: int = 9, hidden: int = 48, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature = nn.Linear(hidden, feature_dim)
        self.hr_head = nn.Linear(feature_dim, 1)
        self.live_head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError("RPPGTCN expects B x T x 9 input.")
        x = x.transpose(1, 2)
        h = self.pool(self.net(x)).squeeze(-1)
        feat = torch.relu(self.feature(h))
        hr = 40.0 + 140.0 * torch.sigmoid(self.hr_head(feat))
        liveness = torch.sigmoid(self.live_head(feat))
        return {"feature": feat, "estimated_hr": hr.squeeze(-1), "rppg_liveness": liveness.squeeze(-1)}

