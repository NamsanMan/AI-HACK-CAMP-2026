import torch
from torch import nn


class ArtifactReXNet(nn.Module):
    """ReXNet artifact branch using timm ImageNet pretrained weights."""

    def __init__(self, model_name: str = "rexnet_100.nav_in1k", feature_dim: int = 64, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("ArtifactReXNet requires timm. Install it with `pip install timm`.") from exc

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = int(getattr(self.backbone, "num_features"))
        self.feature = nn.Linear(in_features, feature_dim)
        self.fake_head = nn.Linear(feature_dim, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        h = self.backbone(x)
        feat = torch.relu(self.feature(h))
        fake = torch.sigmoid(self.fake_head(feat))
        return {"feature": feat, "artifact_fake": fake.squeeze(-1)}

    def optimizer_param_groups(self, backbone_lr: float = 1e-5, head_lr: float = 1e-3):
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": list(self.feature.parameters()) + list(self.fake_head.parameters()), "lr": head_lr},
        ]

