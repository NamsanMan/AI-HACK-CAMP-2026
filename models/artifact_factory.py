from models.artifact_cnn import ArtifactCNN
from models.artifact_rexnet import ArtifactReXNet


def normalize_artifact_backbone(name: str) -> str:
    return (name or "custom").lower().replace("-", "_")


def create_artifact_model(backbone: str = "custom", pretrained: bool = True, feature_dim: int = 64):
    name = normalize_artifact_backbone(backbone)
    if name in ("custom", "cnn", "artifact_cnn"):
        return ArtifactCNN(feature_dim=feature_dim)
    if name in ("rexnet_100", "rexnet100"):
        return ArtifactReXNet("rexnet_100.nav_in1k", feature_dim=feature_dim, pretrained=pretrained)
    if name in ("rexnet_150", "rexnet150"):
        return ArtifactReXNet("rexnet_150.nav_in1k", feature_dim=feature_dim, pretrained=pretrained)
    raise ValueError(f"Unsupported artifact backbone: {backbone}")

