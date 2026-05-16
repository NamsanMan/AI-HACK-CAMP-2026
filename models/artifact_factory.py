from models.artifact_rexnet import ArtifactReXNet

ARTIFACT_BACKBONE = "rexnet_100"
TIMM_MODEL_NAME = "rexnet_100.nav_in1k"


def normalize_artifact_backbone(name: str) -> str:
    return (name or ARTIFACT_BACKBONE).lower().replace("-", "_")


def create_artifact_model(backbone: str = ARTIFACT_BACKBONE, pretrained: bool = True, feature_dim: int = 64):
    name = normalize_artifact_backbone(backbone)
    if name in ("rexnet_100", "rexnet100"):
        return ArtifactReXNet(TIMM_MODEL_NAME, feature_dim=feature_dim, pretrained=pretrained)
    raise ValueError(f"Only {ARTIFACT_BACKBONE} is supported for the artifact branch: {backbone}")
