from dataclasses import dataclass


@dataclass
class Config:
    camera_index: int = 0
    target_fps: int = 30
    fallback_fps: int = 15
    detection_interval: int = 7
    landmark_interval: int = 3
    mediapipe_downscale_width: int = 320
    window_seconds: float = 3.0
    score_interval_seconds: float = 1.0
    face_crop_size: int = 96
    min_partial_rppg_frames: int = 12
    warmup_ema_alpha: float = 0.60
    stable_ema_alpha: float = 0.25
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.55
    roi_alpha: float = 0.28
    device: str = "auto"
    mediapipe_model_dir: str = "utils/models"
    artifact_backbone: str = "rexnet_100"
    artifact_pretrained: bool = True
    rppg_weights: str = "checkpoints/rppg_tcn.pt"
    artifact_weights: str = "checkpoints/artifact_rexnet_100.pt"
    fusion_weights: str = "checkpoints/fusion_model.pt"

    @property
    def window_size(self) -> int:
        return max(12, int(round(self.window_seconds * self.target_fps)))

    @property
    def score_interval_frames(self) -> int:
        return max(1, int(round(self.score_interval_seconds * self.target_fps)))


CFG = Config()

# Backward-compatible constants used by preprocess_ff.py and preprocess_rppg_ff.py.
FF_ROOT = "datasets/FaceForensics++"
ARTIFACT_IN_SIZE = CFG.face_crop_size
