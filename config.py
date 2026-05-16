from dataclasses import dataclass


@dataclass
class Config:
    camera_index: int = 0
    target_fps: int = 30
    fallback_fps: int = 15
    detection_interval: int = 6
    window_seconds: float = 3.0
    score_interval_seconds: float = 1.0
    face_crop_size: int = 128
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.55
    roi_alpha: float = 0.28
    device: str = "auto"

    @property
    def window_size(self) -> int:
        return max(12, int(round(self.window_seconds * self.target_fps)))

    @property
    def score_interval_frames(self) -> int:
        return max(1, int(round(self.score_interval_seconds * self.target_fps)))


CFG = Config()

