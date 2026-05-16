import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config import CFG
from models.artifact_cnn import ArtifactCNN
from models.fusion_model import FusionClassifier
from models.rppg_tcn import RPPGTCN
from utils.face_detector import FaceDetector
from utils.landmark_roi import LandmarkROIExtractor
from utils.rppg_signal import TemporalRGBBuffer, estimate_signal_quality
from utils.tracker import BBoxSmoother
from utils.visualization import draw_overlays


def get_device():
    if CFG.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(CFG.device)


def to_face_tensor(face_bgr, device):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


class RealtimeDeepfakePipeline:
    def __init__(self):
        self.device = get_device()
        self.face_detector = FaceDetector(CFG.min_detection_confidence, CFG.detection_interval)
        self.tracker = BBoxSmoother()
        self.roi_extractor = LandmarkROIExtractor(CFG.min_tracking_confidence)
        self.buffer = TemporalRGBBuffer(CFG.window_size)
        self.rppg = RPPGTCN().to(self.device).eval()
        self.artifact = ArtifactCNN().to(self.device).eval()
        self.fusion = FusionClassifier().to(self.device).eval()
        self.last_scores = {"fake_probability": 0.0, "liveness_score": 0.0, "confidence_score": 0.0, "estimated_hr": 0.0}
        self.frame_idx = 0

    @torch.no_grad()
    def process(self, frame_bgr, fps_hint: float):
        bbox, det_score, _ = self.face_detector.detect(frame_bgr)
        bbox = self.tracker.update(bbox)
        rois, roi_rgb, roi_quality = self.roi_extractor.extract(frame_bgr, bbox=bbox)
        self.buffer.append(roi_rgb)

        should_score = self.frame_idx % CFG.score_interval_frames == 0 and self.buffer.ready()
        self.frame_idx += 1
        if should_score and bbox is not None:
            face = self.roi_extractor.aligned_face_crop(frame_bgr, bbox, CFG.face_crop_size)
            if face is not None:
                x_roi = torch.from_numpy(self.buffer.array()).unsqueeze(0).to(self.device)
                face_tensor = to_face_tensor(face, self.device)
                signal_quality = estimate_signal_quality(self.buffer.array(), fps_hint)
                rppg_out = self.rppg(x_roi)
                art_out = self.artifact(face_tensor)
                quality = torch.tensor(
                    [[roi_quality, det_score, signal_quality["motion_quality"]]],
                    dtype=torch.float32,
                    device=self.device,
                )
                fused = self.fusion(rppg_out["feature"], art_out["feature"], quality)

                confidence = float(fused["confidence_score"].item())
                confidence = 0.55 * confidence + 0.45 * min(roi_quality, det_score, signal_quality["motion_quality"])
                liveness = 0.50 * float(fused["liveness_score"].item()) + 0.30 * signal_quality["pulse_consistency"] + 0.20 * float(rppg_out["rppg_liveness"].item())
                fake = 0.65 * float(fused["fake_probability"].item()) + 0.35 * float(art_out["artifact_fake"].item())
                if confidence < 0.35:
                    fake *= 0.8

                self.last_scores = {
                    "fake_probability": float(np.clip(fake, 0.0, 1.0)),
                    "liveness_score": float(np.clip(liveness, 0.0, 1.0)),
                    "confidence_score": float(np.clip(confidence, 0.0, 1.0)),
                    "estimated_hr": signal_quality["estimated_hr"] or float(rppg_out["estimated_hr"].item()),
                }
        else:
            self.last_scores["confidence_score"] = min(self.last_scores["confidence_score"], self.buffer.fill_ratio())

        return bbox, rois, self.last_scores


def run_capture(source=0, window_name="Real-Time Deepfake rPPG Prototype"):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    pipeline = RealtimeDeepfakePipeline()
    prev = time.time()
    smoothed_fps = 0.0
    fps_hint = cap.get(cv2.CAP_PROP_FPS)
    if not fps_hint or fps_hint < 1:
        fps_hint = CFG.target_fps

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        inst_fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        smoothed_fps = inst_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * inst_fps

        bbox, rois, scores = pipeline.process(frame, fps_hint)
        draw_overlays(frame, bbox=bbox, rois=rois, scores=scores, fps=smoothed_fps)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def normalized_input_path(path: str):
    return str(Path(path).expanduser().resolve())
