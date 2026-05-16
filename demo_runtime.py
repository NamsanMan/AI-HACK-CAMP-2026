import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config import CFG
from data.transforms import normalize_roi_sequence
from models.artifact_factory import create_artifact_model
from models.fusion_model import FusionClassifier
from models.rppg_tcn import RPPGTCN
from utils.face_detector import FaceDetector
from utils.landmark_roi import LandmarkROIExtractor
from utils.rppg_signal import TemporalRGBBuffer, estimate_signal_quality
from utils.tracker import BBoxSmoother
from utils.visualization import draw_overlays
from utils.checkpoints import prefer_fusion_checkpoint


def get_device():
    if CFG.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(CFG.device)


def to_face_tensor(face_bgr, device):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def maybe_load_state(model, path, device):
    if path and Path(path).exists():
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        return True
    return False


class RealtimeDeepfakePipeline:
    def __init__(self):
        self.device = get_device()
        self.face_detector = FaceDetector(CFG.min_detection_confidence, CFG.detection_interval)
        self.tracker = BBoxSmoother()
        self.roi_extractor = LandmarkROIExtractor(CFG.min_tracking_confidence)
        self.buffer = TemporalRGBBuffer(CFG.window_size)
        self.rppg = RPPGTCN().to(self.device).eval()
        self.artifact = create_artifact_model(
            CFG.artifact_backbone,
            pretrained=CFG.artifact_pretrained and not Path(CFG.artifact_weights).exists(),
        ).to(self.device).eval()
        self.fusion = FusionClassifier().to(self.device).eval()
        maybe_load_state(self.rppg, prefer_fusion_checkpoint(CFG.rppg_weights, "rppg_fusion_best.pt"), self.device)
        maybe_load_state(
            self.artifact,
            prefer_fusion_checkpoint(CFG.artifact_weights, f"artifact_{CFG.artifact_backbone}_fusion_best.pt"),
            self.device,
        )
        maybe_load_state(self.fusion, CFG.fusion_weights, self.device)
        self.last_scores = {"fake_probability": 0.0, "liveness_score": 0.0, "confidence_score": 0.0, "estimated_hr": 0.0}
        self.frame_idx = 0

    def close(self):
        self.face_detector.close()
        self.roi_extractor.close()

    def _smooth_scores(self, scores, fill_ratio):
        alpha = CFG.warmup_ema_alpha if fill_ratio < 1.0 else CFG.stable_ema_alpha
        smoothed = {}
        for key, value in scores.items():
            old = self.last_scores.get(key, value)
            if key == "estimated_hr" and value <= 0:
                smoothed[key] = old
            else:
                smoothed[key] = float(alpha * value + (1.0 - alpha) * old)
        return smoothed

    @torch.no_grad()
    def process(self, frame_bgr, fps_hint: float):
        bbox, det_score, _ = self.face_detector.detect(frame_bgr)
        bbox = self.tracker.update(bbox)
        rois, roi_rgb, roi_quality = self.roi_extractor.extract(frame_bgr, bbox=bbox)
        self.buffer.append(roi_rgb)

        should_score = self.frame_idx % CFG.score_interval_frames == 0 and bbox is not None
        self.frame_idx += 1
        if should_score:
            face = self.roi_extractor.aligned_face_crop(frame_bgr, bbox, CFG.face_crop_size)
            if face is not None:
                face_tensor = to_face_tensor(face, self.device)
                art_out = self.artifact(face_tensor)
                artifact_fake = float(art_out["artifact_fake"].item())
                fill_ratio = self.buffer.fill_ratio()
                has_partial_rppg = len(self.buffer) >= CFG.min_partial_rppg_frames

                if has_partial_rppg:
                    roi_window = self.buffer.padded_array()
                    roi_window = normalize_roi_sequence(roi_window)
                    x_roi = torch.from_numpy(roi_window).unsqueeze(0).to(self.device)
                    signal_quality = estimate_signal_quality(self.buffer.array(), fps_hint)
                    rppg_out = self.rppg(x_roi)
                    motion_quality = signal_quality["motion_quality"] * min(fill_ratio, 1.0)
                    quality = torch.tensor(
                        [[roi_quality * fill_ratio, det_score, motion_quality]],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    fused = self.fusion(rppg_out["feature"], art_out["feature"], quality)

                    rppg_weight = min(fill_ratio, 1.0)
                    fusion_fake = float(fused["fake_probability"].item())
                    confidence = float(fused["confidence_score"].item())
                    confidence = 0.55 * confidence + 0.45 * min(roi_quality, det_score, motion_quality)
                    confidence *= 0.35 + 0.65 * rppg_weight
                    liveness = (
                        0.50 * float(fused["liveness_score"].item())
                        + 0.30 * signal_quality["pulse_consistency"]
                        + 0.20 * float(rppg_out["rppg_liveness"].item())
                    )
                    fake = (1.0 - rppg_weight) * artifact_fake + rppg_weight * (
                        0.65 * fusion_fake + 0.35 * artifact_fake
                    )
                    estimated_hr = signal_quality["estimated_hr"] or float(rppg_out["estimated_hr"].item())
                else:
                    confidence = min(det_score, roi_quality) * max(0.15, fill_ratio)
                    liveness = 0.5 * fill_ratio
                    fake = artifact_fake
                    estimated_hr = 0.0

                if confidence < 0.35:
                    fake *= 0.9

                raw_scores = {
                    "fake_probability": float(np.clip(fake, 0.0, 1.0)),
                    "liveness_score": float(np.clip(liveness, 0.0, 1.0)),
                    "confidence_score": float(np.clip(confidence, 0.0, 1.0)),
                    "estimated_hr": float(estimated_hr),
                }
                self.last_scores = self._smooth_scores(raw_scores, fill_ratio)
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

    try:
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
    finally:
        pipeline.close()
        cap.release()
        cv2.destroyAllWindows()


def normalized_input_path(path: str):
    return str(Path(path).expanduser().resolve())
