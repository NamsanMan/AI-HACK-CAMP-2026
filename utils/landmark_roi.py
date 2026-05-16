import cv2
import numpy as np
import mediapipe as mp

from config import CFG
from utils.mediapipe_tasks import FACE_LANDMARKER_URL, ensure_model, mp_image_from_bgr


LEFT_CHEEK = [50, 101, 118, 117, 123, 147, 187, 205, 203, 206, 216, 192]
RIGHT_CHEEK = [280, 330, 347, 346, 352, 376, 411, 425, 423, 426, 436, 416]
FOREHEAD = [109, 10, 338, 337, 336, 296, 334, 293, 300, 151, 70, 63, 105, 66, 107]


class LandmarkROIExtractor:
    def __init__(self, min_tracking_confidence: float = 0.55, interval: int | None = None):
        self.mesh = None
        self.task_landmarker = None
        if hasattr(mp, "solutions"):
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=min_tracking_confidence,
            )
        elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            model_path = ensure_model(
                f"{CFG.mediapipe_model_dir}/face_landmarker.task",
                FACE_LANDMARKER_URL,
            )
            if model_path is not None:
                base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
                options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.task_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.last_landmarks = None
        self.last_rois = None
        self.last_signal = None
        self.last_quality = 0.0
        self.frame_idx = 0
        self.interval = max(1, interval or CFG.landmark_interval)

    def close(self):
        for attr in ("task_landmarker", "mesh"):
            obj = getattr(self, attr, None)
            if obj is not None and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    pass
            setattr(self, attr, None)

    def process(self, frame_bgr):
        if self.mesh is None and self.task_landmarker is None:
            return None
        should_run = self.frame_idx % self.interval == 0 or self.last_landmarks is None
        self.frame_idx += 1
        if not should_run:
            return self.last_landmarks

        original_h, original_w = frame_bgr.shape[:2]
        scale = 1.0
        proc = frame_bgr
        if original_w > CFG.mediapipe_downscale_width:
            scale = CFG.mediapipe_downscale_width / float(original_w)
            proc = cv2.resize(
                frame_bgr,
                (CFG.mediapipe_downscale_width, max(1, int(original_h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        if self.task_landmarker is not None:
            result = self.task_landmarker.detect(mp_image_from_bgr(mp, proc))
            if result.face_landmarks:
                self.last_landmarks = result.face_landmarks[0]
            return self.last_landmarks

        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if result.multi_face_landmarks:
            self.last_landmarks = result.multi_face_landmarks[0].landmark
        return self.last_landmarks

    @staticmethod
    def _points(landmarks, indices, width, height):
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            pts.append([int(lm.x * width), int(lm.y * height)])
        return np.asarray(pts, dtype=np.int32)

    @staticmethod
    def _fallback_rois_from_bbox(bbox):
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        return {
            "left_cheek": np.asarray([
                [x1 + int(0.18 * w), y1 + int(0.48 * h)],
                [x1 + int(0.42 * w), y1 + int(0.45 * h)],
                [x1 + int(0.40 * w), y1 + int(0.68 * h)],
                [x1 + int(0.20 * w), y1 + int(0.68 * h)],
            ], dtype=np.int32),
            "right_cheek": np.asarray([
                [x1 + int(0.58 * w), y1 + int(0.45 * h)],
                [x1 + int(0.82 * w), y1 + int(0.48 * h)],
                [x1 + int(0.80 * w), y1 + int(0.68 * h)],
                [x1 + int(0.60 * w), y1 + int(0.68 * h)],
            ], dtype=np.int32),
            "forehead": np.asarray([
                [x1 + int(0.34 * w), y1 + int(0.16 * h)],
                [x1 + int(0.66 * w), y1 + int(0.16 * h)],
                [x1 + int(0.62 * w), y1 + int(0.30 * h)],
                [x1 + int(0.38 * w), y1 + int(0.30 * h)],
            ], dtype=np.int32),
        }

    def extract(self, frame_bgr, bbox=None):
        landmarks = self.process(frame_bgr)
        if landmarks is None and bbox is None:
            self.last_rois = None
            self.last_signal = None
            self.last_quality = 0.0
            return None, None, 0.0

        h, w = frame_bgr.shape[:2]
        if landmarks is None:
            rois = self._fallback_rois_from_bbox(bbox)
        else:
            rois = {
                "left_cheek": self._points(landmarks, LEFT_CHEEK, w, h),
                "right_cheek": self._points(landmarks, RIGHT_CHEEK, w, h),
                "forehead": self._points(landmarks, FOREHEAD, w, h),
            }
        if rois is None:
            return None, None, 0.0
        means = []
        valid = 0
        for pts in rois.values():
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            pixels = frame_bgr[mask > 0]
            if pixels.size == 0:
                means.extend([0.0, 0.0, 0.0])
                continue
            valid += 1
            rgb_mean = pixels[:, ::-1].mean(axis=0) / 255.0
            means.extend(rgb_mean.tolist())
        self.last_rois = rois
        self.last_signal = np.asarray(means, dtype=np.float32)
        self.last_quality = valid / 3.0
        return self.last_rois, self.last_signal, self.last_quality

    @staticmethod
    def aligned_face_crop(frame_bgr, bbox, size: int = 128):
        if bbox is None:
            return None
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.20)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return cv2.resize(frame_bgr[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


class LandmarkROI:
    """Compatibility wrapper for preprocess_rppg_ff.py."""

    def __init__(self, min_tracking_confidence: float = 0.55):
        self.extractor = LandmarkROIExtractor(min_tracking_confidence)

    def process(self, frame_bgr):
        rois, signal, quality = self.extractor.extract(frame_bgr)
        return signal, rois, quality > 0.0
