import cv2
import mediapipe as mp

from config import CFG
from utils.mediapipe_tasks import FACE_DETECTOR_URL, ensure_model, mp_image_from_bgr


class FaceDetector:
    def __init__(self, min_confidence: float = 0.55, detection_interval: int = 6):
        self.detector = None
        self.task_detector = None
        self.haar = None
        if hasattr(mp, "solutions"):
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_confidence,
            )
        elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            model_path = ensure_model(
                f"{CFG.mediapipe_model_dir}/blaze_face_short_range.tflite",
                FACE_DETECTOR_URL,
            )
            if model_path is not None:
                base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
                options = mp.tasks.vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=min_confidence,
                )
                self.task_detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        if self.detector is None and self.task_detector is None:
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar = cv2.CascadeClassifier(cascade)
        self.detection_interval = max(1, detection_interval)
        self.frame_idx = 0
        self.last_bbox = None
        self.last_score = 0.0

    def close(self):
        for attr in ("task_detector", "detector"):
            obj = getattr(self, attr, None)
            if obj is not None and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    pass
            setattr(self, attr, None)

    def detect(self, frame_bgr):
        run_detection = self.frame_idx % self.detection_interval == 0 or self.last_bbox is None
        self.frame_idx += 1
        if not run_detection:
            return self.last_bbox, self.last_score, False

        if self.detector is None and self.task_detector is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
            if len(faces) == 0:
                self.last_score = 0.0
                return self.last_bbox, self.last_score, True
            x, y, w_box, h_box = max(faces, key=lambda b: b[2] * b[3])
            self.last_bbox = (int(x), int(y), int(x + w_box), int(y + h_box))
            self.last_score = 0.65
            return self.last_bbox, self.last_score, True

        if self.task_detector is not None:
            result = self.task_detector.detect(mp_image_from_bgr(mp, frame_bgr))
            if not result.detections:
                self.last_score = 0.0
                self.last_bbox = None
                return self.last_bbox, self.last_score, True
            h, w = frame_bgr.shape[:2]
            det = max(result.detections, key=lambda d: d.categories[0].score if d.categories else 0.0)
            box = det.bounding_box
            x1 = max(0, int(box.origin_x))
            y1 = max(0, int(box.origin_y))
            x2 = min(w - 1, int(box.origin_x + box.width))
            y2 = min(h - 1, int(box.origin_y + box.height))
            self.last_bbox = (x1, y1, x2, y2)
            self.last_score = float(det.categories[0].score if det.categories else 0.5)
            return self.last_bbox, self.last_score, True

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)
        if not result.detections:
            self.last_score = 0.0
            self.last_bbox = None
            return self.last_bbox, self.last_score, True

        h, w = frame_bgr.shape[:2]
        det = max(result.detections, key=lambda d: d.score[0])
        box = det.location_data.relative_bounding_box
        x1 = max(0, int(box.xmin * w))
        y1 = max(0, int(box.ymin * h))
        x2 = min(w - 1, int((box.xmin + box.width) * w))
        y2 = min(h - 1, int((box.ymin + box.height) * h))
        self.last_bbox = (x1, y1, x2, y2)
        self.last_score = float(det.score[0])
        return self.last_bbox, self.last_score, True
