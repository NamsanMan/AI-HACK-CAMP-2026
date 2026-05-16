import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_confidence: float = 0.55, detection_interval: int = 6):
        self.detector = None
        self.haar = None
        if hasattr(mp, "solutions"):
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_confidence,
            )
        else:
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar = cv2.CascadeClassifier(cascade)
        self.detection_interval = max(1, detection_interval)
        self.frame_idx = 0
        self.last_bbox = None
        self.last_score = 0.0

    def detect(self, frame_bgr):
        run_detection = self.frame_idx % self.detection_interval == 0 or self.last_bbox is None
        self.frame_idx += 1
        if not run_detection:
            return self.last_bbox, self.last_score, False

        if self.detector is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
            if len(faces) == 0:
                self.last_score = 0.0
                return self.last_bbox, self.last_score, True
            x, y, w_box, h_box = max(faces, key=lambda b: b[2] * b[3])
            self.last_bbox = (int(x), int(y), int(x + w_box), int(y + h_box))
            self.last_score = 0.65
            return self.last_bbox, self.last_score, True

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)
        if not result.detections:
            self.last_score = 0.0
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
