from pathlib import Path
from urllib.request import urlopen


FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_model(path: str | Path, url: str) -> Path | None:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=10) as response:
            path.write_bytes(response.read())
    except Exception:
        return None
    return path if path.exists() else None


def mp_image_from_bgr(mp_module, frame_bgr):
    return mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=frame_bgr[:, :, ::-1])
