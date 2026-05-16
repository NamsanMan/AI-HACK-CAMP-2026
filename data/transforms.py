import cv2
import numpy as np
import torch


def augment_realtime_artifacts(frame_bgr):
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    if np.random.rand() < 0.55:
        scale = float(np.random.uniform(0.45, 0.90))
        small = cv2.resize(frame, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    if np.random.rand() < 0.50:
        quality = int(np.random.randint(35, 91))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if np.random.rand() < 0.35:
        k = int(np.random.choice([3, 5]))
        frame = cv2.GaussianBlur(frame, (k, k), 0)

    if np.random.rand() < 0.65:
        alpha = float(np.random.uniform(0.70, 1.35))
        beta = float(np.random.uniform(-28, 28))
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    if np.random.rand() < 0.40:
        gamma = float(np.random.uniform(0.70, 1.45))
        table = ((np.arange(256) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
        frame = cv2.LUT(frame, table)

    if np.random.rand() < 0.35:
        noise = np.random.normal(0.0, np.random.uniform(2.0, 8.0), frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if np.random.rand() < 0.30:
        occ_w = int(w * np.random.uniform(0.10, 0.28))
        occ_h = int(h * np.random.uniform(0.08, 0.24))
        x = int(np.random.randint(0, max(1, w - occ_w)))
        y = int(np.random.randint(0, max(1, h - occ_h)))
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8).tolist()
        cv2.rectangle(frame, (x, y), (x + occ_w, y + occ_h), color, -1)

    return frame


def frame_to_tensor_bgr(frame_bgr, size: int = 128):
    frame = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(frame).permute(2, 0, 1)


def normalize_roi_sequence(seq):
    seq = np.asarray(seq, dtype=np.float32)
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mean) / std
