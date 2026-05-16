import cv2
import numpy as np
import torch


def frame_to_tensor_bgr(frame_bgr, size: int = 128):
    frame = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(frame).permute(2, 0, 1)


def normalize_roi_sequence(seq):
    seq = np.asarray(seq, dtype=np.float32)
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mean) / std

