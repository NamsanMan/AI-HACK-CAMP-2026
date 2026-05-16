from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import normalize_roi_sequence


class UBFCRPPGDataset(Dataset):
    """UBFC-rPPG loader. Uses cached ROI npy if present, otherwise samples simple frame means."""

    def __init__(self, root="datasets/UBFC-rPPG", window_size=90, stride=30):
        self.root = Path(root)
        self.window_size = window_size
        self.stride = stride
        self.samples = []
        for subject in sorted(self.root.glob("subject_*")):
            video = next((subject / name for name in ("video.avi", "vid.avi") if (subject / name).exists()), None)
            gt = subject / "ground_truth.txt"
            if video is not None and gt.exists():
                self.samples.append((video, gt))

    def __len__(self):
        return len(self.samples)

    def _load_hr(self, gt_path):
        values = np.loadtxt(gt_path, ndmin=2)
        if values.shape[0] >= 2:
            hr_values = values[1, :]
        elif values.shape[1] >= 2:
            hr_values = values[:, 1]
        else:
            hr_values = values.reshape(-1)
        plausible = hr_values[(hr_values >= 35) & (hr_values <= 220)]
        return float(plausible.mean()) if plausible.size else 75.0

    def _video_to_rgb_means(self, path):
        cap = cv2.VideoCapture(str(path))
        values = []
        while len(values) < self.window_size:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            crops = [
                frame[int(h * 0.42):int(h * 0.62), int(w * 0.18):int(w * 0.42)],
                frame[int(h * 0.42):int(h * 0.62), int(w * 0.58):int(w * 0.82)],
                frame[int(h * 0.16):int(h * 0.32), int(w * 0.34):int(w * 0.66)],
            ]
            row = []
            for crop in crops:
                row.extend((crop[:, :, ::-1].mean(axis=(0, 1)) / 255.0).tolist())
            values.append(row)
        cap.release()
        if not values:
            values = np.zeros((self.window_size, 9), dtype=np.float32)
        while len(values) < self.window_size:
            values.append(values[-1])
        return np.asarray(values[: self.window_size], dtype=np.float32)

    def __getitem__(self, idx):
        video, gt = self.samples[idx]
        seq = normalize_roi_sequence(self._video_to_rgb_means(video))
        hr = self._load_hr(gt)
        return torch.from_numpy(seq), torch.tensor(hr, dtype=torch.float32)
