from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from data.transforms import frame_to_tensor_bgr


class DeepfakeFrameDataset(Dataset):
    def __init__(self, root="datasets/FaceForensics++", image_size=128, max_frames_per_video=8):
        self.root = Path(root)
        self.image_size = image_size
        self.max_frames_per_video = max_frames_per_video
        self.samples = []
        for label_name, label in (("real", 0.0), ("fake", 1.0)):
            for path in sorted((self.root / label_name).glob("*.mp4")):
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(str(path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = torch.zeros(3, self.image_size, self.image_size)
        else:
            frame = frame_to_tensor_bgr(frame, self.image_size)
        return frame, torch.tensor(label, dtype=torch.float32)


class FusionClipDataset(DeepfakeFrameDataset):
    def __getitem__(self, idx):
        face, label = super().__getitem__(idx)
        roi = torch.zeros(90, 9, dtype=torch.float32)
        return roi, face, label

