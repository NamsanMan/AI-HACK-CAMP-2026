from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import frame_to_tensor_bgr, normalize_roi_sequence


class DeepfakeFrameDataset(Dataset):
    def __init__(self, root="datasets/FaceForensics++", image_size=128, max_frames_per_video=8, frames_dir=None):
        self.root = Path(root)
        self.image_size = image_size
        self.max_frames_per_video = max_frames_per_video
        self.samples = []
        frames_root = Path(frames_dir) if frames_dir else self.root / "frames"
        if frames_root.exists():
            for label_name, label in (("real", 0.0), ("fake", 1.0)):
                for path in sorted((frames_root / label_name).glob("*.jpg")):
                    self.samples.append((path, label, "image"))
            return

        for label_name, label in (("real", 0.0), ("fake", 1.0)):
            for src_name in (label_name, "original" if label_name == "real" else "Deepfakes"):
                src_dir = self.root / src_name
                if not src_dir.exists():
                    continue
                for path in sorted(src_dir.glob("*.mp4")) + sorted(src_dir.glob("*.avi")):
                    self.samples.append((path, label, "video"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, kind = self.samples[idx]
        if kind == "image":
            frame = cv2.imread(str(path))
            if frame is None:
                frame = torch.zeros(3, self.image_size, self.image_size)
            else:
                frame = frame_to_tensor_bgr(frame, self.image_size)
            return frame, torch.tensor(label, dtype=torch.float32)

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


class FusionClipDataset(Dataset):
    def __init__(
        self,
        root="datasets/FaceForensics++",
        image_size=128,
        window_size=90,
        frames_dir=None,
        rppg_dir=None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.window_size = window_size
        self.frames_root = Path(frames_dir) if frames_dir else self.root / "frames"
        self.rppg_root = Path(rppg_dir) if rppg_dir else self.root / "rppg"
        self.samples = []

        if self.frames_root.exists() and self.rppg_root.exists():
            for label_name, label in (("real", 0.0), ("fake", 1.0)):
                for rppg_path in sorted((self.rppg_root / label_name).glob("*_rppg.npy")):
                    stem = rppg_path.stem.removesuffix("_rppg")
                    frames = sorted((self.frames_root / label_name).glob(f"{stem}_f*.jpg"))
                    if frames:
                        self.samples.append((rppg_path, frames, label))

    def __len__(self):
        return len(self.samples)

    def _load_roi_window(self, path):
        seq = np.load(path).astype(np.float32)
        if seq.size == 0:
            seq = np.zeros((self.window_size, 9), dtype=np.float32)
        if np.nanmax(seq) > 2.0:
            seq = seq / 255.0
        if len(seq) >= self.window_size:
            start = np.random.randint(0, len(seq) - self.window_size + 1)
            seq = seq[start : start + self.window_size]
        else:
            pad = np.repeat(seq[-1:], self.window_size - len(seq), axis=0)
            seq = np.concatenate([seq, pad], axis=0)
        return torch.from_numpy(normalize_roi_sequence(seq))

    def __getitem__(self, idx):
        rppg_path, frames, label = self.samples[idx]
        roi = self._load_roi_window(rppg_path)
        frame_path = frames[np.random.randint(0, len(frames))]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            face = torch.zeros(3, self.image_size, self.image_size)
        else:
            face = frame_to_tensor_bgr(frame, self.image_size)
        return roi, face, torch.tensor(label, dtype=torch.float32)
