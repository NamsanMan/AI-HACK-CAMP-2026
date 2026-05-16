"""
Precompute UBFC-rPPG ROI RGB sliding windows for rPPG branch training.

Output:
    datasets/UBFC-rPPG/windows/
        subject_01_s000000.npz

Each npz contains:
    x: (T, 9) ROI RGB sequence, float32 in 0..1
    hr: scalar average HR for the same temporal window
    subject: subject folder name
    start: start frame index
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config
from utils.face_detector import FaceDetector
from utils.landmark_roi import LandmarkROIExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/UBFC-rPPG")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--window-size", type=int, default=config.CFG.window_size)
    parser.add_argument("--stride", type=int, default=config.CFG.score_interval_frames)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--max-subjects", type=int, default=0)
    return parser.parse_args()


def find_video(subject_dir: Path):
    return next((subject_dir / name for name in ("vid.avi", "video.avi") if (subject_dir / name).exists()), None)


def load_hr_series(gt_path: Path):
    gt = np.loadtxt(gt_path, ndmin=2)
    if gt.shape[0] >= 2:
        return gt[1, :].astype(np.float32)
    if gt.shape[1] >= 2:
        return gt[:, 1].astype(np.float32)
    return gt.reshape(-1).astype(np.float32)


def extract_full_signal(video_path: Path, every_n: int):
    detector = FaceDetector(config.CFG.min_detection_confidence, config.CFG.detection_interval)
    roi_extractor = LandmarkROIExtractor(config.CFG.min_tracking_confidence, interval=config.CFG.landmark_interval)
    cap = cv2.VideoCapture(str(video_path))
    signals = []
    quality = []
    last_sig = np.zeros(9, dtype=np.float32)
    last_quality = 0.0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % every_n == 0:
                bbox, det_score, _ = detector.detect(frame)
                _, sig, roi_quality = roi_extractor.extract(frame, bbox=bbox)
                if sig is not None and roi_quality > 0:
                    last_sig = sig.astype(np.float32)
                    last_quality = float(min(det_score, roi_quality) if det_score > 0 else roi_quality)
            signals.append(last_sig.copy())
            quality.append(last_quality)
            frame_idx += 1
    finally:
        detector.close()
        roi_extractor.close()
        cap.release()
    if not signals:
        return np.zeros((0, 9), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.stack(signals).astype(np.float32), np.asarray(quality, dtype=np.float32)


def window_hr(hr_series, start, end, total_frames):
    if hr_series.size == 0:
        return 75.0
    h0 = int(round(start / max(total_frames, 1) * len(hr_series)))
    h1 = int(round(end / max(total_frames, 1) * len(hr_series)))
    h0 = max(0, min(h0, len(hr_series) - 1))
    h1 = max(h0 + 1, min(h1, len(hr_series)))
    vals = hr_series[h0:h1]
    vals = vals[(vals >= 35) & (vals <= 220)]
    return float(vals.mean()) if vals.size else 75.0


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "windows"
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(root.glob("subject_*"))
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]

    total_saved = 0
    for subject in tqdm(subjects, desc="UBFC subjects"):
        video = find_video(subject)
        gt = subject / "ground_truth.txt"
        if video is None or not gt.exists():
            continue
        sig, quality = extract_full_signal(video, args.every_n)
        if len(sig) < 1:
            continue
        hr_series = load_hr_series(gt)
        max_start = max(0, len(sig) - args.window_size)
        starts = list(range(0, max_start + 1, args.stride)) or [0]
        if starts[-1] != max_start:
            starts.append(max_start)
        for start in starts:
            end = min(start + args.window_size, len(sig))
            out_path = out_dir / f"{subject.name}_s{start:06d}.npz"
            if args.resume and out_path.exists():
                continue
            x = sig[start:end]
            q = quality[start:end]
            if len(x) < args.window_size:
                pad = np.repeat(x[-1:], args.window_size - len(x), axis=0)
                x = np.concatenate([x, pad], axis=0)
            hr = window_hr(hr_series, start, end, len(sig))
            np.savez_compressed(
                out_path,
                x=x.astype(np.float32),
                hr=np.float32(hr),
                subject=subject.name,
                start=np.int32(start),
                mean_quality=np.float32(q.mean() if q.size else 0.0),
                valid_ratio=np.float32((q > 0).mean() if q.size else 0.0),
            )
            total_saved += 1
    print(f"saved_windows={total_saved} out_dir={out_dir}")


if __name__ == "__main__":
    main()

