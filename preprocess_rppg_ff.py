"""
Extract rPPG ROI RGB sequences from FaceForensics++ videos.

Output:
    datasets/FaceForensics++/rppg_v2/
        real/{video_stem}_rppg.npy
        fake/{video_stem}_rppg.npy
        real/{video_stem}_meta.npz
        fake/{video_stem}_meta.npz

Each rPPG array has shape (N, 9):
    R_left, G_left, B_left,
    R_right, G_right, B_right,
    R_forehead, G_forehead, B_forehead

The extraction path mirrors the realtime demo:
    frame -> interval face detection -> landmark/bbox ROI -> RGB signal
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config
from utils.face_detector import FaceDetector
from utils.landmark_roi import LandmarkROIExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=config.FF_ROOT)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--every-n", type=int, default=1, help="Run ROI extraction every N frames.")
    parser.add_argument("--detection-interval", type=int, default=config.CFG.detection_interval)
    parser.add_argument("--landmark-interval", type=int, default=config.CFG.landmark_interval)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--max-videos", type=int, default=0, help="Debug limit per split. 0 means all videos.")
    return parser.parse_args()


def find_split_dir(root: Path, names):
    return next((root / name for name in names if (root / name).is_dir()), None)


def extract_rppg(video_path: Path, detector: FaceDetector, roi_extractor: LandmarkROIExtractor, every_n: int):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 1:
        cap.release()
        return np.zeros((0, 9), dtype=np.float32), np.zeros(0, dtype=np.float32)

    signals = []
    qualities = []
    last_sig = np.zeros(9, dtype=np.float32)
    last_quality = 0.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_idx % every_n == 0:
            bbox, det_score, _ = detector.detect(frame)
            _, sig, roi_quality = roi_extractor.extract(frame, bbox=bbox)
            if sig is not None and roi_quality > 0.0:
                last_sig = sig.astype(np.float32)
                last_quality = float(min(det_score, roi_quality) if det_score > 0 else roi_quality)

        signals.append(last_sig.copy())
        qualities.append(last_quality)
        frame_idx += 1

    cap.release()
    if not signals:
        return np.zeros((0, 9), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.stack(signals, axis=0).astype(np.float32), np.asarray(qualities, dtype=np.float32)


def process_split(src_dir: Path, out_dir: Path, label: str, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(src_dir.glob("*.mp4")) + sorted(src_dir.glob("*.avi"))
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    detector = FaceDetector(config.CFG.min_detection_confidence, args.detection_interval)
    roi_extractor = LandmarkROIExtractor(config.CFG.min_tracking_confidence, interval=args.landmark_interval)
    saved = 0
    skipped = 0
    t0 = time.time()

    for i, video_path in enumerate(tqdm(videos, desc=label), start=1):
        out_path = out_dir / f"{video_path.stem}_rppg.npy"
        meta_path = out_dir / f"{video_path.stem}_meta.npz"
        if args.resume and out_path.exists() and meta_path.exists():
            skipped += 1
            continue

        sig, quality = extract_rppg(video_path, detector, roi_extractor, args.every_n)
        np.save(out_path, sig)
        np.savez_compressed(
            meta_path,
            quality=quality,
            mean_quality=float(quality.mean()) if quality.size else 0.0,
            valid_ratio=float((quality > 0).mean()) if quality.size else 0.0,
            every_n=args.every_n,
        )
        saved += 1

        if i % 100 == 0:
            elapsed = (time.time() - t0) / 60.0
            print(f"{label}: {i}/{len(videos)} saved={saved} skipped={skipped} elapsed={elapsed:.1f}min")

    print(f"{label}: saved={saved} skipped={skipped} out={out_dir}")


def main():
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.out_dir) if args.out_dir else root / "rppg_v2"

    real_src = find_split_dir(root, ("real", "original"))
    fake_src = find_split_dir(root, ("fake", "Deepfakes", "deepfakes"))
    if real_src is None or fake_src is None:
        raise RuntimeError(f"Could not find real/original and fake/Deepfakes folders under {root}")

    process_split(real_src, out_root / "real", "real", args)
    process_split(fake_src, out_root / "fake", "fake", args)
    print(f"done: {out_root}")


if __name__ == "__main__":
    main()

