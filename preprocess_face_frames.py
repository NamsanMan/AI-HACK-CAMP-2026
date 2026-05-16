"""
Extract face-cropped JPEG frames for artifact/fusion training.

Output:
    datasets/FaceForensics++/face_frames/
        real/{video_stem}_f00000.jpg
        fake/{video_stem}_f00000.jpg

This matches demo inference, where the artifact branch receives an aligned-ish
face crop instead of a full resized video frame.
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
    parser.add_argument("--root", default=config.FF_ROOT)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--n-frames", type=int, default=10)
    parser.add_argument("--size", type=int, default=config.ARTIFACT_IN_SIZE)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--allow-center-fallback", action="store_true", default=True)
    parser.add_argument("--max-videos", type=int, default=0, help="Debug limit per split. 0 means all videos.")
    return parser.parse_args()


def find_split_dir(root: Path, names):
    return next((root / name for name in names if (root / name).is_dir()), None)


def center_square_crop(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    return frame[y1 : y1 + side, x1 : x1 + side]


def extract_face_frames(video_path: Path, out_dir: Path, detector: FaceDetector, n_frames: int, size: int, resume: bool, allow_center_fallback: bool):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 1:
        cap.release()
        return 0, 0

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    saved = 0
    fallback = 0
    for idx in indices:
        out_path = out_dir / f"{video_path.stem}_f{idx:05d}.jpg"
        if resume and out_path.exists():
            saved += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        bbox, _, _ = detector.detect(frame)
        crop = LandmarkROIExtractor.aligned_face_crop(frame, bbox, size=size)
        if crop is None and allow_center_fallback:
            crop = cv2.resize(center_square_crop(frame), (size, size), interpolation=cv2.INTER_AREA)
            fallback += 1
        if crop is None:
            continue

        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    return saved, fallback


def process_split(src_dir: Path, out_dir: Path, label: str, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(src_dir.glob("*.mp4")) + sorted(src_dir.glob("*.avi"))
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    detector = FaceDetector(config.CFG.min_detection_confidence, detection_interval=1)
    total_saved = 0
    total_fallback = 0
    for video_path in tqdm(videos, desc=label):
        saved, fallback = extract_face_frames(
            video_path,
            out_dir,
            detector,
            args.n_frames,
            args.size,
            args.resume,
            args.allow_center_fallback,
        )
        total_saved += saved
        total_fallback += fallback
    print(f"{label}: saved={total_saved} fallback_center_crops={total_fallback}")


def main():
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.out_dir) if args.out_dir else root / "face_frames"

    real_src = find_split_dir(root, ("real", "original"))
    fake_src = find_split_dir(root, ("fake", "Deepfakes", "deepfakes"))
    if real_src is None or fake_src is None:
        raise RuntimeError(f"Could not find real/original and fake/Deepfakes folders under {root}")

    process_split(real_src, out_root / "real", "real", args)
    process_split(fake_src, out_root / "fake", "fake", args)
    print(f"done: {out_root}")


if __name__ == "__main__":
    main()
