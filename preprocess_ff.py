"""
FaceForensics++ 영상에서 프레임을 JPEG로 사전 추출.
추출된 이미지는 datasets/FaceForensics++/frames/{real,fake}/ 에 저장.

Usage:
    python preprocess_ff.py
    python preprocess_ff.py --n_frames 10 --size 96
"""

import argparse
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root",      type=str, default=config.FF_ROOT)
    p.add_argument("--n_frames",  type=int, default=10,
                   help="영상당 추출 프레임 수")
    p.add_argument("--size",      type=int, default=config.ARTIFACT_IN_SIZE,
                   help="저장 이미지 크기 (정사각형)")
    p.add_argument("--out_dir",   type=str, default=None,
                   help="출력 폴더 (기본: root/frames)")
    return p.parse_args()


def extract_frames(video_path: Path, out_dir: Path, n_frames: int, size: int):
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return 0

    # 균등 간격으로 n_frames개 인덱스 선택
    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_resized = cv2.resize(frame, (size, size))
        fname = out_dir / f"{video_path.stem}_f{idx:05d}.jpg"
        cv2.imwrite(str(fname), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    return saved


def process_split(video_dir: Path, out_dir: Path, n_frames: int, size: int, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.avi"))
    print(f"\n[{label}] {len(videos)}개 영상 → {out_dir}")

    total_frames = 0
    for vp in tqdm(videos, desc=label):
        total_frames += extract_frames(vp, out_dir, n_frames, size)

    print(f"  → 총 {total_frames}개 이미지 저장 완료")
    return total_frames


def main():
    args    = parse_args()
    root    = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "frames"

    # real 폴더 자동 탐지
    real_src = next((root / n for n in ("real", "original") if (root / n).is_dir()), None)
    fake_src = next((root / n for n in ("fake", "Deepfakes", "deepfakes") if (root / n).is_dir()), None)

    if real_src is None or fake_src is None:
        print("ERROR: real/fake 폴더를 찾을 수 없습니다.")
        return

    process_split(real_src, out_dir / "real", args.n_frames, args.size, "real")
    process_split(fake_src, out_dir / "fake", args.n_frames, args.size, "fake")

    print(f"\n완료. 학습 시 --frames_dir {out_dir} 옵션을 사용하세요.")
    print(f"  python train_artifact.py --frames_dir {out_dir}")


if __name__ == "__main__":
    main()
