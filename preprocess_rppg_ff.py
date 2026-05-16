"""
FaceForensics++ 영상에서 rPPG (ROI RGB) 시계열 신호를 추출하여 .npy로 저장.

각 영상 → LandmarkROI → (N, 9) float32 배열 → {video_stem}_rppg.npy

Usage:
    python preprocess_rppg_ff.py                    # 전체 프레임 (느림)
    python preprocess_rppg_ff.py --every_n 2        # 매 2번째 프레임 (권장, ~절반 속도)
    python preprocess_rppg_ff.py --every_n 3        # 매 3번째 프레임 (~1/3 속도)

출력 구조:
    datasets/FaceForensics++/rppg/
        real/  {video_stem}_rppg.npy   shape: (N, 9)
        fake/  {video_stem}_rppg.npy   shape: (N, 9)

처리 후 fusion 학습:
    python train_fusion.py \\
        --frames_dir datasets/FaceForensics++/frames \\
        --rppg_dir   datasets/FaceForensics++/rppg
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils.landmark_roi import LandmarkROI


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root",    type=str, default=config.FF_ROOT,
                   help="FaceForensics++ 루트 폴더")
    p.add_argument("--out_dir", type=str, default=None,
                   help="출력 폴더 (기본: root/rppg)")
    p.add_argument("--every_n", type=int, default=2,
                   help="매 N번째 프레임만 처리 (1=전체, 2=절반). 속도 단축용")
    p.add_argument("--resume",  action="store_true", default=True,
                   help="이미 생성된 .npy 건너뜀 (기본 True)")
    return p.parse_args()


def extract_rppg(video_path: Path, roi: LandmarkROI, every_n: int) -> np.ndarray:
    """
    영상에서 ROI RGB 시계열 신호 추출.
    every_n > 1 이면 중간 프레임은 직전 신호값으로 보간 (rPPG 윈도우 길이 유지).
    Returns: (N_total_frames, 9) float32 — 원래 프레임 수 기준
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return np.zeros((0, 9), dtype=np.float32)

    signals = []
    last_sig = np.zeros(9, dtype=np.float32)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n == 0:
            sig, _, success = roi.process(frame)
            if success and sig is not None:
                last_sig = sig
            # else: keep last_sig (보간)

        signals.append(last_sig.copy())
        frame_idx += 1

    cap.release()
    return np.stack(signals, axis=0) if signals else np.zeros((0, 9), dtype=np.float32)


def process_split(video_dir: Path, out_dir: Path, roi: LandmarkROI,
                  every_n: int, resume: bool, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.avi"))
    print(f"\n[{label}] {len(videos)}개 영상 → {out_dir}")

    skipped = 0
    t0 = time.time()

    for i, vp in enumerate(tqdm(videos, desc=label)):
        out_path = out_dir / f"{vp.stem}_rppg.npy"

        if resume and out_path.exists():
            skipped += 1
            continue

        sig = extract_rppg(vp, roi, every_n)
        np.save(str(out_path), sig)

        # 진행률 출력 (100개마다)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1 - skipped) * (len(videos) - i - 1)
            print(f"  {i+1}/{len(videos)}  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    saved = len(videos) - skipped
    print(f"  → {saved}개 저장, {skipped}개 건너뜀")


def main():
    args = parse_args()
    root    = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "rppg"

    real_src = next(
        (root / n for n in ("real", "original") if (root / n).is_dir()), None
    )
    fake_src = next(
        (root / n for n in ("fake", "Deepfakes", "deepfakes") if (root / n).is_dir()), None
    )

    if real_src is None or fake_src is None:
        print("ERROR: real/fake 폴더를 찾을 수 없습니다.")
        return

    print(f"every_n={args.every_n}  → 처리 속도 약 {args.every_n}배 단축")
    print(f"출력 폴더: {out_dir}")

    roi = LandmarkROI()

    t_start = time.time()
    process_split(real_src, out_dir / "real", roi, args.every_n, args.resume, "real")
    process_split(fake_src, out_dir / "fake", roi, args.every_n, args.resume, "fake")

    total_min = (time.time() - t_start) / 60
    print(f"\n전체 완료: {total_min:.1f}분")
    print(f"\nFusion 학습 명령:")
    print(f"  python train_fusion.py \\")
    print(f"      --frames_dir {root / 'frames'} \\")
    print(f"      --rppg_dir   {out_dir}")


if __name__ == "__main__":
    main()
