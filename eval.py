import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from data.deepfake_dataset import FusionClipDataset
from demo_runtime import get_device
from models.artifact_cnn import ArtifactCNN
from models.fusion_model import FusionClassifier
from models.rppg_tcn import RPPGTCN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/Celeb-DF-v2")
    parser.add_argument("--frames-dir", default="")
    parser.add_argument("--rppg-dir", default="")
    parser.add_argument("--fusion-weights", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    dataset = FusionClipDataset(
        args.data_root,
        window_size=CFG.window_size,
        frames_dir=args.frames_dir or None,
        rppg_dir=args.rppg_dir or None,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No matched evaluation frame/rPPG samples found under {args.data_root}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rppg = RPPGTCN().to(device).eval()
    artifact = ArtifactCNN().to(device).eval()
    fusion = FusionClassifier().to(device).eval()
    if args.fusion_weights:
        fusion.load_state_dict(torch.load(args.fusion_weights, map_location=device))

    correct, total = 0, 0
    with torch.no_grad():
        for roi, face, label in tqdm(loader, desc="eval"):
            roi, face, label = roi.to(device), face.to(device), label.to(device)
            quality = torch.ones((roi.shape[0], 3), device=device) * 0.75
            out = fusion(rppg(roi)["feature"], artifact(face)["feature"], quality)
            pred = (out["fake_probability"] >= 0.5).float()
            correct += int((pred == label).sum().item())
            total += int(label.numel())
    print(f"accuracy={correct / max(total, 1):.4f} samples={total}")


if __name__ == "__main__":
    main()
