import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from data.deepfake_dataset import DeepfakeFrameDataset
from demo_runtime import get_device
from models.artifact_cnn import ArtifactCNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/FaceForensics++")
    parser.add_argument("--frames-dir", default="", help="Preprocessed frames root with real/fake subfolders.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="checkpoints/artifact_cnn.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    dataset = DeepfakeFrameDataset(args.data_root, image_size=CFG.face_crop_size, frames_dir=args.frames_dir or None)
    if len(dataset) == 0:
        raise RuntimeError(f"No FaceForensics++ frames/videos found under {args.data_root}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = ArtifactCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for face, label in tqdm(loader, desc=f"artifact epoch {epoch + 1}/{args.epochs}"):
            face, label = face.to(device), label.to(device)
            pred = model(face)["artifact_fake"]
            loss = loss_fn(pred, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch={epoch + 1} loss={total / max(len(loader), 1):.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
