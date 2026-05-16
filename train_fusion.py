import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.deepfake_dataset import FusionClipDataset
from demo_runtime import get_device
from models.artifact_cnn import ArtifactCNN
from models.fusion_model import FusionClassifier
from models.rppg_tcn import RPPGTCN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/FaceForensics++")
    parser.add_argument("--rppg-weights", default="")
    parser.add_argument("--artifact-weights", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="checkpoints/fusion_model.pt")
    return parser.parse_args()


def maybe_load(model, path, device):
    if path and Path(path).exists():
        model.load_state_dict(torch.load(path, map_location=device))


def main():
    args = parse_args()
    device = get_device()
    dataset = FusionClipDataset(args.data_root)
    if len(dataset) == 0:
        raise RuntimeError(f"No fusion videos found under {args.data_root}/real and /fake")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    rppg = RPPGTCN().to(device).eval()
    artifact = ArtifactCNN().to(device).eval()
    fusion = FusionClassifier().to(device)
    maybe_load(rppg, args.rppg_weights, device)
    maybe_load(artifact, args.artifact_weights, device)
    opt = torch.optim.AdamW(fusion.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    for epoch in range(args.epochs):
        fusion.train()
        total = 0.0
        for roi, face, label in tqdm(loader, desc=f"fusion epoch {epoch + 1}/{args.epochs}"):
            roi, face, label = roi.to(device), face.to(device), label.to(device)
            with torch.no_grad():
                r_feat = rppg(roi)["feature"]
                a_feat = artifact(face)["feature"]
                quality = torch.ones((roi.shape[0], 3), device=device) * 0.75
            pred = fusion(r_feat, a_feat, quality)["fake_probability"]
            loss = loss_fn(pred, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch={epoch + 1} loss={total / max(len(loader), 1):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(fusion.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

