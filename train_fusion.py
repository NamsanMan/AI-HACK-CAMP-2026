import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from data.deepfake_dataset import FusionClipDataset
from demo_runtime import get_device
from models.artifact_factory import create_artifact_model
from models.fusion_model import FusionClassifier
from models.rppg_tcn import RPPGTCN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/FaceForensics++")
    parser.add_argument("--frames-dir", default="", help="Preprocessed frames root with real/fake subfolders.")
    parser.add_argument("--rppg-dir", default="", help="Preprocessed rPPG root with real/fake subfolders.")
    parser.add_argument("--rppg-weights", default="")
    parser.add_argument("--artifact-weights", default="")
    parser.add_argument("--artifact-backbone", default=CFG.artifact_backbone, choices=["custom", "rexnet_100", "rexnet_150"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="checkpoints/fusion_model.pt")
    return parser.parse_args()


def maybe_load(model, path, device):
    if path and Path(path).exists():
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)


def main():
    args = parse_args()
    device = get_device()
    dataset = FusionClipDataset(
        args.data_root,
        image_size=CFG.face_crop_size,
        window_size=CFG.window_size,
        window_stride=CFG.score_interval_frames,
        frames_dir=args.frames_dir or None,
        rppg_dir=args.rppg_dir or None,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No matched frame/rPPG samples found under {args.data_root}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    rppg = RPPGTCN().to(device).eval()
    artifact = create_artifact_model(
        args.artifact_backbone,
        pretrained=not args.no_pretrained and not bool(args.artifact_weights),
    ).to(device).eval()
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
    torch.save(rppg.state_dict(), Path(args.out).with_name("rppg_fusion_best.pt"))
    torch.save(artifact.state_dict(), Path(args.out).with_name(f"artifact_{args.artifact_backbone}_fusion_best.pt"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
