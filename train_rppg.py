import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import CFG
from data.ubfc_dataset import UBFCRPPGDataset
from demo_runtime import get_device
from models.rppg_tcn import RPPGTCN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/UBFC-rPPG")
    parser.add_argument("--windows-dir", default="", help="Precomputed UBFC window cache directory.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--out", default="checkpoints/rppg_tcn.pt")
    return parser.parse_args()


def make_subject_split(dataset, val_ratio: float, seed: int):
    rng = random.Random(seed)
    subject_to_indices = {}
    for idx, sample in enumerate(dataset.samples):
        if getattr(dataset, "mode", "video") == "cache":
            subject = "_".join(Path(sample).stem.split("_")[:2])
        else:
            subject = Path(sample[0]).parent.name
        subject_to_indices.setdefault(subject, []).append(idx)
    subjects = list(subject_to_indices)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_ratio))) if len(subjects) > 1 else 0
    val_subjects = set(subjects[:n_val])
    train_idx, val_idx = [], []
    for subject, indices in subject_to_indices.items():
        if subject in val_subjects:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    for x, hr in tqdm(loader, desc="rPPG val"):
        x, hr = x.to(device), hr.to(device)
        pred = model(x)["estimated_hr"]
        loss = loss_fn(pred, hr)
        total_loss += float(loss.item())
        preds.extend(pred.detach().cpu().numpy().tolist())
        targets.extend(hr.detach().cpu().numpy().tolist())
    preds = np.asarray(preds, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    mae = float(np.mean(np.abs(preds - targets))) if len(targets) else 0.0
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2))) if len(targets) else 0.0
    return total_loss / max(len(loader), 1), mae, rmse


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    dataset = UBFCRPPGDataset(
        args.data_root,
        CFG.window_size,
        CFG.score_interval_frames,
        windows_dir=args.windows_dir or None,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No UBFC-rPPG samples found under {args.data_root}")
    train_idx, val_idx = make_subject_split(dataset, args.val_ratio, args.seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = RPPGTCN().to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best_mae = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"dataset={len(dataset)} train_subjects={len(train_set)} val_subjects={len(val_set)}")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, hr in tqdm(loader, desc=f"rPPG epoch {epoch + 1}/{args.epochs}"):
            x, hr = x.to(device), hr.to(device)
            pred = model(x)["estimated_hr"]
            loss = loss_fn(pred, hr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        train_loss = total / max(len(loader), 1)
        val_loss, val_mae, val_rmse = validate(model, val_loader, loss_fn, device)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.2f}bpm val_rmse={val_rmse:.2f}bpm"
        )
        if val_mae <= best_mae:
            best_mae = val_mae
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                },
                out_path,
            )
            print(f"saved best {out_path} val_mae={val_mae:.2f}bpm")
    print(f"best_mae={best_mae:.2f}bpm checkpoint={out_path}")


if __name__ == "__main__":
    main()
