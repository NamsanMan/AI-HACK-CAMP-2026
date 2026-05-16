import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from data.ubfc_dataset import UBFCRPPGDataset
from demo_runtime import get_device
from models.rppg_tcn import RPPGTCN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/UBFC-rPPG")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="checkpoints/rppg_tcn.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    dataset = UBFCRPPGDataset(args.data_root, CFG.window_size, CFG.score_interval_frames)
    if len(dataset) == 0:
        raise RuntimeError(f"No UBFC-rPPG samples found under {args.data_root}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = RPPGTCN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
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
        print(f"epoch={epoch + 1} loss={total / max(len(loader), 1):.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
