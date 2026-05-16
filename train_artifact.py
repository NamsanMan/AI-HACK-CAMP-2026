import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import CFG
from data.deepfake_dataset import DeepfakeFrameDataset
from demo_runtime import get_device
from models.artifact_factory import create_artifact_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/FaceForensics++")
    parser.add_argument("--frames-dir", default="", help="Preprocessed frames root with real/fake subfolders.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--artifact-backbone", default=CFG.artifact_backbone, choices=["custom", "rexnet_100", "rexnet_150"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", default="", help="Checkpoint path to resume model weights from.")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def video_id_from_sample(sample):
    path = Path(sample[0])
    return path.name.split("_f")[0]


def make_video_split(dataset, val_ratio: float, seed: int):
    rng = random.Random(seed)
    video_to_indices = {}
    video_to_label = {}
    for idx, sample in enumerate(dataset.samples):
        vid = video_id_from_sample(sample)
        video_to_indices.setdefault(vid, []).append(idx)
        video_to_label[vid] = float(sample[1])

    by_label = {0.0: [], 1.0: []}
    for vid, label in video_to_label.items():
        by_label.setdefault(label, []).append(vid)

    train_idx, val_idx = [], []
    for label, vids in by_label.items():
        rng.shuffle(vids)
        n_val = max(1, int(round(len(vids) * val_ratio))) if len(vids) > 1 else 0
        val_vids = set(vids[:n_val])
        for vid in vids:
            target = val_idx if vid in val_vids else train_idx
            target.extend(video_to_indices[vid])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def binary_metrics(probs, labels):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    pred = (probs >= 0.5).astype(np.float32)
    acc = float((pred == labels).mean()) if labels.size else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = 0.0
    return acc, auc


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    probs, labels_all = [], []
    for face, label in tqdm(loader, desc="artifact val"):
        face, label = face.to(device), label.to(device)
        pred = model(face)["artifact_fake"]
        loss = loss_fn(pred, label)
        total_loss += float(loss.item())
        probs.extend(pred.detach().cpu().numpy().tolist())
        labels_all.extend(label.detach().cpu().numpy().tolist())
    acc, auc = binary_metrics(probs, labels_all)
    return total_loss / max(len(loader), 1), acc, auc


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    dataset = DeepfakeFrameDataset(args.data_root, image_size=CFG.face_crop_size, frames_dir=args.frames_dir or None)
    if len(dataset) == 0:
        raise RuntimeError(f"No FaceForensics++ frames/videos found under {args.data_root}")
    train_idx, val_idx = make_video_split(dataset, args.val_ratio, args.seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = create_artifact_model(
        args.artifact_backbone,
        pretrained=not args.no_pretrained,
    ).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
    if hasattr(model, "optimizer_param_groups"):
        opt = torch.optim.AdamW(model.optimizer_param_groups(args.backbone_lr, args.lr))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()
    best_auc = -1.0
    best_path = Path(args.out or f"checkpoints/artifact_{args.artifact_backbone}.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"dataset={len(dataset)} train_frames={len(train_set)} val_frames={len(val_set)}")
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
        train_loss = total / max(len(loader), 1)
        val_loss, val_acc, val_auc = validate(model, val_loader, loss_fn, device)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
        )
        if val_auc >= best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "artifact_backbone": args.artifact_backbone,
                },
                best_path,
            )
            print(f"saved best {best_path} val_auc={val_auc:.4f}")
    print(f"best_auc={best_auc:.4f} checkpoint={best_path}")


if __name__ == "__main__":
    main()
