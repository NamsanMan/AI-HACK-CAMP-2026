import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import CFG
from data.deepfake_dataset import FusionClipDataset
from demo_runtime import get_device
from models.artifact_factory import create_artifact_model
from models.fusion_model import FusionClassifier
from models.rppg_tcn import RPPGTCN


def parse_args():
    parser = argparse.ArgumentParser(description="Train temporal fusion classifier from rPPG and artifact branches.")
    parser.add_argument("--data-root", default="datasets/FaceForensics++")
    parser.add_argument("--frames-dir", default="datasets/FaceForensics++/face_frames")
    parser.add_argument("--rppg-dir", default="datasets/FaceForensics++/rppg_v2")
    parser.add_argument("--rppg-weights", default=CFG.rppg_weights)
    parser.add_argument("--artifact-weights", default=CFG.artifact_weights)
    parser.add_argument("--artifact-backbone", default=CFG.artifact_backbone, choices=["custom", "rexnet_100", "rexnet_150"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rppg-lr", type=float, default=1e-5)
    parser.add_argument("--artifact-lr", type=float, default=1e-6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--finetune-branches", action="store_true")
    parser.add_argument("--artifact-aux-weight", type=float, default=0.20)
    parser.add_argument("--liveness-aux-weight", type=float, default=0.10)
    parser.add_argument("--out", default=CFG.fusion_weights)
    return parser.parse_args()


def maybe_load(model, path, device):
    if path and Path(path).exists():
        ckpt = torch.load(path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        return ckpt if isinstance(ckpt, dict) else {"model": state}
    return None


def sample_video_id(sample):
    rppg_path = Path(sample[0])
    stem = rppg_path.stem.removesuffix("_rppg")
    label = int(sample[2])
    return f"{label}:{stem}"


def make_video_split(dataset, val_ratio: float, seed: int):
    rng = random.Random(seed)
    video_to_indices = {}
    video_to_label = {}
    for idx, sample in enumerate(dataset.samples):
        vid = sample_video_id(sample)
        video_to_indices.setdefault(vid, []).append(idx)
        video_to_label[vid] = int(sample[2])

    videos_by_label = {0: [], 1: []}
    for vid, label in video_to_label.items():
        videos_by_label[label].append(vid)

    val_videos = set()
    for label, vids in videos_by_label.items():
        rng.shuffle(vids)
        n_val = max(1, int(round(len(vids) * val_ratio))) if len(vids) > 1 else 0
        val_videos.update(vids[:n_val])

    train_idx, val_idx = [], []
    for vid, indices in video_to_indices.items():
        if vid in val_videos:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx, len(video_to_indices) - len(val_videos), len(val_videos)


def set_branch_train_mode(rppg, artifact, finetune):
    if finetune:
        rppg.train()
        artifact.train()
    else:
        rppg.eval()
        artifact.eval()


def build_optimizer(args, rppg, artifact, fusion):
    groups = [{"params": fusion.parameters(), "lr": args.lr}]
    if args.finetune_branches:
        groups.append({"params": rppg.parameters(), "lr": args.rppg_lr})
        groups.append({"params": artifact.parameters(), "lr": args.artifact_lr})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def compute_loss(outputs, artifact_out, label, args, loss_fn):
    loss = loss_fn(outputs["fake_probability"], label)
    if args.artifact_aux_weight > 0 and "artifact_fake" in artifact_out:
        loss = loss + args.artifact_aux_weight * loss_fn(artifact_out["artifact_fake"], label)
    if args.liveness_aux_weight > 0:
        live_target = 1.0 - label
        loss = loss + args.liveness_aux_weight * loss_fn(outputs["liveness_score"], live_target)
    return loss


def forward_batch(rppg, artifact, fusion, roi, face, quality, finetune):
    if finetune:
        r_out = rppg(roi)
        a_out = artifact(face)
    else:
        with torch.no_grad():
            r_out = rppg(roi)
            a_out = artifact(face)
    fused = fusion(r_out["feature"], a_out["feature"], quality)
    return fused, a_out


@torch.no_grad()
def validate(rppg, artifact, fusion, loader, args, device, loss_fn):
    rppg.eval()
    artifact.eval()
    fusion.eval()
    total = 0.0
    preds, targets, confidences = [], [], []
    for roi, face, quality, label in tqdm(loader, desc="fusion val"):
        roi = roi.to(device)
        face = face.to(device)
        quality = quality.to(device)
        label = label.to(device)
        outputs, artifact_out = forward_batch(rppg, artifact, fusion, roi, face, quality, False)
        loss = compute_loss(outputs, artifact_out, label, args, loss_fn)
        total += float(loss.item())
        preds.extend(outputs["fake_probability"].detach().cpu().numpy().tolist())
        targets.extend(label.detach().cpu().numpy().tolist())
        confidences.extend(outputs["confidence_score"].detach().cpu().numpy().tolist())

    if len(set(targets)) > 1:
        auc = float(roc_auc_score(targets, preds))
    else:
        auc = 0.5
    acc = float(accuracy_score(targets, np.asarray(preds) >= 0.5)) if targets else 0.0
    conf = float(np.mean(confidences)) if confidences else 0.0
    return total / max(len(loader), 1), acc, auc, conf


def save_checkpoint(path, args, epoch, fusion, rppg, artifact, val_auc, val_acc):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": fusion.state_dict(),
            "epoch": epoch,
            "val_auc": val_auc,
            "val_acc": val_acc,
            "artifact_backbone": args.artifact_backbone,
            "finetuned_branches": bool(args.finetune_branches),
        },
        out_path,
    )
    if args.finetune_branches:
        torch.save(
            {"model": rppg.state_dict(), "epoch": epoch, "val_auc": val_auc},
            out_path.with_name("rppg_fusion_best.pt"),
        )
        torch.save(
            {
                "model": artifact.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "artifact_backbone": args.artifact_backbone,
            },
            out_path.with_name(f"artifact_{args.artifact_backbone}_fusion_best.pt"),
        )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()

    dataset = FusionClipDataset(
        args.data_root,
        image_size=CFG.face_crop_size,
        window_size=CFG.window_size,
        window_stride=CFG.score_interval_frames,
        frames_dir=args.frames_dir or None,
        rppg_dir=args.rppg_dir or None,
        return_quality=True,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No matched frame/rPPG samples found under {args.data_root}")

    train_idx, val_idx, train_videos, val_videos = make_video_split(dataset, args.val_ratio, args.seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    rppg = RPPGTCN().to(device)
    artifact = create_artifact_model(
        args.artifact_backbone,
        pretrained=not args.no_pretrained and not Path(args.artifact_weights).exists(),
    ).to(device)
    fusion = FusionClassifier().to(device)
    rppg_ckpt = maybe_load(rppg, args.rppg_weights, device)
    artifact_ckpt = maybe_load(artifact, args.artifact_weights, device)
    resume_ckpt = None
    if args.resume:
        resume_ckpt = maybe_load(fusion, args.resume, device)

    for p in rppg.parameters():
        p.requires_grad = args.finetune_branches
    for p in artifact.parameters():
        p.requires_grad = args.finetune_branches

    opt = build_optimizer(args, rppg, artifact, fusion)
    loss_fn = nn.BCELoss()
    best_auc = float(resume_ckpt.get("val_auc", -1.0)) if isinstance(resume_ckpt, dict) else -1.0

    print(
        f"dataset={len(dataset)} train_samples={len(train_set)} val_samples={len(val_set)} "
        f"train_videos={train_videos} val_videos={val_videos}"
    )
    print(
        f"loaded_rppg={rppg_ckpt is not None} loaded_artifact={artifact_ckpt is not None} "
        f"finetune_branches={args.finetune_branches}"
    )
    if best_auc >= 0.0:
        print(f"resume_best_auc={best_auc:.4f}; will only overwrite if validation improves")

    for epoch in range(args.epochs):
        set_branch_train_mode(rppg, artifact, args.finetune_branches)
        fusion.train()
        total = 0.0
        for roi, face, quality, label in tqdm(loader, desc=f"fusion epoch {epoch + 1}/{args.epochs}"):
            roi = roi.to(device)
            face = face.to(device)
            quality = quality.to(device)
            label = label.to(device)

            outputs, artifact_out = forward_batch(rppg, artifact, fusion, roi, face, quality, args.finetune_branches)
            loss = compute_loss(outputs, artifact_out, label, args, loss_fn)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            opt.step()
            total += float(loss.item())

        train_loss = total / max(len(loader), 1)
        val_loss, val_acc, val_auc, val_conf = validate(rppg, artifact, fusion, val_loader, args, device, loss_fn)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_auc={val_auc:.4f} val_conf={val_conf:.4f}"
        )
        if val_auc >= best_auc:
            best_auc = val_auc
            save_checkpoint(args.out, args, epoch + 1, fusion, rppg, artifact, val_auc, val_acc)
            print(f"saved best {args.out} val_auc={val_auc:.4f}")

    print(f"best_auc={best_auc:.4f} checkpoint={args.out}")


if __name__ == "__main__":
    main()
