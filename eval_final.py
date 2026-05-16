import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from data.deepfake_dataset import FusionClipDataset
from demo_runtime import get_device
from models.artifact_factory import create_artifact_model
from models.fusion_model import FusionClassifier
from models.rppg_tcn import RPPGTCN
from utils.checkpoints import prefer_fusion_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Final evaluation for fused deepfake/rPPG model.")
    parser.add_argument("--data-root", default="datasets/FaceForensics++")
    parser.add_argument("--frames-dir", default="datasets/FaceForensics++/face_frames")
    parser.add_argument("--rppg-dir", default="datasets/FaceForensics++/rppg_v2")
    parser.add_argument("--rppg-weights", default=CFG.rppg_weights)
    parser.add_argument("--artifact-weights", default=CFG.artifact_weights)
    parser.add_argument("--fusion-weights", default=CFG.fusion_weights)
    parser.add_argument("--prefer-finetuned-branches", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--csv", default="outputs/final_eval_predictions.csv")
    return parser.parse_args()


def load_state(model, path, device, required=True):
    if not path or not Path(path).exists():
        if required:
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return False
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return True


def auc_or_half(y_true, y_score):
    if len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def sample_meta(sample):
    rppg_path = Path(sample[0])
    label = int(sample[2])
    start = int(sample[3])
    stem = rppg_path.stem.removesuffix("_rppg")
    split = "fake" if label == 1 else "real"
    return split, stem, start


def summarize(y_true, y_score, threshold):
    y_pred = (np.asarray(y_score) >= threshold).astype(np.int32)
    y_true_arr = np.asarray(y_true).astype(np.int32)
    acc = float(accuracy_score(y_true_arr, y_pred)) if len(y_true_arr) else 0.0
    auc = auc_or_half(y_true_arr.tolist(), list(y_score))
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    return {
        "acc": acc,
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    args = parse_args()
    device = get_device()
    dataset = FusionClipDataset(
        args.data_root,
        image_size=CFG.face_crop_size,
        window_size=CFG.window_size,
        window_stride=CFG.score_interval_frames,
        frames_dir=args.frames_dir,
        rppg_dir=args.rppg_dir,
        return_quality=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("No matched fusion samples found. Check --frames-dir and --rppg-dir.")

    rppg_weights = args.rppg_weights
    artifact_weights = args.artifact_weights
    if args.prefer_finetuned_branches:
        rppg_weights = prefer_fusion_checkpoint(rppg_weights, "rppg_fusion_best.pt")
        artifact_weights = prefer_fusion_checkpoint(artifact_weights, f"artifact_{CFG.artifact_backbone}_fusion_best.pt")

    rppg = RPPGTCN().to(device).eval()
    artifact = create_artifact_model(CFG.artifact_backbone, pretrained=False).to(device).eval()
    fusion = FusionClassifier().to(device).eval()
    load_state(rppg, rppg_weights, device)
    load_state(artifact, artifact_weights, device)
    load_state(fusion, args.fusion_weights, device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows = []
    offset = 0
    with torch.no_grad():
        for roi, face, quality, label in tqdm(loader, desc="final eval"):
            batch_size = int(label.shape[0])
            roi = roi.to(device)
            face = face.to(device)
            quality = quality.to(device)
            label = label.to(device)
            r_out = rppg(roi)
            a_out = artifact(face)
            out = fusion(r_out["feature"], a_out["feature"], quality)

            fake = out["fake_probability"].detach().cpu().numpy()
            live = out["liveness_score"].detach().cpu().numpy()
            conf = out["confidence_score"].detach().cpu().numpy()
            artifact_fake = a_out["artifact_fake"].detach().cpu().numpy()
            hr = r_out["estimated_hr"].detach().cpu().numpy()
            labels = label.detach().cpu().numpy()

            for i in range(batch_size):
                split, stem, start = sample_meta(dataset.samples[offset + i])
                rows.append(
                    {
                        "split": split,
                        "video": stem,
                        "start": start,
                        "label": int(labels[i]),
                        "fake_probability": float(fake[i]),
                        "artifact_fake": float(artifact_fake[i]),
                        "liveness_score": float(live[i]),
                        "confidence_score": float(conf[i]),
                        "estimated_hr": float(hr[i]),
                    }
                )
            offset += batch_size

    y_true = [r["label"] for r in rows]
    y_score = [r["fake_probability"] for r in rows]
    sample_metrics = summarize(y_true, y_score, args.threshold)

    video_scores = defaultdict(list)
    video_labels = {}
    for row in rows:
        key = (row["split"], row["video"])
        video_scores[key].append(row["fake_probability"])
        video_labels[key] = row["label"]
    video_y_true = []
    video_y_score = []
    for key, scores in video_scores.items():
        video_y_true.append(video_labels[key])
        video_y_score.append(float(np.mean(scores)))
    video_metrics = summarize(video_y_true, video_y_score, args.threshold)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "sample_level "
        f"acc={sample_metrics['acc']:.4f} auc={sample_metrics['auc']:.4f} "
        f"tn={sample_metrics['tn']} fp={sample_metrics['fp']} fn={sample_metrics['fn']} tp={sample_metrics['tp']}"
    )
    print(
        "video_level  "
        f"acc={video_metrics['acc']:.4f} auc={video_metrics['auc']:.4f} "
        f"tn={video_metrics['tn']} fp={video_metrics['fp']} fn={video_metrics['fn']} tp={video_metrics['tp']}"
    )
    print(f"saved_csv={csv_path}")
    print(f"rppg_weights={rppg_weights}")
    print(f"artifact_weights={artifact_weights}")
    print(f"fusion_weights={args.fusion_weights}")


if __name__ == "__main__":
    main()
