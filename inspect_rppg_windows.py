import argparse
import csv
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

from config import CFG
from data.transforms import normalize_roi_sequence
from data.ubfc_dataset import UBFCRPPGDataset
from demo_runtime import get_device
from models.rppg_tcn import RPPGTCN
from train_rppg import make_subject_split


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize rPPG HR predictions against UBFC ground truth windows.")
    parser.add_argument("--data-root", default="datasets/UBFC-rPPG")
    parser.add_argument("--windows-dir", default="datasets/UBFC-rPPG/windows")
    parser.add_argument("--weights", default="checkpoints/rppg_tcn.pt")
    parser.add_argument("--subject", default="", help="Example: subject_01. Empty picks the first validation subject.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means all windows.")
    parser.add_argument("--ema-alpha", type=float, default=0.35, help="Smoothing for the subject timeline.")
    parser.add_argument("--output", default="outputs/rppg_inspect.png")
    parser.add_argument("--csv", default="outputs/rppg_inspect.csv")
    return parser.parse_args()


def subject_from_path(path: Path) -> str:
    parts = path.stem.split("_")
    return "_".join(parts[:2])


def window_start_from_path(path: Path) -> int:
    token = path.stem.split("_")[-1]
    return int(token[1:]) if token.startswith("s") and token[1:].isdigit() else 0


def load_model(weights: str, device: torch.device):
    model = RPPGTCN().to(device).eval()
    ckpt = torch.load(weights, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


@torch.no_grad()
def predict_windows(model, paths, device, batch_size=128):
    rows = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        xs, hrs = [], []
        for path in batch_paths:
            data = np.load(path)
            xs.append(normalize_roi_sequence(data["x"].astype(np.float32)))
            hrs.append(float(data["hr"]))
        x = torch.from_numpy(np.stack(xs)).to(device)
        pred = model(x)["estimated_hr"].detach().cpu().numpy()
        for path, gt, pr in zip(batch_paths, hrs, pred):
            rows.append(
                {
                    "path": str(path),
                    "subject": subject_from_path(path),
                    "start": window_start_from_path(path),
                    "gt_hr": float(gt),
                    "pred_hr": float(pr),
                    "abs_error": abs(float(pr) - float(gt)),
                }
            )
    return rows


def ema(values, alpha):
    out = []
    last = None
    for value in values:
        last = value if last is None else alpha * value + (1.0 - alpha) * last
        out.append(last)
    return np.asarray(out, dtype=np.float32)


def draw_axes(img, left, top, right, bottom, title, x_label="", y_label="HR BPM"):
    cv2.rectangle(img, (left, top), (right, bottom), (70, 70, 70), 1)
    cv2.putText(img, title, (left, top - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(img, y_label, (left, top - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    if x_label:
        cv2.putText(img, x_label, (left, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def scale_points(xs, ys, left, top, right, bottom, x_min, x_max, y_min, y_max):
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    x_span = max(float(x_max - x_min), 1e-6)
    y_span = max(float(y_max - y_min), 1e-6)
    px = left + ((xs - x_min) / x_span * (right - left)).astype(np.int32)
    py = bottom - ((ys - y_min) / y_span * (bottom - top)).astype(np.int32)
    return list(zip(px.tolist(), py.tolist()))


def draw_polyline(img, pts, color, thickness=2):
    if len(pts) >= 2:
        cv2.polylines(img, [np.asarray(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_report(rows, subject_rows, output):
    width, height = 1280, 860
    img = np.full((height, width, 3), (24, 26, 30), dtype=np.uint8)
    gt = np.asarray([r["gt_hr"] for r in rows], dtype=np.float32)
    pred = np.asarray([r["pred_hr"] for r in rows], dtype=np.float32)
    abs_err = np.abs(pred - gt)
    mae = float(abs_err.mean()) if len(abs_err) else 0.0
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2))) if len(abs_err) else 0.0
    corr = float(np.corrcoef(gt, pred)[0, 1]) if len(rows) > 1 and np.std(gt) > 0 and np.std(pred) > 0 else 0.0

    cv2.putText(img, "rPPG HR Prediction Inspection", (36, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"windows={len(rows)}  MAE={mae:.2f} bpm  RMSE={rmse:.2f} bpm  Corr={corr:.3f}",
        (36, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (205, 215, 230),
        2,
        cv2.LINE_AA,
    )

    left, top, right, bottom = 70, 145, 600, 555
    y_min = max(35.0, float(min(gt.min(), pred.min()) - 8.0)) if len(rows) else 40.0
    y_max = min(190.0, float(max(gt.max(), pred.max()) + 8.0)) if len(rows) else 180.0
    draw_axes(img, left, top, right, bottom, "Validation Scatter: GT vs Pred", "GT HR")
    for bpm in range(int(y_min // 20 * 20), int(y_max) + 20, 20):
        y = int(bottom - (bpm - y_min) / max(y_max - y_min, 1e-6) * (bottom - top))
        cv2.line(img, (left, y), (right, y), (40, 42, 48), 1)
        cv2.putText(img, str(bpm), (24, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    diag = scale_points([y_min, y_max], [y_min, y_max], left, top, right, bottom, y_min, y_max, y_min, y_max)
    draw_polyline(img, diag, (90, 160, 255), 2)
    scatter = scale_points(gt, pred, left, top, right, bottom, y_min, y_max, y_min, y_max)
    for x, y in scatter:
        cv2.circle(img, (x, y), 2, (110, 225, 180), -1, cv2.LINE_AA)

    s_gt = np.asarray([r["gt_hr"] for r in subject_rows], dtype=np.float32)
    s_pred_raw = np.asarray([r["pred_hr"] for r in subject_rows], dtype=np.float32)
    s_pred = ema(s_pred_raw, 0.35)
    starts = np.asarray([r["start"] for r in subject_rows], dtype=np.float32)
    times = starts / max(float(CFG.target_fps), 1.0)
    left2, top2, right2, bottom2 = 690, 145, 1230, 555
    title = f"Subject Timeline: {subject_rows[0]['subject']}" if subject_rows else "Subject Timeline"
    draw_axes(img, left2, top2, right2, bottom2, title, "seconds")
    if len(subject_rows):
        sy_min = max(35.0, float(min(s_gt.min(), s_pred_raw.min()) - 8.0))
        sy_max = min(190.0, float(max(s_gt.max(), s_pred_raw.max()) + 8.0))
        x_min, x_max = float(times.min()), float(times.max())
        for bpm in range(int(sy_min // 20 * 20), int(sy_max) + 20, 20):
            y = int(bottom2 - (bpm - sy_min) / max(sy_max - sy_min, 1e-6) * (bottom2 - top2))
            cv2.line(img, (left2, y), (right2, y), (40, 42, 48), 1)
            cv2.putText(img, str(bpm), (left2 - 46, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        draw_polyline(img, scale_points(times, s_gt, left2, top2, right2, bottom2, x_min, x_max, sy_min, sy_max), (80, 220, 255), 2)
        draw_polyline(img, scale_points(times, s_pred_raw, left2, top2, right2, bottom2, x_min, x_max, sy_min, sy_max), (95, 110, 235), 1)
        draw_polyline(img, scale_points(times, s_pred, left2, top2, right2, bottom2, x_min, x_max, sy_min, sy_max), (125, 245, 160), 3)
        cv2.putText(img, "GT", (left2 + 18, bottom2 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Pred raw", (left2 + 78, bottom2 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (95, 110, 235), 2, cv2.LINE_AA)
        cv2.putText(img, "Pred EMA", (left2 + 190, bottom2 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (125, 245, 160), 2, cv2.LINE_AA)

    worst = sorted(rows, key=lambda r: r["abs_error"], reverse=True)[:8]
    cv2.putText(img, "Largest Errors", (70, 645), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    for i, r in enumerate(worst):
        text = f"{i + 1}. {r['subject']} start={r['start']:05d}  GT={r['gt_hr']:.1f}  Pred={r['pred_hr']:.1f}  Err={r['abs_error']:.1f}"
        cv2.putText(img, text, (70, 682 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (205, 215, 230), 1, cv2.LINE_AA)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), img)


def main():
    args = parse_args()
    random.seed(args.seed)
    dataset = UBFCRPPGDataset(args.data_root, CFG.window_size, CFG.score_interval_frames, windows_dir=args.windows_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No windows found in {args.windows_dir}")

    _, val_idx = make_subject_split(dataset, args.val_ratio, args.seed)
    paths = [Path(dataset.samples[i]) for i in val_idx] if val_idx else [Path(p) for p in dataset.samples]
    if args.max_windows > 0:
        paths = paths[: args.max_windows]

    device = get_device()
    model = load_model(args.weights, device)
    rows = predict_windows(model, paths, device)

    subjects = sorted({r["subject"] for r in rows})
    chosen = args.subject or (subjects[0] if subjects else "")
    subject_rows = sorted([r for r in rows if r["subject"] == chosen], key=lambda r: r["start"])
    if not subject_rows and subjects:
        subject_rows = sorted([r for r in rows if r["subject"] == subjects[0]], key=lambda r: r["start"])

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "subject", "start", "gt_hr", "pred_hr", "abs_error"])
            writer.writeheader()
            writer.writerows(rows)

    draw_report(rows, subject_rows, args.output)
    mae = np.mean([r["abs_error"] for r in rows]) if rows else 0.0
    print(f"saved {args.output}")
    if args.csv:
        print(f"saved {args.csv}")
    print(f"windows={len(rows)} subject={subject_rows[0]['subject'] if subject_rows else 'none'} mae={mae:.2f}bpm")


if __name__ == "__main__":
    main()
