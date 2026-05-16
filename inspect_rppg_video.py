import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

from config import CFG
from data.transforms import normalize_roi_sequence
from demo_runtime import get_device
from models.rppg_tcn import RPPGTCN
from utils.face_detector import FaceDetector
from utils.landmark_roi import LandmarkROIExtractor
from utils.rppg_signal import TemporalRGBBuffer, estimate_signal_quality


COLORS = {
    "left_cheek": (80, 220, 255),
    "right_cheek": (120, 240, 150),
    "forehead": (255, 170, 80),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay face bbox, rPPG ROIs, and HR estimates on a video.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--weights", default="checkpoints/rppg_tcn.pt")
    parser.add_argument("--output", default="outputs/rppg_roi_inspect.mp4")
    parser.add_argument("--gt", default="", help="Optional UBFC ground_truth.txt.")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--score-interval-sec", type=float, default=1.0)
    parser.add_argument("--no-model", action="store_true", help="Only draw ROI quality without loading rPPG model.")
    parser.add_argument("--debug-fft", action="store_true", help="Show raw FFT HR and pulse consistency debug values.")
    return parser.parse_args()


def load_gt_hr(path):
    if not path:
        return None
    gt_path = Path(path)
    if not gt_path.exists():
        return None
    values = np.loadtxt(gt_path, ndmin=2)
    if values.shape[0] >= 2:
        hr = values[1, :]
    elif values.shape[1] >= 2:
        hr = values[:, 1]
    else:
        hr = values.reshape(-1)
    hr = hr[(hr >= 35) & (hr <= 220)]
    return float(hr.mean()) if hr.size else None


def load_model(weights, device):
    model = RPPGTCN().to(device).eval()
    ckpt = torch.load(weights, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def draw_label(frame, text, x, y, color=(245, 245, 245), scale=0.62, thickness=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_rois(frame, rois):
    if not rois:
        return
    overlay = frame.copy()
    for name, pts in rois.items():
        color = COLORS.get(name, (255, 255, 255))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        center = pts.mean(axis=0).astype(int)
        draw_label(frame, name.replace("_", " "), int(center[0]) - 48, int(center[1]), color, 0.45, 1)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)


def draw_signal_strip(frame, buffer):
    if len(buffer) < 2:
        return
    x = buffer.padded_array()
    green = x[:, [1, 4, 7]].mean(axis=1)
    green = green - green.mean()
    denom = float(np.max(np.abs(green))) + 1e-6
    green = green / denom
    h, w = frame.shape[:2]
    left, right = 30, min(w - 30, 430)
    top, bottom = h - 110, h - 35
    cv2.rectangle(frame, (left, top), (right, bottom), (30, 30, 30), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), (85, 85, 85), 1)
    xs = np.linspace(left + 4, right - 4, len(green)).astype(np.int32)
    ys = ((top + bottom) / 2 - green * (bottom - top) * 0.42).astype(np.int32)
    pts = np.stack([xs, ys], axis=1)
    cv2.polylines(frame, [pts], False, (120, 240, 150), 2, cv2.LINE_AA)
    draw_label(frame, "rPPG green signal", left + 8, top - 10, (120, 240, 150), 0.46, 1)


@torch.no_grad()
def predict_hr(model, buffer, device):
    x = normalize_roi_sequence(buffer.padded_array())
    tensor = torch.from_numpy(x).unsqueeze(0).to(device)
    out = model(tensor)
    return float(out["estimated_hr"].item()), float(out["rppg_liveness"].item())


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or CFG.target_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    device = get_device()
    model = None if args.no_model else load_model(args.weights, device)
    gt_hr = load_gt_hr(args.gt)
    face_detector = FaceDetector(CFG.min_detection_confidence, CFG.detection_interval)
    roi_extractor = LandmarkROIExtractor(CFG.min_tracking_confidence, interval=CFG.landmark_interval)
    buffer = TemporalRGBBuffer(CFG.window_size)
    score_every = max(1, int(round(args.score_interval_sec * fps)))
    pred_hr = None
    live_score = None

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bbox, det_score, ran_detection = face_detector.detect(frame)
            rois, sig9, roi_quality = roi_extractor.extract(frame, bbox)
            if sig9 is not None:
                buffer.append(sig9)

            if model is not None and len(buffer) >= CFG.min_partial_rppg_frames and frame_idx % score_every == 0:
                pred_hr, live_score = predict_hr(model, buffer, device)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                color = (80, 220, 255) if det_score > 0 else (120, 120, 120)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, f"face {det_score:.2f}", x1, max(24, y1 - 8), color, 0.52, 1)
            draw_rois(frame, rois)
            draw_signal_strip(frame, buffer)

            q = estimate_signal_quality(buffer.padded_array(), fps) if len(buffer) >= CFG.min_partial_rppg_frames else None
            fill = buffer.fill_ratio()
            draw_label(frame, f"ROI quality {roi_quality:.2f}  buffer {fill * 100:.0f}%", 30, 34, (245, 245, 245), 0.68, 2)
            draw_label(frame, f"landmark every {CFG.landmark_interval}f  detect every {CFG.detection_interval}f", 30, 64, (205, 215, 230), 0.52, 1)
            if pred_hr is not None:
                gt_text = f"  GT {gt_hr:.1f}" if gt_hr is not None else ""
                draw_label(frame, f"Pred HR {pred_hr:.1f} bpm{gt_text}", 30, 96, (120, 240, 150), 0.7, 2)
            if live_score is not None:
                draw_label(frame, f"rPPG live {live_score:.2f}", 30, 126, (120, 240, 150), 0.58, 1)
            if q is not None and args.debug_fft:
                draw_label(
                    frame,
                    f"FFT HR {q['estimated_hr']:.1f}  pulse consistency {q['pulse_consistency']:.2f}",
                    30,
                    154,
                    (180, 210, 255),
                    0.52,
                    1,
                )
            if ran_detection:
                cv2.circle(frame, (width - 34, 34), 8, (80, 220, 255), -1, cv2.LINE_AA)

            writer.write(frame)
            if args.display:
                cv2.imshow("rPPG ROI inspection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()
        face_detector.close()
        roi_extractor.close()
        if args.display:
            cv2.destroyAllWindows()

    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
