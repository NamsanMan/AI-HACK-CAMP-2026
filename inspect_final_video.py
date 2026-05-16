import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

from config import CFG
from data.transforms import normalize_roi_sequence
from demo_runtime import get_device, to_face_tensor
from models.artifact_factory import create_artifact_model
from models.fusion_model import FusionClassifier
from models.rppg_tcn import RPPGTCN
from utils.checkpoints import prefer_fusion_checkpoint
from utils.face_detector import FaceDetector
from utils.landmark_roi import LandmarkROIExtractor
from utils.rppg_signal import TemporalRGBBuffer, estimate_signal_quality
from utils.score_smoothing import RiskHysteresis, ScoreSmoother, risk_color_bgr, risk_label
from utils.tracker import BBoxSmoother


ROI_COLORS = {
    "left_cheek": (80, 220, 255),
    "right_cheek": (120, 240, 150),
    "forehead": (255, 170, 80),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render final fused deepfake/rPPG scores on a video.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/final_video_overlay.mp4")
    parser.add_argument("--csv", default="outputs/final_video_scores.csv")
    parser.add_argument("--rppg-weights", default=CFG.rppg_weights)
    parser.add_argument("--artifact-weights", default=CFG.artifact_weights)
    parser.add_argument("--fusion-weights", default=CFG.fusion_weights)
    parser.add_argument("--prefer-finetuned-branches", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--score-interval-sec", type=float, default=1.0)
    parser.add_argument("--hide-roi", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_state(model, path, device):
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)


def draw_text(frame, text, org, color=(255, 255, 255), scale=0.68, thickness=2):
    x, y = org
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_bar(frame, label, value, x, y, width, color):
    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(frame, (x, y), (x + width, y + 14), (35, 35, 35), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + 14), (100, 100, 100), 1)
    cv2.rectangle(frame, (x, y), (x + int(width * value), y + 14), color, -1)
    draw_text(frame, f"{label}: {value:.3f}", (x, y - 7), color, 0.48, 1)


def draw_rois(frame, rois):
    if not rois:
        return
    overlay = frame.copy()
    for name, pts in rois.items():
        color = ROI_COLORS.get(name, (255, 255, 255))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)


def draw_signal(frame, buffer):
    if len(buffer) < 2:
        return
    x = buffer.padded_array()
    green = x[:, [1, 4, 7]].mean(axis=1)
    green = green - green.mean()
    green = green / (float(np.max(np.abs(green))) + 1e-6)
    h, w = frame.shape[:2]
    left, right = 28, min(w - 28, 390)
    top, bottom = h - 92, h - 32
    cv2.rectangle(frame, (left, top), (right, bottom), (25, 25, 25), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), (85, 85, 85), 1)
    xs = np.linspace(left + 4, right - 4, len(green)).astype(np.int32)
    ys = ((top + bottom) / 2 - green * (bottom - top) * 0.42).astype(np.int32)
    pts = np.stack([xs, ys], axis=1)
    cv2.polylines(frame, [pts], False, (120, 240, 150), 2, cv2.LINE_AA)
    draw_text(frame, "rPPG signal", (left, top - 8), (120, 240, 150), 0.45, 1)


@torch.no_grad()
def run_models(rppg, artifact, fusion, face_bgr, buffer, roi_quality, det_score, fps, device):
    face = to_face_tensor(face_bgr, device)
    art_out = artifact(face)
    artifact_fake = float(art_out["artifact_fake"].item())
    fill = buffer.fill_ratio()

    if len(buffer) >= CFG.min_partial_rppg_frames:
        roi_window = normalize_roi_sequence(buffer.padded_array())
        roi_tensor = torch.from_numpy(roi_window).unsqueeze(0).to(device)
        rppg_out = rppg(roi_tensor)
        signal_quality = estimate_signal_quality(buffer.array(), fps)
        motion_quality = signal_quality["motion_quality"] * min(fill, 1.0)
        quality = torch.tensor(
            [[roi_quality * fill, det_score, motion_quality]],
            dtype=torch.float32,
            device=device,
        )
        fused = fusion(rppg_out["feature"], art_out["feature"], quality)
        fake = float(fused["fake_probability"].item())
        liveness = float(fused["liveness_score"].item())
        confidence = float(fused["confidence_score"].item())
        hr = float(rppg_out["estimated_hr"].item())
        pulse = float(signal_quality["pulse_consistency"])
    else:
        fake = artifact_fake
        liveness = 0.5 * fill
        confidence = min(det_score, roi_quality) * max(0.15, fill)
        hr = 0.0
        pulse = 0.0

    return {
        "fake_probability": float(np.clip(fake, 0.0, 1.0)),
        "artifact_fake": float(np.clip(artifact_fake, 0.0, 1.0)),
        "liveness_score": float(np.clip(liveness, 0.0, 1.0)),
        "confidence_score": float(np.clip(confidence, 0.0, 1.0)),
        "estimated_hr": hr,
        "pulse_consistency": pulse,
        "buffer_fill": fill,
    }


def main():
    args = parse_args()
    device = get_device()
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

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or CFG.target_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    score_every = max(1, int(round(args.score_interval_sec * fps)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = []

    detector = FaceDetector(CFG.min_detection_confidence, CFG.detection_interval)
    tracker = BBoxSmoother()
    roi_extractor = LandmarkROIExtractor(CFG.min_tracking_confidence, interval=CFG.landmark_interval)
    buffer = TemporalRGBBuffer(CFG.window_size)
    risk_smoother = ScoreSmoother(CFG.stable_ema_alpha)
    risk_state = RiskHysteresis()
    bad_frame_count = 0
    scores = {
        "fake_probability": 0.0,
        "artifact_fake": 0.0,
        "liveness_score": 0.0,
        "confidence_score": 0.0,
        "estimated_hr": 0.0,
        "pulse_consistency": 0.0,
        "buffer_fill": 0.0,
    }
    smoothed_fps = 0.0
    prev = time.time()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            inst_fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            smoothed_fps = inst_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * inst_fps

            raw_bbox, det_score, ran_detection = detector.detect(frame)
            if ran_detection and raw_bbox is None:
                tracker.reset()
            bbox = tracker.update(raw_bbox)
            rois, sig9, roi_quality = roi_extractor.extract(frame, bbox=bbox)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                face_ratio = min((x2 - x1) / max(width, 1), (y2 - y1) / max(height, 1))
            else:
                face_ratio = 0.0
            verified_input = (
                bbox is not None
                and det_score >= CFG.verify_min_detection
                and roi_quality >= CFG.verify_min_roi_quality
                and face_ratio >= CFG.verify_min_face_size_ratio
                and sig9 is not None
            )
            if verified_input:
                bad_frame_count = 0
                buffer.append(sig9)
            else:
                bad_frame_count += 1
                scores.update(
                    {
                        "fake_probability": 0.0,
                        "artifact_fake": 0.0,
                        "liveness_score": 0.0,
                        "confidence_score": 0.0,
                        "estimated_hr": 0.0,
                        "pulse_consistency": 0.0,
                        "buffer_fill": buffer.fill_ratio(),
                        "risk_state": "Unverified",
                    }
                )
                if bad_frame_count >= CFG.verify_max_bad_frames:
                    buffer.clear()
                    risk_smoother.value = None
                    risk_state.reset("Unverified")

            if verified_input and frame_idx % score_every == 0:
                face = roi_extractor.aligned_face_crop(frame, bbox, CFG.face_crop_size)
                if face is not None:
                    scores = run_models(rppg, artifact, fusion, face, buffer, roi_quality, det_score, fps, device)
                    input_reliability = min(det_score, roi_quality, scores["buffer_fill"])
                    if input_reliability < CFG.high_risk_min_quality:
                        scores["fake_probability"] = min(scores["fake_probability"], 0.84)
                    scores["fake_probability"] = risk_smoother.update(scores["fake_probability"])
                    scores["risk_state"] = risk_state.update(
                        scores["fake_probability"],
                        verified=True,
                        allow_high=input_reliability >= CFG.high_risk_min_quality,
                    )

            state = scores.get("risk_state", "Low")
            state_color = risk_color_bgr(state)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, 2)
                draw_text(frame, f"face {det_score:.2f}", (x1, max(24, y1 - 8)), state_color, 0.52, 1)
            if not args.hide_roi:
                draw_rois(frame, rois)
                draw_signal(frame, buffer)

            draw_text(frame, f"Risk Level: {risk_label(state)}", (28, 36), state_color, 0.78, 2)
            draw_text(frame, f"Risk Score: {scores['fake_probability']:.3f}", (28, 68), state_color, 0.72, 2)
            draw_bar(frame, "Liveness", scores["liveness_score"], 28, 108, 230, (120, 240, 150))
            hr_text = "HR: not reliable" if state in ("High", "Unverified") else f"HR: {scores['estimated_hr']:.1f} bpm"
            draw_text(frame, hr_text, (28, 158), (245, 245, 245), 0.62, 2)
            draw_text(frame, f"FPS: {smoothed_fps:.1f}", (28, 188), (205, 215, 230), 0.52, 1)
            if args.debug:
                draw_text(
                    frame,
                    f"artifact={scores['artifact_fake']:.3f} conf={scores['confidence_score']:.3f} pulse={scores['pulse_consistency']:.2f} buffer={scores['buffer_fill']:.2f}",
                    (28, 216),
                    (205, 215, 230),
                    0.48,
                    1,
                )

            csv_rows.append(
                {
                    "frame": frame_idx,
                    "time_sec": frame_idx / max(float(fps), 1.0),
                    "fake_probability": scores["fake_probability"],
                    "artifact_fake": scores["artifact_fake"],
                    "liveness_score": scores["liveness_score"],
                    "confidence_score": scores["confidence_score"],
                    "estimated_hr": scores["estimated_hr"],
                    "pulse_consistency": scores["pulse_consistency"],
                    "buffer_fill": scores["buffer_fill"],
                    "risk_state": state,
                    "roi_quality": roi_quality,
                    "det_score": det_score,
                }
            )

            writer.write(frame)
            if args.display:
                cv2.imshow("final deepfake/rPPG inspection", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()
        detector.close()
        roi_extractor.close()
        if args.display:
            cv2.destroyAllWindows()

    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(csv_rows)

    print(f"saved_video={out_path}")
    print(f"saved_csv={csv_path}")
    print(f"rppg_weights={rppg_weights}")
    print(f"artifact_weights={artifact_weights}")
    print(f"fusion_weights={args.fusion_weights}")


if __name__ == "__main__":
    main()
