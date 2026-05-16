import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from config import CFG
from demo_runtime import get_device, to_face_tensor
from models.artifact_factory import create_artifact_model
from utils.face_detector import FaceDetector
from utils.landmark_roi import LandmarkROIExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--weights", default=CFG.artifact_weights)
    parser.add_argument("--artifact-backbone", default=CFG.artifact_backbone, choices=["custom", "rexnet_100", "rexnet_150"])
    parser.add_argument("--output", default="", help="Optional output video path.")
    parser.add_argument("--display", action="store_true", help="Show live preview window.")
    parser.add_argument("--sample-every", type=int, default=1, help="Run model every N frames; cached score is reused between runs.")
    return parser.parse_args()


def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)


def draw_score(frame, bbox, score, frame_idx):
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 255), 2)
    label = f"artifact fake: {score:.3f}"
    color = (40, 40, 255) if score >= 0.5 else (40, 220, 80)
    cv2.rectangle(frame, (12, 12), (340, 72), (0, 0, 0), -1)
    cv2.putText(frame, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"frame {frame_idx}", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device()
    model = create_artifact_model(args.artifact_backbone, pretrained=False).to(device).eval()
    load_weights(model, args.weights, device)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    detector = FaceDetector(CFG.min_detection_confidence, detection_interval=1)
    last_score = 0.0
    frame_idx = 0
    scores = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            bbox, _, _ = detector.detect(frame)
            if frame_idx % max(1, args.sample_every) == 0 and bbox is not None:
                crop = LandmarkROIExtractor.aligned_face_crop(frame, bbox, CFG.face_crop_size)
                if crop is not None:
                    pred = model(to_face_tensor(crop, device))["artifact_fake"]
                    last_score = float(pred.item())
                    scores.append(last_score)

            draw_score(frame, bbox, last_score, frame_idx)
            if writer is not None:
                writer.write(frame)
            if args.display:
                cv2.imshow("Artifact Inspection", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            frame_idx += 1
    finally:
        detector.close()
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if scores:
        print(
            "artifact scores:",
            f"frames_scored={len(scores)}",
            f"mean={np.mean(scores):.4f}",
            f"min={np.min(scores):.4f}",
            f"max={np.max(scores):.4f}",
        )
    if args.output:
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()

