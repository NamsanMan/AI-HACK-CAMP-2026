"""
Export the trained neural pipeline to a deployment-oriented ONNX model.

The exported ONNX intentionally contains only the neural inference graph:

    rppg_window  (B, T, 9)       raw ROI RGB means, usually 0..1
    face_crop    (B, 3, 96, 96)  aligned RGB face crop, 0..1
    quality      (B, 3)          [roi_quality, detection_score, motion_quality]

It does not include webcam capture, face detection, MediaPipe landmarks, ROI
polygon extraction, temporal buffering, score smoothing, or UI thresholding.
Those pieces should live in the realtime plugin/runtime so they can use native
video APIs and keep latency low.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from config import CFG
from models.artifact_factory import create_artifact_model
from models.fusion_model import FusionClassifier
from models.rppg_tcn import RPPGTCN
from utils.checkpoints import prefer_fusion_checkpoint


class DeploymentPipelineONNX(nn.Module):
    """Tensor-only ONNX wrapper. No dict outputs, no Python post-processing."""

    def __init__(self, rppg_model: nn.Module, artifact_model: nn.Module, fusion_model: nn.Module):
        super().__init__()
        self.rppg = rppg_model
        self.artifact = artifact_model
        self.fusion = fusion_model

    @staticmethod
    def normalize_rppg_window(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std

    def forward(self, rppg_window: torch.Tensor, face_crop: torch.Tensor, quality: torch.Tensor):
        rppg_window = self.normalize_rppg_window(rppg_window)
        rppg_out = self.rppg(rppg_window)
        artifact_out = self.artifact(face_crop)
        fused = self.fusion(rppg_out["feature"], artifact_out["feature"], quality)

        return (
            fused["fake_probability"],
            fused["liveness_score"],
            fused["confidence_score"],
            rppg_out["estimated_hr"],
            artifact_out["artifact_fake"],
            rppg_out["rppg_liveness"],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Export deployment ONNX for the fused deepfake/rPPG pipeline.")
    parser.add_argument("--out", default="checkpoints/pipeline.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--rppg-weights", default=CFG.rppg_weights)
    parser.add_argument("--artifact-weights", default=CFG.artifact_weights)
    parser.add_argument("--fusion-weights", default=CFG.fusion_weights)
    parser.add_argument("--artifact-backbone", default=CFG.artifact_backbone, choices=["custom", "rexnet_100", "rexnet_150"])
    parser.add_argument("--prefer-finetuned-branches", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=CFG.window_size)
    parser.add_argument("--face-size", type=int, default=CFG.face_crop_size)
    parser.add_argument("--dynamic-batch", action="store_true", default=True)
    parser.add_argument("--static-batch", action="store_false", dest="dynamic_batch")
    parser.add_argument("--dynamic-time", action="store_true", help="Allow variable rPPG time length. Default is fixed T=90.")
    parser.add_argument("--verify", action="store_true", help="Run ONNX Runtime parity check after export if installed.")
    return parser.parse_args()


def load_state(model: nn.Module, path: str, device: torch.device, required: bool = True):
    if not path or not Path(path).exists():
        if required:
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"WARNING: checkpoint not found: {path}")
        return False
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    meta = ""
    if isinstance(ckpt, dict):
        bits = []
        for key in ("epoch", "val_auc", "val_acc", "val_mae", "artifact_backbone", "finetuned_branches"):
            if key in ckpt:
                bits.append(f"{key}={ckpt[key]}")
        meta = " (" + ", ".join(bits) + ")" if bits else ""
    print(f"loaded {path}{meta}")
    return True


def dynamic_axes(args):
    axes = {}
    if args.dynamic_batch:
        axes.update(
            {
                "rppg_window": {0: "batch"},
                "face_crop": {0: "batch"},
                "quality": {0: "batch"},
                "fake_probability": {0: "batch"},
                "liveness_score": {0: "batch"},
                "confidence_score": {0: "batch"},
                "estimated_hr": {0: "batch"},
                "artifact_fake": {0: "batch"},
                "rppg_liveness": {0: "batch"},
            }
        )
    if args.dynamic_time:
        axes.setdefault("rppg_window", {})
        axes["rppg_window"][1] = "time"
    return axes or None


def verify_with_onnxruntime(out_path: str, pt_outputs, dummy_inputs):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed; skipping verification.")
        return

    rppg_window, face_crop, quality = dummy_inputs
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    ort_outputs = sess.run(
        None,
        {
            "rppg_window": rppg_window.detach().cpu().numpy(),
            "face_crop": face_crop.detach().cpu().numpy(),
            "quality": quality.detach().cpu().numpy(),
        },
    )
    names = ["fake_probability", "liveness_score", "confidence_score", "estimated_hr", "artifact_fake", "rppg_liveness"]
    max_diff = 0.0
    for name, pt, ort_out in zip(names, pt_outputs, ort_outputs):
        diff = float(abs(pt.detach().cpu().numpy() - ort_out).max())
        max_diff = max(max_diff, diff)
        print(f"verify {name}: max_abs_diff={diff:.6f}")
    print(f"verify max_abs_diff={max_diff:.6f}")


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"export_device={device}")

    rppg_weights = args.rppg_weights
    artifact_weights = args.artifact_weights
    if args.prefer_finetuned_branches:
        rppg_weights = prefer_fusion_checkpoint(rppg_weights, "rppg_fusion_best.pt")
        artifact_weights = prefer_fusion_checkpoint(artifact_weights, f"artifact_{args.artifact_backbone}_fusion_best.pt")

    rppg = RPPGTCN().to(device).eval()
    artifact = create_artifact_model(args.artifact_backbone, pretrained=False).to(device).eval()
    fusion = FusionClassifier().to(device).eval()
    load_state(rppg, rppg_weights, device)
    load_state(artifact, artifact_weights, device)
    load_state(fusion, args.fusion_weights, device)

    pipeline = DeploymentPipelineONNX(rppg, artifact, fusion).to(device).eval()

    # Deployment contract:
    # - rppg_window is raw ROI RGB means in 0..1. Normalization happens inside ONNX.
    # - face_crop is RGB, CHW, resized to face_size, in 0..1.
    # - quality is [roi_quality, detection_score, motion_quality], each in 0..1.
    dummy_rppg = torch.rand(args.batch_size, args.window_size, 9, dtype=torch.float32, device=device)
    dummy_face = torch.rand(args.batch_size, 3, args.face_size, args.face_size, dtype=torch.float32, device=device)
    dummy_quality = torch.tensor([[1.0, 0.9, 0.8]], dtype=torch.float32, device=device).repeat(args.batch_size, 1)

    with torch.no_grad():
        pt_outputs = pipeline(dummy_rppg, dummy_face, dummy_quality)
    print("forward_ok")
    print(f"  fake_probability={pt_outputs[0].flatten()[0].item():.4f}")
    print(f"  liveness_score={pt_outputs[1].flatten()[0].item():.4f}")
    print(f"  confidence_score={pt_outputs[2].flatten()[0].item():.4f}")
    print(f"  estimated_hr={pt_outputs[3].flatten()[0].item():.2f}")
    print(f"  artifact_fake={pt_outputs[4].flatten()[0].item():.4f}")
    print(f"  rppg_liveness={pt_outputs[5].flatten()[0].item():.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        pipeline,
        (dummy_rppg, dummy_face, dummy_quality),
        str(out_path),
        opset_version=args.opset,
        input_names=["rppg_window", "face_crop", "quality"],
        output_names=[
            "fake_probability",
            "liveness_score",
            "confidence_score",
            "estimated_hr",
            "artifact_fake",
            "rppg_liveness",
        ],
        dynamic_axes=dynamic_axes(args),
        do_constant_folding=True,
    )

    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"exported={out_path} size={mb:.2f}MB")
    if args.verify:
        verify_with_onnxruntime(str(out_path), pt_outputs, (dummy_rppg, dummy_face, dummy_quality))

    print("deployment_note=Run face detection, landmark ROI extraction, buffering, smoothing, and UI thresholding outside ONNX.")


if __name__ == "__main__":
    main()
