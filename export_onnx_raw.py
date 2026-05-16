"""
Export raw-score ONNX for model evaluation/deployment integration.

This ONNX is intentionally threshold-free:

    no UI threshold
    no Low/Watch/High state
    no hysteresis
    no EMA smoothing

Use `fake_probability` as the raw model score for AUC/ROC analysis or apply
thresholds in the host runtime:

    0.50 -> binary classification metric
    0.65 -> Watch UI threshold
    0.85 -> High Risk UI threshold

The runtime still needs to provide preprocessed tensors:

    rppg_window: B x 90 x 9, raw ROI RGB means in 0..1
    face_crop:   B x 3 x 96 x 96, RGB face crop in 0..1
    quality:     B x 3, [roi_quality, detection_score, motion_quality]
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


class RawScorePipelineONNX(nn.Module):
    """Deployment wrapper with raw tensor outputs only."""

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
        rppg_out = self.rppg(self.normalize_rppg_window(rppg_window))
        artifact_out = self.artifact(face_crop)
        fused = self.fusion(rppg_out["feature"], artifact_out["feature"], quality)
        return (
            fused["fake_probability"],
            fused["liveness_score"],
            rppg_out["estimated_hr"],
            artifact_out["artifact_fake"],
            rppg_out["rppg_liveness"],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Export raw-score ONNX without thresholding or smoothing.")
    parser.add_argument("--out", default="checkpoints/pipeline_raw.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--rppg-weights", default=CFG.rppg_weights)
    parser.add_argument("--artifact-weights", default=CFG.artifact_weights)
    parser.add_argument("--fusion-weights", default=CFG.fusion_weights)
    parser.add_argument("--prefer-finetuned-branches", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=CFG.window_size)
    parser.add_argument("--face-size", type=int, default=CFG.face_crop_size)
    parser.add_argument("--dynamic-batch", action="store_true", default=True)
    parser.add_argument("--static-batch", action="store_false", dest="dynamic_batch")
    parser.add_argument("--dynamic-time", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def load_state(model: nn.Module, path: str, device: torch.device):
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    if isinstance(ckpt, dict):
        meta = {k: ckpt[k] for k in ("epoch", "val_auc", "val_acc", "val_mae", "artifact_backbone") if k in ckpt}
        print(f"loaded {path} {meta}")
    else:
        print(f"loaded {path}")


def make_dynamic_axes(args):
    axes = {}
    if args.dynamic_batch:
        for name in (
            "rppg_window",
            "face_crop",
            "quality",
            "fake_probability",
            "liveness_score",
            "estimated_hr",
            "artifact_fake",
            "rppg_liveness",
        ):
            axes[name] = {0: "batch"}
    if args.dynamic_time:
        axes.setdefault("rppg_window", {})
        axes["rppg_window"][1] = "time"
    return axes or None


def verify(out_path, pt_outputs, inputs):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed; skipping verification.")
        return
    rppg_window, face_crop, quality = inputs
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(
        None,
        {
            "rppg_window": rppg_window.detach().cpu().numpy(),
            "face_crop": face_crop.detach().cpu().numpy(),
            "quality": quality.detach().cpu().numpy(),
        },
    )
    names = ["fake_probability", "liveness_score", "estimated_hr", "artifact_fake", "rppg_liveness"]
    for name, pt, ort_out in zip(names, pt_outputs, ort_outputs):
        diff = abs(pt.detach().cpu().numpy() - ort_out).max()
        print(f"verify {name}: max_abs_diff={float(diff):.6f}")


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"export_device={device}")

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

    pipeline = RawScorePipelineONNX(rppg, artifact, fusion).to(device).eval()
    dummy_rppg = torch.rand(args.batch_size, args.window_size, 9, dtype=torch.float32, device=device)
    dummy_face = torch.rand(args.batch_size, 3, args.face_size, args.face_size, dtype=torch.float32, device=device)
    dummy_quality = torch.tensor([[1.0, 0.9, 0.8]], dtype=torch.float32, device=device).repeat(args.batch_size, 1)

    with torch.no_grad():
        outputs = pipeline(dummy_rppg, dummy_face, dummy_quality)
    print("forward_ok")
    print(f"  raw_fake_probability={outputs[0].flatten()[0].item():.4f}")
    print(f"  raw_liveness_score={outputs[1].flatten()[0].item():.4f}")
    print(f"  estimated_hr={outputs[2].flatten()[0].item():.2f}")
    print(f"  artifact_fake={outputs[3].flatten()[0].item():.4f}")
    print(f"  rppg_liveness={outputs[4].flatten()[0].item():.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        pipeline,
        (dummy_rppg, dummy_face, dummy_quality),
        str(out_path),
        opset_version=args.opset,
        input_names=["rppg_window", "face_crop", "quality"],
        output_names=["fake_probability", "liveness_score", "estimated_hr", "artifact_fake", "rppg_liveness"],
        dynamic_axes=make_dynamic_axes(args),
        do_constant_folding=True,
    )
    print(f"exported={out_path} size={out_path.stat().st_size / (1024 * 1024):.2f}MB")
    if args.verify:
        verify(out_path, outputs, (dummy_rppg, dummy_face, dummy_quality))
    print("note=This ONNX outputs raw model scores. Apply thresholding outside ONNX.")


if __name__ == "__main__":
    main()
