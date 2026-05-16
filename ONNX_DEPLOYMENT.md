# ONNX Deployment Contract

The exported ONNX model contains the neural inference graph only.

It does not include:

- webcam/video capture
- face detection
- MediaPipe face landmarks
- cheek/forehead ROI polygon extraction
- temporal buffering
- score smoothing and hysteresis
- UI rendering

Those pieces should run in the realtime plugin/runtime.

## Export

Use the best frozen-branch fusion checkpoint unless branch fine-tuning improved validation AUC.

Raw-score ONNX for evaluation/plugin integration without thresholding:

```powershell
python export_onnx_raw.py --out checkpoints/pipeline_raw.onnx --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt --artifact-backbone rexnet_100 --opset 17 --verify
```

Full tensor-output ONNX, including internal confidence output:

```powershell
python export_onnx.py --out checkpoints/pipeline.onnx --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt --artifact-backbone rexnet_100 --opset 17 --verify
```

If a branch fine-tuned checkpoint is actually better:

```powershell
python export_onnx.py --out checkpoints/pipeline.onnx --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt --artifact-backbone rexnet_100 --prefer-finetuned-branches --opset 17 --verify
```

## Inputs

`rppg_window`

- shape: `B x 90 x 9`
- dtype: `float32`
- value: raw ROI RGB means, normally `0..1`
- channel order per frame:
  - `R_left, G_left, B_left`
  - `R_right, G_right, B_right`
  - `R_forehead, G_forehead, B_forehead`
- The ONNX graph normalizes this window internally across the time dimension.

`face_crop`

- shape: `B x 3 x 96 x 96`
- dtype: `float32`
- value: aligned face crop in RGB order, `0..1`
- The artifact branch performs ImageNet normalization inside the ONNX graph.

`quality`

- shape: `B x 3`
- dtype: `float32`
- value:
  - `quality[:, 0]`: ROI quality, `0..1`
  - `quality[:, 1]`: face detection score, `0..1`
  - `quality[:, 2]`: motion/signal quality, `0..1`

## Outputs

`pipeline_raw.onnx` outputs:

- `fake_probability`: raw fused fake/risk probability, `B`
- `liveness_score`: raw fused liveness score, `B`
- `estimated_hr`: rPPG branch HR estimate in BPM, `B`
- `artifact_fake`: artifact branch fake probability, `B`
- `rppg_liveness`: rPPG branch liveness score, `B`

`pipeline.onnx` outputs:

- `fake_probability`: fused fake/risk probability, `B`
- `liveness_score`: fused liveness score, `B`
- `confidence_score`: internal confidence/reliability score, `B`
- `estimated_hr`: rPPG branch HR estimate in BPM, `B`
- `artifact_fake`: artifact branch fake probability, `B`
- `rppg_liveness`: rPPG branch liveness score, `B`

For the UI, prefer displaying `fake_probability`, `liveness_score`, and `estimated_hr`.
Keep `confidence_score` internal unless it is recalibrated.

## Runtime UI Thresholds

Apply these outside ONNX with smoothing/hysteresis:

- `Unverified`: no face, low detection score, poor ROI quality, small face, or insufficient buffer
- `Low Risk`: risk score below `0.65`
- `Watch`: enters at `0.65`, exits below `0.55`
- `High Risk`: enters at `0.85`, exits below `0.75`

Hide `estimated_hr` or show `HR: not reliable` when the risk state is `High Risk`,
because fake videos can still produce a periodic rPPG-like signal.
