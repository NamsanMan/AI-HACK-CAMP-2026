# Final Test And Visualization

Use these commands after final training. By default, they use the original
rPPG/artifact branch checkpoints and the final fusion checkpoint. Add
`--prefer-finetuned-branches` only when branch fine-tuning improves validation AUC.

## Dataset Evaluation

Sample-level and video-level metrics with CSV export:

```powershell
python eval_final.py --data-root datasets/FaceForensics++ --frames-dir datasets/FaceForensics++/face_frames --rppg-dir datasets/FaceForensics++/rppg_v2 --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt --batch-size 128 --num-workers 2 --csv outputs/final_eval_predictions.csv
```

## Real Video Overlay

```powershell
python inspect_final_video.py --input datasets/FaceForensics++/original/000.mp4 --output outputs/final_real_000.mp4 --csv outputs/final_real_000.csv --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt
```

## Fake Video Overlay

```powershell
python inspect_final_video.py --input datasets/FaceForensics++/Deepfakes/000_003.mp4 --output outputs/final_fake_000_003.mp4 --csv outputs/final_fake_000_003.csv --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt
```

## Useful Options

- `--display`: show the rendered video while saving.
- `--max-frames 300`: render only the first 300 frames for a quick check.
- `--hide-roi`: hide ROI polygons for cleaner presentation output.
- `--debug`: show artifact branch score, pulse consistency, and buffer fill.
- `--threshold 0.5`: adjust evaluation threshold in `eval_final.py`.

## UI Risk Thresholds

The realtime/video UI uses hysteresis thresholds for stable display:

- `Low Risk`: risk score below `0.65`
- `Watch`: enters at `0.65`, exits below `0.55`
- `High Risk`: enters at `0.85`, exits below `0.75`

This keeps short real-video spikes from becoming warnings while preserving clear
high-risk display for strong fake detections.

## Robustness Retraining

For better webcam/deploy generalization, retrain the artifact branch with
realtime augmentations enabled. This does not require re-running preprocessing.

```powershell
python train_artifact.py --data-root datasets/FaceForensics++ --frames-dir datasets/FaceForensics++/face_frames --epochs 15 --batch-size 32 --grad-accum-steps 2 --num-workers 2 --val-ratio 0.1 --lr 7e-4 --backbone-lr 7e-6 --augment
```

Then retrain frozen fusion with augmented face crops:

```powershell
python train_fusion.py --data-root datasets/FaceForensics++ --frames-dir datasets/FaceForensics++/face_frames --rppg-dir datasets/FaceForensics++/rppg_v2 --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --epochs 20 --batch-size 32 --grad-accum-steps 2 --num-workers 2 --val-ratio 0.1 --lr 7e-4 --augment
```

Use smaller batches for robustness. Avoid branch fine-tuning unless validation
AUC improves. In deployment, missing or low-quality face input is shown as
`Unverified`, not `High Risk`.
