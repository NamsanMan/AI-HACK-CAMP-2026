# Judge Guide

이 문서는 심사위원이 프로젝트를 빠르게 파악할 수 있도록 만든 요약 가이드입니다.

## 1. What This Project Does

실시간 영상통화 또는 비디오 입력에서 얼굴을 분석해 deepfake/피싱 위험도를 추정합니다.

핵심은 두 branch를 함께 쓰는 것입니다.

```text
Face crop image -> ReXNet-100 artifact branch
Cheek/forehead RGB sequence -> rPPG temporal branch
Both features + quality -> fusion classifier
```

최종 출력:

- fake probability
- liveness score
- risk state: Low Risk / Watch / High Risk / Unverified

## 2. Why It Is Not Just Image Classification

일반적인 frame-based deepfake classifier는 얼굴 crop의 texture artifact에 크게 의존합니다.

이 프로젝트는 여기에 rPPG 계열 temporal signal을 추가합니다.

```text
left cheek RGB
right cheek RGB
forehead RGB
```

각 frame에서 9개 값을 만들고, 90-frame window로 temporal model에 넣습니다.

```text
rppg_window = 90 x 9
```

rPPG branch는 단독 판별기가 아니라 liveness와 temporal consistency를 보조하는 branch입니다.

## 3. Runtime Pipeline

```text
Raw frame
 -> MediaPipe face detector
 -> face bbox
 -> MediaPipe FaceLandmarker
 -> cheek/forehead ROI polygons
 -> ROI RGB mean
 -> 90 x 9 rPPG buffer
 -> 96 x 96 face crop
 -> ONNX/PyTorch model inference
 -> smoothing + hysteresis + UI state
```

FaceLandmarker는 현재 환경에서 실제 사용한 MediaPipe Tasks API 기반 landmark model입니다.

## 4. Main Files To Review

```text
demo_webcam.py              Webcam demo entry point
demo_video.py               Video demo entry point
demo_runtime.py             Realtime inference pipeline
models/rppg_tcn.py          rPPG temporal branch
models/artifact_rexnet.py   ReXNet-100 artifact branch
models/fusion_model.py      Fusion classifier
utils/face_detector.py      Face detection wrapper
utils/landmark_roi.py       Landmark ROI extraction
export_onnx.py              Full ONNX export
export_onnx_raw.py          Raw-score ONNX export
```

## 5. Reproduce Demo

Webcam:

```powershell
python demo_webcam.py
```

Video:

```powershell
python demo_video.py --input path/to/video.mp4
```

Save visualization:

```powershell
python inspect_final_video.py --input datasets/FaceForensics++/Deepfakes/001_870.mp4 --output outputs/demo_fake_001_870.mp4 --csv outputs/demo_fake_001_870.csv --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt
```

## 6. Training Summary

Stage 1:

```text
UBFC-rPPG -> rPPG TCN branch
```

Stage 2:

```text
FaceForensics++ face frames -> ReXNet-100 artifact branch
```

Stage 3:

```text
FaceForensics++ face frames + rPPG sequences -> fusion classifier
```

## 7. Internal Validation Reference

Current prepared FaceForensics++ split:

```text
sample_level acc=0.8550 auc=0.9414
video_level  acc=0.9070 auc=0.9773
```

These are internal validation numbers for the hackathon prototype. They should be re-measured when dataset splits, checkpoints, or preprocessing change.

## 8. Deployment Boundary

ONNX export includes:

```text
rPPG TCN branch
ReXNet-100 artifact branch
Fusion classifier
```

ONNX export does not include:

```text
video capture
face detector
FaceLandmarker
ROI extraction
temporal buffer
smoothing
UI
```

Those parts must be implemented by the plugin/runtime.

## 9. Known Limitations

- Dynamic motion, strong blur, side/back-facing faces, and unusual lighting can reduce stability.
- rPPG is a supporting signal, not a standalone fake detector.
- A fake video can still produce periodic RGB changes, so HR is hidden or marked unreliable in risky states.
- This is a hackathon prototype and should not be treated as a final security product.

