# Real-Time Video Call Deepfake Detection Plugin Prototype

실시간 영상통화 환경에서 얼굴 영상의 deepfake 또는 피싱 위험도를 추정하는 해커톤 프로토타입입니다.

이 프로젝트는 얼굴 이미지의 시각적 조작 흔적만 보는 방식이 아니라, 얼굴 피부 영역의 RGB temporal signal에서 얻는 rPPG 계열 정보도 함께 사용합니다.

최종 목표는 웹캠 또는 비디오 입력에서 다음 값을 실시간으로 표시하는 것입니다.

- `fake_probability`: deepfake risk score
- `liveness_score`: 입력 얼굴이 실제 생체 신호와 일관적인지에 대한 보조 점수
- `risk state`: `Low Risk`, `Watch`, `High Risk`, `Unverified`
- FPS, face bbox, cheek/forehead ROI overlay

> 이 프로젝트의 출력은 법적/보안적 확정 판정이 아니라 실시간 위험도 보조 지표입니다.

## Key Features

- Webcam / video file inference
- MediaPipe Tasks API 기반 face detection 및 FaceLandmarker
- Landmark 기반 `left cheek`, `right cheek`, `forehead` ROI 추출
- ROI RGB mean 기반 rPPG temporal buffer
- ReXNet-100 artifact branch
- rPPG TCN branch
- Fusion classifier
- ONNX export for plugin/runtime deployment
- Score smoothing, hysteresis, `Unverified` gating

## Pipeline

```text
Raw video frame
 -> MediaPipe face detector
 -> face bbox
 -> MediaPipe FaceLandmarker
 -> cheek/forehead ROI polygons
 -> ROI RGB mean signal, shape T x 9
 -> face crop, shape 3 x 96 x 96
 -> rPPG TCN branch
 -> ReXNet-100 artifact branch
 -> fusion classifier
 -> fake probability / liveness score / confidence score
 -> UI smoothing + risk thresholding
```

## Repository Structure

```text
AI_HACKCAMP_2026/
  config.py
  demo_webcam.py
  demo_video.py
  demo_runtime.py

  preprocess_face_frames.py
  preprocess_rppg_ff.py
  preprocess_ubfc_rppg.py

  train_rppg.py
  train_artifact.py
  train_fusion.py
  eval_final.py

  inspect_final_video.py
  export_onnx.py
  export_onnx_raw.py

  data/
  models/
  utils/
  docs/
```

## Model Design

### 1. rPPG Branch

The rPPG branch receives raw ROI RGB temporal signals, not image feature vectors.

Input:

```text
rppg_window: 90 x 9
```

Each frame contributes 9 values:

```text
R_left, G_left, B_left,
R_right, G_right, B_right,
R_forehead, G_forehead, B_forehead
```

The branch is implemented as a lightweight temporal model and outputs:

- rPPG feature
- estimated HR
- rPPG liveness score

### 2. Artifact Branch

The artifact branch receives a face crop:

```text
face_crop: 3 x 96 x 96
```

The current implementation uses `ReXNet-100` through `timm`.

The branch is trained to detect visual artifacts such as:

- texture inconsistency
- blending artifacts
- boundary artifacts
- compression-sensitive fake patterns

### 3. Fusion Classifier

The fusion classifier combines:

- rPPG feature
- artifact feature
- quality feature

Output:

```text
fake_probability
liveness_score
confidence_score
```

For user-facing UI, confidence is used internally. The UI displays a stable risk state rather than a hard binary judgment.

## Face Crop And Landmark Extraction

### Face Crop

The artifact branch does not receive the full frame.

The face crop is generated from a detected face bbox:

```text
face bbox
 -> x padding: 15%
 -> y padding: 20%
 -> clamp to frame boundary
 -> resize to 96 x 96
 -> RGB float tensor in 0..1
```

ImageNet normalization is handled inside the model/export path where applicable.

### FaceLandmarker

The current environment uses:

```text
mediapipe version = 0.10.35
mp.solutions = not available
mp.tasks = available
```

Therefore landmark extraction is performed with:

```text
MediaPipe Tasks API FaceLandmarker
```

The landmark polygons are used to extract:

- left cheek
- right cheek
- forehead

If landmark extraction fails but a face bbox exists, a bbox-based fallback ROI can be used with lower quality.

## Dataset Layout

Datasets are not included in the repository.

Expected layout:

```text
datasets/
  UBFC-rPPG/
    subject_01/
      vid.avi
      ground_truth.txt
    subject_02/
      vid.avi
      ground_truth.txt

  FaceForensics++/
    original/
      *.mp4
    Deepfakes/
      *.mp4
    face_frames/
      real/*.jpg
      fake/*.jpg
    rppg_v2/
      real/*_rppg.npy
      fake/*_rppg.npy
```

## Installation

```powershell
cd D:\pytorch_projects\AI_HACKCAMP_2026
pip install -r requirements.txt
```

CUDA is used automatically when available. CPU inference is also supported, but FPS may vary by hardware.

## Preprocessing

### 1. Extract Face Frames

```powershell
python preprocess_face_frames.py --root datasets/FaceForensics++ --out-dir datasets/FaceForensics++/face_frames --n-frames 10 --size 96
```

### 2. Extract FaceForensics++ rPPG Signals

```powershell
python preprocess_rppg_ff.py --root datasets/FaceForensics++ --out-dir datasets/FaceForensics++/rppg_v2 --every-n 1
```

### 3. Preprocess UBFC-rPPG Windows

```powershell
python preprocess_ubfc_rppg.py --root datasets/UBFC-rPPG --out-dir datasets/UBFC-rPPG/windows --window-size 90 --stride 30
```

## Training

### Stage 1. Train rPPG Branch

```powershell
python train_rppg.py --data-root datasets/UBFC-rPPG --windows-dir datasets/UBFC-rPPG/windows --epochs 50 --batch-size 32 --val-ratio 0.2 --lr 1e-3
```

### Stage 2. Train ReXNet Artifact Branch

```powershell
python train_artifact.py --data-root datasets/FaceForensics++ --frames-dir datasets/FaceForensics++/face_frames --epochs 15 --batch-size 32 --grad-accum-steps 2 --num-workers 2 --val-ratio 0.1 --lr 7e-4 --backbone-lr 7e-6 --augment
```

### Stage 3. Train Fusion Classifier

```powershell
python train_fusion.py --data-root datasets/FaceForensics++ --frames-dir datasets/FaceForensics++/face_frames --rppg-dir datasets/FaceForensics++/rppg_v2 --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --epochs 20 --batch-size 32 --grad-accum-steps 2 --num-workers 2 --val-ratio 0.1 --lr 7e-4 --augment
```

## Evaluation

```powershell
python eval_final.py --data-root datasets/FaceForensics++ --frames-dir datasets/FaceForensics++/face_frames --rppg-dir datasets/FaceForensics++/rppg_v2 --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt --batch-size 128 --num-workers 2 --csv outputs/final_eval_predictions.csv
```

Example internal validation result on the prepared FaceForensics++ split:

```text
sample_level acc=0.8550 auc=0.9414
video_level  acc=0.9070 auc=0.9773
```

These numbers are provided as a reference for the current hackathon prototype and should be re-evaluated when the dataset split or preprocessing changes.

## Demo

### Webcam

```powershell
python demo_webcam.py
```

### Video File

```powershell
python demo_video.py --input path/to/video.mp4
```

### Save A Visualization Video

```powershell
python inspect_final_video.py --input datasets/FaceForensics++/Deepfakes/001_870.mp4 --output outputs/demo_fake_001_870.mp4 --csv outputs/demo_fake_001_870.csv --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt
```

## ONNX Export

The ONNX model contains only the neural inference graph.

It does not include:

- video capture
- face detection
- FaceLandmarker
- ROI polygon extraction
- RGB mean extraction
- temporal buffering
- smoothing / thresholding
- UI rendering

Raw-score ONNX:

```powershell
python export_onnx_raw.py --out checkpoints/pipeline_raw.onnx --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt --opset 17 --verify
```

Full-output ONNX:

```powershell
python export_onnx.py --out checkpoints/pipeline.onnx --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt --opset 17 --verify
```

ONNX input contract:

```text
rppg_window: B x 90 x 9
face_crop:   B x 3 x 96 x 96
quality:     B x 3
```

See [ONNX_DEPLOYMENT.md](ONNX_DEPLOYMENT.md) for runtime integration details.

## Runtime Risk Policy

The runtime does not immediately treat every high raw score as a final warning.

It applies:

- EMA smoothing
- hysteresis thresholds
- quality gating
- `Unverified` state for poor face/ROI quality

Default UI policy:

```text
Low Risk:   risk < 0.65
Watch:      enters at 0.65, exits below 0.55
High Risk: enters at 0.85, exits below 0.75
Unverified: poor face, landmark, ROI, or buffer quality
```

If the face is missing, turned away, too small, or landmarks are unstable, the system should display `Unverified` rather than forcing `High Risk`.

## Documentation

- [Judge Guide](docs/judge_guide.md)
- [Dataset, Model, Training Overview](docs/dataset_model_training_overview.md)
- [ONNX Deployment Contract](ONNX_DEPLOYMENT.md)
- [Final Test And Visualization Commands](FINAL_TEST.md)

## Limitations

- This is a hackathon prototype, not a production-grade anti-fraud system.
- Generalization can drop on highly dynamic videos, unusual lighting, heavy blur, or unseen deepfake generation methods.
- rPPG is used as a supporting liveness signal, not as a standalone deepfake detector.
- Fake videos can still produce periodic RGB changes, so HR display is hidden or marked unreliable in `High Risk` or `Unverified` states.
- The ONNX model requires a matching runtime preprocessing pipeline.

## Large Files

The repository excludes large datasets, checkpoints, ONNX files, and generated outputs through `.gitignore`.

```text
datasets/
checkpoints/
outputs/
*.pt
*.pth
*.onnx
utils/models/*.tflite
utils/models/*.task
```

For reproduction or judging, place the required datasets and checkpoints in the paths shown above.
