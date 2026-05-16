# Real-Time Video Call Deepfake Detection Plugin Prototype

실시간 영상통화 환경에서 얼굴 영상의 rPPG 신호와 시각적 artifact를 함께 사용해 deepfake/피싱 risk score를 추정하는 프로토타입입니다. 우선순위는 학습 성능보다 webcam/video 데모 파이프라인이 실제로 동작하는 것입니다.

## Pipeline

```text
Video frame
 -> interval face detection + bbox smoothing
 -> MediaPipe Face Mesh landmarks, or OpenCV fallback
 -> landmark ROI: left cheek, right cheek, forehead
 -> raw RGB temporal buffer, X in R^{T x 9}
 -> rPPG TCN branch
 -> face/frame artifact CNN branch
 -> MLP fusion
 -> fake probability, liveness score, confidence score
```

## Install

```bash
cd D:\pytorch_projects\AI_HACKCAMP_2026
pip install -r requirements.txt
```

CUDA가 가능하면 PyTorch가 자동으로 GPU를 사용하고, 아니면 CPU로 실행됩니다.

## Current Dataset Layout

현재 확인된 구조:

```text
datasets/
  UBFC-rPPG/
    subject_01/
      vid.avi
      ground_truth.txt

  FaceForensics++/
    original/*.mp4
    Deepfakes/*.mp4
    frames/
      real/*.jpg
      fake/*.jpg
    rppg/ or rppg_v2/
      real/*_rppg.npy
      fake/*_rppg.npy
```

확인된 전처리 산출물:

- `datasets/FaceForensics++/frames/real`: 10,000 images
- `datasets/FaceForensics++/frames/fake`: 10,000 images
- `datasets/FaceForensics++/rppg/real`: 1,000 `.npy` files
- `datasets/FaceForensics++/rppg/fake`: 1,000 `.npy` files
- `datasets/FaceForensics++/rppg_v2`: final-quality regenerated rPPG output
- rPPG file shape: `(N, 9)`
- frame image shape: `96 x 96 x 3`, 학습 시 `128 x 128`로 resize

## Preprocessing

이미 전처리를 끝냈다면 다시 돌릴 필요는 없습니다.

Frame 추출:

```bash
python preprocess_ff.py --root datasets/FaceForensics++ --n_frames 10 --size 96
```

rPPG ROI RGB sequence 추출:

```bash
python preprocess_rppg_ff.py --root datasets/FaceForensics++ --every_n 2
```

Final-quality rPPG 재추출:

```bash
python preprocess_rppg_ff.py --root datasets/FaceForensics++ --out-dir datasets/FaceForensics++/rppg_v2 --every-n 1
```

## Realtime Demo

Webcam:

```bash
python demo_webcam.py
```

Video file:

```bash
python demo_video.py --input path/to/video.mp4
```

화면에는 frame, face bbox, cheek/forehead ROI, fake risk, liveness, confidence, HR, FPS가 표시됩니다. 종료는 `q` 또는 `Esc`입니다.

## Training

Stage 1: UBFC-rPPG로 rPPG branch 학습

```bash
python train_rppg.py --data-root datasets/UBFC-rPPG --epochs 3
```

Stage 2: 전처리된 FaceForensics++ frame으로 artifact branch 학습

```bash
python train_artifact.py \
  --data-root datasets/FaceForensics++ \
  --frames-dir datasets/FaceForensics++/frames \
  --epochs 3
```

Stage 3: 전처리된 frame + rPPG npy를 매칭해 fusion classifier 학습

```bash
python train_fusion.py \
  --data-root datasets/FaceForensics++ \
  --frames-dir datasets/FaceForensics++/frames \
  --rppg-dir datasets/FaceForensics++/rppg_v2 \
  --rppg-weights checkpoints/rppg_tcn.pt \
  --artifact-weights checkpoints/artifact_cnn.pt \
  --epochs 3
```

Evaluation도 같은 전처리 구조를 사용합니다.

```bash
python eval.py \
  --data-root datasets/FaceForensics++ \
  --frames-dir datasets/FaceForensics++/frames \
  --rppg-dir datasets/FaceForensics++/rppg_v2 \
  --fusion-weights checkpoints/fusion_model.pt
```

## Model Inputs

Artifact branch:

```text
input: image/frame tensor, shape = 3 x 128 x 128
label: 0 real, 1 fake
```

rPPG branch:

```text
input: temporal ROI RGB sequence, shape = T x 9
default: T = 90 at 30 FPS, 3 seconds
```

The 9 channels are:

```text
R_left, G_left, B_left,
R_right, G_right, B_right,
R_forehead, G_forehead, B_forehead
```

Fusion branch:

```text
input 1: rPPG sequence window, shape = 90 x 9
input 2: matched frame image, shape = 3 x 128 x 128
label: 0 real, 1 fake
```

## Config

`config.py`에서 주요 값을 수정할 수 있습니다.

- `detection_interval`: face detection 실행 간격
- `window_seconds`: rPPG temporal window 길이
- `score_interval_seconds`: score 갱신 간격
- `face_crop_size`: artifact CNN 입력 crop 크기
- `target_fps`: temporal window 산정 기준 FPS
- `device`: `auto`, `cpu`, `cuda`

## Limitations

- 현재 artifact 학습은 전처리된 frame 단위입니다. temporal flicker를 더 잘 보려면 clip 기반 artifact branch로 확장해야 합니다.
- fusion 학습은 rPPG `.npy`와 같은 video stem을 가진 frame들을 매칭합니다.
- 데모 score는 판정이 아니라 risk score입니다.
- MediaPipe `solutions.face_mesh`가 없는 환경에서는 OpenCV fallback ROI를 사용합니다.
