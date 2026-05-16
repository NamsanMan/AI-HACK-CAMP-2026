# Real-Time Video Call Deepfake Detection Plugin Prototype

실시간 영상통화 환경에서 얼굴 영상의 rPPG 신호와 시각적 artifact를 함께 사용해 deepfake/피싱 risk score를 추정하는 프로토타입입니다. 현재 구현은 파이프라인 동작과 데모 실행을 우선하며, 모델은 학습 전에도 랜덤 초기화 상태로 score를 출력할 수 있습니다.

## Pipeline

```text
Video frame
 -> interval face detection + bbox smoothing
 -> MediaPipe Face Mesh landmarks
 -> landmark polygon ROI: left cheek, right cheek, forehead
 -> raw RGB temporal buffer, X in R^{T x 9}
 -> rPPG TCN branch
 -> aligned face crop artifact CNN branch
 -> MLP fusion
 -> fake probability, liveness score, confidence score
```

## Install

```bash
cd D:\pytorch_projects\AI_HACKCAMP_2026
pip install -r requirements.txt
```

CUDA가 가능하면 PyTorch가 자동으로 GPU를 사용하고, 아니면 CPU로 실행됩니다.

## Dataset Layout

```text
datasets/
  UBFC-rPPG/
    subject_01/
      video.avi
      ground_truth.txt
  FaceForensics++/
    real/*.mp4
    fake/*.mp4
  Celeb-DF-v2/
    real/*.mp4
    fake/*.mp4
```

데이터셋 다운로드 자동화는 포함하지 않았습니다.

## Realtime Demo

```bash
python demo_webcam.py
```

video file 입력:

```bash
python demo_video.py --input path/to/video.mp4
```

화면에는 webcam/video frame, face bbox, cheek/forehead ROI polygon, fake risk, liveness, confidence, HR, FPS가 표시됩니다. 종료는 `q` 또는 `Esc`입니다.

MediaPipe `solutions.face_mesh` API가 있는 환경에서는 landmark polygon ROI를 사용합니다. 현재 설치 환경처럼 `mediapipe.tasks`만 노출되는 경우에는 데모 실행성을 위해 OpenCV Haar face detector와 bbox 기반 approximate ROI fallback을 사용합니다.

## Training

Stage 1: UBFC-rPPG로 rPPG branch 학습

```bash
python train_rppg.py --data-root datasets/UBFC-rPPG --epochs 3
```

Stage 2: FaceForensics++로 artifact branch 학습

```bash
python train_artifact.py --data-root datasets/FaceForensics++ --epochs 3
```

Stage 3: fusion classifier 학습

```bash
python train_fusion.py --data-root datasets/FaceForensics++ \
  --rppg-weights checkpoints/rppg_tcn.pt \
  --artifact-weights checkpoints/artifact_cnn.pt
```

Evaluation:

```bash
python eval.py --data-root datasets/Celeb-DF-v2 --fusion-weights checkpoints/fusion_model.pt
```

## Config

`config.py`에서 다음 항목을 바꿀 수 있습니다.

- `detection_interval`: face detection 실행 간격
- `window_seconds`: rPPG temporal window 길이, 기본 3초
- `score_interval_seconds`: score 갱신 간격, 기본 1초
- `face_crop_size`: artifact CNN 입력 crop 크기
- `target_fps`: 30 FPS 기준 window 산정
- `device`: `auto`, `cpu`, `cuda`

## Notes And Limitations

- rPPG branch는 DINOv2/YOLO feature가 아니라 ROI 원본 RGB temporal signal만 사용합니다.
- ROI는 bbox 고정 crop이 아니라 MediaPipe landmark polygon으로 구성됩니다.
- 데모 score는 risk score입니다. 높은 fake probability가 즉시 단정이나 판정을 의미하지 않습니다.
- 랜덤 초기화 모델의 출력은 성능 지표가 아니라 파이프라인 sanity check용입니다.
- 조명 변화, 압축, motion blur, 큰 head pose, landmark 실패 시 confidence가 낮아질 수 있습니다.

## Future Work

- UBFC ROI cache 생성기로 학습 속도 개선
- 실제 face alignment와 temporal artifact clip 모델 추가
- pretrained MobileNetV3/EfficientNet-lite option 추가
- blink, lip-sync, compression artifact feature 추가
- calibration set 기반 confidence calibration
