# 데이터셋 준비, 모델 설계, 학습 방법 설명서

## 1. 프로젝트 목표

이 프로젝트는 실시간 영상통화 환경에서 얼굴 영상을 분석해 deepfake 또는 피싱 위험도를 추정하는 프로토타입이다.

핵심 아이디어는 얼굴 이미지의 시각적 조작 흔적만 보는 것이 아니라, 얼굴 피부 영역의 시간적 RGB 변화인 rPPG 신호도 함께 사용한다는 점이다.

```text
Video frame
 -> Face detection
 -> Landmark 기반 cheek/forehead ROI 추출
 -> rPPG temporal branch
 -> Artifact image branch
 -> Fusion classifier
 -> fake probability / liveness score
```

즉, 최종 모델은 다음 두 정보를 함께 본다.

- 얼굴 crop 이미지에서 보이는 합성 흔적
- 볼과 이마 ROI에서 추출한 RGB 시계열 신호

## 2. 데이터셋 구성

현재 학습은 크게 두 종류의 데이터를 기준으로 구성된다.

### 2.1 UBFC-rPPG

UBFC-rPPG는 rPPG branch 학습에 사용한다.

이 데이터셋은 실제 사람 얼굴 영상과 ground truth 생체 신호를 포함한다. 따라서 얼굴 ROI의 RGB 변화와 실제 심박 또는 pulse signal 사이의 관계를 학습할 수 있다.

예상 폴더 구조:

```text
datasets/
  UBFC-rPPG/
    subject_01/
      vid.avi
      ground_truth.txt
    subject_02/
      vid.avi
      ground_truth.txt
```

역할:

- rPPG branch 사전 학습
- ROI RGB 시계열에서 HR 또는 pulse 관련 feature 학습
- real 생체 신호 패턴을 모델이 이해하도록 만드는 단계

### 2.2 FaceForensics++

FaceForensics++는 artifact branch와 fusion classifier 학습에 사용한다.

이 데이터셋에는 real 영상과 fake 영상이 있으므로 얼굴 이미지 기반 deepfake artifact 학습에 적합하다.

현재 사용하는 구조:

```text
datasets/
  FaceForensics++/
    original/
      *.mp4
    Deepfakes/
      *.mp4
    face_frames/
      real/*.jpg
      fake/*.jpg
    rppg_v2/
      real/*.npy
      fake/*.npy
```

역할:

- `face_frames`: artifact branch 학습용 얼굴 crop 이미지
- `rppg_v2`: fusion 학습용 ROI RGB 시계열
- real/fake label을 이용한 최종 risk classifier 학습

## 3. 전처리 과정

학습 중에 매번 영상을 열어서 얼굴 검출, 랜드마크 추출, ROI 계산을 수행하면 너무 느리다.

그래서 학습 전 단계에서 필요한 입력을 미리 추출한다.

### 3.1 얼굴 프레임 추출

FaceForensics++ 영상에서 얼굴 crop 이미지를 추출한다.

결과:

```text
datasets/FaceForensics++/face_frames/
  real/*.jpg
  fake/*.jpg
```

이 이미지는 artifact branch의 입력으로 사용된다.

입력 형태:

```text
face crop: 3 x 96 x 96
label: 0 real, 1 fake
```

현재 artifact branch는 ReXNet-100을 사용하며, 입력 face crop 크기는 `96 x 96`이다.

### 3.1.1 얼굴 crop 기준

얼굴 crop은 전체 프레임을 단순 resize하지 않고, 먼저 얼굴 검출 결과인 bounding box를 기준으로 만든다.

현재 `preprocess_face_frames.py`의 흐름은 다음과 같다.

```text
video frame
 -> FaceDetector로 얼굴 bbox 탐지
 -> bbox 주변에 여유 padding 추가
 -> 얼굴 영역 crop
 -> 96 x 96으로 resize
 -> JPEG 저장
```

crop 함수는 `LandmarkROIExtractor.aligned_face_crop()`을 사용한다.

구체적인 기준:

- 얼굴 detector가 반환한 bbox를 사용한다.
- bbox 너비 기준 좌우로 약 `15%` padding을 추가한다.
- bbox 높이 기준 위아래로 약 `20%` padding을 추가한다.
- padding을 추가한 영역이 프레임 밖으로 나가지 않도록 좌표를 clamp한다.
- 최종 crop을 `96 x 96` 크기로 resize한다.

즉 artifact branch가 보는 입력은 다음과 같다.

```text
원본 frame 전체가 아니라,
얼굴 bbox 주변을 약간 넓게 포함한 face crop
```

이렇게 한 이유:

- 얼굴 내부 texture뿐 아니라 경계 부근의 blending artifact도 일부 포함하기 위해서
- 배경 전체나 옷 같은 불필요한 정보를 줄이기 위해서
- 학습 입력과 실시간 demo 입력을 최대한 비슷하게 맞추기 위해서

얼굴 검출이 실패한 경우에는 전처리 옵션에 따라 center square crop fallback을 사용할 수 있다. 다만 이 fallback은 어디까지나 데이터 손실을 줄이기 위한 예외 처리이고, 정상 경로는 얼굴 bbox 기반 crop이다.

### 3.2 FaceForensics++ rPPG 시계열 추출

`preprocess_rppg_ff.py`는 각 영상에서 얼굴 landmark를 찾고, 볼과 이마 ROI의 RGB mean 값을 저장한다.

사용 ROI:

- left cheek
- right cheek
- forehead

한 프레임마다 다음 9개 값이 생성된다.

```text
R_left, G_left, B_left,
R_right, G_right, B_right,
R_forehead, G_forehead, B_forehead
```

결과 파일:

```text
datasets/FaceForensics++/rppg_v2/
  real/*_rppg.npy
  fake/*_rppg.npy
```

각 `.npy` 파일은 대략 다음 형태를 가진다.

```text
N x 9
```

여기서 `N`은 영상에서 추출된 frame 수다.

### 3.2.1 Landmark를 찾는 방식

rPPG ROI는 bbox를 단순히 비율로 나눈 고정 crop이 아니라, MediaPipe face landmark를 사용해 찾는다.

현재 `utils/landmark_roi.py`의 `LandmarkROIExtractor`가 이 역할을 한다.

현재 실행 환경은 다음과 같았다.

```text
mediapipe version = 0.10.35
mp.solutions = 없음
mp.tasks = 있음
```

따라서 실제 사용한 landmark 모델은 `MediaPipe Tasks API`의 `FaceLandmarker`다.

```text
FaceLandmarker
 -> 얼굴 위의 landmark 좌표 반환
 -> landmark index로 cheek/forehead polygon 구성
 -> polygon 내부 RGB 평균 계산
```

FaceLandmarker가 반환하는 좌표는 정규화된 좌표이므로, 코드에서는 현재 frame의 width와 height를 곱해 pixel 좌표로 변환한다.

참고로 코드에는 구버전 MediaPipe 환경을 위한 `mp.solutions.face_mesh.FaceMesh` 분기도 남아 있지만, 현재 학습/전처리/데모를 실행한 환경에서는 `mp.solutions`가 없기 때문에 FaceMesh가 아니라 FaceLandmarker가 사용된다.

landmark가 실패했지만 얼굴 bbox는 있는 경우에는 bbox 기반 fallback ROI를 사용한다.

현재 사용하는 landmark polygon index는 다음과 같다.

```text
LEFT_CHEEK  = [50, 101, 118, 117, 123, 147, 187, 205, 203, 206, 216, 192]
RIGHT_CHEEK = [280, 330, 347, 346, 352, 376, 411, 425, 423, 426, 436, 416]
FOREHEAD    = [109, 10, 338, 337, 336, 296, 334, 293, 300, 151, 70, 63, 105, 66, 107]
```

이 index들로 polygon ROI를 만들고, 각 polygon 내부 pixel의 RGB 평균을 계산한다.

```text
left cheek polygon 내부 RGB 평균
right cheek polygon 내부 RGB 평균
forehead polygon 내부 RGB 평균
```

그 결과 한 프레임에서 `9`개의 값이 나온다.

```text
[R_left, G_left, B_left,
 R_right, G_right, B_right,
 R_forehead, G_forehead, B_forehead]
```

### 3.2.2 Landmark 처리 최적화

실시간성을 위해 landmark는 매 프레임 무조건 새로 계산하지 않는다.

현재 설정:

```text
detection_interval = 7
landmark_interval = 3
mediapipe_downscale_width = 320
```

의미:

- face detection은 기본적으로 7프레임마다 수행한다.
- landmark는 기본적으로 3프레임마다 수행한다.
- 중간 프레임에서는 이전 bbox 또는 landmark 결과를 재사용한다.
- MediaPipe 입력은 필요하면 width 320 기준으로 downscale해서 처리한다.

이렇게 한 이유:

- 매 프레임 MediaPipe를 돌리면 실시간 FPS가 떨어지기 때문이다.
- 얼굴 위치와 ROI는 프레임마다 급격히 변하지 않으므로, 짧은 구간에서는 캐시 결과를 재사용해도 충분하다.
- 전처리와 실시간 demo 모두 비슷한 방식으로 ROI를 만들기 때문에 학습/추론 입력 차이를 줄일 수 있다.

### 3.2.3 Landmark 실패 시 fallback

landmark가 실패했지만 얼굴 bbox는 있는 경우, bbox 내부 비율을 이용한 fallback ROI를 만든다.

fallback ROI는 다음 위치를 사용한다.

- bbox 왼쪽 중하단: left cheek
- bbox 오른쪽 중하단: right cheek
- bbox 상단 중앙: forehead

이 fallback은 정상적인 landmark ROI보다 정확도는 낮지만, 얼굴이 잠깐 흐려지거나 landmark가 불안정할 때 rPPG buffer가 완전히 끊기는 것을 줄여준다.

단, landmark와 ROI 품질이 낮으면 quality score가 낮아지고, 실시간 UI에서는 `Unverified` 또는 낮은 신뢰 상태로 처리된다.

### 3.3 UBFC-rPPG sliding window 생성

UBFC-rPPG는 rPPG branch 학습을 위해 window 단위로 자른다.

기본 설정:

```text
window size = 90 frames
stride = 30 frames
```

30 FPS 기준으로 보면:

- 90 frames = 3초 window
- 30 frames stride = 1초 간격 score 갱신

생성 명령 예시:

```powershell
python preprocess_ubfc_rppg.py --root datasets/UBFC-rPPG --out-dir datasets/UBFC-rPPG/windows --window-size 90 --stride 30
```

## 4. 모델 설계

전체 모델은 세 부분으로 나뉜다.

```text
rPPG branch
Artifact branch
Fusion classifier
```

## 4.0 ONNX 모델 입력 전까지의 추론 전처리 과정

실시간 inference 또는 비디오 inference에서 ONNX 모델에 원본 frame이 그대로 들어가는 것은 아니다.

ONNX 모델은 이미 가공된 tensor를 입력으로 받는다.

ONNX 입력은 크게 다음 세 가지다.

```text
rppg_window: 90 x 9
face_crop: 3 x 96 x 96
quality: 3
```

따라서 원본 webcam/video frame을 위 입력 형태로 바꾸는 전처리 파이프라인이 필요하다.

전체 흐름은 다음과 같다.

```text
Raw video frame
 -> Face detector
 -> Face bbox
 -> FaceLandmarker
 -> cheek/forehead ROI polygon
 -> ROI RGB mean extraction
 -> temporal rPPG buffer
 -> face crop generation
 -> quality feature generation
 -> ONNX model input tensors
```

### 4.0.1 Face detector로 얼굴 bbox 생성

먼저 입력 frame에서 얼굴이 어디에 있는지 찾는다.

이 단계는 MediaPipe의 사전 학습된 face detection model을 사용한다.

현재 MediaPipe Tasks API 환경에서는 BlazeFace 계열의 face detector `.tflite` 모델을 사용한다.

출력은 얼굴 bounding box다.

```text
bbox = (x1, y1, x2, y2)
```

이 bbox는 세 가지 용도로 사용된다.

- artifact branch 입력용 face crop 생성
- 얼굴이 현재 frame에 존재하는지 판단
- landmark 실패 시 rPPG ROI fallback 기준

실시간성을 위해 face detection은 매 frame 수행하지 않는다.

현재 설정:

```text
detection_interval = 7
```

즉 기본적으로 7 frame마다 새 detection을 수행하고, 중간 frame에서는 이전 bbox를 재사용하거나 tracker/smoother를 통해 부드럽게 이어간다.

### 4.0.2 bbox로 artifact branch용 face crop 생성

artifact branch는 전체 frame을 보지 않고 얼굴 crop만 본다.

face crop 생성 기준은 다음과 같다.

```text
face bbox
 -> 좌우 15% padding
 -> 상하 20% padding
 -> frame 경계 안으로 clamp
 -> 96 x 96 resize
```

최종 결과:

```text
face_crop = 3 x 96 x 96
```

이 입력은 ReXNet-100 artifact branch로 들어간다.

이렇게 crop을 사용하는 이유:

- 얼굴 texture artifact를 집중적으로 보기 위해서
- 합성 경계 근처 정보를 조금 포함하기 위해서
- 배경이나 옷처럼 deepfake 판단에 덜 직접적인 정보를 줄이기 위해서
- 학습 때 만든 face crop과 실시간 inference 입력을 맞추기 위해서

### 4.0.3 FaceLandmarker로 얼굴 landmark 생성

rPPG branch는 face crop 이미지가 아니라 볼과 이마 ROI의 RGB 시계열을 입력으로 받는다.

따라서 얼굴 내부에서 볼과 이마가 어디인지 찾아야 한다.

이를 위해 MediaPipe Tasks API의 사전 학습된 `FaceLandmarker`를 사용한다.

현재 실행 환경:

```text
mediapipe version = 0.10.35
mp.solutions = 없음
mp.tasks = 있음
```

따라서 실제 사용한 landmark 모델은 `FaceMesh`가 아니라 `FaceLandmarker`다.

FaceLandmarker는 얼굴 위의 여러 landmark 좌표를 반환한다.

이 좌표는 정규화된 좌표이므로, 코드에서는 frame width와 height를 곱해 pixel 좌표로 변환한다.

실시간성을 위해 landmark도 매 frame 새로 계산하지 않는다.

현재 설정:

```text
landmark_interval = 3
mediapipe_downscale_width = 320
```

즉 기본적으로 3 frame마다 landmark를 갱신하고, MediaPipe 입력은 필요하면 width 320 기준으로 downscale해서 처리한다.

### 4.0.4 landmark polygon으로 cheek/forehead ROI 생성

FaceLandmarker가 반환한 landmark 중 rPPG에 사용할 영역만 골라 polygon ROI를 만든다.

사용 ROI:

- left cheek
- right cheek
- forehead

현재 사용하는 landmark index:

```text
LEFT_CHEEK  = [50, 101, 118, 117, 123, 147, 187, 205, 203, 206, 216, 192]
RIGHT_CHEEK = [280, 330, 347, 346, 352, 376, 411, 425, 423, 426, 436, 416]
FOREHEAD    = [109, 10, 338, 337, 336, 296, 334, 293, 300, 151, 70, 63, 105, 66, 107]
```

각 index에 해당하는 점들을 이어서 polygon을 만든다.

이 polygon 내부 pixel만 골라 RGB 평균을 계산한다.

한 frame에서 나오는 rPPG raw signal은 다음과 같다.

```text
x_t = [
  R_left, G_left, B_left,
  R_right, G_right, B_right,
  R_forehead, G_forehead, B_forehead
]
```

즉 한 frame당 `9`개의 값이 만들어진다.

### 4.0.5 temporal rPPG buffer 생성

rPPG branch는 한 frame만으로 판단하지 않는다.

피부색 변화는 시간에 따른 신호이므로 일정 길이의 window가 필요하다.

현재 기본 설정:

```text
window_seconds = 3.0
target_fps = 30
window_size = 90
score_interval_seconds = 1.0
```

따라서 rPPG branch 입력은 다음 형태다.

```text
rppg_window = 90 x 9
```

의미:

- 90개 frame
- 각 frame마다 9개 RGB ROI 값
- 약 3초 분량의 temporal signal

실시간 demo에서는 매 frame마다 ROI RGB 값을 buffer에 넣고, 기본적으로 1초마다 score를 갱신한다.

초기에는 buffer가 아직 90 frame을 채우지 못하므로, partial window와 smoothing을 이용해 시작 지연을 줄인다.

### 4.0.6 quality feature 생성

ONNX 모델에는 `quality` 입력도 들어간다.

형태:

```text
quality = 3
```

현재 의미:

```text
quality[0] = ROI quality
quality[1] = face detection score
quality[2] = motion/signal quality
```

이 값은 모델과 runtime이 입력 신뢰도를 판단하는 데 사용한다.

예를 들어 다음 상황에서는 confidence 또는 표시 상태를 낮춘다.

- 얼굴 detection score가 낮음
- landmark ROI가 불안정함
- 얼굴이 너무 작음
- 사용자가 뒤돌아봄
- frame blur 또는 motion이 큼
- rPPG buffer가 충분히 차지 않음

이런 경우 바로 `High Risk`로 단정하지 않고 `Unverified` 또는 낮은 신뢰 상태로 처리한다.

### 4.0.7 ONNX 입력 정리

최종적으로 ONNX 모델에 들어가는 것은 원본 영상이 아니라 다음 세 tensor다.

```text
rppg_window: B x 90 x 9
face_crop:   B x 3 x 96 x 96
quality:     B x 3
```

ONNX 모델 내부에서는 다음 branch들이 실행된다.

```text
rppg_window -> rPPG TCN branch
face_crop   -> ReXNet-100 artifact branch
quality     -> fusion reliability input
```

그리고 fusion classifier가 최종 출력을 만든다.

```text
fake_probability
liveness_score
estimated_hr
artifact_fake
rppg_liveness
```

`confidence_score`는 raw ONNX 출력에 포함하지 않는다. 입력 신뢰도와 표시 안정성은 runtime의 quality gating, smoothing, thresholding에서 처리한다.

정리하면, 배포용 plugin은 ONNX 모델만 실행하는 것이 아니라 ONNX 입력을 만들기 위한 전처리 runtime도 함께 구현해야 한다.

```text
Plugin/runtime responsibility:
  webcam/video capture
  face detection
  FaceLandmarker
  ROI polygon extraction
  RGB mean extraction
  temporal buffer
  quality calculation
  smoothing / thresholding / UI

ONNX responsibility:
  rPPG branch inference
  artifact branch inference
  fusion classifier inference
```

## 4.1 rPPG Branch

rPPG branch는 얼굴 ROI의 RGB 시계열을 입력으로 받는다.

입력:

```text
X: 90 x 9
```

의미:

- 90 frames: 약 3초 길이의 temporal window
- 9 channels: left cheek, right cheek, forehead의 RGB mean

중요한 점:

> rPPG branch는 이미지 feature vector가 아니라 원본 ROI RGB temporal signal을 사용한다.

출력:

- rPPG feature
- estimated HR
- rPPG liveness score

역할:

- 얼굴 피부 영역에서 생체 신호처럼 보이는 주기적 변화를 추정
- 심박과 유사한 temporal consistency를 보조 feature로 제공
- 단독 fake 판별기가 아니라 fusion classifier의 보조 branch로 사용

## 4.2 Artifact Branch

Artifact branch는 얼굴 crop 이미지를 입력으로 받는다.

입력:

```text
face crop: 3 x 96 x 96
```

현재 backbone:

```text
ReXNet-100
```

ReXNet은 NAVER AI Lab/CLOVA AI 계열의 효율적인 CNN 설계 아이디어를 기반으로 한 모델이다. 이 프로젝트에서는 한국 AI model 사용 가산점과 실시간성을 고려해 ReXNet-100을 artifact branch로 사용한다.

역할:

- 얼굴 texture artifact 탐지
- 합성 boundary artifact 탐지
- 압축, blending, 피부 질감 이상 탐지

출력:

- artifact feature
- artifact fake probability

현재 코드는 ReXNet-100 전용으로 정리되어 있다. `custom CNN`, `MobileNet`, `ReXNet-150` 같은 선택지는 제거했다.

## 4.3 Fusion Classifier

Fusion classifier는 rPPG branch와 artifact branch의 feature를 결합한다.

입력:

```text
rPPG feature
artifact feature
quality feature
```

quality feature에는 다음 정보가 들어간다.

- ROI quality
- face detection score
- motion/signal quality

출력:

```text
fake_probability
liveness_score
```

최종 프로토타입 ONNX는 raw score 기준이므로 `confidence_score`를 출력하지 않는다. 입력 품질 판단은 별도의 quality feature와 runtime 후처리에서 담당한다.

최종 UI에서는 대략 다음 식으로 생각하면 된다.

```text
risk score = fake_probability
liveness = liveness_score
state = Low Risk / Watch / High Risk / Unverified
```

## 5. 학습 방법

학습은 3단계로 진행한다.

```text
Stage 1: rPPG branch 학습
Stage 2: artifact branch 학습
Stage 3: fusion classifier 학습
```

## 5.1 Stage 1: rPPG Branch 학습

목표:

```text
ROI RGB 시계열 -> HR 또는 pulse 관련 feature 학습
```

입력:

```text
90 x 9 temporal RGB signal
```

출력:

```text
estimated HR 또는 rPPG feature
```

학습 데이터:

```text
datasets/UBFC-rPPG/windows
```

명령 예시:

```powershell
python train_rppg.py --data-root datasets/UBFC-rPPG --windows-dir datasets/UBFC-rPPG/windows --epochs 50 --batch-size 32 --val-ratio 0.2 --lr 1e-3
```

이 단계는 deepfake 판별을 직접 학습하는 단계가 아니다.

목적은 real face video에서 생체 신호와 관련된 temporal representation을 학습하는 것이다.

## 5.2 Stage 2: Artifact Branch 학습

목표:

```text
face crop image -> real/fake probability
```

입력:

```text
3 x 96 x 96 face crop
```

학습 데이터:

```text
datasets/FaceForensics++/face_frames
```

명령 예시:

```powershell
python train_artifact.py --data-root datasets/FaceForensics++ --frames-dir datasets/FaceForensics++/face_frames --epochs 15 --batch-size 32 --grad-accum-steps 2 --num-workers 2 --val-ratio 0.1 --lr 7e-4 --backbone-lr 7e-6 --augment
```

학습 포인트:

- ReXNet-100 pretrained weight를 사용한다.
- backbone learning rate는 작게 둔다.
- classification head learning rate는 상대적으로 크게 둔다.
- `--augment`를 사용해 webcam-like 변화를 일부 반영한다.

이렇게 하는 이유:

- pretrained feature를 너무 빠르게 망가뜨리지 않기 위해서
- 실제 영상통화 환경의 조명 변화, 압축, blur, 움직임에 조금 더 견디게 하기 위해서

## 5.3 Stage 3: Fusion Classifier 학습

목표:

```text
rPPG feature + artifact feature + quality feature -> final fake probability
```

입력:

```text
rPPG window: 90 x 9
face crop: 3 x 96 x 96
quality: 3
```

학습 데이터:

```text
datasets/FaceForensics++/face_frames
datasets/FaceForensics++/rppg_v2
```

명령 예시:

```powershell
python train_fusion.py --data-root datasets/FaceForensics++ --frames-dir datasets/FaceForensics++/face_frames --rppg-dir datasets/FaceForensics++/rppg_v2 --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --epochs 20 --batch-size 32 --grad-accum-steps 2 --num-workers 2 --val-ratio 0.1 --lr 7e-4 --augment
```

학습 포인트:

- 이미 학습된 rPPG branch와 artifact branch를 불러온다.
- fusion classifier가 두 branch의 정보를 조합하도록 학습한다.
- 일반화 안정성을 위해 branch fine-tuning은 조심해서 사용한다.
- 현재 해커톤 시연 목적에서는 frozen branch 기반 fusion이 더 안정적이다.

## 6. Validation 방식

Artifact와 fusion 학습에서는 video 단위 split을 사용한다.

즉, 같은 영상에서 나온 frame이 train과 validation에 동시에 들어가지 않도록 한다.

이유:

```text
같은 영상의 프레임이 train/val에 섞이면 데이터 리키지가 발생하고,
실제보다 validation 성능이 과대평가될 수 있다.
```

따라서 validation은 단순 frame random split보다 더 엄격한 방식이다.

## 7. 현재 학습 전략 요약

현재 최종 전략은 다음과 같다.

```text
1. UBFC-rPPG로 rPPG branch 학습
2. FaceForensics++ face frames로 ReXNet artifact branch 학습
3. FaceForensics++ rppg_v2 + face_frames로 fusion classifier 학습
4. 실시간 demo에서는 smoothing, hysteresis, Unverified gating 적용
```

모델 성능 측면에서 중요한 점:

- rPPG만으로 fake를 판단하지 않는다.
- artifact branch가 주요 fake detection 역할을 한다.
- rPPG branch는 liveness와 temporal consistency 보조 정보를 제공한다.
- fusion classifier가 두 branch를 결합해 최종 risk score를 낸다.
- 얼굴이 사라지거나 뒤돌아보는 경우는 fake로 단정하지 않고 `Unverified`로 처리한다.

## 8. 발표용 한 문장 요약

이 프로젝트는 FaceForensics++로 학습한 ReXNet 기반 artifact branch와 UBFC-rPPG로 학습한 rPPG temporal branch를 결합해, 실시간 영상통화에서 얼굴 조작 흔적과 생체 신호 일관성을 함께 분석하는 deepfake risk scoring pipeline이다.
