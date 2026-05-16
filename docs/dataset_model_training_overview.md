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

현재 학습은 크게 세 종류의 데이터를 기준으로 구성된다.

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

### 2.3 Celeb-DF-v2

Celeb-DF-v2는 일반화 성능 평가용으로 고려할 수 있다.

현재 핵심 학습은 FaceForensics++ 중심으로 진행했지만, 실제 배포 성능을 높이려면 Celeb-DF-v2 같은 다른 분포의 데이터셋으로 검증하는 것이 좋다.

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
confidence_score
```

사용자 UI에서는 `confidence_score`를 직접 보여주기보다 내부 안정성 판단용으로 사용한다.

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

