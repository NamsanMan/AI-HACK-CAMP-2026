# 심사용 모델 프레임워크 요약

이 문서는 모델 자체의 구조와 학습 방식을 빠르게 파악하기 위한 요약입니다.

## 1. 모델의 핵심 아이디어

이 프로젝트는 얼굴 기반 deepfake risk score를 추정하기 위한 멀티브랜치 모델입니다.

핵심은 두 종류의 feature를 함께 사용하는 것입니다.

```text
얼굴 crop 이미지
  -> ReXNet-100 artifact branch

볼/이마 ROI RGB 시계열
  -> rPPG temporal branch

두 feature + quality
  -> fusion classifier
```

## 2. 전체 모델 구조

![전체 파이프라인](assets/overall_pipeline_korean.png)

```text
입력 프레임
 -> 얼굴 검출
 -> 랜드마크 ROI
 -> rPPG 분석 + artifact 분석
 -> fusion model
 -> 위험도 점수 / liveness 점수
```

그림은 모델 입력 생성과 branch 결합 흐름을 단순화한 것입니다.

## 3. 입력 텐서

모델 기준 입력은 세 가지입니다.

```text
rppg_window: 90 x 9
face_crop:   3 x 96 x 96
quality:     3
```

`rppg_window`는 한 frame마다 다음 9개 값을 갖습니다.

```text
R_left, G_left, B_left,
R_right, G_right, B_right,
R_forehead, G_forehead, B_forehead
```

## 4. Branch별 역할

### rPPG branch

- ROI RGB temporal signal 분석
- estimated HR 및 liveness feature 생성
- 단독 fake detector가 아니라 보조 temporal branch

### Artifact branch

- ReXNet-100 기반 image branch
- 얼굴 crop에서 texture, blending, boundary artifact 분석
- real/fake visual feature 생성

### Fusion classifier

- rPPG feature
- artifact feature
- quality feature

를 결합해 최종 fake probability와 liveness score를 출력합니다.

## 5. 학습 단계

Stage 1:

```text
UBFC-rPPG -> rPPG TCN branch 학습
```

Stage 2:

```text
FaceForensics++ face frames -> ReXNet-100 artifact branch 학습
```

Stage 3:

```text
FaceForensics++ face frames + rppg_v2 -> fusion classifier 학습
```

## 6. 내부 검증 참고값

현재 준비된 FaceForensics++ split 기준:

```text
sample_level acc=0.8550 auc=0.9414
video_level  acc=0.9070 auc=0.9773
```

이 수치는 해커톤 프로토타입 내부 검증 결과이며, 데이터 split이나 preprocessing이 바뀌면 다시 측정해야 합니다.

## 7. ONNX 변환 의미

ONNX export는 학습된 neural network graph를 배포 가능한 추론 그래프로 변환한 것입니다.

포함되는 모델:

```text
rPPG TCN branch
ReXNet-100 artifact branch
fusion classifier
```

ONNX 입력:

```text
rppg_window: B x 90 x 9
face_crop:   B x 3 x 96 x 96
quality:     B x 3
```

## 8. 주요 코드

```text
models/rppg_tcn.py          rPPG temporal branch
models/artifact_rexnet.py   ReXNet-100 artifact branch
models/fusion_model.py      Fusion classifier
train_rppg.py               rPPG 학습
train_artifact.py           artifact 학습
train_fusion.py             fusion 학습
eval_final.py               최종 평가
export_onnx_raw.py          raw-score ONNX 변환
```

## 9. 한계

- rPPG branch는 보조 liveness signal이며 단독 deepfake detector가 아닙니다.
- 모델 성능은 ROI 품질과 face crop 품질에 영향을 받습니다.
- 데이터셋 분포와 다른 입력에서는 일반화 성능이 낮아질 수 있습니다.
- 해커톤 프로토타입 모델 프레임워크이며, 실제 보안 제품 수준의 검증 시스템은 아닙니다.
