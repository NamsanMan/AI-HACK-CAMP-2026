# ONNX Deployment Contract

최종 프로토타입 ONNX는 `export_onnx_raw.py` 기준으로 생성합니다.

이 ONNX 파일은 thresholding, smoothing, hysteresis가 적용되지 않은 **raw model score**를 출력합니다.

## Export Command

```powershell
python export_onnx_raw.py --out checkpoints/pipeline_raw.onnx --rppg-weights checkpoints/rppg_tcn.pt --artifact-weights checkpoints/artifact_rexnet_100.pt --fusion-weights checkpoints/fusion_model.pt --opset 17 --verify
```

최종 배포/플러그인 연동 기준 파일명:

```text
checkpoints/pipeline_raw.onnx
```

## What Is Included

ONNX에 포함되는 부분:

```text
rPPG TCN branch
ReXNet-100 artifact branch
fusion classifier
```

## What Is Not Included

ONNX에 포함되지 않는 부분:

```text
영상 frame capture
face detection
FaceLandmarker
cheek/forehead ROI polygon extraction
ROI RGB mean 계산
temporal rPPG buffer
score smoothing
hysteresis threshold
UI rendering
```

따라서 plugin/runtime 쪽에서 ONNX 입력 tensor를 만들어 넣어야 합니다.

## Inputs

### `rppg_window`

```text
shape: B x 90 x 9
dtype: float32
range: 0..1
```

한 frame의 channel 순서:

```text
R_left, G_left, B_left,
R_right, G_right, B_right,
R_forehead, G_forehead, B_forehead
```

ONNX graph 내부에서 time dimension 기준 normalization을 수행합니다.

### `face_crop`

```text
shape: B x 3 x 96 x 96
dtype: float32
range: 0..1
order: RGB
```

ImageNet normalization은 ONNX graph 내부 artifact branch에서 수행합니다. Runtime에서 별도 ImageNet normalization을 다시 적용하지 않아야 합니다.

### `quality`

```text
shape: B x 3
dtype: float32
range: 0..1
```

의미:

```text
quality[:, 0] = ROI quality
quality[:, 1] = face / landmark detection quality
quality[:, 2] = motion or signal quality
```

## Outputs

`pipeline_raw.onnx` 출력:

```text
fake_probability
liveness_score
estimated_hr
artifact_fake
rppg_liveness
```

출력 의미:

- `fake_probability`: fusion model의 raw fake/risk probability
- `liveness_score`: fusion model의 raw liveness score
- `estimated_hr`: rPPG branch의 HR 추정값
- `artifact_fake`: artifact branch 단독 fake probability
- `rppg_liveness`: rPPG branch 단독 liveness score

`confidence_score`는 최종 raw ONNX 출력에 포함하지 않습니다. 입력 신뢰도와 화면 표시 안정성은 runtime의 quality gating, smoothing, hysteresis에서 처리합니다.

## Runtime Post-processing

ONNX 출력은 raw score이므로 plugin/runtime에서 후처리를 적용해야 합니다.

권장 후처리:

```text
1. EMA smoothing
2. quality gating
3. hysteresis threshold
4. Unverified state
5. HR reliability policy
```

권장 risk state:

```text
Low Risk:   risk < 0.65
Watch:      enters at 0.65, exits below 0.55
High Risk:  enters at 0.85, exits below 0.75
Unverified: poor face, landmark, ROI, signal, or buffer quality
```

`High Risk` 또는 `Unverified` 상태에서는 `estimated_hr`를 신뢰값으로 표시하지 않는 것을 권장합니다. fake 영상에서도 주기적 RGB 변화가 생겨 BPM처럼 보이는 값이 나올 수 있기 때문입니다.

