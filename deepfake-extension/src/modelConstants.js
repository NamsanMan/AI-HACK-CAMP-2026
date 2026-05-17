/** AI-HACK-CAMP-2026 config.py / export 기준 */
export const MODEL_FILENAME = "pipeline_raw2.onnx";

export const FACE_INPUT_SIZE = 96;
export const RPPG_WINDOW_FRAMES = 90;
export const RPPG_CHANNELS = 9;
/** Fusion 추론 최소 rPPG 샘플 (config.min_partial_rppg_frames) */
export const MIN_RPPG_FRAMES_FOR_INFER = 12;

/** camp: LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD (utils/landmark_roi.py) */
export const LEFT_CHEEK_IDX = [
  50, 101, 118, 117, 123, 147, 187, 205, 203, 206, 216, 192,
];

export const RIGHT_CHEEK_IDX = [
  280, 330, 347, 346, 352, 376, 411, 425, 423, 426, 436, 416,
];

export const FOREHEAD_IDX = [
  109, 10, 338, 337, 336, 296, 334, 293, 300, 151, 70, 63, 105, 66, 107,
];
