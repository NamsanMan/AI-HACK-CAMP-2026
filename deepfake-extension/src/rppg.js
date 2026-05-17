import {
  FaceLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";
import {
  FOREHEAD_IDX,
  LEFT_CHEEK_IDX,
  MIN_RPPG_FRAMES_FOR_INFER,
  RIGHT_CHEEK_IDX,
  RPPG_CHANNELS,
  RPPG_WINDOW_FRAMES,
} from "./modelConstants.js";

export { RPPG_WINDOW_FRAMES } from "./modelConstants.js";

const T = RPPG_WINDOW_FRAMES;
const C = RPPG_CHANNELS;

let faceLandmarker = null;
let rppgBuffer = [];
let lastSignal = new Array(C).fill(0);
let lastRoiQuality = 0;
let lastDetScore = 0;

export function isRppgLandmarkerReady() {
  return faceLandmarker !== null;
}

const FACE_LANDMARKER_CDN =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

async function resolveFaceLandmarkerModelUrl(isExtension) {
  const localPath = isExtension
    ? chrome.runtime.getURL("mediapipe/models/face_landmarker.task")
    : "/mediapipe/models/face_landmarker.task";

  try {
    const res = await fetch(localPath, { method: "HEAD" });
    if (res.ok) {
      return localPath;
    }
  } catch {
    // bundled model missing — fall through to CDN
  }

  console.warn(
    "[MediaPipe] public/mediapipe/models/face_landmarker.task 없음 → CDN 사용"
  );
  return FACE_LANDMARKER_CDN;
}

export async function initRppgLandmarker({ runningMode = "VIDEO" } = {}) {
  const isExtension =
    typeof chrome !== "undefined" && chrome.runtime?.getURL;

  const wasmBaseUrl = isExtension
    ? chrome.runtime.getURL("mediapipe/wasm/")
    : "/mediapipe/wasm/";

  const modelUrl = await resolveFaceLandmarkerModelUrl(isExtension);

  console.log("[MediaPipe] wasmBaseUrl:", wasmBaseUrl);
  console.log("[MediaPipe] modelUrl:", modelUrl);
  console.log("[MediaPipe] runningMode:", runningMode);

  const visionFileset = await FilesetResolver.forVisionTasks(wasmBaseUrl);

  faceLandmarker = await FaceLandmarker.createFromOptions(visionFileset, {
    baseOptions: {
      modelAssetPath: modelUrl,
      delegate: "CPU",
    },
    runningMode,
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  console.log("[MediaPipe] FaceLandmarker loaded");
}

export function updateRppgFromVideo(videoEl, roiPreviewCanvas = null) {
  if (!faceLandmarker) {
    throw new Error("FaceLandmarker is not initialized");
  }

  const nowMs = performance.now();
  const result = faceLandmarker.detectForVideo(videoEl, nowMs);

  if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
    lastDetScore = 0.25;
    pushSignal(lastSignal);

    return {
      success: false,
      signal: lastSignal,
      bufferLength: rppgBuffer.length,
      landmarks: null,
      roiQuality: 0,
      detScore: lastDetScore,
    };
  }

  const landmarks = result.faceLandmarks[0];
  const { signal, roiQuality } = extractSignal9FromVideo(
    videoEl,
    landmarks,
    roiPreviewCanvas
  );

  lastSignal = signal;
  lastRoiQuality = roiQuality;
  lastDetScore = 0.92;
  pushSignal(signal);

  return {
    success: true,
    signal,
    bufferLength: rppgBuffer.length,
    landmarks,
    roiQuality,
    detScore: lastDetScore,
  };
}

export function updateRppgFromFrame(rgba, width, height) {
  if (!faceLandmarker) {
    throw new Error("FaceLandmarker is not initialized");
  }

  if (!width || !height || rgba.length < width * height * 4) {
    throw new Error("Invalid frame for rPPG extraction.");
  }

  const frameCanvas = document.createElement("canvas");
  frameCanvas.width = width;
  frameCanvas.height = height;
  const frameCtx = frameCanvas.getContext("2d");
  const pixels =
    rgba instanceof Uint8ClampedArray ? rgba : new Uint8ClampedArray(rgba);
  frameCtx.putImageData(new ImageData(pixels, width, height), 0, 0);

  const result = faceLandmarker.detect(frameCanvas);

  if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
    lastDetScore = 0.25;
    pushSignal(lastSignal);

    return {
      success: false,
      signal: lastSignal,
      bufferLength: rppgBuffer.length,
      landmarks: null,
      roiQuality: 0,
      detScore: lastDetScore,
    };
  }

  const landmarks = result.faceLandmarks[0];
  const { signal, roiQuality } = extractSignal9FromRgba(
    rgba,
    width,
    height,
    landmarks
  );

  lastSignal = signal;
  lastRoiQuality = roiQuality;
  lastDetScore = 0.92;
  pushSignal(signal);

  return {
    success: true,
    signal,
    bufferLength: rppgBuffer.length,
    landmarks,
    roiQuality,
    detScore: lastDetScore,
  };
}

function pushSignal(signal) {
  rppgBuffer.push(signal);

  if (rppgBuffer.length > T) {
    rppgBuffer.shift();
  }
}

export function isRppgReady() {
  return rppgBuffer.length >= MIN_RPPG_FRAMES_FOR_INFER;
}

export function isRppgBufferFull() {
  return rppgBuffer.length >= T;
}

export function getRppgBufferInfo() {
  return { bufferLength: rppgBuffer.length };
}

export function resetRppgBuffer() {
  rppgBuffer = [];
  lastSignal = new Array(C).fill(0);
  lastRoiQuality = 0;
  lastDetScore = 0;
}

/** ONNX rppg_window: raw 0..1 (정규화는 그래프 내부) */
export function getRppgWindow() {
  const window = [];

  if (rppgBuffer.length < T) {
    const padCount = T - rppgBuffer.length;

    for (let i = 0; i < padCount; i++) {
      window.push(new Array(C).fill(0));
    }

    for (const sig of rppgBuffer) {
      window.push(sig);
    }
  } else {
    for (const sig of rppgBuffer) {
      window.push(sig);
    }
  }

  const tensor = new Float32Array(T * C);

  for (let t = 0; t < T; t++) {
    for (let c = 0; c < C; c++) {
      tensor[t * C + c] = window[t][c];
    }
  }

  return tensor;
}

/** @deprecated use getRppgWindow */
export function getRppgTensor() {
  return getRppgWindow();
}

function estimateMotionQuality() {
  if (rppgBuffer.length < 4) {
    return 0.35;
  }

  const greens = rppgBuffer.map((s) => (s[1] + s[4] + s[7]) / 3);
  const mean = greens.reduce((a, b) => a + b, 0) / greens.length;
  let variance = 0;

  for (const g of greens) {
    const d = g - mean;
    variance += d * d;
  }

  variance /= greens.length;
  const pulse = Math.min(1, variance * 80);
  return Math.min(1, Math.max(0.2, pulse));
}

/**
 * quality[:,0]=ROI*fill, [:,1]=det, [:,2]=motion*fill
 */
export function getQualityVector() {
  const fill = Math.min(1, rppgBuffer.length / T);

  return new Float32Array([
    lastRoiQuality * fill,
    lastDetScore,
    estimateMotionQuality() * fill,
  ]);
}

function extractSignal9FromRgba(data, width, height, landmarks) {
  const left = meanRgbInPolygon(data, width, height, landmarks, LEFT_CHEEK_IDX);
  const right = meanRgbInPolygon(data, width, height, landmarks, RIGHT_CHEEK_IDX);
  const forehead = meanRgbInPolygon(
    data,
    width,
    height,
    landmarks,
    FOREHEAD_IDX
  );

  let valid = 0;
  if (left[0] + left[1] + left[2] > 0) {
    valid += 1;
  }
  if (right[0] + right[1] + right[2] > 0) {
    valid += 1;
  }
  if (forehead[0] + forehead[1] + forehead[2] > 0) {
    valid += 1;
  }

  return {
    signal: [
      left[0],
      left[1],
      left[2],
      right[0],
      right[1],
      right[2],
      forehead[0],
      forehead[1],
      forehead[2],
    ],
    roiQuality: valid / 3,
  };
}

function extractSignal9FromVideo(videoEl, landmarks, roiPreviewCanvas = null) {
  const width = videoEl.videoWidth;
  const height = videoEl.videoHeight;

  const frameCanvas = document.createElement("canvas");
  frameCanvas.width = width;
  frameCanvas.height = height;

  const frameCtx = frameCanvas.getContext("2d");
  frameCtx.drawImage(videoEl, 0, 0, width, height);

  const imageData = frameCtx.getImageData(0, 0, width, height);
  const extracted = extractSignal9FromRgba(
    imageData.data,
    width,
    height,
    landmarks
  );

  if (roiPreviewCanvas) {
    drawRoiPreview(videoEl, landmarks, roiPreviewCanvas);
  }

  return extracted;
}

function meanRgbInPolygon(data, width, height, landmarks, indices) {
  const points = indices.map((idx) => {
    const p = landmarks[idx];

    return {
      x: Math.round(p.x * width),
      y: Math.round(p.y * height),
    };
  });

  const minX = clamp(Math.floor(Math.min(...points.map((p) => p.x))), 0, width - 1);
  const maxX = clamp(Math.ceil(Math.max(...points.map((p) => p.x))), 0, width - 1);
  const minY = clamp(Math.floor(Math.min(...points.map((p) => p.y))), 0, height - 1);
  const maxY = clamp(Math.ceil(Math.max(...points.map((p) => p.y))), 0, height - 1);

  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let count = 0;

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (!pointInPolygon(x, y, points)) {
        continue;
      }

      const pixelIndex = (y * width + x) * 4;

      sumR += data[pixelIndex];
      sumG += data[pixelIndex + 1];
      sumB += data[pixelIndex + 2];
      count += 1;
    }
  }

  if (count === 0) {
    return [0, 0, 0];
  }

  return [sumR / count / 255, sumG / count / 255, sumB / count / 255];
}

function pointInPolygon(x, y, polygon) {
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    const intersect =
      yi > y !== yj > y &&
      x < ((xj - xi) * (y - yi)) / ((yj - yi) + 1e-6) + xi;

    if (intersect) {
      inside = !inside;
    }
  }

  return inside;
}

function drawRoiPreview(videoEl, landmarks, canvas) {
  const width = videoEl.videoWidth;
  const height = videoEl.videoHeight;

  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, width, height);

  drawPolygon(ctx, landmarks, LEFT_CHEEK_IDX, width, height, "red");
  drawPolygon(ctx, landmarks, RIGHT_CHEEK_IDX, width, height, "lime");
  drawPolygon(ctx, landmarks, FOREHEAD_IDX, width, height, "cyan");
}

function drawPolygon(ctx, landmarks, indices, width, height, color) {
  const points = indices.map((idx) => ({
    x: landmarks[idx].x * width,
    y: landmarks[idx].y * height,
  }));

  ctx.beginPath();

  points.forEach((p, i) => {
    if (i === 0) {
      ctx.moveTo(p.x, p.y);
    } else {
      ctx.lineTo(p.x, p.y);
    }
  });

  ctx.closePath();
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}