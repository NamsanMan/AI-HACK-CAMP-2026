/**
 * camp aligned_face_crop + to_face_tensor (RGB 0~1, no ImageNet).
 */

import { FACE_INPUT_SIZE } from "./modelConstants.js";

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

/**
 * @param {import("@mediapipe/tasks-vision").NormalizedLandmark[]} landmarks
 */
export function boundingBoxFromLandmarks(
  landmarks,
  width,
  height,
  padX = 0.15,
  padY = 0.2
) {
  if (!landmarks?.length) {
    return null;
  }

  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;

  for (const p of landmarks) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  let x1 = Math.floor(minX * width);
  let y1 = Math.floor(minY * height);
  let x2 = Math.ceil(maxX * width);
  let y2 = Math.ceil(maxY * height);

  const bw = x2 - x1;
  const bh = y2 - y1;

  if (bw <= 0 || bh <= 0) {
    return null;
  }

  x1 = Math.max(0, x1 - Math.floor(bw * padX));
  y1 = Math.max(0, y1 - Math.floor(bh * padY));
  x2 = Math.min(width, x2 + Math.floor(bw * padX));
  y2 = Math.min(height, y2 + Math.floor(bh * padY));

  if (x2 <= x1 || y2 <= y1) {
    return null;
  }

  return { x1, y1, x2, y2 };
}

function rgbaToFaceTensorFromCrop(rgba, frameW, frameH, bbox, faceSize = FACE_INPUT_SIZE) {
  const { x1, y1, x2, y2 } = bbox;
  const cropW = x2 - x1;
  const cropH = y2 - y1;

  const src = document.createElement("canvas");
  src.width = frameW;
  src.height = frameH;
  const srcCtx = src.getContext("2d");
  const pixels =
    rgba instanceof Uint8ClampedArray ? rgba : new Uint8ClampedArray(rgba);
  srcCtx.putImageData(new ImageData(pixels, frameW, frameH), 0, 0);

  const dst = document.createElement("canvas");
  dst.width = faceSize;
  dst.height = faceSize;
  const dstCtx = dst.getContext("2d");
  dstCtx.drawImage(src, x1, y1, cropW, cropH, 0, 0, faceSize, faceSize);

  const data = dstCtx.getImageData(0, 0, faceSize, faceSize).data;
  const tensor = new Float32Array(1 * 3 * faceSize * faceSize);

  for (let y = 0; y < faceSize; y++) {
    for (let x = 0; x < faceSize; x++) {
      const i = (y * faceSize + x) * 4;
      const hw = y * faceSize + x;
      tensor[0 * faceSize * faceSize + hw] = data[i] / 255;
      tensor[1 * faceSize * faceSize + hw] = data[i + 1] / 255;
      tensor[2 * faceSize * faceSize + hw] = data[i + 2] / 255;
    }
  }

  return tensor;
}

function fullFrameRgbaToFaceTensor(rgba, frameW, frameH, faceSize = FACE_INPUT_SIZE) {
  const bbox = { x1: 0, y1: 0, x2: frameW, y2: frameH };
  return rgbaToFaceTensorFromCrop(rgba, frameW, frameH, bbox, faceSize);
}

/**
 * @param {import("@mediapipe/tasks-vision").NormalizedLandmark[] | null} landmarks
 */
export function rgbaFrameToFaceTensor(rgba, width, height, landmarks = null) {
  if (!width || !height) {
    throw new Error("Frame size is not available.");
  }

  const expected = width * height * 4;
  const pixels =
    rgba instanceof Uint8ClampedArray ? rgba : new Uint8ClampedArray(rgba);

  if (pixels.length !== expected) {
    throw new Error(
      `Face tensor input size mismatch: ${pixels.length} vs ${expected}`
    );
  }

  if (landmarks?.length) {
    const bbox = boundingBoxFromLandmarks(landmarks, width, height);
    if (bbox) {
      return rgbaToFaceTensorFromCrop(pixels, width, height, bbox);
    }
  }

  return fullFrameRgbaToFaceTensor(pixels, width, height);
}

/**
 * @param {HTMLVideoElement} videoEl
 * @param {import("@mediapipe/tasks-vision").NormalizedLandmark[] | null} landmarks
 */
export function videoFrameToFaceTensor(videoEl, landmarks = null) {
  const videoWidth = videoEl.videoWidth;
  const videoHeight = videoEl.videoHeight;

  if (!videoWidth || !videoHeight) {
    throw new Error("Video size is not available yet.");
  }

  const frameCanvas = document.createElement("canvas");
  frameCanvas.width = videoWidth;
  frameCanvas.height = videoHeight;
  const frameCtx = frameCanvas.getContext("2d");
  frameCtx.drawImage(videoEl, 0, 0, videoWidth, videoHeight);
  const imageData = frameCtx.getImageData(0, 0, videoWidth, videoHeight);

  return rgbaFrameToFaceTensor(
    imageData.data,
    videoWidth,
    videoHeight,
    landmarks
  );
}
