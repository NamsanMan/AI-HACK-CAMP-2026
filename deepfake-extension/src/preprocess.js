/**
 * @deprecated Use faceAlign.js — re-export for compatibility.
 */
export {
  rgbaFrameToFaceTensor,
  videoFrameToFaceTensor,
} from "./faceAlign.js";

import {
  FACE_INPUT_SIZE,
  RPPG_WINDOW_FRAMES,
  RPPG_CHANNELS,
} from "./modelConstants.js";

export function makeDummyRppgTensor() {
  const tensor = new Float32Array(
    1 * RPPG_WINDOW_FRAMES * RPPG_CHANNELS
  );

  return tensor;
}
