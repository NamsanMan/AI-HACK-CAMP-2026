import { loadOnnxModel, runInferenceWithInputs } from "./inference.js";
import { decodeRgbaPayload } from "./frameUtils.js";
import { MODEL_FILENAME } from "./modelConstants.js";
import { rgbaFrameToFaceTensor } from "./faceAlign.js";
import {
  initRppgLandmarker,
  updateRppgFromFrame,
  getRppgWindow,
  getQualityVector,
  isRppgReady,
  resetRppgBuffer,
  getRppgBufferInfo,
} from "./rppg.js";

let pipelineReady = false;
let initPromise = null;

async function ensurePipeline() {
  if (pipelineReady) {
    return;
  }

  if (initPromise) {
    await initPromise;
    return;
  }

  initPromise = (async () => {
    const modelUrl = chrome.runtime.getURL(`models/${MODEL_FILENAME}`);
    await loadOnnxModel(modelUrl);
    await initRppgLandmarker({ runningMode: "IMAGE" });
    pipelineReady = true;
    console.log("[Offscreen] pipeline ready");
  })();

  await initPromise;
}

let lastLatestGreen = 0.5;

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "offscreen") {
    return false;
  }

  (async () => {
    try {
      if (message.type === "DF_INIT") {
        await ensurePipeline();
        sendResponse({ ok: true });
        return;
      }

      if (message.type === "DF_PROCESS_FRAME") {
        await ensurePipeline();

        if (message.resetRppg) {
          resetRppgBuffer();
          lastLatestGreen = 0.5;
        }

        const { rgba, width, height } = decodeRgbaPayload(message);
        const infer = message.infer !== false;
        const skipRppgUpdate = message.skipRppgUpdate === true;

        let rppgFrame;
        if (skipRppgUpdate) {
          rppgFrame = {
            bufferLength: getRppgBufferInfo().bufferLength,
            landmarks: null,
            success: false,
          };
        } else {
          rppgFrame = updateRppgFromFrame(rgba, width, height);
          if (rppgFrame.success && rppgFrame.signal) {
            lastLatestGreen = (rppgFrame.signal[1] + rppgFrame.signal[4] + rppgFrame.signal[7]) / 3;
          }
        }

        const rppgInfo = {
          bufferLength: rppgFrame.bufferLength,
          success: rppgFrame.success,
          latestGreen: lastLatestGreen,
        };
        const ready = isRppgReady();

        if (!infer || !ready) {
          sendResponse({
            ok: true,
            ready,
            rppgInfo,
            result: null,
          });
          return;
        }

        const faceCrop = rgbaFrameToFaceTensor(
          rgba,
          width,
          height,
          rppgFrame.landmarks
        );
        const rppgWindow = getRppgWindow();
        const quality = getQualityVector();
        const result = await runInferenceWithInputs(rppgWindow, faceCrop, quality);

        sendResponse({
          ok: true,
          ready: true,
          rppgInfo,
          result,
        });
        return;
      }

      sendResponse({ ok: false, error: `Unknown message type: ${message.type}` });
    } catch (err) {
      console.error("[Offscreen] error:", err);
      sendResponse({
        ok: false,
        error: err?.message || String(err),
      });
    }
  })();

  return true;
});

console.log("[Offscreen] document loaded");
