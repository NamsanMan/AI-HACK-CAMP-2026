import "./style.css";

import {
  loadOnnxModel,
  runInferenceWithInputs,
} from "./inference.js";

import {
  videoFrameToFaceTensor,
} from "./preprocess.js";

import {
  initRppgLandmarker,
  updateRppgFromVideo,
  getRppgWindow,
  getQualityVector,
  isRppgReady,
} from "./rppg.js";

import { getRiskStatus } from "./thresholds.js";

document.querySelector("#app").innerHTML = `
  <div>
    <h1>Video File + ONNX + rPPG Test</h1>

    <div style="margin-bottom: 12px;">
      <button id="loadBtn">Load ONNX</button>
      <button id="mpBtn">Load MediaPipe</button>
      <button id="loopBtn">Start Video Loop Inference</button>
      <button id="stopBtn">Stop</button>
    </div>

    <video id="sampleVideo" width="480" controls muted>
      <source src="/sample/992_980.mp4" type="video/mp4" />
    </video>

    <div id="statusCard" class="status-card status-waiting">
      <div class="status-title">분석 대기 중</div>
      <div class="status-message">ONNX와 MediaPipe를 로드한 뒤 영상을 재생하세요.</div>

      <div class="metric-row">
        <span>Fake Probability</span>
        <strong id="fakeProbText">-</strong>
      </div>

      <div class="metric-row">
        <span>Liveness Score</span>
        <strong id="livenessText">-</strong>
      </div>

      <div class="metric-row">
        <span>Confidence</span>
        <strong id="confidenceText">-</strong>
      </div>

      <div class="metric-row">
        <span>rPPG Buffer</span>
        <strong id="rppgText">-</strong>
      </div>
    </div>

    <div style="margin-top: 12px;">
      <p>rPPG ROI preview</p>
      <canvas id="roiPreview"
        style="border: 1px solid #00f; width: 480px;"></canvas>
    </div>

    <pre id="result">Waiting...</pre>
  </div>
`;

const resultEl = document.querySelector("#result");
const videoEl = document.querySelector("#sampleVideo");
const roiPreviewEl = document.querySelector("#roiPreview");

const statusCardEl = document.querySelector("#statusCard");
const fakeProbTextEl = document.querySelector("#fakeProbText");
const livenessTextEl = document.querySelector("#livenessText");
const confidenceTextEl = document.querySelector("#confidenceText");
const rppgTextEl = document.querySelector("#rppgText");

let loopTimer = null;
let isRunning = false;

document.querySelector("#loadBtn").addEventListener("click", async () => {
  try {
    resultEl.textContent = "Loading ONNX model...";
    await loadOnnxModel();
    setOverlayMessage("초기화 중", "ONNX 모델 로드 완료. MediaPipe를 로드하는 중입니다.");
    resultEl.textContent = "ONNX model loaded successfully.";
  } catch (err) {
    console.error(err);
    resultEl.textContent = "ONNX load failed:\n" + err.message;
  }
});

document.querySelector("#mpBtn").addEventListener("click", async () => {
  try {
    resultEl.textContent = "Loading MediaPipe FaceLandmarker...";
    await initRppgLandmarker();
    setOverlayMessage("대기 중", "ONNX와 MediaPipe 로드 완료. 영상통화 화면을 찾는 중입니다.");
    resultEl.textContent = "MediaPipe FaceLandmarker loaded successfully.";
  } catch (err) {
    console.error(err);
    resultEl.textContent = "MediaPipe load failed:\n" + err.message;
  }
});

document.querySelector("#loopBtn").addEventListener("click", async () => {
  try {
    await waitForVideoReady(videoEl);

    if (videoEl.paused) {
      await videoEl.play();
    }

    if (isRunning) {
      return;
    }

    isRunning = true;
    resultEl.textContent = "Loop inference started...";

    loopTimer = setInterval(async () => {
      try {
        if (videoEl.paused || videoEl.ended) {
          return;
        }

        /*
          Artifact branch input:
          전체 video frame -> 96x96 resize -> ImageNet normalize
          preview는 제거했으므로 두 번째 인자는 넘기지 않음.
        */
        const rppgInfo = updateRppgFromVideo(videoEl, roiPreviewEl);
        const faceCrop = videoFrameToFaceTensor(videoEl, rppgInfo.landmarks);
        if (!isRppgReady()) {
          return;
        }

        const rppgWindow = getRppgWindow();
        const quality = getQualityVector();
        const result = await runInferenceWithInputs(rppgWindow, faceCrop, quality);

        const ready = isRppgReady();
        const status = updateStatusUI(result, ready, rppgInfo);

        resultEl.textContent =
          `time: ${videoEl.currentTime.toFixed(2)} sec\n` +
          `riskLevel: ${status.riskLevel}\n` +
          `trustLevel: ${status.trustLevel}\n` +
          `message: ${status.message}\n` +
          `rPPG success: ${rppgInfo.success}\n` +
          `rPPG buffer: ${rppgInfo.bufferLength}/90\n` +
          `rPPG ready: ${ready}\n` +
          JSON.stringify(result, null, 2);

        console.log("loop result:", result);
      } catch (err) {
        console.error("Loop inference error:", err);
      }
    }, 1000);
  } catch (err) {
    console.error(err);
    resultEl.textContent = "Loop inference failed:\n" + err.message;
  }
});

document.querySelector("#stopBtn").addEventListener("click", () => {
  if (loopTimer) {
    clearInterval(loopTimer);
    loopTimer = null;
  }

  isRunning = false;
  resultEl.textContent = "Loop inference stopped.";
});

function waitForVideoReady(videoEl) {
  return new Promise((resolve, reject) => {
    if (videoEl.readyState >= 2) {
      resolve();
      return;
    }

    const timeout = setTimeout(() => {
      reject(new Error("Video is not ready. The file may be unsupported by Chrome."));
    }, 3000);

    videoEl.addEventListener(
      "loadeddata",
      () => {
        clearTimeout(timeout);
        resolve();
      },
      { once: true }
    );
  });
}

function updateStatusUI(result, ready, rppgInfo) {
  const status = getRiskStatus(result, ready);

  statusCardEl.className = `status-card status-${status.riskLevel}`;

  statusCardEl.querySelector(".status-title").textContent = status.title;
  statusCardEl.querySelector(".status-message").textContent = status.message;

  fakeProbTextEl.textContent = result.fakeProb.toFixed(3);
  livenessTextEl.textContent = result.livenessScore.toFixed(3);
  confidenceTextEl.textContent = result.confidence.toFixed(3);
  rppgTextEl.textContent = `${rppgInfo.bufferLength}/90`;

  return status;
}