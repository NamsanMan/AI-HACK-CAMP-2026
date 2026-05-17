import { initPipeline, processVideoFrame } from "./extensionClient.js";
import { getRiskStatus, resetRiskHysteresis } from "./thresholds.js";
import {
  createOverlay,
  hasOverlay,
  setOverlayMessage,
  updateOverlay,
  resetOverlayState,
} from "./overlayUi.js";
import {
  describeMediaSearch,
  findBestMediaSource,
  isMediaSourceReady,
} from "./videoSource.js";
import { isZoomHost } from "./zoomHosts.js";
import { resetScoreHistory } from "./scoreChart.js";

const ZOOM_MODE = isZoomHost();

console.log("[Deepfake Extension] content.js loaded", {
  frame: window.location.href,
  isTop: window === window.top,
  zoomMode: ZOOM_MODE,
});

const ONNX_ONLY_TEST = false;
const MAX_CAPTURE_WIDTH = 320;
/** rPPG 버퍼 채우기. 90샘플 ≈ 9초 @ 100ms */
const RPPG_SAMPLE_INTERVAL_MS = 100;
/** ONNX 점수 갱신 주기 (버퍼가 찬 뒤) */
const INFERENCE_INTERVAL_MS = 1000;
/** Zoom: 영상 탐색만 (무거운 Shadow DOM walk 빈도 제한) */
const ZOOM_DISCOVER_INTERVAL_MS = 2500;
const DF_OVERLAY_MSG = "deepfake-extension-overlay-v1";
const LEADER_CHANNEL = "deepfake-detector-leader-v1";
const LEADER_WAIT_MS = 200;

let inferenceSourceKey = null;
let isRunning = false;
let sampleTimer = null;
let lastInferAt = 0;
let lastInferenceResult = null;
let pipelineReady = false;

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });

function findMainVideoSimple() {
  const videos = Array.from(document.querySelectorAll("video"));
  const validVideos = videos.filter(
    (v) => v.videoWidth > 0 && v.videoHeight > 0 && !v.ended
  );

  if (validVideos.length === 0) {
    return null;
  }

  validVideos.sort(
    (a, b) =>
      b.videoWidth * b.videoHeight - a.videoWidth * a.videoHeight
  );

  const el = validVideos[0];
  return {
    type: "video",
    el,
    area: el.videoWidth * el.videoHeight,
  };
}

function findMediaSource() {
  if (ZOOM_MODE) {
    return findBestMediaSource();
  }

  return findMainVideoSimple();
}

function isMediaReady(source) {
  if (!source?.el) {
    return false;
  }

  if (ZOOM_MODE) {
    return isMediaSourceReady(source);
  }

  if (source.type === "canvas") {
    return source.el.width > 0 && source.el.height > 0;
  }

  const v = source.el;
  return v.readyState >= 2 && !v.ended;
}

function waitingForMediaMessage() {
  if (ZOOM_MODE) {
    return formatSearchHint();
  }

  return "페이지에서 video element를 찾는 중입니다.";
}

async function bootAnalyzer(initialSource = null) {
  if (window === window.top) {
    setOverlayMessage("초기화 중", "분석 엔진을 준비하는 중입니다 (Offscreen WASM)...");
  }

  await initPipeline();

  if (ONNX_ONLY_TEST) {
    if (window === window.top) {
      setOverlayMessage("ONNX 확인", "Offscreen 파이프라인 초기화 성공.");
    }
    return;
  }

  pipelineReady = true;

  if (initialSource) {
    startInferenceLoop(initialSource);
    return;
  }

  if (window === window.top) {
    setOverlayMessage("대기 중", "영상 화면을 찾는 중입니다.");
  }

  waitAndStartStandard();
}

async function main() {
  // 일반 사이트: 최상위 프레임만 (manifest all_frames: false)
  if (!ZOOM_MODE && window !== window.top) {
    return;
  }

  if (ZOOM_MODE && window === window.top) {
    createOverlay();
    installOverlayBridge();
  }

  if (!ZOOM_MODE) {
    createOverlay();
  }

  try {
    if (ZOOM_MODE) {
      runZoomMediaProbe();
      return;
    }

    await bootAnalyzer();
  } catch (err) {
    console.error("[Deepfake Extension] init error:", err);
    const failMsg = `${err?.message || err}\n\nnpm run build 후 확장 프로그램·페이지를 새로고침하세요.`;
    if (window === window.top) {
      setOverlayMessage("초기화 실패", failMsg);
    }
  }
}

function runZoomMediaProbe() {
  if (window === window.top) {
    setOverlayMessage("대기 중", "Zoom 미팅 영상을 찾는 중입니다.");
  }

  const tick = async () => {
    if (isRunning) {
      return true;
    }

    const source = findBestMediaSource();
    if (!source) {
      if (window === window.top) {
        setOverlayMessage("대기 중", formatSearchHint());
      }
      return false;
    }

    if (!isMediaSourceReady(source)) {
      if (window === window.top) {
        setOverlayMessage(
          "대기 중",
          `${formatSearchHint()}\n\n영상 요소는 보이지만 스트림이 아직 없습니다.`
        );
      }
      return false;
    }

    return tryStartAsZoomLeader();
  };

  const finder = setInterval(() => {
    tick().then((started) => {
      if (started) {
        clearInterval(finder);
      }
    });
  }, ZOOM_DISCOVER_INTERVAL_MS);

  tick();
}

async function tryStartAsZoomLeader() {
  if (isRunning) {
    return true;
  }

  const source = findBestMediaSource();
  if (!source || !isMediaSourceReady(source)) {
    return false;
  }

  const isLeader = await electLeader(source.area);
  if (!isLeader) {
    return false;
  }

  try {
    await bootAnalyzer(source);
    return true;
  } catch (err) {
    console.error("[Deepfake Extension] zoom leader boot error:", err);
    return false;
  }
}

function waitAndStartStandard() {
  const finder = setInterval(() => {
    const source = findMainVideoSimple();

    if (!source) {
      setOverlayMessage("대기 중", waitingForMediaMessage());
      return;
    }

    if (source.el.readyState < 2) {
      setOverlayMessage("대기 중", "video가 아직 준비되지 않았습니다.");
      return;
    }

    clearInterval(finder);
    startInferenceLoop(source);
  }, 1000);
}

function installOverlayBridge() {
  window.addEventListener("message", (event) => {
    if (event.data?.type !== DF_OVERLAY_MSG) {
      return;
    }

    const { status, result, ready, rppgInfo } = event.data;
    if (!status || !hasOverlay()) {
      return;
    }

    updateOverlay(status, result, ready, rppgInfo);
  });
}

function publishOverlayUpdate(status, result, ready, rppgInfo) {
  if (window === window.top) {
    if (hasOverlay()) {
      updateOverlay(status, result, ready, rppgInfo);
    }
    return;
  }

  window.top.postMessage(
    {
      type: DF_OVERLAY_MSG,
      status,
      result,
      ready,
      rppgInfo,
    },
    "*"
  );
}

function mediaSourceKey(source) {
  if (!source) {
    return null;
  }
  // DOM 엘리먼트 객체 자체를 반환하여 참조(Reference) 비교가 가능하도록 수정
  return source.el;
}

function formatSearchHint() {
  const info = describeMediaSearch();
  const lines = [
    "영상(video/canvas)을 찾는 중입니다.",
    `프레임: video ${info.videoCount} (스트림 ${info.withStream}), canvas ${info.canvasCount}, player ${info.playerCount ?? 0}`,
  ];

  if (info.inIframe) {
    lines.push("iframe 내부 탐색 중");
  } else if (
    info.videoCount === 0 &&
    info.canvasCount === 0 &&
    (info.playerCount ?? 0) === 0
  ) {
    lines.push("미팅 입장 후 상대/본인 영상이 켜져 있어야 합니다. (Shadow DOM·iframe 탐색 중)");
  }

  return lines.join("\n");
}

function electLeader(myArea) {
  return new Promise((resolve) => {
    let bc;

    try {
      bc = new BroadcastChannel(LEADER_CHANNEL);
    } catch {
      resolve(true);
      return;
    }

    const claims = [{ area: myArea, self: true }];

    const onMessage = (event) => {
      if (event.data?.type === "claim" && typeof event.data.area === "number") {
        claims.push({ area: event.data.area, self: false });
      }
    };

    bc.addEventListener("message", onMessage);
    bc.postMessage({ type: "claim", area: myArea });

    setTimeout(() => {
      bc.removeEventListener("message", onMessage);
      bc.close();

      const maxArea = Math.max(...claims.map((c) => c.area));
      const myMax = claims.filter((c) => c.self)[0]?.area ?? 0;
      resolve(myMax > 0 && myMax >= maxArea);
    }, LEADER_WAIT_MS);
  });
}

function captureVideoFrame(videoEl) {
  const srcW = videoEl.videoWidth;
  const srcH = videoEl.videoHeight;

  let drawW = srcW;
  let drawH = srcH;

  if (drawW > MAX_CAPTURE_WIDTH) {
    const scale = MAX_CAPTURE_WIDTH / drawW;
    drawW = MAX_CAPTURE_WIDTH;
    drawH = Math.max(1, Math.round(srcH * scale));
  }

  drawW = Math.max(1, drawW);
  drawH = Math.max(1, drawH);

  captureCanvas.width = drawW;
  captureCanvas.height = drawH;

  try {
    captureCtx.drawImage(videoEl, 0, 0, drawW, drawH);
    const imageData = captureCtx.getImageData(0, 0, drawW, drawH);
    const expected = drawW * drawH * 4;

    if (!imageData.data.length) {
      throw new Error("캡처된 프레임이 비어 있습니다.");
    }

    if (imageData.data.length !== expected) {
      throw new Error(
        `캡처 버퍼 크기 오류: ${imageData.data.length} / ${expected}`
      );
    }

    return {
      rgba: new Uint8ClampedArray(imageData.data),
      width: drawW,
      height: drawH,
    };
  } catch (err) {
    if (err?.name === "SecurityError" || err instanceof DOMException) {
      throw new Error(
        "영상 프레임을 읽을 수 없습니다 (CORS). crossOrigin='anonymous' 영상만 분석 가능합니다."
      );
    }
    throw err;
  }
}

function captureCanvasFrame(canvasEl) {
  const srcW = canvasEl.width || canvasEl.clientWidth;
  const srcH = canvasEl.height || canvasEl.clientHeight;

  let drawW = srcW;
  let drawH = srcH;

  if (drawW > MAX_CAPTURE_WIDTH) {
    const scale = MAX_CAPTURE_WIDTH / drawW;
    drawW = MAX_CAPTURE_WIDTH;
    drawH = Math.max(1, Math.round(srcH * scale));
  }

  drawW = Math.max(1, drawW);
  drawH = Math.max(1, drawH);

  captureCanvas.width = drawW;
  captureCanvas.height = drawH;

  try {
    captureCtx.drawImage(canvasEl, 0, 0, drawW, drawH);
    const imageData = captureCtx.getImageData(0, 0, drawW, drawH);

    return {
      rgba: new Uint8ClampedArray(imageData.data),
      width: drawW,
      height: drawH,
    };
  } catch (err) {
    if (err?.name === "SecurityError" || err instanceof DOMException) {
      throw new Error(
        "canvas 프레임을 읽을 수 없습니다. Zoom 등은 canvas 대신 video를 쓰는 경우가 있습니다."
      );
    }
    throw err;
  }
}

function captureFromSource(source) {
  if (source.type === "canvas") {
    return captureCanvasFrame(source.el);
  }

  return captureVideoFrame(source.el);
}

function startInferenceLoop(initialSource) {
  if (isRunning) {
    return;
  }

  isRunning = true;
  inferenceSourceKey = mediaSourceKey(initialSource);

  const kind = initialSource.type === "canvas" ? "canvas" : "video";
  publishOverlayUpdate(
    {
      riskLevel: "warming",
      trustLevel: "not_ready",
      title: "분석 시작",
      message: `${kind} 프레임을 분석하는 중입니다.`,
    },
    null,
    false,
    null
  );

  sampleTimer = setInterval(async () => {
    if (!pipelineReady) {
      return;
    }

    try {
      const found = findMediaSource();

      if (!found || !isMediaReady(found)) {
        lastInferenceResult = null;
        lastInferAt = 0;
        publishOverlayUpdate(
          {
            riskLevel: "warming",
            trustLevel: "not_ready",
            title: "대기 중",
            message: waitingForMediaMessage(),
          },
          null,
          false,
          null
        );
        return;
      }

      const key = mediaSourceKey(found);
      const resetRppg = key !== inferenceSourceKey;

      if (resetRppg) {
        inferenceSourceKey = key;
        lastInferenceResult = null;
        lastInferAt = 0;
        resetScoreHistory();
        resetRiskHysteresis();
        resetOverlayState();
      }

      const frame = captureFromSource(found);
      const rppgReply = await processVideoFrame(
        frame.rgba,
        frame.width,
        frame.height,
        { resetRppg, infer: false }
      );

      const ready = rppgReply.ready;
      const rppgInfo = rppgReply.rppgInfo;
      const now = Date.now();
      const shouldInfer =
        ready &&
        (lastInferAt === 0 || now - lastInferAt >= INFERENCE_INTERVAL_MS);

      if (shouldInfer) {
        const inferReply = await processVideoFrame(
          frame.rgba,
          frame.width,
          frame.height,
          { infer: true, skipRppgUpdate: true }
        );

        if (inferReply.result) {
          lastInferenceResult = inferReply.result;
          lastInferAt = now;
        }
      }

      if (!ready) {
        publishOverlayUpdate(
          getRiskStatus(null, false),
          null,
          false,
          rppgInfo
        );
        return;
      }

      const status = getRiskStatus(lastInferenceResult, true);
      publishOverlayUpdate(status, lastInferenceResult, true, rppgInfo);
    } catch (err) {
      const msg = err?.message || String(err);
      console.error("[Deepfake Extension] inference loop error:", err);
      publishOverlayUpdate(
        {
          riskLevel: "warming",
          trustLevel: "not_ready",
          title: "분석 오류",
          message: msg,
        },
        null,
        false,
        null
      );
    }
  }, RPPG_SAMPLE_INTERVAL_MS);
}


if (window.__deepfakeDetectorStarted) {
  console.log("[Deepfake Extension] already started. Skip.");
} else {
  window.__deepfakeDetectorStarted = true;
  main();
}
