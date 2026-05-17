import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const path = resolve(dirname(fileURLToPath(import.meta.url)), "../src/content.js");
let src = readFileSync(path, "utf8");

const replacements = [
  [
    /\/\*\* rPPG 버퍼 채우[^*]+\*\//,
    "/** rPPG 버퍼 채우기. 90샘플 ≈ 9초 @ 100ms */",
  ],
  [
    /\/\*\* ONNX [^*]+\*\//,
    "/** ONNX 점수 갱신 주기 (버퍼가 찬 뒤) */",
  ],
  [
    /\/\*\* Zoom:[^*]+\*\//,
    "/** Zoom: 영상 탐색만 (무거운 Shadow DOM walk 빈도 제한) */",
  ],
  [
    /return "[^"]*video element[^"]*";/,
    'return "페이지에서 video element를 찾는 중입니다.";',
  ],
  [
    /setOverlayMessage\("초기[^"]*", "분석[^"]*"\);/,
    'setOverlayMessage("초기화 중", "분석 엔진을 준비하는 중입니다 (Offscreen WASM)...");',
  ],
  [
    /setOverlayMessage\("ONNX [^"]*", "Offscreen[^"]*"\);/,
    'setOverlayMessage("ONNX 확인", "Offscreen 파이프라인 초기화 성공.");',
  ],
  [
    /setOverlayMessage\("[^"]*", "영상 화면[^"]*"\);/,
    'setOverlayMessage("대기 중", "영상 화면을 찾는 중입니다.");',
  ],
  [/  \/\/ [^\n]*manifest[^\n]*/g, "  // 일반 사이트: 최상위 프레임만 (manifest all_frames: false)"],
  [
    /npm run build [^`]+`/g,
    "npm run build 후 확장 프로그램·페이지를 새로고침하세요.`",
  ],
  [
    /setOverlayMessage\("초기[^"]*패", failMsg\);/,
    'setOverlayMessage("초기화 실패", failMsg);',
  ],
  [
    /setOverlayMessage\("[^"]*", "Zoom 미팅[^"]*"\);/,
    'setOverlayMessage("대기 중", "Zoom 미팅 영상을 찾는 중입니다.");',
  ],
  [
    /title: "[^"]*",\s*\n\s*`\$\{formatSearchHint/,
    'title: "대기 중",\n          `${formatSearchHint',
  ],
  [
    /setOverlayMessage\("[^"]*", waitingForMediaMessage\(\)\);/,
    'setOverlayMessage("대기 중", waitingForMediaMessage());',
  ],
  [
    /setOverlayMessage\("[^"]*", "video가[^"]*"\);/,
    'setOverlayMessage("대기 중", "video가 아직 준비되지 않았습니다.");',
  ],
  [
    /"\?[^"]*video\/canvas[^"]*",/,
    '"영상(video/canvas)을 찾는 중입니다.",',
  ],
  [
    /`\?[^`]*video \$\{info\.videoCount\}[^`]*`/,
    "`프레임: video ${info.videoCount} (스트림 ${info.withStream}), canvas ${info.canvasCount}, player ${info.playerCount ?? 0}`",
  ],
  [/lines\.push\("iframe[^"]*"\);/, 'lines.push("iframe 내부 탐색 중");'],
  [
    /lines\.push\(\s*"미팅[^"]*"\s*\);/,
    'lines.push("미팅 입장 후 상대/본인 영상이 켜져 있어야 합니다. (Shadow DOM·iframe 탐색 중)");',
  ],
  [
    /throw new Error\("캡처[^"]*"\);/,
    'throw new Error("캡처된 프레임이 비어 있습니다.");',
  ],
  [
    /`캡처 버퍼 [^`]*`/,
    "`캡처 버퍼 크기 오류: ${imageData.data.length} / ${expected}`",
  ],
  [
    /throw new Error\(\s*"[^"]*CORS[^"]*"\s*\);/,
    "throw new Error(\n        \"영상 프레임을 읽을 수 없습니다 (CORS). crossOrigin='anonymous' 영상만 분석 가능합니다.\"\n      );",
  ],
  [
    /throw new Error\(\s*"canvas[^"]*"\s*\);/,
    'throw new Error(\n        "canvas 프레임을 읽을 수 없습니다. Zoom 등은 canvas 대신 video를 쓰는 경우가 있습니다."\n      );',
  ],
  [
    /title: "분석 [^"]*",/,
    'title: "분석 시작",',
  ],
  [
    /message: `\$\{kind\} [^`]*`,/,
    "message: `${kind} 프레임을 분석하는 중입니다.`,",
  ],
  [
    /title: "[^"]*",\s*\n\s*message: waitingForMediaMessage/,
    'title: "대기 중",\n            message: waitingForMediaMessage',
  ],
  [
    /title: "분석 [^"]*",\s*\n\s*message: msg,/,
    'title: "분석 오류",\n          message: msg,',
  ],
];

for (const [re, rep] of replacements) {
  src = src.replace(re, rep);
}

// Remove inlined overlay block (moved to overlayUi.js)
const start = src.indexOf("const OVERLAY_PANEL_HTML");
const end = src.indexOf("if (window.__deepfakeDetectorStarted)");

if (start !== -1 && end !== -1) {
  src = `${src.slice(0, start)}\n${src.slice(end)}`;
}

if (!src.includes('from "./overlayUi.js"')) {
  src = src.replace(
    'import { formatHr, getRiskStatus, resetRiskHysteresis } from "./thresholds.js";',
    `import { getRiskStatus, resetRiskHysteresis } from "./thresholds.js";
import {
  createOverlay,
  hasOverlay,
  setOverlayMessage,
  updateOverlay,
} from "./overlayUi.js";`
  );

  src = src.replace(
    `import {
  drawScoreChart,
  recordScore,
  resetScoreHistory,
} from "./scoreChart.js";`,
    'import { resetScoreHistory } from "./scoreChart.js";'
  );

  src = src.replace(
    `import { formatHr, getRiskStatus, resetRiskHysteresis } from "./thresholds.js";
import {
  createOverlay,
  hasOverlay,
  setOverlayMessage,
  updateOverlay,
} from "./overlayUi.js";`,
    `import { getRiskStatus, resetRiskHysteresis } from "./thresholds.js";
import {
  createOverlay,
  hasOverlay,
  setOverlayMessage,
  updateOverlay,
} from "./overlayUi.js";`
  );
}

src = src.replace(
  /let overlayEl = null;\nlet dotEl = null;\nlet panelEl = null;\nlet titleEl = null;\nlet messageEl = null;\nlet fakeEl = null;\nlet liveEl = null;\nlet hrEl = null;\nlet rppgEl = null;\nlet scoreChartEl = null;\n\n/,
  ""
);

src = src.replace(
  /if \(!status \|\| !overlayEl\)/g,
  "if (!status || !hasOverlay())"
);

src = src.replace(
  /if \(overlayEl\) \{\s*updateOverlay/g,
  "if (hasOverlay()) {\n      updateOverlay"
);

writeFileSync(path, src, "utf8");
console.log("[restore-content-ko] done");
