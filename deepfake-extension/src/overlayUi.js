import { formatHr } from "./thresholds.js";
import { drawScoreChart, recordScore, recordRppgSignal } from "./scoreChart.js";

const RPPG_WINDOW_FRAMES = 90;
const HIGH_RISK_PANEL_HOLD_MS = 3000;

let overlayEl = null;
let dotEl = null;
let titleEl = null;
let messageEl = null;
let fakeEl = null;
let liveEl = null;
let hrEl = null;
let rppgEl = null;
let scoreChartEl = null;
let userExpanded = false;
let userCollapsed = false;
let lastRiskState = "normal";

const EMOJI = {
  normal: "😊",
  caution: "😐",
  high: "😠",
};

const OVERLAY_PANEL_HTML = `
    <div class="dashboard-header df-drag-handle">
        <div class="dashboard-title">Only Human Beats</div>
        <div class="status-indicator">
            <div class="status-dot"></div>
            <span id="scan-status" class="df-title">SCANNING</span>
        </div>
    </div>
    <div class="metric-container">
        <div class="metric-label">
            <span>Deepfake Risk</span>
            <span id="dfFake">0%</span>
        </div>
        <div class="gauge-bar"><div id="fake-probability-bar"></div></div>
    </div>
    <div class="metric-container" style="margin-bottom: 0;">
        <div class="metric-label"><span>Live Heart Rate</span></div>
        <div class="hr-display">
            <span class="hr-value" id="dfHr">--</span>
            <span class="hr-unit">BPM</span>
        </div>
        <div class="waveform-container"><canvas id="dfScoreChart"></canvas></div>
    </div>
    <div style="display:none;" id="dfLive"></div>
    <div style="display:none;" id="dfRppg"></div>
`;

function riskLevelToClass(riskLevel) {
  if (riskLevel === "warning") return "df-risk-warning";
  if (riskLevel === "caution") return "df-risk-caution";
  if (riskLevel === "normal") return "df-risk-normal";
  return "df-risk-warming";
}

function setOverlayExpanded(expanded, riskLevel) {
  if (!overlayEl) return;
  overlayEl.classList.toggle("df-compact", !expanded);
  overlayEl.classList.toggle("df-expanded", expanded);
  overlayEl.classList.remove(
    "df-risk-warming",
    "df-risk-normal",
    "df-risk-caution",
    "df-risk-warning"
  );
  overlayEl.classList.add(riskLevelToClass(riskLevel));
}

function compactDotEmoji(status) {
  if (status.riskLevel === "normal") return EMOJI.normal;
  if (status.riskLevel === "caution") return EMOJI.caution;
  if (status.riskLevel === "warming") return "stack";
  return "";
}

function ensureDotLabel(dot) {
  if (dot.querySelector(".df-dot-emoji")) return;
  const legacy = dot.querySelector(".df-dot-label");
  if (legacy) legacy.remove();
  const wrap = document.createElement("div");
  wrap.className = "df-dot-label";
  wrap.setAttribute("aria-hidden", "true");
  const emoji = document.createElement("span");
  emoji.className = "df-dot-emoji";
  const text = document.createElement("span");
  text.className = "df-dot-text";
  text.hidden = true;
  wrap.appendChild(emoji);
  wrap.appendChild(text);
  dot.appendChild(wrap);
}

function renderNoVideoEmoji(emojiEl) {
  emojiEl.className = "df-dot-emoji df-dot-emoji--no-video";
  emojiEl.innerHTML =
    '<span class="df-emoji-stack"><span class="df-emoji-cam">📷</span><span class="df-emoji-ban">🚫</span></span>';
}

function updateCompactDot(status, result) {
  if (!dotEl) return;
  const label = status.title || "분석 중";
  const emojiKind = compactDotEmoji(status);
  dotEl.setAttribute(
    "aria-label",
    status.riskLevel === "warming" ? `영상을 찾을 수 없음 · ${label}` : label
  );
  dotEl.title = label;
  if (result && typeof result.fakeProb === "number") {
    dotEl.title = `${label} (Fake ${result.fakeProb.toFixed(2)})`;
  }
  const labelEl = dotEl.querySelector(".df-dot-label");
  const emojiEl = dotEl.querySelector(".df-dot-emoji");
  const textEl = dotEl.querySelector(".df-dot-text");
  if (emojiEl) {
    if (emojiKind === "stack") renderNoVideoEmoji(emojiEl);
    else {
      emojiEl.className = "df-dot-emoji";
      emojiEl.textContent = emojiKind;
    }
  }
  if (textEl) {
    textEl.textContent = "";
    textEl.hidden = true;
  }
  if (labelEl) {
    labelEl.classList.toggle("df-dot-label--emoji-only", Boolean(emojiKind));
  }
  dotEl.classList.toggle("df-dot--no-video", status.riskLevel === "warming");
}

function bindOverlayElements() {
  dotEl = overlayEl.querySelector(".df-dot");
  if (dotEl) ensureDotLabel(dotEl);
  titleEl = overlayEl.querySelector(".df-title");
  messageEl = overlayEl.querySelector(".df-message");
  fakeEl = overlayEl.querySelector("#dfFake");
  liveEl = overlayEl.querySelector("#dfLive");
  hrEl = overlayEl.querySelector("#dfHr");
  rppgEl = overlayEl.querySelector("#dfRppg");
  scoreChartEl = overlayEl.querySelector("#dfScoreChart");
}

function ensureCompactLayout(root) {
  const existingDot = root.querySelector(".df-dot");
  if (existingDot) {
    ensureDotLabel(existingDot);
    return;
  }
  const panel = document.createElement("div");
  panel.className = "df-panel";
  while (root.firstChild) panel.appendChild(root.firstChild);
  const dot = document.createElement("div");
  dot.className = "df-dot df-drag-handle";
  dot.setAttribute("role", "status");
  dot.setAttribute("aria-live", "polite");
  ensureDotLabel(dot);
  root.appendChild(dot);
  root.appendChild(panel);
}

function formatPanelTitle(status) {
  const title = status.title || "분석 중";
  if (status.riskState === "High" || status.riskLevel === "warning") {
    return `${title} ${EMOJI.high}`;
  }
  return title;
}

function renderExpandedPanel(status, result, ready, rppgInfo) {
  if (titleEl) titleEl.textContent = formatPanelTitle(status);
  if (messageEl) messageEl.textContent = status.message;
  if (result) {
    const percent = Math.round(result.fakeProb * 100);
    if (fakeEl) fakeEl.textContent = `${percent}%`;
    const fakeBar = document.getElementById('fake-probability-bar');
    if (fakeBar) {
      const percentClamped = Math.max(0, Math.min(100, percent));
      fakeBar.style.width = `${percentClamped}%`;
      let barColor = "#00f0ff";
      if (percentClamped > 35 && percentClamped <= 70) {
        barColor = "#ffaa00";
      } else if (percentClamped > 70) {
        barColor = "#ff3e3e";
      }
      fakeBar.style.background = barColor;
      fakeBar.style.boxShadow = `0 0 8px ${barColor}`;
    }
    if (liveEl) liveEl.textContent = result.livenessScore.toFixed(3);
    if (hrEl) {
      const isHigh = status.riskState === "High";
      if (isHigh || !result.hrPred || result.hrPred <= 0) {
        hrEl.textContent = "--";
      } else {
        hrEl.textContent = Math.round(result.hrPred);
      }
    }
  } else {
    if (fakeEl) fakeEl.textContent = "0%";
    const fakeBar = document.getElementById('fake-probability-bar');
    if (fakeBar) {
      fakeBar.style.width = "0%";
      fakeBar.style.background = "#00f0ff";
      fakeBar.style.boxShadow = "none";
    }
    if (liveEl) liveEl.textContent = "-";
    if (hrEl) hrEl.textContent = "--";
  }
  if (rppgEl) {
    const buf = rppgInfo?.bufferLength ?? 0;
    rppgEl.textContent = ready
      ? `${RPPG_WINDOW_FRAMES}/${RPPG_WINDOW_FRAMES}`
      : `${buf}/${RPPG_WINDOW_FRAMES}`;
  }
  if (scoreChartEl) {
    if (result) recordScore(result);
    drawScoreChart(scoreChartEl);
  }
}

function injectOverlayStyle() {
  let style = document.querySelector("#deepfake-detector-overlay-style");
  if (!style) {
    style = document.createElement("style");
    style.id = "deepfake-detector-overlay-style";
    document.head.appendChild(style);
  }
  style.textContent = `
    #deepfake-detector-overlay {
      position: fixed; top: 20px; right: 20px; z-index: 999999;
      color: #111;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      box-shadow: 0 4px 16px rgba(0,0,0,0.22); box-sizing: border-box;
    }
    #deepfake-detector-overlay.df-compact {
      width: 80px; height: 80px; padding: 0; border-radius: 50%;
      border: 2px solid rgba(0,0,0,0.12); background: transparent;
    }
    #deepfake-detector-overlay.df-compact .df-panel { display: none; }
    #deepfake-detector-overlay.df-compact .df-dot {
      position: relative; display: flex; align-items: center; justify-content: center;
      width: 100%; height: 100%; border-radius: 50%;
      cursor: grab; touch-action: none; user-select: none;
    }
    #deepfake-detector-overlay.df-compact .df-dot-label {
      display: none !important; /* 커스텀 로고 사용을 위해 텍스트 라벨 모두 숨김 */
    }
    #deepfake-detector-overlay.df-compact .df-dot-label--emoji-only .df-dot-emoji {
      font-size: 32px; line-height: 1;
    }
    #deepfake-detector-overlay.df-compact .df-dot-emoji { display: none !important; }
    #deepfake-detector-overlay.df-compact .df-dot-text[hidden] { display: none; }
    #deepfake-detector-overlay.df-compact .df-dot-emoji--no-video {
      display: none !important;
    }
    #deepfake-detector-overlay.df-compact .df-emoji-stack {
      position: relative; display: inline-flex;
      align-items: center; justify-content: center;
      width: 44px; height: 44px;
    }
    #deepfake-detector-overlay.df-compact .df-emoji-cam { font-size: 30px; line-height: 1; }
    #deepfake-detector-overlay.df-compact .df-emoji-ban {
      position: absolute; right: 0; bottom: 2px; font-size: 22px; line-height: 1;
    }
    #deepfake-detector-overlay.df-expanded {
      width: 320px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 20px;
      padding: 20px;
      color: white;
      font-family: 'Inter', -apple-system, sans-serif;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 0 0 1px rgba(255, 255, 255, 0.1);
      transition: width 0.5s, height 0.5s, border-radius 0.5s, top 0.3s, left 0.3s;
      overflow: hidden;
    }
    #deepfake-detector-overlay.df-expanded .df-dot { display: none; }
    #deepfake-detector-overlay.df-expanded .df-panel { display: block; }
    #deepfake-detector-overlay.df-risk-normal.df-compact .df-dot {
      background-color: transparent !important;
      background-image: var(--logo-url, none);
      background-size: cover; background-position: center;
      box-shadow: 0 0 0 3px rgba(46,125,50,0.6), 0 0 15px rgba(46,125,50,0.5);
    }
    #deepfake-detector-overlay.df-risk-caution.df-compact .df-dot {
      background-color: transparent !important;
      background-image: var(--logo-url, none);
      background-size: cover; background-position: center;
      box-shadow: 0 0 0 3px rgba(249,168,37,0.8), 0 0 15px rgba(249,168,37,0.6);
    }
    #deepfake-detector-overlay.df-risk-warning.df-compact .df-dot {
      background-color: transparent !important;
      background-image: var(--logo-url, none);
      background-size: cover; background-position: center;
      box-shadow: 0 0 0 4px rgba(255,62,62,0.9), 0 0 20px rgba(255,62,62,0.8);
      animation: df-pulse 1.2s ease-in-out infinite;
    }
    #deepfake-detector-overlay.df-risk-warming.df-compact .df-dot { 
      background-color: transparent !important;
      background-image: var(--logo-url, none);
      background-size: cover; background-position: center;
      box-shadow: 0 0 0 2px rgba(158,158,158,0.5);
    }
    @keyframes df-pulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.06); }
    }
    /* 높은 위험 상태 시 붉은색 강렬한 네온 배경 및 테두리로 반전 */
    #deepfake-detector-overlay.df-expanded.df-risk-warning {
        background: rgba(42, 15, 15, 0.95) !important;
        border: 1px solid rgba(248, 56, 56, 0.8) !important;
        box-shadow: 0 10px 40px rgba(255, 0, 0, 0.5), inset 0 0 0 1px rgba(255, 0, 0, 0.2) !important;
    }
    #deepfake-detector-overlay.df-expanded.df-risk-warning .dashboard-title {
        background: linear-gradient(135deg, #ff3e3e, #ff0000) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    /* New Dashboard Styles */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        cursor: grab;
    }
    .dashboard-header:active {
        cursor: grabbing;
    }
    .dashboard-title {
        font-size: 16px;
        font-weight: 700;
        background: linear-gradient(135deg, #00f0ff, #0080ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.5px;
    }
    #deepfake-detector-overlay .df-title {
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
        color: #fff;
    }
    #deepfake-detector-overlay .df-message {
        display: none;
    }
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00ff88;
        box-shadow: 0 0 8px #00ff88;
        transition: background 0.3s, box-shadow 0.3s;
    }
    #deepfake-detector-overlay.df-risk-warning .status-dot {
        background: #ff3e3e !important;
        box-shadow: 0 0 10px #ff3e3e, 0 0 20px #ff3e3e !important;
        animation: df-pulse 1s infinite !important;
    }
    .metric-container {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .metric-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: #94a3b8;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .metric-label #dfFake {
        font-size: 16px;
        color: #fff;
        font-weight: 700;
    }
    .gauge-bar {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        overflow: hidden;
    }
    #fake-probability-bar {
        height: 100%;
        width: 0%;
        background: #00f0ff;
        border-radius: 3px;
        transition: width 0.3s ease, background-color 0.3s ease, box-shadow 0.3s ease;
    }
    .hr-display {
        display: flex;
        align-items: baseline;
        gap: 4px;
        margin-bottom: 10px;
    }
    .hr-value {
        font-size: 32px;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    .hr-unit {
        font-size: 14px;
        color: #f43f5e;
        font-weight: 600;
    }
    .waveform-container {
        width: 100%;
        height: 40px;
        border-radius: 6px;
        background: rgba(0, 0, 0, 0.5);
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.02);
    }
    .waveform-container canvas {
        display: block; width: 100%; height: 100%;
    }
  `;
}

function ensureOverlayDragging() {
  if (!overlayEl || overlayEl.dataset.dfDragBound === "1") return;
  overlayEl.dataset.dfDragBound = "1";
  let drag = false, isDragging = false, pointerId = null, originX = 0, originY = 0, boxLeft = 0, boxTop = 0;

  const syncPositionToLeftTop = () => {
    const r = overlayEl.getBoundingClientRect();
    overlayEl.style.right = "auto";
    overlayEl.style.left = `${Math.round(r.left)}px`;
    overlayEl.style.top = `${Math.round(r.top)}px`;
    return r;
  };

  const onPointerMove = (e) => {
    if (!drag || e.pointerId !== pointerId) return;

    const dx = Math.abs(e.clientX - originX);
    const dy = Math.abs(e.clientY - originY);
    if (dx > 3 || dy > 3) isDragging = true; // Drag threshold

    if (isDragging) {
      overlayEl.style.left = `${boxLeft + (e.clientX - originX)}px`;
      overlayEl.style.top = `${boxTop + (e.clientY - originY)}px`;
      overlayEl.style.right = "auto";
      e.preventDefault();
    }
  };

  const endDrag = (e) => {
    drag = false;
    pointerId = null;
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", endDrag);
    window.removeEventListener("pointercancel", endDrag);
  };

  overlayEl.addEventListener("pointerdown", (e) => {
    if (!e.target.closest(".df-drag-handle")) return;
    if (e.button !== undefined && e.button !== 0) return;
    e.preventDefault();
    syncPositionToLeftTop();
    const r = overlayEl.getBoundingClientRect();
    originX = e.clientX;
    originY = e.clientY;
    boxLeft = r.left;
    boxTop = r.top;
    drag = true;
    isDragging = false;
    pointerId = e.pointerId;
    window.addEventListener("pointermove", onPointerMove, { passive: false });
    window.addEventListener("pointerup", endDrag);
    window.addEventListener("pointercancel", endDrag);
  }, { passive: false });

  overlayEl.addEventListener("click", (e) => {
    if (isDragging) return; // Prevent toggle if it was a drag
    if (e.target.closest(".df-drag-handle")) {
      userExpanded = !userExpanded;
      userCollapsed = !userExpanded; // Explicit override

      // Immediately reflect state to UI without waiting for next updateOverlay loop
      if (overlayEl.classList.contains("df-expanded") && !userExpanded) {
        overlayEl.classList.remove("df-expanded");
        overlayEl.classList.add("df-compact");
      } else if (overlayEl.classList.contains("df-compact") && userExpanded) {
        overlayEl.classList.remove("df-compact");
        overlayEl.classList.add("df-expanded");
      }
    }
  });
}

function ensureScoreChartInOverlay() {
  const host = overlayEl?.querySelector(".df-panel");
  if (!host) return;
  if (!host.querySelector("#dfScoreChart")) {
    const wrap = document.createElement("div");
    wrap.className = "df-chart-wrap";
    wrap.innerHTML = '<div class="df-chart-title">점수 추이 (0~1)</div><canvas id="dfScoreChart" aria-label="Fake, Liveness 추이"></canvas>';
    host.appendChild(wrap);
    scoreChartEl = wrap.querySelector("#dfScoreChart");
  } else {
    scoreChartEl = host.querySelector("#dfScoreChart");
  }
  drawScoreChart(scoreChartEl);
}

export function createOverlay() {
  const existing = document.querySelector("#deepfake-detector-overlay");
  if (existing) {
    overlayEl = existing;
    ensureCompactLayout(overlayEl);
    bindOverlayElements();
    ensureScoreChartInOverlay();
    injectOverlayStyle();
    ensureOverlayDragging();
    setOverlayExpanded(false, "warming");
    return;
  }
  overlayEl = document.createElement("div");
  overlayEl.id = "deepfake-detector-overlay";
  overlayEl.className = "df-compact df-risk-warming";
  overlayEl.innerHTML =
    '<div class="df-dot df-drag-handle" role="status" aria-live="polite"></div>' +
    '<div class="df-panel">' +
    OVERLAY_PANEL_HTML +
    "</div>";
  document.body.appendChild(overlayEl);

  // 커스텀 로고 URL 변수 바인딩
  const logoUrl = chrome?.runtime?.getURL ? chrome.runtime.getURL('logo.jpg') : '';
  if (logoUrl) overlayEl.style.setProperty('--logo-url', `url("${logoUrl}")`);

  bindOverlayElements();
  injectOverlayStyle();
  ensureOverlayDragging();
  setOverlayExpanded(false, "warming");
  drawScoreChart(scoreChartEl);
}

export function setOverlayMessage(title, message) {
  if (!overlayEl?.classList.contains("df-expanded")) {
    updateCompactDot({ title, riskLevel: "warming" }, null);
    return;
  }
  if (titleEl) titleEl.textContent = title;
  if (messageEl) messageEl.textContent = message;
}

export function updateOverlay(status, result, ready, rppgInfo) {
  if (!overlayEl) return;
  
  if (rppgInfo && typeof rppgInfo.latestGreen === "number") {
    recordRppgSignal(rppgInfo.latestGreen);
  }

  const isHigh = status.riskState === "High";
  // Transition to high risk triggers auto-expansion if not explicitly collapsed
  if (isHigh && lastRiskState !== "High") {
    userExpanded = true;
    userCollapsed = false;
  }
  lastRiskState = status.riskState;
  
  const showExpanded = userExpanded;
  
  setOverlayExpanded(showExpanded, status.riskLevel);
  if (!showExpanded) {
    updateCompactDot(status, result);
    return;
  }
  renderExpandedPanel(
    status,
    result,
    ready,
    rppgInfo
  );
}

export function resetOverlayState() {
  userExpanded = false;
  userCollapsed = false;
  lastRiskState = "normal";
  if (overlayEl) {
    overlayEl.classList.remove("df-expanded");
    overlayEl.classList.add("df-compact");
  }
}

export function hasOverlay() {
  return Boolean(overlayEl);
}
