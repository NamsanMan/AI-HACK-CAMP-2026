// content.js - Sandbox Analysis Version

let dashboard = null;
let rppgCanvas = null;
let rppgCtx = null;
let rppgHistory = new Array(100).fill(0.5);
let sandboxFrame = null;
let modelReady = false;
let lostDetectionCount = 0;
const MAX_LOST_COUNT = 5; // 5회 연속 탐지 실패 시에만 닫기 (약 1초)

// 1. 샌드박스 iframe 생성
function initSandbox() {
    if (document.getElementById('deepfake-sandbox-frame')) return;
    
    sandboxFrame = document.createElement('iframe');
    sandboxFrame.id = 'deepfake-sandbox-frame';
    sandboxFrame.src = chrome.runtime.getURL('sandbox.html');
    sandboxFrame.style.display = 'none'; // 숨김 처리
    document.body.appendChild(sandboxFrame);

    window.addEventListener('message', (event) => {
        if (event.data.type === 'MODEL_LOADED') {
            modelReady = true;
            console.log("[*] Sandbox Model Ready");
            const statusLabel = document.getElementById('scan-status');
            if (statusLabel) statusLabel.textContent = "AI Model Ready. Monitoring...";
            startInferenceLoop(); 
        } else if (event.data.type === 'PREDICT_RESULT') {
            handleInferenceResult(event.data);
        }
    });

    // 샌드박스 초기화 명령
    sandboxFrame.onload = () => {
        sandboxFrame.contentWindow.postMessage({ type: 'INIT' }, '*');
    };
}

// 추론 루프 시작
function startInferenceLoop() {
    console.log("[Content] Starting inference loop...");
    setInterval(() => {
        const activeMedia = getActiveVideo();
        if (activeMedia) {
            sendFrameToSandbox(activeMedia);
        }
    }, 200); 
}

// 2. 비디오 프레임을 샌드박스로 전달
function sendFrameToSandbox(video) {
    if (!modelReady || !sandboxFrame) {
        if (!modelReady) console.warn("[Content] Model not ready yet");
        return;
    }

    const targetSize = 320;
    const canvas = document.createElement('canvas');
    canvas.width = targetSize;
    canvas.height = targetSize;
    const ctx = canvas.getContext('2d');
    
    try {
        // 배경을 검은색으로 채워 레터박스 효과 (비율 유지)
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, targetSize, targetSize);

        // 비디오/캔버스 구분 없이 실제 크기 가져오기
        const videoW = video.videoWidth || video.width || video.offsetWidth || targetSize;
        const videoH = video.videoHeight || video.height || video.offsetHeight || targetSize;
        
        // 가로/세로 비율 유지하며 320x320에 맞추기 (Letterbox)
        const scale = Math.min(targetSize / videoW, targetSize / videoH);
        const drawW = videoW * scale;
        const drawH = videoH * scale;
        const drawX = (targetSize - drawW) / 2;
        const drawY = (targetSize - drawH) / 2;
        
        ctx.drawImage(video, 0, 0, videoW, videoH, drawX, drawY, drawW, drawH);
        const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
        
        // [중요] 전송 확인 로그
        console.log("[Content] Sending PREDICT to sandbox..."); 
        
        sandboxFrame.contentWindow.postMessage({
            type: 'PREDICT',
            imageData: imageData.data,
            width: targetSize,
            height: targetSize
        }, '*'); // Transferable 제거하여 호환성 확보
    } catch (e) {
        console.error("[Content] Failed to capture frame:", e);
    }
}

// 초기 실행
function initAll() {
    console.log("[Content] Initializing Antigravity Deepfake Extension...");
    initUI();
    initSandbox();
}

if (document.readyState === 'complete') {
    initAll();
} else {
    window.addEventListener('load', initAll);
}

// 3. [결과 처리] 샌드박스에서 온 데이터 대시보드에 반영
function handleInferenceResult(data) {
    if (!dashboard) return;
    
    const { faceDetected, score, crop, fakeRisk, ppgWave, hrVal } = data;
    console.log("[Content] Inference result:", { faceDetected, score, hasCrop: !!crop, fakeRisk, hrVal });
    
    // 얼굴 탐지 시 자동으로 대시보드 확장
    if (faceDetected) {
        lostDetectionCount = 0; // 탐지 성공 시 카운트 리셋
        if (dashboard.classList.contains('minimized')) {
            console.log("[Content] Face detected! Expanding dashboard...");
            dashboard.classList.remove('minimized');
        }
    } else {
        lostDetectionCount++;
    }
    
    // 일정 횟수 이상 실패해야만 닫기 (깜빡임 방지)
    if (!faceDetected && lostDetectionCount >= MAX_LOST_COUNT) {
        if (!dashboard.classList.contains('minimized')) {
            dashboard.classList.add('minimized');
        }
    }
    
    const statusText = faceDetected ? `Tracking (${(score * 100).toFixed(1)}%)` : (lostDetectionCount < MAX_LOST_COUNT && !dashboard.classList.contains('minimized') ? "Searching..." : "Monitoring...");
    const statusLabel = document.getElementById('scan-status');
    if (statusLabel) {
        statusLabel.textContent = statusText;
        statusLabel.style.color = faceDetected ? "#00ff88" : "#aaa";
    }
    
    const statusDot = dashboard.querySelector('.status-dot');
    if (statusDot) {
        const dotColor = faceDetected ? "#00ff88" : (lostDetectionCount < MAX_LOST_COUNT ? "#ffaa00" : "#ff3e3e");
        statusDot.style.setProperty('background', dotColor, 'important');
        statusDot.style.setProperty('box-shadow', `0 0 8px ${dotColor}`, 'important');
    }
    
    // 크롭된 얼굴 미리보기 그리기
    if (faceDetected && crop) {
        drawFacePreview(crop);
    }

    // --- [바인딩 1] 딥페이크 위험도 백분율 및 컬러 게이지바 제어 ---
    const fakePercentLabel = document.getElementById('fake-percent');
    const fakeBar = document.getElementById('fake-probability-bar');
    if (faceDetected && typeof fakeRisk !== 'undefined') {
        const percent = Math.round(fakeRisk * 100);
        if (fakePercentLabel) fakePercentLabel.textContent = `${percent}%`;
        if (fakeBar) {
            fakeBar.style.width = `${percent}%`;
            // 위험도 구간별 색상 동적 변경 (안전: 청록, 보통: 오렌지, 위험: 빨강)
            let barColor = "#00f0ff"; // Cyantastic / Safe
            if (percent > 35 && percent <= 70) {
                barColor = "#ffaa00"; // Alert
            } else if (percent > 70) {
                barColor = "#ff3e3e"; // Danger
            }
            fakeBar.style.background = barColor;
            fakeBar.style.boxShadow = `0 0 8px ${barColor}`;
        }
    } else {
        if (fakePercentLabel) fakePercentLabel.textContent = "0%";
        if (fakeBar) {
            fakeBar.style.width = "0%";
            fakeBar.style.background = "#00f0ff";
            fakeBar.style.boxShadow = "none";
        }
    }

    // --- [바인딩 2] 실시간 심박수 텍스트 제어 ---
    const hrLabel = document.getElementById('hr-value');
    if (faceDetected && typeof hrVal !== 'undefined' && hrVal > 0) {
        hrLabel.textContent = Math.round(hrVal);
    } else {
        if (hrLabel) hrLabel.textContent = "--";
    }

    // --- [바인딩 3] 맥박 Waveform 히스토리 누적 ---
    if (faceDetected && ppgWave && ppgWave.length > 0) {
        // 30프레임 중 가장 최신 파동 데이터 1개를 역사에 밀어넣음
        const latestPPG = ppgWave[ppgWave.length - 1];
        rppgHistory.push(latestPPG);
        if (rppgHistory.length > 100) {
            rppgHistory.shift();
        }
    } else {
        // 미감지 시 맥박 신호를 서서히 중간(0.5)으로 수렴하도록 보정 (깜빡임 완화)
        rppgHistory.push(0.5);
        if (rppgHistory.length > 100) {
            rppgHistory.shift();
        }
    }
}

// 얼굴 미리보기 캔버스 그리기
function drawFacePreview(crop) {
    let previewContainer = document.getElementById('face-preview-container');
    if (!previewContainer) {
        previewContainer = document.createElement('div');
        previewContainer.id = 'face-preview-container';
        previewContainer.style.cssText = `
            width: 100%;
            height: 160px;
            background: #000;
            border-radius: 12px;
            margin-bottom: 20px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            border: 1px solid rgba(56, 189, 248, 0.3);
            box-shadow: inset 0 0 20px rgba(56, 189, 248, 0.2);
        `;
        
        const canvas = document.createElement('canvas');
        canvas.id = 'face-preview';
        canvas.style.height = '100%';
        canvas.style.width = 'auto';
        previewContainer.appendChild(canvas);
        
        // 헤더 바로 다음에 삽입
        const header = dashboard.querySelector('.dashboard-header');
        header.parentNode.insertBefore(previewContainer, header.nextSibling);
    }
    
    const canvas = document.getElementById('face-preview');
    if (canvas && crop.pixels) {
        canvas.width = crop.width;
        canvas.height = crop.height;
        const ctx = canvas.getContext('2d');
        const imageData = new ImageData(new Uint8ClampedArray(crop.pixels), crop.width, crop.height);
        ctx.putImageData(imageData, 0, 0);
        // console.log("[Content] Face preview drawn:", crop.width, "x", crop.height);
    }
}

// 4. 프리미엄 대시보드 UI 생성 (기존 유지)
function createDashboard() {
    if (document.getElementById('deepfake-analysis-dashboard')) return;
    dashboard = document.createElement('div');
    dashboard.id = 'deepfake-analysis-dashboard';
    dashboard.classList.add('minimized');
    dashboard.innerHTML = `
        <div class="dashboard-header">
            <div class="dashboard-title">Only Human Beats</div>
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span id="scan-status">SCANNING</span>
            </div>
        </div>
        <div class="metric-container">
            <div class="metric-label">
                <span>Deepfake Risk</span>
                <span id="fake-percent">0%</span>
            </div>
            <div class="gauge-bar"><div id="fake-probability-bar"></div></div>
        </div>
        <div class="metric-container" style="margin-bottom: 0;">
            <div class="metric-label"><span>Live Heart Rate</span></div>
            <div class="hr-display">
                <span class="hr-value" id="hr-value">--</span>
                <span class="hr-unit">BPM</span>
            </div>
            <div class="waveform-container"><canvas id="rppg-canvas"></canvas></div>
        </div>
    `;
    document.body.appendChild(dashboard);
    rppgCanvas = document.getElementById('rppg-canvas');
    const rect = rppgCanvas.parentElement.getBoundingClientRect();
    rppgCanvas.width = rect.width;
    rppgCanvas.height = rect.height;
    rppgCtx = rppgCanvas.getContext('2d');

    const logoUrl = chrome.runtime.getURL('logo.png');
    dashboard.style.setProperty('--logo-url', `url("${logoUrl}")`);

    // 드래그 로직 (기존 유지)
    let isDragging = false, offsetX, offsetY;
    dashboard.addEventListener('mousedown', (e) => {
        isDragging = true;
        const rect = dashboard.getBoundingClientRect();
        offsetX = e.clientX - rect.left; offsetY = e.clientY - rect.top;
        dashboard.style.transition = 'none';
    });
    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        let x = e.clientX - offsetX, y = e.clientY - offsetY;
        x = Math.max(0, Math.min(x, window.innerWidth - dashboard.offsetWidth));
        y = Math.max(0, Math.min(y, window.innerHeight - dashboard.offsetHeight));
        dashboard.style.left = `${x}px`; dashboard.style.top = `${y}px`;
        dashboard.style.right = 'auto';
    });
    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            dashboard.style.transition = 'width 0.5s, height 0.5s, border-radius 0.5s, top 0.3s, left 0.3s';
        }
    });
}

// 초기 실행 및 통합 관리
function initAll() {
    console.log("[Content] Initializing Antigravity Deepfake Extension...");
    if (dashboard) return; // 중복 실행 방지
    
    createDashboard(); // UI 생성
    initSandbox();     // 샌드박스 및 모델 로드 시작
    drawWaveform();    // 그래프 애니메이션 시작 (UI용)
}

// 페이지 로드 상태에 따른 실행
if (document.readyState === 'complete') {
    setTimeout(initAll, 1000); // 유튜브 UI 안정화 대기
} else {
    window.addEventListener('load', () => setTimeout(initAll, 1000));
}

function getActiveVideo() {
    // 비디오 태그와 캔버스 태그 모두 검색 (줌, 구글 미트 등 대응)
    const elements = Array.from(document.querySelectorAll('video, canvas'));
    let activeElement = null, maxVisibleArea = 0;
    
    elements.forEach(el => {
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return;
        
        const rect = el.getBoundingClientRect();
        const visibleWidth = Math.max(0, Math.min(rect.right, window.innerWidth) - Math.max(rect.left, 0));
        const visibleHeight = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0));
        const visibleArea = visibleWidth * visibleHeight;
        
        // 화면의 10% 이상을 차지하는 가장 큰 요소 선택
        if (visibleArea > maxVisibleArea && visibleArea > (window.innerWidth * window.innerHeight * 0.05)) {
            // 비디오인 경우 재생 중인지 확인, 캔버스인 경우 일단 선택
            if (el.tagName === 'VIDEO') {
                if (el.paused || el.ended || el.readyState < 2) return;
            }
            maxVisibleArea = visibleArea; 
            activeElement = el;
        }
    });
    return activeElement;
}

function drawWaveform() {
    if (!rppgCtx) return;
    const w = rppgCanvas.width, h = rppgCanvas.height;
    rppgCtx.clearRect(0, 0, w, h);
    rppgCtx.beginPath();
    rppgCtx.strokeStyle = '#f43f5e';
    rppgCtx.lineWidth = 2;
    rppgCtx.lineJoin = 'round';
    
    const step = w / (rppgHistory.length - 1);
    
    // 실시간 Min-Max 스케일링을 통한 파형 정규화
    const minVal = Math.min(...rppgHistory);
    const maxVal = Math.max(...rppgHistory);
    const range = maxVal - minVal;
    
    for (let i = 0; i < rppgHistory.length; i++) {
        const x = i * step;
        // 신호가 너무 잔잔할 땐 중앙(0.5)에 고정하고, 신호가 존재하면 [0.1~0.9] 스페이스 내에 투사
        const normVal = range > 0.0001 ? (rppgHistory[i] - minVal) / range : 0.5;
        const y = h - (normVal * h * 0.75 + h * 0.125); // 상하 약 12.5% 여백 부여로 파형 잘림 방지
        
        if (i === 0) rppgCtx.moveTo(x, y); else rppgCtx.lineTo(x, y);
    }
    rppgCtx.stroke();
    rppgCtx.lineTo(w, h); rppgCtx.lineTo(0, h);
    const grad = rppgCtx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(244, 63, 94, 0.25)');
    grad.addColorStop(1, 'rgba(244, 63, 94, 0)');
    rppgCtx.fillStyle = grad; rppgCtx.fill();
    requestAnimationFrame(drawWaveform);
}
