/** 최근 추론 점수 이력 (1초 간격 기준 약 2분) */
const MAX_POINTS = 120;

const series = {
  fake: [],
  liveness: [],
};

function clamp01(v) {
  if (typeof v !== "number" || Number.isNaN(v)) {
    return 0;
  }

  return Math.min(1, Math.max(0, v));
}

function trimSeries() {
  const n = series.fake.length;
  if (n <= MAX_POINTS) {
    return;
  }

  const drop = n - MAX_POINTS;
  series.fake.splice(0, drop);
  series.liveness.splice(0, drop);
}

const rppgSignalHistory = [];
const MAX_SIGNAL_POINTS = 120;

export function resetScoreHistory() {
  series.fake.length = 0;
  series.liveness.length = 0;
  rppgSignalHistory.length = 0;
  currentHr = 65;
}

let currentHr = 65;
let phase = 0;

export function recordRppgSignal(val) {
  if (typeof val !== "number" || Number.isNaN(val)) return;
  rppgSignalHistory.push(val);
  if (rppgSignalHistory.length > MAX_SIGNAL_POINTS) {
      rppgSignalHistory.shift();
  }
}

export function recordScore(result) {
  if (!result) return;
  
  series.fake.push(clamp01(result.fakeProb));
  series.liveness.push(clamp01(result.livenessScore));
  trimSeries();
  
  if (result.hrPred && result.hrPred > 0) {
      currentHr = result.hrPred;
  }
}

export function drawScoreChart(canvas) {
  if (!canvas) return;

  const parent = canvas.parentElement;
  if (!parent) return;

  // We only initialize the animation loop once per canvas
  if (canvas.dataset.animating === "true") return;
  canvas.dataset.animating = "true";

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const ctx = canvas.getContext("2d");

  function animate() {
      if (!document.body.contains(canvas)) {
         canvas.dataset.animating = "";
         return; // Stop animation if canvas is removed
      }
      
      const width = parent.clientWidth;
      const height = parent.clientHeight;
      
      if (width === 0 || height === 0) {
          requestAnimationFrame(animate);
          return;
      }
      
      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
          canvas.width = Math.round(width * dpr);
          canvas.height = Math.round(height * dpr);
          canvas.style.width = `${width}px`;
          canvas.style.height = `${height}px`;
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
      
      ctx.clearRect(0, 0, width, height);
      ctx.beginPath();
      ctx.strokeStyle = '#f43f5e'; // Neon pink
      ctx.lineWidth = 2.5;
      ctx.lineJoin = 'round';
      
      if (rppgSignalHistory.length >= 10) {
          // Draw real-time detrended rPPG signal
          const processed = [];
          const windowSize = 15;
          for (let i = 0; i < rppgSignalHistory.length; i++) {
              const start = Math.max(0, i - windowSize);
              const end = Math.min(rppgSignalHistory.length - 1, i + windowSize);
              let sum = 0;
              for (let j = start; j <= end; j++) sum += rppgSignalHistory[j];
              const mean = sum / (end - start + 1);
              processed.push(rppgSignalHistory[i] - mean);
          }
          
          let minVal = Math.min(...processed);
          let maxVal = Math.max(...processed);
          let range = maxVal - minVal;
          
          const step = width / (processed.length - 1);
          for (let i = 0; i < processed.length; i++) {
              const x = i * step;
              const normVal = range > 0.0001 ? (processed[i] - minVal) / range : 0.5;
              const y = height - (normVal * height * 0.7 + height * 0.15);
              
              if (i === 0) ctx.moveTo(x, y); 
              else ctx.lineTo(x, y);
          }
      } else {
          // Simulated ECG fallback wave
          const numPoints = 120;
          const step = width / (numPoints - 1);
          
          // Heart rate frequency (beats per second)
          const freq = (currentHr / 60) * Math.PI * 2; 
          phase += freq / 60; // Assume roughly 60fps for scrolling speed
          
          for (let i = 0; i < numPoints; i++) {
              const x = i * step;
              // Calculate scrolling time parameter for this pixel
              const t = phase - (i * 0.08); 
              
              // Synthesize an ECG-like shape combining sine waves
              let wave = Math.sin(t) + 0.5 * Math.sin(2 * t + 1.5) + 0.25 * Math.sin(3 * t);
              const normVal = (wave + 1.75) / 3.5; // roughly 0 to 1
              
              const y = height - (normVal * height * 0.7 + height * 0.15);
              
              if (i === 0) ctx.moveTo(x, y); 
              else ctx.lineTo(x, y);
          }
      }
      ctx.stroke();
      
      // Gradient fill below the waveform
      ctx.lineTo(width, height); 
      ctx.lineTo(0, height);
      const grad = ctx.createLinearGradient(0, 0, 0, height);
      grad.addColorStop(0, 'rgba(244, 63, 94, 0.4)');
      grad.addColorStop(1, 'rgba(244, 63, 94, 0)');
      ctx.fillStyle = grad; 
      ctx.fill();
      
      requestAnimationFrame(animate);
  }
  
  animate();
}

export function getHistoryLength() {
  return series.fake.length;
}
