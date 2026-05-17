/**
 * Zoom / Teams 등: video가 iframe·Shadow DOM·canvas·video-player 안에 있는 경우.
 */

const MIN_VISIBLE_PX = 48;
const MIN_CANVAS_PX = 120;
const ZOOM_PLAYER_TAGS = new Set([
  "VIDEO-PLAYER",
  "VIDEO-PLAYER-CONTAINER",
]);

function isElementVisible(el) {
  if (!el?.isConnected) {
    return false;
  }

  const style = getComputedStyle(el);
  if (style.display === "none" || style.visibility === "hidden") {
    return false;
  }

  if (parseFloat(style.opacity) < 0.05) {
    return false;
  }

  const rect = el.getBoundingClientRect();
  return rect.width >= MIN_VISIBLE_PX && rect.height >= MIN_VISIBLE_PX;
}

function walkElementTree(root, visit) {
  if (!root) {
    return;
  }

  const node = root instanceof Document ? root.documentElement : root;
  if (!node || node.nodeType !== Node.ELEMENT_NODE) {
    return;
  }

  visit(node);

  if (node.shadowRoot) {
    walkElementTree(node.shadowRoot, visit);
  }

  for (const child of node.children) {
    walkElementTree(child, visit);
  }
}

function addUniqueVideo(target, seen, el) {
  if (!el || el.tagName !== "VIDEO" || seen.has(el)) {
    return;
  }

  seen.add(el);
  target.push(el);
}

export function collectVideos(root = document) {
  const videos = [];
  const seen = new Set();

  walkElementTree(root, (el) => {
    if (el.tagName === "VIDEO") {
      addUniqueVideo(videos, seen, el);
      return;
    }

    if (!ZOOM_PLAYER_TAGS.has(el.tagName)) {
      return;
    }

    if (el.shadowRoot) {
      el.shadowRoot.querySelectorAll("video").forEach((v) => {
        addUniqueVideo(videos, seen, v);
      });
    }

    el.querySelectorAll?.("video").forEach((v) => {
      addUniqueVideo(videos, seen, v);
    });
  });

  return videos;
}

export function collectLargeCanvases(root = document) {
  const canvases = [];

  walkElementTree(root, (el) => {
    if (el.tagName !== "CANVAS") {
      return;
    }

    const w = el.width || el.clientWidth;
    const h = el.height || el.clientHeight;

    if (w >= MIN_CANVAS_PX && h >= MIN_CANVAS_PX && isElementVisible(el)) {
      canvases.push(el);
    }
  });

  return canvases;
}

function scoreVideo(el) {
  const rect = el.getBoundingClientRect();
  const streamW = el.videoWidth || 0;
  const streamH = el.videoHeight || 0;
  const w = streamW > 0 ? streamW : Math.round(rect.width);
  const h = streamH > 0 ? streamH : Math.round(rect.height);
  const area = w * h;
  const visible = isElementVisible(el);
  const hasStream = streamW > 0 && streamH > 0;
  const ready = el.readyState >= 2;
  const playing = !el.paused && !el.ended;

  let score = area;

  if (!visible) {
    score *= 0.01;
  }
  if (hasStream) {
    score *= 1.5;
  }
  if (ready) {
    score *= 1.2;
  }
  if (playing) {
    score *= 1.1;
  }
  if (rect.width >= 200 && rect.height >= 150) {
    score *= 1.15;
  }

  return { el, type: "video", area, score, hasStream, ready, visible };
}

function scoreCanvas(el) {
  const w = el.width || el.clientWidth;
  const h = el.height || el.clientHeight;
  const area = w * h;
  let score = area * 0.85;

  if (!isElementVisible(el)) {
    score *= 0.01;
  }

  return { el, type: "canvas", area, score, hasStream: true, ready: true, visible: true };
}

function pickBest(candidates) {
  if (candidates.length === 0) {
    return null;
  }

  candidates.sort((a, b) => b.score - a.score);
  return candidates[0];
}

/**
 * @returns {{ el: HTMLVideoElement|HTMLCanvasElement, type: 'video'|'canvas', area: number } | null}
 */
export function findBestMediaSource(root = document) {
  const videos = collectVideos(root);
  const videoCandidates = videos
    .filter((v) => {
      if (v.ended) {
        return false;
      }
      if (!isElementVisible(v) && v.videoWidth === 0) {
        return false;
      }
      return true;
    })
    .map(scoreVideo);

  const bestVideo = pickBest(videoCandidates);

  if (bestVideo && (bestVideo.hasStream || bestVideo.visible)) {
    return bestVideo;
  }

  const canvasCandidates = collectLargeCanvases(root).map(scoreCanvas);
  const bestCanvas = pickBest(canvasCandidates);

  if (bestCanvas && bestCanvas.area >= MIN_CANVAS_PX * MIN_CANVAS_PX) {
    return bestCanvas;
  }

  return bestVideo;
}

export function isMediaSourceReady(source) {
  if (!source) {
    return false;
  }

  if (source.type === "canvas") {
    return source.el.width > 0 && source.el.height > 0;
  }

  const v = source.el;
  if (v.videoWidth > 0 && v.videoHeight > 0) {
    return v.readyState >= 2;
  }

  const rect = v.getBoundingClientRect();
  const largeEnough = rect.width >= 80 && rect.height >= 60;

  return largeEnough && isElementVisible(v) && !v.ended;
}

export function describeMediaSearch(root = document) {
  const videos = collectVideos(root);
  const canvases = collectLargeCanvases(root);
  const inIframe = window !== window.top;
  const players = [];

  walkElementTree(root, (el) => {
    if (ZOOM_PLAYER_TAGS.has(el.tagName)) {
      players.push(el);
    }
  });

  return {
    inIframe,
    videoCount: videos.length,
    canvasCount: canvases.length,
    playerCount: players.length,
    withStream: videos.filter((v) => v.videoWidth > 0).length,
    href: location.pathname.slice(0, 40),
  };
}
