const OFFSCREEN_URL = "offscreen.html";

async function hasOffscreenDocument() {
  if (!chrome.offscreen) {
    return false;
  }

  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
  });

  return contexts.length > 0;
}

export async function ensureOffscreenDocument() {
  if (await hasOffscreenDocument()) {
    return;
  }

  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ["WORKERS"],
    justification: "ONNX Runtime and MediaPipe WASM inference for deepfake detection",
  });
}

async function forwardToOffscreen(message, attempts = 8) {
  let lastError = null;

  for (let i = 0; i < attempts; i++) {
    try {
      return await chrome.runtime.sendMessage({
        ...message,
        target: "offscreen",
      });
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }

  throw lastError ?? new Error("Offscreen document is not responding");
}

chrome.runtime.onInstalled.addListener(() => {
  ensureOffscreenDocument().catch(console.error);
});

chrome.runtime.onStartup.addListener(() => {
  ensureOffscreenDocument().catch(console.error);
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target === "offscreen") {
    return false;
  }

  if (message?.type === "DF_ENSURE_OFFSCREEN") {
    (async () => {
      try {
        await ensureOffscreenDocument();
        sendResponse({ ok: true });
      } catch (err) {
        sendResponse({ ok: false, error: err?.message || String(err) });
      }
    })();
    return true;
  }

  if (message?.type === "DF_INIT") {
    (async () => {
      try {
        await ensureOffscreenDocument();
        const reply = await forwardToOffscreen({ type: "DF_INIT" });
        sendResponse(reply);
      } catch (err) {
        sendResponse({ ok: false, error: err?.message || String(err) });
      }
    })();
    return true;
  }

  return false;
});
