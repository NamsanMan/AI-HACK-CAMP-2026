function assertRuntimeAlive() {
  if (!chrome.runtime?.id) {
    throw new Error(
      "Extension context invalidated. 확장 프로그램을 새로고침한 뒤 페이지도 새로고침하세요."
    );
  }
}

function runtimeSend(message) {
  assertRuntimeAlive();

  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      const err = chrome.runtime.lastError;
      if (err) {
        reject(new Error(err.message));
        return;
      }
      resolve(response);
    });
  });
}

function encodeRgbaBase64(rgba) {
  const bytes =
    rgba instanceof Uint8Array
      ? rgba
      : new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength);

  let binary = "";
  const chunkSize = 0x8000;

  for (let i = 0; i < bytes.length; i += chunkSize) {
    const slice = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
    binary += String.fromCharCode.apply(null, slice);
  }

  return btoa(binary);
}

export async function ensureOffscreenDocument() {
  const reply = await runtimeSend({ type: "DF_ENSURE_OFFSCREEN" });
  if (!reply?.ok) {
    throw new Error(reply?.error || "Failed to open offscreen document");
  }
}

export async function initPipeline() {
  await ensureOffscreenDocument();

  const reply = await runtimeSend({
    target: "offscreen",
    type: "DF_INIT",
  });

  if (!reply?.ok) {
    throw new Error(reply?.error || "Pipeline init failed");
  }
}

export async function processVideoFrame(
  rgba,
  width,
  height,
  { resetRppg = false, infer = true, skipRppgUpdate = false } = {}
) {
  const expected = width * height * 4;
  if (!rgba || rgba.length !== expected) {
    throw new Error(
      `캡처된 프레임이 유효하지 않습니다 (${rgba?.length ?? 0}/${expected}).`
    );
  }

  const rgbaB64 = encodeRgbaBase64(rgba);

  if (!rgbaB64) {
    throw new Error("프레임 인코딩에 실패했습니다.");
  }

  const reply = await runtimeSend({
    target: "offscreen",
    type: "DF_PROCESS_FRAME",
    width,
    height,
    rgbaB64,
    resetRppg,
    infer,
    skipRppgUpdate,
  });

  if (!reply?.ok) {
    throw new Error(reply?.error || "Frame processing failed");
  }

  return reply;
}
