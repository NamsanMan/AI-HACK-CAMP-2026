export function expectedRgbaLength(width, height) {
  return width * height * 4;
}

/** content script → offscreen 전송용 (sendMessage에서 TypedArray가 깨지는 경우 대비) */
export function encodeRgbaBase64(rgba) {
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

export function decodeRgbaBase64(b64) {
  const binary = atob(b64);
  const rgba = new Uint8ClampedArray(binary.length);

  for (let i = 0; i < binary.length; i++) {
    rgba[i] = binary.charCodeAt(i);
  }

  return rgba;
}

function fromIndexedObject(obj, expected) {
  if (!obj || typeof obj !== "object" || ArrayBuffer.isView(obj)) {
    return null;
  }

  if (Array.isArray(obj)) {
    return new Uint8ClampedArray(obj);
  }

  if (typeof obj.length === "number" && obj.length > 0) {
    try {
      return new Uint8ClampedArray(obj);
    } catch {
      // fall through
    }
  }

  const keys = Object.keys(obj);
  if (keys.length === 0) {
    return null;
  }

  const rgba = new Uint8ClampedArray(expected);
  for (let i = 0; i < expected; i++) {
    rgba[i] = obj[i] ?? 0;
  }
  return rgba;
}

export function decodeRgbaPayload(message) {
  const width = message.width | 0;
  const height = message.height | 0;
  const expected = expectedRgbaLength(width, height);

  if (width < 1 || height < 1 || expected < 4) {
    throw new Error(`Invalid frame size: ${width}x${height}`);
  }

  let rgba = null;

  if (typeof message.rgbaB64 === "string" && message.rgbaB64.length > 0) {
    rgba = decodeRgbaBase64(message.rgbaB64);
  } else if (message.rgbaBuffer instanceof ArrayBuffer) {
    rgba = new Uint8ClampedArray(message.rgbaBuffer);
  } else if (message.rgba instanceof ArrayBuffer) {
    rgba = new Uint8ClampedArray(message.rgba);
  } else if (ArrayBuffer.isView(message.rgba)) {
    rgba = new Uint8ClampedArray(
      message.rgba.buffer,
      message.rgba.byteOffset,
      message.rgba.byteLength
    );
  } else if (message.rgba) {
    rgba = fromIndexedObject(message.rgba, expected);
  }

  if (!rgba || rgba.length === 0) {
    const hint = typeof message.rgbaB64 === "string"
      ? `rgbaB64 length=${message.rgbaB64.length}`
      : `keys=${Object.keys(message).join(",")}`;
    throw new Error(
      `프레임 픽셀 데이터가 비어 있습니다 (${hint}). 확장 프로그램·페이지를 새로고침하세요.`
    );
  }

  if (rgba.length !== expected) {
    throw new Error(
      `프레임 크기 불일치: ${rgba.length} bytes, expected ${expected} (${width}x${height})`
    );
  }

  return { rgba, width, height };
}
