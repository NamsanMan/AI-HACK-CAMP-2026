import {
  FACE_INPUT_SIZE,
  MODEL_FILENAME,
  RPPG_CHANNELS,
  RPPG_WINDOW_FRAMES,
} from "./modelConstants.js";

let session = null;

const IO = {
  rppg: "rppg_window",
  face: "face_crop",
  quality: "quality",
  fake: "fake_probability",
  liveness: "liveness_score",
  confidence: "confidence_score",
  hr: "estimated_hr",
  artifactFake: "artifact_fake",
  rppgLiveness: "rppg_liveness",
};

function configureOrtWasm(ortRuntime, { simd }) {
  ortRuntime.env.wasm.numThreads = 1;
  ortRuntime.env.wasm.proxy = false;
  ortRuntime.env.wasm.simd = simd;

  const isExtension =
    typeof chrome !== "undefined" && chrome.runtime?.getURL;

  if (isExtension) {
    const ortBase = chrome.runtime.getURL("ort/");
    if (simd) {
      ortRuntime.env.wasm.wasmPaths = {
        "ort-wasm-simd.wasm": ortBase + "ort-wasm-simd.wasm",
        "ort-wasm.wasm": ortBase + "ort-wasm.wasm",
        "ort-wasm-simd.jsep.wasm": ortBase + "ort-wasm-simd.jsep.wasm",
      };
    } else {
      ortRuntime.env.wasm.wasmPaths = {
        "ort-wasm.wasm": ortBase + "ort-wasm.wasm",
      };
    }
  } else {
    ortRuntime.env.wasm.wasmPaths = "/ort/";
  }
}

async function createSession(ortRuntime, modelUrl, simd) {
  configureOrtWasm(ortRuntime, { simd });
  return ortRuntime.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
  });
}

function resolveIoNames(sess) {
  const ins = sess.inputNames ?? [];
  const outs = sess.outputNames ?? [];

  const pick = (list, hints, fallback) =>
    list.find((n) => hints.some((h) => n === h || n.toLowerCase().includes(h))) ??
    fallback;

  IO.rppg = pick(ins, ["rppg_window", "rppg_seq", "rppg"], ins[0]);
  IO.face = pick(ins, ["face_crop", "face"], ins[1]);
  IO.quality = pick(ins, ["quality"], ins[2]);

  IO.fake = pick(outs, ["fake_probability", "fake_prob"], outs[0]);
  IO.liveness = pick(outs, ["liveness_score"], outs[1]);
  IO.confidence = pick(outs, ["confidence_score", "confidence"], outs[2]);
  IO.hr = pick(outs, ["estimated_hr", "hr_pred"], outs[3]);
  IO.artifactFake = pick(outs, ["artifact_fake"], null);
  IO.rppgLiveness = pick(outs, ["rppg_liveness"], null);

  console.log("[ONNX] IO mapping:", IO);
}

function dimSize(dim, fallback) {
  return typeof dim === "number" && dim > 0 ? dim : fallback;
}

export async function loadOnnxModel(modelUrl = null) {
  const ortRuntime = globalThis.ort;

  if (!ortRuntime) {
    throw new Error(
      "ONNX Runtime global object is not loaded. Check ort/ort.wasm.min.js in offscreen.html."
    );
  }

  const finalModelUrl =
    modelUrl ||
    (typeof chrome !== "undefined" && chrome.runtime?.getURL
      ? chrome.runtime.getURL(`models/${MODEL_FILENAME}`)
      : `/models/${MODEL_FILENAME}`);

  console.log("[ONNX] modelUrl:", finalModelUrl);

  try {
    session = await createSession(ortRuntime, finalModelUrl, true);
  } catch (simdErr) {
    console.warn("[ONNX] SIMD wasm failed, retrying without SIMD:", simdErr);
    session = await createSession(ortRuntime, finalModelUrl, false);
  }

  console.log("[ONNX] model loaded");
  console.log("[ONNX] inputs:", session.inputNames);
  console.log("[ONNX] outputs:", session.outputNames);
  resolveIoNames(session);

  return session;
}

function readScalar(outputs, name) {
  if (!name || !outputs[name]) {
    return 0;
  }

  const v = outputs[name].data?.[0];
  return typeof v === "number" && !Number.isNaN(v) ? v : 0;
}

/**
 * @param {Float32Array} rppgWindow raw ROI RGB [1,90,9] in 0..1
 * @param {Float32Array} faceCrop RGB [1,3,96,96] in 0..1
 * @param {Float32Array} quality [roi*fill, det, motion*fill]
 */
export async function runInferenceWithInputs(rppgWindow, faceCrop, quality) {
  if (!session) {
    throw new Error("ONNX session is not loaded");
  }

  const ortRuntime = globalThis.ort;

  const rppgMeta = session.inputMetadata?.[IO.rppg]?.dimensions;
  const faceMeta = session.inputMetadata?.[IO.face]?.dimensions;

  const rppgShape = rppgMeta?.length === 3
    ? rppgMeta.map((d, i) =>
        dimSize(d, [1, RPPG_WINDOW_FRAMES, RPPG_CHANNELS][i])
      )
    : [1, RPPG_WINDOW_FRAMES, RPPG_CHANNELS];

  const faceShape = faceMeta?.length === 4
    ? faceMeta.map((d, i) =>
        dimSize(d, [1, 3, FACE_INPUT_SIZE, FACE_INPUT_SIZE][i])
      )
    : [1, 3, FACE_INPUT_SIZE, FACE_INPUT_SIZE];

  const rppgExpected = rppgShape.reduce((a, b) => a * b, 1);
  const faceExpected = faceShape.reduce((a, b) => a * b, 1);

  if (rppgWindow.length !== rppgExpected) {
    throw new Error(
      `rppg_window size mismatch: ${rppgWindow.length} / ${rppgExpected}`
    );
  }

  if (faceCrop.length !== faceExpected) {
    throw new Error(
      `face_crop size mismatch: ${faceCrop.length} / ${faceExpected}`
    );
  }

  const qualityData =
    quality instanceof Float32Array && quality.length === 3
      ? quality
      : new Float32Array([1, 1, 1]);

  const feeds = {
    [IO.rppg]: new ortRuntime.Tensor("float32", rppgWindow, rppgShape),
    [IO.face]: new ortRuntime.Tensor("float32", faceCrop, faceShape),
    [IO.quality]: new ortRuntime.Tensor("float32", qualityData, [1, 3]),
  };

  const outputs = await session.run(feeds);

  return {
    fakeProb: readScalar(outputs, IO.fake),
    livenessScore: readScalar(outputs, IO.liveness),
    confidence: readScalar(outputs, IO.confidence),
    hrPred: readScalar(outputs, IO.hr),
    artifactFake: readScalar(outputs, IO.artifactFake),
    rppgLiveness: readScalar(outputs, IO.rppgLiveness),
  };
}
