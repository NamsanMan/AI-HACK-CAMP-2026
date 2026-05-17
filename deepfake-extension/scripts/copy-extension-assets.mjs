/**
 * Copies ONNX Runtime + MediaPipe WASM binaries into public/ for the Chrome extension.
 * npm package onnxruntime-web does not ship .wasm in node_modules; MediaPipe .wasm
 * is in @mediapipe/tasks-vision but was not copied to public/.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ORT_VERSION = "1.18.0";

const ONNX_WASM_FILES = [
  "ort-wasm-simd.wasm",
  "ort-wasm.wasm",
  "ort-wasm-simd.jsep.wasm",
];

async function copyOnnxRuntime() {
  const ortDir = path.join(ROOT, "public", "ort");
  fs.mkdirSync(ortDir, { recursive: true });

  const srcJs = path.join(
    ROOT,
    "node_modules",
    "onnxruntime-web",
    "dist",
    "ort.wasm.min.js"
  );
  fs.copyFileSync(srcJs, path.join(ortDir, "ort.wasm.min.js"));

  for (const file of ONNX_WASM_FILES) {
    const dest = path.join(ortDir, file);
    if (fs.existsSync(dest) && fs.statSync(dest).size > 1000) {
      console.log(`  skip ${file} (already present)`);
      continue;
    }

    const url = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/${file}`;
    console.log(`  download ${file} ...`);
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to download ${url}: ${res.status}`);
    }
    fs.writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
  }
}

function copyMediaPipeWasm() {
  const srcDir = path.join(
    ROOT,
    "node_modules",
    "@mediapipe",
    "tasks-vision",
    "wasm"
  );
  const destDir = path.join(ROOT, "public", "mediapipe", "wasm");
  fs.mkdirSync(destDir, { recursive: true });

  for (const name of fs.readdirSync(srcDir)) {
    fs.copyFileSync(path.join(srcDir, name), path.join(destDir, name));
  }
}

console.log("[copy-extension-assets] ONNX Runtime wasm → public/ort/");
await copyOnnxRuntime();

console.log("[copy-extension-assets] MediaPipe wasm → public/mediapipe/wasm/");
copyMediaPipeWasm();

console.log("[copy-extension-assets] done.");
