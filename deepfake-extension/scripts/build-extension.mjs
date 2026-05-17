import esbuild from "esbuild";
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const distDir = resolve(ROOT, "dist");
const publicDir = resolve(ROOT, "public");
const outDir = resolve(distDir, "assets");

if (existsSync(distDir)) {
  rmSync(distDir, { recursive: true, force: true });
}

cpSync(publicDir, distDir, { recursive: true });
mkdirSync(outDir, { recursive: true });

const shared = {
  bundle: true,
  platform: "browser",
  target: "chrome115",
  sourcemap: false,
  minify: true,
  logLevel: "info",
};

await esbuild.build({
  ...shared,
  entryPoints: [resolve(ROOT, "src/content.js")],
  format: "iife",
  outfile: resolve(outDir, "content.js"),
});

await esbuild.build({
  ...shared,
  entryPoints: [resolve(ROOT, "src/offscreen.js")],
  format: "iife",
  outfile: resolve(outDir, "offscreen.js"),
});

await esbuild.build({
  ...shared,
  entryPoints: [resolve(ROOT, "src/background.js")],
  format: "esm",
  outfile: resolve(outDir, "background.js"),
});

console.log("[build-extension] dist/assets/{content,offscreen,background}.js");
