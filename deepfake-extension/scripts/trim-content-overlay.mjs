import { readFileSync, writeFileSync } from "node:fs";

const path = "src/content.js";
const lines = readFileSync(path, "utf8").split("\n");
const cut = lines.findIndex((l, i) => i > 550 && l.startsWith("function createOverlay"));
if (cut === -1) {
  console.log("no cut needed");
  process.exit(0);
}
const tail = [
  "",
  "if (window.__deepfakeDetectorStarted) {",
  '  console.log("[Deepfake Extension] already started. Skip.");',
  "} else {",
  "  window.__deepfakeDetectorStarted = true;",
  "  main();",
  "}",
  "",
];
writeFileSync(path, [...lines.slice(0, cut), ...tail].join("\n"), "utf8");
console.log("trimmed at line", cut + 1);
