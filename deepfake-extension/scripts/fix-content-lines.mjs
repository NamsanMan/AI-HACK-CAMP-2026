import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const path = resolve(dirname(fileURLToPath(import.meta.url)), "../src/content.js");
const lines = readFileSync(path, "utf8").split("\n");

const set = (i, s) => {
  lines[i] = s;
};

set(
  104,
  '    setOverlayMessage("초기화 중", "분석 엔진을 준비하는 중입니다 (Offscreen WASM)...");'
);
set(124, '    setOverlayMessage("대기 중", "영상 화면을 찾는 중입니다.");');
set(163, '    setOverlayMessage("대기 중", "Zoom 미팅 영상을 찾는 중입니다.");');
set(174, '        setOverlayMessage("대기 중", formatSearchHint());');
set(182, "        setOverlayMessage(");
set(183, '          "대기 중",');
set(
  184,
  "          `${formatSearchHint()}\\n\\n영상 요소는 보이지만 스트림이 아직 없습니다.`"
);
set(232, '      setOverlayMessage("대기 중", waitingForMediaMessage());');
set(237, '      setOverlayMessage("대기 중", "video가 아직 준비되지 않았습니다.");');
set(473, '            title: "대기 중",');

writeFileSync(path, lines.join("\n"), "utf8");
console.log("[fix-content-lines] done");
