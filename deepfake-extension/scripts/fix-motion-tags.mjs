import { readFileSync, writeFileSync } from "node:fs";

const path = process.argv[2] || "src/content.js";
let c = readFileSync(path, "utf8");
c = c.replaceAll('createElement("motion")', 'createElement("div")');
c = c.replaceAll("</motion>", "</div>");
writeFileSync(path, c, "utf8");
