import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  optimizeDeps: {
    include: ["@mediapipe/tasks-vision"],
  },
  build: {
    codeSplitting: true,
    emptyOutDir: true,
    commonjsOptions: {
      transformMixedEsModules: true,
    },
    rollupOptions: {
      input: {
        content: resolve(__dirname, "src/content.js"),
        offscreen: resolve(__dirname, "src/offscreen.js"),
        background: resolve(__dirname, "src/background.js"),
      },
      output: {
        format: "iife",
        entryFileNames: "assets/[name].js",
        extend: true,
      },
    },
  },
});
