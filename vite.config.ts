import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import { viteSingleFile } from "vite-plugin-singlefile";
import { fileURLToPath } from "node:url";

const single = !!process.env.SINGLE;
export default defineConfig({
  base: "./",
  plugins: [tailwindcss(), ...(single ? [viteSingleFile()] : [])],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)), "#spawn": fileURLToPath(new URL(single ? "./src/workers/spawn.single.ts" : "./src/workers/spawn.ts", import.meta.url)) } },
  worker: { format: single ? "iife" : "es", rollupOptions: { output: { inlineDynamicImports: single } } },
  build: { target: "es2022", chunkSizeWarningLimit: 1500, assetsInlineLimit: single ? 100_000_000 : 4096, rollupOptions: { output: { inlineDynamicImports: single } } },
  test: { include: ["tests/**/*.test.ts"] },
});
