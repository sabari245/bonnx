/**
 * Worker factories. The default build loads workers as separate files.
 * The single-file build (SINGLE=1) aliases `#spawn` to ./spawn.single.ts, which inlines them as blob workers
 * so the page also works from file:// and as one downloadable HTML file.
 */
export const spawnModelWorker = (): Worker => new Worker(new URL("./model.worker.ts", import.meta.url), { type: "module" });
export const spawnLayoutWorker = (): Worker => new Worker(new URL("./layout.worker.ts", import.meta.url), { type: "module" });
