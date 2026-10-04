import { spawnModelWorker } from "#spawn";
import type { ModelApi, OpenRequest } from "@/onnx/types";

type Pending = {
  resolve(v: unknown): void;
  reject(e: Error): void;
  progress?: (p: { stage: string; frac: number }) => void;
};

/**
 * Main-thread proxy for the model worker.
 * NOTE: ArrayBuffers passed to open()/provideExternal() are TRANSFERRED (detached for the caller).
 */
export function createModelClient(): ModelApi & { terminate(): void } {
  const worker = spawnModelWorker();
  const pending = new Map<number, Pending>();
  let seq = 0;

  worker.onmessage = (ev: MessageEvent) => {
    const m = ev.data as { id: number; ok?: boolean; result?: unknown; error?: { message: string; stack?: string }; progress?: { stage: string; frac: number } };
    const p = pending.get(m.id);
    if (!p) return;
    if (m.progress) return p.progress?.(m.progress);
    pending.delete(m.id);
    if (m.ok) p.resolve(m.result);
    else {
      const e = new Error(m.error?.message ?? "worker error");
      if (m.error?.stack) e.stack = m.error.stack;
      p.reject(e);
    }
  };
  worker.onerror = (ev) => {
    const e = new Error(ev.message || "model worker crashed");
    for (const p of pending.values()) p.reject(e);
    pending.clear();
  };

  function call<T>(method: string, args: unknown[], transfer: Transferable[] = [], progress?: Pending["progress"]): Promise<T> {
    const id = ++seq;
    return new Promise<T>((resolve, reject) => {
      pending.set(id, { resolve: resolve as (v: unknown) => void, reject, progress });
      worker.postMessage({ id, method, args }, transfer);
    });
  }

  return {
    open(req: OpenRequest, onProgress) {
      const transfer: Transferable[] = [];
      if (req.data instanceof ArrayBuffer) transfer.push(req.data);
      for (const b of Object.values(req.external ?? {})) if (!transfer.includes(b)) transfer.push(b);
      return call("open", [req], transfer, onProgress);
    },
    provideExternal: (path, data) => call("provideExternal", [path, data], [data]),
    tensorStats: (id) => call("tensorStats", [id]),
    tensorSlice: (req) => call("tensorSlice", [req]),
    tensorHistogram: (req) => call("tensorHistogram", [req]),
    opSchema: (domain, op, opset) => call("opSchema", [domain, op, opset]),
    dispose: () => call("dispose", []),
    terminate: () => {
      worker.terminate();
      for (const p of pending.values()) p.reject(new Error("terminated"));
      pending.clear();
    },
  };
}
