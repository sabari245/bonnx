/// <reference lib="webworker" />
/**
 * Model worker: owns the model buffer, decoded model and tensor store; serves ModelApi over a tiny RPC.
 *   request   { id, method, args }
 *   response  { id, ok: true, result } | { id, ok: false, error }
 *   progress  { id, progress: { stage, frac } }
 */
import { loadModel, type Loaded } from "@/onnx/load";
import { lookupSchema } from "@/onnx/schema";
import type { HistogramRequest, OpenRequest, SliceRequest } from "@/onnx/types";

interface Scope {
  postMessage(m: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent) => void) | null;
}
const ctx = self as unknown as Scope;

let loaded: Loaded | null = null;
const need = (): Loaded => {
  if (!loaded) throw new Error("no model loaded");
  return loaded;
};

const methods: Record<string, (args: any, id: number) => Promise<{ result: unknown; transfer?: Transferable[] }> | { result: unknown; transfer?: Transferable[] }> = {
  async open([req]: [OpenRequest], id) {
    loaded = null;
    loaded = await loadModel(req, (p) => ctx.postMessage({ id, progress: p }));
    return { result: loaded.view };
  },
  provideExternal([path, data]: [string, ArrayBuffer]) {
    const { store } = need();
    store.setExternal(path, new Uint8Array(data));
    const changed = [];
    for (let i = 0; i < store.entries.length; i++) {
      const e = store.entries[i]!;
      const before = e.info.available;
      if (e.info.external && !before) {
        store.refresh(i);
        if (e.info.available) changed.push(e.info);
      }
    }
    // sparse tensors may depend on newly available values/indices
    for (let i = 0; i < store.entries.length; i++) {
      const e = store.entries[i]!;
      if (e.sparse && !e.info.available) {
        store.refresh(i);
        if (e.info.available) changed.push(e.info);
      }
    }
    return { result: changed };
  },
  tensorStats([tid]: [number]) {
    return { result: need().store.stats(tid) };
  },
  tensorSlice([req]: [SliceRequest]) {
    const r = need().store.slice(req);
    const transfer = r.values instanceof Float64Array ? [r.values.buffer as ArrayBuffer] : undefined;
    return { result: r, transfer };
  },
  tensorHistogram([req]: [HistogramRequest]) {
    return { result: need().store.histogram(req) };
  },
  async opSchema([domain, op, opset]: [string, string, number | null]) {
    return { result: await lookupSchema(domain, op, opset) };
  },
  dispose() {
    loaded = null;
    return { result: null };
  },
};

ctx.onmessage = async (ev: MessageEvent) => {
  const { id, method, args } = ev.data as { id: number; method: string; args: unknown[] };
  try {
    const fn = methods[method];
    if (!fn) throw new Error(`unknown method ${method}`);
    const { result, transfer } = await fn(args, id);
    ctx.postMessage({ id, ok: true, result }, transfer ?? []);
  } catch (e) {
    const err = e as Error;
    ctx.postMessage({ id, ok: false, error: { message: err?.message ?? String(e), stack: err?.stack } });
  }
};
