import { spawnLayoutWorker } from "#spawn";
import type { LayoutOptions, LayoutRequest, LayoutResult } from "../onnx/types";
import type { Scene } from "../render/scene";

export class LayoutCancelled extends Error {
  constructor() { super("layout cancelled"); this.name = "LayoutCancelled"; }
}

interface Pending { token: number; resolve: (r: LayoutResult) => void; reject: (e: Error) => void }

let worker: Worker | null = null;
let pending: Pending | null = null;
let seq = 0;
export let lastLayoutMs = 0;

function spawn(): Worker {
  const w = spawnLayoutWorker();
  w.onmessage = (ev: MessageEvent<{ token: number; res?: LayoutResult; error?: string; ms?: number }>) => {
    const p = pending;
    if (!p || p.token !== ev.data.token) return;
    pending = null;
    if (ev.data.error) p.reject(new Error(ev.data.error));
    else { lastLayoutMs = ev.data.ms ?? 0; p.resolve(ev.data.res!); }
  };
  w.onerror = (e) => {
    const p = pending;
    pending = null;
    p?.reject(new Error(e.message || "layout worker error"));
  };
  return w;
}

/** Abort the in-flight layout (hard: the worker is terminated, since dagre cannot be interrupted). */
export function cancelLayout(): void {
  if (pending) {
    const p = pending;
    pending = null;
    worker?.terminate();
    worker = null;
    p.reject(new LayoutCancelled());
  }
}

export function layoutRequest(req: LayoutRequest): Promise<LayoutResult> {
  cancelLayout(); // a newer request supersedes the older one
  worker ??= spawn();
  const token = ++seq;
  return new Promise<LayoutResult>((resolve, reject) => {
    pending = { token, resolve, reject };
    worker!.postMessage({ token, req });
  });
}

export function disposeLayoutWorker(): void {
  cancelLayout();
  worker?.terminate();
  worker = null;
}

export const layoutOptionsFor = (dir: "TB" | "LR", nodeCount: number): LayoutOptions =>
  dir === "TB"
    ? { dir, rankSep: nodeCount > 3000 ? 30 : 44, nodeSep: 28 }
    : { dir, rankSep: 84, nodeSep: 20 };

/**
 * Lay out a scene in the worker and write x/y/p/width/height back into it.
 * Rejects with LayoutCancelled if superseded by another call or cancelLayout().
 */
export async function layoutScene(scene: Scene, options?: Partial<LayoutOptions>): Promise<void> {
  const base = layoutOptionsFor(scene.options.direction, scene.nodes.length);
  const req: LayoutRequest = {
    nodes: scene.nodes.map((n) => ({ id: n.id, w: n.w, h: n.h })),
    edges: scene.edges.map((e) => ({ id: e.id, s: e.s, t: e.t })),
    options: { ...base, ...options },
  };
  const res = await layoutRequest(req);
  for (const r of res.nodes) { const n = scene.nodes[r.id]; n.x = r.x; n.y = r.y; }
  for (const e of scene.edges) e.p = [];
  for (const r of res.edges) scene.edges[r.id].p = r.p;
  scene.width = res.width;
  scene.height = res.height;
  scene.laidOut = true;
}
