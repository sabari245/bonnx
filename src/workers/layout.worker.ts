/// <reference lib="webworker" />
import dagre from "@dagrejs/dagre";
import type { LayoutRequest, LayoutResult } from "../onnx/types";
import { fastLayout } from "./fast-layout";

type Msg = { token: number; req: LayoutRequest };

/** Catmull-Rom spline through pts → cubic bezier sequence: x0,y0, then (c1x,c1y,c2x,c2y,px,py)*. */
function toBezier(pts: { x: number; y: number }[], dir: "TB" | "LR"): number[] {
  // drop duplicates
  const q: { x: number; y: number }[] = [];
  for (const p of pts) {
    const l = q[q.length - 1];
    if (!l || Math.abs(l.x - p.x) > 0.01 || Math.abs(l.y - p.y) > 0.01) q.push(p);
  }
  if (q.length === 1) q.push({ x: q[0].x, y: q[0].y + 1 });
  const out: number[] = [q[0].x, q[0].y];
  if (q.length === 2) {
    const [a, b] = q;
    if (dir === "TB") {
      const m = (b.y - a.y) / 2;
      out.push(a.x, a.y + m, b.x, b.y - m, b.x, b.y);
    } else {
      const m = (b.x - a.x) / 2;
      out.push(a.x + m, a.y, b.x - m, b.y, b.x, b.y);
    }
    return out;
  }
  for (let i = 0; i < q.length - 1; i++) {
    const p0 = q[Math.max(0, i - 1)], p1 = q[i], p2 = q[i + 1], p3 = q[Math.min(q.length - 1, i + 2)];
    out.push(
      p1.x + (p2.x - p0.x) / 6, p1.y + (p2.y - p0.y) / 6,
      p2.x - (p3.x - p1.x) / 6, p2.y - (p3.y - p1.y) / 6,
      p2.x, p2.y,
    );
  }
  return out;
}

/** dagre handles up to this many nodes with good quality; beyond it (or if it overflows the stack) use fastLayout */
const DAGRE_MAX_NODES = 1500;

function dagreLayout(req: LayoutRequest): LayoutResult {
  const { nodes, edges, options } = req;
  const g = new dagre.graphlib.Graph({ multigraph: true });
  g.setGraph({ rankdir: options.dir, ranksep: options.rankSep, nodesep: options.nodeSep, edgesep: 14, marginx: 24, marginy: 24, ranker: "network-simplex" });
  g.setDefaultEdgeLabel(() => ({}));
  for (const nd of nodes) g.setNode(String(nd.id), { width: nd.w, height: nd.h });
  for (const e of edges) if (e.s !== e.t) g.setEdge(String(e.s), String(e.t), {}, String(e.id));
  dagre.layout(g);
  const graph = g.graph() as { width?: number; height?: number };
  const rn: LayoutResult["nodes"] = [];
  for (const nd of nodes) {
    const l = g.node(String(nd.id)) as { x: number; y: number };
    rn.push({ id: nd.id, x: l.x - nd.w / 2, y: l.y - nd.h / 2 });
  }
  const re: LayoutResult["edges"] = [];
  for (const e of edges) {
    if (e.s === e.t) continue;
    const l = g.edge({ v: String(e.s), w: String(e.t), name: String(e.id) }) as { points?: { x: number; y: number }[] } | undefined;
    if (l?.points?.length) re.push({ id: e.id, p: toBezier(l.points, options.dir).map(round) });
  }
  return { width: graph.width ?? 0, height: graph.height ?? 0, nodes: rn, edges: re };
}

const round = (v: number): number => Math.round(v * 100) / 100;

function layout(req: LayoutRequest): LayoutResult {
  if (req.nodes.length <= DAGRE_MAX_NODES) {
    try { return dagreLayout(req); } catch (err) { if (!(err instanceof RangeError)) throw err; }
  }
  const r = fastLayout(req);
  return {
    width: r.width, height: r.height, nodes: r.nodes,
    edges: r.edges.map((e) => ({ id: e.id, p: toBezier(e.pts, req.options.dir).map(round) })),
  };
}

self.onmessage = (ev: MessageEvent<Msg>) => {
  const { token, req } = ev.data;
  try {
    const t0 = performance.now();
    const res = layout(req);
    (self as unknown as Worker).postMessage({ token, res, ms: performance.now() - t0 });
  } catch (err) {
    (self as unknown as Worker).postMessage({ token, error: String((err as Error)?.stack ?? err) });
  }
};
