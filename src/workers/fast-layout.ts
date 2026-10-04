/**
 * Iterative (non-recursive) layered graph layout for large graphs.
 * Dagre recurses once per rank in several phases and overflows the worker's call stack on deep graphs
 * (≈2–3k ranks), so graphs above a size threshold - or on which dagre throws - are laid out here:
 *   1. cycle removal (iterative DFS)  2. longest-path ranking, sources pulled next to their consumers
 *   3. dummy nodes for long edges     4. barycentre crossing reduction (down/up sweeps)
 *   5. x placement by isotonic regression (pool-adjacent-violators) towards neighbour means
 *   6. polyline routes through the dummy nodes (smoothed to beziers by the caller)
 */
import type { LayoutRequest } from "../onnx/types";

export interface Pt { x: number; y: number }
export interface FastResult {
  width: number;
  height: number;
  nodes: { id: number; x: number; y: number }[];
  edges: { id: number; pts: Pt[] }[];
}

const MARGIN = 24;
const LONG_EDGE = 160; // edges spanning more ranks than this get no dummy chain

export function fastLayout(req: LayoutRequest): FastResult {
  const { nodes, edges, options } = req;
  const LR = options.dir === "LR";
  const n = nodes.length;
  const sep = options.nodeSep, rankSep = options.rankSep;
  const idx = new Map<number, number>();
  nodes.forEach((nd, i) => idx.set(nd.id, i));

  /* node extents along (cross axis, rank axis) */
  const W: number[] = new Array(n), RH: number[] = new Array(n);
  nodes.forEach((nd, i) => { W[i] = LR ? nd.h : nd.w; RH[i] = LR ? nd.w : nd.h; });

  /* unique node pairs */
  const pairOf = new Map<number, number>();
  const pu: number[] = [], pv: number[] = [];
  const edgePair: number[] = new Array(edges.length).fill(-1);
  edges.forEach((e, i) => {
    const u = idx.get(e.s), v = idx.get(e.t);
    if (u === undefined || v === undefined || u === v) return;
    const key = u * n + v;
    let p = pairOf.get(key);
    if (p === undefined) { p = pu.length; pairOf.set(key, p); pu.push(u); pv.push(v); }
    edgePair[i] = p;
  });
  const P = pu.length;

  /* 1. cycle removal */
  const out: number[][] = Array.from({ length: n }, () => []);
  const indeg0 = new Int32Array(n);
  for (let p = 0; p < P; p++) { out[pu[p]].push(p); indeg0[pv[p]]++; }
  const color = new Uint8Array(n), it = new Int32Array(n), rev = new Uint8Array(P);
  const stack: number[] = [];
  const dfsFrom = (s: number): void => {
    color[s] = 1; stack.push(s);
    while (stack.length) {
      const v = stack[stack.length - 1];
      if (it[v] < out[v].length) {
        const p = out[v][it[v]++], w = pv[p];
        if (color[w] === 1) rev[p] = 1;
        else if (color[w] === 0) { color[w] = 1; stack.push(w); }
      } else { color[v] = 2; stack.pop(); }
    }
  };
  for (let v = 0; v < n; v++) if (!indeg0[v] && !color[v]) dfsFrom(v);
  for (let v = 0; v < n; v++) if (!color[v]) dfsFrom(v);

  /* DAG adjacency */
  const from = (p: number): number => (rev[p] ? pv[p] : pu[p]);
  const to = (p: number): number => (rev[p] ? pu[p] : pv[p]);
  const dout: number[][] = Array.from({ length: n }, () => []);
  const din = new Int32Array(n);
  for (let p = 0; p < P; p++) { dout[from(p)].push(p); din[to(p)]++; }

  /* 2. ranking */
  const rank = new Int32Array(n);
  const indeg = Int32Array.from(din);
  const topo: number[] = [];
  for (let v = 0; v < n; v++) if (!indeg[v]) topo.push(v);
  for (let q = 0; q < topo.length; q++) {
    const u = topo[q];
    for (const p of dout[u]) { const b = to(p); if (rank[b] < rank[u] + 1) rank[b] = rank[u] + 1; if (--indeg[b] === 0) topo.push(b); }
  }
  for (let q = topo.length - 1; q >= 0; q--) {
    const v = topo[q];
    if (din[v] || !dout[v].length) continue;
    let m = Infinity;
    for (const p of dout[v]) m = Math.min(m, rank[to(p)]);
    if (m - 1 > rank[v]) rank[v] = m - 1;
  }
  let maxRank = 0;
  for (let v = 0; v < n; v++) maxRank = Math.max(maxRank, rank[v]);
  const R = maxRank + 1;

  /* 3. layered graph with dummies */
  const WW = W.slice(), RK: number[] = Array.from(rank), isDummy: boolean[] = new Array(n).fill(false);
  const up: number[][] = Array.from({ length: n }, () => []), down: number[][] = Array.from({ length: n }, () => []);
  const link = (a: number, b: number): void => { down[a].push(b); up[b].push(a); };
  const chain: (number[] | null)[] = new Array(P).fill(null);
  for (let p = 0; p < P; p++) {
    const a = from(p), b = to(p), len = rank[b] - rank[a];
    if (len === 1) { link(a, b); chain[p] = []; }
    else if (len > 1 && len <= LONG_EDGE) {
      const ds: number[] = [];
      let prev = a;
      for (let k = 1; k < len; k++) {
        const d = WW.length;
        WW.push(8); RK.push(rank[a] + k); isDummy.push(true); up.push([]); down.push([]);
        link(prev, d); ds.push(d); prev = d;
      }
      link(prev, b); chain[p] = ds;
    }
  }
  const total = WW.length;

  /* 4. ordering */
  const layers: number[][] = Array.from({ length: R }, () => []);
  const seen = new Uint8Array(total);
  const visit = (s: number): void => {
    const st = [s]; seen[s] = 1; layers[RK[s]].push(s);
    const iter = new Map<number, number>();
    while (st.length) {
      const v = st[st.length - 1];
      const i = iter.get(v) ?? 0;
      if (i < down[v].length) {
        iter.set(v, i + 1);
        const w = down[v][i];
        if (!seen[w]) { seen[w] = 1; layers[RK[w]].push(w); st.push(w); }
      } else st.pop();
    }
  };
  for (let v = 0; v < total; v++) if (!seen[v] && !up[v].length) visit(v);
  for (let v = 0; v < total; v++) if (!seen[v]) visit(v);
  const pos = new Float64Array(total);
  const setPos = (L: number[]): void => { for (let i = 0; i < L.length; i++) pos[L[i]] = i; };
  layers.forEach(setPos);
  const key = new Float64Array(total);
  const sweep = (r: number, nb: number[][]): void => {
    const L = layers[r];
    for (const v of L) {
      const a = nb[v];
      if (!a.length) { key[v] = pos[v]; continue; }
      let s = 0;
      for (const w of a) s += pos[w];
      key[v] = s / a.length;
    }
    L.sort((a, b) => key[a] - key[b] || pos[a] - pos[b]);
    setPos(L);
  };
  const sweeps = total > 40000 ? 3 : total > 12000 ? 5 : 8;
  for (let s = 0; s < sweeps; s++) {
    for (let r = 1; r < R; r++) sweep(r, up);
    for (let r = R - 2; r >= 0; r--) sweep(r, down);
  }

  /* 5. x placement */
  const x = new Float64Array(total);
  const gap = (a: number, b: number): number => (WW[a] + WW[b]) / 2 + (isDummy[a] || isDummy[b] ? 6 : sep);
  const ys: number[] = [], ws: number[] = [], bs: { sy: number; sw: number; n: number; start: number }[] = [];
  const place = (L: number[], desired: Float64Array, weight: Float64Array): void => {
    const m = L.length;
    if (!m) return;
    // offsets of minimal-spacing packing
    let o = 0;
    ys.length = ws.length = 0; bs.length = 0;
    const offs: number[] = new Array(m);
    for (let i = 0; i < m; i++) { if (i) o += gap(L[i - 1], L[i]); offs[i] = o; }
    for (let i = 0; i < m; i++) {
      const y = desired[L[i]] - offs[i], w = weight[L[i]];
      bs.push({ sy: y * w, sw: w, n: 1, start: i });
      while (bs.length > 1) {
        const b = bs[bs.length - 1], a = bs[bs.length - 2];
        if (a.sy / a.sw <= b.sy / b.sw) break;
        a.sy += b.sy; a.sw += b.sw; a.n += b.n; bs.pop();
      }
    }
    for (const b of bs) { const z = b.sy / b.sw; for (let i = b.start; i < b.start + b.n; i++) x[L[i]] = z + offs[i]; }
  };
  const desired = new Float64Array(total), weight = new Float64Array(total);
  // initial packing
  for (const L of layers) { let c = 0; for (let i = 0; i < L.length; i++) { if (i) c += gap(L[i - 1], L[i]); x[L[i]] = c; } }
  const pass = (r: number, mode: 0 | 1 | 2): void => {
    const L = layers[r];
    for (const v of L) {
      let s = 0, c = 0;
      if (mode !== 1) for (const w of up[v]) { s += x[w]; c++; }
      if (mode !== 0) for (const w of down[v]) { s += x[w]; c++; }
      desired[v] = c ? s / c : x[v];
      weight[v] = (isDummy[v] ? 3 : 1) * Math.max(0.2, c);
    }
    place(L, desired, weight);
  };
  const iters = total > 40000 ? 2 : total > 12000 ? 3 : 5;
  for (let k = 0; k < iters; k++) {
    for (let r = 1; r < R; r++) pass(r, 0);
    for (let r = R - 2; r >= 0; r--) pass(r, 1);
    for (let r = 0; r < R; r++) pass(r, 2);
  }
  let minX = Infinity, maxX = -Infinity;
  for (let v = 0; v < total; v++) { minX = Math.min(minX, x[v] - WW[v] / 2); maxX = Math.max(maxX, x[v] + WW[v] / 2); }
  const dx = MARGIN - minX;
  for (let v = 0; v < total; v++) x[v] += dx;

  /* 6. rank axis + routes */
  const band = new Float64Array(R);
  for (let v = 0; v < n; v++) band[RK[v]] = Math.max(band[RK[v]], RH[v]);
  const rc = new Float64Array(R);
  let top = MARGIN;
  for (let r = 0; r < R; r++) { rc[r] = top + band[r] / 2; top += band[r] + (band[r] ? rankSep : rankSep / 3); }
  const crossMax = maxX + dx + MARGIN, rankMax = top - rankSep + MARGIN;
  const world = (c: number, r: number): Pt => (LR ? { x: r, y: c } : { x: c, y: r });

  const rn = nodes.map((nd, i) => {
    const p = world(x[i], rc[rank[i]]);
    return { id: nd.id, x: p.x - nd.w / 2, y: p.y - nd.h / 2 };
  });
  const re: FastResult["edges"] = [];
  const cache = new Map<number, Pt[]>();
  edges.forEach((e, i) => {
    const p = edgePair[i];
    if (p < 0) return;
    let pts = cache.get(p);
    if (!pts) {
      const a = from(p), b = to(p);
      pts = [world(x[a], rc[rank[a]] + RH[a] / 2)];
      for (const d of chain[p] ?? []) pts.push(world(x[d], rc[RK[d]]));
      pts.push(world(x[b], rc[rank[b]] - RH[b] / 2));
      if (rev[p]) pts = pts.slice().reverse();
      cache.set(p, pts);
    }
    re.push({ id: e.id, pts });
  });
  return { width: LR ? rankMax : crossMax, height: LR ? crossMax : rankMax, nodes: rn, edges: re };
}
