/**
 * Stateless canvas painters shared by the interactive viewer and the PNG exporter.
 * Callers set the world transform; everything here draws in world coordinates.
 */
import { HD, LH, PADX, type FontSet, type RNode, type REdge, type Scene, type SceneOptions } from "./scene";
import type { ColorScheme } from "./colors";
import type { Theme } from "./theme";

export interface PaintState {
  theme: Theme;
  fonts: FontSet;
  scheme: ColorScheme;
  scale: number;
  dir: "TB" | "LR";
  edgeLabels: SceneOptions["showTypes"];
  sel: number;
  hov: number;
  selEdge: number;
  matches: ReadonlySet<number> | null;
  activeMatch: number;
  /** edges to draw in the highlight colour (incident to hovered / selected node) */
  hiEdges: ReadonlySet<number> | null;
  /** 'full' disables level-of-detail (exports) */
  lod: "auto" | "full";
}

export const FLAT_SCALE = 0.22;
export const TEXT_PX = 5.5; // minimum rendered header font size in px for text drawing

/* ───────────────────────── edges ───────────────────────── */

function pathEdge(ctx: CanvasRenderingContext2D, p: number[]): void {
  ctx.moveTo(p[0], p[1]);
  for (let k = 2; k + 5 < p.length; k += 6) ctx.bezierCurveTo(p[k], p[k + 1], p[k + 2], p[k + 3], p[k + 4], p[k + 5]);
}
function arrow(ctx: CanvasRenderingContext2D, p: number[], s: number): void {
  const L = p.length, x = p[L - 2], y = p[L - 1];
  let dx = x - p[L - 4], dy = y - p[L - 3];
  const d = Math.hypot(dx, dy) || 1;
  dx /= d; dy /= d;
  const a = 6.5 * s, b = 3.2 * s;
  ctx.moveTo(x, y);
  ctx.lineTo(x - dx * a - dy * b, y - dy * a + dx * b);
  ctx.lineTo(x - dx * a + dy * b, y - dy * a - dx * b);
  ctx.closePath();
}

/** midpoint of the edge path (for labels) */
export function edgeMid(p: number[]): [number, number] {
  const segs = (p.length / 2 - 1) / 3;
  if (segs % 2 === 0) { const k = (segs / 2) * 6; return [p[k], p[k + 1]]; }
  const k = ((segs - 1) / 2) * 6;
  return [0.125 * p[k] + 0.375 * p[k + 2] + 0.375 * p[k + 4] + 0.125 * p[k + 6], 0.125 * p[k + 1] + 0.375 * p[k + 3] + 0.375 * p[k + 5] + 0.125 * p[k + 7]];
}

export function paintEdges(ctx: CanvasRenderingContext2D, scene: Scene, ids: readonly number[], st: PaintState): void {
  const { scale, theme } = st;
  const E = scene.edges;
  const hi = st.hiEdges;
  const lw = Math.max(1, Math.min(1.1 / scale, 8));
  const arrows = scale > 0.3 || st.lod === "full";
  const as = Math.min(Math.max(1, 0.6 / scale), 4);
  for (let pass = 0; pass < 2; pass++) {
    // pass 0: data edges, pass 1: constant edges
    ctx.beginPath();
    for (const i of ids) { const e = E[i]; if (e.p.length && e.isConst === (pass === 1) && !hi?.has(i)) pathEdge(ctx, e.p); }
    ctx.lineWidth = lw;
    ctx.strokeStyle = theme.edge;
    ctx.globalAlpha = (pass === 1 ? 0.55 : 1) * (scale < 0.15 ? 0.55 : 1);
    if (pass === 1 && scale > 0.4) ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    if (arrows) {
      ctx.fillStyle = theme.edge;
      ctx.globalAlpha = pass === 1 ? 0.55 : 1;
      ctx.beginPath();
      for (const i of ids) { const e = E[i]; if (e.p.length && e.isConst === (pass === 1) && !hi?.has(i)) arrow(ctx, e.p, as); }
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }
  if (hi && hi.size) {
    ctx.strokeStyle = theme.edgeHi;
    ctx.lineWidth = Math.max(2, 2 / scale);
    ctx.beginPath();
    for (const i of hi) { const e = E[i]; if (e.p.length) pathEdge(ctx, e.p); }
    ctx.stroke();
    ctx.fillStyle = theme.edgeHi;
    ctx.beginPath();
    for (const i of hi) { const e = E[i]; if (e.p.length) arrow(ctx, e.p, as * 1.5); }
    ctx.fill();
  }
  if (st.selEdge >= 0 && E[st.selEdge]?.p.length) {
    const e = E[st.selEdge];
    ctx.strokeStyle = theme.sel;
    ctx.lineWidth = Math.max(2.6, 3 / scale);
    ctx.beginPath(); pathEdge(ctx, e.p); ctx.stroke();
    ctx.fillStyle = theme.sel;
    ctx.beginPath(); arrow(ctx, e.p, as * 1.7); ctx.fill();
  }
}

export function edgeLabelOf(e: REdge, mode: SceneOptions["showTypes"]): string {
  return mode === "none" ? "" : mode === "name" ? e.name : mode === "shape" ? e.typeText : `${e.name}  ${e.typeText}`;
}

export function paintEdgeLabels(ctx: CanvasRenderingContext2D, scene: Scene, ids: readonly number[], st: PaintState, limit = 700): void {
  if (st.edgeLabels === "none") return;
  const { theme, dir } = st;
  ctx.font = st.fonts.label;
  ctx.textBaseline = "middle";
  ctx.textAlign = dir === "LR" ? "center" : "left";
  let n = 0;
  const pad = 3;
  for (const i of ids) {
    const e = scene.edges[i];
    if (!e.p.length) continue;
    if (e.isConst) continue; // constants are described by their own row / node
    const span = dir === "TB" ? Math.abs(e.p[e.p.length - 1] - e.p[1]) : Math.abs(e.p[e.p.length - 2] - e.p[0]);
    if (span < 30) continue;
    const text = edgeLabelOf(e, st.edgeLabels);
    if (!text) continue;
    const [mx, my] = edgeMid(e.p);
    const tw = ctx.measureText(text).width;
    if (dir === "LR" && span < tw + 14) continue; // no room between the ranks
    const x = dir === "LR" ? mx : mx + 5, y = dir === "LR" ? my - 8 : my;
    const hot = i === st.selEdge || st.hiEdges?.has(i);
    ctx.globalAlpha = 0.88;
    ctx.fillStyle = theme.bg;
    const bx = dir === "LR" ? x - tw / 2 - pad : x - 2;
    ctx.beginPath(); ctx.roundRect(bx, y - 7, tw + pad * 2, 14, 4); ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = hot ? theme.edgeHi : theme.nodeSub;
    ctx.fillText(text, x + (dir === "LR" ? 0 : pad - 2), y + 0.5);
    if (++n >= limit) break;
  }
}

/* ───────────────────────── nodes ───────────────────────── */

export function paintNodes(ctx: CanvasRenderingContext2D, nodes: readonly RNode[], st: PaintState): void {
  const { scale, scheme } = st;
  if (st.lod !== "full" && scale < FLAT_SCALE) {
    const groups = new Map<number, RNode[]>();
    for (const n of nodes) { const c = scheme.codes[n.id]; let g = groups.get(c); if (!g) groups.set(c, (g = [])); g.push(n); }
    for (const [c, g] of groups) {
      ctx.fillStyle = scheme.paintOfCode(c).fill;
      ctx.beginPath();
      for (const n of g) ctx.rect(n.x, n.y, n.w, n.h);
      ctx.fill();
    }
    return;
  }
  const text = st.lod === "full" || scale * 12 >= TEXT_PX;
  const fine = st.lod === "full" || scale >= 0.45; // borders + decorative detail
  let curFont = "";
  const setFont = (f: string): void => { if (f !== curFont) { ctx.font = f; curFont = f; } };
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  const { theme, fonts } = st;
  const bw = Math.max(0.7, 0.9 / scale);

  for (const n of nodes) {
    const { x, y, w, h } = n;
    const p = scheme.paint(n.id);
    if (n.kind === "op" || n.kind === "fn") {
      const r = 5;
      if (n.kind === "fn" && fine) {
        ctx.fillStyle = theme.nodeRow; ctx.strokeStyle = theme.nodeBorder; ctx.lineWidth = bw;
        ctx.beginPath(); ctx.roundRect(x + 4, y + 4, w, h, r); ctx.fill(); ctx.stroke();
      }
      ctx.fillStyle = theme.nodeBg;
      ctx.beginPath(); ctx.roundRect(x, y, w, h, r); ctx.fill();
      ctx.fillStyle = p.fill;
      ctx.beginPath(); ctx.roundRect(x, y, w, n.hdh, [r, r, 0, 0]); ctx.fill();
      if (fine) {
        ctx.strokeStyle = theme.nodeBorder; ctx.lineWidth = bw;
        ctx.beginPath(); ctx.roundRect(x, y, w, h, r); ctx.stroke();
      }
      if (!text) continue;
      setFont(fonts.title);
      ctx.fillStyle = p.text;
      ctx.fillText(n.title, x + PADX, y + n.hdh / 2 + 0.5);
      if (n.expandable) {
        const gx = x + w - PADX - 12, gy = y + (n.hdh - 12) / 2;
        ctx.strokeStyle = p.text; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.roundRect(gx, gy, 12, 12, 3); ctx.moveTo(gx + 3, gy + 6); ctx.lineTo(gx + 9, gy + 6); ctx.moveTo(gx + 6, gy + 3); ctx.lineTo(gx + 6, gy + 9); ctx.stroke();
      }
      for (const l of n.lines) {
        const ly = y + l.y + LH / 2;
        switch (l.kind) {
          case "const":
            ctx.fillStyle = theme.nodeRow; ctx.fillRect(x + 1, y + l.y, w - 2, LH);
            setFont(fonts.line); ctx.fillStyle = theme.nodeText; ctx.fillText(l.text, x + PADX, ly + 0.5);
            break;
          case "attr": case "name": case "more":
            setFont(fonts.line); ctx.fillStyle = theme.nodeSub; ctx.fillText(l.text, x + PADX, ly + 0.5);
            break;
          case "sub":
            ctx.globalAlpha = 0.14; ctx.fillStyle = theme.sel; ctx.beginPath(); ctx.roundRect(x + 4, y + l.y, w - 8, LH, 4); ctx.fill(); ctx.globalAlpha = 1;
            setFont(fonts.sub); ctx.fillStyle = theme.sel; ctx.fillText(l.text, x + PADX + 4, ly + 0.5);
            break;
          case "metric": {
            const t = scheme.metricText(n);
            if (!t) break;
            setFont(fonts.lineBold); ctx.fillStyle = theme.nodeText; ctx.fillText(t, x + PADX, ly + 0.5, w - 2 * PADX);
            break;
          }
        }
      }
    } else if (n.kind === "input" || n.kind === "output") {
      ctx.fillStyle = p.fill;
      ctx.beginPath(); ctx.roundRect(x, y, w, h, 9); ctx.fill();
      if (!text) continue;
      ctx.textAlign = "center";
      setFont(fonts.io); ctx.fillStyle = p.text; ctx.fillText(n.title, x + w / 2, y + 13, w - 10);
      setFont(fonts.ioSub); ctx.globalAlpha = 0.78; ctx.fillText(n.subtitle, x + w / 2, y + 27, w - 10); ctx.globalAlpha = 1;
      ctx.textAlign = "left";
    } else if (n.kind === "const") {
      ctx.fillStyle = theme.nodeRow; ctx.strokeStyle = theme.nodeBorder; ctx.lineWidth = bw;
      ctx.beginPath(); ctx.roundRect(x, y, w, h, 6); ctx.fill();
      if (fine) ctx.stroke();
      ctx.fillStyle = p.fill; ctx.beginPath(); ctx.roundRect(x, y, 4, h, [6, 0, 0, 6]); ctx.fill();
      if (!text) continue;
      ctx.textAlign = "center";
      setFont(fonts.io); ctx.fillStyle = theme.nodeText; ctx.fillText(n.title, x + w / 2 + 2, y + 13, w - 14);
      setFont(fonts.ioSub); ctx.fillStyle = theme.nodeSub; ctx.fillText(n.subtitle, x + w / 2 + 2, y + 27, w - 14);
      ctx.textAlign = "left";
    } else {
      ctx.strokeStyle = theme.nodeBorder; ctx.lineWidth = bw; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.roundRect(x, y, w, h, 6); ctx.stroke(); ctx.setLineDash([]);
      if (!text) continue;
      ctx.textAlign = "center";
      setFont(fonts.io); ctx.fillStyle = theme.nodeSub; ctx.fillText(n.title, x + w / 2, y + 13, w - 10);
      setFont(fonts.ioSub); ctx.fillText(n.outType, x + w / 2, y + 27, w - 10);
      ctx.textAlign = "left";
    }
  }
}

/** selection / hover / search-match outlines (only for the given visible nodes + sel + hov) */
export function paintOverlays(ctx: CanvasRenderingContext2D, scene: Scene, visible: readonly RNode[], st: PaintState): void {
  const { scale, theme } = st;
  const N = scene.nodes;
  if (st.matches && st.matches.size) {
    ctx.strokeStyle = theme.match;
    ctx.lineWidth = Math.max(2, 2.2 / scale);
    ctx.beginPath();
    for (const n of visible) if (st.matches.has(n.id) && n.id !== st.activeMatch) ctx.roundRect(n.x - 3, n.y - 3, n.w + 6, n.h + 6, 7);
    ctx.stroke();
    const a = N[st.activeMatch];
    if (a && st.matches.has(a.id)) {
      ctx.lineWidth = Math.max(3.5, 4 / scale);
      ctx.beginPath(); ctx.roundRect(a.x - 4, a.y - 4, a.w + 8, a.h + 8, 8); ctx.stroke();
    }
  }
  if (st.hov >= 0 && st.hov !== st.sel && N[st.hov]) {
    const n = N[st.hov];
    ctx.strokeStyle = theme.sel; ctx.globalAlpha = 0.6; ctx.lineWidth = Math.max(1.5, 1.6 / scale);
    ctx.beginPath(); ctx.roundRect(n.x - 2, n.y - 2, n.w + 4, n.h + 4, 6); ctx.stroke(); ctx.globalAlpha = 1;
  }
  if (st.sel >= 0 && N[st.sel]) {
    const n = N[st.sel];
    ctx.strokeStyle = theme.sel; ctx.lineWidth = Math.max(2.5, 3 / scale);
    ctx.beginPath(); ctx.roundRect(n.x - 3, n.y - 3, n.w + 6, n.h + 6, 7); ctx.stroke();
  }
}

/** dotted background tile as a repeating pattern (24 world units) */
export function makeGridPattern(ctx: CanvasRenderingContext2D, theme: Theme): CanvasPattern | null {
  const c = document.createElement("canvas");
  c.width = c.height = 24;
  const g = c.getContext("2d")!;
  g.fillStyle = theme.grid;
  g.beginPath(); g.arc(12, 12, 1, 0, Math.PI * 2); g.fill();
  return ctx.createPattern(c, "repeat");
}

/** which sub-row (if any) of an expandable card is at world point (wx,wy) */
export function subRowAt(n: RNode, wx: number, wy: number): { graph: number; attr: string } | null {
  if (!n.expandable) return null;
  const ry = wy - n.y;
  for (const l of n.lines) if (l.kind === "sub" && l.graph != null && ry >= l.y && ry <= l.y + LH && wx >= n.x + 4 && wx <= n.x + n.w - 4) return { graph: l.graph, attr: l.attr ?? "" };
  void HD;
  return null;
}
