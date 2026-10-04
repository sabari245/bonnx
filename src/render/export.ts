import type { ColorScheme } from "./colors";
import { edgeLabelOf, edgeMid, paintEdgeLabels, paintEdges, paintNodes, type PaintState } from "./paint";
import { HD, LH, PADX, type FontSet, type RNode, type Scene, type SceneOptions } from "./scene";
import type { Theme } from "./theme";

/** Everything the exporters need; obtain from `viewer.exportContext()`. */
export interface ExportContext {
  scene: Scene;
  theme: Theme;
  fonts: FontSet;
  scheme: ColorScheme;
  edgeLabels: SceneOptions["showTypes"];
  /** selected scene node id or -1 (used for `neighborhood` exports) */
  selection: number;
}

export interface ExportOptions {
  /** include background fill (default true) */
  background?: boolean;
  /** world-unit padding around the content (default 24) */
  padding?: number;
  /** export only the selected node and nodes within N hops (needs a selection); default: whole graph */
  neighborhood?: number;
  /** PNG pixel ratio relative to world units: 1, 2 or 3 (default 2) */
  scale?: number;
  /** override edge labels */
  edgeLabels?: SceneOptions["showTypes"];
}

const MAX_PIXELS = 64e6;
const MAX_DIM = 16384;

interface Subset { nodes: RNode[]; edgeIds: number[]; x0: number; y0: number; x1: number; y1: number }

function subset(ctx: ExportContext, o: ExportOptions): Subset {
  const { scene } = ctx;
  let nodes = scene.nodes;
  if (o.neighborhood != null && ctx.selection >= 0) {
    const keep = new Set<number>([ctx.selection]);
    let frontier = [ctx.selection];
    for (let h = 0; h < o.neighborhood; h++) {
      const next: number[] = [];
      for (const e of scene.edges) {
        if (keep.has(e.s) && !keep.has(e.t)) { keep.add(e.t); next.push(e.t); }
        else if (keep.has(e.t) && !keep.has(e.s)) { keep.add(e.s); next.push(e.s); }
      }
      frontier = next;
      if (!frontier.length) break;
    }
    nodes = scene.nodes.filter((n) => keep.has(n.id));
  }
  const ids = new Set(nodes.map((n) => n.id));
  const edgeIds = scene.edges.filter((e) => e.p.length && ids.has(e.s) && ids.has(e.t)).map((e) => e.id);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const n of nodes) { x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x + n.w); y1 = Math.max(y1, n.y + n.h); }
  for (const i of edgeIds) { const p = scene.edges[i].p; for (let k = 0; k < p.length; k += 2) { x0 = Math.min(x0, p[k]); x1 = Math.max(x1, p[k]); y0 = Math.min(y0, p[k + 1]); y1 = Math.max(y1, p[k + 1]); } }
  if (!isFinite(x0)) { x0 = y0 = 0; x1 = y1 = 100; }
  return { nodes, edgeIds, x0, y0, x1, y1 };
}

const xml = (s: string): string => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" })[c]!);
const num = (v: number): string => String(Math.round(v * 100) / 100);

function bezier(p: number[]): string {
  let d = `M${num(p[0])} ${num(p[1])}`;
  for (let k = 2; k + 5 < p.length; k += 6) d += `C${num(p[k])} ${num(p[k + 1])} ${num(p[k + 2])} ${num(p[k + 3])} ${num(p[k + 4])} ${num(p[k + 5])}`;
  return d;
}

/** Vector export. Returns an SVG document string. */
export function exportSVG(ctx: ExportContext, o: ExportOptions = {}): string {
  const { scene, theme: th, scheme } = ctx;
  const pad = o.padding ?? 24;
  const s = subset(ctx, o);
  const x0 = s.x0 - pad, y0 = s.y0 - pad, W = s.x1 - s.x0 + 2 * pad, H = s.y1 - s.y0 + 2 * pad;
  const sans = xml(th.fontSans.replace(/"/g, "'"));
  const labelMode = o.edgeLabels ?? ctx.edgeLabels;
  const out: string[] = [];
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="${num(x0)} ${num(y0)} ${num(W)} ${num(H)}" width="${num(W)}" height="${num(H)}" font-family="${sans}">`);
  out.push(`<title>${xml(scene.graphName || "ONNX graph")}</title>`);
  out.push(`<defs><marker id="arr" viewBox="-7 -4 8 8" refX="0" refY="0" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0L-6.5-3.2L-6.5 3.2z" fill="${th.edge}"/></marker></defs>`);
  if (o.background !== false) out.push(`<rect x="${num(x0)}" y="${num(y0)}" width="${num(W)}" height="${num(H)}" fill="${th.bg}"/>`);

  out.push(`<g fill="none" stroke="${th.edge}" stroke-width="1.1">`);
  for (const i of s.edgeIds) {
    const e = scene.edges[i];
    out.push(`<path d="${bezier(e.p)}" marker-end="url(#arr)"${e.isConst ? ' stroke-dasharray="4 3" opacity="0.6"' : ""}/>`);
  }
  out.push("</g>");

  const F = ctx.fonts;
  const fs = (f: string): { size: string; weight: string } => {
    const m = /^(?:(\d+)\s+)?([\d.]+)px/.exec(f);
    return { weight: m?.[1] ?? "400", size: m?.[2] ?? "11" };
  };
  const text = (t: string, x: number, y: number, font: string, fill: string, anchor = "start", extra = ""): string => {
    const { size, weight } = fs(font);
    return `<text x="${num(x)}" y="${num(y)}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}" dominant-baseline="central"${extra}>${xml(t)}</text>`;
  };
  for (const n of s.nodes) {
    const { x, y, w, h } = n, p = scheme.paint(n.id);
    out.push("<g>");
    if (n.kind === "op" || n.kind === "fn") {
      const r = 5;
      if (n.kind === "fn") out.push(`<rect x="${num(x + 4)}" y="${num(y + 4)}" width="${num(w)}" height="${num(h)}" rx="${r}" fill="${th.nodeRow}" stroke="${th.nodeBorder}" stroke-width="0.9"/>`);
      out.push(`<rect x="${num(x)}" y="${num(y)}" width="${num(w)}" height="${num(h)}" rx="${r}" fill="${th.nodeBg}" stroke="${th.nodeBorder}" stroke-width="0.9"/>`);
      out.push(`<path d="M${num(x)} ${num(y + n.hdh)}V${num(y + r)}Q${num(x)} ${num(y)} ${num(x + r)} ${num(y)}H${num(x + w - r)}Q${num(x + w)} ${num(y)} ${num(x + w)} ${num(y + r)}V${num(y + n.hdh)}Z" fill="${p.fill}"/>`);
      out.push(text(n.title, x + PADX, y + n.hdh / 2 + 0.5, F.title, p.text));
      if (n.expandable) {
        const gx = x + w - PADX - 12, gy = y + (HD - 12) / 2;
        out.push(`<rect x="${num(gx)}" y="${num(gy)}" width="12" height="12" rx="3" fill="none" stroke="${p.text}" stroke-width="1.2"/><path d="M${num(gx + 3)} ${num(gy + 6)}h6M${num(gx + 6)} ${num(gy + 3)}v6" stroke="${p.text}" stroke-width="1.2"/>`);
      }
      for (const l of n.lines) {
        const ly = y + l.y + LH / 2 + 0.5;
        if (l.kind === "const") out.push(`<rect x="${num(x + 1)}" y="${num(y + l.y)}" width="${num(w - 2)}" height="${LH}" fill="${th.nodeRow}"/>`, text(l.text, x + PADX, ly, F.line, th.nodeText));
        else if (l.kind === "sub") out.push(`<rect x="${num(x + 4)}" y="${num(y + l.y)}" width="${num(w - 8)}" height="${LH}" rx="4" fill="${th.sel}" opacity="0.14"/>`, text(l.text, x + PADX + 4, ly, F.sub, th.sel));
        else if (l.kind === "metric") { const t = scheme.metricText(n); if (t) out.push(text(t, x + PADX, ly, F.lineBold, th.nodeText)); }
        else out.push(text(l.text, x + PADX, ly, F.line, th.nodeSub));
      }
    } else if (n.kind === "input" || n.kind === "output") {
      out.push(`<rect x="${num(x)}" y="${num(y)}" width="${num(w)}" height="${num(h)}" rx="9" fill="${p.fill}"/>`, text(n.title, x + w / 2, y + 13, F.io, p.text, "middle"), text(n.subtitle, x + w / 2, y + 27, F.ioSub, p.text, "middle", ' opacity="0.78"'));
    } else if (n.kind === "const") {
      out.push(`<rect x="${num(x)}" y="${num(y)}" width="${num(w)}" height="${num(h)}" rx="6" fill="${th.nodeRow}" stroke="${th.nodeBorder}" stroke-width="0.9"/><path d="M${num(x)} ${num(y + 6)}Q${num(x)} ${num(y)} ${num(x + 6)} ${num(y)}H${num(x + 4)}V${num(y + h)}H${num(x + 6)}Q${num(x)} ${num(y + h)} ${num(x)} ${num(y + h - 6)}Z" fill="${p.fill}"/>`, text(n.title, x + w / 2 + 2, y + 13, F.io, th.nodeText, "middle"), text(n.subtitle, x + w / 2 + 2, y + 27, F.ioSub, th.nodeSub, "middle"));
    } else {
      out.push(`<rect x="${num(x)}" y="${num(y)}" width="${num(w)}" height="${num(h)}" rx="6" fill="none" stroke="${th.nodeBorder}" stroke-width="0.9" stroke-dasharray="4 3"/>`, text(n.title, x + w / 2, y + 13, F.io, th.nodeSub, "middle"), text(n.outType, x + w / 2, y + 27, F.ioSub, th.nodeSub, "middle"));
    }
    out.push("</g>");
  }

  if (labelMode !== "none") {
    const dir = scene.options.direction;
    for (const i of s.edgeIds) {
      const e = scene.edges[i], t = edgeLabelOf(e, labelMode);
      if (!t || e.isConst) continue;
      const span = dir === "TB" ? Math.abs(e.p[e.p.length - 1] - e.p[1]) : Math.abs(e.p[e.p.length - 2] - e.p[0]);
      if (span < 30) continue;
      const [mx, my] = edgeMid(e.p);
      out.push(text(t, dir === "LR" ? mx : mx + 5, dir === "LR" ? my - 8 : my, F.label, th.nodeSub, dir === "LR" ? "middle" : "start", ` stroke="${th.bg}" stroke-width="3" stroke-linejoin="round" paint-order="stroke"`));
    }
  }
  out.push("</svg>");
  return out.join("\n");
}

/** Raster export rendered offscreen at `scale`× world units (reduced automatically to stay under ~64 MP). */
export async function exportPNG(ctx: ExportContext, o: ExportOptions = {}): Promise<Blob> {
  const pad = o.padding ?? 24;
  const s = subset(ctx, o);
  const x0 = s.x0 - pad, y0 = s.y0 - pad, W = s.x1 - s.x0 + 2 * pad, H = s.y1 - s.y0 + 2 * pad;
  let k = o.scale ?? 2;
  k = Math.min(k, MAX_DIM / W, MAX_DIM / H, Math.sqrt(MAX_PIXELS / (W * H)));
  const pw = Math.max(1, Math.floor(W * k)), ph = Math.max(1, Math.floor(H * k));
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(pw, ph) : Object.assign(document.createElement("canvas"), { width: pw, height: ph });
  const g = canvas.getContext("2d") as CanvasRenderingContext2D;
  if (o.background !== false) { g.fillStyle = ctx.theme.bg; g.fillRect(0, 0, pw, ph); }
  g.setTransform(k, 0, 0, k, -x0 * k, -y0 * k);
  const st: PaintState = {
    theme: ctx.theme, fonts: ctx.fonts, scheme: ctx.scheme, scale: Math.max(k, 0.5), dir: ctx.scene.options.direction,
    edgeLabels: o.edgeLabels ?? ctx.edgeLabels, sel: -1, hov: -1, selEdge: -1, matches: null, activeMatch: -1, hiEdges: null, lod: "full",
  };
  paintEdges(g, ctx.scene, s.edgeIds, st);
  paintNodes(g, s.nodes, st);
  paintEdgeLabels(g, ctx.scene, s.edgeIds, st, 5000);
  if (canvas instanceof OffscreenCanvas) return canvas.convertToBlob({ type: "image/png" });
  return new Promise((res, rej) => (canvas as HTMLCanvasElement).toBlob((b) => (b ? res(b) : rej(new Error("PNG encoding failed"))), "image/png"));
}

export const svgBlob = (svg: string): Blob => new Blob([svg], { type: "image/svg+xml;charset=utf-8" });

export function downloadBlob(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
