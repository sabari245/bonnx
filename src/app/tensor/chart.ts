import { h } from "@/lib/dom";
import { cssVar } from "./util";

export interface ChartColors { fg: string; mutedFg: string; border: string; bar: string; barHi: string; bg: string; accent: string }
export const chartColors = (): ChartColors => ({
  fg: cssVar("--foreground", "#111"),
  mutedFg: cssVar("--muted-foreground", "#666"),
  border: cssVar("--border", "#ddd"),
  bar: cssVar("--chart-1", "#3b82f6"),
  barHi: cssVar("--primary", "#2563eb"),
  bg: cssVar("--background", "#fff"),
  accent: cssVar("--chart-5", "#ef4444"),
});

export interface CanvasChart {
  el: HTMLElement;
  canvas: HTMLCanvasElement;
  redraw(): void;
  size(): { w: number; h: number };
  dispose(): void;
}

/** A DPR-aware canvas that redraws on resize and theme change. `draw` receives CSS-pixel dimensions. */
export function createChartCanvas(height: number, draw: (ctx: CanvasRenderingContext2D, w: number, h: number, c: ChartColors) => void, cls = ""): CanvasChart {
  const canvas = h("canvas", { class: "block h-full w-full touch-none", "aria-hidden": "true" }) as HTMLCanvasElement;
  const el = h("div", { class: `relative w-full ${cls}`, style: { height: `${height}px` } }, canvas);
  let w = 0, hh = 0, raf = 0, dead = false;
  const render = () => {
    raf = 0;
    if (dead || !w || !hh) return;
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.round(w * dpr), ph = Math.round(hh * dpr);
    if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, hh);
    draw(ctx, w, hh, chartColors());
  };
  const redraw = () => { if (!raf) raf = requestAnimationFrame(render); };
  const ro = new ResizeObserver((es) => {
    const r = es[0]!.contentRect;
    w = Math.floor(r.width); hh = Math.floor(r.height); redraw();
  });
  ro.observe(el);
  const onTheme = () => { if (!el.isConnected) return; redraw(); };
  window.addEventListener("themechange", onTheme);
  return {
    el, canvas, redraw, size: () => ({ w, h: hh }),
    dispose() { dead = true; ro.disconnect(); window.removeEventListener("themechange", onTheme); cancelAnimationFrame(raf); },
  };
}

/** "nice" tick values spanning [lo, hi] */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const span = hi - lo;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

export function fmtTick(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(1).replace("e+", "e");
  return String(Number(v.toPrecision(4)));
}

export function fmtAxisCount(v: number): string {
  if (v >= 1e9) return +(v / 1e9).toFixed(1) + "B";
  if (v >= 1e6) return +(v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return +(v / 1e3).toFixed(1) + "K";
  return String(Math.round(v));
}

export function tooltipBox(): { el: HTMLElement; show(x: number, y: number, html: Node | string, bounds: DOMRect): void; hide(): void } {
  const el = h("div", { class: "bg-popover text-popover-foreground pointer-events-none absolute z-20 hidden rounded-md border px-2.5 py-1.5 text-xs shadow-md", role: "tooltip" });
  return {
    el,
    show(x, y, content, bounds) {
      el.replaceChildren(typeof content === "string" ? document.createTextNode(content) : content);
      el.classList.remove("hidden");
      const tw = el.offsetWidth, th = el.offsetHeight;
      let lx = x + 14, ly = y - th - 10;
      if (lx + tw > bounds.width - 4) lx = x - tw - 14;
      if (ly < 0) ly = y + 16;
      el.style.left = `${Math.max(2, lx)}px`;
      el.style.top = `${Math.max(2, ly)}px`;
    },
    hide() { el.classList.add("hidden"); },
  };
}
