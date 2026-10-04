import type { GraphViewer } from "./viewer";

export interface MinimapOptions {
  maxWidth: number;
  maxHeight: number;
}

/** Overview of the whole graph with the current viewport; click or drag to move the main view. */
export class Minimap {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private img: HTMLCanvasElement | null = null;
  private s = 1; private ox = 4; private oy = 4;
  private w = 0; private h = 0; private dpr = 1;
  private queued = false;
  private dragging = false;
  private offs: (() => void)[] = [];
  private opts: MinimapOptions;

  constructor(readonly container: HTMLElement, private viewer: GraphViewer, options: Partial<MinimapOptions> = {}) {
    this.opts = { maxWidth: 160, maxHeight: 280, ...options };
    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("aria-label", "Graph overview. Click or drag to move the view.");
    this.canvas.setAttribute("role", "img");
    Object.assign(this.canvas.style, { display: "none", cursor: "pointer", touchAction: "none" });
    container.append(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;

    const c = this.canvas;
    const down = (e: PointerEvent): void => { this.dragging = true; c.setPointerCapture(e.pointerId); this.jump(e); };
    const move = (e: PointerEvent): void => { if (this.dragging) this.jump(e); };
    const up = (): void => { this.dragging = false; };
    c.addEventListener("pointerdown", down); c.addEventListener("pointermove", move);
    c.addEventListener("pointerup", up); c.addEventListener("pointercancel", up);
    this.offs.push(
      () => { c.removeEventListener("pointerdown", down); c.removeEventListener("pointermove", move); c.removeEventListener("pointerup", up); c.removeEventListener("pointercancel", up); },
      viewer.on("scene", () => { this.rebuild(); }),
      viewer.on("viewchange", () => this.schedule()),
    );
    this.rebuild();
  }

  setMaxSize(maxWidth: number, maxHeight: number): void { this.opts = { maxWidth, maxHeight }; this.rebuild(); }

  private rebuild(): void {
    const sc = this.viewer.scene, scheme = this.viewer.scheme;
    if (!sc || !scheme || !sc.nodes.length) { this.canvas.style.display = "none"; this.img = null; return; }
    this.canvas.style.display = "block";
    const b = this.viewer.worldBounds, W = b.x1 - b.x0, H = b.y1 - b.y0;
    const pad = 4;
    this.dpr = window.devicePixelRatio || 1;
    this.s = Math.min((this.opts.maxWidth - 2 * pad) / W, (this.opts.maxHeight - 2 * pad) / H);
    this.w = Math.ceil(W * this.s) + 2 * pad;
    this.h = Math.ceil(H * this.s) + 2 * pad;
    this.ox = pad - b.x0 * this.s; this.oy = pad - b.y0 * this.s;
    const c = this.canvas;
    c.style.width = this.w + "px"; c.style.height = this.h + "px";
    c.width = Math.round(this.w * this.dpr); c.height = Math.round(this.h * this.dpr);
    const oc = document.createElement("canvas");
    oc.width = c.width; oc.height = c.height;
    const o = oc.getContext("2d")!;
    o.scale(this.dpr, this.dpr);
    const th = this.viewer.theme;
    o.fillStyle = th.bg; o.fillRect(0, 0, this.w, this.h);
    // edges as hairlines for orientation
    o.strokeStyle = th.edge; o.globalAlpha = 0.35; o.lineWidth = 0.5; o.beginPath();
    if (sc.edges.length < 20000) for (const e of sc.edges) { const p = e.p; if (p.length >= 4) { o.moveTo(p[0] * this.s + this.ox, p[1] * this.s + this.oy); o.lineTo(p[p.length - 2] * this.s + this.ox, p[p.length - 1] * this.s + this.oy); } }
    o.stroke(); o.globalAlpha = 1;
    const order = sc.nodes.slice().sort((a, b) => scheme.codes[a.id] - scheme.codes[b.id]);
    for (const n of order) {
      o.fillStyle = scheme.paint(n.id).fill;
      o.fillRect(n.x * this.s + this.ox, n.y * this.s + this.oy, Math.max(1.6, n.w * this.s), Math.max(1.6, n.h * this.s));
    }
    this.img = oc;
    this.schedule();
  }

  private schedule(): void {
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => { this.queued = false; this.paint(); });
  }

  private paint(): void {
    if (!this.img) return;
    const ctx = this.ctx, th = this.viewer.theme;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(this.img, 0, 0);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const v = this.viewer.viewport;
    const x = v.x0 * this.s + this.ox, y = v.y0 * this.s + this.oy, w = Math.max(3, (v.x1 - v.x0) * this.s), h = Math.max(3, (v.y1 - v.y0) * this.s);
    ctx.fillStyle = th.sel; ctx.globalAlpha = 0.12; ctx.fillRect(x, y, w, h); ctx.globalAlpha = 1;
    ctx.strokeStyle = th.sel; ctx.lineWidth = 1.5; ctx.strokeRect(x, y, w, h);
  }

  private jump(e: PointerEvent): void {
    const r = this.canvas.getBoundingClientRect();
    this.viewer.centerOn((e.clientX - r.left - this.ox) / this.s, (e.clientY - r.top - this.oy) / this.s);
  }

  dispose(): void {
    for (const f of this.offs) f();
    this.offs = [];
    this.canvas.remove();
  }
}
