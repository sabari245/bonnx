import type { ValueType } from "../onnx/types";
import { ColorScheme, type ColorMode, type Legend } from "./colors";
import { fontSet, type FontSet, type RNode, type Scene, type SceneOptions } from "./scene";
import { makeGridPattern, paintEdgeLabels, paintEdges, paintNodes, paintOverlays, subRowAt, type PaintState } from "./paint";
import { readTheme, type Theme } from "./theme";
import type { ExportContext } from "./export";

/* ───────────────────────── public types ───────────────────────── */

export interface ViewState { scale: number; tx: number; ty: number }
export interface Insets { top: number; right: number; bottom: number; left: number }
export interface ViewerOptions {
  colorMode: ColorMode;
  /** edge labels; defaults to scene.options.showTypes, can change live without relayout */
  edgeLabels: SceneOptions["showTypes"] | null;
  showGrid: boolean;
  /** minimum zoom at which edge labels are drawn */
  labelMinScale: number;
}
export interface FrameStats { ms: number; nodes: number; edges: number; scale: number }

export interface ViewerEventMap {
  select: { node: number | null; scene: RNode | null };
  hover: { node: number | null; x: number; y: number };
  "open-subgraph": { node: number; graph: number; attr: string };
  contextmenu: { node: number | null; edge: number | null; x: number; y: number };
  viewchange: ViewState;
  "edge-select": { edge: number | null; name: string | null; type: ValueType | null; from: number; to: number };
  /** scene, colours or theme changed (minimap redraw) */
  scene: Scene | null;
  frame: FrameStats;
}

type Handler<T> = (e: T) => void;
class Emitter<M> {
  private map = new Map<keyof M, Set<Handler<never>>>();
  on<K extends keyof M>(k: K, fn: Handler<M[K]>): () => void {
    let s = this.map.get(k);
    if (!s) this.map.set(k, (s = new Set()));
    s.add(fn as Handler<never>);
    return () => this.off(k, fn);
  }
  off<K extends keyof M>(k: K, fn: Handler<M[K]>): void { this.map.get(k)?.delete(fn as Handler<never>); }
  emit<K extends keyof M>(k: K, e: M[K]): void { this.map.get(k)?.forEach((f) => (f as Handler<M[K]>)(e)); }
  clear(): void { this.map.clear(); }
}

/* ───────────────────────── spatial grid ───────────────────────── */

class Grid {
  private readonly size = 512;
  private readonly cols: number;
  private readonly rows: number;
  private cells: (number[] | undefined)[];
  constructor(private ox: number, private oy: number, w: number, h: number) {
    this.cols = Math.max(1, Math.ceil(w / this.size) + 1);
    this.rows = Math.max(1, Math.ceil(h / this.size) + 1);
    this.cells = new Array(this.cols * this.rows);
  }
  private cx(x: number): number { return Math.min(this.cols - 1, Math.max(0, Math.floor((x - this.ox) / this.size))); }
  private cy(y: number): number { return Math.min(this.rows - 1, Math.max(0, Math.floor((y - this.oy) / this.size))); }
  insert(id: number, x0: number, y0: number, x1: number, y1: number): void {
    const a = this.cx(x0), b = this.cx(x1), c = this.cy(y0), d = this.cy(y1);
    for (let j = c; j <= d; j++) for (let i = a; i <= b; i++) (this.cells[j * this.cols + i] ??= []).push(id);
  }
  query(x0: number, y0: number, x1: number, y1: number, cb: (id: number) => void): void {
    const a = this.cx(x0), b = this.cx(x1), c = this.cy(y0), d = this.cy(y1);
    for (let j = c; j <= d; j++) for (let i = a; i <= b; i++) { const cell = this.cells[j * this.cols + i]; if (cell) for (const id of cell) cb(id); }
  }
  at(x: number, y: number): readonly number[] { return this.cells[this.cy(y) * this.cols + this.cx(x)] ?? []; }
}

const MIN_SCALE = 0.004;
const MAX_SCALE = 4;
const reduceMotion = (): boolean => matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ───────────────────────── viewer ───────────────────────── */

export class GraphViewer {
  readonly canvas: HTMLCanvasElement;
  readonly stats: FrameStats = { ms: 0, nodes: 0, edges: 0, scale: 1 };
  private ctx: CanvasRenderingContext2D;
  private live: HTMLElement;
  private ev = new Emitter<ViewerEventMap>();
  private ro: ResizeObserver;
  private mo: MutationObserver;
  private disposed = false;

  private _scene: Scene | null = null;
  private _theme: Theme;
  private fonts: FontSet;
  private _scheme: ColorScheme | null = null;
  private opts: ViewerOptions = { colorMode: "op", edgeLabels: null, showGrid: true, labelMinScale: 0.7 };
  private gridPattern: CanvasPattern | null = null;

  private nodeGrid: Grid | null = null;
  private edgeGrid: Grid | null = null;
  private nStamp = new Uint32Array(0);
  private eStamp = new Uint32Array(0);
  private stamp = 0;
  private inE: number[][] = [];
  private outE: number[][] = [];
  private bounds = { x0: 0, y0: 0, x1: 1, y1: 1 };
  private refMap = new Map<number, number>(); // NodeView.id -> scene id

  private scale = 1; private tx = 0; private ty = 0;
  private vw = 0; private vh = 0; private dpr = 1;
  private ins: Insets = { top: 0, right: 0, bottom: 0, left: 0 };

  private sel = -1; private hov = -1; private selEdge = -1;
  private matches: ReadonlySet<number> | null = null;
  private matchList: number[] = [];
  private activeMatch = -1;

  private queued = false;
  private animId = 0;
  private ptr = new Map<number, { x: number; y: number }>();
  private down: { x: number; y: number; button: number } | null = null;
  private moved = false;
  private pinch: number | null = null;
  private cleanups: (() => void)[] = [];

  constructor(readonly container: HTMLElement, options: Partial<ViewerOptions> = {}) {
    Object.assign(this.opts, options);
    if (getComputedStyle(container).position === "static") container.style.position = "relative";
    this._theme = readTheme();
    this.fonts = fontSet({ sans: this._theme.fontSans, mono: this._theme.fontMono });

    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("role", "application");
    this.canvas.setAttribute("aria-label", "Model graph. Use arrow keys to move between nodes, Enter to open a subgraph.");
    this.canvas.tabIndex = 0;
    Object.assign(this.canvas.style, { position: "absolute", inset: "0", width: "100%", height: "100%", display: "block", touchAction: "none", outline: "none", cursor: "grab" });
    this.live = document.createElement("div");
    this.live.setAttribute("aria-live", "polite");
    this.live.setAttribute("role", "status");
    Object.assign(this.live.style, { position: "absolute", width: "1px", height: "1px", overflow: "hidden", clip: "rect(0 0 0 0)", whiteSpace: "nowrap" });
    container.append(this.canvas, this.live);
    this.ctx = this.canvas.getContext("2d", { alpha: false })!;
    this.gridPattern = makeGridPattern(this.ctx, this._theme);

    this.bindEvents();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    const retheme = (): void => this.refreshTheme();
    window.addEventListener("themechange", retheme);
    const mq = matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", retheme);
    this.mo = new MutationObserver(retheme);
    this.mo.observe(document.documentElement, { attributes: true, attributeFilter: ["class", "data-theme"] });
    this.cleanups.push(() => window.removeEventListener("themechange", retheme), () => mq.removeEventListener("change", retheme));
    this.resize();
  }

  /* ───────────── events api ───────────── */
  on<K extends keyof ViewerEventMap>(k: K, fn: Handler<ViewerEventMap[K]>): () => void { return this.ev.on(k, fn); }
  off<K extends keyof ViewerEventMap>(k: K, fn: Handler<ViewerEventMap[K]>): void { this.ev.off(k, fn); }

  /* ───────────── getters ───────────── */
  get scene(): Scene | null { return this._scene; }
  get theme(): Theme { return this._theme; }
  get scheme(): ColorScheme | null { return this._scheme; }
  get colorMode(): ColorMode { return this.opts.colorMode; }
  get selected(): number { return this.sel; }
  get selectedEdge(): number { return this.selEdge; }
  get edgeLabels(): SceneOptions["showTypes"] { return this.opts.edgeLabels ?? this._scene?.options.showTypes ?? "shape"; }
  get viewport(): { x0: number; y0: number; x1: number; y1: number } {
    return { x0: -this.tx / this.scale, y0: -this.ty / this.scale, x1: (this.vw - this.tx) / this.scale, y1: (this.vh - this.ty) / this.scale };
  }
  get worldBounds(): { x0: number; y0: number; x1: number; y1: number } { return this.bounds; }
  get size(): { width: number; height: number } { return { width: this.vw, height: this.vh }; }
  getLegend(): Legend | null { return this._scheme?.legend() ?? null; }
  /** scene node id for a NodeView.id of the current graph (op nodes only) */
  sceneIdOfNode(nodeViewId: number): number | null { return this.refMap.get(nodeViewId) ?? null; }
  getViewState(): ViewState { return { scale: this.scale, tx: this.tx, ty: this.ty }; }
  setViewState(v: ViewState): void { this.cancelAnim(); this.scale = v.scale; this.tx = v.tx; this.ty = v.ty; this.viewChanged(); }
  exportContext(): ExportContext | null {
    if (!this._scene || !this._scheme) return null;
    return { scene: this._scene, theme: this._theme, fonts: this.fonts, scheme: this._scheme, edgeLabels: this.edgeLabels, selection: this.sel };
  }

  /* ───────────── scene ───────────── */

  /**
   * Show a laid-out scene (call layoutScene() first). `keepView` preserves the current transform
   * (e.g. after toggling display options); otherwise the graph is fitted.
   */
  setScene(scene: Scene | null, o: { keepView?: boolean; selectRef?: number } = {}): void {
    this.cancelAnim();
    this._scene = scene;
    this.sel = this.hov = this.selEdge = -1;
    this.matches = null; this.matchList = []; this.activeMatch = -1;
    this.nodeGrid = this.edgeGrid = null;
    this.refMap.clear();
    if (!scene) { this._scheme = null; this.ev.emit("scene", null); this.draw(); return; }
    this.index(scene);
    this._scheme = new ColorScheme(scene, this._theme, this.opts.colorMode);
    if (!o.keepView) this.fit("auto", false);
    this.ev.emit("scene", scene);
    if (o.selectRef != null) { const id = this.refMap.get(o.selectRef); if (id != null) this.select(id, { focus: "ensure", animate: false }); }
    this.draw();
  }

  private index(scene: Scene): void {
    const N = scene.nodes, E = scene.edges;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of N) { x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x + n.w); y1 = Math.max(y1, n.y + n.h); }
    if (!N.length) { x0 = y0 = 0; x1 = y1 = 1; }
    this.bounds = { x0, y0, x1, y1 };
    const g1 = new Grid(x0, y0, x1 - x0, y1 - y0), g2 = new Grid(x0, y0, x1 - x0, y1 - y0);
    for (const n of N) { g1.insert(n.id, n.x, n.y, n.x + n.w, n.y + n.h); if (n.kind === "op" || n.kind === "fn") this.refMap.set(n.ref, n.id); }
    this.inE = N.map(() => []); this.outE = N.map(() => []);
    this.eBox = new Float32Array(E.length * 4);
    E.forEach((e, i) => {
      this.outE[e.s].push(i); this.inE[e.t].push(i);
      const p = e.p;
      if (!p.length) return;
      let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
      for (let k = 0; k < p.length; k += 2) { a = Math.min(a, p[k]); c = Math.max(c, p[k]); b = Math.min(b, p[k + 1]); d = Math.max(d, p[k + 1]); }
      this.eBox.set([a, b, c, d], i * 4);
      g2.insert(i, a, b, c, d);
      x0 = Math.min(x0, a); y0 = Math.min(y0, b); x1 = Math.max(x1, c); y1 = Math.max(y1, d);
    });
    this.bounds = { x0, y0, x1, y1 };
    this.nodeGrid = g1; this.edgeGrid = g2;
    this.nStamp = new Uint32Array(N.length); this.eStamp = new Uint32Array(E.length); this.stamp = 0;
  }
  private eBox = new Float32Array(0);

  /* ───────────── options ───────────── */

  setOptions(o: Partial<ViewerOptions>): void {
    const prevMode = this.opts.colorMode;
    Object.assign(this.opts, o);
    if (o.colorMode && o.colorMode !== prevMode) this.setColorMode(o.colorMode);
    else this.draw();
  }
  setColorMode(mode: ColorMode): void {
    this.opts.colorMode = mode;
    if (this._scene) { this._scheme = new ColorScheme(this._scene, this._theme, mode); this.ev.emit("scene", this._scene); }
    this.draw();
  }
  setEdgeLabels(m: SceneOptions["showTypes"]): void { this.opts.edgeLabels = m; this.draw(); }
  setInsets(i: Partial<Insets>): void { Object.assign(this.ins, i); }

  refreshTheme(): void {
    if (this.disposed) return;
    this._theme = readTheme();
    this.fonts = fontSet({ sans: this._theme.fontSans, mono: this._theme.fontMono });
    this.gridPattern = makeGridPattern(this.ctx, this._theme);
    if (this._scene) { this._scheme = new ColorScheme(this._scene, this._theme, this.opts.colorMode); this.ev.emit("scene", this._scene); }
    this.draw();
  }

  /* ───────────── view control ───────────── */

  private resize(): void {
    this.dpr = window.devicePixelRatio || 1;
    const r = this.container.getBoundingClientRect();
    this.vw = Math.max(1, Math.round(r.width)); this.vh = Math.max(1, Math.round(r.height));
    this.canvas.width = Math.round(this.vw * this.dpr); this.canvas.height = Math.round(this.vh * this.dpr);
    this.draw();
    this.ev.emit("viewchange", this.getViewState());
  }
  private free(): { l: number; t: number; w: number; h: number } {
    const l = this.ins.left, t = this.ins.top;
    return { l, t, w: Math.max(50, this.vw - l - this.ins.right), h: Math.max(50, this.vh - t - this.ins.bottom) };
  }
  private viewChanged(): void { this.draw(); this.ev.emit("viewchange", this.getViewState()); }
  private cancelAnim(): void { this.animId++; }

  zoomAt(f: number, cx: number, cy: number): void {
    this.cancelAnim();
    const ns = Math.min(MAX_SCALE, Math.max(MIN_SCALE, this.scale * f));
    f = ns / this.scale;
    this.tx = cx - (cx - this.tx) * f; this.ty = cy - (cy - this.ty) * f; this.scale = ns;
    this.viewChanged();
  }
  zoomIn(): void { const f = this.free(); this.zoomAt(1.4, f.l + f.w / 2, f.t + f.h / 2); }
  zoomOut(): void { const f = this.free(); this.zoomAt(1 / 1.4, f.l + f.w / 2, f.t + f.h / 2); }
  /** zoom to an absolute scale around the centre of the free area */
  setZoom(scale: number): void { const f = this.free(); this.zoomAt(scale / this.scale, f.l + f.w / 2, f.t + f.h / 2); }
  panBy(dx: number, dy: number): void { this.cancelAnim(); this.tx += dx; this.ty += dy; this.viewChanged(); }

  /** fit the graph: 'contain' whole graph, 'width' fit width aligned to top, 'auto' = contain unless it gets unreadably small */
  fit(mode: "auto" | "contain" | "width" = "auto", animate = true): void {
    if (!this._scene) return;
    const b = this.bounds, f = this.free(), m = 40;
    const w = b.x1 - b.x0, h = b.y1 - b.y0;
    const sw = (f.w - 2 * m) / w, sh = (f.h - 2 * m) / h;
    let s = Math.min(1, sw, sh), top = false;
    if (mode === "width" || (mode === "auto" && s < 0.18 && sw > s)) { s = Math.min(1, sw); top = true; }
    s = Math.max(MIN_SCALE, s);
    const tx = f.l + (f.w - w * s) / 2 - b.x0 * s;
    const ty = top ? f.t + m - b.y0 * s : f.t + (f.h - h * s) / 2 - b.y0 * s;
    this.animateTo(s, tx, ty, animate ? 320 : 0);
  }
  /** fit the selected node together with its direct neighbours */
  fitSelection(animate = true): void {
    if (!this._scene || this.sel < 0) return;
    const ids = new Set<number>([this.sel]);
    for (const i of this.inE[this.sel]) ids.add(this._scene.edges[i].s);
    for (const i of this.outE[this.sel]) ids.add(this._scene.edges[i].t);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const id of ids) { const n = this._scene.nodes[id]; x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x + n.w); y1 = Math.max(y1, n.y + n.h); }
    this.fitRect(x0, y0, x1, y1, 60, animate);
  }
  fitRect(x0: number, y0: number, x1: number, y1: number, margin = 60, animate = true): void {
    const f = this.free();
    const s = Math.min(1.2, Math.max(MIN_SCALE, Math.min((f.w - 2 * margin) / (x1 - x0), (f.h - 2 * margin) / (y1 - y0))));
    this.animateTo(s, f.l + f.w / 2 - ((x0 + x1) / 2) * s, f.t + f.h / 2 - ((y0 + y1) / 2) * s, animate ? 320 : 0);
  }
  centerOn(wx: number, wy: number, animate = false): void {
    const f = this.free();
    this.animateTo(this.scale, f.l + f.w / 2 - wx * this.scale, f.t + f.h / 2 - wy * this.scale, animate ? 260 : 0);
  }
  focusNode(id: number, o: { zoom?: boolean | number; animate?: boolean } = {}): void {
    const n = this._scene?.nodes[id];
    if (!n) return;
    const s = typeof o.zoom === "number" ? o.zoom : o.zoom ? Math.max(this.scale, 0.9) : this.scale;
    const f = this.free();
    this.animateTo(s, f.l + f.w / 2 - (n.x + n.w / 2) * s, f.t + f.h / 2 - (n.y + n.h / 2) * s, o.animate === false ? 0 : 320);
  }
  /** pan just enough that a node is inside the free area */
  ensureVisible(id: number, animate = true): void {
    const n = this._scene?.nodes[id];
    if (!n) return;
    const f = this.free(), m = 48, s = this.scale;
    const x0 = n.x * s + this.tx, x1 = (n.x + n.w) * s + this.tx, y0 = n.y * s + this.ty, y1 = (n.y + n.h) * s + this.ty;
    let dx = 0, dy = 0;
    if (x1 - x0 + 2 * m > f.w) dx = f.l + f.w / 2 - (x0 + x1) / 2;
    else if (x0 < f.l + m) dx = f.l + m - x0;
    else if (x1 > f.l + f.w - m) dx = f.l + f.w - m - x1;
    if (y1 - y0 + 2 * m > f.h) dy = f.t + f.h / 2 - (y0 + y1) / 2;
    else if (y0 < f.t + m) dy = f.t + m - y0;
    else if (y1 > f.t + f.h - m) dy = f.t + f.h - m - y1;
    if (dx || dy) this.animateTo(s, this.tx + dx, this.ty + dy, animate ? 200 : 0);
  }

  private animateTo(s1: number, tx1: number, ty1: number, ms: number): void {
    this.cancelAnim();
    if (!ms || reduceMotion()) { this.scale = s1; this.tx = tx1; this.ty = ty1; this.viewChanged(); return; }
    const id = this.animId, s0 = this.scale, t0 = performance.now();
    const c0x = (this.vw / 2 - this.tx) / s0, c0y = (this.vh / 2 - this.ty) / s0, c1x = (this.vw / 2 - tx1) / s1, c1y = (this.vh / 2 - ty1) / s1;
    const step = (now: number): void => {
      if (id !== this.animId || this.disposed) return;
      let k = Math.min(1, (now - t0) / ms);
      k = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
      // interpolate scale geometrically for a natural zoom
      this.scale = s0 * Math.pow(s1 / s0, k);
      const cx = c0x + (c1x - c0x) * k, cy = c0y + (c1y - c0y) * k;
      this.tx = this.vw / 2 - cx * this.scale; this.ty = this.vh / 2 - cy * this.scale;
      if (k >= 1) { this.scale = s1; this.tx = tx1; this.ty = ty1; }
      this.render();
      this.ev.emit("viewchange", this.getViewState());
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ───────────── selection / search ───────────── */

  select(id: number | null, o: { focus?: false | "ensure" | "center" | "zoom"; animate?: boolean; silent?: boolean } = {}): void {
    const sc = this._scene;
    const nid = id != null && sc?.nodes[id] ? id : -1;
    const changed = nid !== this.sel || this.selEdge >= 0;
    this.sel = nid; this.selEdge = -1;
    if (nid >= 0) {
      const n = sc!.nodes[nid];
      this.announce(`${n.kind === "op" || n.kind === "fn" ? n.op : n.kind} ${n.name}`);
      if (o.focus === "ensure") this.ensureVisible(nid, o.animate !== false);
      else if (o.focus === "center") this.focusNode(nid, { animate: o.animate });
      else if (o.focus === "zoom") this.focusNode(nid, { zoom: true, animate: o.animate });
    }
    this.draw();
    if (changed && !o.silent) this.ev.emit("select", { node: nid >= 0 ? nid : null, scene: nid >= 0 ? sc!.nodes[nid] : null });
  }
  selectEdge(id: number | null): void {
    const e = id != null ? this._scene?.edges[id] : null;
    this.selEdge = e ? id! : -1;
    if (e) { this.sel = -1; this.announce(`value ${e.name}`); this.ev.emit("select", { node: null, scene: null }); }
    this.draw();
    this.ev.emit("edge-select", e ? { edge: id!, name: e.name, type: e.type, from: e.s, to: e.t } : { edge: null, name: null, type: null, from: -1, to: -1 });
  }

  /** highlight search results (scene node ids); optionally move to the active one */
  highlightMatches(ids: readonly number[] | null, activeId?: number, focus = true): void {
    this.matchList = ids ? [...ids].sort((a, b) => a - b) : [];
    this.matches = ids && ids.length ? new Set(ids) : null;
    this.activeMatch = activeId ?? (this.matchList[0] ?? -1);
    if (focus && this.activeMatch >= 0) this.ensureVisible(this.activeMatch);
    this.draw();
  }
  /** step through highlighted matches; returns the new active node id (or -1) */
  stepMatch(delta: 1 | -1, focus = true): number {
    const l = this.matchList;
    if (!l.length) return -1;
    const i = l.indexOf(this.activeMatch);
    this.activeMatch = l[(i + delta + l.length) % l.length];
    if (focus) this.focusNode(this.activeMatch, { animate: true });
    this.draw();
    return this.activeMatch;
  }
  get matchCount(): number { return this.matchList.length; }
  get activeMatchIndex(): number { return this.matchList.indexOf(this.activeMatch); }
  clearMatches(): void { this.highlightMatches(null, undefined, false); }

  private announce(t: string): void { this.live.textContent = t; }

  /* ───────────── hit testing ───────────── */

  /** scene node under a client point, or null */
  nodeAt(clientX: number, clientY: number): RNode | null {
    const r = this.canvas.getBoundingClientRect();
    const id = this.hitNode(clientX - r.left, clientY - r.top);
    return id >= 0 ? this._scene!.nodes[id] : null;
  }
  private hitNode(sx: number, sy: number): number {
    const sc = this._scene, g = this.nodeGrid;
    if (!sc || !g) return -1;
    const wx = (sx - this.tx) / this.scale, wy = (sy - this.ty) / this.scale, tol = Math.max(0, 3 / this.scale);
    const c = g.at(wx, wy);
    for (let i = c.length - 1; i >= 0; i--) {
      const n = sc.nodes[c[i]];
      if (wx >= n.x - tol && wx <= n.x + n.w + tol && wy >= n.y - tol && wy <= n.y + n.h + tol) return n.id;
    }
    return -1;
  }
  private hitEdge(sx: number, sy: number): number {
    const sc = this._scene, g = this.edgeGrid;
    if (!sc || !g) return -1;
    const wx = (sx - this.tx) / this.scale, wy = (sy - this.ty) / this.scale, tol = 6 / this.scale;
    let best = -1, bd = tol * tol;
    for (const i of g.at(wx, wy)) {
      const p = sc.edges[i].p;
      if (p.length < 8) continue;
      let px = p[0], py = p[1];
      for (let k = 2; k + 5 < p.length; k += 6) {
        for (let s = 1; s <= 8; s++) {
          const t = s / 8, u = 1 - t;
          const x = u * u * u * p[k - 2] + 3 * u * u * t * p[k] + 3 * u * t * t * p[k + 2] + t * t * t * p[k + 4];
          const y = u * u * u * p[k - 1] + 3 * u * u * t * p[k + 1] + 3 * u * t * t * p[k + 3] + t * t * t * p[k + 5];
          const d = segDist2(wx, wy, px, py, x, y);
          if (d < bd) { bd = d; best = i; }
          px = x; py = y;
        }
      }
    }
    return best;
  }

  /* ───────────── keyboard navigation ───────────── */

  /** Move selection along the graph. TB: up/down = producer/consumer, left/right = sibling; LR is rotated. */
  navigate(key: "up" | "down" | "left" | "right"): void {
    const sc = this._scene;
    if (!sc || !sc.nodes.length) return;
    if (this.sel < 0) {
      const first = sc.nodes.find((n) => n.kind === "input") ?? sc.nodes[0];
      this.select(first.id, { focus: "ensure" });
      return;
    }
    const cur = sc.nodes[this.sel], lr = sc.options.direction === "LR";
    const flow: "back" | "fwd" | "prev" | "next" =
      !lr ? (key === "up" ? "back" : key === "down" ? "fwd" : key === "left" ? "prev" : "next")
          : (key === "left" ? "back" : key === "right" ? "fwd" : key === "up" ? "prev" : "next");
    const cx = cur.x + cur.w / 2, cy = cur.y + cur.h / 2;
    let target = -1;
    if (flow === "back" || flow === "fwd") {
      const edges = flow === "back" ? this.inE[cur.id] : this.outE[cur.id];
      let bd = Infinity;
      for (const i of edges) {
        const e = sc.edges[i], n = sc.nodes[flow === "back" ? e.s : e.t];
        const d = lr ? Math.abs(n.y + n.h / 2 - cy) : Math.abs(n.x + n.w / 2 - cx);
        if (d < bd) { bd = d; target = n.id; }
      }
    } else {
      const sign = flow === "next" ? 1 : -1;
      let bd = Infinity;
      for (const n of sc.nodes) {
        if (n.id === cur.id) continue;
        const sameRank = lr ? Math.abs(n.x + n.w / 2 - cx) < Math.max(n.w, cur.w) / 2 + 1 : Math.abs(n.y + n.h / 2 - cy) < Math.max(n.h, cur.h) / 2 + 1;
        if (!sameRank) continue;
        const d = lr ? (n.y + n.h / 2 - cy) * sign : (n.x + n.w / 2 - cx) * sign;
        if (d > 0 && d < bd) { bd = d; target = n.id; }
      }
    }
    if (target >= 0) this.select(target, { focus: "ensure" });
  }

  /** open the (first) subgraph / function of the selected node */
  openSelected(): boolean {
    const n = this.sel >= 0 ? this._scene?.nodes[this.sel] : null;
    if (!n?.expandable) return false;
    const l = n.lines.find((x) => x.kind === "sub" && x.graph != null);
    if (!l) return false;
    this.ev.emit("open-subgraph", { node: n.id, graph: l.graph!, attr: l.attr ?? "" });
    return true;
  }

  /* ───────────── input ───────────── */

  private bindEvents(): void {
    const c = this.canvas;
    const add = <K extends keyof HTMLElementEventMap>(t: K, fn: (e: HTMLElementEventMap[K]) => void, o?: AddEventListenerOptions): void => {
      c.addEventListener(t, fn as EventListener, o);
      this.cleanups.push(() => c.removeEventListener(t, fn as EventListener, o));
    };
    const rel = (e: MouseEvent): { x: number; y: number } => { const r = c.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };

    add("pointerdown", (e) => {
      if (e.button === 2) return;
      c.focus({ preventScroll: true });
      c.setPointerCapture(e.pointerId);
      const p = rel(e);
      this.ptr.set(e.pointerId, p);
      this.cancelAnim();
      if (this.ptr.size === 1) { this.down = { ...p, button: e.button }; this.moved = false; c.style.cursor = "grabbing"; }
      if (this.ptr.size === 2) { const [a, b] = [...this.ptr.values()]; this.pinch = Math.hypot(a.x - b.x, a.y - b.y); this.moved = true; }
      this.ev.emit("hover", { node: null, x: 0, y: 0 });
    });
    add("pointermove", (e) => {
      const p = rel(e), prev = this.ptr.get(e.pointerId);
      if (!prev) {
        const id = this.hitNode(p.x, p.y);
        if (id !== this.hov) { this.hov = id; c.style.cursor = id >= 0 ? "pointer" : "grab"; this.draw(); }
        this.ev.emit("hover", { node: id >= 0 ? id : null, x: e.clientX, y: e.clientY });
        return;
      }
      if (this.ptr.size === 2 && this.pinch != null) {
        this.ptr.set(e.pointerId, p);
        const [a, b] = [...this.ptr.values()], d = Math.hypot(a.x - b.x, a.y - b.y);
        this.zoomAt(d / this.pinch, (a.x + b.x) / 2, (a.y + b.y) / 2);
        this.pinch = d;
        return;
      }
      this.ptr.set(e.pointerId, p);
      if (this.down && Math.abs(p.x - this.down.x) + Math.abs(p.y - this.down.y) > 4) this.moved = true;
      if (this.moved) { this.tx += p.x - prev.x; this.ty += p.y - prev.y; this.viewChanged(); }
    });
    const up = (e: PointerEvent): void => {
      const had = this.ptr.has(e.pointerId);
      this.ptr.delete(e.pointerId);
      if (this.ptr.size < 2) this.pinch = null;
      if (this.ptr.size === 0) {
        c.style.cursor = this.hov >= 0 ? "pointer" : "grab";
        if (had && !this.moved && e.type === "pointerup" && this.down?.button === 0) this.click(rel(e));
        this.down = null;
      }
    };
    add("pointerup", up);
    add("pointercancel", up);
    add("pointerleave", () => { if (this.hov >= 0) { this.hov = -1; this.draw(); } this.ev.emit("hover", { node: null, x: 0, y: 0 }); });
    add("wheel", (e) => {
      e.preventDefault();
      const p = rel(e);
      if (e.ctrlKey || e.metaKey || e.deltaMode === 1 || (Math.abs(e.deltaY) >= 50 && e.deltaX === 0 && Number.isInteger(e.deltaY))) this.zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0022)), p.x, p.y);
      else this.panBy(-e.deltaX, -e.deltaY);
      this.ev.emit("hover", { node: null, x: 0, y: 0 });
    }, { passive: false });
    add("dblclick", (e) => {
      const p = rel(e), id = this.hitNode(p.x, p.y);
      if (id < 0) { if (this.hitEdge(p.x, p.y) < 0) this.fit("auto"); return; }
      this.select(id);
      this.openSelected();
    });
    add("contextmenu", (e) => {
      e.preventDefault();
      const p = rel(e), id = this.hitNode(p.x, p.y), ed = id < 0 ? this.hitEdge(p.x, p.y) : -1;
      if (id >= 0) this.select(id); else if (ed >= 0) this.selectEdge(ed);
      this.ev.emit("contextmenu", { node: id >= 0 ? id : null, edge: ed >= 0 ? ed : null, x: e.clientX, y: e.clientY });
    });
    add("keydown", (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const k = ({ ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right" } as const)[e.key as "ArrowUp"];
      if (k) { e.preventDefault(); this.navigate(k); }
      else if (e.key === "Enter") { if (this.openSelected()) e.preventDefault(); }
      else if (e.key === "Escape") { this.select(null); this.selectEdge(null); }
    });
  }

  private click(p: { x: number; y: number }): void {
    const id = this.hitNode(p.x, p.y);
    if (id >= 0) {
      const n = this._scene!.nodes[id];
      const sub = subRowAt(n, (p.x - this.tx) / this.scale, (p.y - this.ty) / this.scale);
      this.select(id);
      if (sub) this.ev.emit("open-subgraph", { node: id, graph: sub.graph, attr: sub.attr });
      return;
    }
    const ed = this.hitEdge(p.x, p.y);
    if (ed >= 0) this.selectEdge(ed);
    else { if (this.selEdge >= 0) this.selectEdge(null); this.select(null); }
  }

  /* ───────────── rendering ───────────── */

  draw(): void {
    if (this.queued || this.disposed) return;
    this.queued = true;
    requestAnimationFrame(() => { this.queued = false; this.render(); });
  }

  private render(): void {
    if (this.disposed) return;
    const t0 = performance.now();
    const ctx = this.ctx, th = this._theme, sc = this._scene, scheme = this._scheme;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = th.bg;
    ctx.fillRect(0, 0, this.vw, this.vh);
    this.ctx.setTransform(this.dpr * this.scale, 0, 0, this.dpr * this.scale, this.dpr * this.tx, this.dpr * this.ty);
    const v = this.viewport, s = this.scale;
    if (this.opts.showGrid && this.gridPattern && s > 0.4) {
      ctx.fillStyle = this.gridPattern;
      ctx.fillRect(v.x0, v.y0, v.x1 - v.x0, v.y1 - v.y0);
    }
    if (!sc || !scheme || !this.nodeGrid || !this.edgeGrid) return;

    const pad = 40 / s, stamp = ++this.stamp;
    const vn: RNode[] = [], ve: number[] = [];
    this.nodeGrid.query(v.x0 - pad, v.y0 - pad, v.x1 + pad, v.y1 + pad, (id) => {
      if (this.nStamp[id] === stamp) return;
      this.nStamp[id] = stamp;
      const n = sc.nodes[id];
      if (n.x + n.w >= v.x0 && n.x <= v.x1 && n.y + n.h >= v.y0 && n.y <= v.y1) vn.push(n);
    });
    this.edgeGrid.query(v.x0 - pad, v.y0 - pad, v.x1 + pad, v.y1 + pad, (i) => {
      if (this.eStamp[i] === stamp) return;
      this.eStamp[i] = stamp;
      const o = i * 4;
      if (this.eBox[o + 2] >= v.x0 - pad && this.eBox[o] <= v.x1 + pad && this.eBox[o + 3] >= v.y0 - pad && this.eBox[o + 1] <= v.y1 + pad) ve.push(i);
    });
    vn.sort((a, b) => a.id - b.id);

    let hi: Set<number> | null = null;
    for (const id of [this.sel, this.hov]) {
      if (id < 0) continue;
      hi ??= new Set();
      for (const i of this.inE[id]) hi.add(i);
      for (const i of this.outE[id]) hi.add(i);
    }
    const st: PaintState = {
      theme: th, fonts: this.fonts, scheme, scale: s, dir: sc.options.direction, edgeLabels: this.edgeLabels,
      sel: this.sel, hov: this.hov, selEdge: this.selEdge, matches: this.matches, activeMatch: this.activeMatch, hiEdges: hi, lod: "auto",
    };
    paintEdges(ctx, sc, ve, st);
    paintNodes(ctx, vn, st);
    if (s >= this.opts.labelMinScale) paintEdgeLabels(ctx, sc, ve, st);
    paintOverlays(ctx, sc, vn, st);

    const ms = performance.now() - t0;
    Object.assign(this.stats, { ms, nodes: vn.length, edges: ve.length, scale: s });
    this.ev.emit("frame", this.stats);
  }

  dispose(): void {
    this.disposed = true;
    this.cancelAnim();
    this.ro.disconnect(); this.mo.disconnect();
    for (const f of this.cleanups) f();
    this.cleanups = [];
    this.ev.clear();
    this.canvas.remove(); this.live.remove();
    this._scene = null; this._scheme = null;
  }
}

function segDist2(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay, l = dx * dx + dy * dy;
  const t = l ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l)) : 0;
  const x = ax + t * dx - px, y = ay + t * dy - py;
  return x * x + y * y;
}
