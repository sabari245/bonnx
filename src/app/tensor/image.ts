import { h } from "@/lib/dom";
import { ChevronLeft, ChevronRight, Download, Maximize, Minus, Plus } from "lucide";
import { alert, button, input, label, select, slider, toggleGroup } from "@/ui";
import { COLORMAPS, INF_COLOR, NAN_COLOR, NORM_MODES, colormapGradient, colormapLUT, type ColormapName, type NormMode } from "@/lib/colormap";
import type { TensorInfo } from "@/onnx/types";
import { debounce, downloadBlob, errMsg, fetchLinear, fmtNum, isComplexDType, parseNum, safeFileName, shapeStr, type TensorCtx } from "./util";

const MAX_ELEMS = 4_000_000;
const MAX_TILES = 2048;
type Layout = "none" | "first" | "last";

interface Geo { H: number; W: number; C: number; lead: number[]; G: number; gE: number }

export function imageTab(ctx: TensorCtx): HTMLElement {
  const t: TensorInfo = ctx.info;
  const root = h("div", { class: "flex h-full min-h-0 flex-col gap-3 p-1" });
  if (!t.available) return h("div", { class: "p-1" }, alert("No data to show", "The payload of this tensor is not available."));
  if (t.dtype === "string" || isComplexDType(t.dtype)) return h("div", { class: "p-1" }, alert("Not drawable", `${t.dtype} tensors cannot be shown as an image.`));
  if (t.n === 0) return h("div", { class: "p-1" }, alert("Empty tensor", `Shape ${shapeStr(t.dims)} contains no elements.`));

  const d = t.dims, r = d.length;

  /* ── state ── */
  let layout: Layout = r >= 3 ? (d[r - 3]! === 3 ? "first" : r === 3 && d[0]! > 4 && d[2]! <= 4 ? "last" : r === 3 && d[0]! <= 4 ? "first" : "none") : "none";
  let rgb = false;
  let cmap: ColormapName = "viridis";
  let norm: NormMode = "minmax";
  let customLo: number | null = null, customHi: number | null = null;
  let colsOverride = 0;
  let wrapSquare = r === 1 ? d[0]! > 512 : false;
  let g0 = 0;
  const crop = { y0: 0, x0: 0, h: 0, w: 0 };
  let token = 0;

  function geometry(): Geo {
    if (r === 0) return { H: 1, W: 1, C: 1, lead: [], G: 1, gE: 1 };
    if (r === 1) { const n = d[0]!; if (wrapSquare || n > 16384) { const W = Math.ceil(Math.sqrt(n)); return { H: Math.ceil(n / W), W, C: 1, lead: [], G: 1, gE: n }; } return { H: 1, W: n, C: 1, lead: [], G: 1, gE: n }; }
    let H: number, W: number, C: number, lead: number[];
    if (layout === "first") { C = d[r - 3]!; H = d[r - 2]!; W = d[r - 1]!; lead = d.slice(0, r - 3); }
    else if (layout === "last") { H = d[r - 3]!; W = d[r - 2]!; C = d[r - 1]!; lead = d.slice(0, r - 3); }
    else { C = 1; H = d[r - 2]!; W = d[r - 1]!; lead = d.slice(0, r - 2); }
    const G = lead.reduce((a, b) => a * b, 1);
    return { H, W, C, lead, G, gE: C * H * W };
  }
  const canRgb = () => r >= 3 && layout !== "none" && geometry().C === 3;
  const oversize = () => geometry().gE > MAX_ELEMS;

  /* ── controls ── */
  const layoutOpts = r >= 3
    ? [{ value: "none", label: `Tiles of ${d[r - 2]}×${d[r - 1]} (no channel axis)` }, { value: "first", label: `Channels first [… ${d[r - 3]}, ${d[r - 2]}, ${d[r - 1]}]` }, { value: "last", label: `Channels last [… ${d[r - 3]}, ${d[r - 2]}, ${d[r - 1]}]` }]
    : [];
  const layoutSel = select(layoutOpts, { value: layout, size: "sm", ariaLabel: "Channel layout", class: "w-64", onChange: (v) => { layout = v as Layout; g0 = 0; rgb = canRgb(); syncControls(); load(); } });
  const modeTg = toggleGroup([{ value: "tiles", label: "Tiles" }, { value: "rgb", label: "RGB" }], { variant: "outline", size: "sm", ariaLabel: "Channel mode", value: "tiles", onChange: (v) => { rgb = v[0] === "rgb"; g0 = Math.floor(g0); syncControls(); load(); } });
  const cmapSel = select(COLORMAPS.map((c) => ({ value: c.value, label: c.label })), { value: cmap, size: "sm", ariaLabel: "Colormap", class: "w-48", onChange: (v) => { cmap = v as ColormapName; legend(); paint(); } });
  const normSel = select(NORM_MODES.map((n) => ({ value: n.value, label: n.label })), { value: norm, size: "sm", ariaLabel: "Normalization", class: "w-36", onChange: (v) => { norm = v as NormMode; if (norm === "symmetric" && cmap !== "coolwarm") { cmap = "coolwarm"; cmapSel.set("coolwarm", true); legend(); } syncControls(); paint(); } });
  const loIn = input({ type: "text", ariaLabel: "Range minimum", class: "h-8 w-24 font-mono text-xs", onChange: (v) => { customLo = parseNum(v); paint(); } });
  const hiIn = input({ type: "text", ariaLabel: "Range maximum", class: "h-8 w-24 font-mono text-xs", onChange: (v) => { customHi = parseNum(v); paint(); } });
  const colsIn = input({ type: "number", value: "0", ariaLabel: "Tiles per row (0 = auto)", class: "h-8 w-16 font-mono text-xs", onChange: (v) => { colsOverride = Math.max(0, Math.floor(Number(v)) || 0); paint(); } });
  const squareTg = toggleGroup([{ value: "row", label: "Row" }, { value: "square", label: "Square" }], { variant: "outline", size: "sm", ariaLabel: "1-D arrangement", value: wrapSquare ? "square" : "row", onChange: (v) => { wrapSquare = v[0] === "square"; load(); } });
  const ctl = (name: string, ...n: Node[]) => h("div", { class: "flex items-center gap-2" }, label(name, { class: "text-muted-foreground text-xs" }), ...n);

  const row1 = h("div", { class: "flex flex-wrap items-center gap-x-4 gap-y-2" });
  const layoutCtl = ctl("Layout", layoutSel.el), modeCtl = ctl("Mode", modeTg.el), cmapCtl = ctl("Color", cmapSel.el), normCtl = ctl("Range", normSel.el);
  const customCtl = h("div", { class: "flex items-center gap-1.5" }, loIn, h("span", { class: "text-muted-foreground" }, "→"), hiIn);
  const colsCtl = ctl("Tiles per row", colsIn), squareCtl = ctl("Arrange", squareTg.el);
  row1.append(layoutCtl, modeCtl, cmapCtl, normCtl, customCtl, colsCtl, squareCtl);

  const row2 = h("div", { class: "flex flex-wrap items-center gap-x-4 gap-y-2 text-xs" });
  const pageInfo = h("span", { class: "text-muted-foreground tabular-nums" });
  const cropBox = h("div", { class: "flex flex-wrap items-center gap-2" });

  /* viewport */
  const view = h("div", { class: "bg-muted/40 relative min-h-[240px] flex-1 cursor-grab touch-none overflow-hidden rounded-lg border active:cursor-grabbing", tabindex: 0, "aria-label": "Tensor image. Scroll to zoom, drag to pan." });
  const cv = h("canvas", { class: "absolute top-0 left-0 origin-top-left", style: "image-rendering: pixelated" }) as HTMLCanvasElement;
  const marker = h("div", { class: "border-primary pointer-events-none absolute hidden border", style: "box-sizing:border-box" });
  const overlay = h("div", { class: "text-muted-foreground absolute inset-0 flex items-center justify-center text-sm" }, "Loading…");
  view.append(cv, marker, overlay);

  const zoomLbl = h("span", { class: "text-muted-foreground w-14 text-right font-mono text-xs tabular-nums" }, "100%");
  const zSlider = slider({ min: -40, max: 80, step: 1, value: 0, ariaLabel: "Zoom", class: "w-40", onInput: (v) => setZoom(2 ** (v / 10), view.clientWidth / 2, view.clientHeight / 2, true) });
  const zoomBar = h("div", { class: "flex flex-wrap items-center gap-2" },
    button(null, { variant: "outline", size: "icon-sm", icon: Minus, ariaLabel: "Zoom out", onClick: () => setZoom(zoom / 1.5, view.clientWidth / 2, view.clientHeight / 2) }),
    zSlider.el,
    button(null, { variant: "outline", size: "icon-sm", icon: Plus, ariaLabel: "Zoom in", onClick: () => setZoom(zoom * 1.5, view.clientWidth / 2, view.clientHeight / 2) }),
    zoomLbl,
    button("1:1", { variant: "outline", size: "sm", onClick: () => { setZoom(1, view.clientWidth / 2, view.clientHeight / 2); } }),
    button("Fit", { variant: "outline", size: "sm", icon: Maximize, onClick: () => fit() }),
    h("span", { class: "flex-1" }),
    h("div", { class: "flex items-center gap-2" }, h("span", { class: "text-muted-foreground font-mono text-xs tabular-nums", "data-role": "lo" }), h("div", { "data-role": "ramp", class: "h-3 w-32 rounded-sm border" }), h("span", { class: "text-muted-foreground font-mono text-xs tabular-nums", "data-role": "hi" })),
    button("Save PNG", { variant: "outline", size: "sm", icon: Download, onClick: () => void savePng() }));
  const rampEl = zoomBar.querySelector<HTMLElement>("[data-role=ramp]")!;
  const loLbl = zoomBar.querySelector<HTMLElement>("[data-role=lo]")!, hiLbl = zoomBar.querySelector<HTMLElement>("[data-role=hi]")!;

  const status = h("div", { class: "text-muted-foreground min-h-5 truncate font-mono text-xs" });
  const info = h("div", { class: "text-muted-foreground truncate text-xs tabular-nums" });
  root.append(row1, row2, zoomBar, view, h("div", { class: "flex items-center justify-between gap-3" }, status, info));

  /* ── data ── */
  interface Loaded { vals: Float64Array; geo: Geo; groups: number; gStart: number; Hc: number; Wc: number; y0: number; x0: number; n1d: number }
  let data: Loaded | null = null;
  let img: { w: number; h: number; tw: number; th: number; cols: number; gap: number; tiles: { g: number; c: number }[] } | null = null;
  let range = { lo: 0, hi: 1 };

  function unravel(i: number, dims: number[]): number[] {
    const out = new Array<number>(dims.length);
    for (let k = dims.length - 1; k >= 0; k--) { out[k] = i % dims[k]!; i = Math.floor(i / dims[k]!); }
    return out;
  }

  const perTile = () => (rgb && canRgb() ? 1 : geometry().C);
  const groupsPerPage = () => { const g = geometry(); return Math.max(1, Math.min(Math.floor(MAX_ELEMS / g.gE) || 1, Math.floor(MAX_TILES / perTile()))); };

  async function load(): Promise<void> {
    const my = ++token;
    overlay.textContent = "Loading…"; overlay.hidden = false;
    const geo = geometry();
    syncControls();
    try {
      let loaded: Loaded;
      if (r === 0 || r === 1) {
        const { values } = await fetchLinear(ctx.api, t, 0, t.n);
        loaded = { vals: values as Float64Array, geo, groups: 1, gStart: 0, Hc: geo.H, Wc: geo.W, y0: 0, x0: 0, n1d: t.n };
      } else if (oversize()) {
        if (crop.h === 0) { const side = Math.min(1024, Math.floor(Math.sqrt(MAX_ELEMS / geo.C))); crop.h = Math.min(geo.H, side); crop.w = Math.min(geo.W, side); }
        crop.h = Math.max(1, Math.min(crop.h, geo.H)); crop.w = Math.max(1, Math.min(crop.w, geo.W));
        while (crop.h * crop.w * geo.C > MAX_ELEMS) { if (crop.h >= crop.w) crop.h = Math.floor(crop.h / 2); else crop.w = Math.floor(crop.w / 2); }
        crop.y0 = Math.max(0, Math.min(crop.y0, geo.H - crop.h)); crop.x0 = Math.max(0, Math.min(crop.x0, geo.W - crop.w));
        const gi = Math.min(g0, geo.G - 1);
        const lead = unravel(gi, geo.lead);
        const ones = (n: number) => new Array<number>(n).fill(1);
        const offset = [...lead], size = ones(lead.length);
        if (layout === "first") { offset.push(0, crop.y0, crop.x0); size.push(geo.C, crop.h, crop.w); }
        else if (layout === "last") { offset.push(crop.y0, crop.x0, 0); size.push(crop.h, crop.w, geo.C); }
        else { offset.push(crop.y0, crop.x0); size.push(crop.h, crop.w); }
        const s = await ctx.api.tensorSlice({ id: t.id, offset, size, maxElems: MAX_ELEMS + 1 });
        loaded = { vals: s.values as Float64Array, geo, groups: 1, gStart: gi, Hc: crop.h, Wc: crop.w, y0: crop.y0, x0: crop.x0, n1d: 0 };
      } else {
        const gpp = groupsPerPage();
        g0 = Math.max(0, Math.min(g0, Math.max(0, geo.G - 1)));
        const g1 = Math.min(geo.G, g0 + gpp);
        const { values } = await fetchLinear(ctx.api, t, g0 * geo.gE, g1 * geo.gE);
        loaded = { vals: values as Float64Array, geo, groups: g1 - g0, gStart: g0, Hc: geo.H, Wc: geo.W, y0: 0, x0: 0, n1d: 0 };
      }
      if (my !== token) return;
      data = loaded;
      overlay.hidden = true;
      await paint(true);
    } catch (e) {
      if (my !== token) return;
      overlay.replaceChildren(alert("Could not load tensor data", errMsg(e), { variant: "destructive" }));
    }
  }

  function val(g: number, c: number, y: number, x: number): number | undefined {
    const D = data!;
    if (r <= 1) { const i = y * D.Wc + x; return i < D.n1d ? D.vals[i] : undefined; }
    const gE = D.geo.C * D.Hc * D.Wc;
    const base = g * gE;
    return layout === "last" ? D.vals[base + (y * D.Wc + x) * D.geo.C + c] : D.vals[base + c * D.Hc * D.Wc + y * D.Wc + x];
  }

  /* ── painting ── */
  async function computeRange(): Promise<void> {
    const s = t.summary;
    const finite = (v: number | null | undefined): v is number => v != null && Number.isFinite(v);
    let lo = 0, hi = 1;
    if (norm === "minmax" || norm === "symmetric" || norm === "percentile") {
      let st: Awaited<ReturnType<TensorCtx["stats"]>> | null = null;
      if (norm === "percentile" || !finite(s?.min) || !finite(s?.max)) { try { st = await ctx.stats(); } catch { st = null; } }
      const mn = finite(s?.min) ? s!.min! : st?.min, mx = finite(s?.max) ? s!.max! : st?.max;
      if (norm === "percentile" && finite(st?.p1) && finite(st?.p99)) { lo = st!.p1!; hi = st!.p99!; }
      else if (finite(mn) && finite(mx)) { lo = mn; hi = mx; }
      if (norm === "symmetric") { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
    } else if (norm === "custom" || norm === "tile") {
      // data range of the loaded window as the default for custom
      let mn = Infinity, mx = -Infinity;
      if (data) for (let i = 0; i < data.vals.length; i++) { const v = data.vals[i]!; if (Number.isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; } }
      lo = customLo ?? (Number.isFinite(mn) ? mn : 0); hi = customHi ?? (Number.isFinite(mx) ? mx : 1);
      if (norm === "custom") { if (customLo == null) loIn.value = fmtNum(lo, 5); if (customHi == null) hiIn.value = fmtNum(hi, 5); }
    }
    if (!(hi > lo)) hi = lo + 1e-12 + Math.abs(lo) * 1e-9;
    range = { lo, hi };
    if (norm === "custom") { lo = customLo ?? lo; hi = customHi ?? hi; if (hi > lo) range = { lo, hi }; }
  }

  async function paint(refit = false): Promise<void> {
    if (!data) return;
    const my = token;
    await computeRange();
    if (my !== token) return;
    const D = data, geo = D.geo;
    const useRgb = rgb && canRgb();
    const tiles: { g: number; c: number }[] = [];
    for (let g = 0; g < D.groups; g++) { if (useRgb) tiles.push({ g, c: -1 }); else for (let c = 0; c < geo.C; c++) tiles.push({ g, c }); }
    const tw = D.Wc, th = D.Hc, T = tiles.length;
    const gap = T > 1 ? 1 : 0;
    const cols = colsOverride > 0 ? Math.min(colsOverride, T) : T === 1 ? 1 : Math.max(1, Math.min(T, Math.round(Math.sqrt((T * (th + gap)) / (tw + gap)))));
    const rows = Math.ceil(T / cols);
    const W = cols * (tw + gap) - gap, H = rows * (th + gap) - gap;
    if (W > 32000 || H > 32000 || W * H > 120_000_000) { overlay.hidden = false; overlay.textContent = "The tile grid is too large to draw — reduce the tiles per row or page."; return; }
    overlay.hidden = true;
    img = { w: W, h: H, tw, th, cols, gap, tiles };
    cv.width = W; cv.height = H;
    const c2 = cv.getContext("2d")!;
    const id = c2.createImageData(W, H);
    const px = id.data;
    const lut = colormapLUT(cmap);
    const perTileNorm = norm === "tile";
    for (let ti = 0; ti < T; ti++) {
      const { g, c } = tiles[ti]!;
      const ox = (ti % cols) * (tw + gap), oy = Math.floor(ti / cols) * (th + gap);
      let lo = range.lo, hi = range.hi;
      if (perTileNorm) {
        let mn = Infinity, mx = -Infinity;
        for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) for (let cc = useRgb ? 0 : c; cc <= (useRgb ? 2 : c); cc++) { const v = val(g, cc, y, x); if (v !== undefined && Number.isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; } }
        if (mn <= mx) { lo = mn; hi = mx > mn ? mx : mn + 1e-12; }
      }
      const inv = 255 / (hi - lo);
      for (let y = 0; y < th; y++) {
        for (let x = 0; x < tw; x++) {
          const o = ((oy + y) * W + ox + x) * 4;
          const comps = useRgb ? 3 : 1;
          let drawn = false;
          for (let k = 0; k < comps; k++) {
            const v = val(g, useRgb ? k : c, y, x);
            if (v === undefined) break;
            drawn = true;
            if (Number.isNaN(v)) { px[o] = NAN_COLOR[0]; px[o + 1] = NAN_COLOR[1]; px[o + 2] = NAN_COLOR[2]; break; }
            if (!Number.isFinite(v)) { px[o] = INF_COLOR[0]; px[o + 1] = INF_COLOR[1]; px[o + 2] = INF_COLOR[2]; break; }
            let q = (v - lo) * inv; q = q < 0 ? 0 : q > 255 ? 255 : Math.round(q);
            if (useRgb) px[o + k] = q; else { const li = q * 3; px[o] = lut[li]!; px[o + 1] = lut[li + 1]!; px[o + 2] = lut[li + 2]!; }
          }
          if (drawn) px[o + 3] = 255;
        }
      }
    }
    c2.putImageData(id, 0, 0);
    legend();
    const gpp = groupsPerPage();
    info.textContent = `${W}×${H} px · ${T.toLocaleString()} tile${T === 1 ? "" : "s"} of ${tw}×${th}${useRgb ? " RGB" : ""} · range ${fmtNum(range.lo, 5)} … ${fmtNum(range.hi, 5)}${perTileNorm ? " (per tile)" : ""}`;
    pageInfo.textContent = oversize()
      ? `Group ${D.gStart + 1} of ${geo.G.toLocaleString()} · showing ${D.Wc}×${D.Hc} crop at (x ${D.x0}, y ${D.y0}) of ${geo.W}×${geo.H}`
      : geo.G > gpp ? `Groups ${D.gStart + 1}–${D.gStart + D.groups} of ${geo.G.toLocaleString()}` : "";
    syncPaging();
    if (refit || !fitted) fit();
    else applyTransform();
  }

  function legend(): void {
    const useRgb = rgb && canRgb();
    rampEl.style.background = useRgb ? "linear-gradient(90deg,#000,#fff)" : colormapGradient(cmap);
    loLbl.textContent = norm === "tile" ? "tile min" : fmtNum(range.lo, 4);
    hiLbl.textContent = norm === "tile" ? "tile max" : fmtNum(range.hi, 4);
  }

  /* ── controls visibility ── */
  function syncControls(): void {
    const geo = geometry();
    layoutCtl.hidden = r < 3;
    modeCtl.hidden = !canRgb();
    modeTg.set(rgb && canRgb() ? "rgb" : "tiles", true);
    cmapCtl.hidden = rgb && canRgb();
    normCtl.hidden = false;
    customCtl.hidden = norm !== "custom";
    colsCtl.hidden = r <= 2;
    squareCtl.hidden = !(r === 1 && d[0]! <= 16384);
    void geo;
  }

  function syncPaging(): void {
    row2.replaceChildren(); cropBox.replaceChildren();
    const geo = geometry(), gpp = groupsPerPage(), over = oversize();
    if (!over && geo.G <= gpp) { row2.append(pageInfo); return; }
    const step = over ? 1 : gpp;
    const num = input({ type: "number", value: String(g0 + 1), ariaLabel: "First group", class: "h-8 w-24 font-mono text-xs" });
    const go = (v: number) => { g0 = Math.max(0, Math.min(geo.G - 1, v)); load(); };
    num.addEventListener("change", () => { const v = Number(num.value); if (Number.isFinite(v)) go(Math.floor(v) - 1); });
    row2.append(
      button(null, { variant: "outline", size: "icon-sm", icon: ChevronLeft, ariaLabel: "Previous page", onClick: () => go(g0 - step) }),
      button(null, { variant: "outline", size: "icon-sm", icon: ChevronRight, ariaLabel: "Next page", onClick: () => go(g0 + step) }),
      h("span", { class: "text-muted-foreground" }, "go to group"), num, pageInfo);
    if (over) {
      const mk = (name: string, key: "y0" | "x0" | "h" | "w") => {
        const i = input({ type: "number", value: String(crop[key]), ariaLabel: name, class: "h-8 w-20 font-mono text-xs" });
        i.addEventListener("change", () => { const v = Math.floor(Number(i.value)); if (Number.isFinite(v)) { crop[key] = v; load(); } });
        return h("label", { class: "flex items-center gap-1" }, h("span", { class: "text-muted-foreground" }, name), i);
      };
      cropBox.append(mk("y", "y0"), mk("x", "x0"), mk("height", "h"), mk("width", "w"));
      row2.append(h("span", { class: "text-muted-foreground" }, "Crop:"), cropBox);
    }
  }

  /* ── pan & zoom ── */
  let zoom = 1, panX = 0, panY = 0, fitted = false;
  function applyTransform(): void {
    cv.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    zoomLbl.textContent = `${Math.round(zoom * 100)}%`;
    zSlider.set(Math.round(Math.log2(zoom) * 10), true);
    marker.classList.add("hidden");
  }
  function fit(): void {
    if (!img) return;
    const vw = view.clientWidth, vh = view.clientHeight;
    if (!vw || !vh) { requestAnimationFrame(fit); return; }
    let z = Math.min((vw - 16) / img.w, (vh - 16) / img.h);
    if (z >= 1) z = Math.floor(z);
    zoom = Math.max(0.02, z);
    panX = (vw - img.w * zoom) / 2; panY = (vh - img.h * zoom) / 2;
    fitted = true; applyTransform();
  }
  function setZoom(z: number, cx: number, cy: number, fromSlider = false): void {
    z = Math.max(0.02, Math.min(256, z));
    panX = cx - (cx - panX) * (z / zoom); panY = cy - (cy - panY) * (z / zoom);
    zoom = z; applyTransform();
    void fromSlider;
  }
  view.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = view.getBoundingClientRect();
    setZoom(zoom * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018)), e.clientX - rect.left, e.clientY - rect.top);
  }, { passive: false });
  let drag: { x: number; y: number; px: number; py: number } | null = null;
  view.addEventListener("pointerdown", (e) => { view.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, px: panX, py: panY }; });
  view.addEventListener("pointerup", () => { drag = null; });
  view.addEventListener("pointermove", (e) => {
    if (drag) { panX = drag.px + e.clientX - drag.x; panY = drag.py + e.clientY - drag.y; applyTransform(); return; }
    inspect(e);
  });
  view.addEventListener("pointerleave", () => { marker.classList.add("hidden"); status.textContent = ""; });
  view.addEventListener("dblclick", () => fit());
  view.addEventListener("keydown", (e) => {
    const k = e.key;
    if (k === "+" || k === "=") setZoom(zoom * 1.25, view.clientWidth / 2, view.clientHeight / 2);
    else if (k === "-") setZoom(zoom / 1.25, view.clientWidth / 2, view.clientHeight / 2);
    else if (k === "0") fit();
    else if (k.startsWith("Arrow")) { const s = 40; if (k === "ArrowLeft") panX += s; if (k === "ArrowRight") panX -= s; if (k === "ArrowUp") panY += s; if (k === "ArrowDown") panY -= s; applyTransform(); }
    else return;
    e.preventDefault();
  });
  new ResizeObserver(debounce(() => { if (!fitted) fit(); }, 60)).observe(view);

  /* ── pixel inspector ── */
  function inspect(e: PointerEvent): void {
    if (!img || !data) return;
    const rect = view.getBoundingClientRect();
    const ix = Math.floor((e.clientX - rect.left - panX) / zoom), iy = Math.floor((e.clientY - rect.top - panY) / zoom);
    if (ix < 0 || iy < 0 || ix >= img.w || iy >= img.h) { marker.classList.add("hidden"); status.textContent = ""; return; }
    const cellW = img.tw + img.gap, cellH = img.th + img.gap;
    const tc = Math.floor(ix / cellW), tr = Math.floor(iy / cellH);
    const x = ix - tc * cellW, y = iy - tr * cellH;
    const ti = tr * img.cols + tc;
    if (x >= img.tw || y >= img.th || ti >= img.tiles.length) { marker.classList.add("hidden"); status.textContent = ""; return; }
    const { g, c } = img.tiles[ti]!;
    const D = data, geo = D.geo;
    const useRgb = rgb && canRgb();
    let text: string;
    if (r <= 1) text = `[${r === 0 ? "" : iy * D.Wc + x}] = ${val(0, 0, y, x) === undefined ? "–" : fmtNum(val(0, 0, y, x), 7)}`;
    else {
      const lead = unravel(D.gStart + g, geo.lead);
      const yy = y + D.y0, xx = x + D.x0;
      const mk = (cc: number) => (layout === "first" ? [...lead, cc, yy, xx] : layout === "last" ? [...lead, yy, xx, cc] : [...lead, yy, xx]);
      if (useRgb) text = `[${mk(0).join(", ")}] R ${fmtNum(val(g, 0, y, x), 6)}  G ${fmtNum(val(g, 1, y, x), 6)}  B ${fmtNum(val(g, 2, y, x), 6)}`;
      else text = `[${mk(c).join(", ")}] = ${fmtNum(val(g, c, y, x), 7)}`;
    }
    status.textContent = `tile ${ti + 1} · ${text}`;
    if (zoom >= 5) {
      marker.classList.remove("hidden");
      marker.style.left = `${panX + (tc * cellW + x) * zoom}px`; marker.style.top = `${panY + (tr * cellH + y) * zoom}px`;
      marker.style.width = `${zoom}px`; marker.style.height = `${zoom}px`;
    } else marker.classList.add("hidden");
  }

  async function savePng(): Promise<void> {
    if (!img) return;
    const k = Math.max(1, Math.min(32, Math.round(Math.max(zoom, 1))), 1);
    let s = k;
    while (s > 1 && (img.w * s > 16384 || img.h * s > 16384 || img.w * img.h * s * s > 100_000_000)) s--;
    const oc = document.createElement("canvas");
    oc.width = img.w * s; oc.height = img.h * s;
    const c2 = oc.getContext("2d")!;
    c2.imageSmoothingEnabled = false;
    c2.drawImage(cv, 0, 0, oc.width, oc.height);
    oc.toBlob((b) => { if (b) downloadBlob(`${safeFileName(t.name)}.png`, b); }, "image/png");
  }

  window.addEventListener("themechange", () => { if (root.isConnected) legend(); });
  rgb = canRgb();
  syncControls();
  void load();
  return root;
}
