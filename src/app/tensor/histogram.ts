import { h } from "@/lib/dom";
import { RotateCcw } from "lucide";
import { alert, button, input, label, skeleton, slider, switchControl, toggleGroup } from "@/ui";
import type { HistogramResult, TensorStats } from "@/onnx/types";
import { createChartCanvas, fmtAxisCount, fmtTick, niceTicks, tooltipBox, type ChartColors } from "./chart";
import { debounce, errMsg, fmtNum, isNumericDType, parseNum, type TensorCtx } from "./util";

const M = { l: 54, r: 14, t: 12, b: 30 };

export function histogramTab(ctx: TensorCtx): HTMLElement {
  const t = ctx.info;
  const root = h("div", { class: "flex flex-col gap-4 p-1 pb-4" });
  if (!t.available || !isNumericDType(t.dtype)) {
    root.append(alert("No histogram", !t.available ? "The tensor payload is not available." : `Histograms are not defined for ${t.dtype} tensors.`));
    return root;
  }
  const holder = h("div", { class: "flex flex-col gap-4" }, skeleton("h-9 w-full"), skeleton("h-[300px] w-full"));
  root.append(holder);
  ctx.stats().then((s) => build(s)).catch((e) => holder.replaceChildren(alert("Could not compute statistics", errMsg(e), { variant: "destructive" })));

  function build(s: TensorStats): void {
    const hasRange = s.min != null && s.max != null && Number.isFinite(s.min) && Number.isFinite(s.max);
    if (!hasRange) {
      holder.replaceChildren(alert("No finite values", "All elements are NaN or infinite, or the tensor is empty."));
      return;
    }
    const dmin = s.min!, dmax = s.max!;
    const logOk = dmax > 0;
    const autoLogLo = Math.max(dmax * 1e-6, 1e-30);
    let bins = 128, ylog = false, xlog = false, cum = false;
    let lo: number | null = null, hi: number | null = null; // null → automatic
    let res: HistogramResult | null = null, token = 0, hoverBin = -1, err = "";
    let brush: { x0: number; x1: number } | null = null;

    const rangeLo = () => (lo ?? (xlog ? autoLogLo : dmin));
    const rangeHi = () => (hi ?? dmax);

    const minIn = input({ type: "text", ariaLabel: "Range minimum", class: "h-8 w-28 font-mono text-xs", onChange: (v) => { const n = parseNum(v); lo = n; fetchNow(); } });
    const maxIn = input({ type: "text", ariaLabel: "Range maximum", class: "h-8 w-28 font-mono text-xs", onChange: (v) => { const n = parseNum(v); hi = n; fetchNow(); } });
    const syncInputs = () => { minIn.value = fmtNum(rangeLo(), 6); maxIn.value = fmtNum(rangeHi(), 6); };
    const binsLbl = h("span", { class: "text-muted-foreground w-8 text-right font-mono text-xs tabular-nums" }, String(bins));
    const binSlider = slider({ min: 16, max: 512, step: 16, value: bins, ariaLabel: "Number of bins", class: "w-40", onInput: (v) => { bins = v; binsLbl.textContent = String(v); fetch_(); } });
    const reset = button("Reset", { variant: "outline", size: "sm", icon: RotateCcw, title: "Reset range to the data range (or double-click the chart)", onClick: () => resetRange() });
    const yScale = toggleGroup([{ value: "linear", label: "Linear" }, { value: "log", label: "Log" }], { variant: "outline", size: "sm", ariaLabel: "Y scale", value: "linear", onChange: (v) => { ylog = v[0] === "log"; chart.redraw(); } });
    const xScale = toggleGroup([{ value: "linear", label: "Linear" }, { value: "log", label: "Log" }], { variant: "outline", size: "sm", ariaLabel: "X scale", value: "linear", onChange: (v) => { xlog = v[0] === "log"; if (xlog && !logOk) { xlog = false; return; } lo = hi = null; syncInputs(); fetchNow(); } });
    if (!logOk) for (const b of xScale.el.querySelectorAll("button")) if (b.textContent === "Log") { b.disabled = true; b.title = "Log-spaced bins need positive values"; }
    const cumSw = switchControl({ ariaLabel: "Cumulative", onChange: (v) => { cum = v; chart.redraw(); } });

    const ctrl = (name: string, ...n: Node[]) => h("div", { class: "flex items-center gap-2" }, label(name, { class: "text-muted-foreground text-xs" }), ...n);
    const controls = h("div", { class: "flex flex-wrap items-center gap-x-5 gap-y-2" },
      ctrl("Bins", binSlider.el, binsLbl), ctrl("Range", minIn, h("span", { class: "text-muted-foreground" }, "→"), maxIn, reset),
      ctrl("Y", yScale.el), ctrl("X bins", xScale.el), ctrl("Cumulative", cumSw.el));

    const tip = tooltipBox();
    const status = h("div", { class: "text-muted-foreground flex flex-wrap gap-x-6 gap-y-1 text-xs tabular-nums" });

    /* geometry helpers */
    const plot = () => { const { w, h: hh } = chart.size(); return { x0: M.l, x1: w - M.r, y0: M.t, y1: hh - M.b }; };
    const total = () => (res ? res.counts.reduce((a, b) => a + b, 0) : 0);
    const series = (): number[] => {
      if (!res) return [];
      if (!cum) return res.counts;
      let a = 0;
      return res.counts.map((c) => (a += c));
    };
    const xToVal = (px: number): number => {
      const p = plot();
      const f = Math.min(1, Math.max(0, (px - p.x0) / (p.x1 - p.x0)));
      const a = res!.min, b = res!.max;
      return xlog ? 10 ** (Math.log10(a) + f * (Math.log10(b) - Math.log10(a))) : a + f * (b - a);
    };
    const valToX = (v: number): number => {
      const p = plot();
      const a = res!.min, b = res!.max;
      const f = xlog ? (Math.log10(Math.max(v, 1e-300)) - Math.log10(a)) / (Math.log10(b) - Math.log10(a)) : (v - a) / (b - a || 1);
      return p.x0 + f * (p.x1 - p.x0);
    };
    const binEdge = (i: number): number => {
      const a = res!.min, b = res!.max, f = i / bins;
      return xlog ? 10 ** (Math.log10(a) + f * (Math.log10(b) - Math.log10(a))) : a + f * (b - a);
    };

    function draw(ctx2: CanvasRenderingContext2D, w: number, hh: number, c: ChartColors): void {
      const p = plot();
      ctx2.font = "11px ui-sans-serif, system-ui, sans-serif";
      if (!res) { ctx2.fillStyle = c.mutedFg; ctx2.textAlign = "center"; ctx2.fillText(err || "Loading…", w / 2, hh / 2); return; }
      const ser = series();
      const maxV = Math.max(1, ...ser);
      const yMap = (v: number) => {
        const f = ylog ? Math.log10(1 + v) / Math.log10(1 + maxV) : v / maxV;
        return p.y1 - f * (p.y1 - p.y0);
      };
      /* y grid + ticks */
      ctx2.textAlign = "right"; ctx2.textBaseline = "middle"; ctx2.lineWidth = 1;
      const yt = ylog ? Array.from({ length: Math.floor(Math.log10(maxV)) + 2 }, (_, i) => (i === 0 ? 0 : 10 ** (i - 1))).filter((v) => v <= maxV) : niceTicks(0, maxV, 4);
      for (const v of yt) {
        const y = Math.round(yMap(v)) + 0.5;
        ctx2.strokeStyle = c.border; ctx2.beginPath(); ctx2.moveTo(p.x0, y); ctx2.lineTo(p.x1, y); ctx2.stroke();
        ctx2.fillStyle = c.mutedFg; ctx2.fillText(fmtAxisCount(v), p.x0 - 6, y);
      }
      /* bars */
      const bw = (p.x1 - p.x0) / bins;
      for (let i = 0; i < bins; i++) {
        const v = ser[i]!;
        if (!v) continue;
        const x = p.x0 + i * bw, y = yMap(v);
        ctx2.fillStyle = i === hoverBin ? c.barHi : c.bar;
        ctx2.globalAlpha = cum ? 0.75 : 1;
        ctx2.fillRect(x + (bw > 3 ? 0.5 : 0), y, Math.max(1, bw - (bw > 3 ? 1 : 0)), p.y1 - y);
      }
      ctx2.globalAlpha = 1;
      /* x ticks */
      ctx2.textAlign = "center"; ctx2.textBaseline = "top";
      const xt = xlog
        ? (() => { const a = Math.ceil(Math.log10(res!.min)), b = Math.floor(Math.log10(res!.max)); const o: number[] = []; for (let e = a; e <= b; e++) o.push(10 ** e); return o.length ? o : [res!.min, res!.max]; })()
        : niceTicks(res.min, res.max, Math.max(3, Math.floor((p.x1 - p.x0) / 90)));
      ctx2.strokeStyle = c.mutedFg;
      ctx2.beginPath(); ctx2.moveTo(p.x0, p.y1 + 0.5); ctx2.lineTo(p.x1, p.y1 + 0.5); ctx2.stroke();
      for (const v of xt) {
        const x = Math.round(valToX(v)) + 0.5;
        if (x < p.x0 - 1 || x > p.x1 + 1) continue;
        ctx2.beginPath(); ctx2.moveTo(x, p.y1); ctx2.lineTo(x, p.y1 + 4); ctx2.stroke();
        ctx2.fillStyle = c.mutedFg; ctx2.fillText(fmtTick(v), x, p.y1 + 7);
      }
      /* zero marker */
      if (!xlog && res.min < 0 && res.max > 0) {
        const x = Math.round(valToX(0)) + 0.5;
        ctx2.strokeStyle = c.accent; ctx2.setLineDash([3, 3]); ctx2.beginPath(); ctx2.moveTo(x, p.y0); ctx2.lineTo(x, p.y1); ctx2.stroke(); ctx2.setLineDash([]);
      }
      /* brush */
      if (brush) {
        const a = Math.min(brush.x0, brush.x1), b = Math.max(brush.x0, brush.x1);
        ctx2.fillStyle = c.barHi; ctx2.globalAlpha = 0.18; ctx2.fillRect(a, p.y0, b - a, p.y1 - p.y0);
        ctx2.globalAlpha = 1; ctx2.strokeStyle = c.barHi; ctx2.strokeRect(a + 0.5, p.y0 + 0.5, b - a, p.y1 - p.y0);
      }
    }

    const chart = createChartCanvas(320, draw, "rounded-lg border bg-card");
    chart.el.append(tip.el);
    chart.canvas.classList.add("cursor-crosshair");

    const upStatus = () => {
      if (!res) return;
      status.replaceChildren(
        h("span", {}, `Range ${fmtNum(res.min, 6)} → ${fmtNum(res.max, 6)}`),
        h("span", {}, `${total().toLocaleString()} values counted`),
        h("span", {}, `below range: ${res.below.toLocaleString()}${xlog ? " (incl. ≤ 0)" : ""}`),
        h("span", {}, `above range: ${res.above.toLocaleString()}`),
        ...(s.nan || s.inf ? [h("span", { class: "text-destructive" }, `${s.nan} NaN / ${s.inf} Inf excluded`)] : []),
        h("span", {}, "Drag to zoom · double-click to reset"));
    };

    function fetchNow(): void {
      const my = ++token;
      let a = rangeLo(), b = rangeHi();
      if (!(b >= a)) [a, b] = [b, a];
      syncInputs();
      ctx.api.tensorHistogram({ id: t.id, bins, min: a, max: b, log: xlog }).then((r) => {
        if (my !== token) return;
        res = r; err = ""; upStatus(); chart.redraw();
      }).catch((e) => { if (my === token) { res = null; err = errMsg(e); chart.redraw(); } });
    }
    const fetch_ = debounce(fetchNow, 90);
    function resetRange(): void { lo = hi = null; brush = null; fetchNow(); }

    /* pointer interaction */
    const cx = (e: PointerEvent | MouseEvent) => e.clientX - chart.canvas.getBoundingClientRect().left;
    const cy = (e: PointerEvent | MouseEvent) => e.clientY - chart.canvas.getBoundingClientRect().top;
    let dragging = false;
    chart.canvas.addEventListener("pointerdown", (e) => {
      if (!res) return;
      const p = plot();
      if (cx(e) < p.x0 - 2 || cx(e) > p.x1 + 2) return;
      dragging = true; chart.canvas.setPointerCapture(e.pointerId);
      brush = { x0: Math.min(p.x1, Math.max(p.x0, cx(e))), x1: Math.min(p.x1, Math.max(p.x0, cx(e))) };
    });
    chart.canvas.addEventListener("pointermove", (e) => {
      if (!res) return;
      const p = plot(), x = cx(e);
      if (dragging && brush) { brush.x1 = Math.min(p.x1, Math.max(p.x0, x)); tip.hide(); chart.redraw(); return; }
      const bi = Math.floor(((x - p.x0) / (p.x1 - p.x0)) * bins);
      if (bi < 0 || bi >= bins || cy(e) > p.y1 + 2) { if (hoverBin !== -1) { hoverBin = -1; chart.redraw(); } tip.hide(); return; }
      if (bi !== hoverBin) { hoverBin = bi; chart.redraw(); }
      const n = res.counts[bi]!, tot = total();
      const cumN = res.counts.slice(0, bi + 1).reduce((a, b) => a + b, 0);
      tip.show(x, cy(e), h("div", { class: "flex flex-col gap-0.5 tabular-nums" },
        h("div", { class: "font-mono" }, `[${fmtNum(binEdge(bi), 5)}, ${fmtNum(binEdge(bi + 1), 5)}${bi === bins - 1 ? "]" : ")"}`),
        h("div", {}, `${n.toLocaleString()} values · ${tot ? ((n / tot) * 100).toFixed(n / tot < 0.001 ? 4 : 2) : 0}%`),
        h("div", { class: "text-muted-foreground" }, `cumulative ${tot ? ((cumN / tot) * 100).toFixed(2) : 0}%`)), chart.el.getBoundingClientRect());
    });
    const endDrag = (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      const b = brush; brush = null;
      if (b && Math.abs(b.x1 - b.x0) > 5 && res) {
        const a = xToVal(Math.min(b.x0, b.x1)), z = xToVal(Math.max(b.x0, b.x1));
        lo = a; hi = z; fetchNow();
      } else chart.redraw();
      void e;
    };
    chart.canvas.addEventListener("pointerup", endDrag);
    chart.canvas.addEventListener("pointercancel", endDrag);
    chart.canvas.addEventListener("pointerleave", () => { hoverBin = -1; tip.hide(); chart.redraw(); });
    chart.canvas.addEventListener("dblclick", resetRange);

    holder.replaceChildren(controls, chart.el, status);
    if (s.channelAbsMax) holder.append(channelChart(s));
    syncInputs();
    fetchNow();
  }
  return root;
}

/** per-output-channel |x| max bar chart (dim 0) */
function channelChart(s: TensorStats): HTMLElement {
  const raw = s.channelAbsMax!;
  let sorted = false, hover = -1;
  const tip = tooltipBox();
  const order = (): number[] => {
    const idx = raw.map((_, i) => i);
    if (sorted) idx.sort((a, b) => raw[b]! - raw[a]!);
    return idx;
  };
  let ord = order();
  const maxV = Math.max(...raw, 1e-30);
  const nz = raw.filter((v) => v > 0);
  const ratio = nz.length ? Math.max(...nz) / Math.min(...nz) : null;
  const chart = createChartCanvas(190, (c2, w, hh, c) => {
    const x0 = M.l, x1 = w - M.r, y0 = 8, y1 = hh - 22, n = raw.length;
    c2.font = "11px ui-sans-serif, system-ui, sans-serif"; c2.textBaseline = "middle"; c2.textAlign = "right";
    for (const v of niceTicks(0, maxV, 3)) {
      const y = Math.round(y1 - (v / maxV) * (y1 - y0)) + 0.5;
      c2.strokeStyle = c.border; c2.beginPath(); c2.moveTo(x0, y); c2.lineTo(x1, y); c2.stroke();
      c2.fillStyle = c.mutedFg; c2.fillText(fmtTick(v), x0 - 6, y);
    }
    const pw = x1 - x0;
    const cols = Math.min(n, Math.max(1, Math.floor(pw)));
    for (let k = 0; k < cols; k++) {
      const a = Math.floor((k * n) / cols), b = Math.max(a + 1, Math.floor(((k + 1) * n) / cols));
      let m = 0, hit = false;
      for (let i = a; i < b; i++) { const v = raw[ord[i]!]!; if (v > m) m = v; if (ord[i] === hover) hit = true; }
      const bw = pw / cols, y = y1 - (m / maxV) * (y1 - y0);
      c2.fillStyle = hit ? c.barHi : c.bar;
      c2.fillRect(x0 + k * bw + (bw > 3 ? 0.5 : 0), y, Math.max(1, bw - (bw > 3 ? 1 : 0)), y1 - y);
    }
    c2.textAlign = "center"; c2.textBaseline = "top"; c2.fillStyle = c.mutedFg;
    c2.fillText(sorted ? "rank" : "0", x0, y1 + 6); c2.fillText(String(n - 1), x1, y1 + 6); c2.fillText(sorted ? "channels sorted by |x| max" : "output channel", (x0 + x1) / 2, y1 + 6);
  }, "rounded-lg border bg-card");
  chart.el.append(tip.el);
  chart.canvas.addEventListener("pointermove", (e) => {
    const r = chart.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top, p = { x0: M.l, x1: r.width - M.r };
    const n = raw.length;
    const k = Math.floor(((x - p.x0) / (p.x1 - p.x0)) * n);
    if (k < 0 || k >= n) { tip.hide(); if (hover !== -1) { hover = -1; chart.redraw(); } return; }
    const ch = ord[k]!;
    if (ch !== hover) { hover = ch; chart.redraw(); }
    tip.show(x, y, h("div", { class: "tabular-nums" }, h("div", { class: "font-mono" }, `channel ${ch}`), h("div", {}, `|x| max ${fmtNum(raw[ch], 5)}`)), r);
  });
  chart.canvas.addEventListener("pointerleave", () => { hover = -1; tip.hide(); chart.redraw(); });
  const sw = switchControl({ ariaLabel: "Sort channels", onChange: (v) => { sorted = v; ord = order(); chart.redraw(); } });
  return h("section", { class: "flex flex-col gap-2" },
    h("div", { class: "flex flex-wrap items-center justify-between gap-2" },
      h("h3", { class: "text-sm font-semibold" }, `Per-channel |x| max`, h("span", { class: "text-muted-foreground ml-2 text-xs font-normal" }, `${raw.length} channels · largest/smallest ${ratio == null ? "–" : fmtNum(ratio, 1)}×`)),
      h("div", { class: "flex items-center gap-2" }, label("Sort", { class: "text-muted-foreground text-xs" }), sw.el)),
    chart.el);
}
