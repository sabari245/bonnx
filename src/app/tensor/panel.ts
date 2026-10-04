import { h } from "@/lib/dom";
import { Image as ImageIcon, Maximize2, Table2 } from "lucide";
import { badge, button } from "@/ui";
import type { ModelApi, TensorInfo, TensorStats } from "@/onnx/types";
import { createChartCanvas } from "./chart";
import { kvGrid } from "./overview";
import { fmtBytes, fmtNum, fmtPct, getStats, isComplexDType, isNumericDType, shapeStr, fmtValue, type ViewerOpts } from "./util";
import { openTensorViewer } from "./viewer";

/**
 * Compact tensor summary for the inspector side panel: header, sparkline histogram, key stats and
 * buttons that open the full explorer. Statistics load lazily when the panel scrolls into view.
 */
export function createTensorPanel(api: ModelApi, tensor: TensorInfo, opts: ViewerOpts = {}): HTMLElement {
  const t = tensor;
  const open = (tab: ViewerOpts["tab"]) => () => openTensorViewer(api, t, { ...opts, tab });
  const root = h("div", { class: "flex flex-col gap-2", "data-slot": "tensor-panel" });

  root.append(h("div", { class: "flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-xs" },
    h("span", {}, t.dtype), h("span", { class: "text-muted-foreground" }, "⟨"), h("span", {}, shapeStr(t.dims)), h("span", { class: "text-muted-foreground" }, "⟩"),
    h("span", { class: "text-muted-foreground" }, `${t.n.toLocaleString()} · ${fmtBytes(t.bytes)}`),
    t.sparse ? badge(`sparse ${fmtPct(t.n ? t.sparse.nnz / t.n : 0)}`, { variant: "secondary" }) : null,
    t.available ? null : badge("external data missing", { variant: "destructive" })));

  const body = h("div", { class: "flex flex-col gap-2" });
  root.append(body);
  const actions = h("div", { class: "flex flex-wrap gap-1.5" },
    button("Explore", { variant: "outline", size: "sm", icon: Maximize2, onClick: open("overview"), title: "Open the tensor explorer" }),
    button("Data", { variant: "ghost", size: "sm", icon: Table2, onClick: open("data") }),
    button("Image", { variant: "ghost", size: "sm", icon: ImageIcon, onClick: open("image"), disabled: t.dtype === "string" || isComplexDType(t.dtype) }));
  root.append(actions);

  if (!t.available) {
    body.append(h("p", { class: "text-muted-foreground text-xs" }, t.external ? `Stored in ${t.external.location}; open the model together with that file to see values.` : "Tensor payload is not available."));
    return root;
  }

  /* tiny tensors: show the values inline */
  if (t.n <= 16) {
    api.tensorSlice({ id: t.id, maxElems: 16 }).then((s) => {
      const ints = /^u?int|^bool/.test(t.dtype);
      const txt = Array.from(s.values as ArrayLike<number | string>, (v) => (typeof v === "string" ? JSON.stringify(v) : fmtValue(v, "auto", 5, ints))).join(", ");
      body.prepend(h("div", { class: "bg-muted/50 rounded-md px-2 py-1.5 font-mono text-xs break-words" }, txt || "—"));
    }).catch(() => undefined);
  }

  if (!isNumericDType(t.dtype)) return root;

  const skeletonEl = h("div", { class: "bg-muted ui-pulse h-12 w-full rounded-md" });
  body.append(skeletonEl);
  const load = () => {
    getStats(api, t.id).then((s) => fill(s)).catch(() => skeletonEl.remove());
  };
  function fill(s: TensorStats): void {
    skeletonEl.remove();
    if (s.hist && s.histMin != null && s.histMax != null) {
      const hist = s.hist, mx = Math.max(...hist, 1);
      const chart = createChartCanvas(48, (c2, w, hh, c) => {
        const bw = w / hist.length;
        for (let i = 0; i < hist.length; i++) {
          const bh = hist[i]! ? Math.max(1, (hist[i]! / mx) * (hh - 2)) : 0;
          c2.fillStyle = c.bar; c2.globalAlpha = 0.9;
          c2.fillRect(i * bw + (bw > 3 ? 0.5 : 0), hh - bh, Math.max(1, bw - (bw > 3 ? 1 : 0)), bh);
        }
        c2.globalAlpha = 1; c2.strokeStyle = c.border; c2.beginPath(); c2.moveTo(0, hh - 0.5); c2.lineTo(w, hh - 0.5); c2.stroke();
      }, "bg-muted/40 rounded-md");
      chart.canvas.setAttribute("role", "img");
      chart.canvas.setAttribute("aria-label", `Histogram from ${fmtNum(s.histMin)} to ${fmtNum(s.histMax)}`);
      body.append(chart.el, h("div", { class: "text-muted-foreground flex justify-between font-mono text-[11px] tabular-nums" }, h("span", {}, fmtNum(s.histMin)), h("span", {}, fmtNum(s.histMax))));
    }
    if (s.nan || s.inf) body.append(badge(`${s.nan} NaN · ${s.inf} Inf`, { variant: "destructive", class: "w-fit" }));
    body.append(kvGrid([
      ["Min / max", `${fmtNum(s.min)} / ${fmtNum(s.max)}`],
      ["Mean / std", `${fmtNum(s.mean)} / ${fmtNum(s.std)}`],
      ["|x| max", fmtNum(s.absmax)],
      ["Zeros", fmtPct(s.zeros)],
    ], "grid-cols-[5.5rem_1fr]"));
    if (s.sampled) body.append(h("p", { class: "text-muted-foreground text-[11px]" }, "Statistics sampled (very large tensor)."));
  }
  if (typeof IntersectionObserver === "undefined") load();
  else {
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); load(); } });
    io.observe(root);
    // if never attached/visible, the observer is released with the element
  }
  return root;
}
