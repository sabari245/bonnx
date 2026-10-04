import { h, icon } from "@/lib/dom";
import { ChartColumn, Image as ImageIcon, Info, Table2 } from "lucide";
import { badge, dialog, tabs, type DialogCtl } from "@/ui";
import type { ModelApi, TensorInfo } from "@/onnx/types";
import { dataTab } from "./data";
import { histogramTab } from "./histogram";
import { imageTab } from "./image";
import { overviewTab } from "./overview";
import { fmtBytes, getStats, invalidateStats, shapeStr, type TensorCtx, type ViewerOpts } from "./util";

const tabLabel = (ic: Parameters<typeof icon>[0], text: string) =>
  h("span", { class: "inline-flex items-center gap-1.5" }, icon(ic), h("span", { class: "max-sm:sr-only" }, text));

/**
 * Netron-style weight explorer: overview, histogram, data table and image views of a tensor.
 * Opens as a large modal dialog and returns its controller.
 */
export function openTensorViewer(api: ModelApi, tensor: TensorInfo, opts: ViewerOpts = {}): DialogCtl {
  let info = tensor;
  const tensors = opts.tensors ? [...opts.tensors] : undefined;
  const d = dialog({
    title: h("span", { class: "block max-w-full truncate font-mono text-base", title: tensor.name }, tensor.name || "(unnamed tensor)"),
    size: "xl",
    class: "h-[min(92vh,880px)] gap-3 p-4 sm:p-6",
  });
  d.body.style.overflow = "hidden";
  const desc = () => d.setDescription(h("span", { class: "flex flex-wrap items-center gap-x-2 gap-y-1" },
    h("span", { class: "font-mono" }, info.dtype), "·", h("span", { class: "font-mono" }, shapeStr(info.dims)), "·",
    h("span", {}, `${info.n.toLocaleString()} elements`), "·", h("span", {}, fmtBytes(info.bytes)), "·", h("span", {}, info.source.replace(/_/g, " ")),
    info.available ? null : badge("data unavailable", { variant: "destructive" })));

  const build = (initial: string): void => {
    desc();
    const ctx: TensorCtx = { api, info, opts: { ...opts, tensors }, stats: () => getStats(api, info.id) };
    const reload = (updated?: TensorInfo[]) => {
      if (updated) {
        for (const u of updated) if (tensors && tensors[u.id]) tensors[u.id] = u;
        const mine = updated.find((u) => u.id === info.id);
        if (mine) info = mine;
      }
      invalidateStats(api);
      build(tb.get());
    };
    const scroll = (el: HTMLElement) => h("div", { class: "h-full overflow-auto pr-1" }, el);
    const tb = tabs([
      { value: "overview", label: tabLabel(Info, "Overview"), content: () => scroll(overviewTab(ctx, reload)) },
      { value: "histogram", label: tabLabel(ChartColumn, "Histogram"), content: () => scroll(histogramTab(ctx)) },
      { value: "data", label: tabLabel(Table2, "Data"), content: () => dataTab(api, info, ctx.opts) },
      { value: "image", label: tabLabel(ImageIcon, "Image"), content: () => imageTab(ctx) },
    ], { value: initial, class: "h-full gap-3", contentClass: "min-h-0 flex-1" });
    tb.list.classList.add("w-fit");
    d.body.replaceChildren(tb.el);
  };

  build(opts.tab ?? "overview");
  d.open();
  return d;
}
