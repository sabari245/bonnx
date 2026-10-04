import { h } from "@/lib/dom";
import { Copy, Upload } from "lucide";
import { alert, badge, button, skeleton } from "@/ui";
import type { TensorInfo } from "@/onnx/types";
import type { TensorCtx } from "./util";
import { copyText, errMsg, fmtBytes, fmtCount, fmtNum, fmtPct, isNumericDType, shapeStr } from "./util";
import type { TensorStats } from "@/onnx/types";

export function kvGrid(rows: [string, string | Node | null | undefined][], cols = "grid-cols-[8.5rem_1fr]"): HTMLElement {
  const g = h("dl", { class: `grid ${cols} gap-x-4 gap-y-1.5 text-sm` });
  for (const [k, v] of rows) {
    if (v == null) continue;
    g.append(h("dt", { class: "text-muted-foreground" }, k), h("dd", { class: "min-w-0 font-mono text-[13px] tabular-nums break-words" }, v));
  }
  return g;
}

const section = (title: string, body: Node, right?: Node) =>
  h("section", { class: "rounded-lg border" },
    h("header", { class: "flex items-center justify-between border-b px-4 py-2.5" }, h("h3", { class: "text-sm font-semibold" }, title), right),
    h("div", { class: "p-4" }, body));

export function statRows(s: TensorStats): [string, string][] {
  return [
    ["Min / max", `${fmtNum(s.min)} / ${fmtNum(s.max)}`],
    ["Mean / median", `${fmtNum(s.mean)} / ${fmtNum(s.median)}`],
    ["Std deviation", fmtNum(s.std)],
    ["|x| mean / max", `${fmtNum(s.absmean)} / ${fmtNum(s.absmax)}`],
    ["P1 / P99", `${fmtNum(s.p1)} / ${fmtNum(s.p99)}`],
    ["L2 norm", fmtNum(s.l2)],
    ["Unique values", fmtCount(s.unique) + (s.unique >= 1000 ? ` (${s.unique.toLocaleString()})` : "")],
    ["Zeros", fmtPct(s.zeros)],
  ];
}

export function overviewTab(ctx: TensorCtx, reload: (updated?: TensorInfo[]) => void): HTMLElement {
  const t = ctx.info;
  const root = h("div", { class: "flex flex-col gap-4 p-1 pb-4" });

  /* payload not available */
  if (!t.available) {
    const pick = button("Provide data file…", { icon: Upload, variant: "outline", size: "sm", onClick: () => {
      const inp = h("input", { type: "file", style: "display:none" }) as HTMLInputElement;
      inp.onchange = async () => {
        const f = inp.files?.[0];
        if (!f) return;
        try {
          const upd = await ctx.api.provideExternal(t.external?.location ?? f.name, await f.arrayBuffer());
          ctx.opts.onExternalProvided?.(upd);
          reload(upd);
        } catch (e) { root.prepend(alert("Could not read the file", errMsg(e), { variant: "destructive" })); }
      };
      document.body.append(inp); inp.click(); inp.remove();
    } });
    root.append(alert("Tensor data is not available",
      h("div", { class: "flex flex-col gap-2" },
        h("p", {}, t.external
          ? `This tensor is stored in an external file (${t.external.location}, offset ${t.external.offset}${t.external.length != null ? `, ${fmtBytes(t.external.length)}` : ""}) that was not loaded alongside the model. Open the model together with its data file, or provide it here.`
          : "The payload of this tensor could not be read from the model file."),
        t.external ? h("div", {}, pick) : null)));
  }

  const info: [string, string | Node | null][] = [
    ["Name", t.name],
    ["Data type", t.dtype],
    ["Shape", `${shapeStr(t.dims)}${t.dims.length ? `  (rank ${t.dims.length})` : ""}`],
    ["Elements", t.n.toLocaleString()],
    ["Size", t.bytes < 1024 ? `${t.bytes} B` : `${fmtBytes(t.bytes)} (${t.bytes.toLocaleString()} bytes)`],
    ["Source", t.source.replace(/_/g, " ")],
    ["Doc", t.docString || null],
    ["External file", t.external ? `${t.external.location} @ ${t.external.offset}${t.external.length != null ? ` + ${t.external.length}` : ""}` : null],
    ["Sparse", t.sparse ? `${t.sparse.nnz.toLocaleString()} stored of ${t.n.toLocaleString()} (${fmtPct(t.n ? t.sparse.nnz / t.n : 0)} dense)` : null],
  ];
  root.append(section("Tensor", kvGrid(info)));

  /* stats */
  const body = h("div", {}, h("div", { class: "grid grid-cols-2 gap-2" }, skeleton("h-5"), skeleton("h-5"), skeleton("h-5"), skeleton("h-5"), skeleton("h-5"), skeleton("h-5")));
  const copyBtn = button("Copy JSON", { variant: "ghost", size: "sm", icon: Copy, disabled: true });
  root.append(section("Statistics", body, copyBtn));

  if (!t.available || !isNumericDType(t.dtype) && t.dtype !== "string") {
    body.replaceChildren(h("p", { class: "text-muted-foreground text-sm" }, t.available ? "Statistics are not computed for this data type." : "Unavailable until the payload is loaded."));
    return root;
  }
  ctx.stats().then((s) => {
    const rows = [...statRows(s)];
    const grid = h("div", { class: "grid gap-x-8 gap-y-1.5 md:grid-cols-2" }, kvGrid(rows.slice(0, 4)), kvGrid(rows.slice(4)));
    const warn: Node[] = [];
    if (s.nan || s.inf)
      warn.push(alert("Non-finite values", `${s.nan.toLocaleString()} NaN and ${s.inf.toLocaleString()} Inf elements — this usually indicates a broken export or overflow.`, { variant: "destructive" }));
    const badges = h("div", { class: "mb-3 flex flex-wrap gap-2" },
      s.sampled ? badge("Sampled (very large tensor)", { variant: "secondary" }) : null,
      s.nan ? badge(`${s.nan} NaN`, { variant: "destructive" }) : null,
      s.inf ? badge(`${s.inf} Inf`, { variant: "destructive" }) : null);
    body.replaceChildren(...warn.map((w) => h("div", { class: "mb-3" }, w)), badges.childNodes.length ? badges : "", grid);
    if (t.dtype === "string") grid.replaceChildren(kvGrid([["Unique strings", s.unique.toLocaleString()]]));
    copyBtn.disabled = false;
    copyBtn.onclick = () => copyText(JSON.stringify({ name: t.name, dtype: t.dtype, shape: t.dims, ...s }, (_, v) => (typeof v === "number" && !Number.isFinite(v) ? String(v) : v), 2), "Statistics copied");
  }).catch((e) => body.replaceChildren(alert("Could not compute statistics", errMsg(e), { variant: "destructive" })));
  return root;
}
