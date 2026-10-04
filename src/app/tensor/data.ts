import { h } from "@/lib/dom";
import { cn } from "@/lib/cn";
import { ChevronLeft, ChevronRight, Copy, Download, ArrowLeftRight } from "lucide";
import { alert, badge, button, createVirtualTable, dropdownMenu, input, label, select, slider, toast } from "@/ui";
import type { ModelApi, TensorInfo } from "@/onnx/types";
import { canExportJson, canExportNpy, csvCell, exportJson, exportNpy } from "./export";
import {
  copyText, downloadBlob, errMsg, fmtValue, isComplexDType, isIntDType, safeFileName, shapeStr, type NumFormat, type ViewerOpts,
} from "./util";

const PAGE = 256;
const COLWIN = 48;
const ROWH = 26;
const MAX_COPY = 1_000_000;
const MAX_CSV = 20_000_000;

type Raw = number | string | [number, number];

interface Plane {
  r0: number; rows: number; c0: number; cols: number; lossy: boolean;
  get(i: number, j: number): Raw;
}
type Page = { state: "loading" } | { state: "ok"; plane: Plane } | { state: "err"; msg: string };

export function dataTab(api: ModelApi, info: TensorInfo, opts: ViewerOpts): HTMLElement {
  const wrap = h("div", { class: "flex h-full min-h-0 flex-col gap-3 p-1" });
  const render = (t: TensorInfo) => wrap.replaceChildren(dataView(api, t, opts, (nt) => render(nt)));
  render(info);
  return wrap;
}

function dataView(api: ModelApi, t: TensorInfo, opts: ViewerOpts, switchTensor: (t: TensorInfo) => void): HTMLElement {
  const root = h("div", { class: "flex h-full min-h-0 flex-col gap-3" });
  const r = t.dims.length;
  if (!t.available) {
    root.append(alert("No data to show", "The payload of this tensor is not available (external data was not provided)."));
    return root;
  }
  if (t.n === 0) {
    root.append(alert("Empty tensor", `Shape ${shapeStr(t.dims)} contains no elements.`));
    return root;
  }
  const cplx = isComplexDType(t.dtype), isStr = t.dtype === "string", isBool = t.dtype === "bool";
  const intLike = isIntDType(t.dtype) || isBool;
  const f32 = /^(float32|float16|bfloat16|float8|float4)/.test(t.dtype);

  /* ── state ── */
  let rowDim = r >= 2 ? r - 2 : 0;
  let colDim: number | null = r >= 2 ? r - 1 : null;
  const idx = new Array<number>(r).fill(0);
  let colStart = 0;
  let nf: NumFormat = "auto";
  let prec = 5;
  let epoch = 0;
  let sel: { ar: number; ac: number; fr: number; fc: number } | null = null;
  let lossy = false;
  const pages = new Map<number, Page>();
  const want = new Set<number>();

  const rowSize = () => (r === 0 ? 1 : t.dims[rowDim]!);
  const colSize = () => (colDim == null ? 1 : t.dims[colDim]!);
  const colCount = () => Math.max(0, Math.min(COLWIN, colSize() - colStart));
  const fixedDims = () => t.dims.map((_, d) => d).filter((d) => d !== rowDim && d !== colDim);

  /* ── plane fetching ── */
  async function fetchPlane(r0: number, r1: number, c0: number, c1: number): Promise<Plane> {
    const offset = [...idx], size = t.dims.map(() => 1);
    if (r > 0) { offset[rowDim] = r0; size[rowDim] = r1 - r0; }
    if (colDim != null) { offset[colDim] = c0; size[colDim] = c1 - c0; }
    const res = await api.tensorSlice({ id: t.id, offset, size, maxElems: (r1 - r0) * (c1 - c0) + 1 });
    const stride = new Array<number>(r).fill(1);
    for (let d = r - 2; d >= 0; d--) stride[d] = stride[d + 1]! * res.size[d + 1]!;
    const sr = r > 0 ? stride[rowDim]! : 0, sc = colDim != null ? stride[colDim]! : 0;
    const v = res.values;
    const get = (i: number, j: number): Raw => {
      const lin = i * sr + j * sc;
      if (isStr) return (v as string[])[lin] ?? "";
      if (cplx) return [(v as Float64Array)[2 * lin]!, (v as Float64Array)[2 * lin + 1]!];
      return (v as Float64Array)[lin]!;
    };
    return { r0, rows: r1 - r0, c0, cols: c1 - c0, lossy: res.lossy, get };
  }

  /* ── formatting ── */
  const fmt1 = (v: number): string => (isBool ? (v ? "true" : "false") : fmtValue(v, nf, prec, intLike));
  const cellText = (raw: Raw): string => {
    if (typeof raw === "string") return raw;
    if (typeof raw === "number") return fmt1(raw);
    const [a, b] = raw;
    return `${fmtValue(a, nf, prec, false)} ${b < 0 || Object.is(b, -0) ? "−" : "+"} ${fmtValue(Math.abs(b), nf, prec, false)}j`;
  };
  const exactNum = (v: number): string => (!Number.isFinite(v) ? (Number.isNaN(v) ? "NaN" : v > 0 ? "Inf" : "-Inf") : f32 ? String(Number(v.toPrecision(9))) : String(v));
  const exactText = (raw: Raw): string => (typeof raw === "string" ? raw : typeof raw === "number" ? exactNum(raw) : `${exactNum(raw[0])}${raw[1] < 0 ? "-" : "+"}${exactNum(Math.abs(raw[1]))}j`);
  const colWidth = () => (isStr ? 220 : cplx ? 190 : nf === "exp" || prec > 6 ? 130 : 104);

  /* ── table ── */
  const colsSpec = () => {
    const cs = [{ header: "index", width: 76, class: "sticky left-0 z-[2] bg-background text-muted-foreground border-r" }];
    if (colDim == null) cs.push({ header: "value", width: Math.max(colWidth(), 220), class: "" });
    else for (let j = 0; j < colCount(); j++) cs.push({ header: String(colStart + j), width: colWidth(), class: "" });
    return cs.map((c) => ({ ...c, align: (c.header === "index" ? "left" : isStr ? "left" : "right") as "left" | "right" }));
  };

  const inSel = (row: number, col: number): boolean => {
    if (!sel) return false;
    const a = Math.min(sel.ar, sel.fr), b = Math.max(sel.ar, sel.fr), c = Math.min(sel.ac, sel.fc), d = Math.max(sel.ac, sel.fc);
    return row >= a && row <= b && col >= c && col <= d;
  };

  const getRaw = (row: number, col: number): Raw | undefined => {
    const p = pages.get(Math.floor(row / PAGE));
    if (!p || p.state !== "ok") return undefined;
    const pl = p.plane;
    if (col < pl.c0 || col >= pl.c0 + pl.cols) return undefined;
    return pl.get(row - pl.r0, col - pl.c0);
  };

  let table: ReturnType<typeof createVirtualTable>;
  const flush = (() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    return () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const [a, b] = table.getRange();
        const p0 = Math.floor(a / PAGE), p1 = Math.floor(Math.max(a, b - 1) / PAGE);
        for (const p of [...want]) {
          want.delete(p);
          if (p < p0 || p > p1 || pages.has(p)) continue;
          pages.set(p, { state: "loading" });
          const my = epoch;
          const r0 = p * PAGE, r1 = Math.min(rowSize(), r0 + PAGE), c0 = colStart, c1 = colStart + colCount();
          fetchPlane(r0, r1, c0, c1).then((plane) => {
            if (my !== epoch) return;
            pages.set(p, { state: "ok", plane });
            if (plane.lossy && !lossy) { lossy = true; lossyBadge.hidden = false; }
            table.refresh(); updateStatus();
          }).catch((e) => { if (my !== epoch) return; pages.set(p, { state: "err", msg: errMsg(e) }); table.refresh(); });
        }
        // anything the render wanted that was skipped (scrolled past) will be re-requested on the next render
      }, 45);
    };
  })();

  function cell(row: number, col: number): string | Node {
    if (col === 0) return String(row);
    const absCol = colDim == null ? 0 : colStart + col - 1;
    const pn = Math.floor(row / PAGE);
    const p = pages.get(pn);
    if (!p || p.state === "loading") {
      if (!p) { want.add(pn); flush(); }
      return h("span", { class: "text-muted-foreground/50 block" }, "…");
    }
    if (p.state === "err") return h("span", { class: "text-destructive block", title: p.msg }, "error");
    const raw = p.plane.get(row - p.plane.r0, absCol - p.plane.c0);
    const num = typeof raw === "number" ? raw : null;
    const bad = num != null && !Number.isFinite(num);
    return h("span", {
      class: cn("block -mx-2 px-2", bad && "text-destructive font-semibold", num === 0 && "text-muted-foreground/60", inSel(row, absCol) && "bg-primary/20 ring-primary/50 ring-1 ring-inset"),
      dataset: { c: absCol },
    }, cellText(raw));
  }

  table = createVirtualTable({ columns: colsSpec(), rowHeight: ROWH, count: rowSize(), cell, class: "rounded-lg border" });
  const tableEl = table.el;
  tableEl.tabIndex = 0;
  tableEl.setAttribute("aria-label", `Values of ${t.name}`);
  const fixHeadWidth = () => {
    const head = tableEl.firstElementChild as HTMLElement | null;
    if (head) head.style.width = `${colsSpec().reduce((s, c) => s + c.width, 0)}px`;
    // pin the corner cell of the sticky header
    const corner = head?.firstElementChild as HTMLElement | null;
    if (corner) corner.classList.add("sticky", "left-0", "z-[3]", "bg-muted");
  };
  fixHeadWidth();

  /* ── selection ── */
  const status = h("div", { class: "text-muted-foreground min-h-5 truncate font-mono text-xs" });
  const lossyBadge = badge("int64 > 2^53 rounded", { variant: "secondary" });
  lossyBadge.hidden = true;

  function indexOf(row: number, col: number): number[] {
    const ix = [...idx];
    if (r > 0) ix[rowDim] = row;
    if (colDim != null) ix[colDim] = col;
    return ix;
  }
  function updateStatus(): void {
    if (!sel) { status.textContent = `${t.dtype} · ${shapeStr(t.dims)} · click a cell to inspect, shift-click or shift+arrows to select a range, Ctrl+C to copy`; return; }
    const raw = getRaw(sel.fr, sel.fc);
    const n = (Math.abs(sel.fr - sel.ar) + 1) * (Math.abs(sel.fc - sel.ac) + 1);
    status.textContent = `[${indexOf(sel.fr, sel.fc).join(", ")}]${raw === undefined ? "" : " = " + exactText(raw)}${n > 1 ? `   ·   ${n.toLocaleString()} cells selected` : ""}`;
  }
  function setSel(ar: number, ac: number, fr: number, fc: number): void {
    sel = { ar, ac, fr, fc }; table.refresh(); updateStatus();
  }
  tableEl.addEventListener("click", (e) => {
    const rowEl = (e.target as HTMLElement).closest<HTMLElement>("[data-index]");
    if (!rowEl) return;
    const row = Number(rowEl.dataset.index);
    const cEl = (e.target as HTMLElement).closest<HTMLElement>("[data-c]");
    if (!cEl) { // index column → select the visible part of the row
      setSel(row, colDim == null ? 0 : colStart, row, colDim == null ? 0 : colStart + colCount() - 1);
      return;
    }
    const c = Number(cEl.dataset.c);
    if (e.shiftKey && sel) setSel(sel.ar, sel.ac, row, c); else setSel(row, c, row, c);
  });
  tableEl.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "c") { e.preventDefault(); void copySelection(); return; }
    if (!sel) { if (e.key.startsWith("Arrow")) { e.preventDefault(); setSel(0, colStart, 0, colStart); } return; }
    let { fr, fc } = sel;
    const cmin = colStart, cmax = colDim == null ? 0 : colStart + colCount() - 1;
    switch (e.key) {
      case "ArrowDown": fr++; break; case "ArrowUp": fr--; break;
      case "ArrowRight": fc++; break; case "ArrowLeft": fc--; break;
      case "PageDown": fr += 12; break; case "PageUp": fr -= 12; break;
      case "Home": fr = 0; break; case "End": fr = rowSize() - 1; break;
      default: return;
    }
    e.preventDefault();
    fr = Math.max(0, Math.min(rowSize() - 1, fr)); fc = Math.max(cmin, Math.min(cmax, fc));
    if (e.shiftKey) setSel(sel.ar, sel.ac, fr, fc); else setSel(fr, fc, fr, fc);
    table.scrollToIndex(fr, "nearest");
  });

  /* ── copy / export ── */
  async function planeText(r0: number, r1: number, c0: number, c1: number, sepCell: string, csv: boolean): Promise<string> {
    const lines: string[] = [];
    const per = Math.max(1, Math.floor(2_000_000 / Math.max(1, c1 - c0)));
    for (let a = r0; a < r1; a += per) {
      const p = await fetchPlane(a, Math.min(r1, a + per), c0, c1);
      for (let i = 0; i < p.rows; i++) {
        const cells: string[] = [];
        for (let j = 0; j < p.cols; j++) { const s = exactText(p.get(i, j)); cells.push(csv ? csvCell(s) : s); }
        lines.push(cells.join(sepCell));
      }
    }
    return lines.join("\n");
  }
  async function copySelection(): Promise<void> {
    let r0: number, r1: number, c0: number, c1: number;
    if (sel) { r0 = Math.min(sel.ar, sel.fr); r1 = Math.max(sel.ar, sel.fr) + 1; c0 = Math.min(sel.ac, sel.fc); c1 = Math.max(sel.ac, sel.fc) + 1; }
    else { const [a, b] = table.getRange(); r0 = a; r1 = Math.max(a + 1, b); c0 = colStart; c1 = colStart + colCount(); }
    if ((r1 - r0) * (c1 - c0) > MAX_COPY) return void toast({ title: "Selection too large to copy", description: `Limit is ${MAX_COPY.toLocaleString()} cells — use Export instead.`, variant: "destructive" });
    try {
      const text = await planeText(r0, r1, c0, c1, "\t", false);
      await copyText(text, `${(r1 - r0) * (c1 - c0)} cell${(r1 - r0) * (c1 - c0) === 1 ? "" : "s"} copied as TSV`);
    } catch (e) { toast({ title: "Copy failed", description: errMsg(e), variant: "destructive" }); }
  }
  async function exportCsv(): Promise<void> {
    const R = rowSize(), C = colSize();
    if (R * C > MAX_CSV) return void toast({ title: "Slice too large for CSV", description: `${(R * C).toLocaleString()} cells; limit ${MAX_CSV.toLocaleString()}. Select a range and copy it, or export as .npy.`, variant: "destructive" });
    const dismiss = toast({ title: "Preparing CSV…", duration: 600_000 });
    try {
      const text = await planeText(0, R, 0, C, ",", true);
      downloadBlob(`${safeFileName(t.name)}${r > 2 ? `_${idx.map((v, d) => (d === rowDim || d === colDim ? ":" : v)).join("_")}` : ""}.csv`, new Blob([text + "\n"], { type: "text/csv" }));
      dismiss();
    } catch (e) { dismiss(); toast({ title: "Export failed", description: errMsg(e), variant: "destructive" }); }
  }

  const npyWhy = canExportNpy(t), jsonWhy = canExportJson(t);
  const exportBtn = button("Copy / export", { variant: "outline", size: "sm", icon: Download });
  dropdownMenu(exportBtn, () => [
    { label: sel ? "Copy selection (TSV)" : "Copy visible rows (TSV)", icon: Copy, shortcut: "Ctrl+C", onSelect: () => void copySelection() },
    { type: "separator" },
    { label: "Export this slice as CSV", onSelect: () => void exportCsv() },
    { label: npyWhy ? `Export tensor as .npy — ${npyWhy}` : "Export tensor as .npy", disabled: !!npyWhy, onSelect: () => void exportNpy(api, t) },
    { label: jsonWhy ? `Export tensor as JSON — ${jsonWhy}` : "Export tensor as JSON", disabled: !!jsonWhy, onSelect: () => void exportJson(api, t) },
  ]);

  /* ── controls ── */
  const dimOpts = () => t.dims.map((d, i) => ({ value: String(i), label: `dim ${i}  ·  ${d}` }));
  const rowSel = select(dimOpts(), { value: String(rowDim), size: "sm", ariaLabel: "Row dimension", class: "w-36", onChange: (v) => { const n = Number(v); if (n === colDim) colDim = rowDim; rowDim = n; resetAll(); } });
  const colSel = select(dimOpts(), { value: String(colDim ?? 0), size: "sm", ariaLabel: "Column dimension", class: "w-36", onChange: (v) => { const n = Number(v); if (n === rowDim) rowDim = colDim!; colDim = n; resetAll(); } });
  const swap = button(null, { variant: "ghost", size: "icon-sm", icon: ArrowLeftRight, title: "Swap rows and columns", ariaLabel: "Swap rows and columns", onClick: () => { if (colDim == null) return; [rowDim, colDim] = [colDim, rowDim]; resetAll(); } });
  const fmtSel = select([{ value: "auto", label: "Auto" }, { value: "fixed", label: "Fixed" }, { value: "exp", label: "Exponent" }], { value: nf, size: "sm", ariaLabel: "Number format", class: "w-28", onChange: (v) => { nf = v as NumFormat; table.setColumns(colsSpec()); fixHeadWidth(); table.refresh(); } });
  const precLbl = h("span", { class: "text-muted-foreground w-5 font-mono text-xs tabular-nums" }, String(prec));
  const precSl = slider({ min: 1, max: 10, step: 1, value: prec, ariaLabel: "Precision", class: "w-24", onInput: (v) => { prec = v; precLbl.textContent = String(v); table.setColumns(colsSpec()); fixHeadWidth(); table.refresh(); } });
  const ctl = (name: string, ...n: Node[]) => h("div", { class: "flex items-center gap-2" }, label(name, { class: "text-muted-foreground text-xs" }), ...n);

  const topRow = h("div", { class: "flex flex-wrap items-center gap-x-4 gap-y-2" });
  topRow.append(ctl("Rows", rowSel.el));
  if (r >= 2) topRow.append(swap, ctl("Columns", colSel.el));
  if (!isStr && !cplx && !isBool && !intLike) topRow.append(ctl("Format", fmtSel.el, precSl.el, precLbl));
  else if (cplx) topRow.append(ctl("Format", fmtSel.el, precSl.el, precLbl));
  topRow.append(h("span", { class: "flex-1" }), lossyBadge, exportBtn);

  /* sparse component switch */
  if (t.sparse && opts.tensors) {
    const sp = t.sparse, all = opts.tensors;
    const vals = all[sp.valuesTid], inds = all[sp.indicesTid];
    if (vals && inds) {
      const s = select([{ value: "dense", label: "Dense (expanded)" }, { value: "values", label: `Stored values (${sp.nnz.toLocaleString()})` }, { value: "indices", label: "Indices" }], { value: "dense", size: "sm", ariaLabel: "Sparse view", class: "w-48", onChange: (v) => { if (v === "values") switchTensor(vals); else if (v === "indices") switchTensor(inds); } });
      topRow.prepend(ctl("Sparse", s.el));
    }
  }
  if (!t.sparse && opts.tensors) {
    for (const o of opts.tensors) if (o.sparse && (o.sparse.valuesTid === t.id || o.sparse.indicesTid === t.id)) {
      topRow.prepend(button("← Back to sparse tensor", { variant: "outline", size: "sm", onClick: () => switchTensor(o) }));
      break;
    }
  }

  const fixedBox = h("div", { class: "flex flex-col gap-1.5" });
  function buildFixed(): void {
    fixedBox.replaceChildren();
    for (const d of fixedDims()) {
      const n = t.dims[d]!;
      const num = input({ type: "number", value: String(idx[d]), ariaLabel: `Index in dim ${d}`, class: "h-8 w-20 font-mono text-xs" });
      const sl = slider({ min: 0, max: Math.max(0, n - 1), step: 1, value: idx[d], ariaLabel: `Index in dim ${d}`, class: "w-48 max-w-full", onInput: (v) => { set(v); } });
      const set = (v: number) => {
        v = Math.max(0, Math.min(n - 1, Math.round(v)));
        if (v === idx[d]) return;
        idx[d] = v; num.value = String(v); sl.set(v, true); softReset();
      };
      num.addEventListener("change", () => { const v = Number(num.value); if (Number.isFinite(v)) set(v); else num.value = String(idx[d]); });
      fixedBox.append(h("div", { class: "flex flex-wrap items-center gap-2" },
        h("span", { class: "text-muted-foreground w-28 text-xs" }, `dim ${d}  ·  size ${n}`),
        button(null, { variant: "ghost", size: "icon-sm", icon: ChevronLeft, ariaLabel: `Previous in dim ${d}`, onClick: () => set(idx[d]! - 1) }), sl.el,
        button(null, { variant: "ghost", size: "icon-sm", icon: ChevronRight, ariaLabel: `Next in dim ${d}`, onClick: () => set(idx[d]! + 1) }), num));
    }
  }

  /* column window controls */
  const winBox = h("div", { class: "flex flex-wrap items-center gap-2 text-xs" });
  function buildWin(): void {
    winBox.replaceChildren();
    if (colDim == null || colSize() <= COLWIN) return;
    const go = (s: number) => { colStart = Math.max(0, Math.min(colSize() - 1, s)); softReset(); };
    const num = input({ type: "number", value: String(colStart), ariaLabel: "First column", class: "h-8 w-24 font-mono text-xs" });
    num.addEventListener("change", () => { const v = Number(num.value); if (Number.isFinite(v)) go(Math.floor(v)); });
    winBox.append(
      h("span", { class: "text-muted-foreground" }, `Columns ${colStart}–${colStart + colCount() - 1} of ${colSize().toLocaleString()}`),
      button(null, { variant: "outline", size: "icon-sm", icon: ChevronLeft, ariaLabel: "Previous columns", onClick: () => go(colStart - COLWIN) }),
      button(null, { variant: "outline", size: "icon-sm", icon: ChevronRight, ariaLabel: "Next columns", onClick: () => go(colStart + COLWIN) }),
      h("span", { class: "text-muted-foreground" }, "start at"), num);
  }

  function softReset(): void {
    epoch++; pages.clear(); want.clear(); sel = null;
    buildWin(); table.setColumns(colsSpec()); fixHeadWidth(); table.setCount(rowSize()); table.refresh(); updateStatus();
  }
  function resetAll(): void {
    colStart = 0; rowSel.set(String(rowDim), true); if (colDim != null) colSel.set(String(colDim), true);
    buildFixed(); softReset();
  }

  buildFixed(); buildWin(); updateStatus();
  const info2 = h("div", { class: "text-muted-foreground flex items-center gap-2 text-xs" },
    h("span", {}, r === 0 ? "scalar" : r === 1 ? `${rowSize().toLocaleString()} values` : `${rowSize().toLocaleString()} rows × ${colSize().toLocaleString()} columns`));
  root.append(topRow, fixedBox, winBox, h("div", { class: "min-h-0 flex-1" }, tableEl), h("div", { class: "flex items-center justify-between gap-3" }, status, info2));
  return root;
}
