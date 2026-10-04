import "../styles/globals.css";
import { h, icon } from "../lib/dom";
import { initTheme, setTheme, getTheme, resolvedTheme } from "../lib/theme";
import {
  alert, accordion, badge, breadcrumb, button, card, cardAction, cardContent, cardDescription, cardFooter, cardHeader, cardTitle, checkbox, collapsible,
  command, commandDialog, contextMenu, createVirtualList, createVirtualTable, dialog, dropdownMenu, emptyState, input, kbd, kbdGroup, label, popover, progress,
  resizablePanels, scrollArea, select, separator, sheet, skeleton, slider, spinner, switchControl, table, tableBody, tableCell, tableHead, tableHeader, tableRow,
  tabs, textarea, toast, toggle, toggleGroup, tooltip, confirmDialog, type CommandItem,
} from "./index";
import { Activity, ArrowRight, Box, Copy, Download, Eye, Image, Layers, Moon, PanelRight, Search, Settings, Sun, Trash2, Upload } from "lucide";

initTheme();
const q = new URLSearchParams(location.search);
if (q.get("theme")) setTheme(q.get("theme") as "light" | "dark");

const section = (title: string, ...c: (Node | string)[]) =>
  h("section", { class: "flex flex-col gap-3", id: title.toLowerCase().replace(/\W+/g, "-") }, h("h3", { class: "text-muted-foreground text-xs font-semibold tracking-wider uppercase" }, title), ...c);
const row = (...c: (Node | string)[]) => h("div", { class: "flex flex-wrap items-center gap-3" }, ...c);

/* graph tokens swatches */
const cats = ["layer", "activation", "norm", "pool", "math", "reduce", "shape", "quant", "control", "other", "input", "const"];
const tokenSwatches = h("div", { class: "flex flex-col gap-3" },
  row(...cats.map((c) => h("div", { class: "flex h-8 min-w-24 items-center justify-center rounded-md px-3 text-xs font-semibold text-white", style: { background: `var(--cat-${c})` } }, c))),
  row(...[0, 1, 2, 3, 4].map((i) => h("div", { class: "flex h-8 w-20 items-center justify-center rounded-md text-xs font-medium", style: { background: `var(--ramp-${i})`, color: i < 2 ? "#111" : "#fff" } }, `ramp-${i}`))),
  h("div", { class: "flex gap-2" }, ...["graph-bg", "graph-grid", "graph-edge", "graph-edge-hi", "graph-node-bg", "graph-node-row", "graph-node-border", "graph-grey", "graph-sel"].map((t) =>
    h("div", { class: "flex flex-col items-center gap-1 text-[10px]" }, h("div", { class: "size-8 rounded border", style: { background: `var(--${t})` } }), t))),
);

/* buttons */
const buttons = row(
  button("Default"), button("Secondary", { variant: "secondary" }), button("Outline", { variant: "outline" }), button("Ghost", { variant: "ghost" }),
  button("Destructive", { variant: "destructive" }), button("Link", { variant: "link" }), button("Disabled", { disabled: true }),
  button("Open model", { icon: Upload }), button("Small", { size: "sm", variant: "outline" }), button("Large", { size: "lg" }),
  button(null, { size: "icon", variant: "outline", icon: Search, ariaLabel: "Search" }), button(null, { size: "icon-sm", variant: "ghost", icon: Settings, ariaLabel: "Settings" }),
);
const tt = button(null, { size: "icon", variant: "outline", icon: Eye, ariaLabel: "Tooltip target" }); tooltip(tt, "Show attributes", { delay: 100 });
const badges = row(badge("Default"), badge("Secondary", { variant: "secondary" }), badge("Outline", { variant: "outline" }), badge("Destructive", { variant: "destructive" }), badge("Conv", { style: "background:var(--cat-layer);color:white" }));

/* form */
const sw = switchControl({ checked: true, ariaLabel: "Show names" });
const cb = checkbox({ checked: true, ariaLabel: "x" }), cb2 = checkbox({ checked: "indeterminate", ariaLabel: "y" }), cb3 = checkbox({ ariaLabel: "z" });
const sl = slider({ min: 0, max: 100, value: 40, step: 1, ariaLabel: "Zoom" });
const selectCtl = select([{ group: "Layout", options: [{ value: "tb", label: "Top → bottom" }, { value: "lr", label: "Left → right" }] }, { group: "Color", options: [{ value: "op", label: "Operator type" }, { value: "params", label: "Parameters" }, { value: "off", label: "Disabled", disabled: true }] }], { value: "tb", ariaLabel: "Layout" });
const form = h("div", { class: "grid max-w-3xl grid-cols-1 gap-4 sm:grid-cols-2" },
  h("div", { class: "flex flex-col gap-2" }, label("Search"), input({ placeholder: "Find layer, tensor or op…" })),
  h("div", { class: "flex flex-col gap-2" }, label("Layout"), selectCtl.el),
  h("div", { class: "flex flex-col gap-2" }, label("Notes"), textarea({ placeholder: "Textarea", rows: 2 })),
  h("div", { class: "flex flex-col gap-3" },
    h("label", { class: "flex items-center gap-2 text-sm" }, sw.el, "Show attributes"),
    h("label", { class: "flex items-center gap-2 text-sm" }, cb.el, cb2.el, cb3.el, "Checkboxes"),
    h("div", { class: "w-60" }, sl.el)),
  h("div", { class: "flex items-center gap-3" }, kbd("/"), kbdGroup("Ctrl+K"), kbd("Esc")),
  h("div", { class: "flex items-center gap-3" }, toggle("Bold", { icon: Eye, pressed: true, variant: "outline" }).el, toggle(null, { icon: Layers, ariaLabel: "layers" }).el, tt),
);
const tg = toggleGroup([{ value: "op", label: "Op type" }, { value: "params", label: "Parameters" }, { value: "wmax", label: "|w| max" }], { variant: "outline", value: "op" });
const tgm = toggleGroup([{ value: "a", icon: Image, ariaLabel: "Image" }, { value: "b", icon: Activity, ariaLabel: "Hist" }, { value: "c", icon: Box, ariaLabel: "Box" }], { type: "multiple", value: ["a"], variant: "outline", size: "sm" });

/* tabs */
const tabsCtl = tabs([
  { value: "stats", label: "Statistics", content: () => h("div", { class: "text-muted-foreground p-2 text-sm" }, "Weight statistics panel") },
  { value: "hist", label: "Histogram", content: h("div", { class: "text-muted-foreground p-2 text-sm" }, "Histogram panel") },
  { value: "data", label: "Data", content: h("div", { class: "text-muted-foreground p-2 text-sm" }, "Data table") },
], { class: "w-full max-w-md" });

/* cards, alerts */
const cards = row(
  card("w-72", cardHeader("", cardTitle("conv1.weight"), cardDescription("float32 · 64×3×7×7"), cardAction("", button(null, { size: "icon-sm", variant: "ghost", icon: Copy, ariaLabel: "Copy" }))),
    cardContent("flex flex-col gap-2", h("div", { class: "text-2xl font-semibold tabular-nums" }, "9,408"), progress(62).el), cardFooter("gap-2", button("View", { size: "sm" }), button("Export", { size: "sm", variant: "outline", icon: Download }))),
  h("div", { class: "flex w-80 flex-col gap-3" }, alert("Heads up", "This model contains 2 constants with NaN values."), alert("Failed to parse", "Unsupported IR version 12.", { variant: "destructive" })),
  h("div", { class: "flex w-56 flex-col gap-2" }, skeleton("h-4 w-3/4"), skeleton("h-4 w-1/2"), skeleton("h-24 w-full"), row(spinner(), h("span", { class: "text-muted-foreground text-sm" }, "Parsing…")), progress(null).el),
);

/* table */
const tbl = table("", tableHeader(tableRow("", tableHead("Op"), tableHead("Count", "text-right"))), tableBody(
  ...[["Conv", 20], ["Relu", 17], ["BatchNormalization", 19], ["Add", 8], ["MaxPool", 1]].map(([a, b]) => tableRow("", tableCell(String(a)), tableCell(String(b), "text-right tabular-nums")))));

/* menus */
const dd = button("Dropdown", { variant: "outline", icon: Settings });
let showNames = true, radio = "tb";
dropdownMenu(dd.valueOf() as HTMLElement, () => [
  { type: "label", label: "View" },
  { type: "checkbox", label: "Show names", checked: showNames, onChange: (v) => (showNames = v) },
  { type: "separator" },
  { type: "radio", group: "dir", value: "tb", label: "Vertical", checked: radio === "tb", onSelect: (v) => (radio = v) },
  { type: "radio", group: "dir", value: "lr", label: "Horizontal", checked: radio === "lr", onSelect: (v) => (radio = v) },
  { type: "separator" },
  { label: "Copy name", icon: Copy, shortcut: "⌘C", onSelect: () => toast({ title: "Copied" }) },
  { type: "sub", label: "Export", icon: Download, items: [{ label: "SVG", onSelect: () => toast({ title: "SVG" }) }, { label: "PNG" }, { label: "Disabled", disabled: true }] },
  { label: "Delete", icon: Trash2, destructive: true },
]);
const ctxBox = h("div", { class: "text-muted-foreground flex h-24 w-72 items-center justify-center rounded-md border border-dashed text-sm", tabindex: 0 }, "Right-click here");
contextMenu(ctxBox, [{ label: "Copy node name", icon: Copy, onSelect: () => toast({ title: "Copied node name", variant: "success" }) }, { label: "Go to producer", icon: ArrowRight }, { type: "separator" }, { label: "Hide node", icon: Eye }]);
const popBtn = button("Popover", { variant: "outline" });
popover(popBtn, () => h("div", { class: "flex flex-col gap-2" }, h("div", { class: "text-sm font-medium" }, "Display options"), h("p", { class: "text-muted-foreground text-sm" }, "Popovers hold arbitrary content."), input({ placeholder: "Focused input" })));

/* dialogs */
const dlg = dialog({ title: "Weight viewer", description: "conv1.weight · float32 · 64×3×7×7", content: h("div", { class: "text-sm" }, "Dialog body. Press Escape or click outside to close."), footer: [button("Close", { variant: "outline", onClick: () => dlg.close() }), button("Export CSV")] });
const sh = sheet({ title: "Conv · conv1", storageKey: "gallery-sheet", width: 380 });
sh.setContent(h("div", { class: "flex flex-col gap-3 p-4 text-sm" }, h("div", { class: "text-muted-foreground" }, "Docked, resizable inspector. Drag its left edge."), accordion([
  { value: "a", title: "Attributes", content: "kernel_shape = [7, 7]\nstrides = [2, 2]", open: true }, { value: "b", title: "Inputs", content: "X, W" }, { value: "c", title: "Outputs", content: "Y" }]),
  collapsible({ title: "Advanced" }, h("p", { class: "text-muted-foreground text-sm" }, "Collapsible body")).el));
const items: CommandItem[] = Array.from({ length: 40 }, (_, i) => ({ id: `n${i}`, label: ["Conv", "Relu", "Add", "MaxPool", "BatchNormalization"][i % 5] + `_${i}`, hint: i % 2 ? "64×56×56" : "1.2M params", group: i < 20 ? "Layers" : "Tensors", swatch: `var(--cat-${cats[i % 5 === 0 ? 0 : i % 5 === 1 ? 1 : 4]})`, onSelect: (it) => toast({ title: "Selected", description: it.label }) }));
const palette = commandDialog({ items, placeholder: "Find layer, tensor or op…" });
const dialogs = row(button("Dialog", { onClick: () => dlg.open() }), button("Side panel", { variant: "outline", icon: PanelRight, onClick: () => (sh.isOpen() ? sh.close() : sh.open()) }),
  button("Command palette", { variant: "outline", icon: Search, onClick: () => palette.open() }), button("Confirm", { variant: "outline", onClick: async () => toast({ title: (await confirmDialog("Discard model?", "Unsaved view state will be lost.", "Discard")) ? "Discarded" : "Cancelled" }) }),
  button("Toast", { variant: "outline", onClick: () => toast({ title: "Model loaded", description: "resnet18-v2-7.onnx · 69 layers", variant: "success" }) }),
  button("Error toast", { variant: "outline", onClick: () => toast({ title: "Parse error", description: "Unsupported IR version", variant: "destructive" }) }));
document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "k") { e.preventDefault(); palette.open(); } });
const inlineCmd = command({ items: items.slice(0, 8), class: "h-72 w-80 rounded-lg border shadow-sm" });

/* virtual */
const vl = createVirtualList({ rowHeight: 28, count: 1_000_000, render: (i) => h("div", { class: "hover:bg-muted/50 flex items-center border-b px-3 font-mono text-xs" }, `row ${i.toLocaleString()}`), class: "h-56 w-64 rounded-md border" });
const vt = createVirtualTable({ columns: [{ header: "#", width: 70, align: "right" }, { header: "c0", width: 110, align: "right" }, { header: "c1", width: 110, align: "right" }, { header: "c2", width: 110, align: "right" }, { header: "c3", width: 110, align: "right" }, { header: "c4", width: 110, align: "right" }], count: 200_000, cell: (r, c) => (c === 0 ? String(r) : (Math.sin(r * 12.9898 + c * 78.233) * 3).toFixed(4)), class: "h-56 w-[420px] rounded-md border" });
const panels = resizablePanels({ a: h("div", { class: "p-3 text-sm" }, "Pane A (drag the divider)"), b: h("div", { class: "p-3 text-sm" }, "Pane B"), size: 160, class: "h-24 w-[420px] rounded-md border" });

const themeBtn = button(null, { variant: "outline", size: "icon", icon: resolvedTheme() === "dark" ? Sun : Moon, ariaLabel: "Toggle theme", onClick: () => { setTheme(getTheme() === "dark" || resolvedTheme() === "dark" ? "light" : "dark"); } });
window.addEventListener("themechange", () => themeBtn.replaceChildren(icon(resolvedTheme() === "dark" ? Sun : Moon)));

document.getElementById("app")!.append(
  h("div", { class: "mx-auto flex max-w-5xl flex-col gap-10 px-6 py-8" },
    h("header", { class: "flex items-center justify-between" }, h("div", null, h("h1", { class: "text-2xl font-semibold tracking-tight" }, "bonnx UI kit"), h("p", { class: "text-muted-foreground text-sm" }, "shadcn/ui re-implemented in vanilla TypeScript · Tailwind v4")), themeBtn),
    section("Buttons", buttons), section("Badges", badges), section("Form controls", form),
    section("Toggle groups", row(tg.el, tgm.el)), section("Tabs", tabsCtl.el),
    section("Breadcrumb", breadcrumb([{ label: "main" }, { label: "If_12 · then_branch" }, { label: "Loop_3 · body" }])),
    section("Cards · alerts · loading", cards), section("Table", h("div", { class: "w-96 rounded-md border" }, tbl)),
    section("Menus & popovers", row(dd, popBtn, ctxBox)), section("Overlays", dialogs),
    section("Command (inline)", inlineCmd.el), section("Virtualized (1M rows · 200k×6 table)", row(vl.el, vt.el)),
    section("Resizable", panels.el), section("Scroll area", scrollArea("h-24 w-64 rounded-md border p-3 text-sm", ...Array.from({ length: 20 }, (_, i) => h("div", null, `Line ${i + 1}`)))),
    section("Empty state", emptyState({ icon: Upload, title: "Drop an ONNX model here", description: "or use Open (Ctrl+O). Everything stays in your browser.", action: button("Choose file", { icon: Upload }) })),
    section("Graph tokens", tokenSwatches), separator(),
  ),
);
if (q.get("open")) {
  const k = q.get("open")!;
  const at = (el: HTMLElement, f: () => void) => () => { const sec = el.closest("section")!; sec.parentElement!.prepend(sec); setTimeout(f, 50); };
  const acts: Record<string, () => void> = {
    dialog: () => dlg.open(), sheet: () => sh.open(), palette: () => { palette.open(); palette.command.setQuery("conv"); },
    toast: () => toast({ title: "Model loaded", description: "resnet18-v2-7.onnx", variant: "success", duration: 0 }),
    dropdown: at(dd, () => dd.click()), select: at(selectCtl.el, () => selectCtl.el.click()), popover: at(popBtn, () => popBtn.click()),
    tooltip: at(tt, () => tt.dispatchEvent(new PointerEvent("pointerenter"))),
    ctx: at(ctxBox, () => { const r = ctxBox.getBoundingClientRect(); ctxBox.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, clientX: r.left + 40, clientY: r.top + 30 })); }),
  };
  setTimeout(acts[k] ?? (() => {}), 150);
}
