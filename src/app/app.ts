import { h, icon, clear } from "../lib/dom";
import {
  AlertCircle, ArrowDown, ArrowUp, Boxes, ChevronRight, Download, FileUp, FolderOpen, Info, Keyboard, Link as LinkIcon,
  Maximize, Minus, Moon, Network, PanelRight, Plus, Search, Settings, Sun, Monitor, Workflow, X,
} from "../lib/icons";
import {
  badge, button, commandDialog, contextMenu, dialog, dropdownMenu, kbd, kbdGroup, popover, progress, select, sheet, showMenu, switchControl, toast, toggleGroup, tooltip, spinner,
  type CommandItem, type MenuItem,
} from "../ui";
import { cycleTheme, getTheme, resolvedTheme } from "../lib/theme";
import { fmtCount, fmtShape } from "../lib/format";
import type { GraphView, ModelApi, ModelView, TensorInfo } from "../onnx/types";
import { createModelClient } from "../workers/client";
import { cancelLayout, LayoutCancelled } from "../workers/layout-client";
import { prepareScene } from "../render/prepare";
import { defaultSceneOptions, OP_CATEGORIES, type RNode, type Scene, type SceneOptions } from "../render/scene";
import { COLOR_MODES, type ColorMode } from "../render/colors";
import { GraphViewer, type ViewState } from "../render/viewer";
import { Minimap } from "../render/minimap";
import { downloadBlob, exportPNG, exportSVG, svgBlob } from "../render/export";
import { Inspector } from "./inspector";
import { createTensorPanel, openTensorViewer } from "./tensor";
import { filesFromDrop, installDropZone, loadEmbedded, loadFromDesktopPath, loadFromFiles, loadFromUrl, pickFiles, type LoadedInput } from "./loader";

/* ───────── persisted settings ───────── */

interface Settings {
  scene: SceneOptions;
  color: ColorMode;
  grid: boolean;
  inspector: boolean;
}
const SKEY = "onnxviz-settings";
function loadSettings(): Settings {
  const d: Settings = { scene: { ...defaultSceneOptions }, color: "op", grid: true, inspector: true };
  try {
    const s = JSON.parse(localStorage.getItem(SKEY) ?? "null") as Partial<Settings> | null;
    if (s) return { ...d, ...s, scene: { ...d.scene, ...s.scene } };
  } catch { /* ignore */ }
  return d;
}

const SHORTCUTS: [string, string][] = [
  ["/  or  Ctrl+K", "Search nodes, tensors and operators"], ["F", "Fit graph to window"], ["+  /  −", "Zoom in / out"], ["Arrow keys", "Move selection along the graph"],
  ["Enter", "Open the selected node's subgraph or function"], ["Backspace  or  Alt+←", "Back to the previous graph"], ["Esc", "Deselect / close"],
  ["M", "Model summary"], ["I", "Toggle inspector panel"], ["D", "Cycle color mode"], ["L", "Toggle layout direction"], ["N", "Toggle node names"], ["A", "Toggle attributes"],
  ["C", "Copy selected node name"], ["E", "Export SVG"], ["T", "Cycle theme"], ["?", "This help"], ["Ctrl+O", "Open a model"],
];

export class App {
  private settings = loadSettings();
  private api: (ModelApi & { terminate(): void }) | null = null;
  private model: ModelView | null = null;
  private graphId = 0;
  private history: { graph: number; view: ViewState | null }[] = [];
  private viewStates = new Map<number, ViewState>();
  private gen = 0;
  private selectedRef: { kind: RNode["kind"]; ref: number } | null = null;

  /* dom */
  private header!: HTMLElement; private stage!: HTMLElement; private main!: HTMLElement;
  private crumb!: HTMLElement; private titleEl!: HTMLElement; private zoomLabel!: HTMLElement;
  private legendEl!: HTMLElement; private miniHost!: HTMLElement; private matchBar!: HTMLElement;
  private welcome!: HTMLElement; private loading!: HTMLElement; private dropOverlay!: HTMLElement;
  private loadBar!: ReturnType<typeof progress>; private loadText!: HTMLElement;
  private viewer!: GraphViewer; private minimap!: Minimap; private inspector: Inspector | null = null;
  private side!: ReturnType<typeof sheet>;
  private palette!: ReturnType<typeof commandDialog>;
  private busy = false;
  private colorSel!: ReturnType<typeof select>;
  private toolsEnabled: HTMLButtonElement[] = [];

  constructor(private root: HTMLElement) {}

  /* ═══════════════════════ boot & layout ═══════════════════════ */

  async boot(): Promise<void> {
    this.buildShell();
    installDropZone((f) => void this.onFiles(f), (a) => this.dropOverlay.classList.toggle("hidden", !a));
    addEventListener("keydown", (e: KeyboardEvent) => this.onKey(e));
    addEventListener("themechange", () => this.refreshThemeButton());
    const qs = new URLSearchParams(location.search);
    const emb = loadEmbedded();
    const desk = window.onnxvizDesktop;
    if (desk) {
      desk.onOpenPath((p) => void this.openInput(() => loadFromDesktopPath(p)).catch((e) => this.fail(e)));
      const first = await desk.ready();
      if (first) await this.openInput(() => loadFromDesktopPath(first)).catch((e) => this.fail(e));
    }
    try {
      if (emb) await this.openInput(emb);
      else if (qs.get("url")) await this.openInput(() => loadFromUrl(qs.get("url")!, (p) => this.progress(p.stage, p.frac), qs.get("external") ?? undefined));
    } catch (e) { this.fail(e); }
  }

  private buildShell(): void {
    const s = this.settings;
    this.root.className = "bg-background text-foreground relative flex h-dvh w-full flex-col overflow-hidden";

    /* header */
    this.titleEl = h("div", { class: "flex min-w-0 items-center gap-2" });
    this.crumb = h("nav", { "aria-label": "Graph path", class: "flex min-w-0 shrink items-center gap-1 text-sm" });
    const searchBtn = h("button", { type: "button", class: "text-muted-foreground bg-background hover:bg-accent dark:bg-input/30 border-input hidden h-8 w-36 shrink-0 items-center gap-2 rounded-md border px-3 text-sm shadow-xs transition-colors md:flex xl:w-64", onclick: () => this.openSearch() },
      icon(Search, "size-4"), h("span", { class: "flex-1 truncate text-left" }, "Find node, tensor or op"), kbdGroup("Ctrl+K"));
    searchBtn.setAttribute("aria-label", "Search");
    const searchIcon = button(null, { variant: "ghost", size: "icon", icon: Search, ariaLabel: "Search", title: "Search ( / )", class: "md:hidden", onClick: () => this.openSearch() });

    this.colorSel = select(COLOR_MODES.map((c) => ({ value: c.key, label: c.label })), { value: s.color, size: "sm", ariaLabel: "Color nodes by", class: "w-36", onChange: (v) => this.setColor(v as ColorMode) });
    const viewBtn = button("View", { variant: "outline", size: "sm", icon: Settings });
    popover(viewBtn, () => this.viewPanel(), { align: "end", class: "w-80 p-4" });
    this.zoomLabel = h("button", { type: "button", class: "text-muted-foreground hover:text-foreground w-12 text-center text-xs tabular-nums", onclick: () => this.viewer?.fit("auto") }, "100%");
    tooltip(this.zoomLabel, "Fit to window ( F )");
    const zoom = h("div", { class: "bg-background dark:bg-input/30 border-input hidden items-center rounded-md border shadow-xs sm:flex" },
      button(null, { variant: "ghost", size: "icon-sm", icon: Minus, ariaLabel: "Zoom out", title: "Zoom out ( − )", onClick: () => this.viewer?.zoomOut() }), this.zoomLabel,
      button(null, { variant: "ghost", size: "icon-sm", icon: Plus, ariaLabel: "Zoom in", title: "Zoom in ( + )", onClick: () => this.viewer?.zoomIn() }),
      button(null, { variant: "ghost", size: "icon-sm", icon: Maximize, ariaLabel: "Fit to window", title: "Fit ( F )", onClick: () => this.viewer?.fit("auto") }));
    const exportBtn = button(null, { variant: "outline", size: "icon", icon: Download, ariaLabel: "Export", title: "Export image ( E )" });
    dropdownMenu(exportBtn, () => this.exportItems(), { align: "end" });
    const infoBtn = button(null, { variant: "ghost", size: "icon", icon: Info, ariaLabel: "Model summary", title: "Model summary ( M )", onClick: () => this.showModelPanel() });
    const panelBtn = button(null, { variant: "ghost", size: "icon", icon: PanelRight, ariaLabel: "Toggle inspector", title: "Inspector ( I )", onClick: () => this.toggleSide() });
    const themeBtn = button(null, { variant: "ghost", size: "icon", icon: Sun, ariaLabel: "Theme", title: "Theme ( T )", onClick: () => { cycleTheme(); this.refreshThemeButton(); } });
    themeBtn.dataset.role = "theme";
    const helpBtn = button(null, { variant: "ghost", size: "icon", icon: Keyboard, ariaLabel: "Keyboard shortcuts", title: "Shortcuts ( ? )", onClick: () => this.showHelp() });
    const openBtn = button("Open", { variant: "default", size: "sm", icon: FolderOpen, onClick: () => void this.openDialog() });
    dropdownMenu(openBtn, [
      { label: "Open file…", icon: FileUp, shortcut: "Ctrl+O", onSelect: () => void this.openDialog() },
      { label: "Open folder (with external data)…", icon: FolderOpen, onSelect: () => void this.openDialog(true) },
      { label: "Open from URL…", icon: LinkIcon, onSelect: () => this.urlDialog() },
    ], { align: "end" });
    this.toolsEnabled = [exportBtn, infoBtn, panelBtn, viewBtn];

    this.header = h("header", { class: "bg-background/95 relative z-40 flex h-12 shrink-0 items-center gap-2 border-b px-3 backdrop-blur" },
      h("div", { class: "flex shrink-0 items-center gap-2" },
        h("div", { class: "bg-primary text-primary-foreground flex size-7 shrink-0 items-center justify-center rounded-md" }, icon(Network, "size-4")),
        h("span", { class: "hidden text-sm font-semibold tracking-tight whitespace-nowrap sm:block" }, "ONNX Viz")),
      h("div", { class: "bg-border mx-1 hidden h-5 w-px sm:block" }), this.titleEl, this.crumb,
      h("div", { class: "flex-1" }), searchBtn, searchIcon, this.colorSel.el, viewBtn, zoom, exportBtn, infoBtn, panelBtn, helpBtn, themeBtn, openBtn);

    /* stage */
    this.main = h("main", { class: "relative min-h-0 flex-1 overflow-hidden" });
    this.stage = h("div", { class: "absolute inset-0" });
    this.miniHost = h("div", { class: "bg-card/90 absolute top-3 left-3 z-10 overflow-hidden rounded-md border shadow-sm backdrop-blur" });
    this.legendEl = h("div", { class: "bg-card/90 text-card-foreground absolute bottom-3 left-3 z-10 max-w-[min(26rem,calc(100%-1.5rem))] rounded-lg border p-3 text-xs shadow-sm backdrop-blur" });
    this.matchBar = h("div", { class: "bg-popover text-popover-foreground absolute bottom-4 left-1/2 z-20 hidden -translate-x-1/2 items-center gap-2 rounded-full border py-1 pr-1 pl-4 text-sm shadow-lg" });
    this.dropOverlay = h("div", { class: "bg-background/80 border-primary absolute inset-3 z-50 hidden items-center justify-center rounded-xl border-2 border-dashed backdrop-blur-sm" },
      h("div", { class: "text-center" }, icon(FileUp, "text-primary mx-auto mb-3 size-10"), h("div", { class: "text-lg font-semibold" }, "Drop an ONNX model"), h("div", { class: "text-muted-foreground text-sm" }, "Drop a folder to include external weight files")));
    this.loadBar = progress(null);
    this.loadText = h("div", { class: "text-sm font-medium" });
    this.loading = h("div", { class: "bg-background/70 absolute inset-0 z-40 hidden items-center justify-center backdrop-blur-sm" },
      h("div", { class: "bg-card flex w-80 flex-col gap-3 rounded-xl border p-5 shadow-lg" }, h("div", { class: "flex items-center gap-2" }, spinner("size-4"), this.loadText), this.loadBar.el));
    this.welcome = this.buildWelcome();
    this.main.append(this.stage, this.miniHost, this.legendEl, this.matchBar, this.welcome, this.loading, this.dropOverlay);
    this.root.append(this.header, this.main);

    this.viewer = new GraphViewer(this.stage, { colorMode: s.color, showGrid: s.grid });
    this.minimap = new Minimap(this.miniHost, this.viewer, { maxWidth: 150, maxHeight: 230 });
    this.side = sheet({ title: "Inspector", mount: this.main, width: 400, storageKey: "onnxviz-side", onOpenChange: (o) => this.onSideChange(o), onResize: (w) => this.viewer.setInsets({ right: w }) });
    this.side.setTitle("Inspector");
    this.bindViewer();
    this.palette = commandDialog({ placeholder: "Search nodes, tensors, operators…", filter: false, maxItems: 120, onQuery: (q) => this.runSearch(q), onSelect: (it) => it.onSelect?.(it) });
    this.setEmptyMode(true);
    this.refreshThemeButton();
  }

  private buildWelcome(): HTMLElement {
    const urlIn = h("input", { type: "url", placeholder: "https://…/model.onnx", class: "border-input bg-background dark:bg-input/30 placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 h-9 w-full min-w-0 rounded-md border px-3 text-sm shadow-xs outline-none focus-visible:ring-[3px]" }) as HTMLInputElement;
    const go = () => { if (urlIn.value.trim()) void this.openInput(() => loadFromUrl(urlIn.value.trim(), (p) => this.progress(p.stage, p.frac))); };
    urlIn.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    return h("div", { class: "bg-background absolute inset-0 z-30 flex items-center justify-center overflow-auto p-6" },
      h("div", { class: "w-full max-w-xl text-center" },
        h("div", { class: "bg-primary/10 text-primary mx-auto mb-5 flex size-14 items-center justify-center rounded-2xl" }, icon(Boxes, "size-7")),
        h("h1", { class: "text-3xl font-semibold tracking-tight" }, "Visualize any ONNX model"),
        h("p", { class: "text-muted-foreground mx-auto mt-2 max-w-md text-sm" }, "Drop a model anywhere, open a file or folder, or paste a URL. Everything is parsed locally in your browser — models never leave your machine."),
        h("div", { class: "mt-6 flex flex-wrap items-center justify-center gap-2" },
          button("Open model", { size: "lg", icon: FileUp, onClick: () => void this.openDialog() }),
          button("Open folder", { size: "lg", variant: "outline", icon: FolderOpen, onClick: () => void this.openDialog(true) })),
        h("div", { class: "mx-auto mt-4 flex max-w-md items-center gap-2" }, urlIn, button("Load", { variant: "secondary", onClick: go })),
        h("div", { id: "welcome-error", class: "mt-4 hidden text-left" }),
        h("div", { class: "text-muted-foreground mt-8 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-xs" },
          h("span", null, ".onnx"), h("span", null, ".onnx.prototxt"), h("span", null, "protobuf JSON"), h("span", null, "subgraphs & functions"), h("span", null, "external data"))));
  }

  private setEmptyMode(empty: boolean): void {
    this.welcome.classList.toggle("hidden", !empty);
    this.miniHost.classList.toggle("hidden", empty);
    this.legendEl.classList.toggle("hidden", empty);
    for (const b of this.toolsEnabled) b.disabled = empty;
    this.colorSel.el.disabled = empty;
    if (empty) { this.side.close(); this.titleEl.replaceChildren(); this.crumb.replaceChildren(); document.title = "ONNX Viz"; }
  }

  /* ═══════════════════════ loading ═══════════════════════ */

  private progress(stage: string, frac: number): void {
    this.loading.classList.remove("hidden"); this.loading.classList.add("flex");
    this.loadText.textContent = stage; this.loadBar.set(Math.round(frac * 100));
  }
  private hideLoading(): void { this.loading.classList.add("hidden"); this.loading.classList.remove("flex"); }

  private fail(e: unknown): void {
    this.hideLoading(); this.busy = false;
    const msg = e instanceof Error ? e.message : String(e);
    console.error(e);
    const box = this.welcome.querySelector("#welcome-error") as HTMLElement;
    box.classList.remove("hidden"); clear(box);
    box.append(h("div", { class: "border-destructive/50 text-destructive bg-destructive/5 flex gap-2 rounded-lg border p-3 text-sm" }, icon(AlertCircle, "mt-0.5 size-4"), h("div", null, h("div", { class: "font-medium" }, "Could not open the model"), h("div", { class: "text-destructive/80 mt-0.5 break-words" }, msg))));
    if (!this.model) this.setEmptyMode(true); else toast({ title: "Could not open the model", description: msg, variant: "destructive", duration: 8000 });
  }

  private async onFiles(files: { path: string; file: File }[]): Promise<void> {
    if (!files.length) return;
    // extra non-model files dropped while a model is open → external tensor data
    const isModel = (n: string) => /\.(onnx|prototxt|pbtxt|json|ort)$/i.test(n);
    if (this.model && this.api && files.every((f) => !isModel(f.file.name))) return void this.provideExternal(files);
    await this.openInput(() => loadFromFiles(files, (p) => this.progress(p.stage, p.frac)));
  }

  private async provideExternal(files: { path: string; file: File }[]): Promise<void> {
    if (!this.api || !this.model) return;
    let n = 0;
    for (const f of files) {
      this.progress(`Reading ${f.file.name}`, 0.3);
      const tensors = await this.api.provideExternal(f.path, await f.file.arrayBuffer());
      for (const t of tensors) this.model.tensors[t.id] = t;
      n++;
    }
    this.hideLoading();
    toast({ title: "External data added", description: `${n} file${n > 1 ? "s" : ""} provided.`, variant: "success" });
    this.reselect();
  }

  async openDialog(directory = false): Promise<void> {
    const files = await pickFiles({ directory });
    if (files.length) await this.onFiles(files);
  }

  private urlDialog(): void {
    const input = h("input", { type: "url", placeholder: "https://…/model.onnx", class: "border-input bg-background h-9 w-full rounded-md border px-3 text-sm outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50" }) as HTMLInputElement;
    const d = dialog({ title: "Open from URL", description: "The server must allow cross-origin requests (CORS).", content: input, size: "sm",
      footer: [button("Open", { onClick: () => { d.close(); if (input.value.trim()) void this.openInput(() => loadFromUrl(input.value.trim(), (p) => this.progress(p.stage, p.frac))); } })] });
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { d.close(); void this.openInput(() => loadFromUrl(input.value.trim(), (p) => this.progress(p.stage, p.frac))); } });
    d.open(); input.focus();
  }

  /** load a model from an input or an async producer of one */
  async openInput(src: LoadedInput | (() => Promise<LoadedInput>)): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    const my = ++this.gen;
    try {
      this.progress("Reading…", 0.02);
      const input = typeof src === "function" ? await src() : src;
      if (my !== this.gen) return;
      this.api?.terminate(); cancelLayout();
      this.api = createModelClient();
      this.model = null; this.selectedRef = null; this.history = []; this.viewStates.clear();
      const model = await this.api.open({ name: input.name, data: input.data, external: input.external }, (p) => this.progress(p.stage, 0.3 + 0.6 * p.frac));
      if (my !== this.gen) return;
      this.model = model;
      this.inspector = new Inspector(this.inspectorHost());
      this.side.setContent(this.inspector.el);
      this.progress("Laying out graph…", 0.95);
      this.welcome.querySelector("#welcome-error")?.classList.add("hidden");
      this.setEmptyMode(false);
      this.renderTitle();
      await this.showGraph(0);
      this.hideLoading(); this.busy = false;
      if (this.settings.inspector && innerWidth >= 1000) { this.showModelPanel(); this.viewer.fit("auto", false); } else this.inspector.showModel();
      const missing = model.tensors.filter((t) => t.external && !t.available);
      if (missing.length) {
        const files = [...new Set(missing.map((t) => t.external!.location))];
        toast({ title: `${missing.length} tensors use external data`, description: `Drop ${files.slice(0, 3).join(", ")}${files.length > 3 ? ` and ${files.length - 3} more` : ""} onto this window (or reopen the folder) to see their values.`, duration: 12000 });
      }
      for (const w of model.warnings.slice(0, 1)) toast({ title: "Model opened with warnings", description: w + (model.warnings.length > 1 ? ` (+${model.warnings.length - 1} more, see Model summary)` : ""), duration: 7000 });
    } catch (e) { this.fail(e); } finally { this.busy = false; }
  }

  /* ═══════════════════════ graph pipeline ═══════════════════════ */

  private async showGraph(id: number, o: { keepView?: boolean; select?: { kind: RNode["kind"]; ref: number } | null; focusNode?: number } = {}): Promise<void> {
    const model = this.model; if (!model) return;
    const my = ++this.gen;
    let scene: Scene;
    try { scene = await prepareScene(model, id, this.settings.scene); } catch (e) { if (e instanceof LayoutCancelled) return; throw e; }
    if (my !== this.gen && !this.busy) return;
    this.graphId = id;
    const saved = !o.keepView && o.focusNode == null ? this.viewStates.get(id) : undefined;
    const selectRef = o.focusNode ?? (o.select?.kind === "op" || o.select?.kind === "fn" ? o.select.ref : undefined);
    this.viewer.setScene(scene, { keepView: o.keepView || !!saved, selectRef });
    if (saved) this.viewer.setViewState(saved);
    if (o.focusNode != null) this.viewer.select(this.viewer.sceneIdOfNode(o.focusNode), { focus: "zoom", animate: false });
    else if (o.select && selectRef == null) this.reselect();
    this.renderCrumb(); this.renderLegend();
    if (!this.selectedRef) this.inspectorIdle();
  }

  private relayout(): void { this.viewStates.clear(); void this.showGraph(this.graphId, { keepView: true, select: this.selectedRef }); }

  openGraph(graph: number, focusNode?: number): void {
    if (graph === this.graphId && focusNode == null) return;
    this.history.push({ graph: this.graphId, view: this.viewer.getViewState() });
    this.viewStates.set(this.graphId, this.viewer.getViewState());
    this.selectedRef = null;
    void this.showGraph(graph, { focusNode });
  }

  private goBack(): void {
    const prev = this.history.pop();
    if (!prev) { const p = this.model?.graphs[this.graphId]?.parent; if (p) void this.showGraph(p.graph, { focusNode: p.node }); return; }
    this.selectedRef = null;
    if (prev.view) this.viewStates.set(prev.graph, prev.view);
    void this.showGraph(prev.graph);
  }

  private graphLabel(g: GraphView): string {
    return g.kind === "function" ? `${g.fnName}${g.fnDomain ? ` (${g.fnDomain})` : ""}` : g.kind === "main" ? "main graph" : g.name || "subgraph";
  }

  private renderCrumb(): void {
    const m = this.model; if (!m) return;
    const chain: GraphView[] = [];
    for (let g: GraphView | undefined = m.graphs[this.graphId]; g; g = g.parent ? m.graphs[g.parent.graph] : undefined) chain.unshift(g);
    if (chain[0]!.kind === "function") chain.unshift(m.graphs[0]!);
    clear(this.crumb);
    this.titleEl.querySelectorAll("[data-stat]").forEach((b) => b.classList.toggle("hidden!", chain.length > 1));
    if (chain.length < 2) return;
    chain.forEach((g, i) => {
      if (i) this.crumb.append(icon(ChevronRight, "text-muted-foreground size-3.5"));
      const last = i === chain.length - 1;
      const parentAttr = g.parent ? m.graphs[g.parent.graph]!.nodes[g.parent.node] : null;
      const label = g.kind === "subgraph" && parentAttr ? `${parentAttr.op} › ${g.parent!.attr}` : this.graphLabel(g);
      this.crumb.append(last ? h("span", { class: "truncate font-medium" }, label) : h("button", { type: "button", class: "text-muted-foreground hover:text-foreground max-w-40 truncate hover:underline", onclick: () => this.openGraph(g.id, g.parent?.node) }, label));
    });
  }

  private renderTitle(): void {
    const m = this.model!.meta;
    document.title = `${m.file} – ONNX Viz`;
    this.titleEl.replaceChildren(h("span", { class: "max-w-56 truncate text-sm font-medium", title: m.file }, m.file),
      badge(`${fmtCount(m.nodeCount)} nodes`, { variant: "secondary", class: "hidden lg:inline-flex" }));
    this.titleEl.lastElementChild?.setAttribute("data-stat", "");
    if (m.params) { const b = badge(`${fmtCount(m.params)} params`, { variant: "secondary", class: "hidden xl:inline-flex" }); b.setAttribute("data-stat", ""); this.titleEl.append(b); }
  }

  private renderLegend(): void {
    const lg = this.viewer.getLegend(); clear(this.legendEl);
    if (!lg) return void this.legendEl.classList.add("hidden");
    this.legendEl.classList.remove("hidden");
    this.legendEl.append(h("div", { class: "mb-1.5 font-medium" }, lg.title));
    if (lg.kind === "categorical") {
      this.legendEl.append(h("div", { class: "flex flex-wrap gap-x-3 gap-y-1" }, ...lg.entries.map((e) => h("button", { type: "button", class: "hover:text-foreground text-muted-foreground inline-flex items-center gap-1.5", title: `Highlight ${e.label}`, onclick: () => this.highlightLegend(e.label) },
        h("i", { class: "size-2.5 rounded-[3px]", style: { background: e.color } }), e.label, h("span", { class: "tabular-nums opacity-70" }, e.count)))));
    } else if (lg.ramp) {
      const r = lg.ramp;
      this.legendEl.append(h("div", { class: "h-2 w-64 rounded-full", style: { background: `linear-gradient(90deg, ${r.stops.join(",")})` } }),
        h("div", { class: "text-muted-foreground mt-1 flex w-64 justify-between tabular-nums" }, h("span", null, r.lo), h("span", null, r.mid), h("span", null, r.hi)),
        ...(r.none ? [h("div", { class: "text-muted-foreground mt-1" }, `${r.none} nodes without a value`)] : []));
    }
  }

  private highlightLegend(label: string): void {
    const sc = this.viewer.scene; if (!sc || this.settings.color !== "op") return;
    const ids = sc.nodes.filter((n) => (n.kind === "op" || n.kind === "fn") && OP_CATEGORIES[n.cat]?.label === label).map((n) => n.id);
    if (ids.length) this.showMatches(ids, label);
  }

  /* ═══════════════════════ viewer events ═══════════════════════ */

  private bindViewer(): void {
    const v = this.viewer;
    v.on("viewchange", (s) => { this.zoomLabel.textContent = `${Math.round(s.scale * 100)}%`; });
    v.on("select", ({ scene }) => this.onSelect(scene));
    v.on("edge-select", (e) => { if (e.name && this.inspector) { this.selectedRef = null; this.inspector.showValue(this.graphId, e.name); this.ensureSide(); } });
    v.on("open-subgraph", (e) => this.openGraph(e.graph));
    v.on("contextmenu", (e) => this.onContext(e));
    v.on("scene", () => this.renderLegend());
  }

  private onSelect(n: RNode | null): void {
    if (!this.inspector) return;
    if (!n) { this.selectedRef = null; this.inspectorIdle(); return; }
    this.selectedRef = { kind: n.kind, ref: n.ref };
    this.reselect(); this.ensureSide();
  }

  private reselect(): void {
    const r = this.selectedRef, ins = this.inspector; if (!r || !ins) return;
    this.side.setTitle(r.kind === "op" || r.kind === "fn" ? this.model!.graphs[this.graphId]!.nodes[r.ref]?.op ?? "Node" : r.kind === "const" ? "Constant" : r.kind === "ghost" ? "Value" : r.kind === "input" ? "Graph input" : "Graph output");
    if (r.kind === "op" || r.kind === "fn") ins.showNode(this.graphId, r.ref);
    else if (r.kind === "input") ins.showIO(this.graphId, "input", r.ref);
    else if (r.kind === "output") ins.showIO(this.graphId, "output", r.ref);
    else if (r.kind === "const") ins.showConst(r.ref);
    else if (r.kind === "ghost") { const sel = this.viewer.scene?.nodes.find((n) => n.kind === "ghost" && n.ref === r.ref); if (sel) ins.showValue(this.graphId, sel.name); }
  }

  private inspectorIdle(): void {
    this.side.setTitle(this.model ? "Model" : "Inspector");
    if (this.inspector && this.model) this.inspector.showModel();
  }

  private onContext(e: { node: number | null; edge: number | null; x: number; y: number }): void {
    const sc = this.viewer.scene; if (!sc) return;
    const items: MenuItem[] = [];
    if (e.node != null) {
      const n = sc.nodes[e.node]!;
      items.push({ type: "label", label: n.title || n.name });
      items.push({ label: "Copy name", shortcut: "C", onSelect: () => this.copy(n.name) });
      if (n.op && (n.kind === "op" || n.kind === "fn")) {
        items.push({ label: "Copy operator type", onSelect: () => this.copy(n.op) });
        items.push({ label: `Find all ${n.op} nodes`, onSelect: () => this.findOp(n.op) });
      }
      for (const s of n.subgraphs) items.push({ label: `Open ${s.attr}`, icon: Network, onSelect: () => this.openGraph(s.graph) });
      if (n.fn != null) items.push({ label: "Open function", icon: Workflow, onSelect: () => this.openGraph(n.fn!) });
      items.push({ type: "separator" }, { label: "Zoom to node", onSelect: () => this.viewer.focusNode(n.id, { zoom: true, animate: true }) });
      items.push({ label: "Inspect", onSelect: () => { this.viewer.select(n.id); this.ensureSide(); } });
    } else {
      items.push({ label: "Fit to window", shortcut: "F", onSelect: () => this.viewer.fit("auto") }, { label: "Export SVG", onSelect: () => this.exportSvg() }, { label: "Model summary", shortcut: "M", onSelect: () => this.showModelPanel() });
    }
    showMenu(items, { x: e.x, y: e.y });
  }

  private copy(text: string): void {
    navigator.clipboard?.writeText(text).then(() => toast({ title: "Copied", description: text, duration: 1400 }), () => toast({ title: "Copy failed", variant: "destructive" }));
  }

  /* ═══════════════════════ inspector host ═══════════════════════ */

  private inspectorHost() {
    const self = this;
    return {
      api: this.api!, model: this.model!,
      currentGraph: () => this.graphId,
      jumpToNode: (id: number) => { const sid = this.viewer.sceneIdOfNode(id); if (sid != null) this.viewer.select(sid, { focus: "ensure", animate: true }); },
      openGraph: (g: number, f?: number) => this.openGraph(g, f),
      findOp: (op: string) => this.findOp(op),
      tensorPanel: (t: TensorInfo) => createTensorPanel(self.api!, t),
      openTensor: (t: TensorInfo) => openTensorViewer(self.api!, t),
      selectValue: (name: string) => {
        const sc = this.viewer.scene; const e = sc?.edges.find((x) => x.name === name);
        if (e) this.viewer.selectEdge(e.id); else this.inspector?.showValue(this.graphId, name);
      },
    };
  }

  private showModelPanel(): void { this.viewer.select(null); this.selectedRef = null; this.side.setTitle("Model"); this.inspector?.showModel(); this.ensureSide(); }
  private ensureSide(): void { if (!this.side.isOpen()) this.side.open(); }
  private toggleSide(): void { this.side.isOpen() ? this.side.close() : this.side.open(); }
  private onSideChange(open: boolean): void {
    this.viewer.setInsets({ right: open ? this.side.getWidth() : 0 });
    this.settings.inspector = open; this.saveSettings();
  }

  /* ═══════════════════════ search ═══════════════════════ */

  private index: { g: number; n: number; text: string; head: string; label: string; hint: string; tensor?: number }[] | null = null;
  private buildIndex() {
    const m = this.model!, idx: NonNullable<App["index"]> = [];
    for (const g of m.graphs) {
      g.nodes.forEach((n) => {
        const tx = [n.op, n.name, ...n.outputs.map((o) => o.name), ...n.inputs.map((i) => i.name), ...n.attrs.map((a) => (a.value.t === "string" ? a.value.v : ""))].join("\u0001").toLowerCase();
        idx.push({ g: g.id, n: n.id, text: tx, head: (n.op + "\u0001" + n.name).toLowerCase(), label: n.name, hint: `${n.op}${g.id ? " · " + this.graphLabel(g) : ""}` });
      });
      for (const tid of g.initializers) {
        const t = m.tensors[tid]!;
        idx.push({ g: g.id, n: -1, text: ("\u0002" + t.name).toLowerCase(), head: t.name.toLowerCase(), label: t.name, hint: `constant ⟨${fmtShape(t.dims)}⟩ ${t.dtype}`, tensor: tid });
      }
    }
    return (this.index = idx);
  }

  openSearch(prefill = ""): void {
    if (!this.model) return;
    this.palette.open();
    this.palette.command.setQuery(prefill);
    this.runSearch(prefill);
    this.palette.command.input.select();
  }

  private runSearch(q: string): void {
    const m = this.model; if (!m) return;
    const idx = this.index ?? this.buildIndex();
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    const items: CommandItem[] = [];
    if (!words.length) {
      items.push({ id: "model", label: "Model summary", icon: Info, group: "Actions", shortcut: "M", onSelect: () => { this.palette.close(); this.showModelPanel(); } });
      for (const [op, n] of m.meta.ops.slice(0, 40)) items.push({ id: "op:" + op, label: op, hint: `${n} nodes`, group: "Operators", onSelect: () => { this.palette.close(); this.findOp(op); } });
    } else {
      const cur: CommandItem[] = [], other: CommandItem[] = [], tens: CommandItem[] = [], weak: CommandItem[] = [];
      let matches = 0;
      for (const e of idx) {
        if (!words.every((w) => e.text.includes(w))) continue;
        matches++;
        const strong = words.every((w) => e.head.includes(w));
        const it: CommandItem = { id: `${e.g}:${e.n}:${e.tensor ?? ""}`, label: e.label, hint: e.hint, group: "", onSelect: () => { this.palette.close(); this.goTo(e); } };
        const bucket = e.tensor != null ? tens : e.g !== this.graphId ? other : strong ? cur : weak;
        if (bucket.length < 80) bucket.push({ ...it, group: e.tensor != null ? "Constants" : e.g === this.graphId ? (strong ? "In this graph" : "Connected to a matching tensor") : "Other graphs" });
      }
      if (matches && cur.length) items.push({ id: "hl", label: `Highlight ${cur.length >= 80 ? "all" : cur.length} matches in this graph`, icon: Search, group: "Actions", onSelect: () => { this.palette.close(); this.highlightQuery(words); } });
      items.push(...cur, ...tens, ...weak, ...other);
    }
    this.palette.command.setItems(items);
  }

  private goTo(e: { g: number; n: number; tensor?: number }): void {
    if (e.tensor != null) {
      const t = this.model!.tensors[e.tensor]!;
      // jump to first consumer if present, else open explorer
      const g = this.model!.graphs[e.g]!;
      const cons = g.values[t.name]?.consumers[0];
      if (cons != null) return void this.goTo({ g: e.g, n: cons });
      return void openTensorViewer(this.api!, t);
    }
    if (e.g !== this.graphId) { this.history.push({ graph: this.graphId, view: this.viewer.getViewState() }); void this.showGraph(e.g, { focusNode: e.n }); return; }
    const sid = this.viewer.sceneIdOfNode(e.n);
    if (sid != null) { this.viewer.select(sid, { focus: "zoom", animate: true }); this.ensureSide(); }
  }

  private highlightQuery(words: string[]): void {
    const g = this.model!.graphs[this.graphId]!, ids: number[] = [];
    for (const e of this.index!) if (e.g === g.id && e.n >= 0 && words.every((w) => e.head.includes(w))) { const s = this.viewer.sceneIdOfNode(e.n); if (s != null) ids.push(s); }
    this.showMatches(ids, words.join(" "));
  }

  findOp(op: string): void {
    const sc = this.viewer.scene; if (!sc) return;
    const ids = sc.nodes.filter((n) => n.op === op && (n.kind === "op" || n.kind === "fn")).map((n) => n.id);
    if (!ids.length) { toast({ title: `No ${op} nodes in this graph`, description: "Try the search to look across nested graphs." }); return; }
    this.showMatches(ids, op);
  }

  private showMatches(ids: number[], label: string): void {
    if (!ids.length) return void toast({ title: "No matches in this graph" });
    this.viewer.highlightMatches(ids, ids[0], true);
    const upd = () => { const i = this.viewer.activeMatchIndex; (this.matchBar.querySelector("[data-count]") as HTMLElement).textContent = `${i + 1} / ${this.viewer.matchCount}`; };
    clear(this.matchBar);
    this.matchBar.append(h("span", { class: "max-w-48 truncate font-medium", title: label }, label), h("span", { "data-count": "", class: "text-muted-foreground tabular-nums" }, ""),
      button(null, { variant: "ghost", size: "icon-sm", icon: ArrowUp, ariaLabel: "Previous match", onClick: () => { this.viewer.stepMatch(-1); upd(); } }),
      button(null, { variant: "ghost", size: "icon-sm", icon: ArrowDown, ariaLabel: "Next match", onClick: () => { this.viewer.stepMatch(1); upd(); } }),
      button(null, { variant: "ghost", size: "icon-sm", icon: X, ariaLabel: "Clear highlight", onClick: () => this.clearMatches() }));
    this.matchBar.classList.remove("hidden"); this.matchBar.classList.add("flex"); upd();
  }
  private clearMatches(): void { this.viewer.clearMatches(); this.matchBar.classList.add("hidden"); this.matchBar.classList.remove("flex"); }

  /* ═══════════════════════ view settings, export, help ═══════════════════════ */

  private saveSettings(): void { try { localStorage.setItem(SKEY, JSON.stringify(this.settings)); } catch { /* ignore */ } }

  private setColor(c: ColorMode): void { this.settings.color = c; this.saveSettings(); this.viewer.setColorMode(c); this.colorSel.set(c, true); this.renderLegend(); }

  private updScene(patch: Partial<SceneOptions>, relayout = true): void {
    Object.assign(this.settings.scene, patch); this.saveSettings();
    if (relayout) this.relayout(); else if ("showTypes" in patch) this.viewer.setEdgeLabels(patch.showTypes!);
  }

  private viewPanel(): HTMLElement {
    const so = this.settings.scene;
    const row = (label: string, ctl: Node, hint?: string) => h("div", { class: "flex items-center justify-between gap-4" }, h("div", null, h("div", { class: "text-sm" }, label), hint ? h("div", { class: "text-muted-foreground text-xs" }, hint) : null), ctl);
    const sw = (key: "showNames" | "showAttributes" | "showInputsOutputs") => switchControl({ checked: so[key], ariaLabel: key, onChange: (v) => this.updScene({ [key]: v }) }).el;
    return h("div", { class: "flex flex-col gap-3" },
      h("div", { class: "text-sm font-medium" }, "Graph view"),
      row("Direction", toggleGroup([{ value: "TB", label: "Vertical" }, { value: "LR", label: "Horizontal" }], { value: so.direction, variant: "outline", size: "sm", ariaLabel: "Layout direction", onChange: (v) => this.updScene({ direction: v[0] as "TB" | "LR" }) }).el),
      row("Initializers", select([{ value: "rows", label: "In node cards" }, { value: "nodes", label: "As nodes" }, { value: "hidden", label: "Hidden" }], { value: so.showInitializers, size: "sm", class: "w-36", ariaLabel: "Initializers", onChange: (v) => this.updScene({ showInitializers: v as SceneOptions["showInitializers"] }) }).el),
      row("Edge labels", select([{ value: "shape", label: "Shape / type" }, { value: "name", label: "Tensor name" }, { value: "both", label: "Name and type" }, { value: "none", label: "None" }], { value: so.showTypes, size: "sm", class: "w-36", ariaLabel: "Edge labels", onChange: (v) => this.updScene({ showTypes: v as SceneOptions["showTypes"] }, false) }).el),
      row("Attributes", sw("showAttributes"), "Show attributes in node cards"), row("Node names", sw("showNames"), "Show names below operator types"), row("Inputs and outputs", sw("showInputsOutputs")),
      row("Grid", switchControl({ checked: this.settings.grid, ariaLabel: "Grid", onChange: (v) => { this.settings.grid = v; this.saveSettings(); this.viewer.setOptions?.({ showGrid: v }); } }).el));
  }

  private exportItems(): MenuItem[] {
    return [
      { type: "label", label: "Export graph" }, { label: "SVG (vector)", shortcut: "E", onSelect: () => this.exportSvg() },
      { label: "PNG 1×", onSelect: () => void this.exportPng(1) }, { label: "PNG 2×", onSelect: () => void this.exportPng(2) }, { label: "PNG 4×", onSelect: () => void this.exportPng(4) },
      { type: "separator" }, { label: "SVG of selection neighborhood", disabled: !this.selectedRef, onSelect: () => this.exportSvg(true) },
    ];
  }
  private fileBase(): string { return (this.model?.meta.file ?? "graph").replace(/\.[^.]+$/, "") + (this.graphId ? `-${this.graphId}` : ""); }
  private exportSvg(neighborhood = false): void {
    const c = this.viewer.exportContext(); if (!c) return;
    downloadBlob(`${this.fileBase()}.svg`, svgBlob(exportSVG(c, { neighborhood: neighborhood ? 2 : undefined })));
  }
  private async exportPng(scale: number): Promise<void> {
    const c = this.viewer.exportContext(); if (!c) return;
    try { downloadBlob(`${this.fileBase()}.png`, await exportPNG(c, { scale })); } catch (e) { toast({ title: "PNG export failed", description: String(e), variant: "destructive" }); }
  }

  private showHelp(): void {
    const body = h("div", { class: "grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm" }, ...SHORTCUTS.flatMap(([k, d]) => [h("div", { class: "flex flex-wrap gap-1" }, ...k.split(/\s{2}or\s{2}/).map((x) => kbd(x))), h("div", { class: "text-muted-foreground" }, d)]));
    const d = dialog({ title: "Keyboard shortcuts", content: body, size: "md" });
    d.open();
  }

  private refreshThemeButton(): void {
    const b = this.header.querySelector('[data-role="theme"]'); if (!b) return;
    const t = getTheme(); const ic = t === "system" ? Monitor : resolvedTheme() === "dark" ? Moon : Sun;
    b.replaceChildren(icon(ic)); b.setAttribute("title", `Theme: ${t} ( T )`);
    queueMicrotask(() => { this.renderLegend(); });
  }

  /* ═══════════════════════ keyboard ═══════════════════════ */

  private onKey(e: KeyboardEvent): void {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (document.querySelector("dialog[open], [role=dialog][data-state=open]") && e.key !== "?") return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === "o") { e.preventDefault(); void this.openDialog(); return; }
    if (mod && e.key.toLowerCase() === "k") { e.preventDefault(); this.openSearch(); return; }
    if (mod || e.altKey && e.key !== "ArrowLeft") return;
    if (e.key === "?") { e.preventDefault(); this.showHelp(); return; }
    if (!this.model) return;
    const k = e.key;
    if (k === "/") { e.preventDefault(); this.openSearch(); }
    else if (k === "f" || k === "F") this.viewer.fit("auto");
    else if (k === "+" || k === "=") this.viewer.zoomIn();
    else if (k === "-" || k === "_") this.viewer.zoomOut();
    else if (k === "Escape") { this.viewer.select(null); this.viewer.selectEdge(null); this.clearMatches(); }
    else if (k === "m" || k === "M") this.showModelPanel();
    else if (k === "i" || k === "I") this.toggleSide();
    else if (k === "d" || k === "D") this.setColor(COLOR_MODES[(COLOR_MODES.findIndex((c) => c.key === this.settings.color) + 1) % COLOR_MODES.length]!.key);
    else if (k === "l" || k === "L") this.updScene({ direction: this.settings.scene.direction === "TB" ? "LR" : "TB" });
    else if (k === "n" || k === "N") this.updScene({ showNames: !this.settings.scene.showNames });
    else if (k === "a" || k === "A") this.updScene({ showAttributes: !this.settings.scene.showAttributes });
    else if (k === "e" || k === "E") this.exportSvg();
    else if (k === "t" || k === "T") { cycleTheme(); this.refreshThemeButton(); }
    else if (k === "c" || k === "C") { const n = this.viewer.selected >= 0 ? this.viewer.scene?.nodes[this.viewer.selected] : null; if (n) this.copy(n.name); }
    else if (k === "Backspace" || (e.altKey && k === "ArrowLeft")) { e.preventDefault(); this.goBack(); }
    else if (k === "Enter") this.viewer.openSelected();
    else if (k === "ArrowUp" || k === "ArrowDown" || k === "ArrowLeft" || k === "ArrowRight") { e.preventDefault(); this.viewer.navigate(k.slice(5).toLowerCase() as "up" | "down" | "left" | "right"); }
  }
}

