import type { ModelView } from "../../onnx/types";
import { cancelLayout, LayoutCancelled, lastLayoutMs } from "../../workers/layout-client";
import { COLOR_MODES, type ColorMode } from "../colors";
import { downloadBlob, exportPNG, exportSVG, svgBlob } from "../export";
import { Minimap } from "../minimap";
import { prepareScene } from "../prepare";
import { defaultSceneOptions, type SceneOptions } from "../scene";
import { notifyThemeChange } from "../theme";
import { GraphViewer } from "../viewer";
import { syntheticBig, syntheticSmall } from "./synthetic";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const params = new URLSearchParams(location.search);

const viewer = new GraphViewer($("stage"));
const mini = new Minimap($("mini"), viewer, { maxWidth: 150, maxHeight: 260 });
let model: ModelView = syntheticSmall();
let graphId = 0;
const trail: number[] = [];
const so: SceneOptions = { ...defaultSceneOptions };
let lastBuildMs = 0;

/* ── toolbar ── */
const bar = $("bar");
const el = <K extends keyof HTMLElementTagNameMap>(tag: K, props: Record<string, unknown> = {}, ...kids: (Node | string)[]): HTMLElementTagNameMap[K] => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...kids);
  return e;
};
const btn = (label: string, fn: () => void): HTMLButtonElement => bar.appendChild(el("button", { textContent: label, onclick: fn }));
const sel = (opts: [string, string][], value: string, fn: (v: string) => void): HTMLSelectElement => {
  const s = el("select", { onchange: () => fn(s.value) });
  for (const [v, l] of opts) s.append(el("option", { value: v, textContent: l, selected: v === value }));
  bar.append(s);
  return s;
};

sel([["small", "small resnet"], ["big", "3k nodes"], ["big10", "10k nodes"]], params.get("m") ?? "small", (v) => { load(v); });
sel(COLOR_MODES.map((c) => [c.key, c.label]), "op", (v) => viewer.setColorMode(v as ColorMode));
sel([["TB", "top→bottom"], ["LR", "left→right"]], so.direction, (v) => rebuild({ direction: v as "TB" | "LR" }));
sel([["rows", "consts: rows"], ["nodes", "consts: nodes"], ["hidden", "consts: hidden"]], so.showInitializers, (v) => rebuild({ showInitializers: v as SceneOptions["showInitializers"] }));
sel([["shape", "labels: shape"], ["name", "labels: name"], ["both", "labels: both"], ["none", "labels: none"]], so.showTypes, (v) => viewer.setEdgeLabels(v as SceneOptions["showTypes"]));
const chk = (label: string, key: "showNames" | "showAttributes"): void => {
  const c = el("input", { type: "checkbox", onchange: () => rebuild({ [key]: c.checked }) });
  bar.append(el("label", {}, c, label));
};
chk("names", "showNames"); chk("attrs", "showAttributes");
btn("−", () => viewer.zoomOut()); btn("+", () => viewer.zoomIn()); btn("fit", () => viewer.fit());
btn("SVG", () => { const c = viewer.exportContext(); if (c) downloadBlob("graph.svg", svgBlob(exportSVG(c))); });
btn("PNG", async () => { const c = viewer.exportContext(); if (c) downloadBlob("graph.png", await exportPNG(c, { scale: 2 })); });
btn("theme", () => { document.documentElement.classList.toggle("dark"); notifyThemeChange(); });
const search = bar.appendChild(el("input", { type: "search", placeholder: "search op/name…" }));
search.addEventListener("input", () => {
  const q = search.value.trim().toLowerCase(), s = viewer.scene;
  if (!q || !s) { viewer.clearMatches(); return; }
  viewer.highlightMatches(s.nodes.filter((n) => (n.op + " " + n.name).toLowerCase().includes(q)).map((n) => n.id));
});
search.addEventListener("keydown", (e) => { if (e.key === "Enter") viewer.stepMatch(e.shiftKey ? -1 : 1); });
const crumb = bar.appendChild(el("div", { id: "crumb" }));

/* ── pipeline ── */
let gen = 0;
async function show(id: number, keepView = false): Promise<void> {
  const my = ++gen;
  const t0 = performance.now();
  try {
    const scene = await prepareScene(model, id, so);
    if (my !== gen) return;
    lastBuildMs = performance.now() - t0;
    graphId = id;
    viewer.setScene(scene, { keepView });
    renderCrumb();
    updateStats();
  } catch (e) { if (!(e instanceof LayoutCancelled)) console.error(e); }
}
function rebuild(o: Partial<SceneOptions>): void { Object.assign(so, o); void show(graphId, true); }
function load(kind: string): void {
  cancelLayout();
  model = kind === "big" ? syntheticBig(3000) : kind === "big10" ? syntheticBig(10000) : syntheticSmall();
  trail.length = 0;
  void show(0);
}
function renderCrumb(): void {
  crumb.replaceChildren(...[...trail, graphId].map((g, i, a) => el("button", { textContent: model.graphs[g].name, disabled: i === a.length - 1, onclick: () => { trail.length = i; void show(g); } })));
}
viewer.on("open-subgraph", (e) => { trail.push(graphId); void show(e.graph); });

/* ── stats + tooltip ── */
const stats = $("stats");
function updateStats(): void {
  const s = viewer.scene, f = viewer.stats;
  stats.textContent = `${s ? s.nodes.length : 0} nodes · ${s ? s.edges.length : 0} edges · build+layout ${lastBuildMs.toFixed(0)} ms (worker ${lastLayoutMs.toFixed(0)})\nframe ${f.ms.toFixed(2)} ms · drawn ${f.nodes}n/${f.edges}e · zoom ${f.scale.toFixed(3)}`;
}
viewer.on("frame", () => { if (!statsQueued) { statsQueued = true; setTimeout(() => { statsQueued = false; updateStats(); }, 200); } });
let statsQueued = false;
const tip = $("tip");
viewer.on("hover", (e) => {
  const n = e.node != null ? viewer.scene?.nodes[e.node] : null;
  if (!n) { tip.style.display = "none"; return; }
  tip.textContent = `${n.op}  ${n.name}  ${n.outType}`;
  Object.assign(tip.style, { display: "block", left: e.x + 14 + "px", top: e.y + 14 + "px" });
});
viewer.on("select", (e) => console.log("select", e.scene?.name));
viewer.on("edge-select", (e) => e.edge != null && console.log("edge", e.name));

/* ── test hooks ── */
Object.assign(window, { __dev: { viewer, mini, load, rebuild, show, so, get model() { return model; } } });

if (params.get("dark") === "1") { document.documentElement.classList.add("dark"); }
if (params.get("dir")) so.direction = params.get("dir") as "TB" | "LR";
load(params.get("m") ?? "small");
