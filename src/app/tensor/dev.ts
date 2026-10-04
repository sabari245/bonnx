import "@/styles/globals.css";
import { h } from "@/lib/dom";
import { initTheme, cycleTheme } from "@/lib/theme";
import { button, select } from "@/ui";
import { createModelClient } from "@/workers/client";
import type { ModelView, TensorInfo } from "@/onnx/types";
import { createTensorPanel, openTensorViewer } from "./index";

initTheme();
const resnet = new URL("../../../models/resnet18-v2-7.onnx", import.meta.url).href;
const fixtures = import.meta.glob("../../../tests/fixtures/*.{onnx,bin}", { query: "?url", import: "default", eager: true }) as Record<string, string>;
const models: Record<string, string> = { "resnet18-v2-7.onnx": resnet };
for (const [p, u] of Object.entries(fixtures)) if (p.endsWith(".onnx")) models[p.split("/").pop()!] = u;
const extUrl = Object.entries(fixtures).find(([p]) => p.endsWith("external.bin"))?.[1];

const api = createModelClient();
const params = new URLSearchParams(location.search);
const app = document.getElementById("app")!;
app.className = "mx-auto flex max-w-5xl flex-col gap-4 p-6";
const panelHost = h("div", { class: "w-80 rounded-lg border p-3" });
const list = h("div", { class: "max-h-[50vh] overflow-auto rounded-lg border" });
const title = h("h1", { class: "text-xl font-semibold" }, "Tensor explorer · dev harness");
let view: ModelView | null = null;

async function load(name: string): Promise<void> {
  const buf = await (await fetch(models[name]!)).arrayBuffer();
  const external: Record<string, ArrayBuffer> = {};
  if (name === "external.onnx" && extUrl && !params.has("noext")) external["external.bin"] = await (await fetch(extUrl)).arrayBuffer();
  view = await api.open({ name, data: buf, external });
  list.replaceChildren(...view.tensors.map((t) => row(t)));
  title.textContent = `${name} · ${view.tensors.length} tensors`;
  const want = params.get("tensor");
  if (want && params.get("model") === name) {
    const t = view.tensors.find((x) => x.name === want || String(x.id) === want);
    if (t) { show(t); if (params.has("tab")) openTensorViewer(api, t, { tensors: view.tensors, tab: params.get("tab") as "data" }); }
  }
}
function row(t: TensorInfo): HTMLElement {
  return h("button", { class: "hover:bg-muted flex w-full items-center justify-between gap-4 border-b px-3 py-1.5 text-left font-mono text-xs", onclick: () => show(t), dataset: { tid: t.id } },
    h("span", { class: "truncate" }, t.name), h("span", { class: "text-muted-foreground shrink-0" }, `${t.dtype} ${t.dims.join("×") || "scalar"}`));
}
function show(t: TensorInfo): void {
  panelHost.replaceChildren(createTensorPanel(api, t, { tensors: view?.tensors }));
}
const pick = select(Object.keys(models).map((m) => ({ value: m, label: m })), { value: params.get("model") ?? "resnet18-v2-7.onnx", class: "w-64", onChange: (v) => void load(v) });
app.append(h("div", { class: "flex items-center gap-3" }, title, h("span", { class: "flex-1" }), pick.el, button("Theme", { variant: "outline", size: "sm", onClick: () => cycleTheme() })),
  h("div", { class: "flex items-start gap-4" }, h("div", { class: "min-w-0 flex-1" }, list), panelHost));
void load(pick.get());
(window as unknown as Record<string, unknown>).__dev = { api, open: (id: number, tab?: string) => openTensorViewer(api, view!.tensors[id]!, { tensors: view!.tensors, tab: tab as "data" }), get view() { return view; } };
