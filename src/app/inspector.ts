import { h, icon, clear } from "../lib/dom";
import { cn } from "../lib/cn";
import { Copy, ChevronRight, Network, Workflow, ArrowRight, CircleAlert, FileText, TriangleAlert } from "../lib/icons";
import { badge, collapsible, toast, tooltip, separator } from "../ui";
import { fmtBytes, fmtCount, fmtNum, fmtShape, fmtType, shortList } from "../lib/format";
import type { AttrValue, GraphView, ModelApi, ModelView, NodeView, OpSchema, TensorInfo, ValueType } from "../onnx/types";

/** Callbacks the inspector needs from the app shell. */
export interface InspectorHost {
  api: ModelApi;
  model: ModelView;
  currentGraph(): number;
  /** select a node in the current graph and move the viewport to it */
  jumpToNode(nodeId: number): void;
  openGraph(graphId: number, focusNode?: number): void;
  findOp(op: string): void;
  /** compact tensor stats + histogram + "open explorer" button */
  tensorPanel(t: TensorInfo): HTMLElement;
  openTensor(t: TensorInfo): void;
  selectValue(name: string): void;
}

const mono = "font-mono text-[11px]";
const copy = (text: string, what = "Copied") =>
  navigator.clipboard?.writeText(text).then(() => toast({ title: what, description: text.length > 80 ? text.slice(0, 80) + "…" : text, duration: 1500 }), () => {});

function copyBtn(text: string, label = "Copy"): HTMLElement {
  const b = h("button", { type: "button", class: "text-muted-foreground hover:text-foreground hover:bg-accent inline-flex size-6 shrink-0 items-center justify-center rounded-md", "aria-label": label, onclick: (e: Event) => { e.stopPropagation(); copy(text); } }, icon(Copy, "size-3.5"));
  tooltip(b, label);
  return b;
}

function section(title: string, ...body: (Node | false | null | undefined)[]): HTMLElement {
  return h("section", { class: "border-b px-4 py-3 last:border-b-0" }, h("h3", { class: "text-muted-foreground mb-2 text-xs font-medium tracking-wide uppercase" }, title), ...body);
}

function kv(rows: [string, Node | string | number | null | undefined][]): HTMLElement {
  return h("dl", { class: "grid grid-cols-[104px_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-xs" },
    ...rows.filter(([, v]) => v != null && v !== "").flatMap(([k, v]) => [h("dt", { class: "text-muted-foreground" }, k), h("dd", { class: "min-w-0 break-words tabular-nums" }, v as Node | string)]));
}

/** link-styled button */
function link(label: string | Node, onClick: () => void, cls = ""): HTMLElement {
  return h("button", { type: "button", class: cn("text-primary min-w-0 truncate text-left hover:underline focus-visible:underline focus-visible:outline-hidden", cls), onclick: onClick }, label);
}

/** minimal markdown renderer for operator docs (paragraphs, fenced/indented code, lists, `code`, **bold**) */
function renderDoc(md: string): HTMLElement {
  const root = h("div", { class: "text-muted-foreground space-y-2 text-xs leading-relaxed" });
  const inline = (s: string): (Node | string)[] =>
    s.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean).map((p) => p.startsWith("`") ? h("code", { class: "bg-muted text-foreground rounded px-1 py-0.5 font-mono text-[11px]" }, p.slice(1, -1)) : p.startsWith("**") ? h("strong", { class: "text-foreground font-semibold" }, p.slice(2, -2)) : p);
  const lines = md.replace(/\r/g, "").split("\n");
  let i = 0;
  while (i < lines.length) {
    const l = lines[i]!;
    if (l.startsWith("```")) {
      const buf: string[] = []; i++;
      while (i < lines.length && !lines[i]!.startsWith("```")) buf.push(lines[i++]!);
      i++; root.append(h("pre", { class: "bg-muted text-foreground overflow-x-auto rounded-md p-2 font-mono text-[11px]" }, buf.join("\n")));
    } else if (/^\s{4,}\S/.test(l)) {
      const buf: string[] = [];
      while (i < lines.length && (/^\s{4,}/.test(lines[i]!) || !lines[i]!.trim())) buf.push(lines[i++]!.replace(/^ {4}/, ""));
      root.append(h("pre", { class: "bg-muted text-foreground overflow-x-auto rounded-md p-2 font-mono text-[11px]" }, buf.join("\n").trim()));
    } else if (/^\s*[-*] /.test(l)) {
      const ul = h("ul", { class: "list-disc space-y-0.5 pl-4" });
      while (i < lines.length && /^\s*[-*] /.test(lines[i]!)) ul.append(h("li", null, ...inline(lines[i++]!.replace(/^\s*[-*] /, ""))));
      root.append(ul);
    } else if (!l.trim()) i++;
    else {
      const buf: string[] = [];
      while (i < lines.length && lines[i]!.trim() && !lines[i]!.startsWith("```") && !/^\s{4,}\S/.test(lines[i]!) && !/^\s*[-*] /.test(lines[i]!)) buf.push(lines[i++]!);
      root.append(h("p", null, ...inline(buf.join(" "))));
    }
  }
  return root;
}

export class Inspector {
  readonly el = h("div", { class: "flex min-h-0 flex-col" });
  constructor(private host: InspectorHost) {}

  private set(...children: Node[]) { clear(this.el); this.el.append(...children); this.el.parentElement?.scrollTo?.({ top: 0 }); }

  /* ───────── model ───────── */
  showModel() {
    const { meta, graphs, tensors, warnings } = this.host.model, host = this.host;
    const opsets = Object.entries(meta.opsets).map(([d, v]) => `${d} ${v}`).join(", ");
    const nodes: Node[] = [];
    nodes.push(section("Model", kv([
      ["File", h("span", { class: mono }, meta.file)], ["Format", meta.format], ["File size", fmtBytes(meta.fileBytes)], ["IR version", meta.ir], ["Opsets", opsets],
      ["Producer", meta.producer], ["Domain", meta.domain], ["Model version", meta.modelVersion == null ? null : String(meta.modelVersion)], ["Graph", meta.graphName],
      ["Nodes", fmtCount(meta.nodeCount) + (graphs.length > 1 ? ` across ${graphs.length} graphs` : "")], ["Operator types", meta.ops.length], ["Parameters", `${meta.params.toLocaleString()} (${fmtCount(meta.params)})`],
      ["Constants", `${tensors.length} (${meta.initializers} initializers)`], ["Functions", meta.functions || null],
      ...Object.entries(meta.weightBytes).map(([d, b]) => [`Weights ${d}`, fmtBytes(b)] as [string, string]),
    ]), meta.docString ? h("p", { class: "bg-muted mt-3 rounded-md p-2 text-xs" }, meta.docString) : null,
      meta.extras.length ? h("div", { class: "mt-2 flex flex-wrap gap-1" }, ...meta.extras.map((x) => badge(x, { variant: "secondary" }))) : null));
    if (warnings.length) nodes.push(section("Warnings", ...warnings.slice(0, 20).map((w) => h("div", { class: "text-destructive mb-1 flex gap-2 text-xs" }, icon(TriangleAlert, "mt-0.5 size-3.5"), w))));
    if (Object.keys(meta.props).length) nodes.push(section("Metadata", kv(Object.entries(meta.props).map(([k, v]) => [k, h("span", { class: mono }, v)] as [string, Node]))));
    const main = graphs[0]!;
    const ioList = (list: GraphViewIO[]) => list.map((x) => h("div", { class: "bg-muted/50 mb-1 flex items-baseline justify-between gap-3 rounded-md border px-2 py-1.5 text-xs" }, h("span", { class: cn(mono, "truncate") }, x.name), h("span", { class: "text-muted-foreground shrink-0 tabular-nums" }, fmtType(x.type))));
    nodes.push(section("Inputs", ...ioList(main.inputs.filter((x) => x.tid == null))), section("Outputs", ...ioList(main.outputs)));
    if (graphs.length > 1) nodes.push(section("Graphs", ...graphs.map((g) => h("button", { type: "button", class: "hover:bg-accent flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs", onclick: () => host.openGraph(g.id) },
      icon(g.kind === "function" ? Workflow : Network, "text-muted-foreground size-3.5"), h("span", { class: "min-w-0 flex-1 truncate", style: { paddingLeft: `${g.depth * 10}px` } }, g.kind === "function" ? `${g.fnDomain ? g.fnDomain + "." : ""}${g.fnName}` : g.name || "(unnamed graph)"),
      badge(g.kind, { variant: "outline" }), h("span", { class: "text-muted-foreground tabular-nums" }, g.nodes.length)))));
    nodes.push(section(`Operators (${meta.ops.length})`, h("div", { class: "grid grid-cols-1 gap-0.5" }, ...meta.ops.map(([op, n]) => h("button", { type: "button", class: "hover:bg-accent flex items-center justify-between rounded-md px-2 py-1 text-xs", onclick: () => host.findOp(op) }, h("span", { class: "truncate" }, op), h("span", { class: "text-muted-foreground tabular-nums" }, n))))));
    const bad = tensors.filter((t) => t.summary && (t.summary.nan || t.summary.inf));
    if (bad.length) nodes.push(section("Numerical health", h("div", { class: "text-destructive mb-2 flex items-center gap-2 text-xs font-medium" }, icon(CircleAlert), `${bad.length} constants contain NaN or Inf`), ...bad.slice(0, 30).map((t) => this.tensorRow(t, `${t.summary!.nan} NaN, ${t.summary!.inf} Inf`))));
    const big = tensors.filter((t) => t.n > 1).sort((a, b) => b.n - a.n).slice(0, 12);
    if (big.length) nodes.push(section("Largest constants", ...big.map((t) => this.tensorRow(t, `${fmtCount(t.n)} ⟨${fmtShape(t.dims)}⟩`))));
    const wide = tensors.filter((t) => t.summary?.absmax != null && /float/.test(t.dtype)).sort((a, b) => b.summary!.absmax! - a.summary!.absmax!).slice(0, 12);
    if (wide.length) nodes.push(section("Widest weight ranges (|x| max)", ...wide.map((t) => this.tensorRow(t, fmtNum(t.summary!.absmax)))));
    const zero = tensors.filter((t) => t.n > 64 && t.summary && t.summary.zeros > 0.5).sort((a, b) => b.summary!.zeros - a.summary!.zeros).slice(0, 12);
    if (zero.length) nodes.push(section("Sparsest constants (zeros)", ...zero.map((t) => this.tensorRow(t, `${(t.summary!.zeros * 100).toFixed(1)} %`))));
    this.set(...nodes);
  }

  private tensorRow(t: TensorInfo, right: string): HTMLElement {
    return h("button", { type: "button", class: "hover:bg-accent flex w-full items-center justify-between gap-3 rounded-md px-2 py-1 text-left text-xs", onclick: () => this.host.openTensor(t) },
      h("span", { class: cn(mono, "truncate"), title: t.name }, t.name), h("span", { class: "text-muted-foreground shrink-0 tabular-nums" }, right));
  }

  /* ───────── graph ───────── */
  showGraph(g: GraphView) {
    const host = this.host;
    this.set(section(g.kind === "function" ? "Function" : g.kind === "subgraph" ? "Subgraph" : "Graph", kv([
      ["Name", g.name || "(unnamed)"], g.fnName ? ["Function", `${g.fnDomain ?? ""}::${g.fnName}${g.fnOverload ? ":" + g.fnOverload : ""}`] : ["", ""],
      ["Nodes", g.nodes.length], ["Constants", g.initializers.length], ["Attributes", g.fnAttrs?.join(", ")],
      ["Parent", g.parent ? link(`${host.model.graphs[g.parent.graph]!.nodes[g.parent.node]?.name ?? "node"} › ${g.parent.attr}`, () => host.openGraph(g.parent!.graph, g.parent!.node)) : null],
    ].filter((r) => r[0]) as [string, Node | string | number | null | undefined][]), g.docString ? h("p", { class: "bg-muted mt-3 rounded-md p-2 text-xs" }, g.docString) : null));
  }

  /* ───────── node ───────── */
  showNode(graphId: number, nodeId: number) {
    const host = this.host, g = host.model.graphs[graphId]!, n = g.nodes[nodeId]!;
    const sections: Node[] = [];
    const schemaBox = h("div", { class: "mt-3" });
    sections.push(section("Node", kv([
      ["Type", h("span", { class: "inline-flex items-center gap-1.5" }, h("span", { class: "font-semibold" }, n.op), n.domain ? badge(n.domain, { variant: "outline" }) : null, n.opset != null ? badge(`opset ${n.opset}`, { variant: "secondary" }) : null)],
      ["Name", h("span", { class: "inline-flex items-center gap-1" }, h("span", { class: cn(mono, "break-all") }, n.name), copyBtn(n.name, "Copy node name"))],
      ["Index", `${n.src}`], ["Parameters", n.params ? n.params.toLocaleString() : null], ["|w| max", n.wmax != null ? fmtNum(n.wmax) : null],
    ]), n.docString ? h("p", { class: "bg-muted mt-3 rounded-md p-2 text-xs" }, n.docString) : null, schemaBox));
    this.loadSchema(n, schemaBox);

    if (n.fn != null || n.subgraphs.length) {
      sections.push(section("Nested graphs", h("div", { class: "flex flex-col gap-1.5" },
        ...(n.fn != null ? [this.graphButton(n.fn, `Open function ${host.model.graphs[n.fn]!.fnName}`)] : []),
        ...n.subgraphs.map((s) => this.graphButton(s.graph, `Open ${s.attr}`, host.model.graphs[s.graph]!.nodes.length)))));
    }
    if (n.attrs.length) sections.push(section(`Attributes (${n.attrs.length})`, h("div", { class: "flex flex-col gap-2" }, ...n.attrs.map((a) => this.attrRow(g, n, a)))));
    if (n.inputs.length) sections.push(section(`Inputs (${n.inputs.length})`, ...n.inputs.map((i) => this.valueRow(g, i.param, i.name, i.type, i.tid))));
    if (n.outputs.length) sections.push(section(`Outputs (${n.outputs.length})`, ...n.outputs.map((o) => this.valueRow(g, o.param, o.name, o.type, null, true))));
    const succ = new Set<number>(), pred = new Set<number>();
    for (const o of n.outputs) for (const c of g.values[o.name]?.consumers ?? []) succ.add(c);
    for (const i of n.inputs) { const p = g.values[i.name]?.producer ?? -1; if (p >= 0) pred.add(p); }
    const nodeLinks = (title: string, set: Set<number>) => set.size ? section(title, ...[...set].map((id) => h("button", { type: "button", class: "hover:bg-accent flex w-full items-center justify-between gap-3 rounded-md px-2 py-1 text-left text-xs", onclick: () => host.jumpToNode(id) }, h("span", { class: "truncate" }, g.nodes[id]!.op), h("span", { class: cn(mono, "text-muted-foreground truncate") }, g.nodes[id]!.name)))) : null;
    const a = nodeLinks("Producers", pred), b = nodeLinks("Consumers", succ);
    if (a) sections.push(a);
    if (b) sections.push(b);
    if (n.metadata && Object.keys(n.metadata).length) sections.push(section("Node metadata", kv(Object.entries(n.metadata).map(([k, v]) => [k, v] as [string, string]))));
    this.set(...sections);
  }

  private graphButton(graph: number, label: string, nodes?: number): HTMLElement {
    return h("button", { type: "button", class: "bg-secondary hover:bg-secondary/80 flex items-center gap-2 rounded-md border px-3 py-2 text-left text-xs font-medium", onclick: () => this.host.openGraph(graph) },
      icon(Network, "size-4"), h("span", { class: "flex-1" }, label), nodes != null ? h("span", { class: "text-muted-foreground font-normal tabular-nums" }, `${nodes} nodes`) : null, icon(ChevronRight, "text-muted-foreground size-4"));
  }

  private async loadSchema(n: NodeView, box: HTMLElement) {
    let s: OpSchema | null = null;
    try { s = await this.host.api.opSchema(n.domain, n.op, n.opset); } catch { /* optional */ }
    if (!s || !box.isConnected) return;
    const doc = collapsible({ title: `Documentation · since opset ${s.sinceVersion}` }, renderDoc(s.doc));
    box.append(doc.el ?? (doc as unknown as Node));
  }

  private attrRow(g: GraphView, n: NodeView, a: { name: string; value: AttrValue; docString?: string }): HTMLElement {
    const host = this.host, v = a.value;
    let body: Node | string;
    const list = (arr: readonly (string | number)[]) => {
      const text = `[${arr.join(", ")}]`;
      return h("span", { class: cn(mono, "break-all") }, arr.length <= 12 ? text : `${shortList(arr as (string | number)[], 12)} · ${arr.length} items`, arr.length > 12 ? copyBtn(text, "Copy list") : null);
    };
    switch (v.t) {
      case "int": case "float": body = h("span", { class: mono }, String(v.v)); break;
      case "string": body = h("span", { class: cn(mono, "break-all") }, v.bytes ? `0x${v.v}` : `"${v.v}"`); break;
      case "ints": case "floats": case "strings": body = list(v.v as (string | number)[]); break;
      case "tensor": case "sparse_tensor": body = host.tensorPanel(host.model.tensors[v.tid]!); break;
      case "tensors": body = h("div", { class: "flex flex-col gap-2" }, ...v.tids.map((t) => host.tensorPanel(host.model.tensors[t]!))); break;
      case "graph": body = this.graphButton(v.graph, "Open subgraph", host.model.graphs[v.graph]!.nodes.length); break;
      case "graphs": body = h("div", { class: "flex flex-col gap-1.5" }, ...v.graphs.map((x, i) => this.graphButton(x, `Open graph ${i}`, host.model.graphs[x]!.nodes.length))); break;
      case "type": body = h("span", { class: mono }, fmtType(v.type, true)); break;
      case "types": body = h("span", { class: mono }, v.types.map((t) => fmtType(t, true)).join(", ")); break;
      case "ref": body = h("span", { class: mono }, `@${v.ref}`); break;
      default: body = "—";
    }
    return h("div", { class: "text-xs" }, h("div", { class: "text-muted-foreground mb-0.5 flex items-center gap-1.5" }, h("span", { class: "text-foreground font-medium" }, a.name), badge(v.t, { variant: "outline", class: "px-1 py-0 text-[10px]" })), body,
      a.docString ? h("p", { class: "text-muted-foreground mt-0.5" }, a.docString) : null);
  }

  private valueRow(g: GraphView, param: string, name: string, type: ValueType | null, tid: number | null, isOut = false): HTMLElement {
    const host = this.host, info = g.values[name];
    const t = tid != null ? host.model.tensors[tid] : null;
    const prod = info && info.producer >= 0 ? info.producer : -1;
    return h("div", { class: "mb-2 overflow-hidden rounded-md border text-xs" },
      h("div", { class: "bg-muted/50 flex items-center justify-between gap-3 px-2 py-1.5" },
        h("div", { class: "min-w-0" }, h("div", { class: "text-muted-foreground text-[10px] font-medium uppercase" }, param),
          h("button", { type: "button", class: cn(mono, "text-primary block max-w-full truncate hover:underline"), title: name, onclick: () => host.selectValue(name) }, name)),
        h("div", { class: "flex shrink-0 items-center gap-1" }, copyBtn(name, "Copy tensor name"), h("span", { class: "text-muted-foreground tabular-nums" }, fmtType(type)))),
      !isOut && prod >= 0 ? h("button", { type: "button", class: "hover:bg-accent text-muted-foreground flex w-full items-center gap-1.5 border-t px-2 py-1 text-left", onclick: () => host.jumpToNode(prod) }, "from", h("span", { class: "text-foreground font-medium" }, g.nodes[prod]!.op), icon(ArrowRight, "size-3")) : null,
      info?.outer ? h("div", { class: "text-muted-foreground border-t px-2 py-1" }, "Captured from an enclosing graph") : null,
      isOut && info && info.consumers.length ? h("div", { class: "text-muted-foreground flex flex-wrap items-center gap-1 border-t px-2 py-1" }, "to", ...info.consumers.slice(0, 8).map((c) => link(g.nodes[c]!.op, () => host.jumpToNode(c), "text-xs font-medium"))) : null,
      t ? h("div", { class: "border-t p-2" }, host.tensorPanel(t)) : null);
  }

  /* ───────── value (edge) ───────── */
  showValue(graphId: number, name: string) {
    const host = this.host, g = host.model.graphs[graphId]!, info = g.values[name];
    const t = info?.tid != null ? host.model.tensors[info.tid] : null;
    this.set(section("Value", kv([
      ["Name", h("span", { class: "inline-flex items-center gap-1" }, h("span", { class: cn(mono, "break-all") }, name), copyBtn(name, "Copy tensor name"))],
      ["Type", info?.type ? fmtType(info.type, true) : "unknown"],
      ["Producer", info && info.producer >= 0 ? link(`${g.nodes[info.producer]!.op} · ${g.nodes[info.producer]!.name}`, () => host.jumpToNode(info.producer)) : "graph input / constant"],
      ["Consumers", info ? info.consumers.length : 0],
    ]), info?.docString ? h("p", { class: "bg-muted mt-3 rounded-md p-2 text-xs" }, info.docString) : null),
      info && info.consumers.length ? section("Consumers", ...info.consumers.map((c) => this.nodeLink(g, c))) : h("span"),
      t ? section("Constant", host.tensorPanel(t)) : h("span"));
  }
  private nodeLink(g: GraphView, id: number): HTMLElement {
    return h("button", { type: "button", class: "hover:bg-accent flex w-full items-center justify-between gap-3 rounded-md px-2 py-1 text-left text-xs", onclick: () => this.host.jumpToNode(id) }, h("span", { class: "truncate font-medium" }, g.nodes[id]!.op), h("span", { class: cn(mono, "text-muted-foreground truncate") }, g.nodes[id]!.name));
  }

  /* ───────── graph input / output / constant ───────── */
  showIO(graphId: number, kind: "input" | "output", index: number) {
    const host = this.host, g = host.model.graphs[graphId]!, io = (kind === "input" ? g.inputs : g.outputs)[index]!;
    const t = io.tid != null ? host.model.tensors[io.tid] : null;
    this.set(section(kind === "input" ? "Graph input" : "Graph output", kv([
      ["Name", h("span", { class: "inline-flex items-center gap-1" }, h("span", { class: cn(mono, "break-all") }, io.name), copyBtn(io.name, "Copy name"))],
      ["Type", fmtType(io.type, true)],
    ]), io.docString ? h("p", { class: "bg-muted mt-3 rounded-md p-2 text-xs" }, io.docString) : null), t ? section("Default value", host.tensorPanel(t)) : h("span"));
  }
  showConst(tid: number) {
    const t = this.host.model.tensors[tid]!;
    this.set(section("Constant", this.host.tensorPanel(t)));
  }
  showEmpty() { this.set(h("div", { class: "text-muted-foreground p-6 text-center text-xs" }, icon(FileText, "mx-auto mb-2 size-5"), "Select a node or edge to inspect it.", separator("horizontal"))); }
}

type GraphViewIO = GraphView["inputs"][number];
