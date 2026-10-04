import type { AttrValue, GraphView, ModelView, NodeView, SceneEdge, SceneNode, TensorInfo, ValueType } from "../onnx/types";
import { dtypeCategory, fmtNum, fmtType, shortList, truncate, typeDType, type DTypeCategory } from "../lib/format";
import type { CatKey } from "./theme";

/* ───────────────────────── options ───────────────────────── */

export interface SceneOptions {
  showInputsOutputs: boolean;
  /** 'rows': constants listed inside the op card; 'nodes': separate const nodes with edges; 'hidden' */
  showInitializers: "rows" | "nodes" | "hidden";
  showAttributes: boolean;
  showNames: boolean;
  /** edge labels (can also be changed live on the viewer without relayout) */
  showTypes: "none" | "shape" | "name" | "both";
  direction: "TB" | "LR";
  /** max inline attribute / constant rows per card */
  maxAttrRows: number;
  maxConstRows: number;
}

export const defaultSceneOptions: SceneOptions = {
  showInputsOutputs: true,
  showInitializers: "rows",
  showAttributes: false,
  showNames: false,
  showTypes: "shape",
  direction: "TB",
  maxAttrRows: 6,
  maxConstRows: 4,
};

/* ───────────────────────── operator categories ───────────────────────── */

export interface OpCategory { key: CatKey; label: string; ops: string[] }
export const OP_CATEGORIES: OpCategory[] = [
  { key: "layer", label: "Layer", ops: ["Conv", "ConvTranspose", "ConvInteger", "QLinearConv", "MatMul", "MatMulInteger", "QLinearMatMul", "MatMulNBits", "Gemm", "Einsum", "LSTM", "GRU", "RNN", "Attention", "MultiHeadAttention", "GroupQueryAttention", "FusedConv", "FusedGemm", "FusedMatMul"] },
  { key: "activation", label: "Activation", ops: ["Relu", "LeakyRelu", "PRelu", "Sigmoid", "Tanh", "Erf", "Gelu", "FastGelu", "BiasGelu", "Clip", "Softmax", "LogSoftmax", "Hardmax", "HardSigmoid", "HardSwish", "Elu", "Selu", "Softplus", "Softsign", "Mish", "Celu", "Swish", "ThresholdedRelu", "Silu"] },
  { key: "norm", label: "Normalization", ops: ["BatchNormalization", "InstanceNormalization", "LayerNormalization", "GroupNormalization", "LpNormalization", "MeanVarianceNormalization", "SimplifiedLayerNormalization", "SkipLayerNormalization", "RMSNormalization", "LRN", "Dropout"] },
  { key: "pool", label: "Pool / resize", ops: ["MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "GlobalLpPool", "LpPool", "MaxRoiPool", "RoiAlign", "MaxUnpool", "Resize", "Upsample"] },
  { key: "math", label: "Math / logic", ops: ["Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log", "Abs", "Neg", "Reciprocal", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh", "Floor", "Ceil", "Round", "Min", "Max", "Mean", "Sum", "Mod", "Sign", "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Not", "And", "Or", "Xor", "Where", "CumSum", "BitShift", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "BitwiseNot", "IsNaN", "IsInf", "Det", "Trilu"] },
  { key: "reduce", label: "Reduce / search", ops: ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp", "ArgMax", "ArgMin", "TopK", "NonMaxSuppression", "Unique"] },
  { key: "shape", label: "Shape / copy", ops: ["Reshape", "Transpose", "Squeeze", "Unsqueeze", "Concat", "Split", "Slice", "Expand", "Gather", "GatherElements", "GatherND", "Flatten", "Shape", "Size", "Cast", "CastLike", "Identity", "Tile", "Pad", "ScatterND", "ScatterElements", "Scatter", "DepthToSpace", "SpaceToDepth", "ConstantOfShape", "Range", "NonZero", "OneHot", "Compress", "ReverseSequence", "Constant", "EyeLike"] },
  { key: "quant", label: "Quantize", ops: ["QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "QLinearAdd", "QLinearMul", "QLinearSigmoid", "QLinearAveragePool", "QLinearGlobalAveragePool", "QLinearConcat", "QLinearSoftmax"] },
  { key: "control", label: "Control flow / function", ops: ["If", "Loop", "Scan", "SequenceMap"] },
  { key: "other", label: "Other", ops: [] },
];
const CAT_INDEX: Map<string, number> = new Map();
OP_CATEGORIES.forEach((c, i) => c.ops.forEach((o) => CAT_INDEX.set(o, i)));
export const CAT_CONTROL = OP_CATEGORIES.findIndex((c) => c.key === "control");
export const CAT_OTHER = OP_CATEGORIES.length - 1;
export const catOfOp = (op: string): number => CAT_INDEX.get(op) ?? CAT_OTHER;

/* ───────────────────────── scene types ───────────────────────── */

export type LineKind = "name" | "const" | "attr" | "sub" | "metric" | "more";
export interface CardLine {
  kind: LineKind;
  text: string;
  /** y offset from node top (set by measure) */
  y: number;
  /** sub rows: graph to open */
  graph?: number;
  attr?: string;
  /** const rows: tensor id */
  tid?: number;
}

export interface RNode extends SceneNode {
  name: string; // node / io / tensor name
  title: string; // header text
  subtitle: string; // second line for io/const nodes (type text)
  op: string;
  domain: string;
  cat: number; // OP_CATEGORIES index (op/fn only)
  params: number;
  wmax: number | null;
  /** lines drawn under the header (op/fn cards) */
  lines: CardLine[];
  hdh: number; // header height (op/fn), else 0
  subgraphs: { graph: number; attr: string }[];
  fn: number | null;
  expandable: boolean;
  outType: string; // first output type text
  dtypeCat: DTypeCategory | null;
  depth: number;
}

export interface REdge extends SceneEdge {
  /** type text (short) */
  typeText: string;
}

export interface Scene {
  graphId: number;
  graphName: string;
  graphKind: GraphView["kind"];
  options: SceneOptions;
  nodes: RNode[];
  edges: REdge[];
  maxDepth: number;
  width: number;
  height: number;
  laidOut: boolean;
  model: ModelView;
  /** graph-level info that has no node (e.g. empty graph) */
  empty: boolean;
}

/* ───────────────────────── text measurement ───────────────────────── */

export interface SceneFonts { sans: string; mono: string }
export const defaultFonts: SceneFonts = {
  sans: 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  mono: 'ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace',
};
export const HD = 24; // card header height
export const LH = 16; // card line height
export const PADX = 8;
export const MAX_W = 340;
export const GLYPH_W = 20;

export interface FontSet { title: string; line: string; lineBold: string; io: string; ioSub: string; label: string; sub: string }
export const fontSet = (f: SceneFonts): FontSet => ({
  title: `600 12px ${f.sans}`,
  line: `10.5px ${f.sans}`,
  lineBold: `600 10.5px ${f.sans}`,
  io: `600 11.5px ${f.sans}`,
  ioSub: `10px ${f.sans}`,
  label: `10px ${f.sans}`,
  sub: `500 10.5px ${f.sans}`,
});

let mctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;
const wcache = new Map<string, number>();
export function textWidth(font: string, text: string): number {
  const key = font + "\u0000" + text;
  let w = wcache.get(key);
  if (w === undefined) {
    mctx ??= (typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(1, 1) : document.createElement("canvas")).getContext("2d") as CanvasRenderingContext2D;
    mctx.font = font;
    w = mctx.measureText(text).width;
    if (wcache.size > 60000) wcache.clear();
    wcache.set(key, w);
  }
  return w;
}
function fitText(font: string, text: string, maxW: number): string {
  if (textWidth(font, text) <= maxW) return text;
  let lo = 1, hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (textWidth(font, text.slice(0, mid) + "…") <= maxW) lo = mid;
    else hi = mid - 1;
  }
  return text.slice(0, lo) + "…";
}

/**
 * Compute w/h (and line y offsets) of a node from real text metrics.
 * Exported so the app can re-measure when the font changes.
 */
export function measureNode(n: RNode, fonts: FontSet): void {
  switch (n.kind) {
    case "op":
    case "fn": {
      let w = textWidth(fonts.title, n.title) + 2 * PADX + (n.expandable ? GLYPH_W : 0);
      for (const l of n.lines) {
        const f = l.kind === "metric" ? fonts.lineBold : l.kind === "sub" ? fonts.sub : fonts.line;
        w = Math.max(w, textWidth(f, l.text) + 2 * PADX + (l.kind === "sub" ? 8 : 0));
      }
      w = Math.ceil(Math.min(MAX_W, Math.max(118, w)));
      let y = HD + 4;
      for (const l of n.lines) {
        l.y = y;
        y += LH;
        const f = l.kind === "metric" ? fonts.lineBold : l.kind === "sub" ? fonts.sub : fonts.line;
        l.text = fitText(f, l.text, w - 2 * PADX - (l.kind === "sub" ? 8 : 0));
      }
      n.w = w;
      n.h = y + 4;
      n.hdh = HD;
      n.title = fitText(fonts.title, n.title, w - 2 * PADX - (n.expandable ? GLYPH_W : 0));
      break;
    }
    case "input":
    case "output":
    case "const": {
      const wt = textWidth(fonts.io, n.title), ws = textWidth(fonts.ioSub, n.subtitle);
      const w = Math.ceil(Math.min(MAX_W, Math.max(n.kind === "const" ? 86 : 74, wt + 24, ws + 24)));
      n.w = w;
      n.h = 38;
      n.title = fitText(fonts.io, n.title, w - 20);
      n.subtitle = fitText(fonts.ioSub, n.subtitle, w - 20);
      break;
    }
    case "ghost": {
      n.w = Math.ceil(Math.min(MAX_W, Math.max(80, textWidth(fonts.io, n.title) + 24)));
      n.h = 38;
      n.title = fitText(fonts.io, n.title, n.w - 20);
      break;
    }
  }
}

/* ───────────────────────── attribute display ───────────────────────── */

export function attrText(a: AttrValue, model?: ModelView): string {
  switch (a.t) {
    case "int": return String(a.v);
    case "float": return fmtNum(a.v);
    case "string": return a.bytes ? `bytes ${truncate(a.v, 18)}` : JSON.stringify(truncate(a.v, 28));
    case "ints": return shortList(a.v, 8);
    case "floats": return shortList(a.v, 6, (x) => fmtNum(x, 3));
    case "strings": return shortList(a.v, 4, (x) => JSON.stringify(truncate(x, 14)));
    case "tensor": case "sparse_tensor": { const t = model?.tensors[a.tid]; return t ? `tensor ${shortTensor(t)}` : "tensor"; }
    case "tensors": return `${a.tids.length} tensors`;
    case "graph": return "graph";
    case "graphs": return `${a.graphs.length} graphs`;
    case "type": return fmtType(a.type);
    case "types": return shortList(a.types, 3, (x) => fmtType(x));
    case "ref": return `@${a.ref}`;
    default: return "?";
  }
}
export const shortTensor = (t: TensorInfo): string => fmtType({ kind: "tensor", dtype: t.dtype, shape: t.dims });

/* ───────────────────────── scene builder ───────────────────────── */

const labelOf = (name: string, type: ValueType | null, mode: SceneOptions["showTypes"]): string =>
  mode === "none" ? "" : mode === "name" ? name : mode === "shape" ? fmtType(type) : `${name}  ${fmtType(type)}`;
export { labelOf as edgeLabelText };

export function buildScene(model: ModelView, graphId: number, opts: Partial<SceneOptions> = {}, fontsIn: SceneFonts = defaultFonts): Scene {
  const o: SceneOptions = { ...defaultSceneOptions, ...opts };
  const g = model.graphs[graphId];
  const fonts = fontSet(fontsIn);
  const nodes: RNode[] = [];
  const edges: REdge[] = [];
  const producer = new Map<string, number>(); // value name -> scene node id
  const edgeKeys = new Set<string>();
  const constNode = new Map<number, number>(); // tid -> scene node id
  const ghost = new Map<string, number>();

  const blank = (kind: RNode["kind"], ref: number, label: string): RNode => ({
    id: nodes.length, kind, ref, label, x: 0, y: 0, w: 0, h: 0, name: label, title: label, subtitle: "", op: label, domain: "", cat: CAT_OTHER,
    params: 0, wmax: null, lines: [], hdh: 0, subgraphs: [], fn: null, expandable: false, outType: "", dtypeCat: null, depth: 0,
  });
  const push = (n: RNode): RNode => (nodes.push(n), n);
  const addEdge = (s: number, t: number, name: string, type: ValueType | null, isConst = false): void => {
    if (s === t) return;
    const key = `${s}>${t}>${name}`;
    if (edgeKeys.has(key)) return;
    edgeKeys.add(key);
    edges.push({ id: edges.length, s, t, name, type, p: [], isConst, typeText: fmtType(type) });
  };
  const ghostFor = (name: string, type: ValueType | null): number => {
    let id = ghost.get(name);
    if (id === undefined) {
      const n = push(blank("ghost", -1, "outer: " + name));
      n.name = name;
      n.outType = fmtType(type);
      n.dtypeCat = dtypeCategory(typeDType(type));
      id = n.id;
      ghost.set(name, id);
    }
    return id;
  };

  // graph inputs
  if (o.showInputsOutputs) {
    g.inputs.forEach((io, i) => {
      if (io.tid != null) return; // initializer defaults are shown as constants
      const n = push(blank("input", i, io.name));
      n.outType = fmtType(io.type);
      n.subtitle = `input · ${n.outType}`;
      n.dtypeCat = dtypeCategory(typeDType(io.type));
      producer.set(io.name, n.id);
    });
  }

  // op nodes
  const opIds: number[] = [];
  for (const nv of g.nodes) {
    const n = buildOpNode(model, g, nv, o, nodes.length);
    push(n);
    opIds.push(n.id);
    for (const out of nv.outputs) producer.set(out.name, n.id);
  }

  // edges into op nodes
  for (const nv of g.nodes) {
    const t = opIds[nv.id];
    for (const inp of nv.inputs) {
      if (inp.tid != null) {
        if (o.showInitializers === "nodes") {
          let c = constNode.get(inp.tid);
          if (c === undefined) {
            const ti = model.tensors[inp.tid];
            const cn = push(blank("const", inp.tid, ti.name));
            cn.subtitle = shortTensor(ti);
            cn.outType = cn.subtitle;
            cn.dtypeCat = dtypeCategory(ti.dtype);
            c = cn.id;
            constNode.set(inp.tid, c);
          }
          addEdge(c, t, inp.name, inp.type, true);
        }
        continue;
      }
      const pid = producer.get(inp.name);
      if (pid !== undefined) addEdge(pid, t, inp.name, inp.type);
      else {
        const vi = g.values[inp.name];
        if (vi?.outer || inp.name) addEdge(ghostFor(inp.name, inp.type ?? vi?.type ?? null), t, inp.name, inp.type);
      }
    }
  }

  // graph outputs
  if (o.showInputsOutputs) {
    g.outputs.forEach((io, i) => {
      const n = push(blank("output", i, io.name));
      n.outType = fmtType(io.type);
      n.subtitle = `output · ${n.outType}`;
      n.dtypeCat = dtypeCategory(typeDType(io.type));
      const pid = producer.get(io.name);
      if (pid !== undefined) addEdge(pid, n.id, io.name, io.type);
      else if (io.tid == null) addEdge(ghostFor(io.name, io.type), n.id, io.name, io.type);
    });
  }

  for (const n of nodes) measureNode(n, fonts);

  // topological depth (Kahn, tolerant of cycles)
  const indeg = new Int32Array(nodes.length);
  const out: number[][] = nodes.map(() => []);
  for (const e of edges) { indeg[e.t]++; out[e.s].push(e.t); }
  const queue: number[] = [];
  for (let i = 0; i < nodes.length; i++) if (!indeg[i]) queue.push(i);
  let maxDepth = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const u = queue[qi];
    for (const v of out[u]) {
      if (nodes[v].depth < nodes[u].depth + 1) nodes[v].depth = nodes[u].depth + 1;
      if (--indeg[v] === 0) queue.push(v);
    }
    maxDepth = Math.max(maxDepth, nodes[u].depth);
  }

  return { graphId, graphName: g.name, graphKind: g.kind, options: o, nodes, edges, maxDepth, width: 0, height: 0, laidOut: false, model, empty: nodes.length === 0 };
}

function buildOpNode(model: ModelView, g: GraphView, nv: NodeView, o: SceneOptions, id: number): RNode {
  const isFn = nv.fn != null || nv.subgraphs.length > 0;
  const fnGraph = nv.fn != null ? model.graphs[nv.fn] : null;
  const n: RNode = {
    id, kind: isFn ? "fn" : "op", ref: nv.id, label: nv.op, x: 0, y: 0, w: 0, h: 0, name: nv.name, title: nv.op, subtitle: "",
    op: nv.op, domain: nv.domain, cat: isFn ? CAT_CONTROL : catOfOp(nv.op), params: nv.params, wmax: nv.wmax, lines: [], hdh: HD,
    subgraphs: nv.subgraphs.map((s) => ({ graph: s.graph, attr: s.attr })), fn: nv.fn, expandable: isFn,
    outType: nv.outputs[0] ? fmtType(nv.outputs[0].type) : "", dtypeCat: nv.outputs[0] ? dtypeCategory(typeDType(nv.outputs[0].type)) : null, depth: 0,
  };
  const L = n.lines;
  if (o.showNames) L.push({ kind: "name", text: nv.name, y: 0 });
  if (o.showInitializers === "rows") {
    const consts = nv.inputs.filter((i) => i.tid != null);
    for (const c of consts.slice(0, o.maxConstRows)) {
      const ti = model.tensors[c.tid!];
      L.push({ kind: "const", text: `${c.param}  ${shortTensor(ti)}`, y: 0, tid: c.tid! });
    }
    if (consts.length > o.maxConstRows) L.push({ kind: "more", text: `+ ${consts.length - o.maxConstRows} more constants`, y: 0 });
  }
  if (o.showAttributes) {
    const attrs = nv.attrs.filter((a) => a.value.t !== "graph" && a.value.t !== "graphs");
    for (const a of attrs.slice(0, o.maxAttrRows)) L.push({ kind: "attr", text: `${a.name} = ${attrText(a.value, model)}`, y: 0 });
    if (attrs.length > o.maxAttrRows) L.push({ kind: "more", text: `+ ${attrs.length - o.maxAttrRows} more attributes`, y: 0 });
  }
  for (const s of nv.subgraphs) L.push({ kind: "sub", text: `▸ ${s.attr} (${model.graphs[s.graph]?.nodes.length ?? 0} nodes)`, y: 0, graph: s.graph, attr: s.attr });
  if (fnGraph) L.push({ kind: "sub", text: `▸ function ${fnGraph.fnName ?? fnGraph.name} (${fnGraph.nodes.length} nodes)`, y: 0, graph: fnGraph.id, attr: "" });
  L.push({ kind: "metric", text: "", y: 0 }); // text is mode dependent; width reserved from output type below
  L[L.length - 1].text = n.outType ? `→ ${n.outType}` : "";
  void g;
  return n;
}
