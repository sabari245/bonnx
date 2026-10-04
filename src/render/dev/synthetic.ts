import type { AttrView, GraphView, ModelView, NodeInput, NodeOutput, NodeView, TensorInfo, ValueType } from "../../onnx/types";

/** Deterministic PRNG so layouts are reproducible */
function rng(seed: number): () => number {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

const f32 = (shape: (number | string)[]): ValueType => ({ kind: "tensor", dtype: "float32", shape });

class Builder {
  graph: GraphView;
  constructor(readonly model: ModelView, id: number, name: string, kind: GraphView["kind"], parent: GraphView["parent"] = null) {
    this.graph = { id, name, kind, parent, nodes: [], inputs: [], outputs: [], values: {}, initializers: [], depth: parent ? 1 : 0 };
    model.graphs[id] = this.graph;
  }
  input(name: string, type: ValueType): void {
    this.graph.inputs.push({ name, type, tid: null });
    this.graph.values[name] = { name, type, producer: -1, consumers: [], tid: null, outer: false };
  }
  output(name: string, type: ValueType): void { this.graph.outputs.push({ name, type, tid: null }); }
  tensor(name: string, dims: number[], dtype = "float32"): number {
    const n = dims.reduce((a, b) => a * b, 1);
    const t: TensorInfo = {
      id: this.model.tensors.length, name, dtype, dims, n, bytes: n * 4, source: "initializer", graph: this.graph.id, available: true,
      summary: { min: -0.5, max: 0.5, mean: 0, std: 0.1, absmax: 0.5 + (n % 7) * 0.3, zeros: 0, nan: 0, inf: 0 },
    };
    this.model.tensors.push(t);
    this.graph.initializers.push(t.id);
    return t.id;
  }
  add(op: string, ins: string[], out: string, type: ValueType, o: { consts?: [string, number[]][]; attrs?: AttrView[]; name?: string; domain?: string; subgraphs?: { attr: string; graph: number }[]; fn?: number; extraOuts?: string[] } = {}): string {
    const id = this.graph.nodes.length;
    const inputs: NodeInput[] = ins.map((name, i) => ({ param: i === 0 ? "X" : `in${i}`, name, index: i, type: this.graph.values[name]?.type ?? type, tid: null }));
    let params = 0, wmax: number | null = null;
    for (const [param, dims] of o.consts ?? []) {
      const tid = this.tensor(`${o.name ?? op + "_" + id}.${param}`, dims);
      const t = this.model.tensors[tid];
      inputs.push({ param, name: t.name, index: inputs.length, type: f32(dims), tid });
      if (t.n > 1) { params += t.n; wmax = Math.max(wmax ?? 0, t.summary!.absmax!); }
    }
    const outputs: NodeOutput[] = [out, ...(o.extraOuts ?? [])].map((name, i) => ({ param: i ? `Y${i}` : "Y", name, index: i, type }));
    const nv: NodeView = {
      id, name: o.name ?? `${op}_${id}`, op, domain: o.domain ?? "", opset: 17, inputs, outputs, attrs: o.attrs ?? [], fn: o.fn ?? null,
      subgraphs: o.subgraphs ?? [], params, wmax, src: id,
    };
    this.graph.nodes.push(nv);
    for (const x of ins) { const v = this.graph.values[x]; if (v) v.consumers.push(id); }
    for (const x of outputs) this.graph.values[x.name] = { name: x.name, type, producer: id, consumers: [], tid: null, outer: false };
    return out;
  }
}

const ia = (name: string, v: number[]): AttrView => ({ name, value: { t: "ints", v } });
const ii = (name: string, v: number): AttrView => ({ name, value: { t: "int", v } });

function emptyModel(file: string): ModelView {
  return {
    meta: { file, fileBytes: 0, format: "onnx", ir: 8, opsets: { "ai.onnx": 17 }, producer: "synthetic", producerName: "synthetic", producerVersion: "", domain: "", modelVersion: null, docString: "", graphName: "main", props: {}, ops: [], params: 0, initializers: 0, weightBytes: {}, nodeCount: 0, functions: 0, extras: [] },
    graphs: [], tensors: [], warnings: [],
  };
}

function finish(m: ModelView): ModelView {
  const c = new Map<string, number>();
  for (const g of m.graphs) for (const n of g.nodes) c.set(n.op, (c.get(n.op) ?? 0) + 1);
  m.meta.ops = [...c].sort((a, b) => b[1] - a[1]);
  m.meta.nodeCount = m.graphs.reduce((a, g) => a + g.nodes.length, 0);
  m.meta.params = m.tensors.reduce((a, t) => a + t.n, 0);
  m.meta.initializers = m.tensors.length;
  return m;
}

function resBlock(b: Builder, x: string, c: number, hw: number, i: number, long = false): string {
  const t = f32([1, c, hw, hw]);
  const p = long ? `stage_${i}/very/long/module/path/block_${i}/` : `b${i}_`;
  const conv = (inp: string, k: number): string => b.add("Conv", [inp], `${p}conv${k}`, t, { name: `${p}conv${k}`, consts: [["W", [c, c, 3, 3]], ["B", [c]]], attrs: [ia("kernel_shape", [3, 3]), ia("pads", [1, 1, 1, 1]), ia("strides", [1, 1]), ii("group", 1)] });
  const bn = (inp: string, k: number): string => b.add("BatchNormalization", [inp], `${p}bn${k}`, t, { name: `${p}bn${k}`, consts: [["scale", [c]], ["B", [c]], ["mean", [c]], ["var", [c]]] });
  const a = bn(conv(x, 1), 1);
  const r = b.add("Relu", [a], `${p}relu1`, t);
  const d = bn(conv(r, 2), 2);
  const s = b.add("Add", [d, x], `${p}add`, t);
  return b.add("Relu", [s], `${p}relu2`, t);
}

/** Small ResNet-like model + an If node (2 subgraphs) + a call to a local function. */
export function syntheticSmall(blocks = 6): ModelView {
  const m = emptyModel("synthetic-resnet.onnx");
  const b = new Builder(m, 0, "resnet_like", "main");
  b.input("images", f32(["N", 3, 224, 224]));
  b.input("flag", { kind: "tensor", dtype: "bool", shape: [] });
  let x = b.add("Conv", ["images"], "stem_conv", f32(["N", 64, 112, 112]), { name: "stem/conv", consts: [["W", [64, 3, 7, 7]], ["B", [64]]], attrs: [ia("kernel_shape", [7, 7]), ia("strides", [2, 2]), ia("pads", [3, 3, 3, 3])] });
  x = b.add("Relu", [x], "stem_relu", f32(["N", 64, 112, 112]));
  x = b.add("MaxPool", [x], "stem_pool", f32(["N", 64, 56, 56]), { attrs: [ia("kernel_shape", [3, 3]), ia("strides", [2, 2])] });
  for (let i = 0; i < blocks; i++) x = resBlock(b, x, 64, 56, i, i === 2);
  // control flow
  const y = b.add("If", ["flag"], "if_out", f32(["N", 64, 56, 56]), { name: "branch/If", subgraphs: [{ attr: "then_branch", graph: 1 }, { attr: "else_branch", graph: 2 }] });
  x = b.add("Add", [x, y], "merge", f32(["N", 64, 56, 56]));
  x = b.add("Custom_fn_call", [x], "fn_out", f32(["N", 64, 56, 56]), { name: "local_fn_call", domain: "my.domain", fn: 3 });
  x = b.add("GlobalAveragePool", [x], "gap", f32(["N", 64, 1, 1]));
  x = b.add("Flatten", [x], "flat", f32(["N", 64]), { attrs: [ii("axis", 1)] });
  x = b.add("Gemm", [x], "logits", f32(["N", 1000]), { consts: [["B", [1000, 64]], ["C", [1000]]], attrs: [ii("transB", 1)] });
  const seq = b.add("SequenceConstruct", [x, x], "seq", { kind: "sequence", elem: f32(["N", 1000]) });
  b.output("logits", f32(["N", 1000]));
  b.output(seq, { kind: "sequence", elem: f32(["N", 1000]) });

  // then / else subgraphs capture the outer value "stem_pool"
  for (const [id, name, op] of [[1, "then_graph", "Relu"], [2, "else_graph", "Sigmoid"]] as const) {
    const s = new Builder(m, id, name, "subgraph", { graph: 0, node: b.graph.nodes.findIndex((n) => n.op === "If"), attr: id === 1 ? "then_branch" : "else_branch" });
    s.graph.values["stem_pool"] = { name: "stem_pool", type: f32(["N", 64, 56, 56]), producer: -1, consumers: [], tid: null, outer: true };
    const t = s.add(op, ["stem_pool"], `${name}_a`, f32(["N", 64, 56, 56]));
    s.add("Identity", [t], `${name}_out`, f32(["N", 64, 56, 56]));
    s.output(`${name}_out`, f32(["N", 64, 56, 56]));
  }
  const fn = new Builder(m, 3, "local_fn", "function");
  fn.graph.fnName = "Custom_fn_call"; fn.graph.fnDomain = "my.domain";
  fn.input("X", f32(["N", 64, 56, 56]));
  const a = fn.add("Mul", ["X", "X"], "sq", f32(["N", 64, 56, 56]));
  fn.add("Sqrt", [a], "Y", f32(["N", 64, 56, 56]));
  fn.output("Y", f32(["N", 64, 56, 56]));
  return finish(m);
}

/** ~n nodes: stacks of inception-like parallel branches joined by Concat. */
export function syntheticBig(n = 3000): ModelView {
  const m = emptyModel(`synthetic-${n}.onnx`);
  const b = new Builder(m, 0, "big", "main");
  const r = rng(7);
  b.input("input", f32([1, 32, 64, 64]));
  let x = "input", i = 0;
  const t = f32([1, 32, 64, 64]);
  while (b.graph.nodes.length < n) {
    const branches = 1 + Math.floor(r() * 3);
    const outs: string[] = [];
    for (let k = 0; k < branches; k++) {
      let y = x;
      const depth = 1 + Math.floor(r() * 3);
      for (let d = 0; d < depth; d++) {
        const op = r() < 0.5 ? "Conv" : r() < 0.5 ? "Relu" : "BatchNormalization";
        y = b.add(op, [y], `v${i++}`, t, op === "Conv" ? { consts: [["W", [32, 32, 3, 3]], ["B", [32]]] } : op === "BatchNormalization" ? { consts: [["scale", [32]], ["B", [32]], ["mean", [32]], ["var", [32]]] } : {});
      }
      outs.push(y);
    }
    x = outs.length > 1 ? b.add("Concat", outs, `v${i++}`, t, { attrs: [ii("axis", 1)] }) : outs[0];
    if (r() < 0.3) x = b.add("Add", [x, outs[0]], `v${i++}`, t);
  }
  b.output(x, t);
  return finish(m);
}
