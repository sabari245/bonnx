/// <reference types="node" />
import { afterEach, describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { loadModel, type Loaded } from "@/onnx/load";
import { decodeModel } from "@/onnx/decode";
import { Warnings, ProtoError } from "@/onnx/proto";
import { detectFormat, parseTextModel } from "@/onnx/text";
import { __forceSlowPath } from "@/onnx/tensor";
import { lookupSchema } from "@/onnx/schema";
import type { TensorInfo, TensorSliceResult } from "@/onnx/types";

const fx = (n: string) => new URL(`./fixtures/${n}`, import.meta.url);
const ab = (n: string): ArrayBuffer => {
  const b = readFileSync(fx(n));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};
const open = (n: string, external?: Record<string, ArrayBuffer>) => loadModel({ name: n, data: ab(n), external });
const openText = (n: string) => loadModel({ name: n, data: readFileSync(fx(n), "utf8") });

const tensor = (l: Loaded, name: string): TensorInfo => {
  const t = l.view.tensors.find((x) => x.name === name);
  if (!t) throw new Error(`no tensor ${name}: ${l.view.tensors.map((x) => x.name)}`);
  return t;
};
const values = (l: Loaded, name: string): number[] | string[] => {
  const r = l.store.slice({ id: tensor(l, name).id });
  return r.values instanceof Float64Array ? Array.from(r.values) : r.values;
};
const slice = (l: Loaded, name: string): TensorSliceResult => l.store.slice({ id: tensor(l, name).id });

afterEach(() => __forceSlowPath(false));

describe("fixtures exist", () => {
  it("were generated", () => expect(existsSync(fx("if.onnx"))).toBe(true));
});

describe("control flow", () => {
  it("If: both branches become subgraphs and capture outer values", async () => {
    const { view } = await open("if.onnx");
    expect(view.graphs).toHaveLength(3);
    const main = view.graphs[0]!;
    expect(main.nodes.map((n) => n.op)).toEqual(["If"]);
    const iff = main.nodes[0]!;
    expect(iff.subgraphs.map((s) => s.attr).sort()).toEqual(["else_branch", "then_branch"]);
    expect(iff.attrs.map((a) => a.value.t)).toEqual(["graph", "graph"]);
    const then = view.graphs[iff.subgraphs.find((s) => s.attr === "then_branch")!.graph]!;
    expect(then.kind).toBe("subgraph");
    expect(then.parent).toEqual({ graph: 0, node: 0, attr: "then_branch" });
    expect(then.depth).toBe(1);
    expect(then.nodes.map((n) => n.name)).toEqual(["then_add"]);
    const w = then.values["w"]!;
    expect(w.outer).toBe(true);
    expect(w.tid).not.toBeNull();
    expect(then.nodes[0]!.inputs[1]).toMatchObject({ name: "w", tid: w.tid, param: "B" });
    expect(then.values["x"]!.outer).toBe(true);
    expect(then.values["x"]!.type).toMatchObject({ kind: "tensor", dtype: "float32", shape: [2] });
    expect(iff.inputs[0]!.param).toBe("cond");
    expect(view.meta.extras).toContain("subgraphs");
    expect(view.meta.nodeCount).toBe(3);
  });

  it("Loop: body graph + sequence typed carried dependency", async () => {
    const { view } = await open("loop_seq.onnx");
    const main = view.graphs[0]!;
    const loop = main.nodes.find((n) => n.op === "Loop")!;
    const body = view.graphs[loop.subgraphs[0]!.graph]!;
    expect(body.inputs.map((i) => i.name)).toEqual(["iter", "cond_in", "seq_in"]);
    expect(body.inputs[2]!.type).toEqual({ kind: "sequence", elem: { kind: "tensor", dtype: "float32", shape: [3] } });
    expect(main.outputs[0]!.type).toEqual({ kind: "sequence", elem: { kind: "tensor", dtype: "float32", shape: [3] } });
    expect(main.values["seq0"]!.type?.kind).toBe("sequence");
    expect(body.nodes.map((n) => n.op)).toEqual(["SequenceInsert", "Identity"]);
  });
});

describe("sparse", () => {
  it("sparse_initializer with linear indices", async () => {
    const l = await open("sparse.onnx");
    const t = tensor(l, "sp_w");
    expect(t).toMatchObject({ source: "sparse_initializer", dims: [4, 4], n: 16, dtype: "float32", available: true });
    expect(t.sparse!.nnz).toBe(3);
    expect(l.view.graphs[0]!.initializers).toContain(t.id);
    expect(l.view.graphs[0]!.initializers).not.toContain(t.sparse!.valuesTid);
    expect(l.view.meta.extras).toContain("sparse_initializer");
    const dense = Array.from(slice(l, "sp_w").values as Float64Array);
    expect(dense).toEqual([0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0]);
    expect(t.summary).toMatchObject({ max: 5, min: 0, zeros: 13 / 16 });
    // sub-slice
    const s = l.store.slice({ id: t.id, offset: [1, 1], size: [2, 2] });
    expect(Array.from(s.values as Float64Array)).toEqual([4, 0, 0, 5]);
  });

  it("sparse_initializer with [nnz, rank] coordinates", async () => {
    const l = await open("sparse2d.onnx");
    const dense = Array.from(slice(l, "sp2").values as Float64Array);
    expect(dense[1]).toBe(7);
    expect(dense[15]).toBe(8);
    expect(dense.filter((x) => x !== 0)).toHaveLength(2);
  });

  it("is dense-counted in params? (float sparse counts dense elements)", async () => {
    const { view } = await open("sparse.onnx");
    expect(view.meta.params).toBe(16);
    expect(view.meta.initializers).toBe(1);
  });
});

describe("types", () => {
  it("sequence / map / optional value types", async () => {
    const { view } = await open("types.onnx");
    const g = view.graphs[0]!;
    const t = Object.fromEntries(g.inputs.map((i) => [i.name, i.type]));
    expect(t.seq).toEqual({ kind: "sequence", elem: { kind: "tensor", dtype: "float32", shape: ["N", 3] } });
    expect(t.m).toEqual({ kind: "map", key: "string", value: { kind: "tensor", dtype: "float32", shape: [] } });
    expect(t.o).toEqual({ kind: "optional", elem: { kind: "tensor", dtype: "int32", shape: [2, "B"] } });
    expect(g.nodes[0]!.outputs[0]!.type?.kind).toBe("sequence");
  });
});

describe("external data", () => {
  it("is flagged unavailable, then resolved by provideExternal", async () => {
    const l = await open("external.onnx");
    const t = tensor(l, "W");
    expect(t.external).toMatchObject({ location: "external.bin" });
    expect(t.available).toBe(false);
    expect(t.summary).toBeUndefined();
    expect(l.view.warnings.some((w) => w.includes("external.bin"))).toBe(true);
    expect(l.view.meta.extras).toContain("external_data");

    const bin = ab("external.bin");
    l.store.setExternal("./sub/external.bin", new Uint8Array(bin)); // basename match
    l.store.refresh(t.id);
    expect(t.available).toBe(true);
    expect(values(l, "W")).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    expect(t.summary).toMatchObject({ min: 0, max: 11 });
  });

  it("can be supplied at open time", async () => {
    const l = await open("external.onnx", { "external.bin": ab("external.bin") });
    const t = tensor(l, "W");
    expect(t.available).toBe(true);
    expect(l.view.warnings).toEqual([]);
    expect(t.summary!.mean).toBeCloseTo(5.5);
  });
});

describe("dtypes", () => {
  it("decodes every dtype", async () => {
    const l = await open("dtypes.onnx");
    expect(values(l, "f16")).toEqual([1, -2, 0.5, 65504]);
    expect(values(l, "bf16")).toEqual([1, -2.5, 3]);
    expect(values(l, "i4")).toEqual([-8, -1, 0, 7]);
    expect(values(l, "u4")).toEqual([0, 7, 15]);
    expect(values(l, "flag")).toEqual([1, 0, 1]);
    expect(values(l, "names")).toEqual(["alpha", "beta"]);
    expect(values(l, "f8")).toEqual([1, -2, 0.5]);
    expect(values(l, "i64")).toEqual([1, -2, 2 ** 40]);
    expect(values(l, "u32")).toEqual([7, 4000000000]);
    expect(values(l, "c64")).toEqual([1, 2, 3, 4]);
    expect(values(l, "f16_raw")).toEqual([1.5, -2.5]);
    expect(values(l, "i32_raw")).toEqual([1, 2, 3, 4, 5, 6]);
    expect(tensor(l, "i4")).toMatchObject({ dtype: "int4", bytes: 2, n: 4 });
    expect(tensor(l, "u4")).toMatchObject({ dtype: "uint4", bytes: 2 });
    expect(tensor(l, "names")).toMatchObject({ dtype: "string", n: 2, bytes: 9 });
    expect(tensor(l, "c64")).toMatchObject({ dtype: "complex64", bytes: 16 });
    expect(tensor(l, "f16").summary).toMatchObject({ min: -2, max: 65504 });
    expect(tensor(l, "names").summary).toBeUndefined();
    expect(l.view.meta.weightBytes.float16).toBe(8 + 4);
  });

  it("flags int64 values beyond 2^53 as lossy", async () => {
    const l = await open("dtypes.onnx");
    expect(slice(l, "big").lossy).toBe(true);
    expect(slice(l, "i64").lossy).toBe(false);
  });

  it("DataView (big-endian-safe) paths agree with the fast paths", async () => {
    const fast = await open("dtypes.onnx");
    __forceSlowPath(true);
    const slow = await open("dtypes.onnx");
    for (const n of ["f16_raw", "i32_raw", "i64", "u32", "f16", "bf16", "i4", "f8"]) expect(values(slow, n)).toEqual(values(fast, n));
    const u = await open("unnamed.onnx");
    __forceSlowPath(false);
    const u2 = await open("unnamed.onnx");
    expect(values(u, "fw")).toEqual(values(u2, "fw"));
    expect(values(u, "w")).toEqual(values(u2, "w"));
  });
});

describe("Constant folding", () => {
  it("folds all Constant forms into tensors", async () => {
    const l = await open("constants.onnx");
    const g = l.view.graphs[0]!;
    expect(g.nodes.map((n) => n.op)).toEqual(["Add"]);
    const srcOf = (n: string) => tensor(l, n).source;
    for (const n of ["c_value", "c_float", "c_floats", "c_int", "c_ints", "c_string", "c_strings"]) expect(srcOf(n)).toBe("constant");
    expect(srcOf("c_sparse")).toBe("sparse_attribute");
    expect(values(l, "c_value")).toEqual([1, 2, 3, 4]);
    expect(tensor(l, "c_value").dims).toEqual([2, 2]);
    expect(values(l, "c_float")).toEqual([2.5]);
    expect(tensor(l, "c_float").dims).toEqual([]);
    expect(values(l, "c_floats")).toEqual([1, 2, 3]);
    expect(values(l, "c_int")).toEqual([7]);
    expect(tensor(l, "c_int").dtype).toBe("int64");
    expect(values(l, "c_ints")).toEqual([1, 2, 3, 4]);
    expect(values(l, "c_string")).toEqual(["hello"]);
    expect(values(l, "c_strings")).toEqual(["a", "bc"]);
    expect(Array.from(slice(l, "c_sparse").values as Float64Array)).toEqual([0, 0, 9, 0]);
    // the Add consumes the folded constant
    expect(g.nodes[0]!.inputs[1]).toMatchObject({ name: "c_float", tid: tensor(l, "c_float").id });
    expect(g.initializers).toContain(tensor(l, "c_int").id);
  });
});

describe("functions", () => {
  it("function bodies are graphs; calls link to them", async () => {
    const { view } = await open("function.onnx");
    expect(view.graphs).toHaveLength(2);
    const fn = view.graphs[1]!;
    expect(fn).toMatchObject({ kind: "function", fnDomain: "local", fnName: "MyAddRelu", depth: 1 });
    expect(fn.name).toBe("local.MyAddRelu");
    expect(fn.nodes.map((n) => n.op)).toEqual(["Add", "Relu"]);
    expect(fn.inputs.map((i) => i.name)).toEqual(["a", "b"]);
    const call = view.graphs[0]!.nodes[0]!;
    expect(call).toMatchObject({ op: "MyAddRelu", domain: "local", fn: 1, opset: 1 });
    expect(call.inputs.map((i) => i.param)).toEqual(["a", "b"]);
    expect(call.outputs[0]!.param).toBe("out");
    expect(view.meta.functions).toBe(1);
    expect(view.meta.opsets).toMatchObject({ "ai.onnx": 21, local: 1 });
    expect(view.meta.nodeCount).toBe(3);
  });
});

describe("quantization", () => {
  it("QDQ nodes: roles and inferred types", async () => {
    const { view } = await open("qdq.onnx");
    const g = view.graphs[0]!;
    expect(g.nodes.map((n) => n.op)).toEqual(["QuantizeLinear", "DequantizeLinear", "Relu"]);
    expect(g.nodes[0]!.inputs.map((i) => i.param)).toEqual(["x", "y_scale", "y_zero_point"]);
    expect(g.nodes[0]!.inputs[1]!.tid).not.toBeNull();
    expect(g.nodes[0]!.outputs[0]!.type).toMatchObject({ dtype: "uint8", shape: [1, 4] });
    expect(g.nodes[1]!.outputs[0]!.type).toMatchObject({ dtype: "float32", shape: [1, 4] });
    expect(g.nodes[2]!.outputs[0]!.type).toMatchObject({ dtype: "float32", shape: [1, 4] });
    expect(g.nodes[0]!.params).toBe(0); // scalars are not parameters
  });
});

describe("attributes", () => {
  it("tensor, graph and type attributes", async () => {
    const { view } = await open("attrs.onnx");
    const g = view.graphs[0]!;
    const cos = g.nodes.find((n) => n.op === "ConstantOfShape")!;
    const a = cos.attrs[0]!.value;
    expect(a.t).toBe("tensor");
    const t = view.tensors[(a as { tid: number }).tid]!;
    expect(t).toMatchObject({ source: "attribute", name: "cos.value", dtype: "float32", node: cos.id });
    const custom = g.nodes.find((n) => n.op === "Custom")!;
    expect(custom.domain).toBe("my.dom");
    expect(custom.attrs.find((x) => x.name === "body")!.value.t).toBe("graph");
    expect(custom.attrs.find((x) => x.name === "ty")!.value).toEqual({ t: "type", type: { kind: "tensor", dtype: "float32", shape: [1, 2] } });
    expect(view.graphs[custom.subgraphs[0]!.graph]!.name).toBe("sub_g");
    // inferred output type for ConstantOfShape is unknown but must not crash
    expect(cos.outputs[0]!.name).toBe("filled");
  });
});

describe("unnamed nodes + shape inference", () => {
  it("synthesizes names and infers missing shapes", async () => {
    const { view } = await open("unnamed.onnx");
    const g = view.graphs[0]!;
    expect(g.nodes.map((n) => n.name)).toEqual(["Conv_0", "Relu_1", "MaxPool_2", "Flatten_3", "Gemm_4"]);
    const shape = (n: number) => (g.nodes[n]!.outputs[0]!.type as { shape: number[] }).shape;
    expect(shape(0)).toEqual([1, 4, 8, 8]);
    expect(shape(1)).toEqual([1, 4, 8, 8]);
    expect(shape(2)).toEqual([1, 4, 4, 4]);
    expect(shape(3)).toEqual([1, 64]);
    expect(shape(4)).toEqual([1, 5]);
    expect(g.nodes[0]!.inputs.map((i) => i.param)).toEqual(["X", "W", "B"]);
    // initializer that is also a graph input keeps the tensor id
    expect(g.inputs.find((i) => i.name === "w")!.tid).not.toBeNull();
    expect(g.inputs.find((i) => i.name === "x")!.tid).toBeNull();
    expect(g.nodes[0]!.params).toBe(4 * 27 + 4);
    expect(g.nodes[0]!.wmax).toBeGreaterThan(0);
  });
});

describe("text and JSON formats", () => {
  it("detectFormat", () => {
    expect(detectFormat("a.onnx", ab("unnamed.onnx"))).toBe("onnx");
    expect(detectFormat("a.onnx.prototxt", readFileSync(fx("unnamed.onnx.prototxt"), "utf8"))).toBe("onnx-text");
    expect(detectFormat("m", new Uint8Array(readFileSync(fx("unnamed.onnx.prototxt"))))).toBe("onnx-text");
    expect(detectFormat("m", new Uint8Array(readFileSync(fx("unnamed.json"))))).toBe("onnx-json");
    expect(detectFormat("x", new Uint8Array([0, 0, 0, 0, 0x4f, 0x52, 0x54, 0x4d]))).toBe("ort");
    expect(detectFormat("m.pbtxt", "{}")).toBe("onnx-json");
  });

  it("prototxt parses to the same view as binary", async () => {
    const a = await open("unnamed.onnx");
    const b = await openText("unnamed.onnx.prototxt");
    expect(b.view.meta.format).toBe("onnx-text");
    expect(b.view.graphs[0]!.nodes.map((n) => [n.op, n.name, n.params, n.inputs.map((i) => i.name)])).toEqual(
      a.view.graphs[0]!.nodes.map((n) => [n.op, n.name, n.params, n.inputs.map((i) => i.name)]),
    );
    expect(b.view.graphs[0]!.nodes[0]!.attrs).toEqual(a.view.graphs[0]!.nodes[0]!.attrs);
    expect(b.view.tensors.map((t) => [t.name, t.dtype, t.dims])).toEqual(a.view.tensors.map((t) => [t.name, t.dtype, t.dims]));
    expect(values(b, "w")).toEqual(values(a, "w"));
    expect(b.view.meta.params).toBe(a.view.meta.params);
    expect(b.view.meta.opsets).toEqual(a.view.meta.opsets);
    expect(b.view.graphs[0]!.nodes[2]!.outputs[0]!.type).toEqual(a.view.graphs[0]!.nodes[2]!.outputs[0]!.type);
  });

  it("protobuf JSON parses to the same view as binary", async () => {
    const a = await open("unnamed.onnx");
    const b = await loadModel({ name: "unnamed.json", data: readFileSync(fx("unnamed.json"), "utf8") });
    expect(b.view.meta.format).toBe("onnx-json");
    expect(b.view.graphs[0]!.nodes.map((n) => n.op)).toEqual(a.view.graphs[0]!.nodes.map((n) => n.op));
    expect(values(b, "fw")).toEqual(values(a, "fw"));
    expect(b.view.graphs[0]!.nodes[0]!.attrs).toEqual(a.view.graphs[0]!.nodes[0]!.attrs);
  });

  it("text dtypes (int32_data bit patterns, strings, enums) match binary", async () => {
    const a = await open("dtypes.onnx");
    const b = await openText("dtypes.onnx.prototxt");
    for (const t of a.view.tensors) {
      expect(tensor(b, t.name)).toMatchObject({ dtype: t.dtype, dims: t.dims, n: t.n });
      expect(values(b, t.name)).toEqual(values(a, t.name));
    }
  });

  it("text If model keeps its subgraphs", async () => {
    const b = await openText("if.onnx.prototxt");
    expect(b.view.graphs).toHaveLength(3);
    expect(b.view.graphs.flatMap((g) => g.nodes.map((n) => n.name))).toEqual(expect.arrayContaining(["then_add", "else_sub"]));
  });

  it("handwritten text format: comments, escapes, concatenated strings, lists, extensions", () => {
    const txt = `# a comment
ir_version: 8
producer_name: "hand\\x41\\101\\n"  "written"
opset_import { domain: "" version: 17 }
metadata_props { key: "k" value: 'v' }
graph {
  name: "g"
  node { input: "x" output: "y" op_type: "Relu" attribute { name: "alpha" type: FLOAT f: inf } attribute { name: "ks" type: INTS ints: [1, 2, 3] ints: 4 } }
  input { name: "x" type { tensor_type { elem_type: 1 shape { dim { dim_value: 2 } dim { dim_param: "N" } dim {} } } } }
  output { name: "y" }
  initializer { name: "i" dims: [2] data_type: 7 int64_data: [-1, 9223372036854775807] }
}`;
    const m = parseTextModel(txt, new Warnings());
    expect(m.ir).toBe(8);
    expect(m.producerName).toBe("handAA\nwritten");
    expect(m.opsets[""]).toBe(17);
    expect(m.props.k).toBe("v");
    const n = m.graph!.nodes[0]!;
    expect(n.attrs[0]!.f).toBe(Infinity);
    expect(n.attrs[1]!.ints).toEqual([1, 2, 3, 4]);
    expect(m.graph!.inputs[0]!.type).toEqual({ kind: "tensor", dtype: "float32", shape: [2, "N", "?"] });
    expect(Array.from(m.graph!.initializers[0]!.int64Data!)).toEqual([-1n, 9223372036854775807n]);
  });

  it("text parse errors are ProtoErrors with a line number", () => {
    expect(() => parseTextModel("graph { name: ", new Warnings())).toThrow(/line/);
  });
});

describe("robustness", () => {
  const resnet = () => new Uint8Array(readFileSync(new URL("../models/resnet18-v2-7.onnx", import.meta.url)));

  it("tolerates a truncated file (partial model + warnings)", async () => {
    const full = resnet();
    for (const frac of [0.9, 0.5, 0.1]) {
      const cut = full.slice(0, Math.floor(full.length * frac));
      const { view } = await loadModel({ name: "cut.onnx", data: cut.buffer as ArrayBuffer });
      expect(view.graphs).toHaveLength(1);
      expect(view.warnings.length).toBeGreaterThan(0);
    }
  });

  it("ignores unknown fields", () => {
    const full = resnet();
    const extra = new Uint8Array([0xf8, 0x06, 0x2a, 0xc2, 0x3e, 0x03, 1, 2, 3]); // field 111 varint, field 999 len-delimited
    const buf = new Uint8Array(full.length + extra.length);
    buf.set(full);
    buf.set(extra, full.length);
    const w = new Warnings();
    const m = decodeModel(buf, w);
    expect(m.graph!.nodes.length).toBeGreaterThan(60);
    expect(w.list).toEqual([]);
  });

  it("rejects non-ONNX data with a clear error", async () => {
    await expect(loadModel({ name: "x.onnx", data: new Uint8Array([1, 2, 3, 4, 5]).buffer })).rejects.toThrow(ProtoError);
    await expect(loadModel({ name: "x.onnx", data: new ArrayBuffer(0) })).rejects.toThrow();
  });

  it("rejects ORT flatbuffers with a helpful message", async () => {
    const b = new Uint8Array(32);
    b.set([0x4f, 0x52, 0x54, 0x4d], 4);
    await expect(loadModel({ name: "m.ort", data: b.buffer })).rejects.toThrow(/ORT/);
  });

  it("empty graph / missing shapes never crash", async () => {
    const { view } = await loadModel({ name: "e.prototxt", data: 'ir_version: 8\ngraph { name: "g" node { op_type: "Foo" input: "a" output: "b" } }' });
    expect(view.graphs[0]!.nodes[0]).toMatchObject({ op: "Foo", name: "Foo_0" });
    expect(view.graphs[0]!.nodes[0]!.inputs[0]!.type).toBeNull();
    expect(view.graphs[0]!.values["a"]!.producer).toBe(-1);
  });
});

describe("stats / slices / histograms (resnet18)", () => {
  let l: Loaded;
  const floats = (name: string): Float32Array => {
    const e = l.store.entries[tensor(l, name).id]!;
    const b = (e.raw!.raw ?? e.raw!.floatData)!;
    const c = new Uint8Array(b); // aligned copy
    return new Float32Array(c.buffer, 0, c.length / 4);
  };
  const big = async () => (l ??= await loadModel({ name: "r.onnx", data: ab_resnet() }));
  const ab_resnet = () => {
    const b = readFileSync(new URL("../models/resnet18-v2-7.onnx", import.meta.url));
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  };

  it("exact stats", async () => {
    await big();
    const name = l.view.tensors.find((t) => t.dims.length === 4 && t.n > 100_000)!.name;
    const f = floats(name);
    const t = tensor(l, name);
    const st = l.store.stats(t.id);
    const sorted = Float64Array.from(f).sort();
    let sum = 0;
    for (const v of f) sum += v;
    const mean = sum / f.length;
    let ss = 0;
    for (const v of f) ss += (v - mean) ** 2;
    expect(st.min).toBe(sorted[0]);
    expect(st.max).toBe(sorted[sorted.length - 1]);
    expect(st.mean).toBeCloseTo(mean, 8);
    expect(st.std).toBeCloseTo(Math.sqrt(ss / f.length), 6);
    const at = (p: number) => {
      const x = (sorted.length - 1) * p;
      const lo = Math.floor(x);
      return sorted[lo]! + (sorted[Math.ceil(x)]! - sorted[lo]!) * (x - lo);
    };
    expect(st.median).toBeCloseTo(at(0.5), 8);
    expect(st.p1).toBeCloseTo(at(0.01), 8);
    expect(st.p99).toBeCloseTo(at(0.99), 8);
    expect(st.unique).toBe(new Set(f).size);
    expect(st.hist!.reduce((a, b) => a + b, 0)).toBe(f.length);
    expect(st.hist).toHaveLength(64);
    expect(st.channelAbsMax).toHaveLength(t.dims[0]!);
    const ch0 = Math.max(...Array.from(f.subarray(0, f.length / t.dims[0]!)).map(Math.abs));
    expect(st.channelAbsMax![0]).toBeCloseTo(ch0, 6);
    expect(st.sampled).toBe(false);
    expect(st.absmax).toBeCloseTo(Math.max(-st.min!, st.max!), 8);
  });

  it("slices are exact sub-rectangles and clip to maxElems", async () => {
    await big();
    const t = l.view.tensors.find((x) => x.dims.length === 4 && x.dims[2] === 3)!;
    const [o, i, h, w] = t.dims as [number, number, number, number];
    const f = floats(t.name);
    const s = l.store.slice({ id: t.id, offset: [1, 2, 0, 1], size: [2, 3, 3, 2] });
    expect(s.size).toEqual([2, 3, 3, 2]);
    const expect_: number[] = [];
    for (let a = 1; a < 3; a++) for (let b = 2; b < 5; b++) for (let c = 0; c < 3; c++) for (let d = 1; d < 3; d++) expect_.push(f[((a * i + b) * h + c) * w + d]!);
    expect(Array.from(s.values as Float64Array)).toEqual(expect_);
    // merged contiguous trailing dims path
    const s2 = l.store.slice({ id: t.id, offset: [3], size: [1] });
    expect(Array.from(s2.values as Float64Array)).toEqual(Array.from(f.subarray(3 * i * h * w, 4 * i * h * w)));
    expect(s2.size).toEqual([1, i, h, w]);
    // clipping
    const c = l.store.slice({ id: t.id, maxElems: 1000 });
    expect(c.truncated).toBe(true);
    expect((c.values as Float64Array).length).toBeLessThanOrEqual(1000);
    expect(c.size[0]).toBeLessThan(o);
    // out-of-range offsets clamp, empty size
    const e = l.store.slice({ id: t.id, offset: [o + 5], size: [2] });
    expect((e.values as Float64Array).length).toBeLessThanOrEqual(i * h * w);
  });

  it("custom-range and log histograms", async () => {
    await big();
    const t = l.view.tensors.find((x) => x.dims.length === 4 && x.n > 100_000)!;
    const st = l.store.stats(t.id);
    const h = l.store.histogram({ id: t.id, bins: 10, min: -0.1, max: 0.1 });
    expect(h.counts).toHaveLength(10);
    expect(h.counts.reduce((a, b) => a + b, 0) + h.below + h.above).toBe(t.n);
    const lg = l.store.histogram({ id: t.id, bins: 8, min: 1e-4, max: st.absmax!, log: true });
    expect(lg.counts.reduce((a, b) => a + b, 0) + lg.below + lg.above).toBe(t.n);
    const dflt = l.store.histogram({ id: t.id, bins: 5 });
    expect(dflt.min).toBe(st.min);
    expect(dflt.counts.reduce((a, b) => a + b, 0)).toBe(t.n);
  });

  it("string and complex tensors don't break stats", async () => {
    const d = await open("dtypes.onnx");
    expect(d.store.stats(tensor(d, "names").id).unique).toBe(2);
    expect(d.store.stats(tensor(d, "c64").id).min).toBeNull();
    expect(d.store.stats(tensor(d, "f16").id).unique).toBe(4);
  });
});

describe("operator schema", () => {
  it("looks up by opset version", async () => {
    const c21 = (await lookupSchema("", "Conv", 21))!;
    expect(c21.name).toBe("Conv");
    expect(c21.sinceVersion).toBe(11);
    expect(c21.inputs.map((i) => i.name)).toEqual(["X", "W", "B"]);
    expect(c21.inputs[2]!.option).toBe("optional");
    expect(c21.attributes.map((a) => a.name)).toContain("strides");
    expect(c21.doc.length).toBeGreaterThan(20);
    expect(c21.typeConstraints[0]!.types).toContain("tensor(float)");
    const c1 = (await lookupSchema("ai.onnx", "Conv", 5))!;
    expect(c1.sinceVersion).toBe(1);
    expect(c1.doc.length).toBeGreaterThan(20); // carried forward from an older record
    expect((await lookupSchema("", "Concat", 21))!.inputs[0]!.option).toBe("variadic");
    expect(await lookupSchema("com.nope", "X", 1)).toBeNull();
    expect((await lookupSchema("ai.onnx.ml", "LinearClassifier", 1))!.domain).toBe("ai.onnx.ml");
  });

  it("variadic params are named name[k]", async () => {
    const l = await loadModel({ name: "m.prototxt", data: 'ir_version: 8 opset_import { domain: "" version: 21 } graph { node { op_type: "Concat" input: "a" input: "b" input: "c" output: "d" attribute { name: "axis" type: INT i: 0 } } }' });
    expect(l.view.graphs[0]!.nodes[0]!.inputs.map((i) => i.param)).toEqual(["inputs[0]", "inputs[1]", "inputs[2]"]);
    expect(l.view.graphs[0]!.nodes[0]!.outputs[0]!.param).toBe("concat_result");
  });
});
