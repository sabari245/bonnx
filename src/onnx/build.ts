/**
 * RawModel → ModelView (+ tensor registrations in a TensorStore).
 * - main graph, nested subgraphs and model-local functions become GraphViews
 * - Constant nodes are folded into tensors
 * - missing shapes are filled by a small best-effort inference pass
 */
import type { RawAttr, RawFunction, RawGraph, RawModel, RawNode, RawSparse, RawTensor, RawValueInfo } from "./decode";
import { ATTR } from "./decode";
import { dtypeName, isFloatDType } from "./dtype";
import { inferOutputs } from "./infer";
import { nameAt, staticRoles, toFormals } from "./roles";
import { lookupSchemaSync, normDomain, type Metadata } from "./schema";
import { denseBytes, type TensorSource, type TensorStore } from "./tensor";
import type { Warnings } from "./proto";
import type {
  AttrValue, AttrView, GraphIO, GraphView, ModelMeta, ModelView, NodeInput, NodeOutput, NodeView, TensorInfo,
  ValueInfo, ValueType,
} from "./types";

export interface BuildOptions {
  fileName: string;
  fileBytes: number;
  format: string;
  warnings: Warnings;
  store: TensorStore;
  metadata?: Metadata | null;
  onProgress?: (stage: string, frac: number) => void;
}

interface Decl {
  type: ValueType | null;
  tid: number | null;
}
interface Ctx {
  decl: Map<string, Decl>;
  parent: Ctx | null;
}

const isDefaultDomain = (d: string) => d === "" || d === "ai.onnx";
const normD = (d: string) => (d === "ai.onnx" ? "" : d);
const utf8Fatal = new TextDecoder("utf-8", { fatal: true });

const prod = (d: number[]): number => {
  let n = 1;
  for (const x of d) n *= x > 0 ? x : x === 0 ? 0 : 0;
  return n;
};

function hex(b: Uint8Array): string {
  let s = "";
  for (let i = 0; i < b.length && i < 4096; i++) s += b[i]!.toString(16).padStart(2, "0");
  return s;
}

function scalarTensor(dataType: number, dims: number[]): RawTensor {
  return {
    name: "", doc: "", dims, dataType, raw: null, floatData: null, doubleData: null, int32Data: null, int64Data: null,
    uint64Data: null, stringData: null, external: null, dataLocationExternal: false,
  };
}

function inferDataType(t: RawTensor): number {
  if (t.dataType) return t.dataType;
  if (t.floatData) return 1;
  if (t.doubleData) return 11;
  if (t.int64Data) return 7;
  if (t.uint64Data) return 13;
  if (t.int32Data) return 6;
  if (t.stringData) return 8;
  return 0;
}

const tensorType = (dtype: string, dims: number[]): ValueType => ({ kind: "tensor", dtype, shape: dims });

class Builder {
  graphs: GraphView[] = [];
  private nextId: number;
  private fnByKey = new Map<string, number>();
  private fnRaw: RawFunction[];
  private modelOpsets: Record<string, number> = {};
  /** auxiliary tensors (sparse values/indices) excluded from counts */
  aux = new Set<number>();
  private formalCache = new Map<string, { in: [string, boolean][] | null; out: [string, boolean][] | null }>();

  constructor(
    private raw: RawModel,
    private o: BuildOptions,
  ) {
    for (const [d, v] of Object.entries(raw.opsets)) this.modelOpsets[normD(d)] = v;
    this.fnRaw = raw.functions;
    raw.functions.forEach((f, i) => {
      const id = 1 + i;
      const k = `${normD(f.domain)}|${f.name}`;
      if (!this.fnByKey.has(k)) this.fnByKey.set(k, id);
      if (f.overload) this.fnByKey.set(`${k}|${f.overload}`, id);
    });
    this.nextId = 1 + raw.functions.length;
  }

  /* ───────── tensors ───────── */

  addTensor(
    t: RawTensor, source: TensorInfo["source"], graph: number, name: string, node?: number,
  ): TensorInfo {
    const dt = inferDataType(t);
    const dtype = dtypeName(dt);
    const n = prod(t.dims);
    const id = this.o.store.add({
      raw: t, sparse: null,
      info: {
        name, dtype, dims: t.dims.slice(), n, bytes: denseBytes(dtype, n, t), source, graph, node,
        docString: t.doc || undefined,
        external: t.external ? { location: t.external.location, offset: t.external.offset, length: t.external.length } : null,
        available: false,
      },
    });
    return this.o.store.entries[id]!.info;
  }

  addSparse(sp: RawSparse, source: "sparse_initializer" | "sparse_attribute", graph: number, name: string, node?: number): TensorInfo {
    const vals = this.addTensor(sp.values, "sparse_initializer", graph, `${name}.values`, node);
    const idx = this.addTensor(sp.indices, "sparse_initializer", graph, `${name}.indices`, node);
    this.aux.add(vals.id);
    this.aux.add(idx.id);
    const n = prod(sp.dims);
    const dtype = vals.dtype;
    const id = this.o.store.add({
      raw: null, sparse: sp,
      info: {
        name, dtype, dims: sp.dims.slice(), n, bytes: denseBytes(dtype, n, null), source, graph, node, available: false,
        sparse: { nnz: vals.n, valuesTid: vals.id, indicesTid: idx.id },
      },
    });
    return this.o.store.entries[id]!.info;
  }

  /* ───────── schema / roles ───────── */

  private opsetFor(opsets: Record<string, number>, domain: string): number | null {
    return opsets[normD(domain)] ?? this.modelOpsets[normD(domain)] ?? null;
  }

  private formals(domain: string, op: string, opset: number | null) {
    const key = `${domain}|${op}|${opset}`;
    let r = this.formalCache.get(key);
    if (r) return r;
    let fi: [string, boolean][] | null = null;
    let fo: [string, boolean][] | null = null;
    const meta = this.o.metadata;
    const s = meta ? lookupSchemaSync(meta, domain, op, opset) : null;
    if (s) {
      fi = s.inputs.map((p) => [p.name, p.option === "variadic"]);
      fo = s.outputs.map((p) => [p.name, p.option === "variadic"]);
    } else {
      const st = staticRoles(domain, op);
      if (st) {
        fi = toFormals(st.inputs);
        fo = toFormals(st.outputs);
      }
    }
    r = { in: fi, out: fo };
    this.formalCache.set(key, r);
    return r;
  }

  /* ───────── attributes ───────── */

  private attr(
    a: RawAttr, ctx: Ctx, gid: number, nodeId: number, nodeName: string, opsets: Record<string, number>, depth: number,
    sink: NodeView["subgraphs"],
  ): AttrValue {
    if (a.ref) return { t: "ref", ref: a.ref };
    const tname = (k?: number) => `${nodeName}.${a.name}${k === undefined ? "" : `[${k}]`}`;
    const sub = (g: RawGraph, k?: number): number => {
      const id = this.nextId++;
      this.addGraph(
        id, g,
        { kind: "subgraph", parent: { graph: gid, node: nodeId, attr: a.name }, depth: depth + 1, name: g.name || tname(k), opsets },
        ctx,
      );
      sink.push({ attr: a.name, graph: id });
      return id;
    };
    switch (a.type) {
      case ATTR.FLOAT: return { t: "float", v: a.f ?? 0 };
      case ATTR.INT: return { t: "int", v: a.i ?? 0 };
      case ATTR.STRING: return strValue(a.s ?? new Uint8Array());
      case ATTR.TENSOR: return a.t ? { t: "tensor", tid: this.addTensor(a.t, "attribute", gid, tname(), nodeId).id } : { t: "undefined" };
      case ATTR.TENSORS: return { t: "tensors", tids: (a.tensors ?? []).map((t, k) => this.addTensor(t, "attribute", gid, tname(k), nodeId).id) };
      case ATTR.GRAPH: return a.g ? { t: "graph", graph: sub(a.g) } : { t: "undefined" };
      case ATTR.GRAPHS: return { t: "graphs", graphs: (a.graphs ?? []).map((g, k) => sub(g, k)) };
      case ATTR.FLOATS: return { t: "floats", v: a.floats ?? [] };
      case ATTR.INTS: return { t: "ints", v: a.ints ?? [] };
      case ATTR.STRINGS: return { t: "strings", v: (a.strings ?? []).map((b) => (strValue(b) as { v: string }).v) };
      case ATTR.SPARSE_TENSOR: return a.st ? { t: "sparse_tensor", tid: this.addSparse(a.st, "sparse_attribute", gid, tname(), nodeId).id } : { t: "undefined" };
      case ATTR.SPARSE_TENSORS: return { t: "tensors", tids: (a.sts ?? []).map((s, k) => this.addSparse(s, "sparse_attribute", gid, tname(k), nodeId).id) };
      case ATTR.TYPE_PROTO: return a.tp ? { t: "type", type: a.tp } : { t: "undefined" };
      case ATTR.TYPE_PROTOS: return { t: "types", types: a.tps ?? [] };
      default: return { t: "undefined" };
    }
  }

  /* ───────── graphs ───────── */

  addGraph(
    id: number, g: RawGraph,
    spec: {
      kind: GraphView["kind"]; parent: GraphView["parent"]; depth: number; name: string; opsets: Record<string, number>;
      fn?: RawFunction;
    },
    outer: Ctx | null,
  ): void {
    const view: GraphView = {
      id, name: spec.name, kind: spec.kind, parent: spec.parent, docString: g.doc || undefined, nodes: [], inputs: [], outputs: [],
      values: Object.create(null) as Record<string, ValueInfo>, initializers: [], depth: spec.depth,
    };
    if (spec.fn) {
      view.fnDomain = spec.fn.domain;
      view.fnName = spec.fn.name;
      if (spec.fn.overload) view.fnOverload = spec.fn.overload;
      view.fnAttrs = spec.fn.attrs.slice();
    }
    this.graphs[id] = view;
    const ctx: Ctx = { decl: new Map(), parent: outer };

    // declared value types
    const declared = new Map<string, RawValueInfo>();
    for (const v of [...g.valueInfo, ...g.outputs, ...g.inputs]) {
      const prev = declared.get(v.name);
      if (!prev || (!prev.type && v.type)) declared.set(v.name, v);
    }
    const dtype = (name: string): ValueType | null => declared.get(name)?.type ?? null;

    // initializers and folded constants
    const initByName = new Map<string, number>();
    for (const t of g.initializers) {
      const info = this.addTensor(t, "initializer", id, t.name);
      view.initializers.push(info.id);
      initByName.set(t.name, info.id);
      ctx.decl.set(t.name, { type: tensorType(info.dtype, info.dims), tid: info.id });
    }
    for (const sp of g.sparseInitializers) {
      const name = sp.values.name || "sparse";
      const info = this.addSparse(sp, "sparse_initializer", id, name);
      view.initializers.push(info.id);
      initByName.set(name, info.id);
      ctx.decl.set(name, { type: { kind: "sparse", dtype: info.dtype, shape: info.dims.slice() }, tid: info.id });
    }
    const folded = new Set<number>();
    g.nodes.forEach((n, i) => {
      if (n.op !== "Constant" || !isDefaultDomain(n.domain) || n.inputs.length || !n.outputs[0]) return;
      const c = this.foldConstant(n, id, i);
      if (!c) return;
      folded.add(i);
      view.initializers.push(c.id);
      const ty: ValueType = c.sparse ? { kind: "sparse", dtype: c.dtype, shape: c.dims.slice() } : tensorType(c.dtype, c.dims);
      ctx.decl.set(n.outputs[0], { type: ty, tid: c.id });
    });
    for (const v of g.inputs) {
      const tid = initByName.get(v.name) ?? null;
      ctx.decl.set(v.name, { type: v.type ?? (tid != null ? ctx.decl.get(v.name)?.type ?? null : null), tid });
    }
    g.nodes.forEach((n, i) => {
      if (folded.has(i)) return;
      for (const o of n.outputs) if (o && !ctx.decl.has(o)) ctx.decl.set(o, { type: dtype(o), tid: null });
    });

    const values = view.values;
    const val = (name: string): ValueInfo => {
      let v = values[name];
      if (!v) {
        const d = ctx.decl.get(name);
        v = values[name] = { name, type: d?.type ?? null, producer: -1, consumers: [], tid: d?.tid ?? null, outer: false };
        const dv = declared.get(name);
        if (dv?.doc) v.docString = dv.doc;
      }
      return v;
    };
    for (const v of g.inputs) val(v.name);
    for (const [name, tid] of initByName) val(name).tid = tid;

    // nodes
    g.nodes.forEach((n, src) => {
      if (folded.has(src)) return;
      const nid = view.nodes.length;
      const opset = this.opsetFor(spec.opsets, n.domain);
      const fnId = this.lookupFn(n);
      const fnRaw = fnId != null ? this.fnRaw[fnId - 1]! : null;
      const f = this.formals(n.domain, n.op, opset);
      const fin = fnRaw ? fnRaw.inputs.map((x): [string, boolean] => [x, false]) : f.in;
      const fout = fnRaw ? fnRaw.outputs.map((x): [string, boolean] => [x, false]) : f.out;
      const name = n.name || `${n.op}_${src}`;

      const inputs: NodeInput[] = [];
      n.inputs.forEach((nm, i) => {
        if (!nm) return;
        const local = ctx.decl.has(nm);
        let d: Decl | null = ctx.decl.get(nm) ?? null;
        let outer = false;
        if (!d) {
          for (let c = ctx.parent; c; c = c.parent) {
            const x = c.decl.get(nm);
            if (x) {
              d = x;
              outer = true;
              break;
            }
          }
        }
        const vi = val(nm);
        if (outer || (!local && d)) {
          vi.outer = true;
          vi.type ??= d?.type ?? null;
          vi.tid ??= d?.tid ?? null;
        }
        if (vi.consumers[vi.consumers.length - 1] !== nid) vi.consumers.push(nid);
        inputs.push({ param: nameAt(fin, i, n.inputs.length, "in"), name: nm, index: i, type: d?.type ?? vi.type, tid: d?.tid ?? null });
      });

      const attrs: AttrView[] = [];
      const subgraphs: NodeView["subgraphs"] = [];
      for (const a of n.attrs) {
        const av: AttrView = { name: a.name, value: this.attr(a, ctx, id, nid, name, spec.opsets, spec.depth, subgraphs) };
        if (a.doc) av.docString = a.doc;
        attrs.push(av);
      }

      // output types: declared, else inferred
      const missing = n.outputs.some((o) => o && !ctx.decl.get(o)?.type);
      if (missing && !fnRaw && n.outputs.length) {
        try {
          const amap = new Map(n.attrs.map((a) => [a.name, a]));
          const inferred = inferOutputs({
            op: n.op, ins: n.inputs.map((x) => (x ? ctx.decl.get(x)?.type ?? this.outerType(ctx, x) : null)), attrs: amap,
            nOut: n.outputs.length, constInts: (i) => this.constInts(ctx, n.inputs[i]),
          });
          if (inferred && isDefaultDomain(n.domain))
            n.outputs.forEach((o, i) => {
              const d = o ? ctx.decl.get(o) : null;
              if (d && !d.type && inferred[i]) d.type = inferred[i]!;
            });
        } catch {
          /* inference is best effort */
        }
      }

      const outputs: NodeOutput[] = [];
      n.outputs.forEach((nm, i) => {
        if (!nm) return;
        const vi = val(nm);
        if (vi.producer >= 0) this.o.warnings.add(`graph "${spec.name}": value "${nm}" is produced by more than one node`);
        vi.producer = nid;
        vi.type = ctx.decl.get(nm)?.type ?? vi.type;
        outputs.push({ param: nameAt(fout, i, n.outputs.length, "out"), name: nm, index: i, type: vi.type });
      });

      const nv: NodeView = {
        id: nid, name, op: n.op, domain: normD(n.domain), opset, inputs, outputs, attrs, fn: fnId, subgraphs, params: 0, wmax: null, src,
      };
      if (n.doc) nv.docString = n.doc;
      if (n.metadata) nv.metadata = n.metadata;
      view.nodes.push(nv);
    });

    // graph io
    const io = (v: RawValueInfo): GraphIO => {
      const vi = val(v.name);
      const out: GraphIO = { name: v.name, type: v.type ?? vi.type, tid: initByName.get(v.name) ?? null };
      if (v.doc) out.docString = v.doc;
      return out;
    };
    view.inputs = g.inputs.map(io);
    view.outputs = g.outputs.map(io);
    for (const o of view.outputs) {
      const vi = val(o.name);
      if (!o.type) o.type = vi.type;
    }
  }

  private outerType(ctx: Ctx, name: string): ValueType | null {
    for (let c = ctx.parent; c; c = c.parent) {
      const d = c.decl.get(name);
      if (d) return d.type;
    }
    return null;
  }

  private constInts(ctx: Ctx, name: string | undefined): number[] | null {
    if (!name) return null;
    let d = ctx.decl.get(name);
    for (let c = ctx.parent; !d && c; c = c.parent) d = c.decl.get(name);
    if (d?.tid == null) return null;
    const e = this.o.store.entries[d.tid]!;
    if (e.info.n > 64 || !/^u?int(8|16|32|64)$/.test(e.info.dtype)) return null;
    const src: TensorSource | null = this.o.store.source(d.tid);
    if (!src || src.kind !== "num") return null;
    const out = new Float64Array(e.info.n);
    src.fill(out, 0, e.info.n);
    return Array.from(out);
  }

  private lookupFn(n: RawNode): number | null {
    if (!this.fnByKey.size) return null;
    const k = `${normD(n.domain)}|${n.op}`;
    return (n.overload ? this.fnByKey.get(`${k}|${n.overload}`) : undefined) ?? this.fnByKey.get(k) ?? null;
  }

  /** Constant node → tensor, or null when it can't be folded (ref attribute, unknown form) */
  private foldConstant(n: RawNode, gid: number, src: number): TensorInfo | null {
    const name = n.outputs[0]!;
    const a = n.attrs.find((x) => !x.ref && x.name.startsWith("value") || x.name === "sparse_value");
    if (!a || n.attrs.some((x) => x.ref)) return null;
    const make = (dt: number, dims: number[]) => scalarTensor(dt, dims);
    let t: RawTensor | null = null;
    switch (a.name) {
      case "value": t = a.t ?? null; break;
      case "sparse_value": return a.st ? this.addSparse(a.st, "sparse_attribute", gid, name, src) : null;
      case "value_float": {
        t = make(1, []);
        const f = new Float32Array([a.f ?? 0]);
        t.floatData = new Uint8Array(f.buffer);
        break;
      }
      case "value_floats": {
        const v = a.floats ?? [];
        t = make(1, [v.length]);
        t.floatData = new Uint8Array(new Float32Array(v).buffer);
        break;
      }
      case "value_int": t = make(7, []); t.int64Data = BigInt64Array.from([BigInt(a.i ?? 0)]); break;
      case "value_ints": {
        const v = a.ints ?? [];
        t = make(7, [v.length]);
        t.int64Data = BigInt64Array.from(v.map((x) => BigInt(x)));
        break;
      }
      case "value_string": t = make(8, []); t.stringData = [a.s ?? new Uint8Array()]; break;
      case "value_strings": {
        const v = a.strings ?? [];
        t = make(8, [v.length]);
        t.stringData = v;
        break;
      }
      default: return null;
    }
    if (!t) return null;
    t = { ...t, name };
    return this.addTensor(t, "constant", gid, name, src);
  }

  /* ───────── model ───────── */

  build(): ModelView {
    const raw = this.raw;
    const mainRaw = raw.graph ?? { name: "", doc: "", nodes: [], initializers: [], sparseInitializers: [], inputs: [], outputs: [], valueInfo: [], hasQuantAnnotation: false };
    const mainOpsets = { ...this.modelOpsets };
    this.o.onProgress?.("graph", 0);
    this.addGraph(0, mainRaw, { kind: "main", parent: null, depth: 0, name: mainRaw.name, opsets: mainOpsets }, null);
    this.fnRaw.forEach((f, i) => {
      const body: RawGraph = {
        name: f.name, doc: f.doc, nodes: f.nodes, initializers: [], sparseInitializers: [],
        inputs: f.inputs.map((n) => ({ name: n, type: null, doc: "" })),
        outputs: f.outputs.map((n) => ({ name: n, type: null, doc: "" })),
        valueInfo: f.valueInfo, hasQuantAnnotation: false,
      };
      const ops: Record<string, number> = { ...mainOpsets };
      for (const [d, v] of Object.entries(f.opsets)) ops[normD(d)] = v;
      this.addGraph(1 + i, body, { kind: "function", parent: null, depth: 1, name: f.overload ? `${f.domain}.${f.name}:${f.overload}` : `${f.domain}.${f.name}`, opsets: ops, fn: f }, null);
    });

    // payload availability + summaries
    this.o.onProgress?.("tensors", 0);
    const store = this.o.store;
    store.refreshAll((f) => this.o.onProgress?.("tensors", f));

    const missingExt = new Set<string>();
    for (const e of store.entries) if (e.info.external && !e.info.available) missingExt.add(e.info.external.location);
    for (const loc of missingExt) this.o.warnings.add(`external data file not provided: ${loc}`);

    // per-node parameters
    for (const g of this.graphs) {
      for (const n of g.nodes) {
        let params = 0;
        let wmax: number | null = null;
        for (const i of n.inputs) {
          if (i.tid == null) continue;
          const t = store.entries[i.tid]!.info;
          if (!isFloatDType(t.dtype) || t.n <= 1) continue;
          params += t.n;
          const m = t.summary?.absmax;
          if (m != null && (wmax == null || m > wmax)) wmax = m;
        }
        n.params = params;
        n.wmax = wmax;
      }
    }

    // meta
    const opCounts = new Map<string, number>();
    let nodeCount = 0;
    for (const g of this.graphs) {
      nodeCount += g.nodes.length;
      for (const n of g.nodes) opCounts.set(n.op, (opCounts.get(n.op) ?? 0) + 1);
    }
    let params = 0;
    let initializers = 0;
    const weightBytes: Record<string, number> = {};
    for (const e of store.entries) {
      if (this.aux.has(e.info.id)) continue;
      const s = e.info.source;
      if (s !== "initializer" && s !== "constant" && s !== "sparse_initializer") continue;
      if (s !== "constant") initializers++;
      if (isFloatDType(e.info.dtype)) params += e.info.n;
      weightBytes[e.info.dtype] = (weightBytes[e.info.dtype] ?? 0) + e.info.bytes;
    }
    const extras: string[] = [];
    if (raw.hasTraining) extras.push("training_info");
    if (raw.hasConfiguration) extras.push("configuration");
    if (mainRaw.hasQuantAnnotation) extras.push("quantization_annotation");
    if (store.entries.some((e) => e.info.external)) extras.push("external_data");
    if (store.entries.some((e) => e.info.source === "sparse_initializer" && !this.aux.has(e.info.id))) extras.push("sparse_initializer");
    if (raw.functions.length) extras.push("functions");
    if (this.graphs.some((g) => g.kind === "subgraph")) extras.push("subgraphs");

    const opsets: Record<string, number> = {};
    for (const [d, v] of Object.entries(raw.opsets)) opsets[normDomain(d)] = v;
    const meta: ModelMeta = {
      file: this.o.fileName,
      fileBytes: this.o.fileBytes,
      format: this.o.format,
      ir: raw.ir,
      opsets,
      producer: `${raw.producerName} ${raw.producerVersion}`.trim(),
      producerName: raw.producerName,
      producerVersion: raw.producerVersion,
      domain: raw.domain,
      modelVersion: raw.modelVersion,
      docString: raw.doc,
      graphName: mainRaw.name,
      props: raw.props,
      ops: [...opCounts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)),
      params,
      initializers,
      weightBytes,
      nodeCount,
      functions: raw.functions.length,
      extras,
    };
    this.o.onProgress?.("done", 1);
    return { meta, graphs: this.graphs, tensors: store.entries.map((e) => e.info), warnings: this.o.warnings.toArray() };
  }
}

function strValue(b: Uint8Array): AttrValue {
  try {
    return { t: "string", v: utf8Fatal.decode(b) };
  } catch {
    return { t: "string", v: hex(b), bytes: true };
  }
}

export function buildModel(raw: RawModel, o: BuildOptions): ModelView {
  return new Builder(raw, o).build();
}

