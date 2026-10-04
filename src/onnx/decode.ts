/**
 * ONNX protobuf (binary) decoder → plain "Raw*" structures. text.ts produces the same structures.
 * Tensor payloads stay as views over the input buffer.
 */
import { Reader, Warnings, readMessage, WIRE_LEN, WIRE_VARINT, WIRE_32, WIRE_64, decodeUtf8, ProtoError } from "./proto";
import { dtypeName } from "./dtype";
import type { ValueType, Dim } from "./types";

export interface RawExternal {
  location: string;
  offset: number;
  length: number | null;
  checksum: string;
}

export interface RawTensor {
  name: string;
  doc: string;
  dims: number[];
  dataType: number; // TensorProto.DataType enum
  raw: Uint8Array | null;
  /** little-endian bytes of float32 values (float_data) */
  floatData: Uint8Array | null;
  /** little-endian bytes of float64 values (double_data) */
  doubleData: Uint8Array | null;
  int32Data: Int32Array | null;
  int64Data: BigInt64Array | null;
  uint64Data: BigUint64Array | null;
  stringData: Uint8Array[] | null;
  external: RawExternal | null;
  dataLocationExternal: boolean;
}

export interface RawSparse {
  values: RawTensor;
  indices: RawTensor;
  dims: number[];
}

export const ATTR = {
  UNDEFINED: 0, FLOAT: 1, INT: 2, STRING: 3, TENSOR: 4, GRAPH: 5, FLOATS: 6, INTS: 7, STRINGS: 8,
  TENSORS: 9, GRAPHS: 10, SPARSE_TENSOR: 11, SPARSE_TENSORS: 12, TYPE_PROTO: 13, TYPE_PROTOS: 14,
} as const;

export interface RawAttr {
  name: string;
  /** AttributeProto.AttributeType (inferred when absent) */
  type: number;
  doc: string;
  ref: string;
  f?: number;
  i?: number | string;
  s?: Uint8Array;
  t?: RawTensor;
  g?: RawGraph;
  floats?: number[];
  ints?: (number | string)[];
  strings?: Uint8Array[];
  tensors?: RawTensor[];
  graphs?: RawGraph[];
  tp?: ValueType;
  tps?: ValueType[];
  st?: RawSparse;
  sts?: RawSparse[];
}

export interface RawNode {
  name: string;
  op: string;
  domain: string;
  overload: string;
  inputs: string[];
  outputs: string[];
  attrs: RawAttr[];
  doc: string;
  metadata?: Record<string, string>;
}

export interface RawValueInfo {
  name: string;
  type: ValueType | null;
  doc: string;
}

export interface RawGraph {
  name: string;
  doc: string;
  nodes: RawNode[];
  initializers: RawTensor[];
  sparseInitializers: RawSparse[];
  inputs: RawValueInfo[];
  outputs: RawValueInfo[];
  valueInfo: RawValueInfo[];
  hasQuantAnnotation: boolean;
}

export interface RawFunction {
  domain: string;
  name: string;
  overload: string;
  inputs: string[];
  outputs: string[];
  attrs: string[];
  attrProtos: RawAttr[];
  nodes: RawNode[];
  doc: string;
  opsets: Record<string, number>;
  valueInfo: RawValueInfo[];
}

export interface RawModel {
  ir: number | null;
  opsets: Record<string, number>; // domain as stored ("" for default)
  producerName: string;
  producerVersion: string;
  domain: string;
  modelVersion: number | string | null;
  doc: string;
  graph: RawGraph | null;
  functions: RawFunction[];
  props: Record<string, string>;
  hasTraining: boolean;
  hasConfiguration: boolean;
}

const MAX_DEPTH = 64;

export function emptyTensor(): RawTensor {
  return {
    name: "", doc: "", dims: [], dataType: 0, raw: null, floatData: null, doubleData: null, int32Data: null,
    int64Data: null, uint64Data: null, stringData: null, external: null, dataLocationExternal: false,
  };
}
export const emptyGraph = (): RawGraph => ({
  name: "", doc: "", nodes: [], initializers: [], sparseInitializers: [], inputs: [], outputs: [], valueInfo: [],
  hasQuantAnnotation: false,
});

function concatBytes(chunks: Uint8Array[]): Uint8Array | null {
  if (!chunks.length) return null;
  if (chunks.length === 1) return chunks[0]!;
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

function f32bytes(v: number[]): Uint8Array {
  const a = new Float32Array(v);
  return new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
}
function f64bytes(v: number[]): Uint8Array {
  const a = new Float64Array(v);
  return new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
}

function decodeStringEntry(r: Reader): [string, string] {
  let k = "";
  let v = "";
  readMessage(r, "StringStringEntry", (f, w) => {
    if (f === 1 && w === WIRE_LEN) k = r.string();
    else if (f === 2 && w === WIRE_LEN) v = r.string();
    else r.skip(w);
  });
  return [k, v];
}

export function decodeTensor(r: Reader): RawTensor {
  const t = emptyTensor();
  const floatChunks: Uint8Array[] = [];
  const doubleChunks: Uint8Array[] = [];
  const floatLoose: number[] = [];
  const doubleLoose: number[] = [];
  const i32: number[] = [];
  const i64: bigint[] = [];
  const u64: bigint[] = [];
  const ext: Record<string, string> = {};
  let hasExt = false;
  readMessage(r, "TensorProto", (f, w) => {
    switch (f) {
      case 1: r.packed(w, (x) => t.dims.push(x.int64()));
        break;
      case 2: t.dataType = r.int32();
        break;
      case 4:
        if (w === WIRE_LEN) floatChunks.push(r.bytes());
        else floatLoose.push(r.float());
        break;
      case 5: r.packed(w, (x) => i32.push(x.int32()));
        break;
      case 6: (t.stringData ??= []).push(r.bytes());
        break;
      case 7: r.packed(w, (x) => i64.push(x.int64big()));
        break;
      case 8: t.name = r.string();
        break;
      case 9: t.raw = r.bytes();
        break;
      case 10:
        if (w === WIRE_LEN) doubleChunks.push(r.bytes());
        else doubleLoose.push(r.double());
        break;
      case 11: r.packed(w, (x) => u64.push(x.uint64big()));
        break;
      case 12: t.doc = r.string();
        break;
      case 13: {
        const [k, v] = decodeStringEntry(r.sub());
        ext[k] = v;
        hasExt = true;
        break;
      }
      case 14: t.dataLocationExternal = r.varint() === 1;
        break;
      default: r.skip(w);
    }
  });
  if (floatChunks.length) t.floatData = concatBytes(floatChunks);
  else if (floatLoose.length) t.floatData = f32bytes(floatLoose);
  if (doubleChunks.length) t.doubleData = concatBytes(doubleChunks);
  else if (doubleLoose.length) t.doubleData = f64bytes(doubleLoose);
  if (i32.length) t.int32Data = Int32Array.from(i32);
  if (i64.length) t.int64Data = BigInt64Array.from(i64);
  if (u64.length) t.uint64Data = BigUint64Array.from(u64);
  if (hasExt || t.dataLocationExternal) {
    t.dataLocationExternal = true;
    t.external = {
      location: ext.location ?? "",
      offset: ext.offset ? Number(ext.offset) || 0 : 0,
      length: ext.length ? Number(ext.length) : null,
      checksum: ext.checksum ?? "",
    };
  }
  return t;
}

export function decodeSparse(r: Reader): RawSparse {
  const s: RawSparse = { values: emptyTensor(), indices: emptyTensor(), dims: [] };
  readMessage(r, "SparseTensorProto", (f, w) => {
    if (f === 1 && w === WIRE_LEN) s.values = decodeTensor(r.sub());
    else if (f === 2 && w === WIRE_LEN) s.indices = decodeTensor(r.sub());
    else if (f === 3) r.packed(w, (x) => s.dims.push(x.int64()));
    else r.skip(w);
  });
  return s;
}

/* ───────────── types ───────────── */

function decodeShape(r: Reader): Dim[] {
  const dims: Dim[] = [];
  readMessage(r, "TensorShapeProto", (f, w) => {
    if (f === 1 && w === WIRE_LEN) {
      const d = r.sub();
      let v: Dim = "?";
      readMessage(d, "Dimension", (df, dw) => {
        if (df === 1 && dw === WIRE_VARINT) v = d.int64();
        else if (df === 2 && dw === WIRE_LEN) {
          const s = d.string();
          v = s || "?";
        } else d.skip(dw);
      });
      dims.push(v);
    } else r.skip(w);
  });
  return dims;
}

function decodeTensorTypeLike(r: Reader, kind: "tensor" | "sparse"): ValueType {
  let elem = 0;
  let shape: Dim[] | null = null;
  readMessage(r, "TypeProto.Tensor", (f, w) => {
    if (f === 1 && w === WIRE_VARINT) elem = r.int32();
    else if (f === 2 && w === WIRE_LEN) shape = decodeShape(r.sub());
    else r.skip(w);
  });
  return { kind, dtype: dtypeName(elem), shape };
}

export function decodeType(r: Reader, depth = 0): ValueType {
  let out: ValueType = { kind: "unknown" };
  if (depth > MAX_DEPTH) return out;
  readMessage(r, "TypeProto", (f, w) => {
    if (w !== WIRE_LEN) return r.skip(w);
    switch (f) {
      case 1: out = decodeTensorTypeLike(r.sub(), "tensor");
        break;
      case 8: out = decodeTensorTypeLike(r.sub(), "sparse");
        break;
      case 4: {
        const s = r.sub();
        let elem: ValueType | null = null;
        readMessage(s, "SequenceType", (sf, sw) => (sf === 1 && sw === WIRE_LEN ? (elem = decodeType(s.sub(), depth + 1)) : s.skip(sw)));
        out = { kind: "sequence", elem };
        break;
      }
      case 9: {
        const s = r.sub();
        let elem: ValueType | null = null;
        readMessage(s, "OptionalType", (sf, sw) => (sf === 1 && sw === WIRE_LEN ? (elem = decodeType(s.sub(), depth + 1)) : s.skip(sw)));
        out = { kind: "optional", elem };
        break;
      }
      case 5: {
        const s = r.sub();
        let key = 0;
        let value: ValueType | null = null;
        readMessage(s, "MapType", (sf, sw) => {
          if (sf === 1 && sw === WIRE_VARINT) key = s.int32();
          else if (sf === 2 && sw === WIRE_LEN) value = decodeType(s.sub(), depth + 1);
          else s.skip(sw);
        });
        out = { kind: "map", key: dtypeName(key), value };
        break;
      }
      case 7: {
        const s = r.sub();
        let domain = "";
        let name = "";
        readMessage(s, "OpaqueType", (sf, sw) => {
          if (sf === 1 && sw === WIRE_LEN) domain = s.string();
          else if (sf === 2 && sw === WIRE_LEN) name = s.string();
          else s.skip(sw);
        });
        out = { kind: "opaque", domain, name };
        break;
      }
      default: r.skip(w);
    }
  });
  return out;
}

export function decodeValueInfo(r: Reader): RawValueInfo {
  const v: RawValueInfo = { name: "", type: null, doc: "" };
  readMessage(r, "ValueInfoProto", (f, w) => {
    if (f === 1 && w === WIRE_LEN) v.name = r.string();
    else if (f === 2 && w === WIRE_LEN) v.type = decodeType(r.sub());
    else if (f === 3 && w === WIRE_LEN) v.doc = r.string();
    else r.skip(w);
  });
  return v;
}

/* ───────────── attributes / nodes / graphs ───────────── */

export function inferAttrType(a: RawAttr): number {
  if (a.type) return a.type;
  if (a.f !== undefined) return ATTR.FLOAT;
  if (a.i !== undefined) return ATTR.INT;
  if (a.s !== undefined) return ATTR.STRING;
  if (a.t) return ATTR.TENSOR;
  if (a.g) return ATTR.GRAPH;
  if (a.st) return ATTR.SPARSE_TENSOR;
  if (a.tp) return ATTR.TYPE_PROTO;
  if (a.floats) return ATTR.FLOATS;
  if (a.ints) return ATTR.INTS;
  if (a.strings) return ATTR.STRINGS;
  if (a.tensors) return ATTR.TENSORS;
  if (a.graphs) return ATTR.GRAPHS;
  if (a.sts) return ATTR.SPARSE_TENSORS;
  if (a.tps) return ATTR.TYPE_PROTOS;
  return ATTR.UNDEFINED;
}

export function decodeAttr(r: Reader, depth: number): RawAttr {
  const a: RawAttr = { name: "", type: 0, doc: "", ref: "" };
  readMessage(r, "AttributeProto", (f, w) => {
    switch (f) {
      case 1: a.name = r.string(); break;
      case 2: a.f = r.float(); break;
      case 3: a.i = r.int64x(); break;
      case 4: a.s = r.bytes(); break;
      case 5: a.t = decodeTensor(r.sub()); break;
      case 6: a.g = decodeGraph(r.sub(), depth + 1); break;
      case 7: r.floats(w, (a.floats ??= [])); break;
      case 8: r.packed(w, (x) => (a.ints ??= []).push(x.int64x())); break;
      case 9: (a.strings ??= []).push(r.bytes()); break;
      case 10: (a.tensors ??= []).push(decodeTensor(r.sub())); break;
      case 11: (a.graphs ??= []).push(decodeGraph(r.sub(), depth + 1)); break;
      case 13: a.doc = r.string(); break;
      case 14: a.tp = decodeType(r.sub()); break;
      case 15: (a.tps ??= []).push(decodeType(r.sub())); break;
      case 20: a.type = r.int32(); break;
      case 21: a.ref = r.string(); break;
      case 22: a.st = decodeSparse(r.sub()); break;
      case 23: (a.sts ??= []).push(decodeSparse(r.sub())); break;
      default: r.skip(w);
    }
  });
  a.type = inferAttrType(a);
  return a;
}

export function decodeNode(r: Reader, depth: number): RawNode {
  const n: RawNode = { name: "", op: "", domain: "", overload: "", inputs: [], outputs: [], attrs: [], doc: "" };
  readMessage(r, "NodeProto", (f, w) => {
    if (w !== WIRE_LEN) return r.skip(w);
    switch (f) {
      case 1: n.inputs.push(r.string()); break;
      case 2: n.outputs.push(r.string()); break;
      case 3: n.name = r.string(); break;
      case 4: n.op = r.string(); break;
      case 5: n.attrs.push(decodeAttr(r.sub(), depth)); break;
      case 6: n.doc = r.string(); break;
      case 7: n.domain = r.string(); break;
      case 8: n.overload = r.string(); break;
      case 9: {
        const [k, v] = decodeStringEntry(r.sub());
        (n.metadata ??= {})[k] = v;
        break;
      }
      default: r.skip(w);
    }
  });
  return n;
}

export function decodeGraph(r: Reader, depth = 0): RawGraph {
  const g = emptyGraph();
  if (depth > MAX_DEPTH) {
    r.warnings.add("graph nesting too deep; truncated");
    return g;
  }
  readMessage(r, "GraphProto", (f, w) => {
    if (w !== WIRE_LEN) return r.skip(w);
    switch (f) {
      case 1: g.nodes.push(decodeNode(r.sub(), depth)); break;
      case 2: g.name = r.string(); break;
      case 5: g.initializers.push(decodeTensor(r.sub())); break;
      case 10: g.doc = r.string(); break;
      case 11: g.inputs.push(decodeValueInfo(r.sub())); break;
      case 12: g.outputs.push(decodeValueInfo(r.sub())); break;
      case 13: g.valueInfo.push(decodeValueInfo(r.sub())); break;
      case 14: g.hasQuantAnnotation = true; r.skip(w); break;
      case 15: g.sparseInitializers.push(decodeSparse(r.sub())); break;
      default: r.skip(w);
    }
  });
  return g;
}

function decodeOpset(r: Reader): [string, number] {
  let d = "";
  let v = 0;
  readMessage(r, "OperatorSetIdProto", (f, w) => {
    if (f === 1 && w === WIRE_LEN) d = r.string();
    else if (f === 2 && w === WIRE_VARINT) v = r.int64();
    else r.skip(w);
  });
  return [d, v];
}

function decodeFunction(r: Reader): RawFunction {
  const fn: RawFunction = {
    domain: "", name: "", overload: "", inputs: [], outputs: [], attrs: [], attrProtos: [], nodes: [], doc: "",
    opsets: {}, valueInfo: [],
  };
  readMessage(r, "FunctionProto", (f, w) => {
    if (w !== WIRE_LEN) return r.skip(w);
    switch (f) {
      case 1: fn.name = r.string(); break;
      case 4: fn.inputs.push(r.string()); break;
      case 5: fn.outputs.push(r.string()); break;
      case 6: fn.attrs.push(r.string()); break;
      case 7: fn.nodes.push(decodeNode(r.sub(), 1)); break;
      case 8: fn.doc = r.string(); break;
      case 9: {
        const [d, v] = decodeOpset(r.sub());
        fn.opsets[d] = v;
        break;
      }
      case 10: fn.domain = r.string(); break;
      case 11: fn.attrProtos.push(decodeAttr(r.sub(), 1)); break;
      case 12: fn.valueInfo.push(decodeValueInfo(r.sub())); break;
      case 13: fn.overload = r.string(); break;
      default: r.skip(w);
    }
  });
  return fn;
}

export function decodeModel(buf: Uint8Array, warnings: Warnings = new Warnings()): RawModel {
  const r = new Reader(buf, 0, buf.length, warnings);
  const m: RawModel = {
    ir: null, opsets: {}, producerName: "", producerVersion: "", domain: "", modelVersion: null, doc: "", graph: null,
    functions: [], props: {}, hasTraining: false, hasConfiguration: false,
  };
  readMessage(r, "ModelProto", (f, w) => {
    switch (f) {
      case 1: if (w !== WIRE_VARINT) return r.skip(w); m.ir = r.int64(); break;
      case 2: if (w !== WIRE_LEN) return r.skip(w); m.producerName = r.string(); break;
      case 3: if (w !== WIRE_LEN) return r.skip(w); m.producerVersion = r.string(); break;
      case 4: if (w !== WIRE_LEN) return r.skip(w); m.domain = r.string(); break;
      case 5: if (w !== WIRE_VARINT) return r.skip(w); m.modelVersion = r.int64x(); break;
      case 6: if (w !== WIRE_LEN) return r.skip(w); m.doc = r.string(); break;
      case 7: if (w !== WIRE_LEN) return r.skip(w); m.graph = decodeGraph(r.sub()); break;
      case 8: {
        if (w !== WIRE_LEN) return r.skip(w);
        const [d, v] = decodeOpset(r.sub());
        m.opsets[d] = v;
        break;
      }
      case 14: {
        if (w !== WIRE_LEN) return r.skip(w);
        const [k, v] = decodeStringEntry(r.sub());
        m.props[k] = v;
        break;
      }
      case 20: m.hasTraining = true; r.skip(w); break;
      case 25: if (w !== WIRE_LEN) return r.skip(w); m.functions.push(decodeFunction(r.sub())); break;
      case 26: m.hasConfiguration = true; r.skip(w); break;
      default: r.skip(w);
    }
  });
  if (!m.graph && m.ir === null && !m.producerName && !Object.keys(m.opsets).length)
    throw new ProtoError("not an ONNX model (no ModelProto fields found)");
  if (!m.graph) warnings.add("model has no graph");
  return m;
}

export { decodeUtf8, WIRE_32, WIRE_64 };
