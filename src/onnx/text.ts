/**
 * ONNX text (protobuf text format, .onnx.prototxt / .pbtxt) and protobuf-JSON readers.
 * Both are parsed to a generic tree, then converted into the same Raw* structures as the binary decoder.
 */
import { ProtoError, Warnings } from "./proto";
import { ENUM_BY_PROTO_NAME, dtypeName } from "./dtype";
import {
  ATTR, emptyGraph, emptyTensor, inferAttrType,
  type RawAttr, type RawFunction, type RawGraph, type RawModel, type RawNode, type RawSparse, type RawTensor, type RawValueInfo,
} from "./decode";
import type { Dim, ValueType } from "./types";

/* ───────────── generic tree ───────────── */

export class Ident {
  constructor(readonly name: string) {}
}
type Scalar = number | bigint | boolean | string | Uint8Array | Ident;
export interface Msg {
  f: Map<string, Val[]>;
}
type Val = Scalar | Msg;

const norm = (k: string): string => k.replace(/_/g, "").toLowerCase();
const isMsg = (v: unknown): v is Msg => typeof v === "object" && v !== null && "f" in (v as object) && (v as Msg).f instanceof Map;

function push(m: Msg, key: string, v: Val): void {
  const k = norm(key);
  const a = m.f.get(k);
  if (a) a.push(v);
  else m.f.set(k, [v]);
}

/* ───────────── text format tokenizer / parser ───────────── */

const NUM_RE = /^[+-]?(?:0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fF]?|inf(?:inity)?|nan)$/i;

class TextParser {
  i = 0;
  constructor(private s: string) {}

  private ws(): void {
    const s = this.s;
    for (;;) {
      const c = s.charCodeAt(this.i);
      if (c === 32 || c === 10 || c === 13 || c === 9) this.i++;
      else if (c === 35) {
        // # comment
        while (this.i < s.length && s.charCodeAt(this.i) !== 10) this.i++;
      } else break;
    }
  }

  private err(msg: string): never {
    const line = this.s.slice(0, this.i).split("\n").length;
    throw new ProtoError(`text format: ${msg} (line ${line})`);
  }

  parseMessage(close: string | null): Msg {
    const m: Msg = { f: new Map() };
    for (;;) {
      this.ws();
      if (this.i >= this.s.length) {
        if (close) this.err(`missing '${close}'`);
        return m;
      }
      const c = this.s[this.i]!;
      if (close && c === close) {
        this.i++;
        return m;
      }
      if (c === ";" || c === ",") {
        this.i++;
        continue;
      }
      let name: string;
      if (c === "[") {
        // extension field name [pkg.name]
        const e = this.s.indexOf("]", this.i);
        if (e < 0) this.err("unterminated extension name");
        name = this.s.slice(this.i + 1, e);
        this.i = e + 1;
      } else name = this.ident();
      this.ws();
      let colon = false;
      if (this.s[this.i] === ":") {
        colon = true;
        this.i++;
        this.ws();
      }
      const c2 = this.s[this.i];
      if (c2 === "{" || c2 === "<") {
        this.i++;
        push(m, name, this.parseMessage(c2 === "{" ? "}" : ">"));
      } else if (c2 === "[") {
        this.i++;
        for (;;) {
          this.ws();
          if (this.s[this.i] === "]") {
            this.i++;
            break;
          }
          if (this.s[this.i] === ",") {
            this.i++;
            continue;
          }
          const c3 = this.s[this.i];
          if (c3 === "{" || c3 === "<") {
            this.i++;
            push(m, name, this.parseMessage(c3 === "{" ? "}" : ">"));
          } else push(m, name, this.scalar());
        }
      } else {
        if (!colon) this.err(`expected ':' after field '${name}'`);
        push(m, name, this.scalar());
      }
    }
  }

  private ident(): string {
    const s = this.s;
    const st = this.i;
    while (this.i < s.length && /[A-Za-z0-9_.]/.test(s[this.i]!)) this.i++;
    if (st === this.i) this.err(`unexpected character '${s[this.i] ?? "EOF"}'`);
    return s.slice(st, this.i);
  }

  private scalar(): Scalar {
    const s = this.s;
    const c = s[this.i]!;
    if (c === '"' || c === "'") {
      const parts: Uint8Array[] = [];
      for (;;) {
        parts.push(this.str());
        this.ws();
        const d = s[this.i];
        if (d !== '"' && d !== "'") break;
      }
      if (parts.length === 1) return parts[0]!;
      let n = 0;
      for (const p of parts) n += p.length;
      const out = new Uint8Array(n);
      let o = 0;
      for (const p of parts) (out.set(p, o), (o += p.length));
      return out;
    }
    const st = this.i;
    while (this.i < s.length && /[A-Za-z0-9_.+\-]/.test(s[this.i]!)) this.i++;
    const tok = s.slice(st, this.i);
    if (!tok) this.err(`unexpected character '${c}'`);
    if (NUM_RE.test(tok)) return parseNum(tok);
    if (tok === "true" || tok === "True" || tok === "t") return true;
    if (tok === "false" || tok === "False" || tok === "f") return false;
    return new Ident(tok);
  }

  private str(): Uint8Array {
    const s = this.s;
    const q = s[this.i++]!;
    const out: number[] = [];
    const enc = new TextEncoder();
    let chunk = "";
    const flush = () => {
      if (chunk) {
        for (const b of enc.encode(chunk)) out.push(b);
        chunk = "";
      }
    };
    for (;;) {
      if (this.i >= s.length) this.err("unterminated string");
      const c = s[this.i++]!;
      if (c === q) break;
      if (c !== "\\") {
        chunk += c;
        continue;
      }
      const e = s[this.i++]!;
      switch (e) {
        case "n": chunk += "\n"; break;
        case "r": chunk += "\r"; break;
        case "t": chunk += "\t"; break;
        case "a": chunk += "\x07"; break;
        case "b": chunk += "\b"; break;
        case "f": chunk += "\f"; break;
        case "v": chunk += "\v"; break;
        case "x": case "X": {
          let h = "";
          while (h.length < 2 && /[0-9a-fA-F]/.test(s[this.i] ?? "")) h += s[this.i++];
          flush();
          out.push(parseInt(h || "0", 16));
          break;
        }
        case "u": case "U": {
          const n = e === "u" ? 4 : 8;
          const h = s.slice(this.i, this.i + n);
          this.i += n;
          chunk += String.fromCodePoint(parseInt(h, 16));
          break;
        }
        default:
          if (e >= "0" && e <= "7") {
            let o = e;
            while (o.length < 3 && /[0-7]/.test(s[this.i] ?? "")) o += s[this.i++];
            flush();
            out.push(parseInt(o, 8) & 255);
          } else chunk += e;
      }
    }
    flush();
    return Uint8Array.from(out);
  }
}

function parseNum(tok: string): number | bigint {
  const t = tok.toLowerCase();
  if (/^[+-]?(inf|infinity)$/.test(t)) return t[0] === "-" ? -Infinity : Infinity;
  if (/^[+-]?nan$/.test(t)) return NaN;
  if (/^[+-]?0x/.test(t)) {
    const b = BigInt(t.replace("+", ""));
    return b >= -(2n ** 53n) && b <= 2n ** 53n ? Number(b) : b;
  }
  if (/^[+-]?\d+$/.test(t)) {
    const n = Number(t);
    return Number.isSafeInteger(n) ? n : BigInt(t.replace("+", ""));
  }
  return Number(t.replace(/f$/, ""));
}

/* ───────────── JSON → tree ───────────── */

function jsonToVal(v: unknown): Val[] {
  if (Array.isArray(v)) return v.flatMap(jsonToVal);
  if (v === null || v === undefined) return [];
  if (typeof v === "object") {
    const m: Msg = { f: new Map() };
    for (const [k, x] of Object.entries(v as object)) for (const y of jsonToVal(x)) push(m, k, y);
    return [m];
  }
  return [v as Scalar];
}

/* ───────────── tree accessors ───────────── */

const all = (m: Msg, k: string): Val[] => m.f.get(norm(k)) ?? [];
const one = (m: Msg, k: string): Val | undefined => m.f.get(norm(k))?.[0];
const has = (m: Msg, k: string): boolean => m.f.has(norm(k));

const utf8 = new TextDecoder();
function str(v: Val | undefined): string {
  if (v === undefined) return "";
  if (typeof v === "string") return v;
  if (v instanceof Uint8Array) return utf8.decode(v);
  if (v instanceof Ident) return v.name;
  return String(v);
}
function bytes(v: Val | undefined): Uint8Array | null {
  if (v === undefined) return null;
  if (v instanceof Uint8Array) return v;
  if (typeof v === "string") {
    // protobuf JSON: base64 (standard or url-safe)
    const bin = atob(v.replace(/-/g, "+").replace(/_/g, "/"));
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  return null;
}
function num(v: Val | undefined): number {
  if (typeof v === "number") return v;
  if (typeof v === "bigint") return Number(v);
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string") return Number(v);
  if (v instanceof Ident) {
    const t = v.name.toLowerCase();
    if (t === "nan") return NaN;
    if (t === "inf" || t === "infinity") return Infinity;
    if (t === "-inf" || t === "-infinity") return -Infinity;
  }
  return NaN;
}
function flt(v: Val | undefined): number {
  if (typeof v === "string") {
    if (v === "NaN") return NaN;
    if (v === "Infinity") return Infinity;
    if (v === "-Infinity") return -Infinity;
  }
  return num(v);
}
function int(v: Val | undefined): number | string {
  if (typeof v === "bigint") return v >= -(2n ** 53n) && v <= 2n ** 53n ? Number(v) : v.toString();
  if (typeof v === "string") {
    if (/^-?\d+$/.test(v)) {
      const n = Number(v);
      return Number.isSafeInteger(n) ? n : v;
    }
  }
  return num(v);
}
const big = (v: Val | undefined): bigint => {
  if (typeof v === "bigint") return v;
  if (typeof v === "number") return BigInt(Math.trunc(v));
  if (typeof v === "string" && /^-?\d+$/.test(v)) return BigInt(v);
  return 0n;
};
function enumNum(v: Val | undefined, table?: Record<string, number>): number {
  if (typeof v === "number") return v;
  if (typeof v === "bigint") return Number(v);
  const s = str(v).toUpperCase();
  if (table && s in table) return table[s]!;
  const n = Number(s);
  return Number.isNaN(n) ? 0 : n;
}
const ATTR_ENUM: Record<string, number> = {
  UNDEFINED: 0, FLOAT: 1, INT: 2, STRING: 3, TENSOR: 4, GRAPH: 5, FLOATS: 6, INTS: 7, STRINGS: 8, TENSORS: 9, GRAPHS: 10,
  SPARSE_TENSOR: 11, SPARSE_TENSORS: 12, TYPE_PROTO: 13, TYPE_PROTOS: 14,
};
const mm = (v: Val): Msg => (isMsg(v) ? v : { f: new Map() });

/* ───────────── tree → Raw* ───────────── */

function toTensor(m: Msg): RawTensor {
  const t = emptyTensor();
  t.name = str(one(m, "name"));
  t.doc = str(one(m, "doc_string"));
  t.dims = all(m, "dims").map(num);
  t.dataType = enumNum(one(m, "data_type"), ENUM_BY_PROTO_NAME);
  const raw = bytes(one(m, "raw_data"));
  if (raw) t.raw = raw;
  const fd = all(m, "float_data");
  if (fd.length) {
    const a = Float32Array.from(fd.map(flt));
    t.floatData = new Uint8Array(a.buffer);
  }
  const dd = all(m, "double_data");
  if (dd.length) {
    const a = Float64Array.from(dd.map(flt));
    t.doubleData = new Uint8Array(a.buffer);
  }
  const i32 = all(m, "int32_data");
  if (i32.length) t.int32Data = Int32Array.from(i32.map((x) => Number(big(x)) | 0));
  const i64 = all(m, "int64_data");
  if (i64.length) t.int64Data = BigInt64Array.from(i64.map((x) => BigInt.asIntN(64, big(x))));
  const u64 = all(m, "uint64_data");
  if (u64.length) t.uint64Data = BigUint64Array.from(u64.map((x) => BigInt.asUintN(64, big(x))));
  const sd = all(m, "string_data");
  if (sd.length) t.stringData = sd.map((x) => bytes(x) ?? new Uint8Array());
  const ext = all(m, "external_data");
  const loc = one(m, "data_location");
  const isExt = ext.length > 0 || (loc !== undefined && (str(loc).toUpperCase() === "EXTERNAL" || num(loc) === 1));
  if (isExt) {
    const kv: Record<string, string> = {};
    for (const e of ext) kv[str(one(mm(e), "key"))] = str(one(mm(e), "value"));
    t.dataLocationExternal = true;
    t.external = {
      location: kv.location ?? "", offset: kv.offset ? Number(kv.offset) || 0 : 0,
      length: kv.length ? Number(kv.length) : null, checksum: kv.checksum ?? "",
    };
  }
  return t;
}

function toSparse(m: Msg): RawSparse {
  const v = one(m, "values");
  const i = one(m, "indices");
  return {
    values: v ? toTensor(mm(v)) : emptyTensor(),
    indices: i ? toTensor(mm(i)) : emptyTensor(),
    dims: all(m, "dims").map(num),
  };
}

function toShape(m: Msg): Dim[] {
  return all(m, "dim").map((d) => {
    const dm = mm(d);
    if (has(dm, "dim_value")) return num(one(dm, "dim_value"));
    const p = str(one(dm, "dim_param"));
    return p || "?";
  });
}

function toTensorType(m: Msg, kind: "tensor" | "sparse"): ValueType {
  const sh = one(m, "shape");
  return { kind, dtype: dtypeName(enumNum(one(m, "elem_type"), ENUM_BY_PROTO_NAME)), shape: sh ? toShape(mm(sh)) : null };
}

function toType(m: Msg): ValueType {
  let v: Val | undefined;
  if ((v = one(m, "tensor_type"))) return toTensorType(mm(v), "tensor");
  if ((v = one(m, "sparse_tensor_type"))) return toTensorType(mm(v), "sparse");
  if ((v = one(m, "sequence_type"))) {
    const e = one(mm(v), "elem_type");
    return { kind: "sequence", elem: e ? toType(mm(e)) : null };
  }
  if ((v = one(m, "optional_type"))) {
    const e = one(mm(v), "elem_type");
    return { kind: "optional", elem: e ? toType(mm(e)) : null };
  }
  if ((v = one(m, "map_type"))) {
    const mt = mm(v);
    const vt = one(mt, "value_type");
    return { kind: "map", key: dtypeName(enumNum(one(mt, "key_type"), ENUM_BY_PROTO_NAME)), value: vt ? toType(mm(vt)) : null };
  }
  if ((v = one(m, "opaque_type"))) return { kind: "opaque", domain: str(one(mm(v), "domain")), name: str(one(mm(v), "name")) };
  return { kind: "unknown" };
}

function toValueInfo(m: Msg): RawValueInfo {
  const t = one(m, "type");
  return { name: str(one(m, "name")), type: t ? toType(mm(t)) : null, doc: str(one(m, "doc_string")) };
}

function toAttr(m: Msg): RawAttr {
  const a: RawAttr = { name: str(one(m, "name")), type: 0, doc: str(one(m, "doc_string")), ref: str(one(m, "ref_attr_name")) };
  const t = one(m, "type");
  if (t !== undefined) a.type = enumNum(t, ATTR_ENUM);
  let v: Val | undefined;
  if (has(m, "f")) a.f = flt(one(m, "f"));
  if (has(m, "i")) a.i = int(one(m, "i"));
  if ((v = one(m, "s")) !== undefined) a.s = bytes(v) ?? new Uint8Array();
  if ((v = one(m, "t"))) a.t = toTensor(mm(v));
  if ((v = one(m, "g"))) a.g = toGraph(mm(v));
  if (has(m, "floats")) a.floats = all(m, "floats").map(flt);
  if (has(m, "ints")) a.ints = all(m, "ints").map(int);
  if (has(m, "strings")) a.strings = all(m, "strings").map((x) => bytes(x) ?? new Uint8Array());
  if (has(m, "tensors")) a.tensors = all(m, "tensors").map((x) => toTensor(mm(x)));
  if (has(m, "graphs")) a.graphs = all(m, "graphs").map((x) => toGraph(mm(x)));
  if ((v = one(m, "tp"))) a.tp = toType(mm(v));
  if (has(m, "type_protos")) a.tps = all(m, "type_protos").map((x) => toType(mm(x)));
  if ((v = one(m, "sparse_tensor"))) a.st = toSparse(mm(v));
  if (has(m, "sparse_tensors")) a.sts = all(m, "sparse_tensors").map((x) => toSparse(mm(x)));
  if (!a.type) a.type = inferAttrType(a);
  return a;
}

function toNode(m: Msg): RawNode {
  const n: RawNode = {
    name: str(one(m, "name")), op: str(one(m, "op_type")), domain: str(one(m, "domain")), overload: str(one(m, "overload")),
    inputs: all(m, "input").map(str), outputs: all(m, "output").map(str), attrs: all(m, "attribute").map((x) => toAttr(mm(x))),
    doc: str(one(m, "doc_string")),
  };
  const md = all(m, "metadata_props");
  if (md.length) {
    n.metadata = {};
    for (const e of md) n.metadata[str(one(mm(e), "key"))] = str(one(mm(e), "value"));
  }
  return n;
}

function toGraph(m: Msg): RawGraph {
  const g = emptyGraph();
  g.name = str(one(m, "name"));
  g.doc = str(one(m, "doc_string"));
  g.nodes = all(m, "node").map((x) => toNode(mm(x)));
  g.initializers = all(m, "initializer").map((x) => toTensor(mm(x)));
  g.sparseInitializers = all(m, "sparse_initializer").map((x) => toSparse(mm(x)));
  g.inputs = all(m, "input").map((x) => toValueInfo(mm(x)));
  g.outputs = all(m, "output").map((x) => toValueInfo(mm(x)));
  g.valueInfo = all(m, "value_info").map((x) => toValueInfo(mm(x)));
  g.hasQuantAnnotation = has(m, "quantization_annotation");
  return g;
}

function toFunction(m: Msg): RawFunction {
  const opsets: Record<string, number> = {};
  for (const o of all(m, "opset_import")) opsets[str(one(mm(o), "domain"))] = num(one(mm(o), "version"));
  return {
    domain: str(one(m, "domain")), name: str(one(m, "name")), overload: str(one(m, "overload")),
    inputs: all(m, "input").map(str), outputs: all(m, "output").map(str), attrs: all(m, "attribute").map(str),
    attrProtos: all(m, "attribute_proto").map((x) => toAttr(mm(x))), nodes: all(m, "node").map((x) => toNode(mm(x))),
    doc: str(one(m, "doc_string")), opsets, valueInfo: all(m, "value_info").map((x) => toValueInfo(mm(x))),
  };
}

export function treeToModel(m: Msg): RawModel {
  const opsets: Record<string, number> = {};
  for (const o of all(m, "opset_import")) opsets[str(one(mm(o), "domain"))] = num(one(mm(o), "version"));
  const props: Record<string, string> = {};
  for (const e of all(m, "metadata_props")) props[str(one(mm(e), "key"))] = str(one(mm(e), "value"));
  const g = one(m, "graph");
  const mv = one(m, "model_version");
  return {
    ir: has(m, "ir_version") ? num(one(m, "ir_version")) : null,
    opsets,
    producerName: str(one(m, "producer_name")),
    producerVersion: str(one(m, "producer_version")),
    domain: str(one(m, "domain")),
    modelVersion: mv === undefined ? null : int(mv),
    doc: str(one(m, "doc_string")),
    graph: g ? toGraph(mm(g)) : null,
    functions: all(m, "functions").concat(all(m, "function")).map((x) => toFunction(mm(x))),
    props,
    hasTraining: has(m, "training_info"),
    hasConfiguration: has(m, "configuration"),
  };
}

export function parseTextModel(text: string, warnings: Warnings): RawModel {
  const p = new TextParser(text);
  const tree = p.parseMessage(null);
  const m = treeToModel(tree);
  if (!m.graph) warnings.add("model has no graph");
  return m;
}

export function parseJsonModel(text: string, warnings: Warnings): RawModel {
  let j: unknown;
  try {
    j = JSON.parse(text);
  } catch (e) {
    throw new ProtoError(`invalid JSON: ${(e as Error).message}`);
  }
  const vals = jsonToVal(j);
  const root = vals[0];
  if (!isMsg(root)) throw new ProtoError("JSON root is not an object");
  const m = treeToModel(root);
  if (!m.graph) warnings.add("model has no graph");
  return m;
}

/* ───────────── format detection ───────────── */

export type DetectedFormat = "onnx" | "onnx-text" | "onnx-json" | "ort" | "unknown";

export function detectFormat(name: string, data: ArrayBuffer | Uint8Array | string): DetectedFormat {
  const lower = name.toLowerCase();
  if (typeof data === "string") return data.trimStart().startsWith("{") ? "onnx-json" : "onnx-text";
  const u = data instanceof Uint8Array ? data : new Uint8Array(data);
  if (u.length >= 8 && u[4] === 0x4f && u[5] === 0x52 && u[6] === 0x54 && u[7] === 0x4d) return "ort"; // FlatBuffer "ORTM"
  const head = u.subarray(0, Math.min(u.length, 4096));
  let printable = head.length > 0;
  for (const b of head) {
    if (b < 9 || (b > 13 && b < 32) || b === 127) {
      printable = false;
      break;
    }
  }
  if (printable) {
    const t = utf8.decode(head).trimStart();
    if (t.startsWith("{") && (lower.endsWith(".json") || /"(irVersion|ir_version|graph)"/.test(t))) return "onnx-json";
    if (
      /\.(prototxt|pbtxt|txt)$/.test(lower) ||
      /^(#[^\n]*\n\s*)*(ir_version|graph|opset_import|producer_name|producer_version|model_version|doc_string|domain|metadata_props)\b/.test(t)
    )
      return "onnx-text";
  }
  if (lower.endsWith(".ort")) return "ort";
  return "onnx";
}

export { ATTR };
