/**
 * Lightweight, best-effort shape/type propagation for the most common operators.
 * Only used for values that have no value_info. Never throws (callers wrap in try/catch anyway);
 * returns null entries when something cannot be determined.
 */
import type { Dim, ValueType } from "./types";
import type { RawAttr } from "./decode";
import { dtypeName } from "./dtype";

export interface InferCtx {
  op: string;
  ins: (ValueType | null)[]; // by input position ("" inputs → null)
  attrs: Map<string, RawAttr>;
  nOut: number;
  /** integer content of a constant input, if known and small */
  constInts(i: number): number[] | null;
}

const T = (dtype: string, shape: Dim[] | null): ValueType => ({ kind: "tensor", dtype, shape });
type TensorT = Extract<ValueType, { kind: "tensor" }>;
const tt = (v: ValueType | null | undefined): TensorT | null => (v && v.kind === "tensor" ? v : null);
const ints = (a: RawAttr | undefined): number[] | null => (a?.ints ? a.ints.map(Number) : null);
const num = (a: RawAttr | undefined, d: number): number => (a?.i !== undefined ? Number(a.i) : d);

const UNARY = new Set(
  ("Relu Sigmoid Tanh Softmax LogSoftmax Hardmax Erf Gelu LeakyRelu Elu Selu HardSigmoid HardSwish Abs Neg Sqrt Exp Log Reciprocal " +
    "Floor Ceil Round Sign Clip Identity Not Sin Cos Tan Asin Acos Atan Sinh Cosh Asinh Acosh Atanh Softplus Softsign Mish Celu " +
    "ThresholdedRelu Shrink LRN LpNormalization MeanVarianceNormalization InstanceNormalization GroupNormalization PRelu BitwiseNot " +
    "IsNaN IsInf CumSum Relu6 Swish").split(" "),
);
const BINARY = new Set("Add Sub Mul Div Pow Max Min Sum Mean Mod And Or Xor BitwiseAnd BitwiseOr BitwiseXor".split(" "));
const COMPARE = new Set("Equal Greater Less GreaterOrEqual LessOrEqual".split(" "));
const REDUCE = new Set("ReduceMean ReduceSum ReduceMax ReduceMin ReduceProd ReduceL1 ReduceL2 ReduceSumSquare ReduceLogSum ReduceLogSumExp".split(" "));

function broadcast(a: Dim[], b: Dim[]): Dim[] {
  const n = Math.max(a.length, b.length);
  const out: Dim[] = [];
  for (let i = 0; i < n; i++) {
    const x = a[a.length - n + i] ?? 1;
    const y = b[b.length - n + i] ?? 1;
    out.push(x === y ? x : x === 1 ? y : y === 1 ? x : typeof x === "number" && typeof y === "number" ? Math.max(x, y) : typeof x === "string" ? x : y);
  }
  return out;
}

function convOut(inp: Dim, k: number, stride: number, dil: number, pb: number, pe: number, ceil = false): Dim {
  if (typeof inp !== "number") return "?";
  const num = inp + pb + pe - dil * (k - 1) - 1;
  const v = (ceil ? Math.ceil(num / stride) : Math.floor(num / stride)) + 1;
  return Math.max(0, v);
}

function spatial(c: InferCtx, x: Dim[], kernel: number[], isPool: boolean, outC: Dim | null): Dim[] {
  const rank = kernel.length;
  const strides = ints(c.attrs.get("strides")) ?? new Array(rank).fill(1);
  const dil = ints(c.attrs.get("dilations")) ?? new Array(rank).fill(1);
  const pads = ints(c.attrs.get("pads")) ?? new Array(rank * 2).fill(0);
  const ap = c.attrs.get("auto_pad")?.s ? new TextDecoder().decode(c.attrs.get("auto_pad")!.s) : "NOTSET";
  const ceil = num(c.attrs.get("ceil_mode"), 0) === 1;
  const out: Dim[] = [];
  for (let i = 0; i < rank; i++) {
    const inp = x[2 + i]!;
    if (ap === "SAME_UPPER" || ap === "SAME_LOWER") out.push(typeof inp === "number" ? Math.ceil(inp / strides[i]!) : "?");
    else if (ap === "VALID") out.push(convOut(inp, kernel[i]!, strides[i]!, dil[i]!, 0, 0, ceil));
    else out.push(convOut(inp, kernel[i]!, strides[i]!, dil[i]!, pads[i]!, pads[i + rank]!, ceil));
  }
  return [x[0]!, outC ?? x[1]!, ...out];
}

export function inferOutputs(c: InferCtx): (ValueType | null)[] | null {
  const x = tt(c.ins[0] ?? null);
  const out = (...v: (ValueType | null)[]) => v;
  const op = c.op;

  if (op === "Cast") {
    if (!x) return null;
    const to = c.attrs.get("to")?.i;
    return out(T(dtypeName(Number(to ?? 0)), x.shape));
  }
  if (op === "CastLike") {
    const y = tt(c.ins[1] ?? null);
    return x && y ? out(T(y.dtype, x.shape)) : null;
  }
  if (op === "QuantizeLinear") {
    const zp = tt(c.ins[2] ?? null);
    return x ? out(T(zp?.dtype ?? "uint8", x.shape)) : null;
  }
  if (op === "DequantizeLinear") {
    const s = tt(c.ins[1] ?? null);
    return x ? out(T(s?.dtype ?? "float32", x.shape)) : null;
  }
  if (op === "DynamicQuantizeLinear") return x ? out(T("uint8", x.shape), T("float32", []), T("uint8", [])) : null;
  if (op === "Dropout") return x ? out(x, T("bool", x.shape)) : null;
  if (op === "BatchNormalization" || op === "LayerNormalization") return x ? out(x, ...new Array(Math.max(0, c.nOut - 1)).fill(null)) : null;
  if (UNARY.has(op)) return x ? out(op === "IsNaN" || op === "IsInf" ? T("bool", x.shape) : x) : null;
  if (COMPARE.has(op) || BINARY.has(op)) {
    const ts = c.ins.map(tt).filter((v) => v);
    if (!ts.length) return null;
    let shape: Dim[] | null = ts[0]!.shape;
    for (const t of ts.slice(1)) shape = shape && t!.shape ? broadcast(shape, t!.shape) : null;
    return out(T(COMPARE.has(op) ? "bool" : ts[0]!.dtype, shape));
  }
  if (op === "Where") {
    const a = tt(c.ins[1] ?? null);
    const b = tt(c.ins[2] ?? null);
    const k = tt(c.ins[0] ?? null);
    if (!a) return null;
    let shape = a.shape;
    for (const t of [b, k]) shape = shape && t?.shape ? broadcast(shape, t.shape) : null;
    return out(T(a.dtype, shape));
  }
  if (op === "Conv" || op === "ConvInteger" || op === "QLinearConv") {
    const wi = op === "QLinearConv" ? 3 : 1;
    const w = tt(c.ins[wi] ?? null);
    if (!x?.shape || !w?.shape) return null;
    const kernel = ints(c.attrs.get("kernel_shape")) ?? (w.shape.slice(2) as number[]);
    const dt = op === "ConvInteger" ? "int32" : op === "QLinearConv" ? w.dtype : x.dtype;
    return out(T(dt, spatial(c, x.shape, kernel, false, w.shape[0]!)));
  }
  if (op === "ConvTranspose") {
    const w = tt(c.ins[1] ?? null);
    if (!x?.shape || !w?.shape) return null;
    const rank = x.shape.length - 2;
    const strides = ints(c.attrs.get("strides")) ?? new Array(rank).fill(1);
    const dil = ints(c.attrs.get("dilations")) ?? new Array(rank).fill(1);
    const pads = ints(c.attrs.get("pads")) ?? new Array(rank * 2).fill(0);
    const opad = ints(c.attrs.get("output_padding")) ?? new Array(rank).fill(0);
    const osh = ints(c.attrs.get("output_shape"));
    const g = num(c.attrs.get("group"), 1);
    const sp: Dim[] = [];
    for (let i = 0; i < rank; i++) {
      const inp = x.shape[2 + i]!;
      const k = w.shape[2 + i];
      if (osh) sp.push(osh[i]!);
      else sp.push(typeof inp === "number" && typeof k === "number" ? strides[i]! * (inp - 1) + opad[i]! + dil[i]! * (k - 1) + 1 - pads[i]! - pads[i + rank]! : "?");
    }
    const oc = typeof w.shape[1] === "number" ? w.shape[1] * g : "?";
    return out(T(x.dtype, [x.shape[0]!, oc, ...sp]));
  }
  if (op === "MaxPool" || op === "AveragePool" || op === "LpPool") {
    const kernel = ints(c.attrs.get("kernel_shape"));
    if (!x?.shape || !kernel) return null;
    const o = T(x.dtype, spatial(c, x.shape, kernel, true, null));
    return out(o, ...(c.nOut > 1 ? [T("int64", (o as { shape: Dim[] }).shape)] : []));
  }
  if (op === "GlobalAveragePool" || op === "GlobalMaxPool" || op === "GlobalLpPool")
    return x?.shape ? out(T(x.dtype, [x.shape[0]!, x.shape[1]!, ...x.shape.slice(2).map(() => 1)])) : null;
  if (op === "MatMul") {
    const a = tt(c.ins[0] ?? null);
    const b = tt(c.ins[1] ?? null);
    if (!a?.shape || !b?.shape) return a ? out(T(a.dtype, null)) : null;
    let as = a.shape;
    let bs = b.shape;
    const aV = as.length === 1;
    const bV = bs.length === 1;
    if (aV) as = [1, as[0]!];
    if (bV) bs = [bs[0]!, 1];
    const batch = broadcast(as.slice(0, -2), bs.slice(0, -2));
    const shape = [...batch, as[as.length - 2]!, bs[bs.length - 1]!];
    if (aV) shape.splice(shape.length - 2, 1);
    if (bV) shape.pop();
    return out(T(a.dtype, shape));
  }
  if (op === "MatMulInteger") {
    const a = tt(c.ins[0] ?? null);
    const b = tt(c.ins[1] ?? null);
    if (!a?.shape || !b?.shape) return out(T("int32", null));
    return out(T("int32", [...broadcast(a.shape.slice(0, -2), b.shape.slice(0, -2)), a.shape[a.shape.length - 2]!, b.shape[b.shape.length - 1]!]));
  }
  if (op === "Gemm") {
    const a = tt(c.ins[0] ?? null);
    const b = tt(c.ins[1] ?? null);
    if (!a?.shape || !b?.shape || a.shape.length !== 2 || b.shape.length !== 2) return a ? out(T(a.dtype, null)) : null;
    const tA = num(c.attrs.get("transA"), 0);
    const tB = num(c.attrs.get("transB"), 0);
    return out(T(a.dtype, [a.shape[tA ? 1 : 0]!, b.shape[tB ? 0 : 1]!]));
  }
  if (op === "Flatten") {
    if (!x?.shape) return null;
    let ax = num(c.attrs.get("axis"), 1);
    if (ax < 0) ax += x.shape.length;
    const prod = (d: Dim[]): Dim => (d.every((v) => typeof v === "number") ? (d as number[]).reduce((p, q) => p * q, 1) : "?");
    return out(T(x.dtype, [prod(x.shape.slice(0, ax)), prod(x.shape.slice(ax))]));
  }
  if (op === "Reshape") {
    const target = c.constInts(1);
    if (!x || !target) return x ? out(T(x.dtype, null)) : null;
    const allowzero = num(c.attrs.get("allowzero"), 0);
    const shape: Dim[] = target.map((v, i) => (v === 0 && !allowzero ? x.shape?.[i] ?? "?" : v));
    const known = x.shape && x.shape.every((v) => typeof v === "number") ? (x.shape as number[]).reduce((p, q) => p * q, 1) : null;
    const neg = shape.indexOf(-1);
    if (neg >= 0) {
      const rest = shape.filter((_, i) => i !== neg);
      shape[neg] = known != null && rest.every((v) => typeof v === "number" && v >= 0) ? (rest as number[]).reduce((p, q) => p * q, 1) === 0 ? 0 : known / (rest as number[]).reduce((p, q) => p * q, 1) : "?";
    }
    return out(T(x.dtype, shape));
  }
  if (op === "Transpose") {
    if (!x?.shape) return null;
    const perm = ints(c.attrs.get("perm")) ?? x.shape.map((_, i) => x.shape!.length - 1 - i);
    return out(T(x.dtype, perm.map((p) => x.shape![p]!)));
  }
  if (op === "Concat") {
    const ts = c.ins.map(tt).filter((v) => v);
    if (!ts.length || ts.some((t) => !t!.shape)) return ts.length ? out(T(ts[0]!.dtype, null)) : null;
    let ax = num(c.attrs.get("axis"), 0);
    const r = ts[0]!.shape!.length;
    if (ax < 0) ax += r;
    const shape = ts[0]!.shape!.slice();
    shape[ax] = ts.every((t) => typeof t!.shape![ax] === "number") ? ts.reduce((p, t) => p + (t!.shape![ax] as number), 0) : "?";
    return out(T(ts[0]!.dtype, shape));
  }
  if (op === "Squeeze" || op === "Unsqueeze") {
    if (!x?.shape) return x ? out(T(x.dtype, null)) : null;
    const axes = ints(c.attrs.get("axes")) ?? c.constInts(1);
    if (op === "Squeeze") {
      if (!axes) return out(T(x.dtype, x.shape.filter((d) => d !== 1)));
      const r = x.shape.length;
      const set = new Set(axes.map((a) => (a < 0 ? a + r : a)));
      return out(T(x.dtype, x.shape.filter((_, i) => !set.has(i))));
    }
    if (!axes) return out(T(x.dtype, null));
    const r = x.shape.length + axes.length;
    const set = new Set(axes.map((a) => (a < 0 ? a + r : a)));
    const shape: Dim[] = [];
    let k = 0;
    for (let i = 0; i < r; i++) shape.push(set.has(i) ? 1 : x.shape[k++]!);
    return out(T(x.dtype, shape));
  }
  if (op === "Shape") return x ? out(T("int64", [x.shape ? x.shape.length : "?"])) : out(T("int64", null));
  if (op === "Size") return out(T("int64", []));
  if (op === "Gather") {
    const idx = tt(c.ins[1] ?? null);
    if (!x?.shape || !idx?.shape) return x ? out(T(x.dtype, null)) : null;
    let ax = num(c.attrs.get("axis"), 0);
    if (ax < 0) ax += x.shape.length;
    return out(T(x.dtype, [...x.shape.slice(0, ax), ...idx.shape, ...x.shape.slice(ax + 1)]));
  }
  if (REDUCE.has(op) || op === "ArgMax" || op === "ArgMin") {
    if (!x?.shape) return x ? out(T(op.startsWith("Arg") ? "int64" : x.dtype, null)) : null;
    const dt = op.startsWith("Arg") ? "int64" : x.dtype;
    const r = x.shape.length;
    const keep = num(c.attrs.get("keepdims"), 1) === 1;
    let axes = op.startsWith("Arg") ? [num(c.attrs.get("axis"), 0)] : ints(c.attrs.get("axes")) ?? c.constInts(1);
    if (!axes) {
      if (c.ins[1] && !op.startsWith("Arg")) return out(T(dt, null));
      axes = x.shape.map((_, i) => i);
      if (num(c.attrs.get("noop_with_empty_axes"), 0) === 1) axes = [];
    }
    const set = new Set(axes.map((a) => (a < 0 ? a + r : a)));
    return out(T(dt, x.shape.flatMap((d, i) => (set.has(i) ? (keep ? [1] : []) : [d]))));
  }
  if (op === "Expand" || op === "Tile" || op === "Slice" || op === "Pad" || op === "Resize" || op === "Split") return x ? out(T(x.dtype, null), ...new Array(Math.max(0, c.nOut - 1)).fill(null)) : null;
  return null;
}
