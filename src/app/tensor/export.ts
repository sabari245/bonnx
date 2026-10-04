import type { ModelApi, TensorInfo } from "@/onnx/types";
import { toast } from "@/ui";
import { downloadBlob, errMsg, fetchLinear, isComplexDType, safeFileName } from "./util";

export const NPY_MAX_ELEMS = 50_000_000;
export const JSON_MAX_ELEMS = 1_000_000;
const CHUNK = 4_194_304;

interface NpyKind { descr: string; make(v: Float64Array): ArrayBufferView }
function npyKind(dtype: string): NpyKind | null {
  switch (dtype) {
    case "float64": return { descr: "<f8", make: (v) => v };
    case "complex128": return { descr: "<c16", make: (v) => v };
    case "float32": case "float16": case "bfloat16": case "float4e2m1":
    case "float8e4m3fn": case "float8e4m3fnuz": case "float8e5m2": case "float8e5m2fnuz": return { descr: "<f4", make: (v) => Float32Array.from(v) };
    case "complex64": return { descr: "<c8", make: (v) => Float32Array.from(v) };
    case "int8": case "int4": return { descr: "|i1", make: (v) => Int8Array.from(v) };
    case "uint8": case "uint4": return { descr: "|u1", make: (v) => Uint8Array.from(v) };
    case "bool": return { descr: "|b1", make: (v) => Uint8Array.from(v, (x) => (x ? 1 : 0)) };
    case "int16": return { descr: "<i2", make: (v) => Int16Array.from(v) };
    case "uint16": return { descr: "<u2", make: (v) => Uint16Array.from(v) };
    case "int32": return { descr: "<i4", make: (v) => Int32Array.from(v) };
    case "uint32": return { descr: "<u4", make: (v) => Uint32Array.from(v) };
    case "int64": return { descr: "<i8", make: (v) => BigInt64Array.from(v, (x) => BigInt(Math.trunc(x))) };
    case "uint64": return { descr: "<u8", make: (v) => BigUint64Array.from(v, (x) => BigInt(Math.max(0, Math.trunc(x)))) };
    default: return null;
  }
}

export function npyHeader(descr: string, dims: number[]): Uint8Array {
  const shape = dims.length === 0 ? "()" : dims.length === 1 ? `(${dims[0]},)` : `(${dims.join(", ")})`;
  let dict = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shape}, }`;
  const pre = 10; // magic(6)+version(2)+len(2)
  const pad = (64 - ((pre + dict.length + 1) % 64)) % 64;
  dict += " ".repeat(pad) + "\n";
  const out = new Uint8Array(pre + dict.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(out.buffer).setUint16(8, dict.length, true);
  for (let i = 0; i < dict.length; i++) out[pre + i] = dict.charCodeAt(i);
  return out;
}

export function canExportNpy(t: TensorInfo): string | null {
  if (!t.available) return "Tensor data is not available";
  if (t.dtype === "string") return "String tensors cannot be exported as .npy";
  if (!npyKind(t.dtype)) return `Unsupported dtype ${t.dtype}`;
  if (t.n > NPY_MAX_ELEMS) return `Tensor has more than ${NPY_MAX_ELEMS.toLocaleString()} elements`;
  return null;
}

/** Stream the whole tensor in slices into a .npy download. */
export async function exportNpy(api: ModelApi, t: TensorInfo): Promise<void> {
  const why = canExportNpy(t);
  if (why) return void toast({ title: "Cannot export .npy", description: why, variant: "destructive" });
  const kind = npyKind(t.dtype)!;
  const dismiss = toast({ title: `Exporting ${t.name}.npy…`, description: "Reading tensor data", duration: 600_000 });
  try {
    const parts: BlobPart[] = [npyHeader(kind.descr, t.dims) as BlobPart];
    let lossy = false;
    for (let lo = 0; lo < t.n; lo += CHUNK) {
      const r = await fetchLinear(api, t, lo, Math.min(t.n, lo + CHUNK));
      lossy ||= r.lossy;
      const a = kind.make(r.values as Float64Array);
      parts.push(a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength) as BlobPart);
    }
    downloadBlob(`${safeFileName(t.name)}.npy`, new Blob(parts, { type: "application/octet-stream" }));
    dismiss();
    const lowered = kind.descr === "<f4" && t.dtype !== "float32";
    toast({ title: "Exported .npy", description: [lowered ? `${t.dtype} stored as float32` : "", lossy ? "int64 values beyond 2^53 were rounded" : ""].filter(Boolean).join("; ") || undefined, variant: "success" });
  } catch (e) {
    dismiss();
    toast({ title: "Export failed", description: errMsg(e), variant: "destructive" });
  }
}

export function canExportJson(t: TensorInfo): string | null {
  if (!t.available) return "Tensor data is not available";
  if (t.n > JSON_MAX_ELEMS) return `JSON export is limited to ${JSON_MAX_ELEMS.toLocaleString()} elements`;
  return null;
}

export async function exportJson(api: ModelApi, t: TensorInfo): Promise<void> {
  const why = canExportJson(t);
  if (why) return void toast({ title: "Cannot export JSON", description: why, variant: "destructive" });
  try {
    const { values } = await fetchLinear(api, t, 0, t.n);
    const cplx = isComplexDType(t.dtype), isBool = t.dtype === "bool";
    const conv = (i: number): unknown => {
      if (t.dtype === "string") return (values as string[])[i];
      const v = values as Float64Array;
      if (cplx) return [num(v[2 * i]!), num(v[2 * i + 1]!)];
      return isBool ? !!v[i] : num(v[i]!);
    };
    const num = (x: number): number | string => (Number.isFinite(x) ? x : Number.isNaN(x) ? "NaN" : x > 0 ? "Infinity" : "-Infinity");
    const nest = (d: number, off: number, stride: number): unknown => {
      if (d === t.dims.length) return conv(off);
      const s = stride / t.dims[d]!;
      return Array.from({ length: t.dims[d]! }, (_, i) => nest(d + 1, off + i * s, s));
    };
    const data = nest(0, 0, t.n || 1);
    downloadBlob(`${safeFileName(t.name)}.json`, new Blob([JSON.stringify({ name: t.name, dtype: t.dtype, shape: t.dims, data })], { type: "application/json" }));
  } catch (e) {
    toast({ title: "Export failed", description: errMsg(e), variant: "destructive" });
  }
}

export function csvCell(s: string): string {
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
