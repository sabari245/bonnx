import type { Dim, ValueType } from "../onnx/types";

/** Number formatting: fixed for mid-range magnitudes, exponent for tiny/huge, "–" for null. */
export function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v == null) return "–";
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(3);
  return Number.isInteger(v) ? String(v) : v.toFixed(digits).replace(/\.?0+$/, "");
}

/** 1234 → 1.2K, 1.5e6 → 1.50M */
export function fmtCount(n: number): string {
  const a = Math.abs(n);
  if (a >= 1e12) return (n / 1e12).toFixed(2) + "T";
  if (a >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e8 ? 0 : a >= 1e7 ? 1 : 2) + "M";
  if (a >= 1e3) return (n / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "K";
  return String(Math.round(n));
}

export function fmtBytes(n: number): string {
  if (n >= 2 ** 40) return (n / 2 ** 40).toFixed(2) + " TB";
  if (n >= 2 ** 30) return (n / 2 ** 30).toFixed(2) + " GB";
  if (n >= 2 ** 20) return (n / 2 ** 20).toFixed(2) + " MB";
  if (n >= 2 ** 10) return (n / 2 ** 10).toFixed(1) + " KB";
  return n + " B";
}

export function fmtPercent(v: number | null | undefined): string {
  if (v == null) return "–";
  return (v * 100).toFixed(v > 0 && v < 0.001 ? 4 : 2) + " %";
}

/** "1×3×224×224", "scalar", "?" for unknown rank */
export function fmtShape(shape: readonly Dim[] | null | undefined, sep = "×"): string {
  if (!shape) return "?";
  if (!shape.length) return "scalar";
  return shape.join(sep);
}

const DT_SHORT: Record<string, string> = {
  float32: "f32", float64: "f64", float16: "f16", bfloat16: "bf16",
  int8: "i8", int16: "i16", int32: "i32", int64: "i64",
  uint8: "u8", uint16: "u16", uint32: "u32", uint64: "u64",
  bool: "bool", string: "str", complex64: "c64", complex128: "c128",
  float8e4m3fn: "f8e4m3fn", float8e4m3fnuz: "f8e4m3fnuz", float8e5m2: "f8e5m2", float8e5m2fnuz: "f8e5m2fnuz",
  int4: "i4", uint4: "u4", float4e2m1: "f4e2m1", undefined: "?",
};
export const dtypeShort = (d: string): string => DT_SHORT[d] ?? d;

export type DTypeCategory = "float" | "lowfloat" | "int" | "int8" | "bool" | "string" | "other";
export const DTYPE_CATEGORIES: { key: DTypeCategory; label: string }[] = [
  { key: "float", label: "float32/64" },
  { key: "lowfloat", label: "float16 / bfloat16 / fp8" },
  { key: "int", label: "int16 / 32 / 64" },
  { key: "int8", label: "int8 / int4" },
  { key: "bool", label: "bool" },
  { key: "string", label: "string" },
  { key: "other", label: "other" },
];
export function dtypeCategory(d: string | null | undefined): DTypeCategory {
  if (!d) return "other";
  if (d === "float32" || d === "float64") return "float";
  if (d === "float16" || d === "bfloat16" || d.startsWith("float8") || d.startsWith("float4")) return "lowfloat";
  if (d === "int8" || d === "uint8" || d === "int4" || d === "uint4") return "int8";
  if (/^u?int(16|32|64)$/.test(d)) return "int";
  if (d === "bool") return "bool";
  if (d === "string") return "string";
  return "other";
}

/** `f32[1,3,224,224]`, `seq<f32[?]>`, `map<i64,f32>`, `optional<…>`, `sparse<f32[10,10]>` */
export function fmtType(t: ValueType | null | undefined, full = false): string {
  if (!t) return "?";
  const dt = (d: string) => (full ? d : dtypeShort(d));
  switch (t.kind) {
    case "tensor": return t.shape ? `${dt(t.dtype)}[${t.shape.join(",")}]` : `${dt(t.dtype)}[?]`;
    case "sparse": return `sparse<${dt(t.dtype)}${t.shape ? `[${t.shape.join(",")}]` : ""}>`;
    case "sequence": return `seq<${fmtType(t.elem, full)}>`;
    case "map": return `map<${dt(t.key)},${fmtType(t.value, full)}>`;
    case "optional": return `optional<${fmtType(t.elem, full)}>`;
    case "opaque": return `opaque<${t.domain ? t.domain + "." : ""}${t.name}>`;
    default: return "?";
  }
}

/** dtype of the innermost tensor in a type (for dtype colouring); null if unknowable */
export function typeDType(t: ValueType | null | undefined): string | null {
  for (let i = 0; t && i < 8; i++) {
    if (t.kind === "tensor" || t.kind === "sparse") return t.dtype;
    if (t.kind === "sequence" || t.kind === "optional") t = t.elem;
    else if (t.kind === "map") t = t.value;
    else return null;
  }
  return null;
}

/** Compact list display: [1, 2, 3, … +40 more] */
export function shortList<T>(arr: readonly T[], max = 6, f: (x: T) => string = String): string {
  if (arr.length <= max) return `[${arr.map(f).join(", ")}]`;
  return `[${arr.slice(0, max).map(f).join(", ")}, … +${arr.length - max}]`;
}

export function truncate(s: string, max: number): string {
  return s.length <= max ? s : s.slice(0, Math.max(1, max - 1)) + "…";
}
