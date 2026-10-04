import type { ModelApi, TensorInfo, TensorStats } from "@/onnx/types";
import { toast } from "@/ui";

/* ───────────── number formatting ───────────── */

export function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v == null) return "–";
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(3);
  return Number.isInteger(v) ? String(v) : v.toFixed(digits).replace(/\.?0+$/, "");
}

export function fmtCount(n: number): string {
  const a = Math.abs(n);
  if (a >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (a >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(Math.round(n));
}

export function fmtBytes(n: number): string {
  if (n >= 2 ** 30) return (n / 2 ** 30).toFixed(2) + " GB";
  if (n >= 2 ** 20) return (n / 2 ** 20).toFixed(2) + " MB";
  if (n >= 2 ** 10) return (n / 2 ** 10).toFixed(1) + " KB";
  return n + " B";
}

export const fmtPct = (v: number | null | undefined): string =>
  v == null ? "–" : (v * 100).toFixed(v > 0 && v < 0.001 ? 4 : 2) + " %";

export const shapeStr = (dims: readonly number[]): string => (dims.length ? dims.join("×") : "scalar");

export type NumFormat = "auto" | "fixed" | "exp";
export function fmtValue(v: number, mode: NumFormat, prec: number, integer: boolean): string {
  if (Number.isNaN(v)) return "NaN";
  if (!Number.isFinite(v)) return v > 0 ? "Inf" : "-Inf";
  if (integer) return String(v);
  if (mode === "fixed") return v.toFixed(prec);
  if (mode === "exp") return v.toExponential(prec);
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(Math.max(1, prec - 1));
  return Number(v.toPrecision(prec + 1)).toString();
}

/* ───────────── dtype helpers ───────────── */

export const isFloatDType = (d: string) => /^(float|bfloat)/.test(d);
export const isIntDType = (d: string) => /^u?int/.test(d);
export const isNumericDType = (d: string) => d !== "string" && !d.startsWith("complex") && d !== "undefined";
export const isComplexDType = (d: string) => d.startsWith("complex");

/* ───────────── stats cache ───────────── */

const statsCache = new WeakMap<ModelApi, Map<number, Promise<TensorStats>>>();
export function getStats(api: ModelApi, id: number): Promise<TensorStats> {
  let m = statsCache.get(api);
  if (!m) statsCache.set(api, (m = new Map()));
  let p = m.get(id);
  if (!p) {
    p = api.tensorStats(id);
    p.catch(() => m!.delete(id));
    m.set(id, p);
  }
  return p;
}
export function invalidateStats(api: ModelApi, id?: number): void {
  const m = statsCache.get(api);
  if (!m) return;
  id == null ? m.clear() : m.delete(id);
}

/* ───────────── shared context ───────────── */

export interface ViewerOpts {
  /** all tensors of the model: enables sparse values/indices switching */
  tensors?: TensorInfo[];
  /** initial tab */
  tab?: "overview" | "histogram" | "data" | "image";
  /** called when the user supplied an external data file and tensor availability changed */
  onExternalProvided?: (tensors: TensorInfo[]) => void;
}

export interface TensorCtx {
  api: ModelApi;
  /** the tensor currently being explored (may switch to a sparse component) */
  info: TensorInfo;
  opts: ViewerOpts;
  /** cached stats for the active tensor */
  stats(): Promise<TensorStats>;
}

/* ───────────── linear range → hyper-rectangles ───────────── */

export interface Box { offset: number[]; size: number[] }

/**
 * Decompose the row-major linear element range [lo, hi) of a tensor with `dims`
 * into a short, ordered list of hyper-rectangles (concatenating them in order yields the range).
 */
export function linearBoxes(dims: number[], lo: number, hi: number): Box[] {
  const r = dims.length;
  const out: Box[] = [];
  if (hi <= lo || dims.some((d) => d === 0)) return out;
  if (r === 0) return [{ offset: [], size: [] }];
  const zeros = (n: number) => new Array<number>(n).fill(0);
  const ones = (n: number) => new Array<number>(n).fill(1);
  const rec = (L: number, a: number, b: number, prefix: number[]): void => {
    let E = 1;
    for (let i = L; i < r; i++) E *= dims[i]!;
    if (a === 0 && b === E) {
      out.push({ offset: [...prefix, ...zeros(r - L)], size: [...ones(prefix.length), ...dims.slice(L)] });
      return;
    }
    if (L === r - 1) {
      out.push({ offset: [...prefix, a], size: [...ones(prefix.length), b - a] });
      return;
    }
    const s = E / dims[L]!;
    const f = Math.floor(a / s);
    const l = Math.floor((b - 1) / s);
    if (f === l) return rec(L + 1, a - f * s, b - f * s, [...prefix, f]);
    if (a % s !== 0) rec(L + 1, a - f * s, s, [...prefix, f]);
    const fs = a % s === 0 ? f : f + 1;
    const fe = b % s === 0 ? l + 1 : l;
    if (fe > fs) out.push({ offset: [...prefix, fs, ...zeros(r - L - 1)], size: [...ones(prefix.length), fe - fs, ...dims.slice(L + 1)] });
    if (b % s !== 0) rec(L + 1, 0, b - l * s, [...prefix, l]);
  };
  rec(0, lo, hi, []);
  return out;
}

/** Fetch the linear element range [lo,hi) (complex: interleaved so 2× values). */
export async function fetchLinear(
  api: ModelApi, info: TensorInfo, lo: number, hi: number,
): Promise<{ values: Float64Array | string[]; lossy: boolean }> {
  const boxes = linearBoxes(info.dims, lo, hi);
  const isStr = info.dtype === "string";
  const w = isComplexDType(info.dtype) ? 2 : 1;
  const parts = await Promise.all(
    boxes.map((b) => api.tensorSlice({ id: info.id, offset: b.offset, size: b.size, maxElems: b.size.reduce((x, y) => x * y, 1) + 1 })),
  );
  const lossy = parts.some((p) => p.lossy);
  if (isStr) return { values: parts.flatMap((p) => p.values as string[]), lossy };
  const out = new Float64Array(Math.max(0, hi - lo) * w);
  let o = 0;
  for (const p of parts) {
    const v = p.values as Float64Array;
    out.set(v.length + o > out.length ? v.subarray(0, out.length - o) : v, o);
    o += v.length;
  }
  return { values: out, lossy };
}

/* ───────────── misc ───────────── */

export function downloadBlob(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export async function copyText(text: string, what = "Copied"): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
    toast({ title: what, variant: "success", duration: 1800 });
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch { /* ignore */ }
    ta.remove();
    toast({ title: ok ? what : "Copy failed", variant: ok ? "success" : "destructive", duration: 1800 });
  }
}

export const safeFileName = (s: string) => s.replace(/[^\w.\-]+/g, "_").slice(0, 80) || "tensor";

export const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** read a design token as a CSS color string */
export function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Debounce with trailing call. */
export function debounce<A extends unknown[]>(fn: (...a: A) => void, ms: number): ((...a: A) => void) & { cancel(): void } {
  let t: ReturnType<typeof setTimeout> | undefined;
  const f = (...a: A) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  f.cancel = () => clearTimeout(t);
  return f;
}

/** Numeric text input that commits on change/Enter and rejects NaN. */
export function parseNum(s: string): number | null {
  const t = s.trim();
  if (!t) return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}
