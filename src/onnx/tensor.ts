/**
 * Tensor payload access: dtype decoding, external data, sparse densification, stats, slices, histograms.
 * All of it lives in the worker; the main thread only sees TensorInfo / results.
 */
import type { RawTensor, RawSparse } from "./decode";
import { dtypeBits, dtypeBytes, isFloatDType } from "./dtype";
import type {
  HistogramRequest, HistogramResult, SliceRequest, TensorInfo, TensorSliceResult, TensorStats, TensorSummary,
} from "./types";

export interface TensorEntry {
  info: TensorInfo;
  raw: RawTensor | null;
  sparse: RawSparse | null;
}

export interface TensorSource {
  dtype: string;
  n: number;
  dims: number[];
  kind: "num" | "string" | "complex";
  /** fills out[0..count) (complex: 2*count, interleaved). Returns true if int64 values were rounded. */
  fill(out: Float64Array, start: number, count: number): boolean;
  strings?(start: number, count: number): string[];
}

/* ───────────── endianness ───────────── */

let LE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
/** test hook: force the DataView (big-endian-safe) code paths */
export function __forceSlowPath(on: boolean): void {
  LE = !on;
}

/* ───────────── float decoding ───────────── */

let F16: Float32Array | null = null;
function f16lut(): Float32Array {
  if (F16) return F16;
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 31;
    const m = h & 1023;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return (F16 = t);
}

const LUT8 = new Map<string, Float32Array>();
function lut8(d: string): Float32Array {
  let t = LUT8.get(d);
  if (t) return t;
  t = new Float32Array(256);
  for (let b = 0; b < 256; b++) {
    const s = b & 0x80 ? -1 : 1;
    let v: number;
    switch (d) {
      case "float8e4m3fn": {
        const e = (b >> 3) & 15;
        const m = b & 7;
        v = e === 15 && m === 7 ? NaN : e === 0 ? s * m * 2 ** -9 : s * (1 + m / 8) * 2 ** (e - 7);
        break;
      }
      case "float8e4m3fnuz": {
        const e = (b >> 3) & 15;
        const m = b & 7;
        v = b === 0x80 ? NaN : e === 0 ? s * m * 2 ** -10 : s * (1 + m / 8) * 2 ** (e - 8);
        break;
      }
      case "float8e5m2": {
        const e = (b >> 2) & 31;
        const m = b & 3;
        v = e === 31 ? (m ? NaN : s * Infinity) : e === 0 ? s * m * 2 ** -16 : s * (1 + m / 4) * 2 ** (e - 15);
        break;
      }
      case "float8e5m2fnuz": {
        const e = (b >> 2) & 31;
        const m = b & 3;
        v = b === 0x80 ? NaN : e === 0 ? s * m * 2 ** -17 : s * (1 + m / 4) * 2 ** (e - 16);
        break;
      }
      case "float8e8m0":
        v = b === 255 ? NaN : 2 ** (b - 127);
        break;
      default:
        v = NaN;
    }
    t[b] = v;
  }
  LUT8.set(d, t);
  return t;
}

const FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
const fp4 = (n: number): number => (n & 8 ? -FP4[n & 7]! : FP4[n & 7]!);

/* ───────────── byte access ───────────── */

/** typed-array view over bytes[start, start+len) — copies only when the byte offset is misaligned */
function view<T extends ArrayBufferView>(
  C: { new (b: ArrayBufferLike, o: number, l: number): T; BYTES_PER_ELEMENT: number },
  bytes: Uint8Array, startByte: number, count: number,
): T {
  const sz = C.BYTES_PER_ELEMENT;
  if (count <= 0) return new C(new ArrayBuffer(0), 0, 0);
  const off = bytes.byteOffset + startByte;
  if (off % sz === 0) return new C(bytes.buffer, off, count);
  const copy = new Uint8Array(count * sz);
  copy.set(bytes.subarray(startByte, startByte + count * sz));
  return new C(copy.buffer, 0, count);
}

/** canonical little-endian byte layout of an in-memory typed field */
function typedToBytes(t: RawTensor, dtype: string): Uint8Array | null {
  const bits = dtypeBits(dtype);
  const eb = Math.max(1, Math.ceil(bits / 8));
  if (t.floatData && (dtype === "float32" || dtype === "complex64")) return t.floatData;
  if (t.doubleData && (dtype === "float64" || dtype === "complex128")) return t.doubleData;
  if (t.floatData && !t.raw) return t.floatData;
  if (t.doubleData && !t.raw) return t.doubleData;
  if (t.int32Data) {
    const a = t.int32Data;
    const out = new Uint8Array(a.length * eb);
    for (let i = 0, o = 0; i < a.length; i++) {
      let v = a[i]!;
      for (let k = 0; k < eb; k++, v >>= 8) out[o++] = v & 255;
    }
    return out;
  }
  const w = t.int64Data ?? t.uint64Data;
  if (w) {
    const lo = new Uint8Array(w.buffer, w.byteOffset, w.byteLength);
    if (eb === 8) return lo;
    const out = new Uint8Array(w.length * eb);
    for (let i = 0; i < w.length; i++) for (let k = 0; k < eb; k++) out[i * eb + k] = lo[i * 8 + k]!;
    return out;
  }
  return null;
}

function normPath(p: string): string {
  return p.replace(/\\/g, "/").replace(/^\.\//, "").replace(/^\/+/, "");
}

/* ───────────── sources ───────────── */

const U32 = 4294967296;
const MAXSAFE_HI = 2097152; // |hi| < 2^21 → value is exactly representable

function numericFill(dtype: string, bytes: Uint8Array, n: number): TensorSource["fill"] {
  const bits = dtypeBits(dtype);
  const avail = Math.floor((bytes.length * 8) / (bits || 8));
  const clipN = (start: number, count: number) => Math.max(0, Math.min(count, avail - start));
  const dv = !LE ? new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength) : null;

  switch (dtype) {
    case "float32":
      return (out, start, count) => {
        const m = clipN(start, count);
        if (dv) for (let i = 0; i < m; i++) out[i] = dv.getFloat32((start + i) * 4, true);
        else {
          const a = view(Float32Array, bytes, start * 4, m);
          for (let i = 0; i < m; i++) out[i] = a[i]!;
        }
        out.fill(0, m, count);
        return false;
      };
    case "float64":
      return (out, start, count) => {
        const m = clipN(start, count);
        if (dv) for (let i = 0; i < m; i++) out[i] = dv.getFloat64((start + i) * 8, true);
        else out.set(view(Float64Array, bytes, start * 8, m));
        out.fill(0, m, count);
        return false;
      };
    case "float16":
    case "bfloat16": {
      const bf = dtype === "bfloat16";
      return (out, start, count) => {
        const m = clipN(start, count);
        if (bf) {
          const tmp = new Uint32Array(1);
          const f = new Float32Array(tmp.buffer);
          for (let i = 0; i < m; i++) {
            const o = (start + i) * 2;
            tmp[0] = (bytes[o]! | (bytes[o + 1]! << 8)) << 16;
            out[i] = f[0]!;
          }
        } else {
          const lut = f16lut();
          for (let i = 0; i < m; i++) {
            const o = (start + i) * 2;
            out[i] = lut[bytes[o]! | (bytes[o + 1]! << 8)]!;
          }
        }
        out.fill(0, m, count);
        return false;
      };
    }
    case "float8e4m3fn":
    case "float8e4m3fnuz":
    case "float8e5m2":
    case "float8e5m2fnuz":
    case "float8e8m0": {
      const lut = lut8(dtype);
      return (out, start, count) => {
        const m = clipN(start, count);
        for (let i = 0; i < m; i++) out[i] = lut[bytes[start + i]!]!;
        out.fill(0, m, count);
        return false;
      };
    }
    case "int8":
      return (out, start, count) => {
        const m = clipN(start, count);
        for (let i = 0; i < m; i++) out[i] = (bytes[start + i]! << 24) >> 24;
        out.fill(0, m, count);
        return false;
      };
    case "uint8":
    case "bool":
      return (out, start, count) => {
        const m = clipN(start, count);
        for (let i = 0; i < m; i++) out[i] = bytes[start + i]!;
        out.fill(0, m, count);
        return false;
      };
    case "int16":
    case "uint16": {
      const sg = dtype === "int16";
      return (out, start, count) => {
        const m = clipN(start, count);
        for (let i = 0; i < m; i++) {
          const o = (start + i) * 2;
          const v = bytes[o]! | (bytes[o + 1]! << 8);
          out[i] = sg ? (v << 16) >> 16 : v;
        }
        out.fill(0, m, count);
        return false;
      };
    }
    case "int32":
    case "uint32": {
      const sg = dtype === "int32";
      return (out, start, count) => {
        const m = clipN(start, count);
        if (dv) for (let i = 0; i < m; i++) out[i] = sg ? dv.getInt32((start + i) * 4, true) : dv.getUint32((start + i) * 4, true);
        else if (sg) {
          const a = view(Int32Array, bytes, start * 4, m);
          for (let i = 0; i < m; i++) out[i] = a[i]!;
        } else {
          const a = view(Uint32Array, bytes, start * 4, m);
          for (let i = 0; i < m; i++) out[i] = a[i]!;
        }
        out.fill(0, m, count);
        return false;
      };
    }
    case "int64":
    case "uint64": {
      const sg = dtype === "int64";
      return (out, start, count) => {
        const m = clipN(start, count);
        let lossy = false;
        if (dv) {
          for (let i = 0; i < m; i++) {
            const lo = dv.getUint32((start + i) * 8, true);
            const hi = sg ? dv.getInt32((start + i) * 8 + 4, true) : dv.getUint32((start + i) * 8 + 4, true);
            if (hi >= MAXSAFE_HI || hi < -MAXSAFE_HI) lossy = true;
            out[i] = hi * U32 + lo;
          }
        } else {
          const a = view(Int32Array, bytes, start * 8, m * 2);
          for (let i = 0; i < m; i++) {
            const lo = a[2 * i]! >>> 0;
            let hi = a[2 * i + 1]!;
            if (!sg) hi >>>= 0;
            if (hi >= MAXSAFE_HI || hi < -MAXSAFE_HI) lossy = true;
            out[i] = hi * U32 + lo;
          }
        }
        out.fill(0, m, count);
        return lossy;
      };
    }
    case "int4":
    case "uint4":
    case "float4e2m1": {
      return (out, start, count) => {
        const m = clipN(start, count);
        for (let i = 0; i < m; i++) {
          const k = start + i;
          const b = bytes[k >> 1]!;
          const nib = k & 1 ? b >> 4 : b & 15;
          out[i] = dtype === "uint4" ? nib : dtype === "int4" ? (nib & 8 ? nib - 16 : nib) : fp4(nib);
        }
        out.fill(0, m, count);
        return false;
      };
    }
    default:
      return (out, _s, count) => {
        out.fill(NaN, 0, count);
        return false;
      };
  }
  void n;
}

function complexFill(dtype: string, bytes: Uint8Array): TensorSource["fill"] {
  const dbl = dtype === "complex128";
  const sz = dbl ? 16 : 8;
  const avail = Math.floor(bytes.length / sz);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return (out, start, count) => {
    const m = Math.max(0, Math.min(count, avail - start));
    for (let i = 0; i < m; i++) {
      const o = (start + i) * sz;
      out[2 * i] = dbl ? dv.getFloat64(o, true) : dv.getFloat32(o, true);
      out[2 * i + 1] = dbl ? dv.getFloat64(o + 8, true) : dv.getFloat32(o + 4, true);
    }
    out.fill(0, 2 * m, 2 * count);
    return false;
  };
}

export class TensorStore {
  entries: TensorEntry[] = [];
  private ext = new Map<string, Uint8Array>();
  private extBase = new Map<string, Uint8Array>();
  private bytesCache = new Map<number, Uint8Array | null>();
  private srcCache = new Map<number, TensorSource | null>();
  private statsCache = new Map<number, TensorStats>();

  add(e: Omit<TensorEntry, "info"> & { info: Omit<TensorInfo, "id"> }): number {
    const id = this.entries.length;
    this.entries.push({ ...e, info: { ...e.info, id } as TensorInfo });
    return id;
  }

  setExternal(path: string, data: Uint8Array): void {
    const p = normPath(path);
    this.ext.set(p, data);
    this.extBase.set(p.split("/").pop()!, data);
    this.bytesCache.clear();
    this.srcCache.clear();
    this.statsCache.clear();
  }

  private findExternal(loc: string): Uint8Array | null {
    const p = normPath(loc);
    return this.ext.get(p) ?? this.extBase.get(p.split("/").pop()!) ?? null;
  }

  /** raw payload bytes in canonical little-endian layout; null if not available */
  private bytesOf(id: number): Uint8Array | null {
    if (this.bytesCache.has(id)) return this.bytesCache.get(id)!;
    const e = this.entries[id]!;
    const t = e.raw;
    let out: Uint8Array | null = null;
    if (t) {
      if (t.external) {
        const f = this.findExternal(t.external.location);
        if (f && t.external.offset <= f.length) {
          const end = t.external.length != null ? Math.min(f.length, t.external.offset + t.external.length) : f.length;
          out = f.subarray(t.external.offset, end);
        }
      } else if (t.raw) out = t.raw;
      else out = typedToBytes(t, e.info.dtype);
    }
    this.bytesCache.set(id, out);
    return out;
  }

  /** is a payload present for this tensor? */
  resolve(id: number): boolean {
    const e = this.entries[id]!;
    if (e.sparse) return true;
    if (e.info.dtype === "string") return !!e.raw?.stringData || e.info.n === 0;
    return e.info.n === 0 || this.bytesOf(id) !== null;
  }

  source(id: number): TensorSource | null {
    if (this.srcCache.has(id)) return this.srcCache.get(id)!;
    const s = this.makeSource(id);
    this.srcCache.set(id, s);
    return s;
  }

  private makeSource(id: number): TensorSource | null {
    const e = this.entries[id]!;
    const { dtype, n, dims } = e.info;
    if (e.sparse) return this.sparseSource(e);
    if (dtype === "string") {
      const sd = e.raw?.stringData;
      if (!sd) return n === 0 ? { dtype, n, dims, kind: "string", fill: () => false, strings: () => [] } : null;
      const dec = new TextDecoder();
      return {
        dtype, n, dims, kind: "string", fill: () => false,
        strings: (s, c) => sd.slice(s, s + c).map((b) => dec.decode(b)),
      };
    }
    if (dtype === "undefined" || dtype.startsWith("dtype#")) return null;
    const bytes = this.bytesOf(id) ?? (n === 0 ? new Uint8Array(0) : null);
    if (!bytes) return null;
    if (dtype === "complex64" || dtype === "complex128") return { dtype, n, dims, kind: "complex", fill: complexFill(dtype, bytes) };
    return { dtype, n, dims, kind: "num", fill: numericFill(dtype, bytes, n) };
  }

  private sparseSource(e: TensorEntry): TensorSource | null {
    const sp = e.info.sparse!;
    const vs = this.source(sp.valuesTid);
    const is = this.source(sp.indicesTid);
    if (!vs || !is || vs.kind !== "num" || is.kind !== "num") return null;
    const { dims, n, dtype } = e.info;
    const nnz = Math.min(vs.n, is.n);
    const vals = new Float64Array(vs.n);
    vs.fill(vals, 0, vs.n);
    const rank = dims.length;
    const idxDims = this.entries[sp.indicesTid]!.info.dims;
    const lin = new Float64Array(vs.n);
    if (idxDims.length === 2 && idxDims[1] === rank && rank > 0) {
      const raw = new Float64Array(is.n);
      is.fill(raw, 0, is.n);
      const stride = new Array<number>(rank).fill(1);
      for (let d = rank - 2; d >= 0; d--) stride[d] = stride[d + 1]! * dims[d + 1]!;
      for (let i = 0; i < nnz; i++) {
        let l = 0;
        for (let d = 0; d < rank; d++) l += raw[i * rank + d]! * stride[d]!;
        lin[i] = l;
      }
    } else is.fill(lin, 0, nnz);
    let order: Uint32Array | null = null;
    for (let i = 1; i < nnz; i++)
      if (lin[i]! < lin[i - 1]!) {
        order = Uint32Array.from({ length: nnz }, (_, k) => k).sort((a, b) => lin[a]! - lin[b]!);
        break;
      }
    const pos = order ? Float64Array.from(order, (k) => lin[k]!) : lin.subarray(0, nnz);
    const val = order ? Float64Array.from(order, (k) => vals[k]!) : vals;
    return {
      dtype, n, dims, kind: "num",
      fill(out, start, count) {
        out.fill(0, 0, count);
        let lo = 0;
        let hi = nnz;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (pos[mid]! < start) lo = mid + 1;
          else hi = mid;
        }
        for (let k = lo; k < nnz && pos[k]! < start + count; k++) out[pos[k]! - start] = val[k]!;
        return vs.dtype === "int64" || vs.dtype === "uint64" ? false : false;
      },
    };
  }

  /* ───────── summaries ───────── */

  /** recompute availability + summary for one tensor */
  refresh(id: number): void {
    const e = this.entries[id]!;
    e.info.available = this.resolve(id);
    e.info.summary = undefined;
    if (!e.info.available) return;
    const src = this.source(id);
    if (src && src.kind === "num" && src.n > 0) e.info.summary = scan(src, {}).summary;
  }

  refreshAll(onProgress?: (frac: number) => void): void {
    const total = this.entries.reduce((a, e) => a + e.info.n, 0) || 1;
    let done = 0;
    for (let i = 0; i < this.entries.length; i++) {
      const e = this.entries[i]!;
      if (e.info.sparse && !e.raw) continue;
      this.refresh(i);
      done += e.info.n;
      onProgress?.(done / total);
    }
    // sparse tensors: summary from the densified source is cheap relative to nnz; derive lazily instead
    for (let i = 0; i < this.entries.length; i++) {
      const e = this.entries[i]!;
      if (e.sparse) {
        e.info.available = this.resolve(i) && !!this.source(i);
        if (e.info.available && e.info.n <= 50_000_000) {
          const src = this.source(i);
          if (src) e.info.summary = scan(src, {}).summary;
        }
      }
    }
  }

  /* ───────── stats ───────── */

  stats(id: number): TensorStats {
    const hit = this.statsCache.get(id);
    if (hit) return hit;
    const e = this.entries[id]!;
    const src = this.source(id);
    const blank: TensorStats = {
      min: null, max: null, mean: null, std: null, absmax: null, zeros: 0, nan: 0, inf: 0, median: null, absmean: null,
      p1: null, p99: null, l2: null, unique: 0, hist: null, histMin: null, histMax: null, channelAbsMax: null, sampled: false,
    };
    if (!src) return blank;
    if (src.kind === "string") {
      const all = src.strings!(0, Math.min(src.n, 1_000_000));
      return { ...blank, unique: new Set(all).size };
    }
    if (src.kind === "complex") return blank;
    const dims = e.info.dims;
    const ch = dims.length >= 2 && dims[0]! > 1 ? dims[0]! : 0;
    const r = scan(src, { channels: ch, collect: true });
    const s = r.summary;
    let median: number | null = null;
    let p1: number | null = null;
    let p99: number | null = null;
    let unique = 0;
    if (r.sorted && r.sorted.length) {
      const a = r.sorted;
      const q = (p: number) => {
        const x = (a.length - 1) * p;
        const lo = Math.floor(x);
        const hi = Math.ceil(x);
        return a[lo]! + (a[hi]! - a[lo]!) * (x - lo);
      };
      median = q(0.5);
      p1 = q(0.01);
      p99 = q(0.99);
      unique = 1;
      for (let i = 1; i < a.length; i++) if (a[i] !== a[i - 1]) unique++;
    }
    let hist: number[] | null = null;
    let histMin: number | null = null;
    let histMax: number | null = null;
    if (s.min != null && s.max != null && isFinite(s.min) && isFinite(s.max) && !(src.dtype === "bool" && false)) {
      const h = histogramPass(src, 64, s.min, s.max, false);
      hist = h.counts;
      histMin = s.min;
      histMax = s.max;
    }
    const out: TensorStats = {
      ...s, median, absmean: r.absmean, p1, p99, l2: r.l2, unique, hist, histMin, histMax,
      channelAbsMax: r.channels ? Array.from(r.channels) : null, sampled: r.sampled,
    };
    this.statsCache.set(id, out);
    return out;
  }

  histogram(req: HistogramRequest): HistogramResult {
    const src = this.source(req.id);
    if (!src || src.kind !== "num") return { counts: new Array(req.bins).fill(0), min: req.min ?? 0, max: req.max ?? 0, below: 0, above: 0 };
    let { min, max } = req;
    if (min === undefined || max === undefined) {
      const s = this.entries[req.id]!.info.summary ?? scan(src, {}).summary;
      min ??= s.min ?? 0;
      max ??= s.max ?? 0;
    }
    return histogramPass(src, Math.max(1, Math.floor(req.bins)), min, max, !!req.log);
  }

  /* ───────── slices ───────── */

  slice(req: SliceRequest): TensorSliceResult {
    const e = this.entries[req.id]!;
    const dims = e.info.dims;
    const rank = dims.length;
    const maxElems = req.maxElems ?? 4_000_000;
    const off = dims.map((d, i) => Math.min(Math.max(0, req.offset?.[i] ?? 0), Math.max(0, d - 1)));
    const size = dims.map((d, i) => Math.max(0, Math.min(req.size?.[i] ?? d - off[i]!, d - off[i]!)));
    const empty = (): TensorSliceResult => ({
      dtype: e.info.dtype, dims, offset: off, size, values: e.info.dtype === "string" ? [] : new Float64Array(0), lossy: false, truncated: false,
    });
    const src = this.source(req.id);
    if (!src) return empty();
    let total = size.reduce((a, b) => a * b, 1);
    let truncated = false;
    if (total > maxElems) {
      truncated = true;
      for (let d = 0; d < rank && total > maxElems; d++) {
        const others = total / size[d]!;
        size[d] = Math.max(1, Math.floor(maxElems / others));
        total = size.reduce((a, b) => a * b, 1);
      }
      if (total > maxElems) {
        size[rank - 1] = maxElems;
        total = size.reduce((a, b) => a * b, 1);
      }
    }
    if (total === 0) return { ...empty(), size };
    const stride = new Array<number>(rank).fill(1);
    for (let d = rank - 2; d >= 0; d--) stride[d] = stride[d + 1]! * dims[d + 1]!;

    if (src.kind === "string") {
      const vals: string[] = [];
      if (rank === 0) return { dtype: e.info.dtype, dims, offset: off, size, values: src.strings!(0, 1), lossy: false, truncated };
      forRuns(dims, off, size, stride, (lin, len) => {
        for (const s of src.strings!(lin, len)) vals.push(s);
      });
      return { dtype: e.info.dtype, dims, offset: off, size, values: vals, lossy: false, truncated };
    }
    const w = src.kind === "complex" ? 2 : 1;
    const out = new Float64Array(total * w);
    let lossy = false;
    let o = 0;
    if (rank === 0) lossy = src.fill(out, 0, 1);
    else
      forRuns(dims, off, size, stride, (lin, len) => {
        const tmp = len * w === out.length - o ? out.subarray(o) : out.subarray(o, o + len * w);
        if (src.fill(tmp, lin, len)) lossy = true;
        o += len * w;
      });
    return { dtype: e.info.dtype, dims, offset: off, size, values: out, lossy, truncated };
  }
}

/** iterate contiguous runs of a hyper-rectangular slice in row-major order */
function forRuns(
  dims: number[], off: number[], size: number[], stride: number[], cb: (linearStart: number, len: number) => void,
): void {
  const rank = dims.length;
  let k = rank - 1;
  while (k > 0 && off[k] === 0 && size[k] === dims[k]) k--;
  const inner = stride[k]!; // elements per index step of dim k
  const runLen = size[k]! * inner;
  const idx = new Array<number>(k).fill(0);
  for (;;) {
    let lin = off[k]! * inner;
    for (let d = 0; d < k; d++) lin += (off[d]! + idx[d]!) * stride[d]!;
    cb(lin, runLen);
    let d = k - 1;
    while (d >= 0) {
      if (++idx[d]! < size[d]!) break;
      idx[d] = 0;
      d--;
    }
    if (d < 0) return;
  }
}

/* ───────────── scanning ───────────── */

const CHUNK = 1 << 16;
export const SAMPLE_LIMIT = 20_000_000;
const SAMPLE_TARGET = 4_000_000;

interface ScanResult {
  summary: TensorSummary;
  absmean: number | null;
  l2: number | null;
  sorted: Float64Array | null;
  sampled: boolean;
  channels: Float64Array | null;
  finite: number;
}

export function scan(src: TensorSource, opts: { channels?: number; collect?: boolean }): ScanResult {
  const n = src.n;
  const buf = new Float64Array(CHUNK);
  let nan = 0, inf = 0, zeros = 0, finite = 0;
  let mn = Infinity, mx = -Infinity, amax = 0;
  let K = 0, s1 = 0, s2 = 0, sabs = 0, sq = 0;
  const sampled = !!opts.collect && n > SAMPLE_LIMIT;
  const stride = sampled ? Math.ceil(n / SAMPLE_TARGET) : 1;
  const coll = opts.collect ? new Float64Array(sampled ? Math.ceil(n / stride) + 1 : n) : null;
  let cn = 0;
  const nch = opts.channels ?? 0;
  const channels = nch ? new Float64Array(nch) : null;
  const inner = nch ? Math.max(1, Math.floor(n / nch)) : 1;
  let ch = 0;
  let rem = 0;
  for (let start = 0; start < n; start += CHUNK) {
    const m = Math.min(CHUNK, n - start);
    src.fill(buf, start, m);
    for (let j = 0; j < m; j++) {
      const x = buf[j]!;
      if (channels) {
        if (rem === inner) {
          ch++;
          rem = 0;
        }
        rem++;
        if (x === x && x !== Infinity && x !== -Infinity) {
          const a = x < 0 ? -x : x;
          if (a > channels[ch]!) channels[ch] = a;
        }
      }
      if (x !== x) {
        nan++;
        continue;
      }
      if (x === Infinity || x === -Infinity) {
        inf++;
        continue;
      }
      if (finite === 0) K = x;
      finite++;
      if (x === 0) zeros++;
      if (x < mn) mn = x;
      if (x > mx) mx = x;
      const a = x < 0 ? -x : x;
      if (a > amax) amax = a;
      const d = x - K;
      s1 += d;
      s2 += d * d;
      sabs += a;
      sq += x * x;
      if (coll && (!sampled || (start + j) % stride === 0)) coll[cn++] = x;
    }
  }
  let sorted: Float64Array | null = null;
  if (coll) {
    sorted = coll.subarray(0, cn);
    sorted.sort();
  }
  const mean = finite ? K + s1 / finite : null;
  const variance = finite ? Math.max(0, s2 / finite - (s1 / finite) ** 2) : 0;
  return {
    summary: {
      min: finite ? mn : null, max: finite ? mx : null, mean, std: finite ? Math.sqrt(variance) : null,
      absmax: finite ? amax : null, zeros: n ? zeros / n : 0, nan, inf,
    },
    absmean: finite ? sabs / finite : null,
    l2: finite ? Math.sqrt(sq) : null,
    sorted, sampled, channels, finite,
  };
}

export function histogramPass(src: TensorSource, bins: number, lo: number, hi: number, log: boolean): HistogramResult {
  const counts = new Array<number>(bins).fill(0);
  let below = 0;
  let above = 0;
  const buf = new Float64Array(CHUNK);
  const llo = log ? Math.log10(Math.max(lo, Number.MIN_VALUE)) : lo;
  const lhi = log ? Math.log10(Math.max(hi, Number.MIN_VALUE)) : hi;
  const span = lhi - llo;
  for (let start = 0; start < src.n; start += CHUNK) {
    const m = Math.min(CHUNK, src.n - start);
    src.fill(buf, start, m);
    for (let j = 0; j < m; j++) {
      const x = buf[j]!;
      if (x !== x || x === Infinity || x === -Infinity) continue;
      if (log && x <= 0) {
        below++;
        continue;
      }
      if (x < lo) {
        below++;
        continue;
      }
      if (x > hi) {
        above++;
        continue;
      }
      let b = span > 0 ? Math.floor(((log ? Math.log10(x) : x) - llo) / span * bins) : 0;
      if (b >= bins) b = bins - 1;
      if (b < 0) b = 0;
      counts[b]!++;
    }
  }
  return { counts, min: lo, max: hi, below, above };
}

/** dense payload size (bytes) of a tensor */
export function denseBytes(dtype: string, n: number, raw: RawTensor | null): number {
  if (dtype === "string") return raw?.stringData?.reduce((a, b) => a + b.length, 0) ?? 0;
  return dtypeBytes(dtype, n);
}

export { isFloatDType };
