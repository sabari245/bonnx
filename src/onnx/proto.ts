/**
 * Minimal, allocation-light protobuf wire reader.
 * - never copies payloads: bytes() returns subarray views over the input
 * - truncated / malformed data raises ProtoError; callers decide whether to recover
 */

export class ProtoError extends Error {}

export class Warnings {
  list: string[] = [];
  dropped = 0;
  constructor(public limit = 40) {}
  add(msg: string): void {
    if (this.list.length < this.limit) this.list.push(msg);
    else this.dropped++;
  }
  toArray(): string[] {
    return this.dropped ? [...this.list, `… and ${this.dropped} more warnings`] : this.list.slice();
  }
}

const utf8 = new TextDecoder("utf-8");
export const decodeUtf8 = (b: Uint8Array): string => utf8.decode(b);

export const WIRE_VARINT = 0;
export const WIRE_64 = 1;
export const WIRE_LEN = 2;
export const WIRE_32 = 5;

export class Reader {
  pos: number;
  /** result of the last v64() call */
  lo = 0;
  hi = 0;
  private dv: DataView | null = null;

  constructor(
    readonly buf: Uint8Array,
    pos = 0,
    readonly end = buf.length,
    readonly warnings: Warnings = new Warnings(),
  ) {
    this.pos = pos;
  }

  eof(): boolean {
    return this.pos >= this.end;
  }

  private view(): DataView {
    return (this.dv ??= new DataView(this.buf.buffer, this.buf.byteOffset, this.buf.byteLength));
  }

  /** unsigned varint, exact up to 2^53 (higher bits lose precision) */
  varint(): number {
    const b = this.buf;
    let p = this.pos;
    let x = b[p++]!;
    let r = x & 0x7f;
    if (x < 0x80) return this.fin(p, r);
    x = b[p++]!;
    r |= (x & 0x7f) << 7;
    if (x < 0x80) return this.fin(p, r);
    x = b[p++]!;
    r |= (x & 0x7f) << 14;
    if (x < 0x80) return this.fin(p, r);
    x = b[p++]!;
    r |= (x & 0x7f) << 21;
    if (x < 0x80) return this.fin(p, r);
    let mult = 268435456; // 2^28
    for (let i = 0; i < 6; i++) {
      if (p >= this.end) throw new ProtoError("truncated varint");
      x = b[p++]!;
      r += (x & 0x7f) * mult;
      mult *= 128;
      if (x < 0x80) return this.fin(p, r);
    }
    throw new ProtoError("varint too long");
  }

  private fin(p: number, r: number): number {
    if (p > this.end) throw new ProtoError("truncated varint");
    this.pos = p;
    return r;
  }

  /** 64-bit varint into this.lo / this.hi (two's complement halves, as int32) */
  v64(): void {
    const b = this.buf;
    let p = this.pos;
    let lo = 0;
    let hi = 0;
    for (let i = 0; i < 10; i++) {
      if (p >= this.end) throw new ProtoError("truncated varint");
      const x = b[p++]!;
      const v = x & 0x7f;
      if (i < 4) lo |= v << (7 * i);
      else if (i === 4) {
        lo |= v << 28;
        hi = v >> 4;
      } else hi |= v << (7 * (i - 5) + 3);
      if (x < 0x80) {
        this.pos = p;
        this.lo = lo;
        this.hi = hi;
        return;
      }
    }
    throw new ProtoError("varint too long");
  }

  /** signed int64 as a JS number (rounded beyond 2^53) */
  int64(): number {
    this.v64();
    return this.hi * 4294967296 + (this.lo >>> 0);
  }

  /** signed int64: number when safe, otherwise exact decimal string */
  int64x(): number | string {
    this.v64();
    const hi = this.hi;
    if (hi >= -2097152 && hi < 2097152) return hi * 4294967296 + (this.lo >>> 0);
    return BigInt.asIntN(64, (BigInt(hi >>> 0) << 32n) | BigInt(this.lo >>> 0)).toString();
  }

  /** signed int64 as bigint */
  int64big(): bigint {
    this.v64();
    return BigInt.asIntN(64, (BigInt(this.hi >>> 0) << 32n) | BigInt(this.lo >>> 0));
  }

  uint64big(): bigint {
    this.v64();
    return (BigInt(this.hi >>> 0) << 32n) | BigInt(this.lo >>> 0);
  }

  /** int32 (varint, sign extended) */
  int32(): number {
    this.v64();
    return this.lo | 0;
  }

  float(): number {
    if (this.pos + 4 > this.end) throw new ProtoError("truncated fixed32");
    const v = this.view().getFloat32(this.pos, true);
    this.pos += 4;
    return v;
  }

  double(): number {
    if (this.pos + 8 > this.end) throw new ProtoError("truncated fixed64");
    const v = this.view().getFloat64(this.pos, true);
    this.pos += 8;
    return v;
  }

  /** length-delimited payload as a view; clamps (with a warning) when the data is truncated */
  bytes(): Uint8Array {
    const len = this.varint();
    let end = this.pos + len;
    if (end > this.end) {
      this.warnings.add(`truncated field: wanted ${len} bytes, only ${this.end - this.pos} left`);
      end = this.end;
    }
    const out = this.buf.subarray(this.pos, end);
    this.pos = end;
    return out;
  }

  string(): string {
    const b = this.bytes();
    return b.length ? utf8.decode(b) : "";
  }

  /** reader over a length-delimited sub-message (shares the buffer) */
  sub(): Reader {
    const len = this.varint();
    let end = this.pos + len;
    if (end > this.end) {
      this.warnings.add(`truncated message: wanted ${len} bytes, only ${this.end - this.pos} left`);
      end = this.end;
    }
    const r = new Reader(this.buf, this.pos, end, this.warnings);
    this.pos = end;
    return r;
  }

  skip(wire: number): void {
    switch (wire) {
      case WIRE_VARINT:
        this.v64();
        break;
      case WIRE_64:
        if (this.pos + 8 > this.end) throw new ProtoError("truncated fixed64");
        this.pos += 8;
        break;
      case WIRE_LEN: {
        const len = this.varint();
        if (this.pos + len > this.end) throw new ProtoError("truncated field");
        this.pos += len;
        break;
      }
      case WIRE_32:
        if (this.pos + 4 > this.end) throw new ProtoError("truncated fixed32");
        this.pos += 4;
        break;
      default:
        throw new ProtoError(`unsupported wire type ${wire}`);
    }
  }

  /** packed (or single, if wire !== 2) float values */
  floats(wire: number, out: number[]): void {
    if (wire === WIRE_LEN) {
      const s = this.sub();
      while (s.pos + 4 <= s.end) out.push(s.float());
    } else out.push(this.float());
  }

  /** packed (or single) varints decoded with fn */
  packed(wire: number, fn: (r: Reader) => void): void {
    if (wire === WIRE_LEN) {
      const s = this.sub();
      while (!s.eof()) fn(s);
    } else fn(this);
  }
}

/**
 * Run a message-decoding loop; on a ProtoError the partial result is kept and a warning is recorded.
 */
export function readMessage(r: Reader, what: string, onField: (field: number, wire: number) => void): void {
  try {
    while (!r.eof()) {
      const tag = r.varint();
      const field = tag >>> 3;
      const wire = tag & 7;
      if (field === 0) throw new ProtoError("field number 0");
      onField(field, wire);
    }
  } catch (e) {
    if (e instanceof ProtoError) r.warnings.add(`${what}: ${e.message}`);
    else throw e;
  }
}
