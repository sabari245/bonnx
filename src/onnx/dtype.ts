import type { DTypeName } from "./types";

/** TensorProto.DataType enum → name */
export const DTYPE_BY_ENUM: Record<number, DTypeName> = {
  0: "undefined",
  1: "float32",
  2: "uint8",
  3: "int8",
  4: "uint16",
  5: "int16",
  6: "int32",
  7: "int64",
  8: "string",
  9: "bool",
  10: "float16",
  11: "float64",
  12: "uint32",
  13: "uint64",
  14: "complex64",
  15: "complex128",
  16: "bfloat16",
  17: "float8e4m3fn",
  18: "float8e4m3fnuz",
  19: "float8e5m2",
  20: "float8e5m2fnuz",
  21: "uint4",
  22: "int4",
  23: "float4e2m1",
  24: "float8e8m0",
};

export const ENUM_BY_DTYPE: Record<string, number> = Object.fromEntries(
  Object.entries(DTYPE_BY_ENUM).map(([k, v]) => [v, +k]),
);

/** upper-case proto enum identifiers (text/JSON format) → enum number */
export const ENUM_BY_PROTO_NAME: Record<string, number> = {
  UNDEFINED: 0, FLOAT: 1, UINT8: 2, INT8: 3, UINT16: 4, INT16: 5, INT32: 6, INT64: 7, STRING: 8, BOOL: 9,
  FLOAT16: 10, DOUBLE: 11, UINT32: 12, UINT64: 13, COMPLEX64: 14, COMPLEX128: 15, BFLOAT16: 16,
  FLOAT8E4M3FN: 17, FLOAT8E4M3FNUZ: 18, FLOAT8E5M2: 19, FLOAT8E5M2FNUZ: 20, UINT4: 21, INT4: 22,
  FLOAT4E2M1: 23, FLOAT8E8M0: 24,
};

export const dtypeName = (e: number): DTypeName => DTYPE_BY_ENUM[e] ?? `dtype#${e}`;

/** bits per element (string = 0) */
export function dtypeBits(d: DTypeName): number {
  switch (d) {
    case "float64": case "int64": case "uint64": return 64;
    case "complex64": return 64;
    case "complex128": return 128;
    case "float32": case "int32": case "uint32": return 32;
    case "float16": case "bfloat16": case "int16": case "uint16": return 16;
    case "int8": case "uint8": case "bool": case "float8e4m3fn": case "float8e4m3fnuz": case "float8e5m2":
    case "float8e5m2fnuz": case "float8e8m0": return 8;
    case "int4": case "uint4": case "float4e2m1": return 4;
    default: return 0;
  }
}

export const isFloatDType = (d: DTypeName): boolean =>
  d === "float32" || d === "float64" || d === "float16" || d === "bfloat16" || d.startsWith("float8") || d.startsWith("float4");

export const isIntDType = (d: DTypeName): boolean => /^u?int(4|8|16|32|64)$/.test(d);

/** dense payload size in bytes for n elements */
export function dtypeBytes(d: DTypeName, n: number): number {
  const bits = dtypeBits(d);
  return bits ? Math.ceil((n * bits) / 8) : 0;
}
