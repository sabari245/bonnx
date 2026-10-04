/**
 * Static formal input/output names. Used as a fallback when operator metadata has no entry
 * (custom domains such as com.microsoft) or has not loaded. A trailing "..." marks a variadic tail
 * (names become "name[0]", "name[1]", …).
 */
export interface Roles {
  inputs: string[];
  outputs?: string[];
}

const R = (inputs: string[], outputs?: string[]): Roles => ({ inputs, outputs });

export const ROLES: Record<string, Roles> = {
  // ai.onnx core
  Conv: R(["X", "W", "B"], ["Y"]),
  ConvTranspose: R(["X", "W", "B"], ["Y"]),
  ConvInteger: R(["x", "w", "x_zero_point", "w_zero_point"], ["y"]),
  QLinearConv: R(["x", "x_scale", "x_zero_point", "w", "w_scale", "w_zero_point", "y_scale", "y_zero_point", "B"], ["y"]),
  MatMul: R(["A", "B"], ["Y"]),
  MatMulInteger: R(["A", "B", "a_zero_point", "b_zero_point"], ["Y"]),
  QLinearMatMul: R(["a", "a_scale", "a_zero_point", "b", "b_scale", "b_zero_point", "y_scale", "y_zero_point"], ["y"]),
  Gemm: R(["A", "B", "C"], ["Y"]),
  Add: R(["A", "B"], ["C"]), Sub: R(["A", "B"], ["C"]), Mul: R(["A", "B"], ["C"]), Div: R(["A", "B"], ["C"]),
  Pow: R(["X", "Y"], ["Z"]),
  Mod: R(["A", "B"], ["C"]),
  Equal: R(["A", "B"], ["C"]), Greater: R(["A", "B"], ["C"]), Less: R(["A", "B"], ["C"]),
  GreaterOrEqual: R(["A", "B"], ["C"]), LessOrEqual: R(["A", "B"], ["C"]),
  And: R(["A", "B"], ["C"]), Or: R(["A", "B"], ["C"]), Xor: R(["A", "B"], ["C"]),
  Where: R(["condition", "X", "Y"], ["output"]),
  PRelu: R(["X", "slope"], ["Y"]),
  Clip: R(["input", "min", "max"], ["output"]),
  Reshape: R(["data", "shape"], ["reshaped"]),
  Expand: R(["input", "shape"], ["output"]),
  Tile: R(["input", "repeats"], ["output"]),
  Slice: R(["data", "starts", "ends", "axes", "steps"], ["output"]),
  Pad: R(["data", "pads", "constant_value", "axes"], ["output"]),
  Resize: R(["X", "roi", "scales", "sizes"], ["Y"]),
  Upsample: R(["X", "scales"], ["Y"]),
  Squeeze: R(["data", "axes"], ["squeezed"]),
  Unsqueeze: R(["data", "axes"], ["expanded"]),
  Split: R(["input", "split"], ["outputs..."]),
  Concat: R(["inputs..."], ["concat_result"]),
  Sum: R(["data_0..."], ["sum"]), Max: R(["data_0..."], ["max"]), Min: R(["data_0..."], ["min"]), Mean: R(["data_0..."], ["mean"]),
  Transpose: R(["data"], ["transposed"]),
  Gather: R(["data", "indices"], ["output"]),
  GatherElements: R(["data", "indices"], ["output"]),
  GatherND: R(["data", "indices"], ["output"]),
  ScatterND: R(["data", "indices", "updates"], ["output"]),
  ScatterElements: R(["data", "indices", "updates"], ["output"]),
  Scatter: R(["data", "indices", "updates"], ["output"]),
  TopK: R(["X", "K"], ["Values", "Indices"]),
  ConstantOfShape: R(["input"], ["output"]),
  Range: R(["start", "limit", "delta"], ["output"]),
  CumSum: R(["x", "axis"], ["y"]),
  OneHot: R(["indices", "depth", "values"], ["output"]),
  ReduceMean: R(["data", "axes"], ["reduced"]), ReduceSum: R(["data", "axes"], ["reduced"]),
  ReduceMax: R(["data", "axes"], ["reduced"]), ReduceMin: R(["data", "axes"], ["reduced"]),
  ReduceProd: R(["data", "axes"], ["reduced"]), ReduceL1: R(["data", "axes"], ["reduced"]),
  ReduceL2: R(["data", "axes"], ["reduced"]), ReduceSumSquare: R(["data", "axes"], ["reduced"]),
  ReduceLogSum: R(["data", "axes"], ["reduced"]), ReduceLogSumExp: R(["data", "axes"], ["reduced"]),
  BatchNormalization: R(["X", "scale", "B", "input_mean", "input_var"], ["Y", "running_mean", "running_var"]),
  InstanceNormalization: R(["input", "scale", "B"], ["output"]),
  LayerNormalization: R(["X", "Scale", "B"], ["Y", "Mean", "InvStdDev"]),
  GroupNormalization: R(["X", "scale", "bias"], ["Y"]),
  LRN: R(["X"], ["Y"]),
  Dropout: R(["data", "ratio", "training_mode"], ["output", "mask"]),
  LSTM: R(["X", "W", "R", "B", "sequence_lens", "initial_h", "initial_c", "P"], ["Y", "Y_h", "Y_c"]),
  GRU: R(["X", "W", "R", "B", "sequence_lens", "initial_h"], ["Y", "Y_h"]),
  RNN: R(["X", "W", "R", "B", "sequence_lens", "initial_h"], ["Y", "Y_h"]),
  NonMaxSuppression: R(["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], ["selected_indices"]),
  RoiAlign: R(["X", "rois", "batch_indices"], ["Y"]),
  QuantizeLinear: R(["x", "y_scale", "y_zero_point"], ["y"]),
  DequantizeLinear: R(["x", "x_scale", "x_zero_point"], ["y"]),
  DynamicQuantizeLinear: R(["x"], ["y", "y_scale", "y_zero_point"]),
  Cast: R(["input"], ["output"]),
  CastLike: R(["input", "target_type"], ["output"]),
  Shape: R(["data"], ["shape"]),
  Size: R(["data"], ["size"]),
  Flatten: R(["input"], ["output"]),
  Softmax: R(["input"], ["output"]),
  If: R(["cond"], ["outputs..."]),
  Loop: R(["M", "cond", "v_initial..."], ["v_final_and_scan_outputs..."]),
  Scan: R(["initial_state_and_scan_inputs..."], ["final_state_and_scan_outputs..."]),
  Einsum: R(["Inputs..."], ["Output"]),
  // com.microsoft (onnxruntime contrib ops)
  "com.microsoft:Attention": R(["input", "weights", "bias", "mask_index", "past", "attention_bias", "past_sequence_length"], ["output", "present"]),
  "com.microsoft:MultiHeadAttention": R(["query", "key", "value", "bias", "key_padding_mask", "attention_bias", "past_key", "past_value"], ["output", "present_key", "present_value"]),
  "com.microsoft:SkipLayerNormalization": R(["input", "skip", "gamma", "beta", "bias"], ["output", "mean", "inv_std_var", "input_skip_bias_sum"]),
  "com.microsoft:SimplifiedLayerNormalization": R(["X", "scale"], ["Y", "inv_std_var"]),
  "com.microsoft:SkipSimplifiedLayerNormalization": R(["input", "skip", "gamma", "bias"], ["output", "mean", "inv_std_var", "input_skip_bias_sum"]),
  "com.microsoft:BiasGelu": R(["A", "B"], ["C"]),
  "com.microsoft:FastGelu": R(["X", "bias"], ["Y"]),
  "com.microsoft:QuickGelu": R(["X"], ["Y"]),
  "com.microsoft:EmbedLayerNormalization": R(["input_ids", "segment_ids", "word_embedding", "position_embedding", "segment_embedding", "gamma", "beta", "mask", "position_ids"], ["output", "mask_index", "embedding_sum"]),
  "com.microsoft:MatMulNBits": R(["A", "B", "scales", "zero_points", "g_idx", "bias"], ["Y"]),
  "com.microsoft:GroupQueryAttention": R(["query", "key", "value", "past_key", "past_value", "seqlens_k", "total_sequence_length", "cos_cache", "sin_cache"], ["output", "present_key", "present_value"]),
  "com.microsoft:RotaryEmbedding": R(["input", "position_ids", "cos_cache", "sin_cache"], ["output"]),
  "com.microsoft:QLinearAdd": R(["A", "A_scale", "A_zero_point", "B", "B_scale", "B_zero_point", "C_scale", "C_zero_point"], ["C"]),
  "com.microsoft:QLinearMul": R(["A", "A_scale", "A_zero_point", "B", "B_scale", "B_zero_point", "C_scale", "C_zero_point"], ["C"]),
  "com.microsoft:QLinearSigmoid": R(["X", "X_scale", "X_zero_point", "Y_scale", "Y_zero_point"], ["Y"]),
  "com.microsoft:QLinearAveragePool": R(["X", "x_scale", "x_zero_point", "y_scale", "y_zero_point"], ["Y"]),
  "com.microsoft:QLinearGlobalAveragePool": R(["X", "x_scale", "x_zero_point", "y_scale", "y_zero_point"], ["Y"]),
  "com.microsoft:Gelu": R(["X"], ["Y"]),
};

/** look up static roles for (domain, op) */
export function staticRoles(domain: string, op: string): Roles | null {
  if (domain && domain !== "ai.onnx") return ROLES[`${domain}:${op}`] ?? null;
  return ROLES[op] ?? null;
}

/**
 * Name the i-th (0-based) of n items given formal names. Variadic names ("x...") and schema-variadic
 * parameters expand to "x[k]". Items past the formals are "<prefix><i>".
 */
export function nameAt(formals: readonly (readonly [string, boolean])[] | null, i: number, total: number, prefix: string): string {
  if (!formals) return `${prefix}${i}`;
  // Walk formals; a variadic formal (flag true) absorbs everything not claimed by the formals after it.
  let pos = 0;
  for (let f = 0; f < formals.length; f++) {
    const [name, variadic] = formals[f]!;
    if (variadic) {
      const after = formals.length - f - 1;
      const span = Math.max(1, total - pos - after);
      if (i < pos + span) return `${name}[${i - pos}]`;
      pos += span;
    } else {
      if (i === pos) return name;
      pos++;
    }
  }
  return `${prefix}${i}`;
}

/** convert a static Roles name list into [name, variadic] tuples */
export function toFormals(names: string[] | undefined): [string, boolean][] | null {
  if (!names) return null;
  return names.map((n) => (n.endsWith("...") ? [n.slice(0, -3), true] : [n, false]));
}
