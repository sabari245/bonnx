/**
 * Shared contract between the model worker (src/onnx, src/workers), the renderer (src/render)
 * and the app shell (src/app). Everything here is structured-cloneable plain data:
 * the main thread NEVER holds tensor payloads; it asks the worker for slices/stats via ModelApi.
 */

/* ───────────────────────── types of values ───────────────────────── */

export type Dim = number | string; // number = fixed, string = symbolic dim_param, "?" = unknown

export type ValueType =
  | { kind: "tensor"; dtype: string; shape: Dim[] | null } // shape null = rank unknown
  | { kind: "sparse"; dtype: string; shape: Dim[] | null }
  | { kind: "sequence"; elem: ValueType | null }
  | { kind: "map"; key: string; value: ValueType | null }
  | { kind: "optional"; elem: ValueType | null }
  | { kind: "opaque"; domain: string; name: string }
  | { kind: "unknown" };

/** dtype strings: float32 float64 float16 bfloat16 int8 int16 int32 int64 uint8 uint16 uint32 uint64
 *  bool string complex64 complex128 float8e4m3fn float8e4m3fnuz float8e5m2 float8e5m2fnuz int4 uint4
 *  float4e2m1 undefined */
export type DTypeName = string;

/* ───────────────────────── tensors (constants) ───────────────────────── */

export interface TensorSummary {
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
  absmax: number | null;
  zeros: number; // fraction 0..1
  nan: number;
  inf: number;
}

export type TensorSource = "initializer" | "constant" | "attribute" | "sparse_initializer" | "sparse_attribute";

export interface TensorInfo {
  id: number; // index in ModelView.tensors, used for all RPC
  name: string; // initializer name, Constant output name, or "<node>.<attr>" for attribute tensors
  dtype: DTypeName;
  dims: number[];
  n: number; // element count (dense)
  bytes: number; // byte size of the dense payload
  source: TensorSource;
  graph: number; // owning graph index
  node?: number; // owning node index (for Constant / attribute tensors)
  docString?: string;
  /** external data reference; `available` is false until the user provides the file */
  external?: { location: string; offset: number; length: number | null } | null;
  available: boolean;
  /** present for numeric tensors when payload is available */
  summary?: TensorSummary;
  /** sparse: stored non-zero count and index layout */
  sparse?: { nnz: number; valuesTid: number; indicesTid: number };
}

export interface TensorStats extends TensorSummary {
  median: number | null;
  absmean: number | null;
  p1: number | null;
  p99: number | null;
  l2: number | null;
  unique: number;
  /** histogram over [histMin, histMax] of finite values (numeric tensors) */
  hist: number[] | null;
  histMin: number | null;
  histMax: number | null;
  /** |x| max for each index of dim 0 (rank >= 2, dim0 > 1) */
  channelAbsMax: number[] | null;
  /** true if stats came from a sample (huge tensors) */
  sampled: boolean;
}

/** Row-major slice of a tensor. `values` is Float64Array for numeric/bool, string[] for string tensors,
 *  complex tensors are interleaved (re, im). `lossy` = int64/uint64 values beyond 2^53 were rounded. */
export interface TensorSliceResult {
  dtype: DTypeName;
  dims: number[]; // full tensor dims
  offset: number[]; // slice start per dim
  size: number[]; // slice extent per dim
  values: Float64Array | string[];
  lossy: boolean;
  truncated: boolean; // request exceeded maxElems and was clipped
}

/* ───────────────────────── attributes ───────────────────────── */

export type AttrValue =
  | { t: "int"; v: number | string } // string if beyond 2^53 (decimal)
  | { t: "float"; v: number }
  | { t: "string"; v: string; bytes?: boolean } // bytes: not valid utf-8, v is hex
  | { t: "ints"; v: (number | string)[] }
  | { t: "floats"; v: number[] }
  | { t: "strings"; v: string[] }
  | { t: "tensor"; tid: number }
  | { t: "tensors"; tids: number[] }
  | { t: "sparse_tensor"; tid: number }
  | { t: "graph"; graph: number } // index into ModelView.graphs
  | { t: "graphs"; graphs: number[] }
  | { t: "type"; type: ValueType }
  | { t: "types"; types: ValueType[] }
  | { t: "ref"; ref: string } // ref_attr_name inside a function body
  | { t: "undefined" };

export interface AttrView {
  name: string;
  value: AttrValue;
  docString?: string;
}

/* ───────────────────────── graph ───────────────────────── */

export interface NodeInput {
  /** formal parameter name from the op schema ("X", "W", "B", …) or "in<i>" when unknown */
  param: string;
  /** tensor/value name ("" never appears; optional-absent inputs are skipped) */
  name: string;
  /** position in node.input */
  index: number;
  type: ValueType | null;
  /** set when this value is a constant of this graph (initializer / Constant node / outer-scope initializer) */
  tid: number | null;
}

export interface NodeOutput {
  param: string;
  name: string;
  index: number;
  type: ValueType | null;
}

export interface NodeView {
  id: number; // index in GraphView.nodes
  name: string; // node name or synthesized "<Op>_<i>"
  op: string;
  domain: string; // "" = ai.onnx
  opset: number | null; // opset version applying to this node's domain
  docString?: string;
  inputs: NodeInput[];
  outputs: NodeOutput[];
  attrs: AttrView[];
  /** index of the local FunctionProto graph this node calls, if any */
  fn: number | null;
  /** subgraph indices referenced by graph/graphs attributes, in attribute order */
  subgraphs: { attr: string; graph: number }[];
  /** total float parameters in constant inputs with >1 element */
  params: number;
  /** largest |w| across float constant inputs (null if none) */
  wmax: number | null;
  /** index of the node in the original GraphProto.node (Constant nodes are folded away, so ids ≠ source index) */
  src: number;
  metadata?: Record<string, string>;
}

export interface ValueInfo {
  name: string;
  type: ValueType | null;
  docString?: string;
  /** node id producing it, -1 = graph input / initializer / outer scope */
  producer: number;
  consumers: number[];
  /** constant tensor id when the value is an initializer/Constant */
  tid: number | null;
  /** true if the name is defined in an enclosing graph (subgraph captures) */
  outer: boolean;
}

export interface GraphIO {
  name: string;
  type: ValueType | null;
  docString?: string;
  /** graph inputs that also have an initializer (defaults) carry the tensor id */
  tid: number | null;
}

export interface GraphView {
  id: number;
  name: string;
  kind: "main" | "subgraph" | "function";
  /** where the graph is referenced from (subgraphs) or null */
  parent: { graph: number; node: number; attr: string } | null;
  /** function graphs: identity of the function */
  fnDomain?: string;
  fnName?: string;
  fnOverload?: string;
  fnAttrs?: string[];
  docString?: string;
  nodes: NodeView[];
  inputs: GraphIO[];
  outputs: GraphIO[];
  /** all named values appearing in this graph (not outer ones) keyed by name */
  values: Record<string, ValueInfo>;
  /** constants that belong to this graph (tensor ids) */
  initializers: number[];
  /** nesting depth for UI breadcrumbs (main = 0) */
  depth: number;
}

/* ───────────────────────── model ───────────────────────── */

export interface ModelMeta {
  file: string;
  fileBytes: number;
  format: "onnx" | "onnx-text" | "onnx-json" | "ort" | string;
  ir: number | null;
  opsets: Record<string, number>; // domain ("ai.onnx" for "") → version
  producer: string; // "name version"
  producerName: string;
  producerVersion: string;
  domain: string;
  modelVersion: number | string | null;
  docString: string;
  graphName: string;
  props: Record<string, string>; // metadata_props
  ops: [string, number][]; // op type counts across all graphs, descending
  params: number; // float parameters across all initializers
  initializers: number;
  weightBytes: Record<string, number>; // dtype → bytes
  nodeCount: number; // all graphs
  functions: number;
  /** training info, quantization annotation presence etc. */
  extras: string[];
}

export interface ModelView {
  meta: ModelMeta;
  graphs: GraphView[]; // graphs[0] is the main graph
  tensors: TensorInfo[];
  warnings: string[];
}

/* ───────────────────────── worker RPC ───────────────────────── */

export interface OpenRequest {
  name: string;
  /** binary protobuf (ArrayBuffer) or text (prototxt/json string); format is auto-detected */
  data: ArrayBuffer | string;
  /** extra files by relative path (external tensor data) */
  external?: Record<string, ArrayBuffer>;
}

export interface SliceRequest {
  id: number;
  /** per-dim start / extent; omitted = whole tensor. Clipped to maxElems (default 4_000_000) */
  offset?: number[];
  size?: number[];
  maxElems?: number;
}

export interface HistogramRequest {
  id: number;
  bins: number;
  min?: number; // default tensor min
  max?: number; // default tensor max
  /** ignore values outside [min,max] */
  log?: boolean;
}

export interface HistogramResult {
  counts: number[];
  min: number;
  max: number;
  below: number;
  above: number;
}

export interface ModelApi {
  open(req: OpenRequest, onProgress?: (p: { stage: string; frac: number }) => void): Promise<ModelView>;
  /** provide an external data file after open() and re-resolve tensors */
  provideExternal(path: string, data: ArrayBuffer): Promise<TensorInfo[]>;
  tensorStats(id: number): Promise<TensorStats>;
  tensorSlice(req: SliceRequest): Promise<TensorSliceResult>;
  tensorHistogram(req: HistogramRequest): Promise<HistogramResult>;
  /** operator documentation lookups (lazy loaded metadata) */
  opSchema(domain: string, op: string, opset: number | null): Promise<OpSchema | null>;
  dispose(): Promise<void>;
}

export interface OpSchema {
  domain: string;
  name: string;
  sinceVersion: number;
  doc: string;
  inputs: { name: string; description: string; types: string[]; option: "single" | "optional" | "variadic"; typeStr: string }[];
  outputs: { name: string; description: string; types: string[]; option: "single" | "optional" | "variadic"; typeStr: string }[];
  attributes: { name: string; description: string; type: string; required: boolean; default?: string | number | (string | number)[] }[];
  typeConstraints: { name: string; description: string; types: string[] }[];
  /** inclusive op-set versions this operator definition spans [since, nextSince) */
  versions?: number[];
  category?: string;
}

/* ───────────────────────── scene + layout (main thread) ───────────────────────── */

export type SceneNodeKind = "op" | "input" | "output" | "const" | "fn" | "ghost";

export interface SceneNode {
  id: number;
  kind: SceneNodeKind;
  /** NodeView.id for kind op/fn; GraphIO index for input/output; tensor id for const */
  ref: number;
  label: string; // op type, io name, or const name
  /** layout results (world coords, top-left) */
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface SceneEdge {
  id: number;
  s: number; // SceneNode.id
  t: number;
  name: string; // value name
  type: ValueType | null;
  /** bezier path: x0,y0 followed by triplets (c1,c2,p) ; filled by layout */
  p: number[];
  /** flat list of points for polyline fallback / SVG export */
  /** true for edges that carry a constant */
  isConst: boolean;
}

export interface LayoutOptions {
  dir: "TB" | "LR";
  rankSep: number;
  nodeSep: number;
}

export interface LayoutRequest {
  nodes: { id: number; w: number; h: number }[];
  edges: { id: number; s: number; t: number }[];
  options: LayoutOptions;
}

export interface LayoutResult {
  width: number;
  height: number;
  nodes: { id: number; x: number; y: number }[];
  edges: { id: number; p: number[] }[];
}
