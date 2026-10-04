/** Parse + build entry point shared by the worker and tests. */
import { decodeModel } from "./decode";
import { buildModel } from "./build";
import { ProtoError, Warnings } from "./proto";
import { detectFormat, parseJsonModel, parseTextModel } from "./text";
import { loadMetadata, type Metadata } from "./schema";
import { TensorStore } from "./tensor";
import type { ModelView, OpenRequest } from "./types";

export interface Loaded {
  view: ModelView;
  store: TensorStore;
}

export type Progress = (p: { stage: string; frac: number }) => void;

export async function loadModel(req: OpenRequest, onProgress?: Progress): Promise<Loaded> {
  const warnings = new Warnings();
  const format = detectFormat(req.name, req.data);
  const report = (stage: string, frac: number) => onProgress?.({ stage, frac });
  if (format === "ort") throw new ProtoError("ORT format (.ort FlatBuffers) is not supported; convert it with onnxruntime to .onnx");

  report("parse", 0);
  const metaP: Promise<Metadata | null> = loadMetadata().catch(() => null);
  let size: number;
  let raw;
  if (format === "onnx") {
    const u8 = req.data instanceof ArrayBuffer ? new Uint8Array(req.data) : new TextEncoder().encode(req.data);
    size = u8.length;
    raw = decodeModel(u8, warnings);
  } else {
    const text = typeof req.data === "string" ? req.data : new TextDecoder().decode(req.data);
    size = typeof req.data === "string" ? new TextEncoder().encode(req.data).length : req.data.byteLength;
    raw = format === "onnx-json" ? parseJsonModel(text, warnings) : parseTextModel(text, warnings);
  }
  report("parse", 1);

  const store = new TensorStore();
  for (const [path, buf] of Object.entries(req.external ?? {})) store.setExternal(path, new Uint8Array(buf));
  const metadata = await metaP;
  const view = buildModel(raw, {
    fileName: req.name, fileBytes: size, format, warnings, store, metadata,
    onProgress: (stage, frac) => report(stage, frac),
  });
  return { view, store };
}
