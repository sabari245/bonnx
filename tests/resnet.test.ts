/// <reference types="node" />
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { loadModel } from "@/onnx/load";
import { decodeModel } from "@/onnx/decode";

const path = new URL("../models/resnet18-v2-7.onnx", import.meta.url);
const buf = () => readFileSync(path);
const ab = () => {
  const b = buf();
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

describe("resnet18-v2-7", () => {
  it("matches the numbers from the legacy tool", async () => {
    const { view } = await loadModel({ name: "resnet18-v2-7.onnx", data: ab() });
    const g = view.graphs[0]!;
    expect(view.graphs.length).toBe(1);
    expect(g.nodes.length).toBe(69);
    expect(view.tensors.length).toBe(99);
    expect(view.meta.params).toBe(11_695_796);
    expect(view.meta.ir).toBe(3);
    expect(view.meta.opsets["ai.onnx"]).toBe(7);
    expect(view.meta.format).toBe("onnx");
    expect(view.warnings).toEqual([]);
    // every constant has a payload + summary, nothing heavy in the view
    expect(view.tensors.every((t) => t.available && t.summary)).toBe(true);
    const conv = g.nodes.find((n) => n.op === "Conv")!;
    expect(conv.inputs.map((i) => i.param)).toEqual(["X", "W"]);
    expect(conv.params).toBeGreaterThan(0);
    expect(conv.wmax).toBeGreaterThan(0);
    expect(conv.outputs[0]!.type).toMatchObject({ kind: "tensor", dtype: "float32" });
  });

  it("decodes quickly", () => {
    const u8 = new Uint8Array(buf());
    decodeModel(u8); // warm
    const t = performance.now();
    for (let i = 0; i < 5; i++) decodeModel(u8);
    const per = (performance.now() - t) / 5;
    console.log(`decode: ${per.toFixed(2)} ms`);
    expect(per).toBeLessThan(100);
  });

  it("does not copy raw payloads", () => {
    const u8 = new Uint8Array(buf());
    const m = decodeModel(u8);
    const t = m.graph!.initializers.find((i) => (i.raw ?? i.floatData)!?.length > 1000)!;
    expect((t.raw ?? t.floatData)!.buffer).toBe(u8.buffer);
  });
});
