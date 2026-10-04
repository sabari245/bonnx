/** Getting model bytes into the app: file picker, drag-drop (files + folders), URL, python embed hook. */

export interface LoadedInput {
  name: string;
  data: ArrayBuffer | string;
  external: Record<string, ArrayBuffer>;
}

export type Progress = (p: { stage: string; frac: number }) => void;

const TEXT_EXT = /\.(prototxt|pbtxt|txt|json)$/i;
const MODEL_EXT = /\.(onnx|ort|pb|prototxt|pbtxt|json|txt|onnx\.txt)$/i;

declare global {
  interface Window {
    onnxvizDesktop?: {
      ready(): Promise<string | null>;
      readModel(path: string): Promise<{ name: string; data: Uint8Array; external: Record<string, Uint8Array> }>;
      pickModel(): Promise<string | null>;
      onOpenPath(cb: (path: string) => void): void;
    };
    __ONNXVIZ_EMBED__?: { name: string; b64: string };
  }
}

async function readBuffer(file: Blob): Promise<ArrayBuffer> {
  return file.arrayBuffer();
}

async function toInput(name: string, buf: ArrayBuffer): Promise<{ name: string; data: ArrayBuffer | string }> {
  if (TEXT_EXT.test(name)) return { name, data: new TextDecoder().decode(buf) };
  // binary protobuf never starts with '{' or typical text; text prototxt starts with ascii identifier. Let the worker detect.
  return { name, data: buf };
}

/** Pick the model among several files (largest .onnx first), the rest become external data candidates. */
export async function loadFromFiles(files: { path: string; file: File }[], onProgress?: Progress): Promise<LoadedInput> {
  if (!files.length) throw new Error("No files");
  const rank = (f: { path: string; file: File }) => (/\.onnx$/i.test(f.path) ? 0 : MODEL_EXT.test(f.path) ? 1 : 2);
  const sorted = files.slice().sort((a, b) => rank(a) - rank(b) || b.file.size - a.file.size);
  const main = sorted[0]!;
  onProgress?.({ stage: `Reading ${main.file.name}`, frac: 0.05 });
  const buf = await readBuffer(main.file);
  const { name, data } = await toInput(main.file.name, buf);
  const external: Record<string, ArrayBuffer> = {};
  const mainDir = main.path.includes("/") ? main.path.slice(0, main.path.lastIndexOf("/") + 1) : "";
  let i = 0;
  for (const f of sorted.slice(1)) {
    if (MODEL_EXT.test(f.path) && !/\.(pb|bin|data|weights)$/i.test(f.path) && /\.onnx$/i.test(f.path)) continue; // other models
    if (f.file.size > 4 * 1024 ** 3) continue;
    onProgress?.({ stage: `Reading ${f.file.name}`, frac: 0.05 + 0.2 * (++i / Math.max(1, sorted.length - 1)) });
    const rel = f.path.startsWith(mainDir) ? f.path.slice(mainDir.length) : f.path;
    external[rel] = await readBuffer(f.file);
  }
  return { name, data, external };
}

/* ───────── drag & drop with directory support ───────── */

async function walk(entry: FileSystemEntry, base: string, out: { path: string; file: File }[]): Promise<void> {
  if (entry.isFile) {
    const file = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
    out.push({ path: base + entry.name, file });
  } else if (entry.isDirectory) {
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    for (;;) {
      const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
      if (!batch.length) break;
      for (const e of batch) await walk(e, base + entry.name + "/", out);
    }
  }
}

export async function filesFromDrop(dt: DataTransfer): Promise<{ path: string; file: File }[]> {
  const out: { path: string; file: File }[] = [];
  const entries = [...dt.items].map((i) => (i.kind === "file" ? i.webkitGetAsEntry?.() : null)).filter(Boolean) as FileSystemEntry[];
  if (entries.length) {
    for (const e of entries) await walk(e, "", out);
    // strip the dropped folder prefix so external paths are relative to the model
    return out;
  }
  return [...dt.files].map((f) => ({ path: f.name, file: f }));
}

export function pickFiles(opts: { directory?: boolean } = {}): Promise<{ path: string; file: File }[]> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    if (opts.directory) input.setAttribute("webkitdirectory", "");
    else input.accept = ".onnx,.ort,.pb,.prototxt,.pbtxt,.json,.txt,.bin,.data,.weights";
    input.onchange = () => resolve([...(input.files ?? [])].map((f) => ({ path: (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name, file: f })));
    input.oncancel = () => resolve([]);
    input.click();
  });
}

/** Install page-wide drag/drop. `onActive` toggles the overlay. */
export function installDropZone(onFiles: (f: { path: string; file: File }[]) => void, onActive: (active: boolean) => void): () => void {
  let depth = 0;
  const hasFiles = (e: DragEvent) => !!e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  const enter = (e: DragEvent) => { if (!hasFiles(e)) return; e.preventDefault(); if (++depth === 1) onActive(true); };
  const over = (e: DragEvent) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer!.dropEffect = "copy"; };
  const leave = (e: DragEvent) => { if (!hasFiles(e)) return; if (--depth <= 0) { depth = 0; onActive(false); } };
  const drop = async (e: DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); depth = 0; onActive(false);
    onFiles(await filesFromDrop(e.dataTransfer!));
  };
  addEventListener("dragenter", enter); addEventListener("dragover", over); addEventListener("dragleave", leave); addEventListener("drop", drop);
  return () => { removeEventListener("dragenter", enter); removeEventListener("dragover", over); removeEventListener("dragleave", leave); removeEventListener("drop", drop); };
}

/* ───────── URL ───────── */

/** Fetch with streaming progress; works with huge files as long as they fit in an ArrayBuffer. */
export async function loadFromUrl(url: string, onProgress?: Progress, externalManifest?: string): Promise<LoadedInput> {
  const abs = new URL(url, location.href);
  const name = decodeURIComponent(abs.pathname.split("/").pop() || "model.onnx");
  onProgress?.({ stage: `Downloading ${name}`, frac: 0 });
  const res = await fetch(abs);
  if (!res.ok) throw new Error(`Could not fetch ${abs} (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  let buf: ArrayBuffer;
  if (res.body && total) {
    const out = new Uint8Array(total);
    const reader = res.body.getReader();
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (got + value.length > out.length) throw new Error("Server sent more bytes than Content-Length");
      out.set(value, got);
      got += value.length;
      onProgress?.({ stage: `Downloading ${name}`, frac: (got / total) * 0.9 });
    }
    buf = got === total ? out.buffer : out.buffer.slice(0, got);
  } else buf = await res.arrayBuffer();
  const { data } = await toInput(name, buf);
  const external: Record<string, ArrayBuffer> = {};
  if (externalManifest) {
    try {
      const man = (await (await fetch(new URL(externalManifest, location.href))).json()) as { files?: string[] } | string[];
      const list = Array.isArray(man) ? man : man.files ?? [];
      const dir = abs.pathname.slice(0, abs.pathname.lastIndexOf("/") + 1);
      for (const rel of list) {
        if (rel === name) continue;
        const r = await fetch(new URL(dir + rel.split("/").map(encodeURIComponent).join("/"), abs));
        if (r.ok) external[rel] = await r.arrayBuffer();
      }
    } catch { /* external data is optional */ }
  }
  return { name, data, external };
}

/* ───────── embedded model (python `onnxviz --out x.html`) ───────── */

export function loadEmbedded(): LoadedInput | null {
  const e = window.__ONNXVIZ_EMBED__;
  if (!e) return null;
  const bin = atob(e.b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  delete window.__ONNXVIZ_EMBED__; // let the big string be collected
  const data: ArrayBuffer | string = TEXT_EXT.test(e.name) ? new TextDecoder().decode(bytes) : bytes.buffer;
  return { name: e.name, data, external: {} };
}

/* ───────── Electron desktop bridge ───────── */

const toBuf = (u: Uint8Array): ArrayBuffer => (u.byteOffset === 0 && u.byteLength === u.buffer.byteLength ? (u.buffer as ArrayBuffer) : (u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer));

export async function loadFromDesktopPath(path: string): Promise<LoadedInput> {
  const m = await window.onnxvizDesktop!.readModel(path);
  const { data } = await toInput(m.name, toBuf(m.data));
  const external: Record<string, ArrayBuffer> = {};
  for (const [k, v] of Object.entries(m.external)) external[k] = toBuf(v);
  return { name: m.name, data, external };
}
