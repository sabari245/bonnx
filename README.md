# onnxviz

An ONNX model viewer that runs entirely in the browser. Open a model by drag and drop, file picker or URL; it is
parsed in a Web Worker and drawn on a canvas, with per-weight statistics, subgraph navigation and SVG/PNG export.
No server and no Python are needed to *view* a model. A small Python launcher is included for the
`onnxviz model.onnx` workflow and for producing one self-contained HTML file.

> Status of this README: the web app (`src/`) is built by several people/tracks; the Python launcher was tested against
> stub builds. Sections that describe the app reflect the source tree at the time of writing and have not been
> verified in a running build — run `make test` and open the app to confirm.

## Install (Linux x86_64)

```bash
curl -fsSL https://raw.githubusercontent.com/sabari245/onnxviz/main/setup.sh | bash
```

The script detects your system and installs the [latest release](https://github.com/sabari245/onnxviz/releases/latest):
a `.deb` via `apt` on Debian/Ubuntu-family distros (asks for sudo), otherwise a user-local AppImage in `~/.local`
(no root; adds a launcher entry and the `onnxviz` command). Downloads are checked against the release's SHA-256 sums.

Then run `onnxviz model.onnx`, or open **ONNX Viz** from your app launcher.

```bash
# options: pin a version, force the no-root AppImage install, or remove it again
curl -fsSL https://raw.githubusercontent.com/sabari245/onnxviz/main/setup.sh | bash -s -- --version v0.1.1
curl -fsSL https://raw.githubusercontent.com/sabari245/onnxviz/main/setup.sh | bash -s -- --appimage
curl -fsSL https://raw.githubusercontent.com/sabari245/onnxviz/main/setup.sh | bash -s -- --uninstall
```

Prefer to do it by hand? Grab the `.deb`, `.AppImage`, `.tar.gz` or the single-file `standalone.html` from the
[Releases page](https://github.com/sabari245/onnxviz/releases).

## Quickstart

### Web app

```bash
make install          # bun install   (if it hangs, see Troubleshooting)
make dev              # http://localhost:5173, then drop a .onnx file on the page
make build            # production build -> dist/
```

The page also takes `?url=<model url>` (and `&external=<manifest url>` for external-data files).

### From Python (like `netron model.onnx`)

```bash
make build                                   # once; the launcher serves dist/
PYTHONPATH=python python -m onnxviz model.onnx          # serve on http://127.0.0.1:8080 and open the browser
PYTHONPATH=python python -m onnxviz models/ --port 9000 # a directory: every .onnx / .onnx.prototxt in it
```

After `make package && pip install ./python` the same thing is just `onnxviz model.onnx` (the build is bundled
into the package; the package itself has no dependencies).

| Option | |
|---|---|
| `--port N` | default 8080; if busy, the next free port is used |
| `--host ADDR` | default `127.0.0.1`; `0.0.0.0` shares it on your network (a warning is printed) |
| `--no-browser` | do not open a window |
| `--dist DIR` | web build to serve (default: bundled `web/`, then `<repo>/dist`) |
| `-o/--out FILE`, `--embed` | write a single HTML file instead of serving (below) |
| `--template FILE`, `--force`, `-v` | single-file build to use; embed models over 200 MB; log requests |

The model is streamed from disk with HTTP `Range` support (it is never read into memory by the server). Files next to
the model and referenced as external tensor data are served too: the page fetches `/__model__/manifest.json`, which lists
them as `{"files": ["weights.bin", ...]}`. Only the model's own directory is exposed; `..` escapes, symlinks leaving it and
dotfiles are refused, no CORS headers are sent, and on loopback addresses requests with a foreign `Host` header are
rejected (DNS-rebinding protection).

### One self-contained HTML file

```bash
make single                                  # once: dist-single/index.html with everything inlined
make viz MODEL=model.onnx OUT=model.html     # == python -m onnxviz model.onnx --out model.html
```

The file opens offline by double-click and can be mailed or attached to a ticket. Mechanism: the model is base64-encoded
(streamed, in chunks) into a classic `<script>window.__ONNXVIZ_EMBED__={"name":…,"size":…,"b64":"…"};</script>` placed
right after `<head>`; the app finds it on startup (`loadEmbedded()` in `src/app/loader.ts`). Limits: models above 200 MB
are refused unless `--force`, and above ~380 MB always (browsers cap string length at ~512M characters). External-data
files are not embedded — serve those models instead.

### Python API

```python
import onnxviz
onnxviz.show("model.onnx")                  # path, bytes, or an onnx.ModelProto (serialized in memory)
srv = onnxviz.show(model, block=False)      # background thread; srv.page_url(), srv.stop()
onnxviz.save_html(model, "model.html")      # single-file viewer
onnxviz.display(model)                      # Jupyter: IFrame on a free localhost port (needs IPython)
```

`onnx` is imported only if you pass it a `ModelProto`.

## Features

- **Open anything at runtime**: drag and drop (files or a whole folder, so external-data files come along), file
  dialog (Ctrl+O), URL, `?url=`, embedded HTML. Binary protobuf, ONNX text (`.onnx.prototxt`) and ONNX JSON.
- **Whole-model fidelity**: nested subgraphs (If/Loop/Scan/any graph attribute) you can click into and back out of,
  model-local functions, sparse initializers, sequence/map/optional/sparse types, tensor- and graph-valued attributes,
  `Constant` nodes folded into constants, outer-scope captures, metadata, doc strings.
- **Canvas renderer** with level-of-detail drawing, culling, minimap, hover/selection edge highlighting, arrow-key
  navigation, TB/LR layout (dagre, in a worker), edge labels, color by operator / parameters / |w| max.
- **Weights**: summary statistics for every constant (min/max/mean/std/percentiles/zeros/NaN/Inf/unique), histogram with
  custom range, linear/log scale, data table with CSV export, per-channel |max|, sparse tensors dense or as stored values,
  all numeric dtypes including float16/bfloat16/float8/int4. Tensors are read in the worker; the page never holds the payload.
- **Operator documentation** from the ONNX schema, matched to the model's opset.
- **Export** the graph as vector SVG or PNG (1×–4×), or the neighborhood of the selection.
- shadcn-style light/dark design system, command palette search, keyboard shortcuts.

## Keyboard shortcuts

(From `SHORTCUTS` in `src/app/app.ts`; press `?` in the app for the live list.)

| Key | Action |
|---|---|
| `/` or `Ctrl+K` | Search nodes, tensors and operators |
| `Ctrl+O` | Open a model |
| `F` | Fit graph to window |
| `+` / `−` | Zoom in / out |
| Arrow keys | Move selection along the graph |
| `Enter` | Open the selected node's subgraph or function |
| `Backspace` / `Alt+←` | Back to the previous graph |
| `Esc` | Deselect / close |
| `M` | Model summary |
| `I` | Toggle inspector panel |
| `D` | Cycle color mode |
| `L` | Toggle layout direction |
| `N` / `A` | Toggle node names / attributes |
| `C` | Copy selected node name |
| `E` | Export SVG |
| `T` | Cycle theme |
| `?` | Shortcut help |

## Architecture

```
index.html, src/main.ts      entry
src/app/                     app shell: loader (drop/URL/embed), inspector, search, tensor viewer (src/app/tensor)
src/onnx/                    protobuf reader, ONNX/text/JSON decoding, graph builder, tensor math, op schema metadata
src/workers/                 model.worker (owns the model bytes; answers stats/slice RPC), layout.worker (dagre)
src/render/                  scene building, canvas viewer, minimap, SVG/PNG export
src/ui/, src/styles/         shadcn-style vanilla components on Tailwind v4, design tokens (light/dark)
python/onnxviz/              launcher: static + Range server, single-file embedder, Python API
```

The main thread holds only a light `ModelView` (src/onnx/types.ts); tensor data stays in the worker and is fetched as
slices, statistics or histograms on demand. No React: the UI is plain TypeScript (`h()` in `src/lib/dom.ts`) with Tailwind classes
copied from shadcn/ui.

## Development

```bash
make dev            # vite dev server
make test           # vitest (parser) + python unittest
make test-py        # python only:  PYTHONPATH=python python -m unittest discover -s python/tests -t python/tests
make build / single / package / clean
bun run typecheck
```

## Netron feature parity

| | |
|---|---|
| Open at runtime (drag-drop, dialog, URL, local server via `onnxviz`) | ✅ |
| Subgraphs (If/Loop/Scan) and local functions, click-through | ✅ |
| Tensor-valued / graph-valued attributes, sparse initializers, sequence/map types | ✅ |
| Weight views: statistics, histogram, data table | ✅ |
| Weight view as image (conv kernels) | ⏳ planned in the tensor viewer, not in the source tree yet |
| Export SVG / PNG | ✅ |
| Edge labels, hover highlight, arrow-key navigation, copy name, context menu | ✅ |
| ONNX text format (`.onnx.prototxt`) and JSON | ✅ |
| Local server / Python package / Jupyter | ✅ (`python/`) |
| Single-file shareable viewer | ✅ (`--out`) |
| Per-weight statistics, NaN/Inf audit | ✅ (not in Netron) |
| Other formats (TFLite, PyTorch, Core ML, OpenVINO, Keras, GGUF, …) | ❌ ONNX only |
| ORT (FlatBuffers) models | ❌ rejected with a message |
| Desktop app (Electron/Tauri), PWA | ❌ use the web app or `onnxviz` |
| Per-layer accuracy comparison (old `build_viewer`) | ❌ not ported |

## Limitations

- `com.microsoft` (and other non-ONNX-domain) operators get parameter names from a static table but no documentation.
- Statistics of tensors above 20M elements are computed on a stride sample and flagged as such; `int64` values beyond 2^53 are flagged as rounded.
- Shape inference is partial and only used when `value_info` is missing.
- Models must fit in an `ArrayBuffer` of the browser (about 2 GB); use external data for larger ones.
- Python `show()` needs a built web app (`make build` or `make package`).

## Troubleshooting

- **`bun install` hangs.** bun resolves `registry.npmjs.org` to IPv6 first and does not fall back to IPv4, so on a host
  with broken IPv6 it never finishes (npm/pnpm are fine). Use `make install-mirror`, which runs
  `bun install --registry=https://registry.npmmirror.com` and rewrites the lockfile URLs back to `registry.npmjs.org`
  (`sed -i 's#https://registry.npmmirror.com/#https://registry.npmjs.org/#g' bun.lock`).
- **graphviz is no longer needed.** Layout is done with dagre in the browser; the old `dot` / micromamba setup can be dropped.
- **`onnxviz: error: the web app is not built`** — run `make build`, pass `--dist DIR`, or install a package made with `make package`.
- **`the single-file web build is missing`** — run `make single`.
- **Port in use** — the launcher picks the next free port and prints it.

## Desktop app (Linux, Electron)

```bash
bun run desktop        # build + run in Electron
bun run desktop:dist   # builds release/onnxviz-<ver>-x86_64.AppImage, .deb and .tar.gz
```

Install: `chmod +x onnxviz-*.AppImage && ./onnxviz-*.AppImage`, or `sudo apt install ./onnxviz-*.deb`
(the deb registers `.onnx` file association; `onnxviz model.onnx` opens a model, and a second
invocation reuses the running window). External-data files next to the model are loaded automatically.
