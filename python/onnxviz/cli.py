"""Command line: `onnxviz model.onnx` (serve) and `onnxviz model.onnx --out view.html` (embed)."""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from ._dist import MISSING_DIST, find_dist
from ._sources import DIR_EXTS, MODEL_EXTS, ModelSource, split_ext, unique_names
from .embed import EmbedError, write_embedded_html
from .server import VizServer


def _err(msg: str) -> int:
    print(f"onnxviz: error: {msg}", file=sys.stderr)
    return 2


def _is_model_name(name: str, exts: Sequence[str]) -> bool:
    low = name.lower()
    return any(low.endswith(e) for e in exts)


def collect_models(paths: Sequence[str]) -> list[ModelSource]:
    """Files are taken as given (extension checked); directories contribute their .onnx / .onnx.prototxt files."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            items = sorted(c for c in p.iterdir() if c.is_file() and not c.name.startswith(".") and _is_model_name(c.name, DIR_EXTS))
            if not items:
                raise ValueError(f"no .onnx or .onnx.prototxt files in directory {p}")
            found += items
        elif p.is_file():
            if not _is_model_name(p.name, MODEL_EXTS):
                raise ValueError(f"unsupported file type: {p.name} (expected {', '.join(MODEL_EXTS)})")
            found.append(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
    seen: set[Path] = set()
    out: list[ModelSource] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(ModelSource(p.name, path=rp))
    return unique_names(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="onnxviz", description="View ONNX models in the browser: serve on localhost, or write one self-contained HTML file.",
        epilog="examples:\n  onnxviz model.onnx\n  onnxviz models/ --port 9000 --no-browser\n  onnxviz model.onnx --out view.html",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", metavar="MODEL", help=".onnx / .onnx.prototxt / .json file(s) or directories; none = open the empty viewer")
    ap.add_argument("--port", type=int, default=8080, help="port to serve on (default 8080; the next free one is used if busy)")
    ap.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1; use 0.0.0.0 to share on your network)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    ap.add_argument("--dist", metavar="DIR", help="directory with the built web app (default: bundled web/, then <repo>/dist)")
    ap.add_argument("-o", "--out", metavar="FILE", help="write a single self-contained HTML file instead of serving")
    ap.add_argument("--embed", action="store_true", help="same as --out; the file defaults to <model>.html in the current directory")
    ap.add_argument("--template", metavar="FILE", help="single-file web build to embed into (default: bundled, then <repo>/dist-single/index.html)")
    ap.add_argument("--force", action="store_true", help="embed models above the 200 MB limit")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every HTTP request")
    ap.add_argument("--version", action="version", version=f"onnxviz {__version__}")
    return ap


def _embed(args: argparse.Namespace, sources: list[ModelSource]) -> int:
    if not sources:
        return _err("--out/--embed needs a model")
    if len(sources) > 1:
        return _err("--out/--embed embeds a single model; pass one file")
    src = sources[0]
    out = Path(args.out) if args.out else Path.cwd() / (split_ext(src.name)[0] + ".html")
    if src.path is not None and out.resolve() == src.path:
        return _err("the output file would overwrite the model")
    try:
        n = write_embedded_html(src, out, template=args.template, force=args.force)
    except (EmbedError, FileNotFoundError) as e:
        return _err(str(e))
    print(f"wrote {out} ({n / 1e6:.1f} MB) with {src.name} embedded")
    return 0


def _serve(args: argparse.Namespace, sources: list[ModelSource]) -> int:
    try:
        dist = find_dist(args.dist)
    except FileNotFoundError as e:
        return _err(str(e))
    if dist is None:
        return _err(MISSING_DIST)
    try:
        srv = VizServer(sources, dist, args.host, args.port, verbose=args.verbose)
    except OSError as e:
        return _err(f"cannot listen on {args.host}:{args.port}: {e}")
    url = srv.page_url()
    print(f"onnxviz {__version__}", flush=True)
    for s in sources:
        print(f"  model:  {s.path or s.name} ({s.size / 1e6:.1f} MB)", flush=True)
    if srv.port_changed:
        print(f"  port {args.port} is busy, using {srv.port}", flush=True)
    print(f"  viewer: {url}", flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("  warning: the model and any files next to it are reachable by other machines on this network", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        srv.stop()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sources = collect_models(args.paths)
    except (FileNotFoundError, ValueError) as e:
        return _err(str(e))
    return _embed(args, sources) if (args.out or args.embed) else _serve(args, sources)


if __name__ == "__main__":
    raise SystemExit(main())
