"""Local HTTP server: the built web app plus the model(s), streamed with Range support.

Routes
  /                         the web app (static files from the dist directory)
  /__model__/<name>         a model (file streamed from disk, or served from memory)
  /__model__/<rel/path>     any non-hidden file below a model's directory (external tensor data)
  /__model__/index.json     {"models": [{"name", "url", "size", "manifest"}]}
  /__model__/manifest.json  {"model", "detected", "files": ["weights.bin", ...]}  (?model=<name>)

No CORS headers are sent, hidden files/directories are never served, paths cannot leave their root, and when
bound to a loopback address requests whose Host header is not a loopback name are refused (DNS rebinding).
"""
from __future__ import annotations

import errno
import ipaddress
import json
import mimetypes
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

from ._sources import ModelSource, unique_names

MODEL_PREFIX = "/__model__/"
SCAN_LIMIT = 64 << 20  # models up to this size are scanned for the external-data names they reference
SCAN_MAX_CANDIDATES = 64
LIST_MAX = 256

_MIME = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json", ".map": "application/json", ".wasm": "application/wasm",
    ".svg": "image/svg+xml", ".woff2": "font/woff2", ".woff": "font/woff", ".ico": "image/x-icon", ".png": "image/png",
    ".webmanifest": "application/manifest+json", ".txt": "text/plain; charset=utf-8",
}


class RangeError(Exception):
    """Range header is satisfiable by no byte of the resource (-> 416)."""


def parse_range(header: str, size: int) -> Optional[tuple[int, int]]:
    """Return (start, end_inclusive), None to ignore the header (serve 200), or raise RangeError (416).

    Only a single byte range is honoured; multi-range and syntactically invalid headers are ignored, as RFC 9110 allows.
    """
    header = header.strip()
    if not header.lower().startswith("bytes=") or "," in header:
        return None
    spec = header[6:].strip()
    a, sep, b = spec.partition("-")
    if not sep:
        return None
    a, b = a.strip(), b.strip()
    try:
        if a == "":  # suffix: last n bytes
            n = int(b)
            if n < 0 or not b.isdigit():
                return None
            if n == 0 or size == 0:
                raise RangeError
            return max(0, size - n), size - 1
        if not a.isdigit() or (b and not b.isdigit()):
            return None
        start = int(a)
        end = int(b) if b else size - 1
    except ValueError:
        return None
    if b and end < start:
        return None
    if start >= size:
        raise RangeError
    return start, min(end, size - 1)


def safe_join(root: Path, rel: str) -> Optional[Path]:
    """Resolve `rel` (already URL-decoded, '/'-separated) below `root`; None if it escapes, is hidden or missing."""
    if "\0" in rel or "\\" in rel:
        return None
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." or p.startswith(".") for p in parts):
        return None
    if os.name == "nt" and any(":" in p for p in parts):
        return None
    try:
        real = root.joinpath(*parts).resolve(strict=True)
        root_real = root.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if real != root_real and root_real not in real.parents:
        return None  # e.g. a symlink pointing out of the root
    return real


def _list_candidates(root: Path, exclude: Optional[Path]) -> list[str]:
    """Regular, non-hidden files up to two levels below `root`, as '/'-separated relative paths."""
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".")) if depth < 2 else []
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            p = Path(dirpath, fn)
            if exclude is not None and p == exclude:
                continue
            if p.is_file():
                out.append(p.relative_to(root).as_posix())
            if len(out) >= 5000:
                return out
    return out


def find_referenced(path: Path, names: list[str], chunk: int = 8 << 20) -> set[str]:
    """Which of `names` (or, for nested paths, their basenames) occur as raw bytes in the file, streamed."""
    needles: dict[bytes, set[str]] = {}
    for n in names:
        for key in {n, n.rsplit("/", 1)[-1]}:
            if len(key) >= 3:
                needles.setdefault(key.encode("utf-8"), set()).add(n)
    found: set[str] = set()
    if not needles:
        return found
    keep = max(len(k) for k in needles) - 1
    tail = b""
    with open(path, "rb") as f:
        while needles:
            data = f.read(chunk)
            if not data:
                break
            buf = tail + data
            for k in [k for k in needles if k in buf]:
                found |= needles.pop(k)
            tail = buf[-keep:] if keep else b""
    return found


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _host_of(header: str) -> str:
    """Hostname from a Host header value ('example:80', '[::1]:80', '::1' is not valid HTTP but tolerated)."""
    header = header.strip().lower()
    if header.startswith("["):
        return header[1 : header.find("]")] if "]" in header else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64
    allow_reuse_address = os.name != "nt"  # on Windows SO_REUSEADDR lets two sockets share a port

    def handle_error(self, request, client_address):  # clients hanging up mid-stream are routine
        import sys

        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class _Server6(_Server):
    address_family = socket.AF_INET6


class VizServer:
    def __init__(self, sources: list[ModelSource], dist: Optional[Path] = None, host: str = "127.0.0.1",
                 port: int = 8080, verbose: bool = False, port_tries: int = 50):
        self.sources = unique_names(list(sources))
        self.by_name = {s.name: s for s in self.sources}
        self.dist = Path(dist).resolve() if dist is not None else None
        self.verbose = verbose
        self.roots: list[Path] = []
        for s in self.sources:
            if s.path is not None and s.path.parent not in self.roots:
                self.roots.append(s.path.parent)
        self._manifests: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._serving = False
        self.requested_port = port
        self.host = host
        self.httpd = self._bind(host, port, port_tries)
        self.httpd.app = self  # type: ignore[attr-defined]
        self.port = self.httpd.server_address[1]
        self._allowed_hosts = None if not _is_loopback(host) else {"localhost", "127.0.0.1", "::1", host.strip("[]").lower()}

    # ── lifecycle ──
    def _bind(self, host: str, port: int, tries: int) -> _Server:
        cls = _Server6 if ":" in host else _Server
        ports = [0] if port == 0 else [*range(port, min(port + tries, 65536)), 0]
        err: Optional[OSError] = None
        for p in ports:
            try:
                return cls((host, p), _Handler)
            except OSError as e:
                if e.errno not in (errno.EADDRINUSE, errno.EACCES, getattr(errno, "WSAEADDRINUSE", -1)):
                    raise
                err = e
        raise err or OSError("could not bind a port")

    @property
    def port_changed(self) -> bool:
        return self.requested_port != 0 and self.port != self.requested_port

    def serve_forever(self) -> None:
        self._serving = True
        self.httpd.serve_forever(poll_interval=0.2)

    def start_background(self) -> "VizServer":
        self._thread = threading.Thread(target=self.serve_forever, name="onnxviz-http", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._serving:  # shutdown() would block forever if serve_forever() never ran
            self.httpd.shutdown()
            self._serving = False
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # ── urls ──
    @property
    def base_url(self) -> str:
        h = self.host
        if h in ("0.0.0.0", "", "::"):
            h = "localhost"
        elif ":" in h and not h.startswith("["):
            h = f"[{h}]"
        return f"http://{h}:{self.port}"

    def page_url(self) -> str:
        if not self.sources:
            return self.base_url + "/"
        first = self.sources[0]
        q = {"url": MODEL_PREFIX + quote(first.name), "external": MODEL_PREFIX + "manifest.json"}
        if len(self.sources) > 1:
            q["index"] = MODEL_PREFIX + "index.json"
        return f"{self.base_url}/?{urlencode(q, safe='/')}"

    # ── request helpers (called from handler threads) ──
    def host_ok(self, header: Optional[str]) -> bool:
        return self._allowed_hosts is None or header is None or _host_of(header) in self._allowed_hosts

    def resolve_external(self, rel: str) -> Optional[Path]:
        for root in self.roots:
            p = safe_join(root, rel)
            if p is not None and p.is_file():
                return p
        return None

    def index(self) -> dict:
        return {"models": [
            {"name": s.name, "url": MODEL_PREFIX + quote(s.name), "size": s.size,
             "manifest": f"{MODEL_PREFIX}manifest.json?model={quote(s.name)}"} for s in self.sources]}

    def manifest(self, name: Optional[str]) -> Optional[dict]:
        src = self.by_name.get(name) if name else (self.sources[0] if self.sources else None)
        if src is None:
            return None
        with self._lock:
            cached = self._manifests.get(src.name)
        if cached is not None:
            return cached
        files: list[str] = []
        detected = False
        if src.path is not None:
            cand = _list_candidates(src.path.parent, src.path)
            if len(cand) <= SCAN_MAX_CANDIDATES and src.size <= SCAN_LIMIT:
                files, detected = sorted(find_referenced(src.path, cand)), True
            else:
                files = cand[:LIST_MAX]
        man = {"model": src.name, "detected": detected, "files": files}
        with self._lock:
            self._manifests[src.name] = man
        return man


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "onnxviz"
    sys_version = ""
    timeout = 120

    def log_message(self, fmt, *args):
        if self.server.app.verbose:  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        self._dispatch(False)

    def do_HEAD(self) -> None:
        self._dispatch(True)

    # ── routing ──
    def _dispatch(self, head: bool) -> None:
        app: VizServer = self.server.app  # type: ignore[attr-defined]
        try:
            if not app.host_ok(self.headers.get("Host")):
                return self._text(403, "Forbidden: unexpected Host header", head)
            if self.path.startswith("/"):  # don't let urlsplit read '//x/y' as a netloc
                raw, _, query = self.path.partition("#")[0].partition("?")
            else:
                u = urlsplit(self.path)
                raw, query = u.path, u.query
            path = unquote(raw)
            if "\0" in path or not path.startswith("/"):
                return self._text(400, "Bad request", head)
            if path.startswith(MODEL_PREFIX):
                return self._model(app, path[len(MODEL_PREFIX):], parse_qs(query), head)
            return self._static(app, path, head)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            self.close_connection = True

    def _model(self, app: VizServer, rel: str, query: dict, head: bool) -> None:
        if rel == "index.json":
            return self._json(app.index(), head)
        if rel == "manifest.json":
            man = app.manifest((query.get("model") or [None])[0])
            return self._json(man, head) if man is not None else self._text(404, "Not found", head)
        src = app.by_name.get(rel)
        if src is not None:
            return self._blob(src, src.content_type, head)
        p = app.resolve_external(rel)
        if p is not None:
            return self._blob(ModelSource(p.name, path=p), "application/octet-stream", head)
        self._text(404, "Not found", head)

    def _static(self, app: VizServer, path: str, head: bool) -> None:
        if app.dist is None:
            return self._text(503, "onnxviz: the web app is not built (run `make build`).", head)
        p = safe_join(app.dist, path)
        if p is not None and p.is_dir():
            p = safe_join(app.dist, path.rstrip("/") + "/index.html")
        if p is None or not p.is_file():
            return self._text(404, "Not found", head)
        ctype = _MIME.get(p.suffix.lower()) or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        self._blob(ModelSource(p.name, path=p), ctype, head)

    # ── responses ──
    def _headers(self, status: int, ctype: str, length: int, extra: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _text(self, status: int, msg: str, head: bool) -> None:
        body = msg.encode("utf-8")
        self._headers(status, "text/plain; charset=utf-8", len(body))
        if not head:
            self.wfile.write(body)

    def _json(self, obj: dict, head: bool) -> None:
        body = json.dumps(obj).encode("utf-8")
        self._headers(200, "application/json", len(body))
        if not head:
            self.wfile.write(body)

    def _blob(self, blob: ModelSource, ctype: str, head: bool) -> None:
        try:
            size = blob.size
        except OSError:
            return self._text(404, "Not found", head)
        status, start, end = 200, 0, size - 1
        extra = {"Accept-Ranges": "bytes"}
        rng = self.headers.get("Range")
        if rng:
            try:
                r = parse_range(rng, size)
            except RangeError:
                return self._range_not_satisfiable(size, head)
            if r is not None:
                status, (start, end) = 206, r
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = max(0, end - start + 1)
        self._headers(status, ctype, length, extra)
        if head or length == 0:
            return
        sent = 0
        try:
            for chunk in blob.iter_range(start, end + 1):
                self.wfile.write(chunk)
                sent += len(chunk)
        except OSError:
            self.close_connection = True
            raise
        if sent != length:  # file shrank while streaming: the framing is broken, drop the connection
            self.close_connection = True

    def _range_not_satisfiable(self, size: int, head: bool) -> None:
        body = b"Range not satisfiable"
        self._headers(416, "text/plain; charset=utf-8", len(body), {"Content-Range": f"bytes */{size}"})
        if not head:
            self.wfile.write(body)
