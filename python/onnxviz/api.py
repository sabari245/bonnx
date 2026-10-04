"""Python API: show / save_html / display."""
from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from typing import Any, Optional

from ._dist import MISSING_DIST, find_dist
from ._sources import as_source
from .embed import write_embedded_html
from .server import VizServer

_ACTIVE: list[VizServer] = []


def _open_later(url: str) -> None:
    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()


def show(model: Any, port: int = 8080, host: str = "127.0.0.1", block: bool = True, open_browser: bool = True,
         name: Optional[str] = None, dist: Optional[str | Path] = None, verbose: bool = False) -> VizServer:
    """Serve `model` (path, bytes, or an onnx.ModelProto) in the viewer.

    block=True serves until Ctrl-C; block=False starts a background thread and returns the server
    (use `.url` and `.stop()`). If `port` is busy the next free one is used (see `.port`).
    """
    src = as_source(model, name)
    web = find_dist(dist)
    if web is None:
        raise FileNotFoundError(MISSING_DIST)
    srv = VizServer([src], web, host, port, verbose=verbose)
    url = srv.page_url()
    if open_browser:
        _open_later(url)
    if not block:
        _ACTIVE.append(srv)
        return srv.start_background()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return srv


def save_html(model: Any, out: str | Path, name: Optional[str] = None, force: bool = False,
              template: Optional[str | Path] = None) -> Path:
    """Write a single self-contained HTML viewer with `model` embedded. Returns the output path."""
    out = Path(out)
    write_embedded_html(as_source(model, name), out, template=template, force=force)
    return out


def display(model: Any, width: Any = "100%", height: Any = 600, name: Optional[str] = None,
            dist: Optional[str | Path] = None):
    """Jupyter: serve in the background on a free port and return an IPython IFrame.

    The iframe loads from localhost, so the notebook kernel and the browser must run on the same machine.
    """
    try:
        from IPython.display import IFrame
    except ImportError as e:  # pragma: no cover
        raise ImportError("onnxviz.display() needs IPython (pip install ipython)") from e
    srv = show(model, port=0, block=False, open_browser=False, name=name, dist=dist)
    return IFrame(src=srv.page_url(), width=width, height=height)


def stop_all() -> None:
    """Stop every background server started by show(block=False) / display()."""
    while _ACTIVE:
        _ACTIVE.pop().stop()
