"""Build one self-contained HTML file: the single-file web build with the model embedded as base64.

The model is injected as a classic script placed right after <head>, so it runs before the (deferred) app script:

    <script>window.__BONNX_EMBED__={"name":"model.onnx","size":1234,"b64":"..."};</script>

The viewer reads it on startup (`loadEmbedded()` in src/app/loader.ts), decodes it and deletes the global.
External-data files are not embedded (use the server mode for models that keep weights in side files).
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional

from ._dist import MISSING_SINGLE, find_single
from ._sources import ModelSource

EMBED_LIMIT = 200 << 20  # refuse larger models unless force=True
# Browsers cap a string at ~512M characters; base64 is 4/3 of the model, so ~380 MB is the hard ceiling.
HARD_LIMIT = 380 << 20
_B64_CHUNK = 3 << 20  # multiple of 3 so chunks concatenate into valid base64

_HEAD = re.compile(r"<head(?:\s[^>]*)?>", re.I)
_TITLE_END = re.compile(r"</title\s*>", re.I)
_HTML = re.compile(r"<html(?:\s[^>]*)?>", re.I)
_SCRIPT = re.compile(r"<script[\s>]", re.I)


class EmbedError(Exception):
    pass


def inject_position(html: str) -> int:
    """Index at which the model script is inserted: after <head>, else after </title>, <html>, else before the first <script>."""
    script = _SCRIPT.search(html)
    first_script = script.start() if script else len(html)
    for rx in (_HEAD, _TITLE_END, _HTML):
        m = rx.search(html, 0, first_script)  # a '<head>' inside a script's text must not count
        if m:
            return m.end()
    return first_script if script else 0


def _js_string(s: str) -> str:
    # json.dumps gives a valid JS string; escaping every '<' means nothing in it can end the <script> element
    # early or open a comment ("</script>", "<!--"), and the result stays valid JSON.
    return json.dumps(s).replace("<", "\\u003c")


def write_embedded_html(source: ModelSource, out: str | os.PathLike, template: Optional[str | Path] = None,
                        force: bool = False, limit: int = EMBED_LIMIT, hard_limit: int = HARD_LIMIT) -> int:
    """Write the viewer with `source` embedded to `out` (atomically); returns the output size in bytes."""
    size = source.size
    mb = size / 1e6
    if size > hard_limit:
        raise EmbedError(
            f"{source.name} is {mb:.0f} MB; browsers cannot hold an embedded model above ~{hard_limit // (1 << 20)} MB. "
            "Serve it instead: `bonnx MODEL` (no --out).")
    if size > limit and not force:
        raise EmbedError(
            f"{source.name} is {mb:.0f} MB, above the {limit // (1 << 20)} MB embed limit. Pass --force to embed anyway "
            "(the HTML will be ~33% larger than the model and slow to open), or serve it with `bonnx MODEL`.")
    tpl = find_single(template)
    if tpl is None:
        raise EmbedError(MISSING_SINGLE)
    html = tpl.read_text(encoding="utf-8")
    at = inject_position(html)
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(html[:at])
            f.write(f'<script>window.__BONNX_EMBED__={{"name":{_js_string(source.name)},"size":{size},"b64":"')
            for chunk in source.iter_range(0, size, _B64_CHUNK):
                f.write(base64.b64encode(chunk).decode("ascii"))
            f.write('"};</script>')
            f.write(html[at:])
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return out.stat().st_size
