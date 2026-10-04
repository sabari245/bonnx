"""Locate the built web app."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

PKG = Path(__file__).resolve().parent
REPO = PKG.parent.parent  # python/bonnx -> python -> repo root


def _ok(p: Path) -> bool:
    return (p / "index.html").is_file()


def find_dist(explicit: Optional[str | Path] = None) -> Optional[Path]:
    """Directory holding the multi-file web build: --dist, then package data, then <repo>/dist."""
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not _ok(p):
            raise FileNotFoundError(f"no index.html in --dist directory: {p}")
        return p
    for p in (PKG / "web", REPO / "dist"):
        if _ok(p):
            return p
    return None


def find_single(explicit: Optional[str | Path] = None) -> Optional[Path]:
    """The single-file build (index.html with everything inlined): --template, package data, <repo>/dist-single."""
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"template not found: {p}")
        return p
    for p in (PKG / "web_single" / "index.html", REPO / "dist-single" / "index.html"):
        if p.is_file():
            return p
    return None


MISSING_DIST = (
    "the web app is not built. Run `make build` (or `bun run build`) in the repo, "
    "pass --dist DIR, or install a package that bundles it (`make package`)."
)
MISSING_SINGLE = (
    "the single-file web build is missing. Run `make single` (or `bun run build:single`) in the repo, "
    "or pass --template FILE."
)
