"""Model sources: a file on disk or bytes in memory, both streamable by byte range."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterator, Optional

# Longest first so ".onnx.prototxt" wins over ".prototxt".
MODEL_EXTS = (".onnx.prototxt", ".onnx", ".prototxt", ".json", ".pb")
# Extensions picked up when a *directory* is given on the command line.
DIR_EXTS = (".onnx", ".onnx.prototxt")
# Names under /__model__/ that belong to the server itself.
RESERVED = frozenset({"index.json", "manifest.json"})
CHUNK = 1 << 20


def split_ext(name: str) -> tuple[str, str]:
    low = name.lower()
    for e in MODEL_EXTS:
        if low.endswith(e) and len(name) > len(e):
            return name[: -len(e)], name[-len(e):]
    return os.path.splitext(name)


def clean_name(name: str) -> str:
    """Last path component, without control characters; never empty."""
    name = re.sub(r"[\x00-\x1f\x7f]", "", name.replace("\\", "/").rsplit("/", 1)[-1]).strip()
    return name if name not in ("", ".", "..") else "model.onnx"


class ModelSource:
    """One model the server can hand out. Exactly one of `path` / `data` is set."""

    __slots__ = ("name", "path", "data")

    def __init__(self, name: str, path: Optional[Path] = None, data: Optional[Any] = None):
        if (path is None) == (data is None):
            raise ValueError("exactly one of path / data is required")
        self.name = clean_name(name)
        self.path = path
        self.data = data  # bytes or contiguous byte memoryview

    @property
    def size(self) -> int:
        return self.path.stat().st_size if self.path is not None else len(self.data)

    @property
    def content_type(self) -> str:
        low = self.name.lower()
        if low.endswith(".json"):
            return "application/json"
        if low.endswith(".prototxt"):
            return "text/plain; charset=utf-8"
        return "application/octet-stream"

    def iter_range(self, start: int, stop: int, chunk: int = CHUNK) -> Iterator[bytes]:
        """Yield bytes [start, stop) without ever holding more than `chunk` bytes of a file."""
        if self.path is not None:
            with open(self.path, "rb") as f:
                f.seek(start)
                left = stop - start
                while left > 0:
                    b = f.read(min(chunk, left))
                    if not b:
                        return
                    left -= len(b)
                    yield b
        else:
            mv = memoryview(self.data)
            for i in range(start, stop, chunk):
                yield bytes(mv[i : min(i + chunk, stop)])

    def __repr__(self) -> str:  # pragma: no cover
        return f"ModelSource({self.name!r}, {'file' if self.path else 'memory'}, {self.size} bytes)"


def as_source(obj: Any, name: Optional[str] = None) -> ModelSource:
    """Accept a path, bytes, an onnx.ModelProto (anything with SerializeToString) or a binary file object."""
    if isinstance(obj, (str, os.PathLike)):
        p = Path(obj).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"model not found: {p}")
        return ModelSource(name or p.name, path=p.resolve())
    if isinstance(obj, (bytes, bytearray, memoryview)):
        data = obj if isinstance(obj, bytes) else bytes(memoryview(obj).cast("B"))
        return ModelSource(name or "model.onnx", data=data)
    if hasattr(obj, "SerializeToString"):
        try:
            data = obj.SerializeToString()
        except Exception as e:  # protobuf raises ValueError for >2 GB messages
            raise ValueError(
                "could not serialize the ModelProto in memory (models over 2 GB must be saved with "
                f"external data first, then passed by path): {e}"
            ) from e
        return ModelSource(name or "model.onnx", data=data)
    if hasattr(obj, "read"):
        data = obj.read()
        return ModelSource(name or "model.onnx", data=data.encode("utf-8") if isinstance(data, str) else bytes(data))
    raise TypeError(f"cannot use {type(obj).__name__} as a model; pass a path, bytes or an onnx.ModelProto")


def unique_names(sources: list[ModelSource]) -> list[ModelSource]:
    """Make names unique (and not clash with the server's reserved names). Mutates and returns `sources`."""
    seen: set[str] = set()
    for s in sources:
        name = s.name
        if name in RESERVED:
            name = "_" + name
        base, ext = split_ext(name)
        n = 2
        while name in seen:
            name = f"{base}-{n}{ext}"
            n += 1
        seen.add(name)
        s.name = name
    return sources
