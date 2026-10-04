import http.client
import os
import tempfile
import unittest
from pathlib import Path

from bonnx._sources import ModelSource
from bonnx.server import VizServer

STUB_INDEX = "<!doctype html><html><head><title>stub</title></head><body>app</body></html>"
SINGLE_INDEX = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>bonnx</title>'
                '<script type="module">console.log("app")</script></head><body></body></html>')


def make_dist(root: Path) -> Path:
    d = root / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(STUB_INDEX)
    (d / "assets" / "app.js").write_text("export default 1")
    (d / ".secret").write_text("nope")
    return d


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)


class ServerCase(TmpCase):
    """A running server over a stub dist and a model dir: model.onnx (1 MB random), weights.bin, notes.txt."""

    def setUp(self):
        super().setUp()
        self.dist = make_dist(self.tmp)
        self.mdir = self.tmp / "models"
        self.mdir.mkdir()
        self.blob = os.urandom(2_100_003)
        self.weights = os.urandom(5000)
        (self.mdir / "weights.bin").write_bytes(self.weights)
        (self.mdir / "notes.txt").write_text("unrelated")
        (self.mdir / ".hidden").write_text("hidden")
        # the "model" references weights.bin by name, like an external-data location string
        self.model_bytes = self.blob + b"\x0a\x0blocation\x12\x0bweights.bin"
        (self.mdir / "model.onnx").write_bytes(self.model_bytes)
        (self.tmp / "secret.txt").write_text("outside")
        self.srv = self.start([ModelSource("model.onnx", path=self.mdir / "model.onnx")])

    def start(self, sources, dist="default", **kw):
        srv = VizServer(sources, self.dist if dist == "default" else dist, "127.0.0.1", 0, **kw).start_background()
        self.addCleanup(srv.stop)
        return srv

    def get(self, path, headers=None, method="GET", srv=None, host=None):
        srv = srv or self.srv
        c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
        try:
            h = dict(headers or {})
            c.putrequest(method, path, skip_host=host is not None)
            if host is not None:
                c.putheader("Host", host)
            for k, v in h.items():
                c.putheader(k, v)
            c.endheaders()
            r = c.getresponse()
            return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
        finally:
            c.close()
