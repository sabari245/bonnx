import base64
import json
import re
import unittest

from onnxviz._sources import ModelSource
from onnxviz.embed import EmbedError, inject_position, write_embedded_html
from support import SINGLE_INDEX, TmpCase


def extract(html: str) -> dict:
    m = re.search(r"<script>window\.__ONNXVIZ_EMBED__=(\{.*?\});</script>", html, re.S)
    assert m, "embed script not found"
    return json.loads(m.group(1))


class Embed(TmpCase):
    def template(self, text=SINGLE_INDEX):
        p = self.tmp / "single.html"
        p.write_text(text, encoding="utf-8")
        return p

    def embed(self, src, text=SINGLE_INDEX, **kw):
        out = self.tmp / "out.html"
        n = write_embedded_html(src, out, template=self.template(text), **kw)
        self.assertEqual(n, out.stat().st_size)
        return out.read_text(encoding="utf-8")

    def test_roundtrip_all_remainders(self):
        for n in (0, 1, 2, 3, 4, 5, (3 << 20) - 1, (3 << 20), (3 << 20) + 1, 7_000_001):
            data = bytes(i % 251 for i in range(n)) if n < 100 else (b"\x00\xff\x10" * (n // 3 + 1))[:n]
            html = self.embed(ModelSource("m.onnx", data=data))
            e = extract(html)
            self.assertEqual(base64.b64decode(e["b64"]), data, n)
            self.assertEqual((e["name"], e["size"]), ("m.onnx", n))

    def test_file_source_streams(self):
        p = self.tmp / "big.onnx"
        p.write_bytes(bytes(range(256)) * 50_000)
        e = extract(self.embed(ModelSource("big.onnx", path=p)))
        self.assertEqual(base64.b64decode(e["b64"]), p.read_bytes())

    def test_script_goes_before_the_app_script_and_page_is_intact(self):
        html = self.embed(ModelSource("m.onnx", data=b"abc"))
        self.assertLess(html.index("__ONNXVIZ_EMBED__"), html.index('<script type="module">'))
        self.assertTrue(html.startswith('<!doctype html><html lang="en"><head><script>window.__ONNXVIZ_EMBED__'))
        self.assertTrue(html.endswith('console.log("app")</script></head><body></body></html>'))

    def test_inject_positions(self):
        self.assertEqual(inject_position("<html><HEAD data-x><title>t</title>"), len("<html><HEAD data-x>"))
        self.assertEqual(inject_position("<html><title>t</title><script>x</script>"), len("<html><title>t</title>"))
        self.assertEqual(inject_position('<html lang="en"><body><script>x</script>'), len('<html lang="en">'))
        self.assertEqual(inject_position("<p>hi</p><script>x</script>"), len("<p>hi</p>"))
        self.assertEqual(inject_position("plain"), 0)
        # <header> is not <head>, and a '<head>' appearing later in JS is ignored
        s = '<html><header></header><script>"<head>"</script>'
        self.assertEqual(inject_position(s), len("<html>"))

    def test_templates_without_head(self):
        html = self.embed(ModelSource("m.onnx", data=b"xyz"), text="<title>t</title><script>app()</script>")
        self.assertLess(html.index("__ONNXVIZ_EMBED__"), html.index("app()"))
        self.assertEqual(base64.b64decode(extract(html)["b64"]), b"xyz")

    def test_hostile_name_cannot_break_out_of_the_script(self):
        name = 'a"</script><script>alert(1)</script><!--.onnx'
        html = self.embed(ModelSource(name, data=b"x"))
        head = html[: html.index('<script type="module">')]
        self.assertEqual(head.lower().count("</script>"), 1)
        self.assertEqual(head.count("<script"), 1)
        self.assertEqual(extract(html)["name"], ModelSource(name, data=b"x").name)

    def test_unicode_name(self):
        html = self.embed(ModelSource("модель.onnx", data=b"x"))
        self.assertEqual(extract(html)["name"], "модель.onnx")

    def test_size_limits(self):
        src = ModelSource("m.onnx", data=b"x" * 100)
        with self.assertRaises(EmbedError) as cm:
            self.embed(src, limit=50)
        self.assertIn("--force", str(cm.exception))
        self.assertEqual(base64.b64decode(extract(self.embed(src, limit=50, force=True))["b64"]), b"x" * 100)
        with self.assertRaises(EmbedError) as cm:
            self.embed(src, limit=50, force=True, hard_limit=99)
        self.assertIn("cannot hold", str(cm.exception))

    def test_failure_leaves_no_partial_output(self):
        out = self.tmp / "out.html"
        with self.assertRaises(EmbedError):
            write_embedded_html(ModelSource("m.onnx", data=b"x" * 10), out, template=self.template(), limit=1)
        self.assertFalse(out.exists())
        self.assertFalse((self.tmp / "out.html.tmp").exists())

    def test_missing_template(self):
        with self.assertRaises(FileNotFoundError):
            write_embedded_html(ModelSource("m.onnx", data=b"x"), self.tmp / "o.html", template=self.tmp / "nope.html")


if __name__ == "__main__":
    unittest.main()
