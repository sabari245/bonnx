import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import bonnx
from bonnx import cli
from bonnx._sources import as_source
from support import SINGLE_INDEX, ServerCase, TmpCase, make_dist


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main(list(argv))
        except SystemExit as e:  # argparse
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


class Cli(TmpCase):
    def setUp(self):
        super().setUp()
        self.m = self.tmp / "a.onnx"
        self.m.write_bytes(b"model-bytes")
        self.single = self.tmp / "single.html"
        self.single.write_text(SINGLE_INDEX)

    def test_version(self):
        rc, out, _ = run_cli("--version")
        self.assertEqual(rc, 0)
        self.assertIn(bonnx.__version__, out)

    def test_embed_out(self):
        out = self.tmp / "v.html"
        rc, so, _ = run_cli(str(self.m), "--out", str(out), "--template", str(self.single))
        self.assertEqual(rc, 0, so)
        self.assertIn("__BONNX_EMBED__", out.read_text())

    def test_embed_flag_defaults_to_stem_in_cwd(self):
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            rc, _, err = run_cli(str(self.m), "--embed", "--template", str(self.single))
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 0, err)
        self.assertTrue((self.tmp / "a.html").exists())

    def test_embed_errors(self):
        self.assertEqual(run_cli("--out", str(self.tmp / "x.html"))[0], 2)
        b = self.tmp / "b.onnx"
        b.write_bytes(b"2")
        rc, _, err = run_cli(str(self.m), str(b), "--out", str(self.tmp / "x.html"), "--template", str(self.single))
        self.assertEqual(rc, 2)
        self.assertIn("single model", err)
        rc, _, err = run_cli(str(self.m), "--out", str(self.m), "--template", str(self.single))
        self.assertEqual(rc, 2)
        self.assertIn("overwrite", err)
        self.assertEqual(self.m.read_bytes(), b"model-bytes")

    def test_input_validation(self):
        self.assertEqual(run_cli(str(self.tmp / "missing.onnx"))[0], 2)
        txt = self.tmp / "x.txt"
        txt.write_text("t")
        rc, _, err = run_cli(str(txt))
        self.assertEqual(rc, 2)
        self.assertIn("unsupported", err)
        empty = self.tmp / "empty"
        empty.mkdir()
        self.assertEqual(run_cli(str(empty))[0], 2)

    def test_accepted_extensions_and_directory_expansion(self):
        d = self.tmp / "dir"
        d.mkdir()
        for n in ("b.onnx", "a.onnx", "c.onnx.prototxt", "notes.txt", "d.json", ".hid.onnx"):
            (d / n).write_bytes(b"1")
        names = [s.name for s in cli.collect_models([str(d)])]
        self.assertEqual(names, ["a.onnx", "b.onnx", "c.onnx.prototxt"])  # dirs: .onnx / .onnx.prototxt only
        self.assertEqual([s.name for s in cli.collect_models([str(d / "d.json")])], ["d.json"])  # explicit file: .json ok
        both = cli.collect_models([str(d), str(d / "a.onnx")])
        self.assertEqual(len(both), 3)  # duplicate dropped

    def test_serve_without_a_build_fails_cleanly(self):
        with mock.patch.object(cli, "find_dist", return_value=None):
            rc, _, err = run_cli(str(self.m), "--no-browser")
        self.assertEqual(rc, 2)
        self.assertIn("not built", err)

    def test_serve_runs_until_interrupted(self):
        dist = make_dist(self.tmp)
        started = {}

        def fake_serve(self_):
            started["url"] = self_.page_url()
            raise KeyboardInterrupt

        with mock.patch("bonnx.server.VizServer.serve_forever", fake_serve):
            rc, out, _ = run_cli(str(self.m), "--no-browser", "--port", "0", "--dist", str(dist))
        self.assertEqual(rc, 0)
        self.assertIn("viewer: http://127.0.0.1:", out)
        self.assertIn("url=/__model__/a.onnx", started["url"])


class Api(ServerCase):
    def test_sources(self):
        self.assertEqual(as_source(b"abc").size, 3)
        self.assertEqual(as_source(bytearray(b"abcd"), name="x.onnx").name, "x.onnx")
        self.assertEqual(as_source(memoryview(b"ab")).size, 2)
        self.assertEqual(as_source(str(self.mdir / "model.onnx")).path, (self.mdir / "model.onnx").resolve())
        with self.assertRaises(FileNotFoundError):
            as_source(self.tmp / "nope.onnx")
        with self.assertRaises(TypeError):
            as_source(12)

    def test_show_nonblocking_serves_bytes_from_memory(self):
        srv = bonnx.show(b"hello-model", port=0, block=False, open_browser=False, dist=self.dist, name="h.onnx")
        self.addCleanup(srv.stop)
        self.assertEqual(self.get("/__model__/h.onnx", srv=srv)[2], b"hello-model")

    def test_show_without_build_raises(self):
        with mock.patch("bonnx.api.find_dist", return_value=None):
            with self.assertRaises(FileNotFoundError):
                bonnx.show(b"x", block=False, open_browser=False)

    def test_save_html(self):
        single = self.tmp / "single.html"
        single.write_text(SINGLE_INDEX)
        out = bonnx.save_html(b"bytes", self.tmp / "o.html", template=single)
        self.assertIn("__BONNX_EMBED__", Path(out).read_text())

    def test_modelproto_is_serialised_in_memory(self):
        class FakeProto:  # stands in for onnx.ModelProto
            def SerializeToString(self):
                return b"proto-bytes"

        src = as_source(FakeProto())
        self.assertEqual((src.name, src.data), ("model.onnx", b"proto-bytes"))

    @unittest.skipUnless(__import__("importlib").util.find_spec("onnx"), "onnx not installed")
    def test_real_onnx_modelproto(self):
        import onnx
        from onnx import TensorProto, helper

        g = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        m = helper.make_model(g)
        srv = bonnx.show(m, port=0, block=False, open_browser=False, dist=self.dist)
        self.addCleanup(srv.stop)
        body = self.get("/__model__/model.onnx", srv=srv)[2]
        self.assertEqual(onnx.load_from_string(body).graph.name, "g")
        out = bonnx.save_html(m, self.tmp / "m.html", template=self._single())
        self.assertIn("__BONNX_EMBED__", Path(out).read_text())

    def _single(self):
        p = self.tmp / "single.html"
        p.write_text(SINGLE_INDEX)
        return p

    def test_stop_all(self):
        bonnx.show(b"x", port=0, block=False, open_browser=False, dist=self.dist)
        bonnx.stop_all()


if __name__ == "__main__":
    unittest.main()
