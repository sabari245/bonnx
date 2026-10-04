import json
import os
import socket
import unittest

from bonnx._sources import ModelSource, unique_names
from bonnx.server import RangeError, VizServer, find_referenced, parse_range, safe_join
from support import ServerCase, TmpCase


class RangeParsing(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(parse_range("bytes=0-9", 100), (0, 9))
        self.assertEqual(parse_range("bytes=90-", 100), (90, 99))
        self.assertEqual(parse_range("bytes=90-500", 100), (90, 99))
        self.assertEqual(parse_range("bytes=-10", 100), (90, 99))
        self.assertEqual(parse_range("bytes=-500", 100), (0, 99))

    def test_ignored(self):
        for h in ("bytes=0-1,5-6", "items=0-1", "bytes=abc", "bytes=5-2", "bytes=-", "bytes="):
            self.assertIsNone(parse_range(h, 100), h)

    def test_unsatisfiable(self):
        for h, size in (("bytes=100-", 100), ("bytes=200-300", 100), ("bytes=-0", 100), ("bytes=0-", 0)):
            with self.assertRaises(RangeError, msg=h):
                parse_range(h, size)


class Serving(ServerCase):
    def test_full_model_has_length_and_accepts_ranges(self):
        st, h, body = self.get("/__model__/model.onnx")
        self.assertEqual(st, 200)
        self.assertEqual(h["content-length"], str(len(self.model_bytes)))
        self.assertEqual(h["accept-ranges"], "bytes")
        self.assertEqual(body, self.model_bytes)

    def test_range(self):
        st, h, body = self.get("/__model__/model.onnx", {"Range": "bytes=10-19"})
        self.assertEqual((st, body), (206, self.model_bytes[10:20]))
        self.assertEqual(h["content-range"], f"bytes 10-19/{len(self.model_bytes)}")
        self.assertEqual(h["content-length"], "10")

    def test_range_suffix_open_ended_and_multi_chunk(self):
        n = len(self.model_bytes)
        self.assertEqual(self.get("/__model__/model.onnx", {"Range": "bytes=-5"})[2], self.model_bytes[-5:])
        self.assertEqual(self.get("/__model__/model.onnx", {"Range": f"bytes={n - 7}-"})[2], self.model_bytes[-7:])
        # crosses the 1 MiB streaming chunk boundary
        st, _, body = self.get("/__model__/model.onnx", {"Range": "bytes=1048000-1048700"})
        self.assertEqual(st, 206)
        self.assertEqual(body, self.model_bytes[1048000:1048701])

    def test_range_not_satisfiable(self):
        n = len(self.model_bytes)
        st, h, _ = self.get("/__model__/model.onnx", {"Range": f"bytes={n}-"})
        self.assertEqual(st, 416)
        self.assertEqual(h["content-range"], f"bytes */{n}")

    def test_head(self):
        st, h, body = self.get("/__model__/model.onnx", method="HEAD")
        self.assertEqual((st, body), (200, b""))
        self.assertEqual(h["content-length"], str(len(self.model_bytes)))

    def test_no_cors_headers(self):
        for path in ("/", "/__model__/model.onnx", "/__model__/manifest.json"):
            _, h, _ = self.get(path)
            self.assertFalse([k for k in h if k.startswith("access-control")], path)

    def test_keepalive_connection_serves_several_requests(self):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=10)
        for _ in range(3):
            c.request("GET", "/__model__/model.onnx", headers={"Range": "bytes=0-3"})
            r = c.getresponse()
            self.assertEqual(r.read(), self.model_bytes[:4])
        c.close()

    def test_static_app(self):
        st, h, body = self.get("/")
        self.assertEqual(st, 200)
        self.assertIn("text/html", h["content-type"])
        self.assertIn(b"stub", body)
        st, h, _ = self.get("/assets/app.js")
        self.assertEqual(st, 200)
        self.assertIn("javascript", h["content-type"])
        self.assertEqual(self.get("/nope.js")[0], 404)

    def test_missing_dist_is_503_not_a_crash(self):
        srv = self.start([], dist=None)
        self.assertEqual(self.get("/", srv=srv)[0], 503)

    def test_page_url_points_at_the_model_and_manifest(self):
        u = self.srv.page_url()
        self.assertIn("url=/__model__/model.onnx", u)
        self.assertIn("external=/__model__/manifest.json", u)
        self.assertTrue(u.startswith(f"http://127.0.0.1:{self.srv.port}/?"))


class External(ServerCase):
    def test_sibling_file_served(self):
        st, _, body = self.get("/__model__/weights.bin")
        self.assertEqual((st, body), (200, self.weights))
        self.assertEqual(self.get("/__model__/weights.bin", {"Range": "bytes=2-4"})[2], self.weights[2:5])

    def test_manifest_lists_only_referenced_files(self):
        st, _, body = self.get("/__model__/manifest.json")
        man = json.loads(body)
        self.assertEqual(st, 200)
        self.assertEqual(man["files"], ["weights.bin"])
        self.assertTrue(man["detected"])
        self.assertEqual(man["model"], "model.onnx")

    def test_manifest_lists_candidates_when_not_scannable(self):
        for i in range(70):
            (self.mdir / f"extra{i:02}.bin").write_bytes(b"x")
        srv = self.start([ModelSource("model.onnx", path=self.mdir / "model.onnx")])
        man = json.loads(self.get("/__model__/manifest.json", srv=srv)[2])
        self.assertFalse(man["detected"])
        self.assertIn("weights.bin", man["files"])
        self.assertNotIn(".hidden", man["files"])
        self.assertNotIn("model.onnx", man["files"])

    def test_nested_location_found_by_basename_and_served(self):
        (self.mdir / "sub").mkdir()
        (self.mdir / "sub" / "w2.bin").write_bytes(b"nested")
        (self.mdir / "model.onnx").write_bytes(self.model_bytes + b"w2.bin")
        srv = self.start([ModelSource("model.onnx", path=self.mdir / "model.onnx")])
        man = json.loads(self.get("/__model__/manifest.json", srv=srv)[2])
        self.assertEqual(sorted(man["files"]), ["sub/w2.bin", "weights.bin"])
        self.assertEqual(self.get("/__model__/sub/w2.bin", srv=srv)[2], b"nested")

    def test_find_referenced_across_chunk_boundary(self):
        p = self.mdir / "big.onnx"
        p.write_bytes(b"a" * 98 + b"weights.bin" + b"b" * 50)
        self.assertEqual(find_referenced(p, ["weights.bin", "other.bin"], chunk=100), {"weights.bin"})

    def test_unknown_model_manifest_404(self):
        self.assertEqual(self.get("/__model__/manifest.json?model=zzz")[0], 404)


class Traversal(ServerCase):
    def test_model_prefix_escapes_are_rejected(self):
        bad = ["/__model__/../secret.txt", "/__model__/%2e%2e/secret.txt", "/__model__/..%2fsecret.txt",
               "/__model__/%2e%2e%2fsecret.txt", "/__model__/sub/../../secret.txt", "/__model__/..\\secret.txt",
               "/__model__/%5c..%5csecret.txt", "/__model__//etc/passwd", "/__model__/%2fetc/passwd",
               "/__model__/./../secret.txt", "/__model__/%00", "/__model__/model.onnx%00.txt"]
        for p in bad:
            st, _, body = self.get(p)
            self.assertIn(st, (400, 404), p)
            self.assertNotIn(b"outside", body, p)

    def test_static_escapes_are_rejected(self):
        for p in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt", "/..%2fsecret.txt", "/.secret", "/assets/.%2e/.secret"):
            st, _, body = self.get(p)
            self.assertIn(st, (400, 404), p)
            self.assertNotIn(b"nope", body)
            self.assertNotIn(b"outside", body)

    def test_hidden_files_not_served(self):
        self.assertEqual(self.get("/__model__/.hidden")[0], 404)

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_symlink_out_of_root_rejected(self):
        os.symlink(self.tmp / "secret.txt", self.mdir / "link.txt")
        st, _, body = self.get("/__model__/link.txt")
        self.assertEqual(st, 404)
        self.assertNotIn(b"outside", body)

    def test_safe_join_unit(self):
        self.assertIsNone(safe_join(self.mdir, "../secret.txt"))
        self.assertIsNone(safe_join(self.mdir, "a\\b"))
        self.assertEqual(safe_join(self.mdir, "weights.bin"), (self.mdir / "weights.bin").resolve())
        self.assertEqual(safe_join(self.mdir, "./weights.bin"), (self.mdir / "weights.bin").resolve())

    def test_dns_rebinding_host_refused_on_loopback(self):
        self.assertEqual(self.get("/", host="evil.example.com")[0], 403)
        self.assertEqual(self.get("/__model__/model.onnx", host="evil.example.com:8080")[0], 403)
        self.assertEqual(self.get("/", host=f"localhost:{self.srv.port}")[0], 200)
        self.assertEqual(self.get("/", host=f"127.0.0.1:{self.srv.port}")[0], 200)


class Multiple(ServerCase):
    def test_index_and_names(self):
        (self.mdir / "other.onnx").write_bytes(b"second")
        (self.mdir / "index.json").write_bytes(b"{}")  # a model literally called index.json
        srcs = [ModelSource("model.onnx", path=self.mdir / "model.onnx"),
                ModelSource("other.onnx", path=self.mdir / "other.onnx"),
                ModelSource("index.json", path=self.mdir / "index.json")]
        srv = self.start(srcs)
        idx = json.loads(self.get("/__model__/index.json", srv=srv)[2])["models"]
        self.assertEqual([m["name"] for m in idx], ["model.onnx", "other.onnx", "_index.json"])
        self.assertEqual(self.get(idx[1]["url"], srv=srv)[2], b"second")
        self.assertEqual(self.get(idx[2]["url"], srv=srv)[2], b"{}")
        self.assertIn("index=/__model__/index.json", srv.page_url())

    def test_duplicate_names_get_suffix(self):
        a = ModelSource("m.onnx", data=b"a")
        b = ModelSource("m.onnx", data=b"b")
        c = ModelSource("m.onnx.prototxt", data=b"c")
        d = ModelSource("m.onnx.prototxt", data=b"d")
        self.assertEqual([s.name for s in unique_names([a, b, c, d])], ["m.onnx", "m-2.onnx", "m.onnx.prototxt", "m-2.onnx.prototxt"])

    def test_in_memory_model_with_range(self):
        srv = self.start([ModelSource("mem.onnx", data=b"0123456789")])
        self.assertEqual(self.get("/__model__/mem.onnx", srv=srv)[2], b"0123456789")
        st, h, body = self.get("/__model__/mem.onnx", {"Range": "bytes=3-5"}, srv=srv)
        self.assertEqual((st, body, h["content-range"]), (206, b"345", "bytes 3-5/10"))

    def test_url_encoded_name(self):
        (self.mdir / "my model #1.onnx").write_bytes(b"spaced")
        srv = self.start([ModelSource("my model #1.onnx", path=self.mdir / "my model #1.onnx")])
        self.assertIn("url=/__model__/my%2520model%2520%25231.onnx", srv.page_url())
        self.assertEqual(self.get("/__model__/my%20model%20%231.onnx", srv=srv)[2], b"spaced")


class PortFallback(TmpCase):
    def test_busy_port_moves_on(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        try:
            srv = VizServer([], None, "127.0.0.1", busy)
            try:
                self.assertNotEqual(srv.port, busy)
                self.assertTrue(srv.port_changed)
                self.assertGreater(srv.port, 0)
            finally:
                srv.stop()
        finally:
            s.close()

    def test_free_port_is_used_as_is(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
        s.close()
        srv = VizServer([], None, "127.0.0.1", free)
        try:
            self.assertEqual((srv.port, srv.port_changed), (free, False))
        finally:
            srv.stop()


if __name__ == "__main__":
    unittest.main()
