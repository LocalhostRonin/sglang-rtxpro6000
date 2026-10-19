import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("verify_ple_identity.py")
SPEC = importlib.util.spec_from_file_location("verify_ple_identity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

PLE = "model.language_model.layers.1.ple."
NGRAM0 = PLE + "ple_embedding.ngram_embedding.shard_0.weight"
NGRAM1 = PLE + "ple_embedding.ngram_embedding.shard_1.weight"
KEEP = PLE + "ple_embedding.layer_multipliers"
ATTN = "model.language_model.layers.0.attn.qkv.weight"


class PureFunctionTest(unittest.TestCase):
    def test_ple_tensor_names_filters_by_key(self):
        weight_map = {NGRAM0: "a.safetensors", KEEP: "a.safetensors", ATTN: "b.safetensors"}
        self.assertEqual(MODULE.ple_tensor_names(weight_map, ".ple."),
                         sorted([NGRAM0, KEEP]))

    def test_ple_tensor_names_custom_key(self):
        weight_map = {NGRAM0: "a", "other.ple.x": "a"}
        self.assertEqual(MODULE.ple_tensor_names(weight_map, "ngram"), [NGRAM0])

    def test_shards_proven_identical_splits_on_lfs_oid(self):
        tree_a = {
            "clean.safetensors": {"lfs": {"oid": "abc"}, "size": 10},
            "dirty.safetensors": {"lfs": {"oid": "ddd"}, "size": 20},
        }
        tree_b = {
            "clean.safetensors": {"lfs": {"oid": "abc"}, "size": 10},
            "dirty.safetensors": {"lfs": {"oid": "eee"}, "size": 20},
        }
        proven, needs = MODULE.shards_proven_identical(
            ["clean.safetensors", "dirty.safetensors"], tree_a, tree_b)
        self.assertEqual(proven, {"clean.safetensors": "abc"})
        self.assertEqual(needs, ["dirty.safetensors"])

    def test_shards_proven_identical_requires_matching_size(self):
        def tree(oid, size):
            return {"s": {"lfs": {"oid": oid}, "size": size}}
        proven, needs = MODULE.shards_proven_identical(
            ["s"], tree("abc", 10), tree("abc", 11))
        self.assertEqual(proven, {})
        self.assertEqual(needs, ["s"])

    def test_shards_without_lfs_metadata_need_range_hash(self):
        # Non-LFS files carry no oid: file-level proof is unavailable.
        entry = {"size": 500}
        proven, needs = MODULE.shards_proven_identical(["s"], {"s": entry}, {"s": entry})
        self.assertEqual(proven, {})
        self.assertEqual(needs, ["s"])

    def test_parse_safetensors_header_roundtrip(self):
        body = json.dumps({NGRAM0: {"dtype": "F8_E4M3", "shape": [2],
                                    "data_offsets": [0, 2]}}).encode("utf-8")
        raw = struct.pack("<Q", len(body)) + body
        header = MODULE.parse_safetensors_header(raw)
        self.assertEqual(header[NGRAM0]["dtype"], "F8_E4M3")

    def test_tensor_inventory_skips_metadata_and_non_ple(self):
        header = {
            "__metadata__": {"format": "pt"},
            NGRAM0: {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [0, 2]},
            KEEP: {"dtype": "BF16", "shape": [2], "data_offsets": [2, 6]},
            ATTN: {"dtype": "F32", "shape": [2], "data_offsets": [6, 14]},
        }
        inv = MODULE.tensor_inventory(header, ".ple.")
        self.assertEqual(set(inv), {NGRAM0, KEEP})
        self.assertEqual(inv[KEEP]["dtype"], "BF16")

    def test_dtype_size_covers_fp8(self):
        self.assertEqual(MODULE.DTYPE_SIZE["F8_E4M3"], 1)
        self.assertEqual(MODULE.DTYPE_SIZE["BF16"], 2)


class FakeHub:
    """Stand-in for the Hub: serves absolute byte ranges over in-memory files.

    HTTP Range offsets are absolute in the file, exactly as a real server (and
    huggingface.co) treats them, so a range built from safetensors
    ``data_offsets`` without the file's header size fetches the wrong bytes.
    """

    def __init__(self):
        self.blobs: dict[str, bytes] = {}
        self.docs: dict[str, object] = {}
        self.ranges: list[tuple[str, str]] = []
        self.doc_requests: list[str] = []

    def fetch(self, url: str, rng: str | None = None, timeout: int = 300) -> bytes:
        if url in self.docs:
            self.doc_requests.append(url)
            return json.dumps(self.docs[url]).encode("utf-8")
        blob = self.blobs[url]
        if rng is None:
            return blob
        self.ranges.append((url, rng))
        start, stop = (int(part) for part in rng.removeprefix("bytes=").split("-"))
        return blob[start:stop + 1]

    def fetch_json(self, cache_key: str, url: str) -> dict:
        return json.loads(self.fetch(url))

    def tensor_ranges(self) -> list[tuple[str, str]]:
        """Range requests beyond the two header probes of a shard."""
        return [(url, rng) for url, rng in self.ranges
                if not rng.startswith("bytes=0-")]


def hub_url(repo: str, rev: str, fname: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"


def shard_bytes(tensors: dict, payload: bytes, extra_pad: int = 0) -> bytes:
    """A real safetensors prefix: 8-byte size, 8-aligned header, then payload."""
    body = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    body += b" " * ((-len(body)) % 8)
    return struct.pack("<Q", len(body) + extra_pad) + body + b" " * extra_pad + payload


def data_start(blob: bytes) -> int:
    return 8 + struct.unpack("<Q", blob[:8])[0]


class RangeIdentityTest(unittest.TestCase):
    """End-to-end ``verify()`` over synthetic shards, asserting real ranges."""

    SHARD = "model-ple-00000.safetensors"
    REPO_A, REV_A = "Org/Base", "a" * 40
    REPO_B, REV_B = "Org/Swift", "b" * 40

    def setUp(self):
        self.hub = FakeHub()
        original = MODULE._Fetcher
        MODULE._Fetcher = lambda cache_dir: self.hub
        self.addCleanup(setattr, MODULE, "_Fetcher", original)
        weight_map = {NGRAM0: self.SHARD, ATTN: "model-00001.safetensors"}
        for repo, rev in ((self.REPO_A, self.REV_A), (self.REPO_B, self.REV_B)):
            self.hub.docs[f"https://huggingface.co/api/models/{repo}"] = {"sha": rev}
            self.hub.docs[hub_url(repo, rev, "model.safetensors.index.json")] = {
                "weight_map": weight_map}
            self.hub.docs[f"https://huggingface.co/api/models/{repo}/tree/{rev}"
                          "?recursive=true&blobs=true"] = self.tree(repo, rev)

    def tree(self, repo: str, rev: str) -> list[dict]:
        return [{"type": "file", "path": self.SHARD, "size": 100, "lfs": {"oid": None}}]

    def serve(self, repo: str, rev: str, blob: bytes, oid: str) -> None:
        self.hub.blobs[hub_url(repo, rev, self.SHARD)] = blob
        self.hub.docs[f"https://huggingface.co/api/models/{repo}/tree/{rev}"
                      "?recursive=true&blobs=true"][0]["lfs"]["oid"] = oid

    def run_verify(self):
        return MODULE.verify(self.REPO_A, self.REV_A, self.REPO_B, self.REV_B)

    def test_changed_payload_bytes_are_a_mismatch(self):
        header = {NGRAM0: {"dtype": "F8_E4M3", "shape": [8],
                           "data_offsets": [0, 8]}}
        self.serve(self.REPO_A, self.REV_A, shard_bytes(header, b"A" * 8), "oid-a")
        self.serve(self.REPO_B, self.REV_B, shard_bytes(header, b"B" * 8), "oid-b")
        report = self.run_verify()
        self.assertEqual(report["mismatches"], [NGRAM0])

    def test_changed_first_and_last_payload_bytes_are_a_mismatch(self):
        header = {NGRAM0: {"dtype": "F8_E4M3", "shape": [8],
                           "data_offsets": [0, 8]}}
        self.serve(self.REPO_A, self.REV_A, shard_bytes(header, b"A" * 7 + b"B"), "1")
        self.serve(self.REPO_B, self.REV_B, shard_bytes(header, b"A" + b"A" * 7), "2")
        self.assertEqual(self.run_verify()["mismatches"], [NGRAM0])
        self.serve(self.REPO_A, self.REV_A, shard_bytes(header, b"B" + b"A" * 7), "3")
        self.serve(self.REPO_B, self.REV_B, shard_bytes(header, b"A" * 8), "4")
        self.assertEqual(self.run_verify()["mismatches"], [NGRAM0])

    def test_tensor_range_uses_each_files_own_data_offset(self):
        # Same tensor bytes, same parsed header, different header padding:
        # the data sections start at different absolute offsets.
        header = {NGRAM0: {"dtype": "F8_E4M3", "shape": [8],
                           "data_offsets": [0, 8]}}
        blob_a = shard_bytes(header, b"S" * 8)
        blob_b = shard_bytes(header, b"S" * 8, extra_pad=48)
        self.assertNotEqual(data_start(blob_a), data_start(blob_b))
        self.assertEqual(MODULE.parse_safetensors_header(blob_a),
                         MODULE.parse_safetensors_header(blob_b))
        self.serve(self.REPO_A, self.REV_A, blob_a, "1")
        self.serve(self.REPO_B, self.REV_B, blob_b, "2")
        report = self.run_verify()
        self.assertEqual(report["mismatches"], [])
        self.assertEqual(
            sorted(self.hub.tensor_ranges()),
            [(hub_url(self.REPO_A, self.REV_A, self.SHARD),
              f"bytes={data_start(blob_a)}-{data_start(blob_a) + 7}"),
             (hub_url(self.REPO_B, self.REV_B, self.SHARD),
              f"bytes={data_start(blob_b)}-{data_start(blob_b) + 7}")])
        self.assertEqual(report["tensors"][NGRAM0]["byte_ranges"],
                         {"a": [data_start(blob_a), data_start(blob_a) + 8],
                          "b": [data_start(blob_b), data_start(blob_b) + 8]})

    def test_cli_exit_code_carries_the_verdict(self):
        # main() is the operator-facing proof: 0 = VERIFIED, 1 = NOT VERIFIED.
        header = {NGRAM0: {"dtype": "F8_E4M3", "shape": [8],
                           "data_offsets": [0, 8]}}
        argv = ["--repo-a", self.REPO_A, "--rev-a", self.REV_A,
                "--repo-b", self.REPO_B, "--rev-b", self.REV_B]
        self.serve(self.REPO_A, self.REV_A, shard_bytes(header, b"A" * 8), "1")
        self.serve(self.REPO_B, self.REV_B, shard_bytes(header, b"B" * 8), "2")
        self.assertEqual(MODULE.main(argv), 1)
        self.serve(self.REPO_B, self.REV_B, shard_bytes(header, b"A" * 8), "3")
        self.assertEqual(MODULE.main(argv), 0)

    def test_equal_shard_file_sha256_needs_no_tensor_range(self):
        header = {NGRAM0: {"dtype": "F8_E4M3", "shape": [8],
                           "data_offsets": [0, 8]}}
        blob = shard_bytes(header, b"S" * 8)
        self.serve(self.REPO_A, self.REV_A, blob, "same-oid")
        self.serve(self.REPO_B, self.REV_B, blob, "same-oid")
        report = self.run_verify()
        self.assertEqual(report["mismatches"], [])
        self.assertEqual(report["identical_via_file_sha256"], 1)
        self.assertEqual(self.hub.tensor_ranges(), [])


class MutableHeadCacheTest(unittest.TestCase):
    """The revision-drift check reads mutable metadata; it may not be cached."""

    SHARD = "model-ple-00000.safetensors"
    REPO_A, REPO_B = "Org/Base", "Org/Swift"
    REV_A1, REV_A2, REV_B = "a" * 40, "c" * 40, "b" * 40

    def setUp(self):
        self.hub = FakeHub()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fetcher = MODULE._Fetcher(Path(tmp.name))  # the real cache, fake transport
        fetcher.fetch = self.hub.fetch
        original = MODULE._Fetcher
        MODULE._Fetcher = lambda cache_dir: fetcher
        self.addCleanup(setattr, MODULE, "_Fetcher", original)
        blob = shard_bytes({NGRAM0: {"dtype": "F8_E4M3", "shape": [8],
                                     "data_offsets": [0, 8]}}, b"S" * 8)
        for repo, rev in ((self.REPO_A, self.REV_A1), (self.REPO_A, self.REV_A2),
                          (self.REPO_B, self.REV_B)):
            self.hub.blobs[hub_url(repo, rev, self.SHARD)] = blob
            self.hub.docs[hub_url(repo, rev, "model.safetensors.index.json")] = {
                "weight_map": {NGRAM0: self.SHARD}}
            self.hub.docs[f"https://huggingface.co/api/models/{repo}/tree/{rev}"
                          "?recursive=true&blobs=true"] = [
                {"type": "file", "path": self.SHARD, "size": 100,
                 "lfs": {"oid": f"{repo}@{rev[:1]}"}}]
        self.head = f"https://huggingface.co/api/models/{self.REPO_A}"
        self.hub.docs[self.head] = {"sha": self.REV_A1}
        self.hub.docs[f"https://huggingface.co/api/models/{self.REPO_B}"] = {
            "sha": self.REV_B}

    def argv(self, rev_a: str) -> list[str]:
        return ["--repo-a", self.REPO_A, "--rev-a", rev_a,
                "--repo-b", self.REPO_B, "--rev-b", self.REV_B]

    def test_advanced_head_is_not_reused_from_cache(self):
        self.assertEqual(MODULE.main(self.argv(self.REV_A1)), 0)
        # Org/Base gains a commit and the operator re-pins to it; the pinned
        # artifacts are unchanged, so only the mutable HEAD moved.
        self.hub.docs[self.head] = {"sha": self.REV_A2}
        self.assertEqual(MODULE.main(self.argv(self.REV_A2)), 0)
        self.assertEqual(MODULE.main(self.argv(self.REV_A2)), 0)
        # Pinned, immutable artifacts still come from the warm cache.
        self.assertEqual(
            self.hub.doc_requests.count(
                hub_url(self.REPO_A, self.REV_A2, "model.safetensors.index.json")), 1)


class CacheTest(unittest.TestCase):
    def test_fetcher_cache_writes_and_reuses(self):
        tmp = tempfile.TemporaryDirectory()
        cache = Path(tmp.name)
        fetcher = MODULE._Fetcher(cache)
        calls = []

        def fake_fetch(url, rng=None, timeout=300):
            calls.append(url)
            return b'{"sha": "x"}'

        fetcher.fetch = fake_fetch
        first = fetcher.fetch_json("k.json", "https://example/api")
        second = fetcher.fetch_json("k.json", "https://example/api")
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)  # second read came from disk
        self.assertTrue(list(cache.glob("k.json-*.json")))
        tmp.cleanup()

    def test_cache_identity_carries_the_url(self):
        # The same cache label serves both repos and both revisions of a run;
        # keying on the label alone reuses another repo's metadata, so a
        # re-run against a different pair reports a stale verdict.
        tmp = tempfile.TemporaryDirectory()
        fetcher = MODULE._Fetcher(Path(tmp.name))
        calls = []

        def fake_fetch(url, rng=None, timeout=300):
            calls.append(url)
            return json.dumps({"url": url}).encode("utf-8")

        fetcher.fetch = fake_fetch
        one = fetcher.fetch_json("a_meta.json", "https://huggingface.co/api/models/Org/One")
        two = fetcher.fetch_json("a_meta.json", "https://huggingface.co/api/models/Org/Two")
        self.assertEqual(one["url"], "https://huggingface.co/api/models/Org/One")
        self.assertEqual(two["url"], "https://huggingface.co/api/models/Org/Two")
        self.assertEqual(len(calls), 2)
        tmp.cleanup()

    def test_fetcher_without_cache_always_fetches(self):
        fetcher = MODULE._Fetcher(None)
        calls = []
        fetcher.fetch = lambda url, rng=None, timeout=300: calls.append(url) or b"{}"
        fetcher.fetch_json("k.json", "https://example/api")
        fetcher.fetch_json("k.json", "https://example/api")
        self.assertEqual(len(calls), 2)


class CliTest(unittest.TestCase):
    def test_arg_parser_requires_pins(self):
        parser = MODULE.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--repo-a", "x"])  # missing rev pins
        args = parser.parse_args([
            "--repo-a", "A/b", "--rev-a", "a" * 40,
            "--repo-b", "C/d", "--rev-b", "b" * 40])
        self.assertEqual(args.ple_key, MODULE.DEFAULT_PLE_KEY)


if __name__ == "__main__":
    unittest.main()
