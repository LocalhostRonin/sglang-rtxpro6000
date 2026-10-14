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
        self.assertTrue((cache / "k.json").is_file())
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
