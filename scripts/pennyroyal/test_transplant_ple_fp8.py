import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("transplant_ple_fp8.py")
SPEC = importlib.util.spec_from_file_location("transplant_ple_fp8", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

PLE_MODULE = MODULE.PLE_MODULE
NGRAM0 = PLE_MODULE + "ple_embedding.ngram_embedding.shard_0.weight"
NGRAM1 = PLE_MODULE + "ple_embedding.ngram_embedding.shard_1.weight"
NGRAM_FLAT = PLE_MODULE + "ple_embedding.ngram_embedding.weight"
WEIGHT_SCALE = PLE_MODULE + "ple_embedding.ngram_embedding.weight_scale"
STRAY_SCALE = PLE_MODULE + "conv1d.weight_scale"
KEEP_PLE = [
    PLE_MODULE + "conv1d.weight",
    PLE_MODULE + "ple_embedding.layer_multipliers",
    PLE_MODULE + "ple_embedding.ngram_heads_offsets",
]
KEEP_LAYERS = ["model.language_model.layers.0.attn.qkv.weight",
               "model.language_model.embed_tokens.weight"]
BASE_PLE_SHARD = "model-plebf16-00000.safetensors"
BASE_KEEP_SHARD = "model-layer-00001.safetensors"
FP8_SHARD = "model-plefp8-00000.safetensors"


def write_safetensors(path: Path, tensors: dict[str, dict]) -> None:
    """Write a minimal safetensors file: real header, dummy payload bytes."""
    body = dict(tensors)
    blob = json.dumps(body, separators=(",", ":")).encode("utf-8")
    pad = (-len(blob)) % 8
    blob += b" " * pad
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 64)


def base_config_payload() -> dict:
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {"ple_embedding_dtype": "bfloat16", "hidden_size": 8},
        "quantization_config": {
            "quant_method": "modelopt",
            "ignore": ["lm_head", "model.language_model.embed_tokens",
                       PLE_MODULE + "*"],
        },
    }


def build_base(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(base_config_payload()))
    write_safetensors(directory / BASE_PLE_SHARD, {
        NGRAM0: {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]},
        NGRAM1: {"dtype": "BF16", "shape": [4, 4], "data_offsets": [32, 64]},
    })
    keep = {name: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
            for name in KEEP_LAYERS}
    keep.update({name: {"dtype": "BF16", "shape": [2], "data_offsets": [8, 12]}
                 for name in KEEP_PLE})
    write_safetensors(directory / BASE_KEEP_SHARD, keep)
    weight_map = {NGRAM0: BASE_PLE_SHARD, NGRAM1: BASE_PLE_SHARD}
    weight_map.update({name: BASE_KEEP_SHARD for name in KEEP_LAYERS + KEEP_PLE})
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 76}, "weight_map": weight_map}))
    (directory / "hf_quant_config.json").write_text("{}")


def build_ple_source(directory: Path) -> None:
    directory.mkdir(parents=True)
    write_safetensors(directory / FP8_SHARD, {
        NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]},
        NGRAM1: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [16, 32]},
        WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [32, 34]},
    })
    for name in MODULE.LEADER_FILES:  # leader files ship from the FP8 side
        (directory / name).write_bytes(f"leader:{name}".encode("utf-8"))


class TransplantFixture(unittest.TestCase):
    """Shared fixture: a BF16-PLE base, an FP8 PLE source, and helpers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base = self.root / "base"
        self.ple = self.root / "plesrc"
        build_base(self.base)
        build_ple_source(self.ple)
        self.addCleanup(self._tmp.cleanup)

    def assemble(self, out_name="out", **overrides):
        kwargs = dict(ple_glob="model-plefp8-*.safetensors", dry_run=False)
        kwargs.update(overrides)
        return MODULE.assemble(self.base, self.ple, self.root / out_name, **kwargs)

    def refuse(self, **overrides):
        with self.assertRaises(MODULE.Refuse) as caught:
            self.assemble(**overrides)
        return str(caught.exception)


class TransplantTest(TransplantFixture):

    def test_happy_path_writes_complete_checkpoint(self):
        report = self.assemble()
        out = self.root / "out"
        index = json.loads((out / "model.safetensors.index.json").read_text())
        config = json.loads((out / "config.json").read_text())
        names = set(index["weight_map"])
        self.assertEqual(config["text_config"]["ple_embedding_dtype"], "float8_e4m3fn")
        self.assertEqual(config["text_config"]["hidden_size"], 8)
        self.assertNotEqual((out / "config.json").read_bytes(),
                            (self.base / "config.json").read_bytes())
        # config.json is the ONLY mutated file: every copy is byte-identical
        for name in ("hf_quant_config.json", BASE_KEEP_SHARD):
            self.assertEqual((out / name).read_bytes(), (self.base / name).read_bytes())
        self.assertEqual((out / FP8_SHARD).read_bytes(), (self.ple / FP8_SHARD).read_bytes())
        for name in MODULE.LEADER_FILES:  # leaders come from the FP8 side
            self.assertEqual((out / name).read_bytes(), (self.ple / name).read_bytes())
        # ngram tables now come from the FP8 side; base PLE shard is gone
        self.assertEqual(index["weight_map"][NGRAM0], FP8_SHARD)
        self.assertEqual(index["weight_map"][NGRAM1], FP8_SHARD)
        self.assertNotIn(BASE_PLE_SHARD, index["weight_map"].values())
        # keep tensors preserved (layers + non-ngram PLE components)
        for name in KEEP_LAYERS + KEEP_PLE:
            self.assertEqual(index["weight_map"][name], BASE_KEEP_SHARD)
        self.assertEqual(names, set(KEEP_LAYERS + KEEP_PLE + [NGRAM0, NGRAM1, WEIGHT_SCALE]))
        self.assertEqual(report["tensors"], len(names))
        self.assertEqual(report["ple_tensors"], 3)
        self.assertEqual(report["total_size"], 12 + 34)  # max data_offsets per claimed shard
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(
            ["config.json", "hf_quant_config.json", "model.safetensors.index.json",
             BASE_KEEP_SHARD, FP8_SHARD, *MODULE.LEADER_FILES]))
        self.assertFalse(list(self.root.glob("out.staging.*")))

    def test_dry_run_writes_nothing(self):
        report = self.assemble(dry_run=True)
        self.assertTrue(report["dry_run"])
        self.assertFalse((self.root / "out").exists())

    def test_refuses_existing_out_dir(self):
        (self.root / "out").mkdir()
        self.assertIn("already exists", self.refuse())

    def test_refuses_base_already_fp8(self):
        config = json.loads((self.base / "config.json").read_text())
        config["text_config"]["ple_embedding_dtype"] = "float8_e4m3fn"
        (self.base / "config.json").write_text(json.dumps(config))
        self.assertIn("not a BF16-PLE", self.refuse())

    def test_assembles_from_pruned_base(self):
        # The production scenario: base dir with its pure-PLE shards and index
        # removed. Assembly succeeds; FP8 side supplies the ngram tables.
        (self.base / BASE_PLE_SHARD).unlink()
        (self.base / "model.safetensors.index.json").unlink()
        report = self.assemble()
        index = json.loads(
            (self.root / "out" / "model.safetensors.index.json").read_text())
        self.assertEqual(index["weight_map"][NGRAM0], FP8_SHARD)
        self.assertNotIn(BASE_PLE_SHARD, index["weight_map"].values())
        self.assertEqual(report["ple_shards_transplanted"], 1)

    def test_refuses_index_claiming_absent_shard(self):
        (self.base / BASE_PLE_SHARD).unlink()
        self.assertIn("file absent", self.refuse())

    def test_refuses_no_glob_match(self):
        self.assertIn("no shards matching", self.refuse(ple_glob="model-plefp8-z*.safetensors"))

    def test_refuses_dirty_ple_shard_outside_module(self):
        write_safetensors(self.ple / FP8_SHARD, {
            NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [16, 18]},
            "model.language_model.layers.0.attn.qkv.weight":
                {"dtype": "F32", "shape": [2], "data_offsets": [18, 26]},
        })
        self.assertIn("outside the PLE module", self.refuse())

    def test_refuses_fp8_source_without_scale(self):
        write_safetensors(self.ple / FP8_SHARD, {
            NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]},
        })
        self.assertIn("weight_scale", self.refuse())

    def test_refuses_base_ngram_table_missing_from_source(self):
        write_safetensors(self.ple / FP8_SHARD, {
            NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [16, 18]},
        })
        self.assertIn("tables the PLE source lacks", self.refuse())

    def test_refuses_unaligned_dtype_shapes(self):
        write_safetensors(self.ple / FP8_SHARD, {
            NGRAM0: {"dtype": "F8_E5M2", "shape": [4, 4], "data_offsets": [0, 16]},
            NGRAM1: {"dtype": "F8_E5M2", "shape": [4, 4], "data_offsets": [16, 32]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [32, 34]},
        })
        self.assertIn("dtype F8_E5M2", self.refuse())

    def test_refuses_missing_ignore_coverage(self):
        config = json.loads((self.base / "config.json").read_text())
        config["quantization_config"]["ignore"] = ["lm_head"]
        (self.base / "config.json").write_text(json.dumps(config))
        self.assertIn("not covered by quantization_config.ignore", self.refuse())

    def test_refuses_missing_quantization_config(self):
        config = json.loads((self.base / "config.json").read_text())
        del config["quantization_config"]
        (self.base / "config.json").write_text(json.dumps(config))
        self.assertIn("no quantization_config", self.refuse())

    def test_refuses_tensor_claimed_twice_across_sides(self):
        # NGRAM1 lives in the pure-PLE shard AND in a kept shard: the kept
        # copy is stale BF16 that would ship beside the FP8 bytes.
        write_safetensors(self.base / BASE_KEEP_SHARD, {
            **{name: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
               for name in KEEP_LAYERS + KEEP_PLE},
            NGRAM1: {"dtype": "BF16", "shape": [4, 4], "data_offsets": [8, 40]},
        })
        weight_map = json.loads(
            (self.base / "model.safetensors.index.json").read_text())["weight_map"]
        weight_map[NGRAM1] = BASE_KEEP_SHARD
        (self.base / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}))
        self.assertIn("claimed twice", self.refuse())

    def test_refuses_stale_ngram_in_kept_shard(self):
        # NGRAM1 lives ONLY in a kept shard: not a duplicate name, a
        # misaligned boundary. build_index must refuse to ship stale bytes.
        write_safetensors(self.base / BASE_PLE_SHARD, {
            NGRAM0: {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]},
        })
        write_safetensors(self.base / BASE_KEEP_SHARD, {
            **{name: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
               for name in KEEP_LAYERS + KEEP_PLE},
            NGRAM1: {"dtype": "BF16", "shape": [4, 4], "data_offsets": [8, 40]},
        })
        weight_map = json.loads(
            (self.base / "model.safetensors.index.json").read_text())["weight_map"]
        weight_map[NGRAM1] = BASE_KEEP_SHARD
        (self.base / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}))
        self.assertIn("not shard-aligned", self.refuse())

    def test_refuses_missing_leader_file(self):
        (self.ple / "tokenizer.json").unlink()
        self.assertIn("leader files", self.refuse())

    def test_refuses_truncated_header(self):
        (self.ple / FP8_SHARD).write_bytes(b"\xff\xff\xff\xff\xff\xff\xff\xff")
        self.assertIn("unreasonable header size", self.refuse())

    def test_base_ple_shards_is_pure_only(self):
        weight_map = json.loads(
            (self.base / "model.safetensors.index.json").read_text())["weight_map"]
        self.assertEqual(MODULE.base_ple_shards(weight_map), {BASE_PLE_SHARD})


class PLETableCensusTest(TransplantFixture):
    """Every ngram table is FP8, rank-2, and has its own module's scale."""

    def refuse_tables(self, **overrides):
        tensors = {
            NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]},
            NGRAM1: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [16, 32]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [32, 34]},
        }
        for name, value in overrides.items():
            if value is None:
                del tensors[name]
            else:
                tensors[name] = value
        write_safetensors(self.ple / FP8_SHARD, tensors)
        return self.refuse()

    def test_refuses_bf16_table_beside_fp8_tables(self):
        # Mixed BF16/FP8 donors were accepted: the BF16 bytes reach the
        # FP8-resident PLE path and the table reads as garbage at half length.
        message = self.refuse_tables(**{NGRAM1: {"dtype": "BF16", "shape": [2, 4],
                                                 "data_offsets": [16, 32]}})
        self.assertIn("dtype BF16", message)

    def test_refuses_flat_ngram_table(self):
        # The loader only accepts shard_N tables and raises on a flat
        # ngram_embedding.weight, whatever its dtype.
        message = self.refuse_tables(**{
            NGRAM_FLAT: {"dtype": "F8_E4M3", "shape": [4, 4],
                         "data_offsets": [32, 48]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [48, 50]},
        })
        self.assertIn("unsupported PLE weight layout", message)

    def test_refuses_scale_that_is_not_bound_to_the_tables(self):
        # A weight_scale on another module does not dequantize the ngram
        # tables: the loader only fills the embedding's own buffer.
        message = self.refuse_tables(**{WEIGHT_SCALE: None,
                                        STRAY_SCALE: {"dtype": "BF16", "shape": [1],
                                                      "data_offsets": [32, 34]}})
        self.assertIn(WEIGHT_SCALE, message)

    def test_refuses_scale_that_is_not_one_element(self):
        # The loader compares the checkpoint scale against its own one-element
        # buffer and raises on any other shape; REFUSE here instead.
        message = self.refuse_tables(**{WEIGHT_SCALE: {"dtype": "BF16", "shape": [2],
                                                       "data_offsets": [32, 36]}})
        self.assertIn("single element", message)

    def test_refuses_integer_scale(self):
        message = self.refuse_tables(**{WEIGHT_SCALE: {"dtype": "I64", "shape": [1],
                                                       "data_offsets": [32, 40]}})
        self.assertIn("must be a float tensor", message)

    def test_refuses_rank_one_table(self):
        message = self.refuse_tables(**{NGRAM0: {"dtype": "F8_E4M3", "shape": [16],
                                                 "data_offsets": [0, 16]}})
        self.assertIn("rank-2", message)

    def test_refuses_tables_of_differing_width(self):
        message = self.refuse_tables(**{NGRAM1: {"dtype": "F8_E4M3", "shape": [4, 8],
                                                 "data_offsets": [16, 48]}})
        self.assertIn("columns", message)

    def test_refuses_table_bytes_that_disagree_with_its_shape(self):
        message = self.refuse_tables(**{NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4],
                                                 "data_offsets": [0, 8]}})
        self.assertIn("8 bytes", message)

    def test_accepts_unequal_row_counts_and_other_float_scale_dtypes(self):
        # The last shard of a split table is legitimately shorter and any float
        # scale dtype reaches the loader's cast: neither may be refused.
        write_safetensors(self.ple / FP8_SHARD, {
            NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]},
            NGRAM1: {"dtype": "F8_E4M3", "shape": [2, 4], "data_offsets": [16, 24]},
            WEIGHT_SCALE: {"dtype": "F32", "shape": [1], "data_offsets": [24, 28]},
        })
        self.assertEqual(self.assemble()["ple_tensors"], 3)


class CliTest(unittest.TestCase):
    def test_main_reports_verdict(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        build_base(root / "base")
        build_ple_source(root / "plesrc")
        rc = MODULE.main([
            "--base-dir", str(root / "base"),
            "--ple-dir", str(root / "plesrc"),
            "--out-dir", str(root / "out"),
        ])
        self.assertEqual(rc, 0)
        self.assertTrue((root / "out" / "config.json").is_file())
        rc = MODULE.main([
            "--base-dir", str(root / "base"),
            "--ple-dir", str(root / "plesrc"),
            "--out-dir", str(root / "out"),
        ])
        self.assertEqual(rc, 1)  # second run: out-dir exists
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
