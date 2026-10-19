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
NGRAM2 = PLE_MODULE + "ple_embedding.ngram_embedding.shard_2.weight"
NGRAM00 = PLE_MODULE + "ple_embedding.ngram_embedding.shard_00.weight"
NGRAM7 = PLE_MODULE + "ple_embedding.ngram_embedding.shard_7.weight"
NGRAM_FLAT = PLE_MODULE + "ple_embedding.ngram_embedding.weight"
WEIGHT_SCALE = PLE_MODULE + "ple_embedding.ngram_embedding.weight_scale"
STRAY_SCALE = PLE_MODULE + "conv1d.weight_scale"
UNBOUND0 = PLE_MODULE + "unknown_embedding.ngram_embedding.shard_0.weight"
UNBOUND1 = PLE_MODULE + "unknown_embedding.ngram_embedding.shard_1.weight"
UNBOUND_SCALE = PLE_MODULE + "unknown_embedding.ngram_embedding.weight_scale"
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
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 128)


def base_config_payload() -> dict:
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        # A real, tiny Qwen4-Exp geometry. One ngram head ((2-1)*1) over
        # ple_embed_dim 4: tables are 4 columns wide. The head vocabulary is
        # nextprime(7-1)=7 rows, divisible_by 1 leaves 7, and split_ngram_parts
        # 2 splits it into ceil(7/2)=4 rows per part, so the loader fills part
        # 0 with 4 rows and the short final part 1 with 3. ple_layer_ids [2]
        # enables PLE at checkpoint layer 1, the layer this recipe transplants.
        "text_config": {"ple_embedding_dtype": "bfloat16", "hidden_size": 8,
                        "ple_layer_ids": [2], "ple_embed_dim": 4,
                        "ngram_size": 2, "heads_per_ngram": 1,
                        "ngram_vocab_size_base": 7,
                        "make_ngram_vocab_size_divisible_by": 1,
                        "split_ngram_parts": 2},
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
        NGRAM1: {"dtype": "BF16", "shape": [3, 4], "data_offsets": [32, 56]},
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
        NGRAM1: {"dtype": "F8_E4M3", "shape": [3, 4], "data_offsets": [16, 28]},
        WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [28, 30]},
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
        self.assertEqual(report["total_size"], 12 + 30)  # max data_offsets per claimed shard
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
            NGRAM1: {"dtype": "F8_E4M3", "shape": [3, 4], "data_offsets": [16, 28]},
        })
        self.assertIn("weight_scale", self.refuse())

    def test_refuses_base_ngram_table_missing_from_source(self):
        # A base table outside the parts the donor covers: the alignment check
        # still runs once the donor itself satisfies the base's geometry.
        write_safetensors(self.base / BASE_PLE_SHARD, {
            NGRAM0: {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]},
            NGRAM2: {"dtype": "BF16", "shape": [3, 4], "data_offsets": [32, 56]},
        })
        self.assertIn("tables the PLE source lacks", self.refuse())

    def test_refuses_ngram_part_missing_from_the_source(self):
        # The same misalignment seen from the donor side, caught by the part
        # coverage proof before any byte is written.
        write_safetensors(self.ple / FP8_SHARD, {
            NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [16, 18]},
        })
        self.assertIn("shard_1", self.refuse())

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
    """Every ngram table must match the layout the loader will build."""

    def refuse_tables(self, **overrides):
        tensors = {
            NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]},
            NGRAM1: {"dtype": "F8_E4M3", "shape": [3, 4], "data_offsets": [16, 28]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [28, 30]},
        }
        for name, value in overrides.items():
            if value is None:
                del tensors[name]
            else:
                tensors[name] = value
        write_safetensors(self.ple / FP8_SHARD, tensors)
        return self.refuse()

    def prune_base(self):
        (self.base / BASE_PLE_SHARD).unlink()
        (self.base / "model.safetensors.index.json").unlink()

    def test_refuses_bf16_table_beside_fp8_tables(self):
        # Mixed BF16/FP8 donors were accepted: the BF16 bytes reach the
        # FP8-resident PLE path and the table reads as garbage at half length.
        message = self.refuse_tables(**{NGRAM1: {"dtype": "BF16", "shape": [3, 4],
                                                 "data_offsets": [16, 28]}})
        self.assertIn("dtype BF16", message)

    def test_refuses_flat_ngram_table(self):
        # The loader only binds shard_N tables and raises on a flat
        # ngram_embedding.weight, whatever its dtype.
        message = self.refuse_tables(**{
            NGRAM_FLAT: {"dtype": "F8_E4M3", "shape": [4, 4],
                         "data_offsets": [30, 46]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [46, 48]},
        })
        self.assertIn("unsupported PLE weight layout", message)

    def test_refuses_scale_that_is_not_bound_to_the_tables(self):
        # A weight_scale on another module does not dequantize the ngram
        # tables: the loader only fills the embedding's own buffer.
        message = self.refuse_tables(**{WEIGHT_SCALE: None,
                                        STRAY_SCALE: {"dtype": "BF16", "shape": [1],
                                                      "data_offsets": [28, 30]}})
        self.assertIn(WEIGHT_SCALE, message)

    def test_refuses_scale_that_is_not_one_element(self):
        # The loader compares the checkpoint scale against its own one-element
        # buffer and raises on any other shape; REFUSE here instead.
        message = self.refuse_tables(**{WEIGHT_SCALE: {"dtype": "BF16", "shape": [2],
                                                       "data_offsets": [28, 32]}})
        self.assertIn("single element", message)

    def test_refuses_integer_scale(self):
        message = self.refuse_tables(**{WEIGHT_SCALE: {"dtype": "I64", "shape": [1],
                                                       "data_offsets": [28, 36]}})
        self.assertIn("must be a float tensor", message)

    def test_refuses_rank_one_table(self):
        message = self.refuse_tables(**{NGRAM0: {"dtype": "F8_E4M3", "shape": [16],
                                                 "data_offsets": [0, 16]}})
        self.assertIn("rank-2", message)

    def test_refuses_tables_of_differing_width(self):
        message = self.refuse_tables(**{NGRAM1: {"dtype": "F8_E4M3", "shape": [3, 8],
                                                 "data_offsets": [16, 40]}})
        self.assertIn("columns", message)

    def test_refuses_table_bytes_that_disagree_with_its_shape(self):
        message = self.refuse_tables(**{NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4],
                                                 "data_offsets": [0, 8]}})
        self.assertIn("8 bytes", message)

    def test_refuses_table_narrower_than_the_target_embedding(self):
        # 4 columns of target geometry, 1-column donor: torch copies by
        # broadcasting the single column across the row, no error, wrong table.
        message = self.refuse_tables(**{
            NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 1], "data_offsets": [0, 4]},
            NGRAM1: {"dtype": "F8_E4M3", "shape": [3, 1], "data_offsets": [4, 7]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [7, 9]},
        })
        self.assertIn("1 columns wide", message)

    def test_refuses_table_wider_than_the_target_embedding(self):
        # Consistent donor widths are not enough: the target is 4 columns, so
        # 2-column tables fail when they are copied into the embedding.
        message = self.refuse_tables(**{
            NGRAM0: {"dtype": "F8_E4M3", "shape": [8, 2], "data_offsets": [0, 16]},
            NGRAM1: {"dtype": "F8_E4M3", "shape": [6, 2], "data_offsets": [16, 28]},
        })
        self.assertIn("2 columns wide", message)

    def test_refuses_short_interior_part(self):
        # The loader places part i at i * ceil(rows/parts) and copies only the
        # rows the tensor carries, so 1 row in a 4-row part leaves rows 1..3
        # holding whatever the empty embedding had. Arbitrary short parts are
        # not valid; only the final one may be short.
        message = self.refuse_tables(**{
            NGRAM0: {"dtype": "F8_E4M3", "shape": [1, 4], "data_offsets": [0, 4]},
            NGRAM1: {"dtype": "F8_E4M3", "shape": [3, 4], "data_offsets": [4, 16]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [16, 18]},
        })
        self.assertIn("part 0", message)
        self.assertIn("4 rows", message)
        self.assertFalse((self.root / "out").exists())

    def test_refuses_part_with_too_many_rows(self):
        message = self.refuse_tables(**{
            NGRAM0: {"dtype": "F8_E4M3", "shape": [5, 4], "data_offsets": [0, 20]},
            NGRAM1: {"dtype": "F8_E4M3", "shape": [3, 4], "data_offsets": [20, 32]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [32, 34]},
        })
        self.assertIn("part 0", message)

    def test_refuses_missing_final_part(self):
        # Pruned-base mode: the base's BF16 headers are gone, so a donor that
        # carries only part 0 would leave rows 4..6 of 7 unfilled.
        self.prune_base()
        message = self.refuse_tables(**{NGRAM1: None})
        self.assertIn("shard_1", message)
        self.assertFalse((self.root / "out").exists())
        self.assertFalse(list(self.root.glob("out.staging.*")))

    def test_refuses_duplicate_part_index(self):
        # shard_00 and shard_0 are different tensors, the same part.
        message = self.refuse_tables(**{
            NGRAM00: {"dtype": "F8_E4M3", "shape": [3, 4], "data_offsets": [28, 40]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [40, 42]},
        })
        self.assertIn("part 0", message)

    def test_refuses_part_index_beyond_the_split(self):
        message = self.refuse_tables(**{
            NGRAM2: {"dtype": "F8_E4M3", "shape": [3, 4], "data_offsets": [28, 40]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [40, 42]},
        })
        self.assertIn("beyond the split", message)

    def test_refuses_tables_of_a_module_the_engine_does_not_build(self):
        # Valid tables and a valid scale under an invented module name: the
        # model creates ple_embedding.ngram_embedding and nothing else, so
        # these bytes would never be bound and the PLE stays BF16 garbage.
        message = self.refuse_tables(**{
            NGRAM0: None, NGRAM1: None, WEIGHT_SCALE: None,
            UNBOUND0: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]},
            UNBOUND1: {"dtype": "F8_E4M3", "shape": [3, 4], "data_offsets": [16, 28]},
            UNBOUND_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [28, 30]},
        })
        self.assertIn("unknown_embedding", message)
        self.assertIn("ple_embedding", message)
        self.assertFalse((self.root / "out").exists())

    def test_refuses_base_without_the_documented_single_ple_layer(self):
        # The recipe transplants the one PLE layer it documents; a base with a
        # second one, or with PLE at another checkpoint layer, is not it.
        for ids in ([2, 34], [1], []):
            self.set_text_config(ple_layer_ids=ids)
            self.assertIn("ple_layer_ids", self.refuse())

    def test_accepts_the_configured_parts_with_a_short_final_part(self):
        # Control: 4 + 3 rows of the 7-row target, part 1 legitimately short,
        # and a float scale dtype other than BF16 reaches the loader's cast.
        write_safetensors(self.ple / FP8_SHARD, {
            NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]},
            NGRAM1: {"dtype": "F8_E4M3", "shape": [3, 4], "data_offsets": [16, 28]},
            WEIGHT_SCALE: {"dtype": "F32", "shape": [1], "data_offsets": [28, 32]},
        })
        self.assertEqual(self.assemble()["ple_tensors"], 3)

    def test_accepts_every_part_of_a_finer_split_including_an_empty_tail(self):
        # 7 rows over 8 parts: ceil(7/8)=1 row each, 7 full parts and an empty
        # tail that the loader copies nothing into, so it may be absent too.
        self.set_text_config(split_ngram_parts=8)
        tensors = {f"{PLE_MODULE}ple_embedding.ngram_embedding.shard_{i}.weight":
                   {"dtype": "F8_E4M3", "shape": [1, 4],
                    "data_offsets": [4 * i, 4 * i + 4]} for i in range(7)}
        tensors[WEIGHT_SCALE] = {"dtype": "BF16", "shape": [1],
                                 "data_offsets": [28, 30]}
        write_safetensors(self.ple / FP8_SHARD, tensors)
        self.assertTrue(self.assemble()["files_copied"])
        tensors[NGRAM7] = {"dtype": "F8_E4M3", "shape": [0, 4],
                           "data_offsets": [30, 30]}
        write_safetensors(self.ple / FP8_SHARD, tensors)
        self.assertTrue(self.assemble("out2")["files_copied"])

    def test_refuses_table_layout_without_base_ple_shards(self):
        # Pruned-base mode derives the same geometry from config.json alone.
        self.prune_base()
        message = self.refuse_tables(**{
            NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 1], "data_offsets": [0, 4]},
            NGRAM1: {"dtype": "F8_E4M3", "shape": [3, 1], "data_offsets": [4, 7]},
            WEIGHT_SCALE: {"dtype": "BF16", "shape": [1], "data_offsets": [7, 9]},
        })
        self.assertIn("1 columns wide", message)

    def test_assembles_pruned_base_with_matching_layout(self):
        self.prune_base()
        self.assertEqual(self.assemble()["ple_shards_transplanted"], 1)

    def set_text_config(self, **values):
        config = json.loads((self.base / "config.json").read_text())
        config["text_config"].update(values)
        for key in ("hidden_size", "ple_embed_dim"):
            if key in values and values[key] is None:
                del config["text_config"][key]
        (self.base / "config.json").write_text(json.dumps(config))

    def test_refuses_base_without_usable_ple_geometry(self):
        self.set_text_config(ple_embed_dim=None, hidden_size=None)
        self.assertIn("geometry cannot be established", self.refuse())

    def test_refuses_base_geometry_the_loader_itself_rejects(self):
        # 6 columns over 4 heads: the engine raises on the indivisible config.
        self.set_text_config(ple_embed_dim=6, ngram_size=3, heads_per_ngram=2)
        self.assertIn("not divisible", self.refuse())
        self.set_text_config(ple_embed_dim=4, ngram_size=1)
        self.assertIn("ngram_size", self.refuse())


class TruncationTest(TransplantFixture):
    """A complete header is not a complete file: the payload must be there."""

    def truncate(self, path: Path, keep: int) -> None:
        blob = path.read_bytes()
        header_size = struct.unpack("<Q", blob[:8])[0]
        path.write_bytes(blob[:8 + header_size + keep])

    def test_refuses_truncated_payload_before_staging(self):
        # The FP8 header declares 30 bytes of tensors (16 + 12 + 2), 24 on disk.
        self.truncate(self.ple / FP8_SHARD, 24)
        message = self.refuse()
        self.assertIn("truncated payload", message)
        self.assertIn("30", message)
        self.assertFalse((self.root / "out").exists())
        self.assertFalse(list(self.root.glob("out.staging.*")))

    def test_refuses_truncated_base_shard(self):
        self.truncate(self.base / BASE_PLE_SHARD, 55)  # declares 32 + 24
        self.assertIn("truncated payload", self.refuse())

    def test_accepts_a_shard_that_is_exactly_its_declared_size(self):
        self.truncate(self.ple / FP8_SHARD, 30)  # no spare padding, still whole
        self.assertTrue(self.assemble()["total_size"])


class TotalSizeTest(TransplantFixture):
    """A shard name coming from both parents must not be counted twice."""

    def test_stray_shard_named_like_a_base_shard_counts_once(self):
        # The donor directory also holds model-layer-00001.safetensors, which
        # the glob ignores and the copy never takes; summing the claimed names
        # in both parents still added its 8 bytes to the assembled index (42).
        write_safetensors(self.ple / BASE_KEEP_SHARD, {
            "model.language_model.layers.9.attn.qkv.weight":
                {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        })
        self.assertEqual(self.assemble()["total_size"], 12 + 30)

    def test_refuses_shard_name_claimed_by_both_parents(self):
        # The same name with different contents: the PLE side's file overwrites
        # the base's copy, so the index would claim the base tensor from bytes
        # that never ship and total_size would add both.
        write_safetensors(self.base / FP8_SHARD, {
            "model.language_model.layers.7.attn.qkv.weight":
                {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        })
        index_path = self.base / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text())["weight_map"]
        weight_map["model.language_model.layers.7.attn.qkv.weight"] = FP8_SHARD
        index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
        self.assertIn("is a shard of the base checkpoint and of the PLE source",
                      self.refuse())


class CliTest(unittest.TestCase):
    def test_main_refuses_a_missing_input_directory(self):
        # A typo in --base-dir is the commonest operator error and must print
        # REFUSE and exit 1, not unwind a traceback past the argument check.
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        build_base(root / "base")
        build_ple_source(root / "plesrc")
        rc = MODULE.main([
            "--base-dir", str(root / "gone"),
            "--ple-dir", str(root / "plesrc"),
            "--out-dir", str(root / "out"),
        ])
        self.assertEqual(rc, 1)
        self.assertFalse((root / "out").exists())
        tmp.cleanup()

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
