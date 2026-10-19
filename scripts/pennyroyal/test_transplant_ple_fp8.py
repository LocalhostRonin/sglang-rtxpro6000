import importlib.util
import json
import math
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

try:  # test-only: the real reader judges the files this tool writes and reads
    from safetensors import safe_open
except ImportError:
    safe_open = None

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


# The fixture writes files, so it keeps its own element sizes; the assembler's
# map is checked against these in RealReaderTest, and against the reader itself
# by the controls there.
TEST_ELEMENT_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
                     "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
                     "U16": 2, "U32": 4, "U64": 8, "BOOL": 1,
                     "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}


def packed(entries) -> dict[str, dict]:
    """Tensors laid out the way safetensors demands: packed in order, no gaps.

    Each entry is ``(name, dtype, shape)`` and the byte range follows from the
    dtype and shape, so a fixture that intends to break one invariant does not
    also break the file layout and mask its own assertion.
    """
    tensors = {}
    cursor = 0
    for name, dtype, shape in entries:
        nbytes = math.prod(shape) * TEST_ELEMENT_SIZE[dtype]
        tensors[name] = {"dtype": dtype, "shape": list(shape),
                         "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
    return tensors


def write_safetensors(path: Path, tensors: dict[str, dict]) -> None:
    """Write a minimal safetensors file: real header, dummy payload bytes.

    The data section is exactly the bytes the header claims, because the real
    reader rejects both a short one and one with undeclared bytes at the end.
    """
    body = json.dumps(dict(tensors), separators=(",", ":")).encode("utf-8")
    body += b" " * ((-len(body)) % 8)
    payload = max((meta["data_offsets"][1] for meta in tensors.values()), default=0)
    path.write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * payload)


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
    write_safetensors(directory / BASE_PLE_SHARD, packed([
        (NGRAM0, "BF16", (4, 4)), (NGRAM1, "BF16", (3, 4))]))
    keep = packed([(name, "F32", (2,)) for name in KEEP_LAYERS]
                  + [(name, "BF16", (2,)) for name in KEEP_PLE])
    write_safetensors(directory / BASE_KEEP_SHARD, keep)
    weight_map = {NGRAM0: BASE_PLE_SHARD, NGRAM1: BASE_PLE_SHARD}
    weight_map.update({name: BASE_KEEP_SHARD for name in KEEP_LAYERS + KEEP_PLE})
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 84}, "weight_map": weight_map}))
    (directory / "hf_quant_config.json").write_text("{}")


def build_ple_source(directory: Path) -> None:
    directory.mkdir(parents=True)
    write_safetensors(directory / FP8_SHARD, packed([
        (NGRAM0, "F8_E4M3", (4, 4)), (NGRAM1, "F8_E4M3", (3, 4)),
        (WEIGHT_SCALE, "BF16", (1,))]))
    for name in MODULE.LEADER_FILES:  # leader files ship from the FP8 side
        (directory / name).write_bytes(f"leader:{name}".encode("utf-8"))


class TransplantFixture(unittest.TestCase):
    """Shared fixture: a BF16-PLE base, an FP8 PLE source, and helpers."""

    # The donor layout the base geometry in base_config_payload() asks for.
    VALID = ((NGRAM0, "F8_E4M3", (4, 4)),
             (NGRAM1, "F8_E4M3", (3, 4)),
             (WEIGHT_SCALE, "BF16", (1,)))

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
        self.assertEqual(report["total_size"], 28 + 30)  # max data_offsets per claimed shard
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
        write_safetensors(self.ple / FP8_SHARD, packed([
            (NGRAM0, "F8_E4M3", (4, 4)), (WEIGHT_SCALE, "BF16", (1,)),
            ("model.language_model.layers.0.attn.qkv.weight", "F32", (2,))]))
        self.assertIn("outside the PLE module", self.refuse())

    def test_refuses_fp8_source_without_scale(self):
        write_safetensors(self.ple / FP8_SHARD, packed([
            (NGRAM0, "F8_E4M3", (4, 4)), (NGRAM1, "F8_E4M3", (3, 4))]))
        self.assertIn("weight_scale", self.refuse())

    def test_refuses_base_ngram_table_missing_from_source(self):
        # A base table outside the parts the donor covers: the alignment check
        # still runs once the donor itself satisfies the base's geometry.
        write_safetensors(self.base / BASE_PLE_SHARD, packed([
            (NGRAM0, "BF16", (4, 4)), (NGRAM2, "BF16", (3, 4))]))
        self.assertIn("tables the PLE source lacks", self.refuse())

    def test_refuses_ngram_part_missing_from_the_source(self):
        # The same misalignment seen from the donor side, caught by the part
        # coverage proof before any byte is written.
        write_safetensors(self.ple / FP8_SHARD, packed([
            (NGRAM0, "F8_E4M3", (4, 4)), (WEIGHT_SCALE, "BF16", (1,))]))
        self.assertIn("shard_1", self.refuse())

    def test_refuses_unaligned_dtype_shapes(self):
        write_safetensors(self.ple / FP8_SHARD, packed([
            (NGRAM0, "F8_E5M2", (4, 4)), (NGRAM1, "F8_E5M2", (3, 4)),
            (WEIGHT_SCALE, "BF16", (1,))]))
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
        write_safetensors(self.base / BASE_KEEP_SHARD, packed(
            [(name, "F32", (2,)) for name in KEEP_LAYERS]
            + [(name, "BF16", (2,)) for name in KEEP_PLE]
            + [(NGRAM1, "BF16", (4, 4))]))
        weight_map = json.loads(
            (self.base / "model.safetensors.index.json").read_text())["weight_map"]
        weight_map[NGRAM1] = BASE_KEEP_SHARD
        (self.base / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}))
        self.assertIn("claimed twice", self.refuse())

    def test_refuses_stale_ngram_in_kept_shard(self):
        # NGRAM1 lives ONLY in a kept shard: not a duplicate name, a
        # misaligned boundary. build_index must refuse to ship stale bytes.
        write_safetensors(self.base / BASE_PLE_SHARD, packed([(NGRAM0, "BF16", (4, 4))]))
        write_safetensors(self.base / BASE_KEEP_SHARD, packed(
            [(name, "F32", (2,)) for name in KEEP_LAYERS]
            + [(name, "BF16", (2,)) for name in KEEP_PLE]
            + [(NGRAM1, "BF16", (4, 4))]))
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

    def donor(self, *entries):
        write_safetensors(self.ple / FP8_SHARD, packed(entries or self.VALID))

    def refuse_donor(self, *entries):
        self.donor(*entries)
        return self.refuse()

    def prune_base(self):
        (self.base / BASE_PLE_SHARD).unlink()
        (self.base / "model.safetensors.index.json").unlink()

    def test_refuses_bf16_table_beside_fp8_tables(self):
        # Mixed BF16/FP8 donors were accepted: the BF16 bytes reach the
        # FP8-resident PLE path and the table reads as garbage at half length.
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (4, 4)),
                                    (NGRAM1, "BF16", (3, 4)),
                                    (WEIGHT_SCALE, "BF16", (1,)))
        self.assertIn("dtype BF16", message)

    def test_refuses_flat_ngram_table(self):
        # The loader only binds shard_N tables and raises on a flat
        # ngram_embedding.weight, whatever its dtype.
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (4, 4)),
                                    (NGRAM1, "F8_E4M3", (3, 4)),
                                    (NGRAM_FLAT, "F8_E4M3", (4, 4)),
                                    (WEIGHT_SCALE, "BF16", (1,)))
        self.assertIn("unsupported PLE weight layout", message)

    def test_refuses_scale_that_is_not_bound_to_the_tables(self):
        # A weight_scale on another module does not dequantize the ngram
        # tables: the loader only fills the embedding's own buffer.
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (4, 4)),
                                    (NGRAM1, "F8_E4M3", (3, 4)),
                                    (STRAY_SCALE, "BF16", (1,)))
        self.assertIn(WEIGHT_SCALE, message)

    def test_refuses_scale_that_is_not_one_element(self):
        # The loader compares the checkpoint scale against its own one-element
        # buffer and raises on any other shape; REFUSE here instead.
        message = self.refuse_donor(*self.VALID[:2], (WEIGHT_SCALE, "BF16", (2,)))
        self.assertIn("single element", message)

    def test_refuses_integer_scale(self):
        message = self.refuse_donor(*self.VALID[:2], (WEIGHT_SCALE, "I64", (1,)))
        self.assertIn("must be a float tensor", message)

    def test_refuses_rank_one_table(self):
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (16,)), *self.VALID[1:])
        self.assertIn("rank-2", message)

    def test_refuses_tables_of_differing_width(self):
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (4, 4)),
                                    (NGRAM1, "F8_E4M3", (3, 8)),
                                    (WEIGHT_SCALE, "BF16", (1,)))
        self.assertIn("columns", message)

    def test_refuses_table_bytes_that_disagree_with_its_shape(self):
        tensors = packed(self.VALID)
        tensors[NGRAM0]["data_offsets"] = [0, 8]
        write_safetensors(self.ple / FP8_SHARD, tensors)
        self.assertIn("8 bytes", self.refuse())

    def test_refuses_table_narrower_than_the_target_embedding(self):
        # 4 columns of target geometry, 1-column donor: torch copies by
        # broadcasting the single column across the row, no error, wrong table.
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (4, 1)),
                                    (NGRAM1, "F8_E4M3", (3, 1)),
                                    (WEIGHT_SCALE, "BF16", (1,)))
        self.assertIn("1 columns wide", message)

    def test_refuses_table_wider_than_the_target_embedding(self):
        # Consistent donor widths are not enough: the target is 4 columns, so
        # 2-column tables fail when they are copied into the embedding.
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (8, 2)),
                                    (NGRAM1, "F8_E4M3", (6, 2)),
                                    (WEIGHT_SCALE, "BF16", (1,)))
        self.assertIn("2 columns wide", message)

    def test_refuses_short_interior_part(self):
        # The loader places part i at i * ceil(rows/parts) and copies only the
        # rows the tensor carries, so 1 row in a 4-row part leaves rows 1..3
        # holding whatever the empty embedding had. Arbitrary short parts are
        # not valid; only the final one may be short.
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (1, 4)),
                                    (NGRAM1, "F8_E4M3", (3, 4)),
                                    (WEIGHT_SCALE, "BF16", (1,)))
        self.assertIn("part 0", message)
        self.assertIn("4 rows", message)
        self.assertFalse((self.root / "out").exists())

    def test_refuses_part_with_too_many_rows(self):
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (5, 4)), *self.VALID[1:])
        self.assertIn("part 0", message)

    def test_refuses_missing_final_part(self):
        # Pruned-base mode: the base's BF16 headers are gone, so a donor that
        # carries only part 0 would leave rows 4..6 of 7 unfilled.
        self.prune_base()
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (4, 4)),
                                    (WEIGHT_SCALE, "BF16", (1,)))
        self.assertIn("shard_1", message)
        self.assertFalse((self.root / "out").exists())
        self.assertFalse(list(self.root.glob("out.staging.*")))

    def test_refuses_duplicate_part_index(self):
        # shard_00 and shard_0 are different tensors, the same part.
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (4, 4)),
                                    (NGRAM00, "F8_E4M3", (3, 4)), *self.VALID[1:])
        self.assertIn("part 0", message)

    def test_refuses_part_index_beyond_the_split(self):
        message = self.refuse_donor(*self.VALID, (NGRAM2, "F8_E4M3", (3, 4)))
        self.assertIn("beyond the split", message)

    def test_refuses_tables_of_a_module_the_engine_does_not_build(self):
        # Valid tables and a valid scale under an invented module name: the
        # model creates ple_embedding.ngram_embedding and nothing else, so
        # these bytes would never be bound and the PLE stays BF16 garbage.
        message = self.refuse_donor((UNBOUND0, "F8_E4M3", (4, 4)),
                                    (UNBOUND1, "F8_E4M3", (3, 4)),
                                    (UNBOUND_SCALE, "BF16", (1,)))
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
        self.donor((NGRAM0, "F8_E4M3", (4, 4)), (NGRAM1, "F8_E4M3", (3, 4)),
                   (WEIGHT_SCALE, "F32", (1,)))
        self.assertEqual(self.assemble()["ple_tensors"], 3)

    def test_accepts_every_part_of_a_finer_split_including_an_empty_tail(self):
        # 7 rows over 8 parts: ceil(7/8)=1 row each, 7 full parts and an empty
        # tail that the loader copies nothing into, so it may be absent too.
        self.set_text_config(split_ngram_parts=8)
        parts = [(f"{PLE_MODULE}ple_embedding.ngram_embedding.shard_{i}.weight",
                  "F8_E4M3", (1, 4)) for i in range(7)]
        self.donor(*parts, (WEIGHT_SCALE, "BF16", (1,)), (NGRAM7, "F8_E4M3", (0, 4)))
        self.assertTrue(self.assemble()["files_copied"])
        self.donor(*parts, (WEIGHT_SCALE, "BF16", (1,)))
        self.assertTrue(self.assemble("out2")["files_copied"])

    def test_refuses_table_layout_without_base_ple_shards(self):
        # Pruned-base mode derives the same geometry from config.json alone.
        self.prune_base()
        message = self.refuse_donor((NGRAM0, "F8_E4M3", (4, 1)),
                                    (NGRAM1, "F8_E4M3", (3, 1)),
                                    (WEIGHT_SCALE, "BF16", (1,)))
        self.assertIn("1 columns wide", message)

    def test_assembles_pruned_base_with_matching_layout(self):
        self.prune_base()
        self.assertEqual(self.assemble()["ple_shards_transplanted"], 1)

    def set_text_config(self, **values):
        config = json.loads((self.base / "config.json").read_text())
        config["text_config"].update(values)
        for key, value in values.items():
            if value is None:
                del config["text_config"][key]
        (self.base / "config.json").write_text(json.dumps(config))
        return config

    def write_config(self, config: dict):
        (self.base / "config.json").write_text(json.dumps(config))

    def test_ignores_split_ngram_parts_outside_text_config(self):
        # Qwen4ExpConfig drops the top-level split_ngram_parts copy whenever a
        # nested text_config exists, so the loader never sees it and uses its
        # own default of 512: seven 1-row parts, not the 4+3 the raw dictionary
        # suggests. Reading the raw copy planned part 1 at row 4 while the
        # runtime puts it at row 1, overwriting rows and leaving the tail empty.
        for top in (2, 512, None):
            config = self.set_text_config(split_ngram_parts=None)
            if top is not None:
                config["split_ngram_parts"] = top
            self.write_config(config)
            message = self.refuse()
            self.assertIn("part 0", message)
            self.assertIn("1 rows", message)

    def test_nested_split_ngram_parts_is_authoritative(self):
        # The same top-level value beside a nested one changes nothing: the
        # donor's 4+3 parts are what the runtime will copy.
        config = self.set_text_config(split_ngram_parts=2)
        config["split_ngram_parts"] = 512
        self.write_config(config)
        self.assertTrue(self.assemble()["files_copied"])

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

    def declared(self, path: Path) -> int:
        blob = path.read_bytes()
        header_size = struct.unpack("<Q", blob[:8])[0]
        header = json.loads(blob[8:8 + header_size])
        return max(meta["data_offsets"][1] for name, meta in header.items()
                   if name != "__metadata__")

    def test_refuses_truncated_payload_before_staging(self):
        # The FP8 header declares 30 bytes of tensors (16 + 12 + 2), 24 on disk.
        declared = self.declared(self.ple / FP8_SHARD)
        self.assertEqual(declared, 30)
        self.truncate(self.ple / FP8_SHARD, 24)
        message = self.refuse()
        self.assertIn("truncated payload", message)
        self.assertIn(str(declared), message)
        self.assertFalse((self.root / "out").exists())
        self.assertFalse(list(self.root.glob("out.staging.*")))

    def test_refuses_truncated_base_shard(self):
        declared = self.declared(self.base / BASE_PLE_SHARD)  # 32 + 24
        self.truncate(self.base / BASE_PLE_SHARD, declared - 1)
        self.assertIn("truncated payload", self.refuse())

    def test_accepts_a_shard_that_is_exactly_its_declared_size(self):
        declared = self.declared(self.ple / FP8_SHARD)
        self.truncate(self.ple / FP8_SHARD, declared)  # no spare bytes at all
        self.assertTrue(self.assemble()["total_size"])


class TotalSizeTest(TransplantFixture):
    """A shard name coming from both parents must not be counted twice."""

    def test_stray_shard_named_like_a_base_shard_counts_once(self):
        # The donor directory also holds model-layer-00001.safetensors, which
        # the glob ignores and the copy never takes; summing the claimed names
        # in both parents still added its 8 bytes to the assembled index (58).
        write_safetensors(self.ple / BASE_KEEP_SHARD, packed(
            [("model.language_model.layers.9.attn.qkv.weight", "F32", (2,))]))
        self.assertEqual(self.assemble()["total_size"], 28 + 30)

    def test_refuses_shard_name_claimed_by_both_parents(self):
        # The same name with different contents: the PLE side's file overwrites
        # the base's copy, so the index would claim the base tensor from bytes
        # that never ship and total_size would add both.
        write_safetensors(self.base / FP8_SHARD, packed(
            [("model.language_model.layers.7.attn.qkv.weight", "F32", (2,))]))
        index_path = self.base / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text())["weight_map"]
        weight_map["model.language_model.layers.7.attn.qkv.weight"] = FP8_SHARD
        index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
        self.assertIn("is a shard of the base checkpoint and of the PLE source",
                      self.refuse())


@unittest.skipIf(safe_open is None, "safetensors (test-only) is not installed")
class RealReaderTest(TransplantFixture):
    """The real reader, not our own parser, judges the files we write."""

    def shapes_from_source(self, shard: str) -> dict:
        header = MODULE.read_safetensors_header(
            (self.ple if shard == FP8_SHARD else self.base) / shard)
        return {name: list(meta["shape"]) for name, meta in header.items()
                if name != "__metadata__"}

    def test_assembler_element_sizes_match_the_fixture(self):
        self.assertEqual(getattr(MODULE, "ELEMENT_SIZE", None), TEST_ELEMENT_SIZE)

    def test_assembled_shards_open_in_the_real_reader(self):
        self.assemble()
        index = json.loads(
            (self.root / "out" / "model.safetensors.index.json").read_text())
        for shard in sorted(set(index["weight_map"].values())):
            claimed = sorted(name for name, file in index["weight_map"].items()
                             if file == shard)
            with safe_open(self.root / "out" / shard, framework="np") as reader:
                self.assertEqual(sorted(reader.keys()), claimed)
                if shard == FP8_SHARD:  # the donor's tensors, unmodified
                    shapes = self.shapes_from_source(shard)
                    for name in claimed:
                        self.assertEqual(list(reader.get_slice(name).get_shape()),
                                         shapes[name])

    def test_refuses_tensors_that_overlap(self):
        # The safetensors reviewer's case: two FP8 tables claiming the same
        # bytes, with a scale after them and an exact 18-byte data section.
        tensors = {NGRAM0: {"dtype": "F8_E4M3", "shape": [4, 4],
                            "data_offsets": [0, 16]},
                   NGRAM1: {"dtype": "F8_E4M3", "shape": [3, 4],
                            "data_offsets": [0, 12]},
                   WEIGHT_SCALE: {"dtype": "BF16", "shape": [1],
                                  "data_offsets": [16, 18]}}
        write_safetensors(self.ple / FP8_SHARD, tensors)
        with self.assertRaises(Exception) as caught:
            safe_open(self.ple / FP8_SHARD, framework="np").keys()
        self.assertIn("invalid offset", str(caught.exception))
        self.assertIn("overlaps", self.refuse())
        self.assertFalse((self.root / "out").exists())

    def test_refuses_a_gap_between_tensors(self):
        tensors = packed(self.VALID)
        tensors[WEIGHT_SCALE]["data_offsets"] = [32, 34]
        write_safetensors(self.ple / FP8_SHARD, tensors)
        with self.assertRaises(Exception) as caught:
            safe_open(self.ple / FP8_SHARD, framework="np").keys()
        self.assertIn("invalid offset", str(caught.exception))
        self.assertIn("gap", self.refuse())

    def test_refuses_bytes_the_header_does_not_claim(self):
        blob = (self.ple / FP8_SHARD).read_bytes()
        (self.ple / FP8_SHARD).write_bytes(blob + b"\x00\x00")
        with self.assertRaises(Exception) as caught:
            safe_open(self.ple / FP8_SHARD, framework="np").keys()
        self.assertIn("not fully covered", str(caught.exception))
        self.assertIn("not claimed by any tensor", self.refuse())

    def test_refuses_a_tensor_entry_that_is_not_a_tensor(self):
        tensors = packed(self.VALID)
        tensors[NGRAM1]["dtype"] = "F16"  # 6 rows of 8 bytes, not 12
        write_safetensors(self.ple / FP8_SHARD, tensors)
        self.assertIn("needs 24", self.refuse())


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
