#!/usr/bin/env python3
"""Assemble a checkpoint that serves FP8 PLE tables from one config patch.

Transplants the FP8 PLE shards of a source checkpoint (e.g.
RadixArk/Qwen3.8-Flash-Next-NVFP4, which ships model-plefp8-*.safetensors)
into a base checkpoint whose PLE is BF16 (e.g. ukisai/Swift-1.5-Qwen3.8-
Flash-Next-NVFP4), and flips ``text_config.ple_embedding_dtype`` to
``float8_e4m3fn`` so the engine's FP8-resident PLE path accepts the result.

No weight math happens here. Every shard is copied byte-for-byte from a
parent; the only files this tool creates are the merged safetensors index and
the patched config. All validation runs before a single byte is written, and
the output is staged in a temp directory next to the destination, then moved
into place atomically. An existing destination is never touched.

The transplant is sound when the two parents' PLE tables descend from the
same BF16 bytes; verify that first with verify_ple_identity.py. This tool
checks structure (shard alignment, dtype census, quantization ignore-list
coverage), not quantization quality.

Usage:
    python3 transplant_ple_fp8.py \
        --base-dir /path/to/Swift-1.5-Qwen3.8-Flash-Next-NVFP4 \
        --ple-dir  /path/to/Qwen3.8-Flash-Next-NVFP4 \
        --out-dir  /path/to/assembled-checkpoint \
        [--ple-glob 'model-plefp8-*.safetensors'] [--dry-run]

Exit code 0 = assembled (or validated, with --dry-run); any refusal exits 1
with a REFUSE line naming the failed invariant.

Uses only the standard library.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

PLE_LAYER = 1  # the one PLE layer this recipe documents and transplants
PLE_MODULE = f"model.language_model.layers.{PLE_LAYER}.ple."
PLE_MODULE_TAIL = PLE_MODULE + "ple_embedding"  # the module the engine builds
PLE_CONFIG_KEY = "ple_embedding_dtype"
FP8_PLE_DTYPE = "float8_e4m3fn"
INDEX_NAME = "model.safetensors.index.json"
HEADER_SIZE_MAX = 128 * 1024 * 1024
NGRAM_MARKER = ".ngram_embedding."
# The engine binds exactly one weight per ngram table: <module>.ngram_embedding
# .shard_<N>.weight (models/qwen4_exp.py raises on any other PLE weight).
NGRAM_TABLE = re.compile(r"^shard_(?P<part>\d+)\.weight$")
NGRAM_SCALE = "weight_scale"
# Qwen4ExpTextConfig defaults; the engine derives the table layout from them.
NGRAM_SIZE_DEFAULT = 3
HEADS_PER_NGRAM_DEFAULT = 8
VOCAB_BASE_DEFAULT = 20000000
VOCAB_DIVISIBLE_BY_DEFAULT = 128
SPLIT_PARTS_DEFAULT = 512
# Element sizes for every dtype the safetensors format defines (verified against
# the reader), used to prove a declared shape matches the bytes its range holds.
ELEMENT_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
                "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
                "U16": 2, "U32": 4, "U64": 8, "BOOL": 1,
                "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}
FP8_TABLE_DTYPE = "F8_E4M3"
ALLOWED_SCALES = ("F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2")
# Non-shard files the assembled checkpoint takes from the FP8 PLE source
# (tokenizer/serialization assets); config and index are generated here.
LEADER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


class Refuse(Exception):
    """A failed invariant; the assembler writes nothing when it is raised."""


def refuse(why: str) -> None:
    raise Refuse(why)


def read_safetensors_header(path: Path) -> dict:
    """Parse a safetensors file's tensor metadata (names, dtypes, offsets) and
    prove the data section it describes is complete and correctly laid out.

    These are the reader's own rules, checked here so a bad shard is refused
    before staging instead of halfway through a model load, and so the payload
    proof cannot be satisfied by a complete header over a truncated download:
      * every entry is a tensor: a known dtype, a shape, and a byte range of
        exactly prod(shape) * itemsize bytes (0 bytes for an empty tensor);
      * the ranges pack the data section in order, no overlaps and no gaps;
      * the file ends where the last tensor ends, so nothing is truncated and
        no undeclared bytes ride along.
    """
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            refuse(f"{path.name}: not a safetensors file (no 8-byte header size)")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size > HEADER_SIZE_MAX:
            refuse(f"{path.name}: unreasonable header size {header_size}")
        raw = stream.read(header_size)
    if len(raw) != header_size:
        refuse(f"{path.name}: truncated header")
    header = json.loads(raw)
    ranges = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta.get("dtype") if isinstance(meta, dict) else None
        if dtype not in ELEMENT_SIZE:
            refuse(f"{path.name}: tensor {name} is not a tensor entry the format "
                   f"defines (dtype {dtype!r})")
        shape, offsets = meta.get("shape"), meta.get("data_offsets")
        if (not isinstance(shape, list)
                or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(value, int) and value >= 0 for value in offsets)
                or offsets[0] > offsets[1]):
            refuse(f"{path.name}: tensor {name} has a malformed shape or "
                   f"data_offsets: {meta}")
        needed = math.prod(shape) * ELEMENT_SIZE[dtype]
        if offsets[1] - offsets[0] != needed:
            refuse(f"{path.name}: tensor {name} reserves {offsets[1] - offsets[0]} "
                   f"bytes for shape {shape} of {dtype}, which needs {needed}")
        ranges.append((offsets[0], offsets[1], name))
    cursor = 0
    for begin, end, name in sorted(ranges):
        if begin != cursor:
            relation = ("overlaps" if begin < cursor
                        else "leaves a gap after the bytes already claimed")
            refuse(f"{path.name}: tensor {name} {relation}: safetensors packs the "
                   "tensors it declares in offset order")
        cursor = end
    present = path.stat().st_size - (8 + header_size)
    if present < cursor:
        refuse(f"{path.name}: truncated payload, the header declares {cursor} "
               f"bytes of tensors and {max(present, 0)} are on disk")
    if present > cursor:
        refuse(f"{path.name}: {present - cursor} bytes at the end of the data "
               "section are not claimed by any tensor")
    return header


def load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        refuse(f"{label} missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        refuse(f"{label} is not valid JSON ({path}): {exc}")
    return {}  # unreachable; refuse exits


def is_ngram_tensor(name: str) -> bool:
    return name.startswith(PLE_MODULE) and NGRAM_MARKER in name


def validate_base(base_dir: Path) -> tuple[dict, dict[str, str]]:
    """Check the base checkpoint: config wants BF16 PLE, and the on-disk
    shards carry the PLE modules. The weight map is scanned from the shard
    headers, so a pruned base (index deleted with dropped shards) assembles
    the same as a full download. When an index is present it must agree."""
    config = load_json(base_dir / "config.json", "base config.json")
    text_config = config.get("text_config")
    if text_config is None:
        refuse("base config.json has no text_config; this tool targets the "
               "Qwen3.8-Flash-Next (qwen4_exp) family")
    dtype = text_config.get(PLE_CONFIG_KEY)
    if dtype == FP8_PLE_DTYPE:
        refuse(f"base config.json already declares {PLE_CONFIG_KEY}={FP8_PLE_DTYPE}; "
               "the base is not a BF16-PLE checkpoint")
    if dtype not in (None, "bfloat16"):
        refuse(f"base {PLE_CONFIG_KEY}={dtype!r}; expected absent or 'bfloat16' "
               "(BF16) for a transplant source")

    base_weight_map: dict[str, str] = {}
    shards = sorted(p.name for p in base_dir.glob("*.safetensors"))
    if not shards:
        refuse(f"no *.safetensors shards found in {base_dir}")
    for shard in shards:
        header = read_safetensors_header(base_dir / shard)
        for name in header:
            if name == "__metadata__":
                continue
            if name in base_weight_map:
                refuse(f"base tensor {name} claimed twice (in "
                       f"{base_weight_map[name]} and {shard})")
            base_weight_map[name] = shard
    if not any(name.startswith(PLE_MODULE) for name in base_weight_map):
        refuse(f"base shards contain no tensors under {PLE_MODULE}; "
               "this is not the expected architecture")
    if (base_dir / INDEX_NAME).is_file():
        claimed = set(load_json(base_dir / INDEX_NAME, "base index")
                      .get("weight_map", {}).values())
        on_disk = set(shards)
        if claimed - on_disk:
            refuse(f"base index claims {sorted(claimed - on_disk)[0]}, "
                   f"file absent in {base_dir}")
        if on_disk - claimed:
            refuse(f"base shard {sorted(on_disk - claimed)[0]} not claimed "
                   f"by the base index in {base_dir}")
    return config, base_weight_map


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _nth_prime_after(start: int, count: int) -> int:
    """``sympy.nextprime`` applied ``count`` times, as the engine's helper does."""
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def ple_target(config: dict) -> dict:
    """The layout the engine builds for the PLE layer this recipe targets.

    models/qwen4_exp.py sizes the ngram embedding from the base config alone:
    each head table is ``ple_embed_dim // ((ngram_size - 1) * heads_per_ngram)``
    columns wide, the row count comes from ``_build_head_vocab_and_offsets``
    (one prime per head, from ``ngram_vocab_size_base``) padded up to
    ``make_ngram_vocab_size_divisible_by``, and ``load_weights`` copies the
    tensor named ``shard_N.weight`` into rows ``N * ceil(rows / parts)`` onward,
    advancing only by the rows that tensor carries. Only the single documented
    PLE layer is supported, so its ``ple_layer_index`` is 0 and the head primes
    are the first ``ngram_heads`` after the base. Defaults follow
    Qwen4ExpTextConfig, and only its nested text_config is authoritative here:
    Qwen4ExpConfig drops a top-level split_ngram_parts copy as soon as a nested
    text_config exists, and a base without one is refused for having no
    text_config at all, so the loader's top-level fallback can never apply.
    """
    text = config["text_config"]
    ids = text.get("ple_layer_ids")
    if ids != [PLE_LAYER + 1]:
        refuse(f"base text_config.ple_layer_ids={ids!r}; this recipe transplants one "
               f"PLE layer, the one it documents, enabled by ple_layer_ids="
               f"[{PLE_LAYER + 1}] for {PLE_MODULE}*")
    dim = text.get("ple_embed_dim") or text.get("hidden_size")
    if not isinstance(dim, int) or dim <= 0:
        refuse("base config.json gives no usable ple_embed_dim or hidden_size; the "
               "PLE target geometry cannot be established")
    ngram_size = text.get("ngram_size", NGRAM_SIZE_DEFAULT)
    heads = text.get("heads_per_ngram", HEADS_PER_NGRAM_DEFAULT)
    if not isinstance(ngram_size, int) or ngram_size < 2:
        refuse(f"base ngram_size={ngram_size!r}; the loader requires an integer "
               "of 2 or more")
    if not isinstance(heads, int) or heads <= 0:
        refuse(f"base heads_per_ngram={heads!r}; the loader requires a positive "
               "integer")
    ngram_heads = (ngram_size - 1) * heads
    if dim % ngram_heads:
        refuse(f"base ple_embed_dim={dim} is not divisible by the {ngram_heads} "
               "ngram heads the config declares; the loader rejects this "
               "checkpoint before any table is read")
    vocab_base = text.get("ngram_vocab_size_base", VOCAB_BASE_DEFAULT)
    divisible = text.get("make_ngram_vocab_size_divisible_by",
                         VOCAB_DIVISIBLE_BY_DEFAULT)
    parts = text.get("split_ngram_parts", SPLIT_PARTS_DEFAULT)
    for label, value in (("ngram_vocab_size_base", vocab_base),
                         ("make_ngram_vocab_size_divisible_by", divisible),
                         ("split_ngram_parts", parts)):
        if not isinstance(value, int) or value <= 0:
            refuse(f"base {label}={value!r}; the loader expects a positive integer")
    rows = sum(_nth_prime_after(vocab_base - 1, head + 1)
               for head in range(ngram_heads))
    rows = -(-rows // divisible) * divisible
    shard_rows = -(-rows // parts)
    return {"module": PLE_MODULE_TAIL, "width": dim // ngram_heads, "rows": rows,
            "shard_rows": shard_rows,
            "parts": [min(shard_rows, max(0, rows - part * shard_rows))
                      for part in range(parts)]}


def validate_ple_source(ple_dir: Path, ple_glob: str,
                        target: dict) -> tuple[list[str], dict[str, dict]]:
    """Check the FP8 PLE shards against the target layout: present, bound to the
    module the engine builds, every part FP8 and exactly as large as its slice,
    the module's scale present, pure-PLE; and the leader files the checkpoint
    takes from this side are present."""
    shards = sorted(p.name for p in ple_dir.glob(ple_glob))
    if not shards:
        refuse(f"no shards matching {ple_glob!r} in {ple_dir}")
    missing = [name for name in LEADER_FILES if not (ple_dir / name).is_file()]
    if missing:
        refuse(f"PLE source is missing leader files the checkpoint needs: {missing}")
    bound_prefix = target["module"] + NGRAM_MARKER
    inventory: dict[str, dict] = {}
    tables: list[str] = []
    seen: dict[int, str] = {}
    for shard in shards:
        header = read_safetensors_header(ple_dir / shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not name.startswith(PLE_MODULE):
                refuse(f"{shard}: tensor {name} is outside the PLE module "
                       f"({PLE_MODULE}*); PLE shards must be pure-PLE")
            if name in inventory:
                refuse(f"tensor {name} claimed twice (in {inventory[name]['shard']} "
                       f"and {shard})")
            suffix = None
            if NGRAM_MARKER in name:
                if not name.startswith(bound_prefix):
                    refuse(f"{shard}: tensor {name} sits under a module the engine "
                           f"never builds; the FP8 PLE path binds {bound_prefix}"
                           "shard_N.weight and its weight_scale, so these bytes "
                           "would load as nothing")
                suffix = name[len(bound_prefix):]
            if suffix == NGRAM_SCALE:
                _validate_ple_scale(shard, name, meta)
            elif suffix is not None and (match := NGRAM_TABLE.match(suffix)):
                _validate_ple_table(shard, name, meta, target, int(match.group("part")),
                                    seen)
                tables.append(name)
            elif suffix is not None and suffix.endswith("weight"):
                refuse(f"{shard}: unsupported PLE weight layout (expected "
                       f"ngram_embedding.shard_N.weight): {name}")
            elif meta["dtype"] not in {FP8_TABLE_DTYPE, "BF16"}:
                refuse(f"{shard}: tensor {name} has dtype {meta['dtype']}; expected "
                       "F8_E4M3 tables (BF16 allowed only for weight_scale)")
            inventory[name] = {"shard": shard, **meta}
    if not tables:
        refuse("no F8_E4M3 ngram tables found; the PLE source is not an FP8 checkpoint")
    absent = [f"shard_{part}" for part, rows in enumerate(target["parts"])
              if rows and part not in seen]
    if absent:
        refuse(f"{len(absent)} of the {len(target['parts'])} ngram parts the base "
               f"geometry requires are missing (first: {absent[0]}); the loader "
               "leaves those rows of the embedding unfilled")
    scale = bound_prefix + NGRAM_SCALE
    if scale not in inventory:
        refuse(f"{len(tables)} FP8 PLE tables without {scale}: an FP8 table without "
               "its per-tensor scale loads as garbage")
    return shards, inventory


def _validate_ple_scale(shard: str, name: str, meta: dict) -> None:
    """The loader's weight_scale buffer is one element, and it casts to BF16."""
    if meta["dtype"] not in ALLOWED_SCALES:
        refuse(f"{shard}: {name} has dtype {meta['dtype']}; a PLE weight_scale must "
               "be a float tensor")
    if meta["shape"] != [1]:
        refuse(f"{shard}: {name} has shape {meta['shape']}; the loader's "
               "weight_scale buffer is a single element and a differing shape "
               "raises at load time")


def _validate_ple_table(shard: str, name: str, meta: dict, target: dict,
                        part: int, seen: dict[int, str]) -> None:
    """An ngram part must be FP8, as wide as the embedding it fills, and hold
    exactly the rows the loader copies for that part index."""
    if meta["dtype"] != FP8_TABLE_DTYPE:
        refuse(f"{shard}: PLE table {name} has dtype {meta['dtype']}; every ngram "
               f"table must be {FP8_TABLE_DTYPE}, because the transplanted "
               "checkpoint declares FP8 PLE end to end")
    shape = meta["shape"]
    if len(shape) != 2 or shape[0] < 0 or shape[1] <= 0:
        refuse(f"{shard}: PLE table {name} has shape {shape}; ngram tables must be "
               "rank-2 [rows, columns] like the embedding they fill")
    if shape[1] != target["width"]:
        refuse(f"{shard}: PLE table {name} is {shape[1]} columns wide, but the base "
               f"config declares {target['width']} columns per ngram head table "
               "(ple_embed_dim // ((ngram_size - 1) * heads_per_ngram)); the load "
               "would broadcast or fail on this table")
    if part >= len(target["parts"]):
        refuse(f"{shard}: PLE table {name} is part {part}, beyond the split: the base "
               f"divides the {target['rows']} PLE rows into {len(target['parts'])} "
               f"parts, shard_0 to shard_{len(target['parts']) - 1}")
    if part in seen:
        refuse(f"{shard}: PLE table {name} is part {part} of the PLE rows, already "
               f"carried by {seen[part]}; the loader copies one tensor per part")
    if shape[0] != target["parts"][part]:
        refuse(f"{shard}: PLE table {name} carries {shape[0]} rows; part {part} covers "
               f"{target['parts'][part]} rows from row "
               f"{part * target['shard_rows']} of the base's {target['rows']}, and "
               "the loader advances only by the rows a part carries, so the rest of "
               "the embedding stays empty (only the final part may be short)")
    seen[part] = name


def validate_quant_ignore(config: dict, ple_inventory: dict[str, dict]) -> None:
    """The base's NVFP4 quantization_config must ignore every transplanted tensor.

    Without this, the engine offers FP8 bytes to the NVFP4 dequant path: a
    silent, wrong load. fnmatch semantics match the runtime's own matching.
    """
    quant = config.get("quantization_config")
    if quant is None:
        refuse("base config.json has no quantization_config; cannot prove the FP8 "
               "bytes are excluded from the NVFP4 path")
    ignore = quant.get("ignore")
    if not ignore:
        refuse("quantization_config has no ignore list; the transplanted FP8 tensors "
               "would be offered to the NVFP4 dequant path")
    uncovered = [
        name for name in ple_inventory
        if not any(fnmatch.fnmatchcase(name, pat) for pat in ignore)
    ]
    if uncovered:
        refuse(f"{len(uncovered)} transplanted tensors are not covered by "
               f"quantization_config.ignore (first: {sorted(uncovered)[0]})")


def base_ple_shards(base_weight_map: dict[str, str]) -> set[str]:
    """Base shards whose tensors are ALL ngram tables: replaced wholesale."""
    per_shard: dict[str, set[str]] = {}
    for name, shard in base_weight_map.items():
        per_shard.setdefault(shard, set()).add(
            "ngram" if is_ngram_tensor(name) else "keep")
    return {shard for shard, kinds in per_shard.items() if kinds == {"ngram"}}


def build_index(base_weight_map: dict[str, str], ple_inventory: dict[str, dict]) -> dict:
    """Merge the two weight maps; prove no double claim and table alignment."""
    replaced = {name for name in base_weight_map if is_ngram_tensor(name)}
    fp8_ngram = {name for name in ple_inventory if NGRAM_MARKER in name}
    if replaced - fp8_ngram:
        refuse(f"base ships ngram tables the PLE source lacks ({len(replaced - fp8_ngram)} "
               f"names, first: {sorted(replaced - fp8_ngram)[0]}); the tables are not aligned")
    # The ngram bytes must live ONLY in the FP8 shards. Any base shard we keep
    # (one holding a non-ngram tensor) must therefore be free of ngram tensors;
    # a kept shard that also carries an ngram table is a misaligned boundary,
    # and its stale BF16 bytes would ship beside the transplanted FP8 ones.
    kept_shards = {shard for name, shard in base_weight_map.items()
                   if name not in replaced}
    strays = sorted(name for name in replaced if base_weight_map[name] in kept_shards)
    if strays:
        refuse(f"base keeps {base_weight_map[strays[0]]} for non-PLE tensors, but it also "
               f"carries {len(strays)} ngram table(s) (first: {strays[0]}); the PLE "
               "boundary is not shard-aligned, refusing to ship stale BF16 bytes")
    merged = {name: shard for name, shard in base_weight_map.items()
              if name not in replaced}
    for name, meta in ple_inventory.items():
        if name in merged:
            refuse(f"tensor {name} exists in both the base shards and the PLE shards; "
                   "the transplant boundary is not shard-aligned")
        merged[name] = meta["shard"]
    return {"metadata": {"total_size": 0}, "weight_map": merged}


def index_total_size(shard_sources: dict[str, Path]) -> int:
    """Sum each copied shard's data region, the safetensors total_size convention.

    Keyed by destination name and resolved to the file that actually ships, so
    a shard name present under both parents is counted once instead of twice.
    """
    total = 0
    for path in sorted(shard_sources.values()):
        header = read_safetensors_header(path)
        ends = [meta["data_offsets"][1] for name, meta in header.items()
                if name != "__metadata__"]
        total += max(ends) if ends else 0
    return total


def plan_copies(base_dir: Path, ple_dir: Path, ple_shards: list[str],
                base_weight_map: dict[str, str]) -> list[tuple[Path, str]]:
    """Every file the output needs: base files minus config, index, and the
    pure-PLE base shards; plus the FP8 shards and the leader files from the
    PLE source."""
    replaced = base_ple_shards(base_weight_map)
    from_ple = set(ple_shards)
    plan: list[tuple[Path, str]] = []
    for path in sorted(base_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in (INDEX_NAME, "config.json", *LEADER_FILES) or path.name in replaced:
            continue
        if path.name in from_ple:
            refuse(f"{path.name} is a shard of the base checkpoint and of the PLE "
                   "source; the copy ships the PLE side's bytes, so the tensors the "
                   "base claims from it would not be in the assembled checkpoint")
        plan.append((path, path.name))
    for shard in sorted(ple_shards):
        plan.append((ple_dir / shard, shard))
    for name in LEADER_FILES:
        plan.append((ple_dir / name, name))
    return plan


def json_bytes(obj: dict) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def assemble(base_dir: Path, ple_dir: Path, out_dir: Path, ple_glob: str,
             dry_run: bool = False) -> dict:
    """Validate everything, then stage and move into out_dir. Refuses freely."""
    if out_dir.exists():
        refuse(f"out-dir already exists: {out_dir}; the assembler never writes into "
               "an existing directory")
    base_config, base_weight_map = validate_base(base_dir)
    ple_shards, ple_inventory = validate_ple_source(
        ple_dir, ple_glob, ple_target(base_config))
    validate_quant_ignore(base_config, ple_inventory)
    index = build_index(base_weight_map, ple_inventory)

    # Cross-check the plan against the index: every claimed shard is copied,
    # every copied shard is index-covered.
    plan = plan_copies(base_dir, ple_dir, ple_shards, base_weight_map)
    copied_shards = {dst for _, dst in plan if dst.endswith(".safetensors")}
    claimed = set(index["weight_map"].values())
    if copied_shards != claimed:
        refuse(f"copy plan and index disagree "
               f"(missing={sorted(claimed - copied_shards)[:3]}, "
               f"extra={sorted(copied_shards - claimed)[:3]})")
    index["metadata"]["total_size"] = index_total_size(
        {dst: src for src, dst in plan if dst.endswith(".safetensors")})

    report = {
        "files_copied": len(plan),
        "ple_shards_transplanted": len(ple_shards),
        "tensors": len(index["weight_map"]),
        "ple_tensors": len(ple_inventory),
        "total_size": index["metadata"]["total_size"],
        "out_dir": str(out_dir),
        "dry_run": dry_run,
    }
    if dry_run:
        return report

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=out_dir.name + ".staging."))
    try:
        for src, dst_name in plan:
            shutil.copyfile(src, staged / dst_name)
        (staged / INDEX_NAME).write_bytes(json_bytes(index))
        (staged / "config.json").write_bytes(json_bytes(patch_config(base_config)))
        os.replace(staged, out_dir)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return report


def patch_config(config: dict) -> dict:
    """The one reviewed change: FP8 PLE dtype. Everything else ships verbatim."""
    patched = json.loads(json.dumps(config))
    patched["text_config"][PLE_CONFIG_KEY] = FP8_PLE_DTYPE
    return patched


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-dir", required=True, type=Path,
                        help="directory of the BF16-PLE base checkpoint (downloaded)")
    parser.add_argument("--ple-dir", required=True, type=Path,
                        help="directory of the FP8 PLE source checkpoint (downloaded)")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="destination directory; must not exist")
    parser.add_argument("--ple-glob", default="model-plefp8-*.safetensors",
                        help="glob for pure-FP8-PLE shards in --ple-dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="run every validation, write nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        for label, path in (("base-dir", args.base_dir), ("ple-dir", args.ple_dir)):
            if not path.is_dir():
                refuse(f"{label} is not a directory: {path}")
        report = assemble(args.base_dir, args.ple_dir, args.out_dir,
                          args.ple_glob, dry_run=args.dry_run)
    except Refuse as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    print("VERDICT:", "VALIDATED (dry run)" if report["dry_run"] else "ASSEMBLED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
