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

PLE_MODULE = "model.language_model.layers.1.ple."
PLE_CONFIG_KEY = "ple_embedding_dtype"
FP8_PLE_DTYPE = "float8_e4m3fn"
INDEX_NAME = "model.safetensors.index.json"
HEADER_SIZE_MAX = 128 * 1024 * 1024
NGRAM_MARKER = ".ngram_embedding."
# The engine binds exactly one weight per ngram table: <module>.ngram_embedding
# .shard_<N>.weight (models/qwen4_exp.py raises on any other PLE weight).
NGRAM_TABLE = re.compile(r"^shard_\d+\.weight$")
NGRAM_SCALE = "weight_scale"
# Element sizes for the dtypes this tool may meet in a PLE shard, used to prove
# a declared shape matches the bytes the header reserves for it.
ELEMENT_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1}
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
    """Parse a safetensors file's tensor metadata (names, dtypes, offsets).

    A partial download has a complete header and a short data section, so the
    declared payload is checked against the bytes on disk here: every caller
    runs before staging, and a shard missing tensor bytes must REFUSE rather
    than ship a checkpoint whose index points at absent bytes.
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
    declared = max((meta["data_offsets"][1] for name, meta in header.items()
                    if name != "__metadata__"), default=0)
    present = path.stat().st_size - (8 + header_size)
    if present < declared:
        refuse(f"{path.name}: truncated payload, the header declares {declared} "
               f"bytes of tensors and {max(present, 0)} are on disk")
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


def validate_ple_source(ple_dir: Path, ple_glob: str) -> tuple[list[str], dict[str, dict]]:
    """Check the FP8 PLE shards: present, every ngram table FP8 and well formed,
    each module's scale bound, pure-PLE; and the leader files the checkpoint
    takes from this side are present."""
    shards = sorted(p.name for p in ple_dir.glob(ple_glob))
    if not shards:
        refuse(f"no shards matching {ple_glob!r} in {ple_dir}")
    missing = [name for name in LEADER_FILES if not (ple_dir / name).is_file()]
    if missing:
        refuse(f"PLE source is missing leader files the checkpoint needs: {missing}")
    inventory: dict[str, dict] = {}
    tables: dict[str, list[str]] = {}
    widths: dict[str, int] = {}
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
            suffix = (name.rsplit(NGRAM_MARKER, 1)[1]
                      if NGRAM_MARKER in name else None)
            if suffix == NGRAM_SCALE:
                _validate_ple_scale(shard, name, meta)
            elif suffix is not None and NGRAM_TABLE.match(suffix):
                module = name[: name.index(NGRAM_MARKER)]
                shape = _validate_ple_table(shard, name, meta)
                if widths.setdefault(module, shape[1]) != shape[1]:
                    refuse(f"{shard}: PLE table {name} is {shape[1]} columns wide, "
                           f"but {module} tables are {widths[module]} columns; every "
                           "table of one PLE module fills the same embedding")
                tables.setdefault(module, []).append(name)
            elif suffix is not None and name.endswith(".weight"):
                refuse(f"{shard}: unsupported PLE weight layout (expected "
                       f"ngram_embedding.shard_N.weight): {name}")
            elif meta["dtype"] not in {FP8_TABLE_DTYPE, "BF16"}:
                refuse(f"{shard}: tensor {name} has dtype {meta['dtype']}; expected "
                       "F8_E4M3 tables (BF16 allowed only for weight_scale)")
            inventory[name] = {"shard": shard, **meta}
    if not tables:
        refuse("no F8_E4M3 ngram tables found; the PLE source is not an FP8 checkpoint")
    for module in sorted(tables):
        scale = module + NGRAM_MARKER + NGRAM_SCALE
        if scale not in inventory:
            refuse(f"{len(tables[module])} FP8 PLE tables under {module} without "
                   f"{scale}: an FP8 table without its per-tensor scale loads as "
                   "garbage")
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
    _require_declared_bytes(shard, name, meta, [1])


def _validate_ple_table(shard: str, name: str, meta: dict) -> list[int]:
    """An ngram table must be FP8 and a rank-2 block of exactly that many bytes."""
    if meta["dtype"] != FP8_TABLE_DTYPE:
        refuse(f"{shard}: PLE table {name} has dtype {meta['dtype']}; every ngram "
               f"table must be {FP8_TABLE_DTYPE}, because the transplanted "
               "checkpoint declares FP8 PLE end to end")
    shape = meta["shape"]
    if len(shape) != 2 or min(shape) <= 0:
        refuse(f"{shard}: PLE table {name} has shape {shape}; ngram tables must be "
               "rank-2 [rows, columns] like the embedding they fill")
    _require_declared_bytes(shard, name, meta, shape)
    return shape


def _require_declared_bytes(shard: str, name: str, meta: dict, shape: list[int]) -> None:
    """A PLE tensor's byte span must cover its declared shape exactly."""
    span = meta["data_offsets"][1] - meta["data_offsets"][0]
    needed = math.prod(shape) * ELEMENT_SIZE[meta["dtype"]]
    if span != needed:
        refuse(f"{shard}: PLE tensor {name} reserves {span} bytes for shape "
               f"{shape} of {meta['dtype']}, which needs {needed}")


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
    ple_shards, ple_inventory = validate_ple_source(ple_dir, ple_glob)
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
    for label, path in (("base-dir", args.base_dir), ("ple-dir", args.ple_dir)):
        if not path.is_dir():
            refuse(f"{label} is not a directory: {path}")
    try:
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
