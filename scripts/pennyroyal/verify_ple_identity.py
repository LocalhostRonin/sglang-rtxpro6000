#!/usr/bin/env python3
"""Verify PLE tensor byte-identity between two checkpoints on the Hugging Face Hub.

Two checkpoints share a PLE (per-layer embedding) table only if the raw tensor
bytes agree. File-level SHA256 proves that instantly; when a shard's full-file
hash differs, this tool range-downloads exactly the PLE tensor bytes from both
sides and hashes them per tensor. Nothing else is downloaded.

Use it before transplanting PLE tables from one checkpoint into another: the
transplant is sound only when the source tables are quantized from the same
BF16 bytes the target's PLE carries.

Usage:
    python3 verify_ple_identity.py \
        --repo-a Qwen/Qwen3.8-Flash-Next --rev-a <40-char sha> \
        --repo-b ukisai/Swift1.5-Qwen3.8-Flash-Next --rev-b <40-char sha> \
        [--ple-key .ple.] [--cache-dir DIR]

Exit code 0 = VERIFIED (every PLE tensor matches), 1 = NOT VERIFIED.
Uses only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_PLE_KEY = ".ple."
DTYPE_SIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
    "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}
_CHUNK = 64 * 1024 * 1024


def ple_tensor_names(weight_map: dict[str, str], ple_key: str) -> list[str]:
    """Every tensor name in the weight map that contains the PLE marker."""
    return sorted(name for name in weight_map if ple_key in name)


def shards_proven_identical(
    ple_shards: list[str],
    tree_a: dict[str, dict],
    tree_b: dict[str, dict],
) -> tuple[dict[str, str], list[str]]:
    """Split PLE shards into (proven by full-file SHA256, needing tensor hashing).

    For LFS files the Hub API's ``oid`` is the SHA256 of the whole shard, so two
    shards with the same oid and size are byte-identical with zero downloads.
    """
    proven: dict[str, str] = {}
    needs_range: list[str] = []
    for shard in ple_shards:
        entry_a, entry_b = tree_a[shard], tree_b[shard]
        oid_a = entry_a.get("lfs", {}).get("oid")
        oid_b = entry_b.get("lfs", {}).get("oid")
        if oid_a and oid_a == oid_b and entry_a["size"] == entry_b["size"]:
            proven[shard] = oid_a
        else:
            needs_range.append(shard)
    return proven, needs_range


def parse_safetensors_header(raw: bytes) -> dict:
    """Decode a safetensors header block (the JSON after the 8-byte LE size)."""
    header_size = int.from_bytes(raw[:8], "little")
    return json.loads(raw[8 : 8 + header_size])


def tensor_inventory(header: dict, ple_key: str) -> dict[str, dict]:
    """Name/shape/dtype/offsets for every PLE tensor in a shard header."""
    inventory = {}
    for name, meta in header.items():
        if name == "__metadata__" or ple_key not in name:
            continue
        inventory[name] = dict(meta)
    return inventory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-a", required=True, help="repo id of the first checkpoint")
    parser.add_argument("--rev-a", required=True, help="pinned 40-char revision for repo-a")
    parser.add_argument("--repo-b", required=True, help="repo id of the second checkpoint")
    parser.add_argument("--rev-b", required=True, help="pinned 40-char revision for repo-b")
    parser.add_argument("--ple-key", default=DEFAULT_PLE_KEY,
                        help=f"substring marking PLE tensors (default {DEFAULT_PLE_KEY!r})")
    parser.add_argument("--cache-dir", default=None,
                        help="directory to cache revision-pinned metadata and "
                             "fetched headers (default: temp, no reuse)")
    return parser


class _Fetcher:
    def __init__(self, cache_dir: Path | None):
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, url: str, rng: str | None = None, timeout: int = 300) -> bytes:
        request = urllib.request.Request(url, headers={"Range": rng} if rng else {})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def fetch_json(self, cache_key: str, url: str) -> dict:
        if self.cache_dir is not None:
            # cache_key labels the artifact ("a_meta.json"); the URL digest is
            # what keeps two repos, revisions or endpoints from sharing one
            # entry. Without it a re-run for another pair reads stale metadata.
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            path = self.cache_dir / f"{cache_key}-{digest}.json"
            if path.exists():
                return json.loads(path.read_bytes())
            payload = self.fetch(url)
            path.write_bytes(payload)
            return json.loads(payload)
        return json.loads(self.fetch(url))


def _resolve(repo: str, rev: str, fname: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"


def verify(repo_a: str, rev_a: str, repo_b: str, rev_b: str,
           ple_key: str = DEFAULT_PLE_KEY, cache_dir: Path | None = None) -> dict:
    """Run the full comparison and return the report dict."""
    fetcher = _Fetcher(cache_dir)
    sides = {"a": (repo_a, rev_a), "b": (repo_b, rev_b)}

    for tag, (repo, rev) in sides.items():
        # Model metadata is the one mutable document this tool reads: its URL
        # carries no revision, so HEAD moves under the same address and a
        # cached copy would abort a valid re-pin on phantom drift. The index and
        # tree below are revision-pinned, therefore cacheable.
        meta = json.loads(fetcher.fetch(f"https://huggingface.co/api/models/{repo}"))
        if meta["sha"] != rev:
            raise SystemExit(
                f"{tag}: revision drift! {repo} HEAD is {meta['sha']}, pinned {rev}")

    index = {}
    for tag, (repo, rev) in sides.items():
        index[tag] = fetcher.fetch_json(
            f"{tag}_index.json", _resolve(repo, rev, "model.safetensors.index.json"))
    if index["a"]["weight_map"] != index["b"]["weight_map"]:
        raise SystemExit("weight maps differ between the two repos; "
                         "PLE tensors are not aligned by name/shard")
    weight_map = index["a"]["weight_map"]
    ple_names = ple_tensor_names(weight_map, ple_key)
    if not ple_names:
        raise SystemExit(f"no tensors containing {ple_key!r} found; nothing to verify")
    ple_shards = sorted({weight_map[name] for name in ple_names})
    print(f"PLE tensors: {len(ple_names)} in {len(ple_shards)} shards")

    tree = {}
    for tag, (repo, rev) in sides.items():
        listing = fetcher.fetch_json(
            f"{tag}_tree.json",
            f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true&blobs=true")
        tree[tag] = {e["path"]: e for e in listing if e["type"] == "file"}

    proven, diff_shards = shards_proven_identical(ple_shards, tree["a"], tree["b"])
    print(f"shards proven identical by file SHA256: {len(proven)}/{len(ple_shards)}")
    print(f"shards needing tensor-level comparison : {diff_shards}")

    def header_json(tag: str, shard: str) -> tuple[dict, int]:
        """Parsed header of one side's shard plus where its data section starts.

        ``data_offsets`` are relative to the byte right after this file's own
        8-byte size field and header block, so the absolute offset depends on
        that file's header length: two layout-identical shards can pad their
        headers differently and still carry identical tensors.
        """
        key = (tag, shard)
        if key not in headers:
            url = _resolve(*sides[tag], shard)
            size = int.from_bytes(fetcher.fetch(url, "bytes=0-7"), "little")
            raw = fetcher.fetch(url, f"bytes=0-{size + 7}")
            headers[key] = (parse_safetensors_header(raw), 8 + size)
        return headers[key]

    headers: dict[tuple[str, str], tuple[dict, int]] = {}
    inventory: dict[str, dict] = {}
    for shard in ple_shards:
        header, _ = header_json("a", shard)
        for name, meta in tensor_inventory(header, ple_key).items():
            inventory.setdefault(name, {"shard": shard, **meta})
    data_start: dict[tuple[str, str], int] = {}
    for shard in diff_shards:
        header_a, start_a = header_json("a", shard)
        header_b, start_b = header_json("b", shard)
        if header_a != header_b:
            raise SystemExit(f"header mismatch in {shard}: layouts are not aligned")
        data_start[("a", shard)], data_start[("b", shard)] = start_a, start_b

    mismatches: list[str] = []

    def hash_range(tag: str, shard: str, start: int, end: int) -> str:
        """SHA256 of [start, end) of one side's shard, absolute file offsets."""
        url = _resolve(*sides[tag], shard)
        digest = hashlib.sha256()
        pos = start
        while pos < end:
            stop = min(pos + _CHUNK, end) - 1
            chunk = fetcher.fetch(url, f"bytes={pos}-{stop}")
            if len(chunk) != stop - pos + 1:
                raise SystemExit(f"short read on {url} at {pos}")
            digest.update(chunk)
            pos = stop + 1
        return digest.hexdigest()

    to_check = [name for name in ple_names if inventory[name]["shard"] in diff_shards]
    print(f"range-hashing {len(to_check)} PLE tensors in differing shards "
          "(layout-identical headers confirmed)")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for name in to_check:
            shard = inventory[name]["shard"]
            lo, hi = inventory[name]["data_offsets"]
            ranges = {tag: (lo + data_start[(tag, shard)],
                            hi + data_start[(tag, shard)]) for tag in ("a", "b")}
            inventory[name]["byte_ranges"] = {tag: list(span)
                                              for tag, span in ranges.items()}
            for tag, (start, end) in ranges.items():
                futures[(name, tag)] = pool.submit(hash_range, tag, shard, start, end)
        for name in to_check:
            hash_a = futures[(name, "a")].result()
            hash_b = futures[(name, "b")].result()
            inventory[name]["sha256_a"] = hash_a
            inventory[name]["sha256_b"] = hash_b
            inventory[name]["match"] = hash_a == hash_b
            if hash_a != hash_b:
                mismatches.append(name)
            print(f"  {'MATCH   ' if hash_a == hash_b else 'MISMATCH'}  {name}")

    return {
        "repos": {"a": {"repo": repo_a, "rev": rev_a}, "b": {"repo": repo_b, "rev": rev_b}},
        "ple_tensor_count": len(ple_names),
        "identical_via_file_sha256": len([n for n in ple_names
                                          if inventory[n]["shard"] in proven]),
        "identical_via_tensor_range_sha256": len(to_check),
        "mismatches": mismatches,
        "shard_file_hashes": proven,
        "diff_shards": diff_shards,
        "tensors": inventory,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = verify(
        repo_a=args.repo_a, rev_a=args.rev_a,
        repo_b=args.repo_b, rev_b=args.rev_b,
        ple_key=args.ple_key,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    matched = report["ple_tensor_count"] - len(report["mismatches"])
    print(f"\nPLE tensors compared : {report['ple_tensor_count']}")
    print(f"  - {report['identical_via_file_sha256']} inside shards whose full-shard SHA256 "
          "is identical")
    print(f"  - {report['identical_via_tensor_range_sha256']} hashed byte-for-byte from "
          "differing shards")
    print(f"Exact matches        : {matched}")
    print(f"Mismatches           : {len(report['mismatches'])}"
          f"  {' '.join(report['mismatches']) or ''}")
    print("VERDICT:", "VERIFIED" if not report["mismatches"] else "NOT VERIFIED")
    return 0 if not report["mismatches"] else 1


if __name__ == "__main__":
    sys.exit(main())
