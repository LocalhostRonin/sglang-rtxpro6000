# Swift-1.5 Qwen3.8-Flash-Next NVFP4 with FP8 PLE: validation

Swift-1.5-Qwen3.8-Flash-Next-NVFP4 served with FP8 E4M3 PLE tables on the
FP8-resident PLE path, instead of the BF16 staging path described in
`tools/ple_nvme/BF16-STAGING-HOTFIX.md` (folded into 2.5.3+). The NVFP4 PLE
path rejects this checkpoint because its 137 ngram PLE tensors ship as BF16
(`text_config.ple_embedding_dtype = "bfloat16"`), which puts ~95.4 GiB of
host RAM into the tables. The FP8 path halves that to ~47.68 GiB, the sizing
the FP8-resident PLE path is built for.

This is a recipe for a derived checkpoint, validated by the community operator
who runs it as a daily driver. It is not a runtime change: the serving engine
is the stock published image, unmodified.

## The recipe

Assemble the checkpoint from two published parents; no weight math, no
requantization. All RadixArk bytes ship verbatim.

| Part | Source | Files |
|---|---|---|
| Base weights, config, tokenizer | [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) @ `0bd4fe22` | 49 shards + non-shard files |
| FP8 PLE shards | [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) @ `7b719225` | 10 `model-plefp8-*.safetensors` |
| Config patch | one key | `text_config.ple_embedding_dtype = "float8_e4m3fn"` |

The result is 70 files, 135,250,559,793 bytes, 296,475 tensors, with an
index byte-identical in name-set to the published FP8-PLE checkpoint.
Download both parents with `hf download <repo> --revision <sha> --local-dir
<dir>` first (the RadixArk repo is gated; accept the terms).

```sh
python3 scripts/pennyroyal/transplant_ple_fp8.py \
    --base-dir /models/swift15-nvfp4 \
    --ple-dir  /models/qwen38-flash-next-nvfp4 \
    --out-dir  /models/swift15-flash-next-plefp8 \
    --dry-run
# review the report, then re-run without --dry-run
```

The assembler validates before it writes: base PLE dtype is BF16, the base
config enables exactly the one PLE layer this recipe transplants
(`text_config.ple_layer_ids = [2]`, checkpoint layer 1), FP8 shards are
pure-PLE (`model.language_model.layers.1.ple.*` only) and bound to the module
the engine builds (`ple_embedding.ngram_embedding`), every ngram table is
F8_E4M3, rank-2, byte-complete, exactly `ple_embed_dim // ((ngram_size - 1) *
heads_per_ngram)` columns wide, and covers its slice of the target rows: the
part set is `shard_0 .. shard_(split_ngram_parts-1)`, each part
`ceil(rows / split_ngram_parts)` rows after the head-vocab padding, only the
final part legitimately shorter, with the module's own per-tensor
`weight_scale`. All of that comes from the base `config.json`, so a base whose
own PLE shards were pruned is checked the same way. The base's
`quantization_config.ignore`
covers every transplanted tensor (the FP8 bytes are never offered to the NVFP4
dequant path), the transplant boundary is shard-aligned, and the copy plan
agrees with the generated index. Every refusal prints `REFUSE:` with the
failed invariant. Output is staged in a temp directory and moved atomically;
an existing destination is never written.

## Why the transplant is sound

PLE lineage, checked with `scripts/pennyroyal/verify_ple_identity.py`:

- Swift's 137 PLE tensors are byte-identical (SHA256) to the base
  Qwen3.8-Flash-Next PLE. The fine-tune did not touch the tables.
- RadixArk's FP8 tables are quantized from that same base BF16 PLE.
  Quantizing Swift's own tables would produce the tables we transplant.

```sh
python3 scripts/pennyroyal/verify_ple_identity.py \
    --repo-a Qwen/Qwen3.8-Flash-Next       --rev-a <40-char-sha> \
    --repo-b ukisai/Swift1.5-Qwen3.8-Flash-Next --rev-b <40-char-sha>
# VERDICT: VERIFIED
```

Shards whose full-file SHA256 (the LFS `oid`) matches are proven with zero
downloads; any shard whose file hash differs is compared per tensor by HTTP
range requests over exactly the PLE byte ranges. Pinned revisions are checked
against the Hub HEAD; drift aborts.

## Serving validation

Ran for 7 days as daily driver before this document existed.

- Image: `ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.2`, stock published
  digest, unmodified. One RTX PRO 6000 Blackwell (SM120), TP=1, Linux.
- 50,790 chat-completion requests over the 7 days, all HTTP 200, zero
  restarts, zero failures. Health generate 1 s on the final day.
- The operator then moved the instance to `v3.0.1` on 2026-10-12, same
  checkpoint and flags: 4,165 chat-completion requests over 2 days, all
  HTTP 200, RestartCount 0.
- Launch: FR-Spec NEXTN profile, 524,288-token context (YaRN 2.0),
  mem-fraction 0.981, FP8 E4M3 KV (824,384-token pool), HiCache + NIXL host
  tier (32 GiB), EAGLE draft from the same checkpoint, `flash-next-64k.pt`
  token map, qwen3 / qwen3_coder parsers.
- Engine logs at load: "Detected nvfp4 checkpoint", "Flash-Next online FP8
  enabled on SM120", PLE host-RAM path active.
- Boot warnings, same class as the vanilla profile: context_length above the
  derived value (the standard YaRN override) and one benign Triton
  kernel-build warning. The server stayed clean.

## Not tested

- No controlled FP8-PLE vs BF16-PLE quality A/B on Swift. The byte-identity
  lineage is the safety argument; there is no measured quality diff.
- No NIXL restart cache-reuse test for this checkpoint (the same gap
  [SWIFT-27B.md](SWIFT-27B.md) flags for Swift 27B).
- No formal benchmarks; no performance claim. The memory numbers are
  arithmetic on measured tensor sizes.

The [validation repository](https://github.com/jpezzulli/pennyroyal-validation)
covers the reasoning and tool cases. For the launch configuration, see the
Flash-Next FR-Spec recipe in [RUN.md](RUN.md).
