# Penny Royal 3.0.2

This maintenance release reduces host-memory overhead in persistent caching and brings focused improvements to speculative decoding and client compatibility for both supported model profiles.

Released October 17, 2026. Container build status is recorded on the GitHub release page.

## What changes

- **Less host RAM used by NIXL caching.** Mamba host-state slots are now page-aligned, allowing direct storage I/O to use the existing host pool without two redundant bounce buffers. In the measured Flash-Next configuration, this recovered **about 14.1 GiB of host RAM**. Thanks to Kaspar Holmgren ([@hiyazakite](https://github.com/hiyazakite)) for [PR73](https://github.com/jpezzulli/sglang-rtxpro6000/pull/73).
- **Less rapid MTP width switching on Flash-Next.** The existing acceptance-based selector continues learning while a hold period limits repeated switches between four- and eight-token speculation for a single request. The hold starts at 40 completed feedback batches and backs off to 80 after repeated reversals. Two or more concurrent requests still select four-token speculation immediately.
- **Complete compact draft-attention windows on 27B.** DFlash2 now includes the verification block when calculating the host-side attention length after applying the window limit. This keeps the draft's intended attention window complete.
- **Correct grammar-mask handling on 27B.** DFlash2 clears stale request-level grammar masks before speculative verification. The verifier continues to apply the grammar constraints for each token position.
- **Better assistant-history compatibility.** Both profiles accept the `reasoning` field when `reasoning_content` is absent or null. An explicitly supplied `reasoning_content` string, including an empty string, retains precedence.

## Validation and model settings

The profiles remain Qwen3.8-27B FP8 with DFlash2 and Qwen3.8 Flash-Next NVFP4 with NEXTN MTP, both using HiCache/NIXL. Model weights, quantization, default precision and configured capacities are unchanged.

Local single-GPU qualification covered both profiles: reasoning, 30/30 streaming tool checks each, images, 489,921-token cold prefill and recall, and disk-restart restoration of KV plus Mamba state.

## Updating

Follow the [native build instructions](https://github.com/jpezzulli/sglang-rtxpro6000/blob/pennyroyal-v3.0.2/BUILD.md) or [container guide](https://github.com/jpezzulli/sglang-rtxpro6000/blob/pennyroyal-v3.0.2/docker/pennyroyal/README.md). Use `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.0.2` with the matching 3.0.2 startup recipes or configurator output, preserving your paths and settings. Custom startup scripts that pin the adaptive-policy hash need the matching updated recipe. This release does not require downloading model weights again or deleting persistent caches.

Thanks to Kaspar Holmgren for the original NIXL contribution and Penny (`Pennyroyal@agentmail.to`) for the accompanying fixes.
