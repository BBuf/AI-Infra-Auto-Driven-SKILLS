---
name: sglang-mimo-v2-flash-optimization
description: PR-backed optimization manual for MiMo-V2 / MiMo-V2-Flash / MiMo-V2.5 in SGLang. Use when an engineer needs to audit, debug, extend, or document MiMo-V2 inference-centric MoE runtime, flashinfer/TRT-LLM fused all-reduce, overlap, MTP/EAGLE, multimodal/pro variants, and reasoning parser behavior.
---

# SGLang MiMo-V2 / MiMo-V2.5 Optimization

## Overview

This skill covers the MiMo-V2 family in SGLang: historical MiMo-V2-Flash,
current `MiMoV2ForCausalLM` compatibility, MiMo-V2.5/Pro deployment recipes,
MTP/EAGLE paths, flashinfer/TRT-LLM fused all-reduce, overlap, multimodal
variants, and reasoning parser behavior.

Evidence snapshot:

- SGLang `origin/main`: `6fbad22fe` on `2026-04-28`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Latest family update: `#23808` renamed the runtime to `mimo_v2.py` /
  `mimo_v2_nextn.py` while keeping the old Flash architecture alias loadable;
  `#23851` added the MiMo-V2.5 cookbook and command generator.
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/sglang/mimo-v2-flash/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- Current runtime: `sglang/python/sglang/srt/models/mimo_v2.py`
- Current MTP runtime: `sglang/python/sglang/srt/models/mimo_v2_nextn.py`
- Historical names still relevant to older PR cards:
  `sglang/python/sglang/srt/models/mimo_v2_flash.py`,
  `sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py`
- Config and server integration:
  `sglang/python/sglang/srt/configs/model_config.py`,
  `sglang/python/sglang/srt/server_args.py`
- Current docs/generator:
  `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`,
  `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`

## Current Main Summary

- MiMo-V2-Flash is the historical throughput-oriented landing.
- Current mainline accepts both `MiMoV2ForCausalLM` and the old
  `MiMoV2FlashForCausalLM` alias through the renamed `mimo_v2.py` module.
- MiMo-V2.5-Pro adds 1.02T text/pro serving with EAGLE speculative decoding;
  MiMo-V2.5 covers the 310B multimodal line and TP=4-interleaved fused
  `qkv_proj` constraints.
- All-reduce fusion, overlap, hybrid SWA/full attention, MTP/EAGLE, and
  reasoning parser behavior matter more than generic loader work.

## Key Landed PRs

- [#15207](https://github.com/sgl-project/sglang/pull/15207) `MiMo-V2-Flash day0 support`: Initial MiMo-V2-Flash landing.
- [#15464](https://github.com/sgl-project/sglang/pull/15464) `Optimize MiMo-V2-Flash by flashinfer fused allreduce`: Targeted decode-side communication cost.
- [#15488](https://github.com/sgl-project/sglang/pull/15488) `Respect `--swa-full-tokens-ratio``: Fixed a concrete runtime flag integration bug.
- [#17634](https://github.com/sgl-project/sglang/pull/17634) `Support two batch overlap`: Added overlap / throughput optimization.
- [#21414](https://github.com/sgl-project/sglang/pull/21414) `Add mimo reasoning parser`: Completed the parser path for thinking outputs.
- [#23808](https://github.com/sgl-project/sglang/pull/23808) `Xiaomi MiMo-V2.5-Pro day0 support`: Renamed runtime files to `mimo_v2.py` / `mimo_v2_nextn.py`, added `MiMoV2ForCausalLM`, and kept the Flash alias.
- [#23851](https://github.com/sgl-project/sglang/pull/23851) `Add MiMo-V2.5 docs`: Added current MiMo-V2.5/Pro deployment cookbook and command generator.

## Validation Lanes

- Startup on MiMo-V2-Flash and current MiMo-V2.5/Pro checkpoints or cookbook routes.
- Re-run the parser, quantization, or multimodal lane that matches this family.
- Re-check MTP/EAGLE, `qkv_proj` loading, hybrid SWA/full attention, and registered/manual tests after touching loader or processor code.

## References

- `references/pr-history.md`: diff-reviewed MiMo-V2 family PR cards; it includes both historical Flash-file cards and current `mimo_v2.py` / MiMo-V2.5 updates.
