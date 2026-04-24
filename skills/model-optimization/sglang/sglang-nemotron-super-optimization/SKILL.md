---
name: sglang-nemotron-super-optimization
description: PR-backed optimization manual for Nemotron Super / Nano Hybrid in SGLang. Use when Codex needs to audit, debug, extend, or document NemotronH, Nemotron 3 Super, Nemotron Nano hybrid Mamba+Attention+MoE, MTP, NVFP4, and VL adjacencies.
---

# SGLang Nemotron Super / Nano Hybrid Optimization

## Overview

This skill covers NemotronH, Nemotron 3 Super, Nemotron Nano hybrid Mamba+Attention+MoE, MTP, NVFP4, and VL adjacencies.

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/sglang/nemotron-super/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `sglang/python/sglang/srt/models/nemotron_h.py`
- `sglang/python/sglang/srt/models/nemotron_h_mtp.py`
- `sglang/python/sglang/srt/models/nano_nemotron_vl.py`

## Current Main Summary

- Nemotron is a hybrid family: attention, Mamba, MoE, MTP, and VL all intersect.
- Because of that, graph execution, cache dtype, and quantized MoE correctness are the primary risk areas.

## Key Landed PRs

- [#16172](https://github.com/sgl-project/sglang/pull/16172) `NemotronH PP support`: Opened pipeline parallelism on NemotronH.
- [#16227](https://github.com/sgl-project/sglang/pull/16227) `Add latent MoE support`: Added the hybrid latent-MoE path.
- [#19903](https://github.com/sgl-project/sglang/pull/19903) `Enable Piecewise CUDA Graph for NemotronH Hybrid Models`: Improved hybrid serving efficiency.
- [#20407](https://github.com/sgl-project/sglang/pull/20407) `Support Nemotron 3 Super NVFP4`: Added the key quantized Super checkpoint path.
- [#20575](https://github.com/sgl-project/sglang/pull/20575) `Add Nemotron 3 Super CI tests for BF16 and NVFP4`: Added regression coverage for the production checkpoint variants.

## Validation Lanes

- Startup on the cookbook checkpoint or example route.
- Re-run the parser, quantization, or multimodal lane that matches this family.
- Re-check registered or manual tests after touching loader or processor code.
