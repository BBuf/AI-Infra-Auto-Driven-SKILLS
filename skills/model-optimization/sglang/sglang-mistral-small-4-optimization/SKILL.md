---
name: sglang-mistral-small-4-optimization
description: PR-backed optimization manual for Mistral Small 4 in SGLang. Use when an engineer needs to audit, debug, extend, or document Mistral Small 4, Leanstral, and closely related Mistral Large 3 / Ministral serving behavior, including multimodal and EAGLE paths.
---

# SGLang Mistral Small 4 Optimization

## Overview

This skill covers Mistral Small 4, Leanstral, and closely related Mistral Large 3 / Ministral serving behavior, including multimodal and EAGLE paths.

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/sglang/mistral-small-4/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `sglang/python/sglang/srt/models/mistral_large_3.py`
- `sglang/python/sglang/srt/models/mistral_large_3_eagle.py`
- `sglang/python/sglang/srt/models/mistral.py`
- `sglang/python/sglang/srt/models/ministral3.py`

## Current Main Summary

- Mistral Small 4 sits on top of the larger Mistral Large 3 / Ministral runtime work.
- Startup format mismatches, multimodal projector behavior, and Eagle / MoE integration are the main risk areas.

## Key Landed PRs

- [#14213](https://github.com/sgl-project/sglang/pull/14213) `Add Mistral Large 3 support`: Historical base runtime reused by later Small 4 work.
- [#14466](https://github.com/sgl-project/sglang/pull/14466) `Add Mistral Large 3 Eagle Support`: Enabled speculative decode on the underlying family.
- [#15049](https://github.com/sgl-project/sglang/pull/15049) `Mistral Large 3 NVFP4 TRTLLM MoE support`: Added the first serious quantized MoE path.
- [#20708](https://github.com/sgl-project/sglang/pull/20708) `Add Mistral Small 4 support`: Brought Mistral Small 4 / Pixtral-style runtime into mainline.
- [#21620](https://github.com/sgl-project/sglang/pull/21620) `Mistral Small 4 fails to start due to config/weight format mismatch`: Closed a startup regression after launch.

## Validation Lanes

- Startup on the cookbook checkpoint or example route.
- Re-run the parser, quantization, or multimodal lane that matches this family.
- Re-check registered or manual tests after touching loader or processor code.
