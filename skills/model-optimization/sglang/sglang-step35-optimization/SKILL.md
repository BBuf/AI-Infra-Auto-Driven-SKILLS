---
name: sglang-step35-optimization
description: PR-backed optimization manual for Step3.5 / Step3-VL in SGLang. Use when an engineer needs to audit, debug, extend, or document Step3.5-Flash and Step3-VL-10B serving, MTP, MoE all-reduce, tool/reasoning parser, and processor evolution.
---

# SGLang Step3.5 / Step3-VL Optimization

## Overview

This skill covers Step3.5-Flash and Step3-VL-10B serving, MTP, MoE all-reduce, tool/reasoning parser, and processor evolution.

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/sglang/step35/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `sglang/python/sglang/srt/models/step3p5.py`
- `sglang/python/sglang/srt/models/step3p5_mtp.py`
- `sglang/python/sglang/srt/models/step3_vl.py`
- `sglang/python/sglang/srt/models/step3_vl_10b.py`

## Current Main Summary

- Step3.5 is split between text/MTP and VL processor work.
- All-reduce efficiency and parser behavior are the main axes to track.

## Key Landed PRs

- [#8583](https://github.com/sgl-project/sglang/pull/8583) `Support Step3V`: Initial Step3 visual model support.
- [#8699](https://github.com/sgl-project/sglang/pull/8699) `Support DP Attention for step3_vl`: Enabled multi-GPU VL serving.
- [#9695](https://github.com/sgl-project/sglang/pull/9695) `Add step3 tool parser`: Added tool-call parsing.
- [#18564](https://github.com/sgl-project/sglang/pull/18564) `Implement the standard multi-layer MTP for step3p5`: Added Step3.5 draft-model support.
- [#22773](https://github.com/sgl-project/sglang/pull/22773) `Optimize allreduce in MoE layers`: Targeted the Step3.5 MoE hot path.

## Validation Lanes

- Startup on the cookbook checkpoint or example route.
- Re-run the parser, quantization, or multimodal lane that matches this family.
- Re-check registered or manual tests after touching loader or processor code.
