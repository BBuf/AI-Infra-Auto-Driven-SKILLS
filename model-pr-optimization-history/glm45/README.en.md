# SGLang GLM-4.5 Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: GLM-4.5, GLM-4.5-Air, GLM-4.5 MoE, A2A/DeepEP, reduce-scatter behavior, NVFP4 padding, and GLM45 parser behavior.

## Summary

GLM-4.5 is the baseline GLM MoE lane. Later GLM-4.6/4.7/5.x work inherits many MoE, A2A, quantization, parser, and platform decisions from it. GLM-4.5V belongs to the VLM/OCR lane unless shared text MoE code is the root cause.

## Code Surfaces

- `python/sglang/srt/models/glm4.py`
- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/function_call/glm4_moe_detector.py`
- `docs/basic_usage/glm45.md`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5.mdx`

## Merged PRs

Key merged PRs: `#11665`, `#11692`, `#11800`, `#12497`, `#17714`, and `#11017`.

## Open Radar

Track `#13711`, `#19106`, `#19728`, `#20917`, and `#23067`.

## Cookbook Evidence

Track `sgl-cookbook#92` and `#95`.

## Next Work

Build fixed BF16/FP8/NVFP4/A2A/DeepEP/parser streaming smoke tests, keep parser changes separate from MoE perf work, and route GLM-4.5V issues through the VLM/OCR lane first.

## 2026-04-23 Three-Pass Addendum

Detailed ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`. GLM4-MoE path additions: `#8224`, `#8456`, `#8647`, `#8729`, `#8804`, `#8883`, `#9136`, `#9223`, `#9264`, `#10008`, `#11847`, `#12162`, `#12572`, `#12834`, `#12957`, and `#14668`. Parser/tooling additions: `#12456`, `#13989`, `#15333`, `#15753`, `#15754`, and `#20543`.

Public evidence adds official SGLang GLM45 docs and the LMSYS GLM-4.5 day-one blog; preserve `glm45` parser behavior, EAGLE, thinking budget/custom logit processor, 128k context, native function calling, and MTP deployment facts.
