# SGLang GLM-4.6/4.7 Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: GLM-4.6, GLM-4.7, GLM-4.7-Flash, GLM4-MoE-Lite, GLM-4.7 parser, NVFP4/MTP, shared expert fusion, and dual-stream MoE.

## Summary

GLM-4.6/4.7 optimization is MoE fusion plus parser/loading correctness. GLM-4.6 introduced shared expert fusion and MoE GEMM overlap; GLM-4.7 adds parser behavior, NVFP4/MTP, Flash/lite import formats, and AMD/NPU lanes.

## Code Surfaces

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/function_call/glm47_moe_detector.py`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7-Flash.mdx`
- `test/registered/8-gpu-models/test_glm_46.py`

## Merged PRs

Key merged PRs: `#13786`, `#13873`, `#14585`, `#15333`, `#15520`, `#15753`, `#15754`, `#17166`, `#17247`, `#21403`, `#21534`, `#21851`, `#22509`, `#22720`, and `#22823`.

## Open Radar

Track `#11951`, `#17869`, `#18930`, `#19040`, `#19106`, `#22315`, and `#22801`.

## Cookbook Evidence

Track `sgl-cookbook#93`, `#94`, and `#109`.

## Next Work

Add streaming/complex-schema/empty-name parser tests, validate GLM-4.7-Flash quantized loading and draft quant config propagation, and pair shared-expert fusion changes with logits and performance evidence.

## 2026-04-23 Three-Pass Addendum

Detailed ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`. This pass adds `#13989`, `#14668`, `#19246`, `#20543`, `#21135`, and `#21660`. The LMSYS/Novita GLM4-MoE production blog ties this lane to shared-expert fusion, QKNorm fusion, async transfer, suffix decoding, and EAGLE/speculative decoding; validate suffix decoding separately from shared-expert fusion.
