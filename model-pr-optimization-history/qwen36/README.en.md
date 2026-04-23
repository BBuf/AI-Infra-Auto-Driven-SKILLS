# SGLang Qwen3.6 Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: Qwen3.6-35B-A3B, Qwen3.6-27B dense, FP8/BF16 deployment, multimodal input, thinking preservation, MTP, Mamba scheduler, Qwen3 reasoning parser, and Qwen3-Coder tool parser.

## Summary

Qwen3.6 is currently a docs/deployment layer over shared hybrid Qwen runtime. Do not add a dedicated runtime fork until Qwen3-Next, Qwen3.5, Qwen VLM, and Qwen3-Coder parser paths are ruled out.

## Code Surfaces

- `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`
- `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`
- `docs_new/docs.json`
- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/function_call/qwen3_coder_detector.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`

## Merged PRs

- `#23034`: add Qwen3.6 docs and update Qwen3.5/GLM-5 docs.
- `#23467`: dot-boundary FP8 `modules_to_not_convert`.
- `#23486`: add Qwen3.6-27B dense cookbook.

## Open Radar

Track `#23474` for hybrid linear-attention CPU offload, plus Qwen3-Next and Qwen VLM open PRs.

## Cookbook Evidence

- `sgl-cookbook#245`: Qwen cookbook refresh.

## Next Work

Build smoke tests for text-only, image, video, reasoning, tool calls, and MTP. Reuse Qwen3-Next validation for CPU offload and hybrid cache issues.

## 2026-04-23 Three-Pass Addendum

Detailed ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`. The audit still found no dedicated `qwen3_6.py` runtime surface; direct PRs remain `#23034`, `#23467`, and `#23486`. Continue tracking shared Qwen3-Next, Qwen3.5, Qwen VLM, and Qwen3-Coder parser paths for Qwen3.6 work.
