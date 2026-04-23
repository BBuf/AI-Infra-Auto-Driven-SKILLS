# SGLang GLM VLM/OCR Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: GLM-4V, GLM-4.1V, GLM-4.5V, GLM-4.6V, GLM-Glyph, and GLM-OCR.

## Summary

GLM VLM/OCR risk sits in multimodal processors, vision encoder DP/PP, GLM4V MoE loading, transformers field compatibility, Conv3D/linear projection, and GLM-OCR NextN, not ordinary GLM text MoE.

## Code Surfaces

- `python/sglang/srt/models/glm4v.py`
- `python/sglang/srt/models/glm4v_moe.py`
- `python/sglang/srt/models/glm_ocr.py`
- `python/sglang/srt/models/glm_ocr_nextn.py`
- `python/sglang/srt/multimodal/processors/glm4v.py`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-Glyph.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-OCR.mdx`

## Merged PRs

Key merged PRs: `#8798`, `#9059`, `#9245`, `#9554`, `#10228`, `#12117`, `#14097`, `#14720`, `#14927`, `#14998`, `#17122`, `#17420`, `#17582`, `#18885`, `#20033`, `#20463`, `#20740`, and `#21134`.

## Open Radar

Track `#9349`, `#14662`, `#19728`, and `#22961`.

## Cookbook Evidence

Track `sgl-cookbook#95`, `#131`, and `#136`.

## Next Work

Use OCR-specific examples for GLM-OCR, add defensive transformers field detection, and keep a no-DP baseline for vision encoder DP/PP changes.

## 2026-04-23 Three-Pass Addendum

Detailed ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`. This pass adds `#9884`, `#10147`, `#11166`, `#11388`, `#11922`, `#13228`, `#15205`, `#15434`, `#20282`, and `#21134`. Official GLM VLM docs add FP8/BF16 launch, `--keep-mm-feature-on-device`, `--mm-attention-backend`, `--mm-max-concurrent-calls`, `--mm-enable-dp-encoder`, `SGLANG_USE_CUDA_IPC_TRANSPORT=1`, `SGLANG_VLM_CACHE_SIZE_MB=0`, and GLM thinking budget/custom logit processor coverage.
