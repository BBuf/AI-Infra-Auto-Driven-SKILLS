# SGLang GLM VLM/OCR 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 GLM-4V、GLM-4.1V、GLM-4.5V、GLM-4.6V、GLM-Glyph、GLM-OCR。

结论：GLM VLM/OCR 的主要风险在 multimodal processor、vision encoder DP/PP、GLM4V MoE loading、transformers 字段兼容、Conv3D/linear projection、GLM-OCR NextN，而不是普通 GLM 文本 MoE。

## 代码面

- `python/sglang/srt/models/glm4v.py`
- `python/sglang/srt/models/glm4v_moe.py`
- `python/sglang/srt/models/glm_ocr.py`
- `python/sglang/srt/models/glm_ocr_nextn.py`
- `python/sglang/srt/multimodal/processors/glm4v.py`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-Glyph.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-OCR.mdx`

## 已合入主线 PR

- `#8798`：GLM-4.1V/4.5V 支持。
- `#9059`：GLM-4.1V/4.5V vision transformer dummy-head TP。
- `#9245`：GLM-4.5V 默认 FA3 attention backend。
- `#9554`：GLM45V torch.compile launch bug。
- `#10228`：GLM-4.5V `capture_aux_hidden_states`。
- `#12117`：GLM-4-0414 和 GLM-4.1V refactor。
- `#14097`：GLM-V vision model DP。
- `#14720`：GLM-4.6V PP。
- `#14927`：`glm4v_moe` nightly CI。
- `#14998`：GLM-4.6V MoE transformers version validation。
- `#17122`：GLM-4V bugfix。
- `#17420`：GLM4V `get_rope_index`。
- `#17582`：GLM-OCR 支持。
- `#18885`：`glm_ocr` 不可用时 GLM-4V processor registration。
- `#20033`：GLM4V Conv3D projection 替换为 linear。
- `#20463`、`#20740`：GLM-4.6V vision regression 与 revert。
- `#21134`：GLM-V/OCR transformers 5.x 字段检测和 MTP omission。

## Open PR 雷达

- `#9349`：GLM-4.5V FP8。
- `#14662`：GLM4.6V ktransformers。
- `#19728`：ROCm GLM-4.5V-FP8 startup。
- `#22961`：NPU GLM-4.5V。

## sgl-cookbook / 文档

- `sgl-cookbook#95`：GLM-4.5V AMD MI300X/MI325X/MI355X。
- `sgl-cookbook#131`：GLM-4.5V/4.6V MI325X。
- `sgl-cookbook#136`：GLM-OCR cookbook。

## 下一步优化建议

1. GLM-OCR 必须有 OCR 专属样例，不要只用 caption smoke。
2. transformers 字段兼容应做 defensive detection。
3. vision encoder DP/PP 改动必须保留 no-DP baseline。

## 2026-04-23 三轮复查补充

详细账本见 `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`。本轮补入 `#9884`、`#10147`、`#11166`、`#11388`、`#11922`、`#13228`、`#15205`、`#15434`、`#20282`、`#21134`。官方 GLM VLM 文档补充了 FP8/BF16 launch、`--keep-mm-feature-on-device`、`--mm-attention-backend`、`--mm-max-concurrent-calls`、`--mm-enable-dp-encoder`、`SGLANG_USE_CUDA_IPC_TRANSPORT=1`、`SGLANG_VLM_CACHE_SIZE_MB=0` 和 GLM thinking budget/custom logit processor。
