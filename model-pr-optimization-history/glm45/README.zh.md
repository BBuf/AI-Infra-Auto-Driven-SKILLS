# SGLang GLM-4.5 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 GLM-4.5、GLM-4.5-Air、GLM-4.5 MoE、A2A/DeepEP、reduce-scatter、NVFP4 padding、GLM45 parser。

结论：GLM-4.5 是 GLM MoE 的基线。后续 GLM-4.6/4.7/5.x 的 MoE、A2A、量化、parser 和平台行为很多都继承自这条线。GLM-4.5V 属于 VLM/OCR lane，只有共享文本 MoE 代码出问题时才回到本页。

## 代码面

- `python/sglang/srt/models/glm4.py`
- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/function_call/glm4_moe_detector.py`
- `docs/basic_usage/glm45.md`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5.mdx`
- `docs_new/src/snippets/autoregressive/glm-45-deployment.jsx`

## 已合入主线 PR

- `#11665`：GLM-4.5 路径禁用 reduce scatter。
- `#11692`：补 GLM-4.5 MoE block 缺失的 A2A backend init。
- `#11800`：重构 GLM-4.5/4.5V。
- `#12497`：移除 NVFP4 weight scale padding assertion。
- `#17714`：GLM45 tool interruption support。
- `#11017`：GLM-4.5 文档更新。

## Open PR 雷达

- `#13711`：GLM-4.5-Air/4.5V RTX Pro 6000 fused-MoE TP2 config。
- `#19106`：GLM4 MoE Lite compressed-tensors。
- `#19728`：ROCm GLM-4.5V-FP8 startup。
- `#20917`：Qwen3/GLM45 `enable_thinking`。
- `#23067`：GLM45 detector `continue_final_message`。

## sgl-cookbook / 文档

- `sgl-cookbook#92`：GLM-4.5 AMD MI300X/MI325X/MI355X。
- `sgl-cookbook#95`：GLM-4.5V AMD 支持，作为同期 release 背景。

## 下一步优化建议

1. 给 GLM-4.5 建 BF16、FP8、NVFP4、A2A/DeepEP、parser streaming 的固定 smoke。
2. parser 的 `continue_final_message` 和 `enable_thinking` 不要和 MoE perf 混在一个 PR 做。
3. GLM-4.5V 问题优先走 GLM VLM/OCR 文档。

## 2026-04-23 三轮复查补充

详细账本见 `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`。GLM4-MoE path 补入 `#8224`、`#8456`、`#8647`、`#8729`、`#8804`、`#8883`、`#9136`、`#9223`、`#9264`、`#10008`、`#11847`、`#12162`、`#12572`、`#12834`、`#12957`、`#14668`；parser/tooling 补入 `#12456`、`#13989`、`#15333`、`#15753`、`#15754`、`#20543`。

外部资料补充 SGLang GLM45 官方文档与 LMSYS GLM-4.5 day-one 博客：需要保留 `glm45` parser、EAGLE、thinking budget/custom logit processor、128k context、native function calling 和 MTP 这些 deployment 事实。
