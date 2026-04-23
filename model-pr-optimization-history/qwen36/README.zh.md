# SGLang Qwen3.6 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 Qwen3.6-35B-A3B、Qwen3.6-27B dense、FP8/BF16 部署、multimodal、thinking preservation、MTP、Mamba scheduler、Qwen3 reasoning parser 与 Qwen3-Coder tool parser。

结论：Qwen3.6 当前主要是文档/部署层和共享 hybrid Qwen runtime 的组合，不应急着新增专属 runtime fork。排查要先看 Qwen3-Next/Qwen3.5/Qwen VLM/Qwen3-Coder parser 这些共享路径。

## 代码面

- `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`
- `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`
- `docs_new/docs.json`
- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/function_call/qwen3_coder_detector.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`

## 已合入主线 PR

- `#23034`：修 docs 链接、增加 Qwen3.6，并更新 Qwen3.5/GLM-5 文档。
- `#23467`：FP8 `modules_to_not_convert` dot-boundary match，影响 Qwen3.6 FP8 文档/加载边界。
- `#23486`：增加 Qwen3.6-27B dense cookbook。

## Open PR 雷达

- `#23474`：hybrid linear-attention 模型 `--cpu-offload-gb` 修复。
- 继续跟踪 Qwen3-Next 的 GDN/Mamba/MTP open PR。
- 继续跟踪 Qwen VLM 的 multimodal encoder open PR。

## sgl-cookbook / 文档

- `sgl-cookbook#245`：Qwen cookbook 刷新。

## 下一步优化建议

1. 不要先写 Qwen3.6 专属模型类；先证明 shared runtime 不能覆盖。
2. 把 text-only、image、video、reasoning、tool-call、MTP 六类请求做成最小 smoke 集。
3. CPU offload 和 hybrid cache 问题优先复用 Qwen3-Next 验证路径。

## 2026-04-23 三轮复查补充

详细账本见 `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`。复查后仍未发现专属 `qwen3_6.py` runtime 面；直接相关 PR 保持为 `#23034`、`#23467`、`#23486`。后续补漏要继续跟 Qwen3-Next、Qwen3.5、Qwen VLM 和 Qwen3-Coder parser 这些共享路径。
