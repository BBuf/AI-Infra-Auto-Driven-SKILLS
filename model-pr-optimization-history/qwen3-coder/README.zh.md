# SGLang Qwen3-Coder 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 Qwen3-Coder-480B-A35B、Qwen3-Coder-Next、`qwen3_coder` tool parser、streaming tool arguments、NVFP4/FP8、AMD/NPU/Blackwell cookbook。

结论：Qwen3-Coder 需要把 parser 正确性和模型性能分开。Qwen3-Coder-Next 的 runtime 多数归入 Qwen3-Next lane，但 parser 是独立高风险路径，并且被 Qwen3.6 文档复用。

## 代码面

- `python/sglang/srt/function_call/qwen3_coder_detector.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder-Next.mdx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-next-deployment.jsx`
- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `test/registered/amd/test_qwen3_coder_next_eval_mi35x.py`

## 已合入主线 PR

- `#16744`：新增 `qwen3_coder_detector`。
- `#17965`：Triton TP MoE SwapAB，覆盖 Qwen3-Coder。
- `#18195`：Qwen3-Coder-Next FP8 H100 TP=2 fused MoE config。
- `#18224`：ModelOpt Qwen3-Next-Coder NVFP4。
- `#18355`：AMD Qwen3-Coder-Next。
- `#18608`：AMD Qwen3-Coder-Next MI35x accuracy/functionality。
- `#18700`：NPU Qwen3-Coder-Next weight transpose。
- `#19736`：AMD Qwen3-Coder-Next AITER backend k/v scale args。

## Open PR 雷达

- `#13411`：schema-aware Qwen3CoderDetector。
- `#13979`：Qwen3-Coder-480B nightly tests。
- `#21829`：incremental streaming tool-call arguments。

## sgl-cookbook / 文档

- `sgl-cookbook#86`：Qwen3-Coder-480B-A35B AMD MI300X。
- `sgl-cookbook#112`：MI325X/MI355X。
- `sgl-cookbook#143`：Qwen3-Coder-Next。
- `sgl-cookbook#174`：NVIDIA B200/GB200。

## 下一步优化建议

1. 先补 parser-only 单测：复杂 schema、空函数名、多工具、增量 streaming。
2. Qwen3-Coder-Next runtime 改动应同步跑 Qwen3-Next MTP/cache 测试。
3. Cookbook 命令必须显式带 `--tool-call-parser qwen3_coder`。

## 2026-04-23 三轮复查补充

详细账本见 `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`。parser 历史补入 `#8357`、`#8371`、`#8445`、`#12226`、`#13163`。这几条主要覆盖 XML-like grammar、streaming、unknown tool calls、EBNF Composer 移除和共享 detector 语义变化。
