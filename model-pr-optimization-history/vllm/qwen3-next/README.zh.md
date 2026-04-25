# vLLM Qwen3-Next 支持与 PR 历史

本文记录 vLLM 中与 Qwen3-Next 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- Qwen3-Next is its own runtime family because of Gated DeltaNet attention and its MTP path.
- The practical risks are PP, MTP varlen handling, quantized shared-expert naming, and GDN-specific CUDA graph bugs.

## 主要代码面

- `vllm/vllm/model_executor/models/qwen3_next.py`
- `vllm/vllm/model_executor/models/qwen3_next_mtp.py`

## 已合入 PR

- [#24709](https://github.com/vllm-project/vllm/pull/24709) `Fix Qwen3-Next PP`：Corrected pipeline-parallel execution on Qwen3-Next.
- [#24957](https://github.com/vllm-project/vllm/pull/24957) `Fix the varlen issue in qwen3-next MTP implementation`：Removed a concrete MTP correctness bug on variable-length batches.
- [#24960](https://github.com/vllm-project/vllm/pull/24960) `Add prefixes to shared_expert in qwen3-next`：Fixed ignored-parameter and quantized weight loading for shared experts.
- [#25743](https://github.com/vllm-project/vllm/pull/25743) `Fix cuda graph capture bug in GDN metadata and a stride bug`：Stabilized GDN execution under CUDA graphs.
- [#31722](https://github.com/vllm-project/vllm/pull/31722) `Speed-up of GDN attention decode part`：Improved decode throughput on the GDN attention path.
- [#33657](https://github.com/vllm-project/vllm/pull/33657) `Initial support for GDN attention on Qwen3-next/Qwen3.5 (XPU)`：Extended the family beyond CUDA with XPU GDN coverage.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-qwen3-next-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen3-next-optimization/references/pr-history.md`
