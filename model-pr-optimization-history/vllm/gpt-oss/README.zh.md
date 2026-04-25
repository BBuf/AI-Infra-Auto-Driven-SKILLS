# vLLM GPT-OSS 支持与 PR 历史

本文记录 vLLM 中与 GPT-OSS 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- GPT-OSS is a flagship MoE family in vLLM.
- Quantization, expert-parallel topology, and reasoning/tool parser behavior all evolved quickly and need per-PR tracking.

## 主要代码面

- `vllm/vllm/model_executor/models/gpt_oss.py`

## 已合入 PR

- [#22327](https://github.com/vllm-project/vllm/pull/22327) `Add GPT-OSS model code and config`：Initial GPT-OSS landing in vLLM.
- [#23819](https://github.com/vllm-project/vllm/pull/23819) `Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE`：Opened large-scale GPT-OSS serving topologies.
- [#25246](https://github.com/vllm-project/vllm/pull/25246) `Enable Eagle3 speculative decoding for GPT-OSS model`：Added draft-model acceleration.
- [#25515](https://github.com/vllm-project/vllm/pull/25515) `Structure_Tag support for gpt-oss tool-call in cot`：Improved tool calling in reasoning-mode outputs.
- [#30647](https://github.com/vllm-project/vllm/pull/30647) `Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE`：Targeted the hot MXFP4/MXFP8 path for throughput.

## 配套 skill

- `skills/model-optimization/vllm/vllm-gpt-oss-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-gpt-oss-optimization/references/pr-history.md`
