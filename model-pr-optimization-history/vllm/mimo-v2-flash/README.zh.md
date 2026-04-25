# vLLM MiMo-V2-Flash 支持与 PR 历史

本文记录 vLLM 中与 MiMo-V2-Flash 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- MiMo-V2-Flash is a throughput-oriented MoE serving family in vLLM.
- MTP correctness and the split between older MiMo checkpoints and V2-Flash are the key maintenance points.

## 主要代码面

- `vllm/vllm/model_executor/models/mimo_v2_flash.py`
- `vllm/vllm/model_executor/models/mimo.py`
- `vllm/vllm/model_executor/models/mimo_mtp.py`

## 已合入 PR

- [#17433](https://github.com/vllm-project/vllm/pull/17433) `Support MiMo-7B inference with MTP`：Historical base for the MiMo family.
- [#25136](https://github.com/vllm-project/vllm/pull/25136) `Fix MTP inference path for MiMo-7B model`：Closed a concrete draft-path bug.
- [#30836](https://github.com/vllm-project/vllm/pull/30836) `Add MiMo-V2-Flash support`：Landed the dedicated V2-Flash runtime.

## 配套 skill

- `skills/model-optimization/vllm/vllm-mimo-v2-flash-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-mimo-v2-flash-optimization/references/pr-history.md`
