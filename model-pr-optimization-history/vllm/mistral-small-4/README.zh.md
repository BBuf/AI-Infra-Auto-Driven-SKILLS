# vLLM Mistral Small 4 支持与 PR 历史

本文记录 vLLM 中与 Mistral Small 4 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Mistral Small 4 sits on top of the larger Mistral Large 3 / Ministral runtime work.
- MoE execution and multimodal projector behavior are the main risk areas.

## 主要代码面

- `vllm/vllm/model_executor/models/mistral_large_3.py`
- `vllm/vllm/model_executor/models/mistral_large_3_eagle.py`
- `vllm/vllm/model_executor/models/mistral3.py`

## 已合入 PR

- [#29757](https://github.com/vllm-project/vllm/pull/29757) `Add Mistral Large 3 and Ministral 3`：Landed the runtime family that Mistral Small 4 deployments build on in vLLM.
- [#33174](https://github.com/vllm-project/vllm/pull/33174) `Add support for Mistral Large 3 inference with Flashinfer MoE`：Improved the practical MoE serving path for the same family.

## 配套 skill

- `skills/model-optimization/vllm/vllm-mistral-small-4-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-mistral-small-4-optimization/references/pr-history.md`
