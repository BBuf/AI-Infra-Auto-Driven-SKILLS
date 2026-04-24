# vLLM Nemotron Super / Nano Hybrid 支持与 PR 历史

本文记录 vLLM 中与 Nemotron Super / Nano Hybrid 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Nemotron is a hybrid family: attention, Mamba, MoE, MTP, and VL all intersect.
- Because of that, graph execution, cache dtype, and quantized MoE correctness are the primary risk areas.

## 主要代码面

- `vllm/vllm/model_executor/models/nemotron_h.py`
- `vllm/vllm/model_executor/models/nemotron_h_mtp.py`
- `vllm/vllm/model_executor/models/nano_nemotron_vl.py`
- `vllm/vllm/model_executor/models/nemotron_vl.py`

## 已合入 PR

- [#18863](https://github.com/vllm-project/vllm/pull/18863) `NemotronH support`：Initial NemotronH landing in vLLM.
- [#25863](https://github.com/vllm-project/vllm/pull/25863) `Add MoE support for NemotronH`：Extended the hybrid family to routed experts.
- [#33726](https://github.com/vllm-project/vllm/pull/33726) `Nemotron-H MTP and Mamba Speculative Decoding Support`：Opened the MTP / spec-decode path.
- [#36803](https://github.com/vllm-project/vllm/pull/36803) `E2E Nemotron-3-Super tests`：Added direct Super-family regression coverage.
- [#37803](https://github.com/vllm-project/vllm/pull/37803) `Enable NemotronHPuzzle + NemotronHMTP`：Expanded hybrid and MTP coverage for the family.

## 配套 skill

- `skills/model-optimization/vllm/vllm-nemotron-super-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-nemotron-super-optimization/references/pr-history.md`
