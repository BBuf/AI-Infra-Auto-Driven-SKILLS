# vLLM Ernie4.5 / Ernie4.5-VL 支持与 PR 历史

本文记录 vLLM 中与 Ernie4.5 / Ernie4.5-VL 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Ernie4.5 spans dense, MoE, and VL paths in vLLM.
- The highest-risk work items are shared-expert behavior, VL rotary/timestamp logic, and long-input stability.

## 主要代码面

- `vllm/vllm/model_executor/models/ernie45.py`
- `vllm/vllm/model_executor/models/ernie45_moe.py`
- `vllm/vllm/model_executor/models/ernie45_vl.py`

## 已合入 PR

- [#20220](https://github.com/vllm-project/vllm/pull/20220) `Add Ernie4.5 and Ernie4.5MoE Model Support`：Landed text and MoE support.
- [#21717](https://github.com/vllm-project/vllm/pull/21717) `Fix Ernie4_5_MoeForCausalLM shared experts`：Fixed shared-expert correctness.
- [#22514](https://github.com/vllm-project/vllm/pull/22514) `Add Ernie4.5 VL Model Support`：Added the multimodal Ernie4.5-VL lane.
- [#24074](https://github.com/vllm-project/vllm/pull/24074) `Fix Ernie4.5-VL hanging on long inputs`：Closed a production long-input stall.
- [#31274](https://github.com/vllm-project/vllm/pull/31274) `Support video metadata for timestamp rendering`：Improved VL video output fidelity.

## 配套 skill

- `skills/model-optimization/vllm/vllm-ernie45-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-ernie45-optimization/references/pr-history.md`
