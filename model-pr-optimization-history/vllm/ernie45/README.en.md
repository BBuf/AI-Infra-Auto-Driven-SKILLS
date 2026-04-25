# vLLM Ernie4.5 / Ernie4.5-VL Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for Ernie4.5 / Ernie4.5-VL.

- Status: supported on current mainline

## Key Conclusions

- Ernie4.5 spans dense, MoE, and VL paths in vLLM.
- The highest-risk work items are shared-expert behavior, VL rotary/timestamp logic, and long-input stability.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/ernie45.py`
- `vllm/vllm/model_executor/models/ernie45_moe.py`
- `vllm/vllm/model_executor/models/ernie45_vl.py`

## Landed PRs

- [#20220](https://github.com/vllm-project/vllm/pull/20220) `Add Ernie4.5 and Ernie4.5MoE Model Support`: Landed text and MoE support.
- [#21717](https://github.com/vllm-project/vllm/pull/21717) `Fix Ernie4_5_MoeForCausalLM shared experts`: Fixed shared-expert correctness.
- [#22514](https://github.com/vllm-project/vllm/pull/22514) `Add Ernie4.5 VL Model Support`: Added the multimodal Ernie4.5-VL lane.
- [#24074](https://github.com/vllm-project/vllm/pull/24074) `Fix Ernie4.5-VL hanging on long inputs`: Closed a production long-input stall.
- [#31274](https://github.com/vllm-project/vllm/pull/31274) `Support video metadata for timestamp rendering`: Improved VL video output fidelity.

## Matching Skill

- `skills/model-optimization/vllm/vllm-ernie45-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-ernie45-optimization/references/pr-history.md`
