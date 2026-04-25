# vLLM MiMo-V2-Flash Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for MiMo-V2-Flash.

- Status: supported on current mainline

## Key Conclusions

- MiMo-V2-Flash is a throughput-oriented MoE serving family in vLLM.
- MTP correctness and the split between older MiMo checkpoints and V2-Flash are the key maintenance points.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/mimo_v2_flash.py`
- `vllm/vllm/model_executor/models/mimo.py`
- `vllm/vllm/model_executor/models/mimo_mtp.py`

## Landed PRs

- [#17433](https://github.com/vllm-project/vllm/pull/17433) `Support MiMo-7B inference with MTP`: Historical base for the MiMo family.
- [#25136](https://github.com/vllm-project/vllm/pull/25136) `Fix MTP inference path for MiMo-7B model`: Closed a concrete draft-path bug.
- [#30836](https://github.com/vllm-project/vllm/pull/30836) `Add MiMo-V2-Flash support`: Landed the dedicated V2-Flash runtime.

## Matching Skill

- `skills/model-optimization/vllm/vllm-mimo-v2-flash-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-mimo-v2-flash-optimization/references/pr-history.md`
