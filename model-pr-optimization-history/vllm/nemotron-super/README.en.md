# vLLM Nemotron Super / Nano Hybrid Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for Nemotron Super / Nano Hybrid.

- Status: supported on current mainline

## Key Conclusions

- Nemotron is a hybrid family: attention, Mamba, MoE, MTP, and VL all intersect.
- Because of that, graph execution, cache dtype, and quantized MoE correctness are the primary risk areas.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/nemotron_h.py`
- `vllm/vllm/model_executor/models/nemotron_h_mtp.py`
- `vllm/vllm/model_executor/models/nano_nemotron_vl.py`
- `vllm/vllm/model_executor/models/nemotron_vl.py`

## Landed PRs

- [#18863](https://github.com/vllm-project/vllm/pull/18863) `NemotronH support`: Initial NemotronH landing in vLLM.
- [#25863](https://github.com/vllm-project/vllm/pull/25863) `Add MoE support for NemotronH`: Extended the hybrid family to routed experts.
- [#33726](https://github.com/vllm-project/vllm/pull/33726) `Nemotron-H MTP and Mamba Speculative Decoding Support`: Opened the MTP / spec-decode path.
- [#36803](https://github.com/vllm-project/vllm/pull/36803) `E2E Nemotron-3-Super tests`: Added direct Super-family regression coverage.
- [#37803](https://github.com/vllm-project/vllm/pull/37803) `Enable NemotronHPuzzle + NemotronHMTP`: Expanded hybrid and MTP coverage for the family.

## Matching Skill

- `skills/model-optimization/vllm/vllm-nemotron-super-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-nemotron-super-optimization/references/pr-history.md`
