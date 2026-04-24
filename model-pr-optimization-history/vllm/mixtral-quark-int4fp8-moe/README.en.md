# vLLM Mixtral Quark / INT4-FP8 MoE Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Mixtral Quark / INT4-FP8 MoE.

- Status: partially supported or only adjacent architectures landed on current mainline

## Key Conclusions

- vLLM has rich Mixtral MoE support, but not every Quark-branded checkpoint path is called out by name.
- The closest production evidence is the Mixtral fused-MoE, FP8, ModelOpt, and EPLB work already merged.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/mixtral.py`
- `vllm/vllm/model_executor/layers/fused_moe/layer.py`

## Landed PRs

- [#2011](https://github.com/vllm-project/vllm/pull/2011) `Mixtral 8x7B support`: Initial Mixtral model-family support.
- [#2090](https://github.com/vllm-project/vllm/pull/2090) `Optimize Mixtral with expert parallelism`: Added early expert-parallel scaling instead of pure TP execution.
- [#2542](https://github.com/vllm-project/vllm/pull/2542) `Fused MOE for Mixtral`: Brought fused-MoE kernels into the Mixtral serving path.
- [#4527](https://github.com/vllm-project/vllm/pull/4527) `Support MoE FP8 checkpoints for Mixtral`: Added the first serious FP8 checkpoint path for Mixtral MoE.
- [#15961](https://github.com/vllm-project/vllm/pull/15961) `Support ModelOpt quantization of Mixtral model`: Extended the family to NVIDIA ModelOpt quantization flows.
- [#22842](https://github.com/vllm-project/vllm/pull/22842) `Support EPLB for Mixtral Model`: Added expert-parallel load balancing to the Mixtral family.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-mixtral-quark-int4fp8-moe-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-mixtral-quark-int4fp8-moe-optimization/references/pr-history.md`
