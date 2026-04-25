# vLLM Mistral Small 4 Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for Mistral Small 4.

- Status: supported on current mainline

## Key Conclusions

- Mistral Small 4 sits on top of the larger Mistral Large 3 / Ministral runtime work.
- MoE execution and multimodal projector behavior are the main risk areas.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/mistral_large_3.py`
- `vllm/vllm/model_executor/models/mistral_large_3_eagle.py`
- `vllm/vllm/model_executor/models/mistral3.py`

## Landed PRs

- [#29757](https://github.com/vllm-project/vllm/pull/29757) `Add Mistral Large 3 and Ministral 3`: Landed the runtime family that Mistral Small 4 deployments build on in vLLM.
- [#33174](https://github.com/vllm-project/vllm/pull/33174) `Add support for Mistral Large 3 inference with Flashinfer MoE`: Improved the practical MoE serving path for the same family.

## Matching Skill

- `skills/model-optimization/vllm/vllm-mistral-small-4-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-mistral-small-4-optimization/references/pr-history.md`
