# vLLM GLM-4.6 / 4.7 Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for GLM-4.6 / 4.7.

- Status: supported on current mainline

## Key Conclusions

- The 4.6/4.7 generation mainly extends the 4.5 base with new tuning tables, parser behavior, and Lite variants.
- AWQ / Marlin compatibility and content-normalization in tool parsing are the recurring pitfalls.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/glm4.py`
- `vllm/vllm/model_executor/models/glm4_moe.py`
- `vllm/vllm/model_executor/models/glm4v.py`

## Landed PRs

- [#26818](https://github.com/vllm-project/vllm/pull/26818) `Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on B200`: Added fused-MoE tuning configs for the new Blackwell deployment lane.
- [#30210](https://github.com/vllm-project/vllm/pull/30210) `Fix glm46 awq marlin moe compatibility`: Closed an incompatibility between GLM-4.6 AWQ checkpoints and Marlin MoE assumptions.
- [#30876](https://github.com/vllm-project/vllm/pull/30876) `GLM-4.7 Tool Parser and Doc Update`: Brought parser behavior and docs up to date for 4.7 / 4.7-Flash.
- [#31386](https://github.com/vllm-project/vllm/pull/31386) `GLM Model support for GLM-Lite`: Extended the same runtime family to the Lite checkpoint line.
- [#37386](https://github.com/vllm-project/vllm/pull/37386) `Improve tool call parsing and content normalization for glm47`: Fixed concrete parsing errors that surfaced in newer GLM-4.7 outputs.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-glm46-glm47-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-glm46-glm47-optimization/references/pr-history.md`
