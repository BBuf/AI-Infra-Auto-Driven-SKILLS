# vLLM GLM-4.5 / 4.5V Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for GLM-4.5 / 4.5V.

- Status: supported on current mainline

## Key Conclusions

- The GLM-4.5 lane is where vLLM reorganized the GLM family around text, MoE, and vision variants.
- Most regressions are in MoE gate behavior, tie-word-embedding policy, and vendor-specific fused MoE tuning.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/glm4.py`
- `vllm/vllm/model_executor/models/glm4_moe.py`
- `vllm/vllm/model_executor/models/glm4v.py`

## Landed PRs

- [#22171](https://github.com/vllm-project/vllm/pull/22171) `Modify the organization of GLM series`: Reworked the family layout so 4.5-era models reused a cleaner GLM structure.
- [#22460](https://github.com/vllm-project/vllm/pull/22460) `not tie_word_embeddings for glm-4.5 and glm-4.5v`: Aligned the loader with the real 4.5 checkpoint contract instead of forcing tied embeddings.
- [#22832](https://github.com/vllm-project/vllm/pull/22832) `Modify the gate implementation of glm4_moe`: Changed the GLM4.5 MoE gating path used by text and VL variants.
- [#23695](https://github.com/vllm-project/vllm/pull/23695) `Add triton fused moe config for GLM-4.5-Air-FP8 on B200`: Added a production kernel-tuning lane for the 4.5 Air FP8 deployment path.
- [#24589](https://github.com/vllm-project/vllm/pull/24589) `Add documentation for GLM-4.5 series tool-calling and reasoning parser`: Codified the parser choices needed to serve 4.5 reasoning / tool checkpoints correctly.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-glm45-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-glm45-optimization/references/pr-history.md`
