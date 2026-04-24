# vLLM InternVL3.5 Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for InternVL3.5.

- Status: supported on current mainline

## Key Conclusions

- InternVL3.5 is mostly a processor / encoder / video problem in vLLM.
- Video handling, native HF loading, and backend compatibility dominate the risk surface.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/internvl.py`

## Landed PRs

- [#6514](https://github.com/vllm-project/vllm/pull/6514) `Initialize support for InternVL2 series models`: Historical base for current InternVL runtime code.
- [#18499](https://github.com/vllm-project/vllm/pull/18499) `Initialize video input support for InternVL models`: Added video processing to the family.
- [#23658](https://github.com/vllm-project/vllm/pull/23658) `Enable video support for InternVL3.5 models`: Carried video support into the 3.5 checkpoints.
- [#23742](https://github.com/vllm-project/vllm/pull/23742) `Enable native HF format InternVL support`: Removed reliance on ad hoc checkpoint rewrites.
- [#38049](https://github.com/vllm-project/vllm/pull/38049) `Add torch.compile support for InternVL vision encoder`: Modernized the encoder execution path.

## Matching Skill

- `skills/model-optimization/vllm/vllm-internvl35-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-internvl35-optimization/references/pr-history.md`
