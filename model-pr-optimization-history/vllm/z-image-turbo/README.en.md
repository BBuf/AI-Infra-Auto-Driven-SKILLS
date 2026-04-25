# vLLM Z-Image-Turbo Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Z-Image-Turbo.

- Status: not supported on current mainline

## Key Conclusions

- vLLM current mainline does not ship a Z-Image diffusion runtime.
- Keep the family explicitly unsupported instead of inferring support from generic multimodal work.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/registry.py`

## Landed PRs

- No landed PR is recorded in this dossier yet.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-z-image-turbo-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-z-image-turbo-optimization/references/pr-history.md`
