# vLLM Qwen-Image Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Qwen-Image.

- Status: not supported on current mainline

## Key Conclusions

- vLLM current mainline does not ship a Qwen-Image diffusion model runtime.
- The family should stay marked unsupported rather than being backfilled from Qwen text/VL support.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/registry.py`

## Landed PRs

- No landed PR is recorded in this dossier yet.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-qwen-image-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen-image-optimization/references/pr-history.md`
