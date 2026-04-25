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

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Z-Image-Turbo` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Public PR search conclusion

- Search terms checked against `vllm-project/vllm`: `Z-Image, Z Image Turbo, z-image-turbo`. No public PR was confirmed as part of this model support or optimization path.
- If a future model, processor, kernel, or benchmark PR appears, add it with the card format below.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
