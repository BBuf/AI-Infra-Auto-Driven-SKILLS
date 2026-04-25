# vLLM Qwen3.6 Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Qwen3.6.

- Status: not supported on current mainline

## Key Conclusions

- Qwen3.6 should not be treated as automatically covered just because Qwen3 / Qwen3.5 are supported.
- At the current checked commit, there is no dedicated `Qwen3.6` model module or registry alias.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/registry.py`

## Landed PRs

- No landed PR is recorded in this dossier yet.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-qwen36-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen36-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Qwen3.6` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Public PR search conclusion

- Search terms checked against `vllm-project/vllm`: `Qwen3.6, Qwen 3.6`. No public PR was confirmed as part of this model support or optimization path.
- If a future model, processor, kernel, or benchmark PR appears, add it with the card format below.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
