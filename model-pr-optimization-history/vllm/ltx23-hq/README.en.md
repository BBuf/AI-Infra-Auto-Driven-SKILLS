# vLLM LTX 2.3 HQ Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for LTX 2.3 HQ.

- Status: not supported on current mainline

## Key Conclusions

- Current vLLM mainline does not ship a dedicated LTX diffusion/video generation runtime.
- Treat this dossier as an explicit unsupported marker, not as hidden partial support.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/registry.py`

## Landed PRs

- No landed PR is recorded in this dossier yet.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-ltx23-hq-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-ltx23-hq-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `LTX-Video 2.3 HQ` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Public PR search conclusion

- Search terms checked against `vllm-project/vllm`: `LTX-Video, LTX 2.3, ltx23`. No public PR was confirmed as part of this model support or optimization path.
- If a future model, processor, kernel, or benchmark PR appears, add it with the card format below.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
