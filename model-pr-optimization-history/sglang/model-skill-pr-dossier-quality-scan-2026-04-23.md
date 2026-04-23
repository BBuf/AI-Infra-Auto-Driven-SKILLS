# Model Skill PR Dossier Quality Scan - 2026-04-23

Scope: SGLang model optimization skills under `skills/model-optimization/sglang/` and matching histories under `model-pr-optimization-history/sglang/`.

Standard: a model PR dossier is strict-compliant only when the skill-side `references/pr-history.md` uses per-PR cards with motivation, key implementation, key code excerpt, and validation/risk. Title-only PR rows, chronological lists, or grouped narratives are not enough for new production docs.

## Strict-Compliant After This Pass

These skill-side PR histories have explicit per-PR motivation, implementation, and code-fragment fields:

- `sglang-qwen3-core-optimization`
- `sglang-qwen3-next-optimization`
- `sglang-qwen35-optimization`
- `sglang-qwen36-optimization`
- `sglang-qwen3-coder-optimization`
- `sglang-qwen-vlm-omni-asr-optimization`
- `sglang-qwen-image-optimization`
- `sglang-glm45-optimization`
- `sglang-glm46-glm47-optimization`
- `sglang-glm5-glm51-optimization`
- `sglang-glm-vlm-ocr-optimization`

## Legacy Format Requiring Rewrite Before New Publication

These skills contain useful historical research, but their skill-side `references/pr-history.md` files were written before the strict per-PR card requirement and should not be used as final production dossiers without rewrite:

- `sglang-deepseek-v3-r1-optimization`: large chronological table plus grouped code-level narrative; many PRs do not have individual motivation/code-excerpt cards.
- `sglang-deepseek-v31-optimization`: more detailed than a plain list, but not uniformly per-PR carded.
- `sglang-deepseek-v32-optimization`: broad model history and many code paths, but still grouped by subsystem instead of every PR card.
- `sglang-kimi-k2-k25-optimization`: many entries include code blocks and benchmark evidence, but motivation/implementation fields are not consistently explicit for every PR.
- `sglang-minimax-m2-series-optimization`: model history is useful, and the matching `README.zh.md` is more detailed, but the skill-side PR history still has shallow PR bullets and must be rewritten.

## Directory Refactor Result

- SGLang model skills now live under `skills/model-optimization/sglang/`.
- Future vLLM model skills should live under `skills/model-optimization/vllm/`.
- SGLang model PR histories now live under `model-pr-optimization-history/sglang/`.
- Future vLLM model PR histories should live under `model-pr-optimization-history/vllm/`.

## Follow-Up Rule

When a legacy DeepSeek/Kimi/MiniMax skill is next touched, first rewrite its `references/pr-history.md` into strict cards before adding new PRs. Do not append more title-level PR rows.
