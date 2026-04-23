# Qwen3.6 PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched paths: Qwen3.6 docs/snippets, Qwen3-Next/Qwen3.5 adjacent runtime files, Qwen VLM docs, Qwen3-Coder parser.
- Searched PR terms: `Qwen3.6`, `Qwen36`, `qwen36`, `qwen3_6`.

## Runtime and Docs Surfaces

- `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`
- `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`
- `docs_new/docs.json`
- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/function_call/qwen3_coder_detector.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`

## Merged/Current-Main PRs

- [#23034](https://github.com/sgl-project/sglang/pull/23034): fix links, add Qwen3.6, and update Qwen3.5/GLM-5 docs.
- [#23467](https://github.com/sgl-project/sglang/pull/23467): dot-boundary match in FP8 `modules_to_not_convert`, relevant to Qwen3.6 FP8 docs.
- [#23486](https://github.com/sgl-project/sglang/pull/23486): add Qwen3.6-27B dense cookbook docs.

## Open PR Radar

- [#23474](https://github.com/sgl-project/sglang/pull/23474): CPU offload bugfix for hybrid linear-attention models.
- Follow Qwen3-Next open PRs for GDN/Mamba/MTP changes.
- Follow Qwen VLM open PRs for multimodal encoder changes.

## Cookbook Evidence

- sgl-cookbook [#245](https://github.com/sgl-project/sgl-cookbook/pull/245): Qwen cookbook refresh.

## Validation Notes

- Qwen3.6 currently depends more on docs/config accuracy than on a dedicated model class.
- Because docs recommend both reasoning and tool-call parsers, parser validation must cover the combined mode.
- Multimodal examples should be tested separately from text-only MTP.

## Three-Pass Completeness Addendum (2026-04-23)

Full audit ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`.

Qwen3.6 remains a docs/config/shared-runtime lane as of the audited SGLang snapshot. Keep [#23034](https://github.com/sgl-project/sglang/pull/23034), [#23467](https://github.com/sgl-project/sglang/pull/23467), and [#23486](https://github.com/sgl-project/sglang/pull/23486) as the direct Qwen3.6 entries, then re-check Qwen3-Next, Qwen3.5, Qwen VLM, and Qwen3-Coder parser PRs before changing runtime behavior.

The three-pass audit found no dedicated `python/sglang/srt/models/qwen3_6.py` surface. Do not create a Qwen3.6-specific model fork unless shared Qwen3-Next/Qwen3.5/Qwen VLM runtime paths are proven insufficient.
