# SGLang Hunyuan 3 Preview Support and Optimization Timeline

Scope: Tencent Hunyuan 3 Preview BF16 cookbook support, hardware TP sizing, reasoning/tool parsers, MTP/EAGLE flags, Blackwell attention backend, and trust-remote-code launch guidance.

Evidence snapshot: SGLang `origin/main` `bca3dd958` (`2026-04-24`). Full dossier: `skills/model-optimization/sglang/sglang-hunyuan3-preview-optimization/references/pr-history.md`.

## Diff-Reviewed PR

#23532 added the Hunyuan 3 Preview cookbook and command generator. Full diff reviewed: `1309` lines, `3` files. The generator maps H200/B200 to TP=8, B300/GB300 to TP=4, adds `hunyuan` parser toggles, MTP/EAGLE flags, `--trust-remote-code`, and Blackwell `--attention-backend trtllm_mha`.

Current caveat: this is docs/recipe support; no new Hunyuan3-specific SGLang model implementation file landed in the PR.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Hunyuan3 Preview` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-23 | [#23532](https://github.com/sgl-project/sglang/pull/23532) | merged | docs: add Hunyuan 3 Preview cookbook | docs/config | `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`, `docs_new/docs.json` |

### File-level PR diff reading notes

### PR #23532 - docs: add Hunyuan 3 Preview cookbook

- Link: https://github.com/sgl-project/sglang/pull/23532
- Status/date: `merged`, created 2026-04-23, merged 2026-04-23; author `JustinTong0323`.
- Diff scope read: `3` files, `+707/-0`; areas: docs/config; keywords: doc, attention, config, eagle, moe, spec, topk, benchmark, expert, flash.
- Code diff details:
  - `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` added +527/-0 (527 lines); hunks: +---; symbols: GPUs
  - `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` added +174/-0 (174 lines); hunks: +export const Hunyuan3PreviewDeployment = () => {; symbols: GPUs
  - `docs_new/docs.json` modified +6/-0 (6 lines); hunks: "pages": [
- Optimization/support interpretation: The concrete diff surface is `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`, `docs_new/docs.json`; keywords observed in patches: doc, attention, config, eagle, moe, spec. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`, `docs_new/docs.json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 1; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
