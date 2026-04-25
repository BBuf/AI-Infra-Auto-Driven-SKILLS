# SGLang DeepSeek-V4 Support and Optimization Timeline

Scope: DeepSeek-V4 Flash/Pro cookbook support, H200 FP8 checkpoint selection, Blackwell FP4 recipes, DeepEP budgets, MTP/EAGLE, CP, and PD-disaggregation command generation.

Evidence snapshot: SGLang `origin/main` `bca3dd958` (`2026-04-24`). Full dossier: `skills/model-optimization/sglang/sglang-deepseek-v4-optimization/references/pr-history.md`.

## Diff-Reviewed PRs

- #23605 added the DeepSeek-V4 cookbook and command generator. Diff reviewed: `2113` lines, `4` files. It introduced the Flash/Pro, B200/GB300/H200 matrix and commented-out unverified recipe behavior.
- #23617 changed H200 Flash from a placeholder to `sgl-project/DeepSeek-V4-Flash-FP8`. Diff reviewed: `33` lines, `1` file.
- #23628 added a cookbook note warning H200 users to use `sgl-project` checkpoints. Diff reviewed: `24` lines, `1` file.
- #23622 added the Docker launch skeleton, expanded verified recipe rows, and cleaned CP `--max-running-requests`. Diff reviewed: `292` lines, `2` files.
- #23634 changed H200 Pro to `sgl-project/DeepSeek-V4-Pro-FP8`. Diff reviewed: `26` lines, `1` file.

## Current Contract

H200 commands must use SGLang FP8 checkpoints; Blackwell commands use DeepSeek Flash/Pro repos. Unverified command-generator cells are intentionally commented out. Parser toggles are `deepseek-v4` for reasoning and `deepseekv4` for tool calls.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `DeepSeek V4` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-24 | [#23605](https://github.com/sgl-project/sglang/pull/23605) | merged | Add DeepSeek V4 cookbook | docs/config | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx` |
| 2026-04-24 | [#23617](https://github.com/sgl-project/sglang/pull/23617) | merged | Further update Deepseek V4 docs | docs/config | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-24 | [#23622](https://github.com/sgl-project/sglang/pull/23622) | merged | Again update DeepSeek V4 cookbook | docs/config | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23628](https://github.com/sgl-project/sglang/pull/23628) | merged | [codex] docs: note H200 DeepSeek-V4 checkpoint | docs/config | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23634](https://github.com/sgl-project/sglang/pull/23634) | merged | Update pro fp8 checkpoint in DeepSeek V4 cookbook | docs/config | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |

### File-level PR diff reading notes

### PR #23605 - Add DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23605
- Status/date: `merged`, created 2026-04-24, merged 2026-04-24; author `wisclmy0611`.
- Diff scope read: `4` files, `+1024/-1`; areas: docs/config; keywords: doc, attention, config, cuda, deepep, eagle, expert, flash, fp4, fp8.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0 (569 lines); hunks: +export const DeepSeekV4Deployment = () => {; symbols: uses
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunks: +---
  - `docs_new/cookbook/autoregressive/intro.mdx` modified +1/-1 (2 lines); hunks: metatags:
  - `docs_new/docs.json` modified +1/-0 (1 lines); hunks: {
- Optimization/support interpretation: The concrete diff surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`; keywords observed in patches: doc, attention, config, cuda, deepep, eagle. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23617 - Further update Deepseek V4 docs

- Link: https://github.com/sgl-project/sglang/pull/23617
- Status/date: `merged`, created 2026-04-24, merged 2026-04-24; author `fzyzcjy`.
- Diff scope read: `1` files, `+5/-6`; areas: docs/config; keywords: doc, flash, fp4, fp8, kv, spec.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6 (11 lines); hunks: export const DeepSeekV4Deployment = () => {
- Optimization/support interpretation: The concrete diff surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; keywords observed in patches: doc, flash, fp4, fp8, kv, spec. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23622 - Again update DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23622
- Status/date: `merged`, created 2026-04-24, merged 2026-04-24; author `fzyzcjy`.
- Diff scope read: `2` files, `+32/-9`; areas: docs/config; keywords: doc, cache, cuda, deepep, kv, router, spec, test.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9 (28 lines); hunks: export const DeepSeekV4Deployment = () => {; export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: Please refer to the [official SGLang installation guide](../../../docs/get-start
- Optimization/support interpretation: The concrete diff surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; keywords observed in patches: doc, cache, cuda, deepep, kv, router. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23628 - [codex] docs: note H200 DeepSeek-V4 checkpoint

- Link: https://github.com/sgl-project/sglang/pull/23628
- Status/date: `merged`, created 2026-04-24, merged 2026-04-24; author `zijiexia`.
- Diff scope read: `1` files, `+4/-0`; areas: docs/config; keywords: config, doc, spec.
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: Please refer to the [official SGLang installation guide](../../../docs/get-start
- Optimization/support interpretation: The concrete diff surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; keywords observed in patches: config, doc, spec. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23634 - Update pro fp8 checkpoint in DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23634
- Status/date: `merged`, created 2026-04-24, merged 2026-04-24; author `fzyzcjy`.
- Diff scope read: `1` files, `+2/-2`; areas: docs/config; keywords: doc, flash, fp4, fp8, kv, spec.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2 (4 lines); hunks: export const DeepSeekV4Deployment = () => {
- Optimization/support interpretation: The concrete diff surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; keywords observed in patches: doc, flash, fp4, fp8, kv, spec. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 5; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
