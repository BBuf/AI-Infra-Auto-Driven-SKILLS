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
