---
name: sglang-deepseek-v4-optimization
description: PR-backed and current-main optimization manual for DeepSeek-V4 in SGLang. Use when Codex needs to audit or extend DeepSeek-V4 Flash/Pro serving recipes, FP4-vs-FP8 checkpoint selection, H200/B200/GB300 launch commands, DeepEP dispatch-token budgets, context-parallel and PD-disaggregation recipes, MTP/EAGLE settings, or DeepSeek-V4 parser flags.
---

# SGLang DeepSeek-V4 Optimization

## Overview

DeepSeek-V4 is documented as a cookbook/command-generator lane in SGLang current main. The inspected PRs are docs/snippet-only, but they are serving-critical because they encode the hardware/checkpoint matrix, DeepEP environment budgets, MTP settings, verified recipes, and parser flags users copy into production.

Current evidence snapshot:

- SGLang `origin/main`: `bca3dd958` on `2026-04-24`
- Main docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`
- Command generator: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`
- Diff-reviewed PRs: #23605, #23617, #23628, #23622, #23634

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Capture:

- variant: DeepSeek-V4-Flash or DeepSeek-V4-Pro
- hardware: B200, GB300, or H200
- checkpoint dtype: Blackwell FP4 mixed checkpoint or H200 `sgl-project/*-FP8`
- recipe: low-latency, balanced, max-throughput, context-parallel, or PD-disagg
- parser flags: `--reasoning-parser deepseek-v4`, `--tool-call-parser deepseekv4`
- MTP settings and `SGLANG_ENABLE_SPEC_V2`
- DeepEP dispatch-token env budget and `--max-running-requests`

## Core Principle

Treat the DeepSeek-V4 docs as an executable deployment matrix, not ordinary prose.

- H200 must use `sgl-project/DeepSeek-V4-*-FP8`, not the default DeepSeek FP4-mixed repos.
- Blackwell uses the DeepSeek Flash/Pro repos directly.
- Unverified generator cells are intentionally rendered as commented shell no-ops.
- Recipe verification state is part of the serving contract.

## PR Dossier Rule

Before adding DeepSeek-V4 evidence, open the PR diff/source and update `references/pr-history.md` with motivation, key implementation, short code/config excerpts, reviewed files, and validation implications. Docs-only PRs still need exact command/config lines.

## Validation Lanes

- B200 Flash/Pro low-latency, balanced, max-throughput, and CP recipe command generation.
- H200 Flash low-latency, balanced, and max-throughput command generation with `sgl-project/DeepSeek-V4-Flash-FP8`.
- H200 Pro command generation with `sgl-project/DeepSeek-V4-Pro-FP8` and TP=16 multinode note.
- Parser flags toggled on/off in generated commands.
- PD-disagg commands checked for router port and commented/uncommented state.

## References

- `references/pr-history.md`: diff-reviewed DeepSeek-V4 PR cards.
