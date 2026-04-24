---
name: vllm-ltx23-hq-optimization
description: PR-backed optimization manual for LTX 2.3 HQ in vLLM. Use when Codex needs to audit, debug, extend, or document Tracking note for LTX 2.3 HQ diffusion/video style models, which are outside current vLLM autoregressive runtime coverage.
---

# vLLM LTX 2.3 HQ Optimization

## Overview

This skill covers Tracking note for LTX 2.3 HQ diffusion/video style models, which are outside current vLLM autoregressive runtime coverage.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: not supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/ltx23-hq/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/registry.py`

## Current Main Summary

- Current vLLM mainline does not ship a dedicated LTX diffusion/video generation runtime.
- Treat this dossier as an explicit unsupported marker, not as hidden partial support.

## Key Landed PRs

- No landed PR is recorded yet for this family.

## Open Radar

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- Keep the family marked unsupported until a real model module and registry alias land.
