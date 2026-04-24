---
name: vllm-qwen36-optimization
description: PR-backed optimization manual for Qwen3.6 in vLLM. Use when Codex needs to audit, debug, extend, or document Tracking note for Qwen3.6-specific documentation; current vLLM mainline does not expose a dedicated Qwen3.6 architecture alias.
---

# vLLM Qwen3.6 Optimization

## Overview

This skill covers Tracking note for Qwen3.6-specific documentation; current vLLM mainline does not expose a dedicated Qwen3.6 architecture alias.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: not supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/qwen36/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/registry.py`

## Current Main Summary

- Qwen3.6 should not be treated as automatically covered just because Qwen3 / Qwen3.5 are supported.
- At the current checked commit, there is no dedicated `Qwen3.6` model module or registry alias.

## Key Landed PRs

- No landed PR is recorded yet for this family.

## Open Radar

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- Re-check the registry and PR search before changing the status.
