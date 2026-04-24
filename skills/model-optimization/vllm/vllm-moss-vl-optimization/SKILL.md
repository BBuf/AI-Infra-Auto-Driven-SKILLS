---
name: vllm-moss-vl-optimization
description: PR-backed optimization manual for MOSS-VL in vLLM. Use when Codex needs to audit, debug, extend, or document Tracking note for MOSS-VL, which does not have a native vLLM mainline model module today.
---

# vLLM MOSS-VL Optimization

## Overview

This skill covers Tracking note for MOSS-VL, which does not have a native vLLM mainline model module today.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: not supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/moss-vl/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/registry.py`

## Current Main Summary

- No `moss` or `moss_vl` model module is present in current vLLM mainline.
- Keep the family explicitly marked unsupported until that changes.

## Key Landed PRs

- No landed PR is recorded yet for this family.

## Open Radar

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- Re-run source and PR search before claiming any MOSS-VL support.
