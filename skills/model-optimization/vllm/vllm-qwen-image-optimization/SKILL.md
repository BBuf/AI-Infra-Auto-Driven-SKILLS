---
name: vllm-qwen-image-optimization
description: PR-backed optimization manual for Qwen-Image in vLLM. Use when Codex needs to audit, debug, extend, or document Tracking note for Qwen-Image diffusion generation, which is outside the current vLLM runtime surface.
---

# vLLM Qwen-Image Optimization

## Overview

This skill covers Tracking note for Qwen-Image diffusion generation, which is outside the current vLLM runtime surface.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: not supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/qwen-image/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/registry.py`

## Current Main Summary

- vLLM current mainline does not ship a Qwen-Image diffusion model runtime.
- The family should stay marked unsupported rather than being backfilled from Qwen text/VL support.

## Key Landed PRs

- No landed PR is recorded yet for this family.

## Open Radar

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- Require a real model module and registry alias before changing the status.
