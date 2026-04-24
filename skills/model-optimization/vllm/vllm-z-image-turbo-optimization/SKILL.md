---
name: vllm-z-image-turbo-optimization
description: PR-backed optimization manual for Z-Image-Turbo in vLLM. Use when Codex needs to audit, debug, extend, or document Tracking note for Z-Image-Turbo diffusion generation, which is outside vLLM mainline today.
---

# vLLM Z-Image-Turbo Optimization

## Overview

This skill covers Tracking note for Z-Image-Turbo diffusion generation, which is outside vLLM mainline today.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: not supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/z-image-turbo/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/registry.py`

## Current Main Summary

- vLLM current mainline does not ship a Z-Image diffusion runtime.
- Keep the family explicitly unsupported instead of inferring support from generic multimodal work.

## Key Landed PRs

- No landed PR is recorded yet for this family.

## Open Radar

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- Require a real model implementation before changing the status.
