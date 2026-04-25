---
name: sglang-moss-vl-optimization
description: PR-backed and current-main optimization manual for Moss-VL in SGLang. Use when Codex needs to audit or extend Moss-VL multimodal runtime support, Qwen3VL-like vision encoder plumbing, cross-attention custom masks, vision position ids, image/video processor behavior, conversation template registration, flashinfer prefill requirements, or Moss-VL weight loading.
---

# SGLang Moss-VL Optimization

## Overview

Moss-VL landed as a native SGLang runtime path in #23454. It is not a thin alias: the PR added a dedicated model file, processor, multimodal scheduling fields, prompt template, encoder-prefix handling, and a flashinfer prefill requirement for cross-attention custom masks.

Current evidence snapshot:

- SGLang `origin/main`: `bca3dd958` on `2026-04-24`
- Primary PR: #23454
- Runtime files: `moss_vl.py`, `multimodal/processors/moss_vl.py`, `schedule_batch.py`, `conversation.py`, `server_args.py`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Capture:

- image vs video input count and shape
- `grid_thw`, `media_nums_per_sample`, `visible_frame_counts`, and `vision_position_ids`
- whether encoder KV is cached
- prefill backend; Moss-VL requires flashinfer prefill
- conversation template selection (`moss-vl`)
- MRoPE enabled or disabled
- cross-attention custom mask behavior for extend vs decode

## Core Principle

Debug Moss-VL at the multimodal boundary.

- Encoder-prefix placeholder tokens are stripped before text extend.
- Vision tokens include per-frame separator tokens.
- Frame-level visibility comes from processor `cross_attention_mask` and becomes a packed flashinfer custom mask.
- Heavy vision tensors should be released after encoder KV is produced.

## PR Dossier Rule

Before adding Moss-VL evidence, open the PR diff/source and update `references/pr-history.md` with motivation, key implementation, real code excerpts, reviewed files, and validation implications.

## Validation Lanes

- text-only Moss-VL request, to verify cross-attention skip behavior
- single image and multi-image prefill
- video input from URL/path/data URI
- prefix-cache reuse after encoder KV is cached
- extend-stage cross-attention custom mask with frame-level visibility
- `--prefill-attention-backend flashinfer` and failure when another backend is forced

## References

- `references/pr-history.md`: diff-reviewed Moss-VL PR cards.
