---
name: vllm-minimax-optimization
description: PR-backed optimization manual for MiniMax M1 / M2 / VL in vLLM. Use when an engineer needs to audit, debug, extend, or document MiniMaxText01, MiniMax-M1, MiniMax-M2, MiniMax-VL-01, LoRA, and Eagle3 support in vLLM.
---

# vLLM MiniMax M1 / M2 / VL Optimization

## Overview

This skill covers MiniMaxText01, MiniMax-M1, MiniMax-M2, MiniMax-VL-01, LoRA, and Eagle3 support in vLLM.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/minimax/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/minimax_text_01.py`
- `vllm/vllm/model_executor/models/minimax_m2.py`
- `vllm/vllm/model_executor/models/minimax_vl_01.py`

## Current Main Summary

- MiniMax support evolved from the text-01 path into M1/M2 and VL variants.
- Today the key production surfaces are linear-attention correctness, VL processor behavior, LoRA, and Eagle3 on M2.

## Key Landed PRs

- [#13454](https://github.com/vllm-project/vllm/pull/13454) `Support MiniMaxText01 model inference`: Landed the original MiniMax text runtime.
- [#16328](https://github.com/vllm-project/vllm/pull/16328) `support MiniMax-VL-01 model`: Added the multimodal MiniMax-VL path.
- [#19677](https://github.com/vllm-project/vllm/pull/19677) `Add support for MiniMaxM1ForCausalLM`: Connected the M1 checkpoint alias to the shared MiniMax runtime.
- [#27535](https://github.com/vllm-project/vllm/pull/27535) `Support MiniMax-M2 Model`: Brought the M2 generation into mainline.
- [#32763](https://github.com/vllm-project/vllm/pull/32763) `Complete LoRA support for MiniMaxM2`: Finished missing adapter wiring in the M2 family.
- [#37512](https://github.com/vllm-project/vllm/pull/37512) `MiniMax-M2: add Eagle3 speculative decoding support`: Enabled the draft-model acceleration path for MiniMax M2.

## Open Optimization Items

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- MiniMax text and VL startup.
- M2 LoRA adapter application.
- Eagle3 speculative decode on M2.
- Reasoning and tool parser streaming for MiniMax checkpoints.
