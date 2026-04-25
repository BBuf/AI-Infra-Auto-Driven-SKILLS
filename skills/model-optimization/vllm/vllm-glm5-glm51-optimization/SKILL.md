---
name: vllm-glm5-glm51-optimization
description: PR-backed optimization manual for GLM-5 / GLM-5.1 in vLLM. Use when Codex needs to audit, debug, extend, or document the current partial GLM-5 bring-up in vLLM, especially the `GlmMoeDsaForCausalLM` aliasing into the DeepSeek-V2/V3 runtime, rope interleave handling, and GLM-5 MTP correctness.
---

# vLLM GLM-5 / 5.1 Optimization

## Overview

GLM-5 support in vLLM is not a standalone `glm5.py` runtime. The current landed
path adapts GLM-5 into the existing DeepSeek-V2/V3 MLA/MoE implementation and
then fixes the GLM-5 MTP draft-model correctness bug separately.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: partially supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/glm5-glm51/README.zh.md`
  and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the bar.
For GLM-5, do not pretend support comes from the older `glm4*` modules; the
actual landed path is the DeepSeek-based alias plus MTP follow-up.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/models/registry.py`
- `vllm/vllm/config/speculative.py`
- `vllm/vllm/transformers_utils/model_arch_config_convertor.py`
- `vllm/vllm/v1/spec_decode/eagle.py`
- `vllm/tests/models/registry.py`
- `vllm/tests/models/test_initialization.py`

## Current Main Summary

- `#34124` adds `GlmMoeDsaForCausalLM` as a DeepSeek-V2-derived runtime rather
  than introducing a dedicated GLM-5 file.
- The same PR also flips rope style through
  `indexer_rope_interleave` and teaches speculative config conversion to treat
  `glm_moe_dsa` like a DeepSeek MTP family.
- `#34385` fixes GLM-5 MTP accuracy by explicitly sharing the target
  `lm_head` into every MTP layer `shared_head.head`; without that, logits could
  become zero or NaN.

## Key Landed PRs

- [#34124](https://github.com/vllm-project/vllm/pull/34124) `GLM adaptation`
- [#34385](https://github.com/vllm-project/vllm/pull/34385) `Fix MTP accuracy for GLM-5`

## Open Radar

- Re-run PR search before claiming broader GLM-5 router, parser, or multimodal
  support beyond the checked mainline commit.

## Validation Lanes

- Plain GLM-5 launch through the `GlmMoeDsaForCausalLM` alias.
- MTP correctness on held-out prompts after `#34385`.
- Rope/interleaving checks when touching the DeepSeek indexer path, because the
  GLM adaptation changed `is_neox_style` handling in `deepseek_v2.py`.

## References

- `references/pr-history.md`: diff-reviewed GLM-5 / 5.1 cards.


## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.
