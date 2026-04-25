---
name: vllm-mixtral-quark-int4fp8-moe-optimization
description: PR-backed optimization manual for Mixtral Quark / INT4-FP8 MoE in vLLM. Use when Codex needs to audit, debug, extend, or document Mixtral MoE, expert parallelism, FP8 / ModelOpt quantization, and EPLB in vLLM, which together form the nearest equivalent to Quark INT4-FP8 Mixtral serving.
---

# vLLM Mixtral Quark / INT4-FP8 MoE Optimization

## Overview

This skill covers Mixtral MoE, expert parallelism, FP8 / ModelOpt quantization, and EPLB in vLLM, which together form the nearest equivalent to Quark INT4-FP8 Mixtral serving.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: partially supported or only adjacent architectures landed on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/mixtral-quark-int4fp8-moe/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/mixtral.py`
- `vllm/vllm/model_executor/layers/fused_moe/layer.py`

## Current Main Summary

- vLLM has rich Mixtral MoE support, but not every Quark-branded checkpoint path is called out by name.
- The closest production evidence is the Mixtral fused-MoE, FP8, ModelOpt, and EPLB work already merged.

## Key Landed PRs

- [#2011](https://github.com/vllm-project/vllm/pull/2011) `Mixtral 8x7B support`: Initial Mixtral model-family support.
- [#2090](https://github.com/vllm-project/vllm/pull/2090) `Optimize Mixtral with expert parallelism`: Added early expert-parallel scaling instead of pure TP execution.
- [#2542](https://github.com/vllm-project/vllm/pull/2542) `Fused MOE for Mixtral`: Brought fused-MoE kernels into the Mixtral serving path.
- [#4527](https://github.com/vllm-project/vllm/pull/4527) `Support MoE FP8 checkpoints for Mixtral`: Added the first serious FP8 checkpoint path for Mixtral MoE.
- [#15961](https://github.com/vllm-project/vllm/pull/15961) `Support ModelOpt quantization of Mixtral model`: Extended the family to NVIDIA ModelOpt quantization flows.
- [#22842](https://github.com/vllm-project/vllm/pull/22842) `Support EPLB for Mixtral Model`: Added expert-parallel load balancing to the Mixtral family.

## Open Radar

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- FP16 / BF16 Mixtral baseline.
- FP8 or ModelOpt quantized checkpoint launch.
- Expert-parallel / EPLB topology checks on multi-GPU setups.
