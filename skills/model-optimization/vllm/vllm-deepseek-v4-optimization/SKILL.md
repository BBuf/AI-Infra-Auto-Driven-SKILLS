---
name: vllm-deepseek-v4-optimization
description: PR-backed optimization manual for DeepSeek V4 in vLLM. Use when Codex needs to audit, debug, extend, or document DeepSeek V4 open-radar work in vLLM, including the proposed model module, MTP path, tokenizer/renderer, DSML tool parser, and BF16 persistent-topk follow-up before mainline support lands.
---

# vLLM DeepSeek V4 Optimization

## Overview

This skill tracks DeepSeek V4 in the "not merged yet, but large open PRs exist"
state. The checked mainline commit still does not ship a `DeepseekV4ForCausalLM`
alias, tokenizer, renderer, or tool parser. The useful evidence is therefore in
open PR diffs, not in current-main runtime behavior.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: not landed on current mainline
- Open PRs reviewed: `#40760`, `#40811`, `#40806`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/deepseek-v4/README.zh.md`
  and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the bar.
DeepSeek V4 must stay marked as open radar until the registry alias, config,
tokenizer/renderer, and parser work are all merged on mainline.

## Runtime Surfaces

- Current main evidence: `vllm/vllm/model_executor/models/registry.py`
- Proposed model path in open PR: `vllm/vllm/model_executor/models/deepseek_v4.py`
- Proposed draft path in open PR: `vllm/vllm/model_executor/models/deepseek_v4_mtp.py`
- Proposed tokenizer/render path in open PR:
  `vllm/vllm/tokenizers/deepseek_v4.py`,
  `vllm/vllm/renderers/deepseek_v4.py`
- Proposed parser path in open PR:
  `vllm/vllm/tool_parsers/deepseekv4_tool_parser.py`
- Kernel follow-up in open PR:
  `vllm/csrc/persistent_topk.cuh`, `vllm/csrc/topk.cu`
- Spec-decode follow-up in open PR:
  `vllm/vllm/v1/spec_decode/eagle.py`

## Current Main Summary

- Current mainline has no `DeepseekV4ForCausalLM` alias in
  `vllm/model_executor/models/registry.py`.
- Open PR `#40760` is a large bring-up stack: model class, MTP model,
  DeepSeek-V4 tokenizer/renderer, DSML tool parser, quant config rewrites, and
  spec-decode wiring.
- Open PR `#40811` extends persistent top-k from FP32-only assumptions to
  BF16 input support, which matters for the DeepSeek V4 sparse indexer path.
- Open PR `#40806` fixes a streaming parser leak where a partial
  `<｜DSML｜function_calls>` sentinel could be emitted as normal content.

## Open Radar

- [#40760](https://github.com/vllm-project/vllm/pull/40760) `[New Model] Support DeepseekV4`
- [#40811](https://github.com/vllm-project/vllm/pull/40811) `[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4`
- [#40806](https://github.com/vllm-project/vllm/pull/40806) `[Bugfix] Fix the DSML token leakage in DSV4/3.2`

## Validation Lanes

- Re-check mainline `registry.py` before claiming support.
- If `#40760` merges, validate tokenizer/renderer parity, DSML tool calls, base
  generation, and MTP speculative decoding.
- If `#40811` merges, rerun kernel tests in
  `tests/kernels/test_top_k_per_row.py`, especially BF16 decode and long-context
  cases.
- If `#40806` merges, rerun streaming tool-parser tests to ensure DSML sentinels
  never leak into user-visible content.

## References

- `references/pr-history.md`: diff-reviewed DeepSeek V4 open-radar cards.


## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.
