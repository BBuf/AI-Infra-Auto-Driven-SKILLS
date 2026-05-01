---
name: vllm-deepseek-v4-optimization
description: PR-backed optimization manual for DeepSeek V4 in vLLM. Use when an engineer needs to audit, debug, extend, or document DeepSeek V4 current-main support in vLLM, including the model module, MTP path, tokenizer/renderer, DSML tool parser, expert-dtype handling, and BF16 persistent-topk follow-up.
---

# vLLM DeepSeek V4 Optimization

## Overview

This skill tracks DeepSeek V4 as a current-main vLLM model family. The old
open bring-up PR was superseded by the rebased mainline landing, and the
runtime now includes the registry aliases, model/MTP files, tokenizer/renderer,
tool parser, and follow-up handling for FP4-vs-FP8 expert checkpoints.

Evidence snapshot:

- vLLM `origin/main`: `fd74c90d9` on `2026-04-27`
- Support status: landed on current mainline after `#40860`
- Diff-reviewed PRs: `#40760`, `#40860`, `#40806`, `#41006`, `#40811`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/deepseek-v4/README.zh.md`
  and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the bar.
DeepSeek V4 support claims must be tied to current-main files and diff-reviewed
PR cards, not only PR titles.

## Runtime Surfaces

- Registry evidence: `vllm/vllm/model_executor/models/registry.py`
- Model path: `vllm/vllm/model_executor/models/deepseek_v4.py`
- Draft/MTP path: `vllm/vllm/model_executor/models/deepseek_v4_mtp.py`
- Attention/compressor paths:
  `vllm/vllm/model_executor/layers/deepseek_v4_attention.py`,
  `vllm/vllm/model_executor/layers/deepseek_compressor.py`
- Tokenizer/render paths:
  `vllm/vllm/tokenizers/deepseek_v4.py`,
  `vllm/vllm/renderers/deepseek_v4.py`
- Tool parser path:
  `vllm/vllm/tool_parsers/deepseekv4_tool_parser.py`
- Kernel follow-up in open PR:
  `vllm/csrc/persistent_topk.cuh`, `vllm/csrc/topk.cu`

## Current Main Summary

- `#40860` merged the rebased DeepSeek V4 bring-up: model class, MTP model,
  MLA attention, tokenizer/renderer, registry aliases, and DSML parser.
- `#40760` is now a closed predecessor, useful only as historical context for
  the initial bring-up shape.
- `#41006` added expert-dtype-aware dispatch so FP4 expert checkpoints keep the
  MXFP4 path while Flash-Base / FP8 expert checkpoints use the FP8 MoE path and
  suffix mapping.
- `#40806` fixed DSML marker leakage for DSV3.2/DSV4 streaming parser paths.
- Open PR `#40811` extends persistent top-k from FP32-only assumptions to
  BF16 input support, which matters for the DeepSeek V4 sparse indexer path.

## Open Optimization Items

- [#40860](https://github.com/vllm-project/vllm/pull/40860) `[Feat] DeepSeek V4 Rebased`: mainline support landing.
- [#41006](https://github.com/vllm-project/vllm/pull/41006) `Support DSV4 base`: FP4/FP8 expert-dtype handling and mapper split.
- [#40806](https://github.com/vllm-project/vllm/pull/40806) `[Bugfix] Fix the DSML token leakage in DSV4/3.2`: merged parser leak fix.
- [#40811](https://github.com/vllm-project/vllm/pull/40811) `[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4`: still open follow-up.
- [#40760](https://github.com/vllm-project/vllm/pull/40760) `[New Model] Support DeepseekV4`: closed predecessor superseded by `#40860`.

## Validation Lanes

- Re-check mainline `registry.py` before claiming a new architecture alias.
- Validate tokenizer/renderer parity, DSML tool calls, base generation, and MTP
  speculative decoding after edits to model, tokenizer, renderer, or parser code.
- For base / Flash-Base checkpoints, verify `expert_dtype="fp8"` selects FP8 MoE
  scales and does not route through MXFP4.
- If `#40811` merges, rerun kernel tests in
  `tests/kernels/test_top_k_per_row.py`, especially BF16 decode and long-context
  cases.
- Rerun streaming tool-parser tests to ensure DSML sentinels never leak into
  user-visible content.

## References

- `references/pr-history.md`: diff-reviewed DeepSeek V4 cards.
