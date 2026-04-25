# vLLM Qwen3 Core Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Qwen3 Core.

- Status: supported on current mainline

## Key Conclusions

- Qwen3 and Qwen3MoE are first-class mainline runtimes in vLLM.
- The highest-risk regressions show up in quantized MoE weight loading, packed-module mappings, and embedding / reranker special cases.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen3.py`
- `vllm/vllm/model_executor/models/qwen3_moe.py`
- `vllm/vllm/model_executor/models/voyage.py`

## Landed PRs

- [#15289](https://github.com/vllm-project/vllm/pull/15289) `Add Qwen3 and Qwen3MoE`: Initial Qwen3 dense and MoE support landed here.
- [#19260](https://github.com/vllm-project/vllm/pull/19260) `Support Qwen3 Embedding & Reranker`: Extended the family to bidirectional embedding / reranker models.
- [#19598](https://github.com/vllm-project/vllm/pull/19598) `Skip loading extra parameters for modelopt Qwen3 MoE model`: Fixed a concrete ModelOpt launch failure on Qwen3 MoE.
- [#22017](https://github.com/vllm-project/vllm/pull/22017) `KeyError for Qwen3-MoE with GPTQ on ROCm`: Closed a GPTQ loading failure in the Qwen3 MoE path.
- [#22785](https://github.com/vllm-project/vllm/pull/22785) `Fix GGUF loader for Qwen3 MoE`: Made the Qwen3 MoE loader accept GGUF weights again.
- [#23490](https://github.com/vllm-project/vllm/pull/23490) `Fix Qwen3 MoE GPTQ inference`: Patched runtime correctness after GPTQ startup succeeded.
- [#26485](https://github.com/vllm-project/vllm/pull/26485) `Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE`: Enabled the draft-model path on top of the base Qwen3 MoE runtime.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-qwen3-core-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen3-core-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Qwen3 Core` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-03-21 | [#15289](https://github.com/vllm-project/vllm/pull/15289) | merged | [Model] Add Qwen3 and Qwen3MoE | model wrapper, MoE/router, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen2.py` |
| 2025-06-06 | [#19260](https://github.com/vllm-project/vllm/pull/19260) | merged | [New Model]: Support Qwen3 Embedding & Reranker | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3.py`, `tests/models/language/pooling/test_qwen3_reranker.py`, `examples/offline_inference/qwen3_reranker.py` |
| 2025-06-13 | [#19598](https://github.com/vllm-project/vllm/pull/19598) | merged | [Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-07-31 | [#22017](https://github.com/vllm-project/vllm/pull/22017) | merged | [BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-13 | [#22785](https://github.com/vllm-project/vllm/pull/22785) | merged | Fix GGUF loader for Qwen3 MoE. | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/model_loader/gguf_loader.py`, `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-24 | [#23490](https://github.com/vllm-project/vllm/pull/23490) | merged | [Bugfix] Fix Qwen3 MoE GPTQ inference | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-10-09 | [#26485](https://github.com/vllm-project/vllm/pull/26485) | merged | Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen3_moe.py` |

### File-level PR diff reading notes

### PR #15289 - [Model] Add Qwen3 and Qwen3MoE

- Link: https://github.com/vllm-project/vllm/pull/15289
- Status/date: `merged`, created 2025-03-21, merged 2025-04-07; author `YamPengLi`.
- Diff scope read: `6` files, `+893/-5`; areas: model wrapper, MoE/router, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, cache, config, quant, attention, kv, processor, spec, doc, expert.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` added +531/-0 (531 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Qwen3MoeMLP, __init__, forward, Qwen3MoeSparseMoeBlock
  - `vllm/model_executor/models/qwen3.py` added +329/-0 (329 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Qwen3Attention, __init__, forward, Qwen3DecoderLayer
  - `vllm/model_executor/models/qwen2.py` modified +11/-5 (16 lines); hunks: def forward(; def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: forward, Qwen2Model, __init__, __init__
  - `docs/source/models/supported_models.md` modified +10/-0 (10 lines); hunks: See this page (#generative-models) for more information on how to use generativ
  - `tests/models/registry.py` modified +10/-0 (10 lines); hunks: def check_available_online(; symbols: check_available_online
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen2.py`; keywords observed in patches: moe, cache, config, quant, attention, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19260 - [New Model]: Support Qwen3 Embedding & Reranker

- Link: https://github.com/vllm-project/vllm/pull/19260
- Status/date: `merged`, created 2025-06-06, merged 2025-06-11; author `noooop`.
- Diff scope read: `8` files, `+396/-19`; areas: model wrapper, scheduler/runtime, tests/benchmarks, docs/config; keywords: test, attention, config, doc, lora, moe, kv, processor, quant, spec.
- Code diff details:
  - `vllm/model_executor/models/qwen3.py` modified +123/-2 (125 lines); hunks: from vllm.model_executor.layers.linear import (QKVParallelLinear,; def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, Qwen3ForSequenceClassification, __init__, forward
  - `tests/models/language/pooling/test_qwen3_reranker.py` added +87/-0 (87 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: vllm_reranker, hf_reranker, process_inputs, compute_logits
  - `examples/offline_inference/qwen3_reranker.py` added +77/-0 (77 lines); hunks: +# SPDX-License-Identifier: Apache-2.0
  - `tests/models/language/pooling/test_qwen3_reranker_seq_cls.py` added +73/-0 (73 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: vllm_reranker, hf_reranker, process_inputs, compute_logits
  - `docs/models/supported_models.md` modified +25/-17 (42 lines); hunks: See this page (./pooling_models.md) for more information on how to use pooling; If your model is not in the above list, we will try to automatically convert th
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3.py`, `tests/models/language/pooling/test_qwen3_reranker.py`, `examples/offline_inference/qwen3_reranker.py`; keywords observed in patches: test, attention, config, doc, lora, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3.py`, `tests/models/language/pooling/test_qwen3_reranker.py`, `examples/offline_inference/qwen3_reranker.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19598 - [Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model

- Link: https://github.com/vllm-project/vllm/pull/19598
- Status/date: `merged`, created 2025-06-13, merged 2025-06-30; author `noiji`.
- Diff scope read: `1` files, `+15/-9`; areas: model wrapper, MoE/router, scheduler/runtime; keywords: expert, fp8, moe.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +15/-9 (24 lines); hunks: def load_weights(self, weights: Iterable[tuple[str,; def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, load_weights, load_weights, load_weights
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_moe.py`; keywords observed in patches: expert, fp8, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22017 - [BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm

- Link: https://github.com/vllm-project/vllm/pull/22017
- Status/date: `merged`, created 2025-07-31, merged 2025-08-11; author `JartX`.
- Diff scope read: `1` files, `+1/-1`; areas: model wrapper, MoE/router, scheduler/runtime; keywords: config, expert, moe, quant.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__, forward
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_moe.py`; keywords observed in patches: config, expert, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22785 - Fix GGUF loader for Qwen3 MoE.

- Link: https://github.com/vllm-project/vllm/pull/22785
- Status/date: `merged`, created 2025-08-13, merged 2025-08-13; author `Gh0u1L5`.
- Diff scope read: `2` files, `+12/-0`; areas: model wrapper, MoE/router, scheduler/runtime; keywords: config, moe, expert, quant.
- Code diff details:
  - `vllm/model_executor/model_loader/gguf_loader.py` modified +11/-0 (11 lines); hunks: def _get_gguf_weights_map(self, model_config: ModelConfig):; symbols: _get_gguf_weights_map
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/model_loader/gguf_loader.py`, `vllm/model_executor/models/qwen3_moe.py`; keywords observed in patches: config, moe, expert, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/model_loader/gguf_loader.py`, `vllm/model_executor/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23490 - [Bugfix] Fix Qwen3 MoE GPTQ inference

- Link: https://github.com/vllm-project/vllm/pull/23490
- Status/date: `merged`, created 2025-08-24, merged 2025-08-25; author `Isotr0py`.
- Diff scope read: `1` files, `+18/-6`; areas: model wrapper, MoE/router, scheduler/runtime; keywords: config, expert, marlin, moe, processor, quant.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +18/-6 (24 lines); hunks: RowParallelLinear); def __init__(; symbols: __init__, _maybe_ignore_quant_config, forward, load_weights
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_moe.py`; keywords observed in patches: config, expert, marlin, moe, processor, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #26485 - Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE

- Link: https://github.com/vllm-project/vllm/pull/26485
- Status/date: `merged`, created 2025-10-09, merged 2025-10-11; author `rahul-tuli`.
- Diff scope read: `1` files, `+33/-4`; areas: model wrapper, MoE/router, scheduler/runtime; keywords: config, eagle, expert, kv, lora, moe, spec.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +33/-4 (37 lines); hunks: from vllm.model_executor.models.utils import sequence_parallel_chunk; def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, get_input_embeddings, forward, forward
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_moe.py`; keywords observed in patches: config, eagle, expert, kv, lora, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 7; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
