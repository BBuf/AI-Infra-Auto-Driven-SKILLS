# vLLM Qwen3.5 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: Qwen3.5 dense / MoE / GDN runtime, MTP, FP8 and NVFP4 quantization, LoRA, and Eagle3 in vLLM.


## Landed PRs

### PR #34110 - Adding Support for Qwen3.5 Models

- Link: https://github.com/vllm-project/vllm/pull/34110
- Why it mattered: Landed the Qwen3.5 runtime family.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #34697 - Redo Qwen3.5/Qwen3-Next GDN projector fusion

- Link: https://github.com/vllm-project/vllm/pull/34697
- Why it mattered: Reworked an earlier fusion that had to be reverted.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #35289 - Fix Qwen3.5 FP8 quantization tuple shard_id weight loading

- Link: https://github.com/vllm-project/vllm/pull/35289
- Why it mattered: Closed a concrete FP8 weight-loading failure.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #36658 - Add Eagle3 support for Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/36658
- Why it mattered: Enabled the draft-model fast path.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37975 - Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/37975
- Why it mattered: Reduced duplicated GDN logic across related families.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #39181 - Fix EP precision for Qwen3.5, Qwen3-Next

- Link: https://github.com/vllm-project/vllm/pull/39181
- Why it mattered: Patched a serving-precision bug under expert parallelism.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM Qwen3.5 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-09 | [#34110](https://github.com/vllm-project/vllm/pull/34110) | merged | [MODEL] Adding Support for Qwen3.5 Models | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `tests/models/registry.py` |
| 2026-02-17 | [#34697](https://github.com/vllm-project/vllm/pull/34697) | merged | [Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion | model wrapper, scheduler/runtime | `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_next.py` |
| 2026-02-25 | [#35289](https://github.com/vllm-project/vllm/pull/35289) | merged | [Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading | scheduler/runtime | `vllm/model_executor/layers/linear.py` |
| 2026-03-10 | [#36658](https://github.com/vllm-project/vllm/pull/36658) | merged | Add: Eagle3 support for Qwen3.5 | model wrapper, scheduler/runtime | `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-24 | [#37975](https://github.com/vllm-project/vllm/pull/37975) | merged | [Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5 | model wrapper, scheduler/runtime | `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py` |
| 2026-04-07 | [#39181](https://github.com/vllm-project/vllm/pull/39181) | merged | [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py` |

## Diff Cards

### PR #34110 - [MODEL] Adding Support for Qwen3.5 Models

- Link: https://github.com/vllm-project/vllm/pull/34110
- Status/date: `merged`, created 2026-02-09, merged 2026-02-09; author `JJJYmmm`.
- Diff scope read: `11` files, `+1501/-9`; areas: model wrapper, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, config, spec, attention, cache, expert, quant, kv, processor, flash.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` added +993/-0 (993 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Qwen3_5ProcessingInfo, get_hf_config, Qwen3_5MoeProcessingInfo, get_hf_config
  - `vllm/model_executor/models/qwen3_5_mtp.py` added +447/-0 (447 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `tests/models/registry.py` modified +20/-0 (20 lines); hunks: def check_available_online(; symbols: check_available_online
  - `vllm/model_executor/models/qwen3_next.py` modified +6/-6 (12 lines); hunks: class Qwen3NextSparseMoeBlock(nn.Module):; def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: Qwen3NextSparseMoeBlock, __init__, __init__, Qwen3NextModel
  - `vllm/config/speculative.py` modified +11/-0 (11 lines); hunks: "ernie_mtp",; def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:; symbols: hf_config_override
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `tests/models/registry.py`; keywords observed in patches: moe, config, spec, attention, cache, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `tests/models/registry.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #34697 - [Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion

- Link: https://github.com/vllm-project/vllm/pull/34697
- Status/date: `merged`, created 2026-02-17, merged 2026-02-18; author `Isotr0py`.
- Diff scope read: `3` files, `+102/-192`; areas: model wrapper, scheduler/runtime; keywords: quant, config, kv, spec, attention, cache, expert, fp8, moe, processor.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +43/-170 (213 lines); hunks: import torch; ); symbols: get_hf_config, Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering
  - `vllm/model_executor/layers/linear.py` modified +32/-12 (44 lines); hunks: def weight_loader(; def weight_loader(; symbols: weight_loader, weight_loader, weight_loader, _load_fused_module_from_checkpoint
  - `vllm/model_executor/models/qwen3_next.py` modified +27/-10 (37 lines); hunks: from vllm.model_executor.layers.layernorm import RMSNormGated; def __init__(; symbols: __init__, __init__, create_qkvz_proj, fix_query_key_value_ordering
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_next.py`; keywords observed in patches: quant, config, kv, spec, attention, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #35289 - [Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading

- Link: https://github.com/vllm-project/vllm/pull/35289
- Status/date: `merged`, created 2026-02-25, merged 2026-02-26; author `Alibaba-HZY`.
- Diff scope read: `1` files, `+19/-8`; areas: scheduler/runtime; keywords: quant, spec.
- Code diff details:
  - `vllm/model_executor/layers/linear.py` modified +19/-8 (27 lines); hunks: def weight_loader(; def weight_loader(; symbols: weight_loader, weight_loader, weight_loader
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/linear.py`; keywords observed in patches: quant, spec. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/linear.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #36658 - Add: Eagle3 support for Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/36658
- Status/date: `merged`, created 2026-03-10, merged 2026-03-11; author `rahul-tuli`.
- Diff scope read: `2` files, `+25/-2`; areas: model wrapper, scheduler/runtime; keywords: expert, config, eagle, lora.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +14/-2 (16 lines); hunks: def get_layer(prefix: str):; def forward(; symbols: get_layer, embed_input_ids, forward, forward
  - `vllm/model_executor/models/qwen3_5.py` modified +11/-0 (11 lines); hunks: IsHybrid,; def get_layer(prefix: str):; symbols: get_layer, load_fused_expert_weights, load_weights, Qwen3_5ForCausalLMBase
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py`; keywords observed in patches: expert, config, eagle, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #37975 - [Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/37975
- Status/date: `merged`, created 2026-03-24, merged 2026-03-27; author `wxsIcey`.
- Diff scope read: `3` files, `+1053/-1126`; areas: model wrapper, scheduler/runtime; keywords: attention, config, kv, lora, quant, spec, benchmark, cache, cuda, expert.
- Code diff details:
  - `vllm/model_executor/layers/mamba/gdn_linear_attn.py` added +1046/-0 (1046 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: fi_chunk_gated_delta_rule, ChunkGatedDeltaRule, __init__, forward_cuda
  - `vllm/model_executor/models/qwen3_next.py` modified +3/-975 (978 lines); hunks: from itertools import islice; get_current_vllm_config,; symbols: fi_chunk_gated_delta_rule, ChunkGatedDeltaRule, __init__, forward_cuda
  - `vllm/model_executor/models/qwen3_5.py` modified +4/-151 (155 lines); hunks: from collections.abc import Callable, Iterable; from vllm.model_executor.layers.layernorm import (; symbols: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py`; keywords observed in patches: attention, config, kv, lora, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #39181 - [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next

- Link: https://github.com/vllm-project/vllm/pull/39181
- Status/date: `merged`, created 2026-04-07, merged 2026-04-08; author `USTCKAY`.
- Diff scope read: `2` files, `+4/-0`; areas: model wrapper, MoE/router, scheduler/runtime; keywords: config, expert, quant, moe.
- Code diff details:
  - `vllm/model_executor/models/qwen2_moe.py` modified +3/-0 (3 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__
  - `vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunks: def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`; keywords observed in patches: config, expert, quant, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
