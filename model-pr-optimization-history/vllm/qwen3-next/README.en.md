# vLLM Qwen3-Next Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Qwen3-Next.

- Status: supported on current mainline

## Key Conclusions

- Qwen3-Next is its own runtime family because of Gated DeltaNet attention and its MTP path.
- The practical risks are PP, MTP varlen handling, quantized shared-expert naming, and GDN-specific CUDA graph bugs.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen3_next.py`
- `vllm/vllm/model_executor/models/qwen3_next_mtp.py`

## Landed PRs

- [#24709](https://github.com/vllm-project/vllm/pull/24709) `Fix Qwen3-Next PP`: Corrected pipeline-parallel execution on Qwen3-Next.
- [#24957](https://github.com/vllm-project/vllm/pull/24957) `Fix the varlen issue in qwen3-next MTP implementation`: Removed a concrete MTP correctness bug on variable-length batches.
- [#24960](https://github.com/vllm-project/vllm/pull/24960) `Add prefixes to shared_expert in qwen3-next`: Fixed ignored-parameter and quantized weight loading for shared experts.
- [#25743](https://github.com/vllm-project/vllm/pull/25743) `Fix cuda graph capture bug in GDN metadata and a stride bug`: Stabilized GDN execution under CUDA graphs.
- [#31722](https://github.com/vllm-project/vllm/pull/31722) `Speed-up of GDN attention decode part`: Improved decode throughput on the GDN attention path.
- [#33657](https://github.com/vllm-project/vllm/pull/33657) `Initial support for GDN attention on Qwen3-next/Qwen3.5 (XPU)`: Extended the family beyond CUDA with XPU GDN coverage.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-qwen3-next-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen3-next-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Qwen3 Next` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-09-12 | [#24709](https://github.com/vllm-project/vllm/pull/24709) | merged | [BugFix] Fix Qwen3-Next PP | model wrapper, scheduler/runtime | `vllm/model_executor/models/qwen3_next.py` |
| 2025-09-16 | [#24957](https://github.com/vllm-project/vllm/pull/24957) | merged | [Bugfix][Qwen3-Next] fixes the varlen issue in qwen3-next's MTP implementation. | model wrapper, attention/backend, scheduler/runtime | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`, `vllm/v1/attention/backends/gdn_attn.py`, `vllm/model_executor/models/qwen3_next.py` |
| 2025-09-16 | [#24960](https://github.com/vllm-project/vllm/pull/24960) | merged | [Bugfix][Qwen3-Next] add prefixes to shared_expert in qwen3-next and mlp in qwen2moe to successfully load ignored params in quantized models | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py` |
| 2025-09-26 | [#25743](https://github.com/vllm-project/vllm/pull/25743) | merged | [Qwen3-Next][GDN] fixes cuda graph capturing bug in GDN metadata and a stride bug in causal_conv_1d. | attention/backend, scheduler/runtime | `vllm/v1/attention/backends/gdn_attn.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` |
| 2026-01-05 | [#31722](https://github.com/vllm-project/vllm/pull/31722) | merged | [PERF] Speed-up of GDN attention decode part (Qwen3-Next) | scheduler/runtime | `vllm/model_executor/layers/fla/ops/fused_recurrent.py` |
| 2026-02-03 | [#33657](https://github.com/vllm-project/vllm/pull/33657) | merged | [XPU] Initial support for GDN attention on Qwen3-next/Qwen3.5 | scheduler/runtime | `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/platforms/xpu.py`, `vllm/model_executor/layers/layernorm.py` |

### File-level PR diff reading notes

### PR #24709 - [BugFix] Fix Qwen3-Next PP

- Link: https://github.com/vllm-project/vllm/pull/24709
- Status/date: `merged`, created 2025-09-12, merged 2025-09-12; author `njhill`.
- Diff scope read: `1` files, `+7/-3`; areas: model wrapper, scheduler/runtime; keywords: config.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +7/-3 (10 lines); hunks: # SPDX-FileCopyrightText: Copyright contributors to the vLLM project; def get_layer(prefix: str):; symbols: get_layer, get_input_embeddings, forward
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_next.py`; keywords observed in patches: config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #24957 - [Bugfix][Qwen3-Next] fixes the varlen issue in qwen3-next's MTP implementation.

- Link: https://github.com/vllm-project/vllm/pull/24957
- Status/date: `merged`, created 2025-09-16, merged 2025-09-17; author `sighingnow`.
- Diff scope read: `3` files, `+139/-34`; areas: model wrapper, attention/backend, scheduler/runtime; keywords: cache, spec, kv, attention, config, cuda, scheduler.
- Code diff details:
  - `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` modified +116/-16 (132 lines); hunks: def _causal_conv1d_update_kernel(; def _causal_conv1d_update_kernel(; symbols: _causal_conv1d_update_kernel, _causal_conv1d_update_kernel, _causal_conv1d_update_kernel, _causal_conv1d_update_kernel
  - `vllm/v1/attention/backends/gdn_attn.py` modified +20/-11 (31 lines); hunks: class GDNAttentionMetadata:; def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],; symbols: GDNAttentionMetadata:, __init__, build, build
  - `vllm/model_executor/models/qwen3_next.py` modified +3/-7 (10 lines); hunks: def _forward(; def _forward(; symbols: _forward, _forward, _forward
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`, `vllm/v1/attention/backends/gdn_attn.py`, `vllm/model_executor/models/qwen3_next.py`; keywords observed in patches: cache, spec, kv, attention, config, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`, `vllm/v1/attention/backends/gdn_attn.py`, `vllm/model_executor/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #24960 - [Bugfix][Qwen3-Next] add prefixes to shared_expert in qwen3-next and mlp in qwen2moe to successfully load ignored params in quantized models

- Link: https://github.com/vllm-project/vllm/pull/24960
- Status/date: `merged`, created 2025-09-16, merged 2025-09-18; author `toncao`.
- Diff scope read: `2` files, `+26/-23`; areas: model wrapper, MoE/router, scheduler/runtime; keywords: config, expert, quant, attention, kv, moe.
- Code diff details:
  - `vllm/model_executor/models/qwen2_moe.py` modified +25/-23 (48 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
  - `vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`; keywords observed in patches: config, expert, quant, attention, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #25743 - [Qwen3-Next][GDN] fixes cuda graph capturing bug in GDN metadata and a stride bug in causal_conv_1d.

- Link: https://github.com/vllm-project/vllm/pull/25743
- Status/date: `merged`, created 2025-09-26, merged 2025-09-26; author `sighingnow`.
- Diff scope read: `3` files, `+50/-45`; areas: attention/backend, scheduler/runtime; keywords: attention, cache, cuda, spec, config, kv, scheduler.
- Code diff details:
  - `vllm/v1/attention/backends/gdn_attn.py` modified +26/-35 (61 lines); hunks: def build( # type: ignore[override]; def build( # type: ignore[override]; symbols: build, build, build_for_cudagraph_capture
  - `vllm/v1/worker/gpu_model_runner.py` modified +16/-7 (23 lines); hunks: def __init__(; def _prepare_inputs(; symbols: __init__, _prepare_inputs, _prepare_inputs
  - `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` modified +8/-3 (11 lines); hunks: def _causal_conv1d_fwd_kernel( # continuous batching; def _causal_conv1d_fwd_kernel( # continuous batching; symbols: _causal_conv1d_fwd_kernel, _causal_conv1d_fwd_kernel, _causal_conv1d_fwd_kernel, causal_conv1d_fn
- Optimization/support interpretation: The concrete diff surface is `vllm/v1/attention/backends/gdn_attn.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`; keywords observed in patches: attention, cache, cuda, spec, config, kv. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/v1/attention/backends/gdn_attn.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #31722 - [PERF] Speed-up of GDN attention decode part (Qwen3-Next)

- Link: https://github.com/vllm-project/vllm/pull/31722
- Status/date: `merged`, created 2026-01-05, merged 2026-01-06; author `vadiklyutiy`.
- Diff scope read: `1` files, `+1/-1`; areas: scheduler/runtime; keywords: triton.
- Code diff details:
  - `vllm/model_executor/layers/fla/ops/fused_recurrent.py` modified +1/-1 (2 lines); hunks: def fused_recurrent_gated_delta_rule_fwd(; symbols: fused_recurrent_gated_delta_rule_fwd
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/fla/ops/fused_recurrent.py`; keywords observed in patches: triton. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/fla/ops/fused_recurrent.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #33657 - [XPU] Initial support for GDN attention on Qwen3-next/Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/33657
- Status/date: `merged`, created 2026-02-03, merged 2026-04-03; author `yma11`.
- Diff scope read: `3` files, `+150/-0`; areas: scheduler/runtime; keywords: attention, cache, cuda, kv, spec, config, doc, lora.
- Code diff details:
  - `vllm/model_executor/layers/mamba/gdn_linear_attn.py` modified +94/-0 (94 lines); hunks: def __init__(; def forward(; symbols: __init__, forward, forward_cuda, forward
  - `vllm/platforms/xpu.py` modified +51/-0 (51 lines); hunks: def check_and_update_config(cls, vllm_config: VllmConfig) -> None:; symbols: check_and_update_config, update_block_size_for_backend, support_hybrid_kv_cache
  - `vllm/model_executor/layers/layernorm.py` modified +5/-0 (5 lines); hunks: def forward_cuda(; symbols: forward_cuda, forward_xpu, LayerNorm
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/platforms/xpu.py`, `vllm/model_executor/layers/layernorm.py`; keywords observed in patches: attention, cache, cuda, kv, spec, config. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/platforms/xpu.py`, `vllm/model_executor/layers/layernorm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 6; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
