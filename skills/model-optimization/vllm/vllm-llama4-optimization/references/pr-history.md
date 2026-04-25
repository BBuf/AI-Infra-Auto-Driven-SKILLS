# vLLM Llama 4 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: Llama4 text and multimodal runtime, FP8/FP4 quantization, router behavior, long-context attention, and Eagle support.

## Landed PRs

### PR #16104 - Support Llama4 in vLLM

- Link: https://github.com/vllm-project/vllm/pull/16104
- Why it mattered: Initial Llama4 landing.
- Runtime path: vllm/vllm/model_executor/models/llama4.py, vllm/vllm/model_executor/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #20419 - Enable ModelOpt Llama4 fp8 checkpoint deployment

- Link: https://github.com/vllm-project/vllm/pull/20419
- Why it mattered: Added ModelOpt FP8 coverage.
- Runtime path: vllm/vllm/model_executor/models/llama4.py, vllm/vllm/model_executor/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #20591 - Llama4 EAGLE Support

- Link: https://github.com/vllm-project/vllm/pull/20591
- Why it mattered: Opened speculative decoding for Llama4.
- Runtime path: vllm/vllm/model_executor/models/llama4.py, vllm/vllm/model_executor/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22511 - Fix Llama4 FlashInfer FP4 MoE issues

- Link: https://github.com/vllm-project/vllm/pull/22511
- Why it mattered: Stabilized the FP4 MoE path.
- Runtime path: vllm/vllm/model_executor/models/llama4.py, vllm/vllm/model_executor/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25889 - Fix misplaced dtype cast in Llama4VisionRotaryEmbedding

- Link: https://github.com/vllm-project/vllm/pull/25889
- Why it mattered: Patched a multimodal rotary bug.
- Runtime path: vllm/vllm/model_executor/models/llama4.py, vllm/vllm/model_executor/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM Llama 4 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-04-05 | [#16104](https://github.com/vllm-project/vllm/pull/16104) | merged | [Model] Support Llama4 in vLLM | model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/v1/attention/backends/flash_attn.py` |
| 2025-07-03 | [#20419](https://github.com/vllm-project/vllm/pull/20419) | merged | Enable ModelOpt Llama4 fp8 checkpoint deployment | model wrapper, MoE/router, quantization, scheduler/runtime | `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py` |
| 2025-07-07 | [#20591](https://github.com/vllm-project/vllm/pull/20591) | merged | [Meta] Llama4 EAGLE Support | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/llama4_eagle.py`, `tests/v1/e2e/test_spec_decode.py`, `tests/models/registry.py` |
| 2025-08-08 | [#22511](https://github.com/vllm-project/vllm/pull/22511) | merged | Fix Llama4 FlashInfer FP4 MoE issues | attention/backend, MoE/router, quantization, scheduler/runtime | `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` |
| 2025-09-29 | [#25889](https://github.com/vllm-project/vllm/pull/25889) | merged | [Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding` | multimodal/processor, scheduler/runtime | `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` |

## Diff Cards

### PR #16104 - [Model] Support Llama4 in vLLM

- Link: https://github.com/vllm-project/vllm/pull/16104
- Status/date: `merged`, created 2025-04-05, merged 2025-04-06; author `houseroad`.
- Diff scope read: `35` files, `+2369/-142`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: cache, config, attention, kv, quant, moe, spec, vision, expert, processor.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` added +886/-0 (886 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward
  - `vllm/model_executor/models/llama4.py` added +530/-0 (530 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Llama4MoE, custom_routing_function, __init__, forward
  - `vllm/v1/attention/backends/flash_attn.py` modified +236/-14 (250 lines); hunks: class FlashAttentionMetadata:; def reorder_batch(self, input_batch: "InputBatch",; symbols: FlashAttentionMetadata:, LocalAttentionMetadata:, make_local_attention_virtual_batches, FlashAttentionMetadataBuilder:
  - `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` added +200/-0 (200 lines); hunks: +{
  - `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: test_processor_override
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/v1/attention/backends/flash_attn.py`; keywords observed in patches: cache, config, attention, kv, quant, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/v1/attention/backends/flash_attn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20419 - Enable ModelOpt Llama4 fp8 checkpoint deployment

- Link: https://github.com/vllm-project/vllm/pull/20419
- Status/date: `merged`, created 2025-07-03, merged 2025-07-12; author `Edwardf0t1`.
- Diff scope read: `5` files, `+501/-35`; areas: model wrapper, MoE/router, quantization, scheduler/runtime; keywords: expert, fp8, kv, moe, config, quant, spec, attention, cache, fp4.
- Code diff details:
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +261/-5 (266 lines); hunks: class ModelOptFp8Config(QuantizationConfig):; def get_config_filenames(cls) -> list[str]:; symbols: ModelOptFp8Config, __init__, get_config_filenames, from_config
  - `vllm/model_executor/models/mllama4.py` modified +144/-20 (164 lines); hunks: class Llama4ForConditionalGeneration(nn.Module, SupportsMultiModal,; def _consolidate_qkv_weights(; symbols: Llama4ForConditionalGeneration, _consolidate_qkv_weights, load_weights, _rename_weight_for_modelopt_checkpoint
  - `vllm/model_executor/models/llama4.py` modified +55/-4 (59 lines); hunks: RowParallelLinear); def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, load_weights
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +31/-6 (37 lines); hunks: def create_weights(self, layer: torch.nn.Module, num_experts: int,; def weight_loader(self,; symbols: create_weights, uses_weight_scale_2_pattern, maybe_make_prepare_finalize, weight_loader
  - `vllm/model_executor/model_loader/weight_utils.py` modified +10/-0 (10 lines); hunks: def maybe_remap_kv_scale_name(name: str, params_dict: dict) -> Optional[str]:; symbols: maybe_remap_kv_scale_name
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`; keywords observed in patches: expert, fp8, kv, moe, config, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20591 - [Meta] Llama4 EAGLE Support

- Link: https://github.com/vllm-project/vllm/pull/20591
- Status/date: `merged`, created 2025-07-07, merged 2025-07-16; author `morgendave`.
- Diff scope read: `6` files, `+258/-18`; areas: model wrapper, scheduler/runtime, tests/benchmarks, docs/config; keywords: eagle, config, spec, test, cache, cuda, kv, moe, processor, quant.
- Code diff details:
  - `vllm/model_executor/models/llama4_eagle.py` added +214/-0 (214 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: LlamaModel, __init__, forward, load_weights
  - `tests/v1/e2e/test_spec_decode.py` modified +31/-17 (48 lines); hunks: from typing import Any; def model_name():; symbols: model_name, eagle_model_name, eagle3_model_name, test_ngram_correctness
  - `tests/models/registry.py` modified +6/-1 (7 lines); hunks: def check_available_online(; def find_hf_info(self, model_id: str) -> _HfExamplesInfo:; symbols: check_available_online, find_hf_info
  - `tests/models/test_initialization.py` modified +5/-0 (5 lines); hunks: def test_can_initialize(model_arch: str, monkeypatch: pytest.MonkeyPatch):; symbols: test_can_initialize, hf_overrides
  - `examples/offline_inference/spec_decode.py` modified +1/-0 (1 lines); hunks: def main():; symbols: main
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/llama4_eagle.py`, `tests/v1/e2e/test_spec_decode.py`, `tests/models/registry.py`; keywords observed in patches: eagle, config, spec, test, cache, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/llama4_eagle.py`, `tests/v1/e2e/test_spec_decode.py`, `tests/models/registry.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22511 - Fix Llama4 FlashInfer FP4 MoE issues

- Link: https://github.com/vllm-project/vllm/pull/22511
- Status/date: `merged`, created 2025-08-08, merged 2025-08-12; author `nvpohanh`.
- Diff scope read: `3` files, `+9/-5`; areas: attention/backend, MoE/router, quantization, scheduler/runtime; keywords: expert, flash, moe, quant, router, topk.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` modified +6/-1 (7 lines); hunks: def prepare(; symbols: prepare
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +3/-2 (5 lines); hunks: def apply(; symbols: apply
  - `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` modified +0/-2 (2 lines); hunks: def apply(; symbols: apply
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`; keywords observed in patches: expert, flash, moe, quant, router, topk. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #25889 - [Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding`

- Link: https://github.com/vllm-project/vllm/pull/25889
- Status/date: `merged`, created 2025-09-29, merged 2025-09-30; author `cjackal`.
- Diff scope read: `1` files, `+3/-1`; areas: multimodal/processor, scheduler/runtime; keywords: cache, vision.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +3/-1 (4 lines); hunks: def forward_native( # type: ignore[override]; symbols: forward_native
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; keywords observed in patches: cache, vision. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
