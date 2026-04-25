# SGLang Mistral Small 4 Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for Mistral Small 4.

- Status: 当前 mainline 已支持

## Key Conclusions

- Mistral Small 4 sits on top of the larger Mistral Large 3 / Ministral runtime work.
- Startup format mismatches, multimodal projector behavior, and Eagle / MoE integration are the main risk areas.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/mistral_large_3.py`
- `sglang/python/sglang/srt/models/mistral_large_3_eagle.py`
- `sglang/python/sglang/srt/models/mistral.py`
- `sglang/python/sglang/srt/models/ministral3.py`

## Landed PRs

- [#14213](https://github.com/sgl-project/sglang/pull/14213) `Add Mistral Large 3 support`: Historical base runtime reused by later Small 4 work.
- [#14466](https://github.com/sgl-project/sglang/pull/14466) `Add Mistral Large 3 Eagle Support`: Enabled speculative decode on the underlying family.
- [#15049](https://github.com/sgl-project/sglang/pull/15049) `Mistral Large 3 NVFP4 TRTLLM MoE support`: Added the first serious quantized MoE path.
- [#20708](https://github.com/sgl-project/sglang/pull/20708) `Add Mistral Small 4 support`: Brought Mistral Small 4 / Pixtral-style runtime into mainline.
- [#21620](https://github.com/sgl-project/sglang/pull/21620) `Mistral Small 4 fails to start due to config/weight format mismatch`: Closed a startup regression after launch.

## Matching Skill

- `skills/model-optimization/sglang/sglang-mistral-small-4-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-mistral-small-4-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Mistral Small 4 / Ministral 3` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-12-01 | [#14213](https://github.com/sgl-project/sglang/pull/14213) | merged | Add Mistral Large 3 support. | model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/utils/mistral_utils.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` |
| 2025-12-05 | [#14466](https://github.com/sgl-project/sglang/pull/14466) | merged | Add Mistral Large 3 Eagle Support | model wrapper, attention/backend, quantization, docs/config | `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-13 | [#15049](https://github.com/sgl-project/sglang/pull/15049) | merged | Mistral Large 3 NVFP4 TRTLLM MoE support | MoE/router, quantization, kernel | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2026-03-16 | [#20708](https://github.com/sgl-project/sglang/pull/20708) | merged | Add Mistral Small 4 (Pixtral) support | model wrapper, attention/backend, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/multimodal/processors/pixtral.py`, `python/sglang/srt/utils/hf_transformers_utils.py`, `benchmark/mmmu/bench_sglang.py` |
| 2026-03-29 | [#21620](https://github.com/sgl-project/sglang/pull/21620) | merged | fix: Mistral Small 4 fails to start due to config/weight format mismatch | model wrapper, tests/benchmarks | `python/sglang/srt/server_args.py`, `test/registered/models/test_ministral4_models.py` |

### File-level PR diff reading notes

### PR #14213 - Add Mistral Large 3 support.

- Link: https://github.com/sgl-project/sglang/pull/14213
- Status/date: `merged`, created 2025-12-01, merged 2025-12-04; author `dcampora`.
- Diff scope read: `16` files, `+1400/-120`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, tests/benchmarks, docs/config; keywords: quant, fp8, config, kv, attention, expert, moe, vision, cache, fp4.
- Code diff details:
  - `python/sglang/srt/models/pixtral.py` modified +565/-3 (568 lines); hunks: Using mistral-community/pixtral-12b as reference.; from sglang.srt.layers.layernorm import RMSNorm; symbols: VisionEncoderArgs:, PixtralForConditionalGeneration, get_placeholder_str, __init__
  - `python/sglang/srt/utils/mistral_utils.py` added +295/-0 (295 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: adapt_config_dict, _remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` modified +127/-63 (190 lines); hunks: # Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors; from sglang.srt.layers.quantization.fp; symbols: CompressedTensorsW8A8Fp8, __init__, CompressedTensorsW8A8Fp8, __init__
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +81/-1 (82 lines); hunks: from compressed_tensors import CompressionFormat; def get_moe_method(; symbols: get_moe_method, __init__, create_weights, create_weights
  - `python/sglang/srt/models/mistral_large_3.py` added +81/-0 (81 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MistralLarge3ForCausalLM, load_weights, _iterable_remap_mistral_to_ds
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/utils/mistral_utils.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`; keywords observed in patches: quant, fp8, config, kv, attention, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/utils/mistral_utils.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14466 - Add Mistral Large 3 Eagle Support

- Link: https://github.com/sgl-project/sglang/pull/14466
- Status/date: `merged`, created 2025-12-05, merged 2025-12-05; author `elvischenv`.
- Diff scope read: `9` files, `+313/-62`; areas: model wrapper, attention/backend, quantization, docs/config; keywords: config, quant, attention, eagle, expert, fp8, kv, mla, moe, spec.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8.py` modified +161/-36 (197 lines); hunks: def process_weights_after_loading(self, layer: Module) -> None:; def apply_with_router_logits(; symbols: process_weights_after_loading, process_weights_hip_int4, apply_with_router_logits, apply_with_router_logits
  - `python/sglang/srt/models/mistral_large_3_eagle.py` added +105/-0 (105 lines); hunks: +from typing import Optional; symbols: MistralLarge3Model, __init__, forward, MistralLarge3ForCausalLMEagle
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-6 (20 lines); hunks: def __init__(; def forward(; symbols: __init__, forward, DeepseekV2ForCausalLM, __init__
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` modified +6/-10 (16 lines); hunks: def create_weights(; def create_weights(; symbols: create_weights, create_weights, process_weights_after_loading, apply_weights
  - `python/sglang/srt/configs/model_config.py` modified +11/-1 (12 lines); hunks: def _derive_model_shapes(self):; def _verify_quantization(self) -> None:; symbols: _derive_model_shapes, _verify_quantization
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: config, quant, attention, eagle, expert, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15049 - Mistral Large 3 NVFP4 TRTLLM MoE support

- Link: https://github.com/sgl-project/sglang/pull/15049
- Status/date: `merged`, created 2025-12-13, merged 2025-12-18; author `elvischenv`.
- Diff scope read: `7` files, `+340/-151`; areas: MoE/router, quantization, kernel; keywords: moe, fp8, quant, flash, config, expert, fp4, topk, cache, router.
- Code diff details:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +193/-21 (214 lines); hunks: from compressed_tensors import CompressionFormat; from sglang.srt.layers.quantization.utils import (; symbols: __init__, create_weights, create_weights, process_weights_after_loading
  - `python/sglang/srt/layers/quantization/utils.py` modified +140/-0 (140 lines); hunks: def swizzle_blockscale(scale: torch.Tensor):; symbols: swizzle_blockscale, reorder_w1w3_to_w3w1, prepare_static_weights_for_trtllm_fp4_moe
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +2/-125 (127 lines); hunks: convert_to_channelwise,; def create_weights(; symbols: create_weights, prepare_static_weights_for_kernel, process_weights_after_loading, _slice_scale
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-1 (3 lines); hunks: def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):; symbols: get_moe_impl_class
  - `python/sglang/srt/server_args.py` modified +2/-1 (3 lines); hunks: def _handle_moe_kernel_config(self):; symbols: _handle_moe_kernel_config
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`; keywords observed in patches: moe, fp8, quant, flash, config, expert. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20708 - Add Mistral Small 4 (Pixtral) support

- Link: https://github.com/sgl-project/sglang/pull/20708
- Status/date: `merged`, created 2026-03-16, merged 2026-03-18; author `JustinTong0323`.
- Diff scope read: `18` files, `+360/-124`; areas: model wrapper, attention/backend, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, processor, spec, vision, attention, benchmark, cache, eagle, expert, flash.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/pixtral.py` modified +71/-44 (115 lines); hunks: -import asyncio; class PixtralProcessor(BaseMultimodalProcessor):; symbols: PixtralProcessor, get_patch_grid_size, __init__, defined
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +55/-6 (61 lines); hunks: import os; def _load_deepseek_v32_model(; symbols: _load_deepseek_v32_model, _load_mistral_large_3_for_causal_LM, _load_mistral_large_3_for_causal_LM, get_config
  - `benchmark/mmmu/bench_sglang.py` modified +49/-10 (59 lines); hunks: import argparse; def _get_prefix_suffix(prompt: str) -> Tuple[str, str]:; symbols: _get_prefix_suffix, process_sample, process_sample_with_semaphore, eval_mmmu
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +32/-13 (45 lines); hunks: def _process_messages(; def _apply_jinja_template(; symbols: _process_messages, and, _apply_jinja_template, _apply_jinja_template
  - `python/sglang/srt/utils/mistral_utils.py` modified +27/-3 (30 lines); hunks: def adapt_config_dict(; def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_mistral_yarn_args
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/multimodal/processors/pixtral.py`, `python/sglang/srt/utils/hf_transformers_utils.py`, `benchmark/mmmu/bench_sglang.py`; keywords observed in patches: config, processor, spec, vision, attention, benchmark. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/multimodal/processors/pixtral.py`, `python/sglang/srt/utils/hf_transformers_utils.py`, `benchmark/mmmu/bench_sglang.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21620 - fix: Mistral Small 4 fails to start due to config/weight format mismatch

- Link: https://github.com/sgl-project/sglang/pull/21620
- Status/date: `merged`, created 2026-03-29, merged 2026-03-30; author `LiYomi`.
- Diff scope read: `2` files, `+59/-7`; areas: model wrapper, tests/benchmarks; keywords: attention, config, cuda, test.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +27/-7 (34 lines); hunks: def _handle_load_format(self):; symbols: _handle_load_format, _is_mistral_native_format, _check_format
  - `test/registered/models/test_ministral4_models.py` added +32/-0 (32 lines); hunks: +import unittest; symbols: TestMistralSmall4TextOnly, TestMistralSmall4MMMU
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `test/registered/models/test_ministral4_models.py`; keywords observed in patches: attention, config, cuda, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `test/registered/models/test_ministral4_models.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 5; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
