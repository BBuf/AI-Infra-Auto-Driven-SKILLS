# vLLM Mistral Small 4 Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for Mistral Small 4.

- Status: supported on current mainline

## Key Conclusions

- Mistral Small 4 sits on top of the larger Mistral Large 3 / Ministral runtime work.
- MoE execution and multimodal projector behavior are the main risk areas.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/mistral_large_3.py`
- `vllm/vllm/model_executor/models/mistral_large_3_eagle.py`
- `vllm/vllm/model_executor/models/mistral3.py`

## Landed PRs

- [#29757](https://github.com/vllm-project/vllm/pull/29757) `Add Mistral Large 3 and Ministral 3`: Landed the runtime family that Mistral Small 4 deployments build on in vLLM.
- [#33174](https://github.com/vllm-project/vllm/pull/33174) `Add support for Mistral Large 3 inference with Flashinfer MoE`: Improved the practical MoE serving path for the same family.

## Matching Skill

- `skills/model-optimization/vllm/vllm-mistral-small-4-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-mistral-small-4-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Mistral Small 4 / Ministral 3` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-11-30 | [#29757](https://github.com/vllm-project/vllm/pull/29757) | merged | Add Mistral Large 3 and Ministral 3 | model wrapper, attention/backend, MoE/router, quantization, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/mistral_large_3_eagle.py`, `tests/tokenizers_/test_mistral.py`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8.json` |
| 2026-01-27 | [#33174](https://github.com/vllm-project/vllm/pull/33174) | merged | Add support for Mistral Large 3 inference with Flashinfer MoE | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` |

### File-level PR diff reading notes

### PR #29757 - Add Mistral Large 3 and Ministral 3

- Link: https://github.com/vllm-project/vllm/pull/29757
- Status/date: `merged`, created 2025-11-30, merged 2025-12-02; author `juliendenize`.
- Diff scope read: `16` files, `+724/-30`; areas: model wrapper, attention/backend, MoE/router, quantization, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, spec, kv, attention, eagle, expert, fp8, moe, quant, test.
- Code diff details:
  - `vllm/model_executor/models/mistral_large_3_eagle.py` added +165/-0 (165 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: EagleMistralLarge3Model, __init__, forward, EagleMistralLarge3ForCausalLM
  - `tests/tokenizers_/test_mistral.py` modified +151/-7 (158 lines); hunks: ],; def test_decode(; symbols: test_prepare_apply_chat_template_tools_and_messages, test_decode, test_decode_empty, test_decode_int
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: +{
  - `vllm/transformers_utils/configs/mistral.py` modified +62/-12 (74 lines); hunks: def adapt_config_dict(; def _remap_general_mistral_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_general_mistral_args, _remap_mistral_quantization_args, _remap_mistral_audio_args
  - `vllm/model_executor/models/deepseek_v2.py` modified +59/-7 (66 lines); hunks: def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:; def __init__(; symbols: yarn_get_mscale, _get_llama_4_scaling, DeepseekV2Attention, __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/mistral_large_3_eagle.py`, `tests/tokenizers_/test_mistral.py`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8.json`; keywords observed in patches: config, spec, kv, attention, eagle, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/mistral_large_3_eagle.py`, `tests/tokenizers_/test_mistral.py`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8.json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #33174 - Add support for Mistral Large 3 inference with Flashinfer MoE

- Link: https://github.com/vllm-project/vllm/pull/33174
- Status/date: `merged`, created 2026-01-27, merged 2026-01-31; author `dbari`.
- Diff scope read: `16` files, `+1104/-31`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, moe, triton, fp8, benchmark, expert, quant, topk.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0 (147 lines); hunks: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200.json` added +147/-0 (147 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json`; keywords observed in patches: config, moe, triton, fp8, benchmark, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 2; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
