# SGLang Nemotron Super / Nano Hybrid Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for Nemotron Super / Nano Hybrid.

- Status: 当前 mainline 已支持

## Key Conclusions

- Nemotron is a hybrid family: attention, Mamba, MoE, MTP, and VL all intersect.
- Because of that, graph execution, cache dtype, and quantized MoE correctness are the primary risk areas.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/nemotron_h.py`
- `sglang/python/sglang/srt/models/nemotron_h_mtp.py`
- `sglang/python/sglang/srt/models/nano_nemotron_vl.py`

## Landed PRs

- [#16172](https://github.com/sgl-project/sglang/pull/16172) `NemotronH PP support`: Opened pipeline parallelism on NemotronH.
- [#16227](https://github.com/sgl-project/sglang/pull/16227) `Add latent MoE support`: Added the hybrid latent-MoE path.
- [#19903](https://github.com/sgl-project/sglang/pull/19903) `Enable Piecewise CUDA Graph for NemotronH Hybrid Models`: Improved hybrid serving efficiency.
- [#20407](https://github.com/sgl-project/sglang/pull/20407) `Support Nemotron 3 Super NVFP4`: Added the key quantized Super checkpoint path.
- [#20575](https://github.com/sgl-project/sglang/pull/20575) `Add Nemotron 3 Super CI tests for BF16 and NVFP4`: Added regression coverage for the production checkpoint variants.

## Matching Skill

- `skills/model-optimization/sglang/sglang-nemotron-super-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-nemotron-super-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Nemotron Super / Nano` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-12-30 | [#16172](https://github.com/sgl-project/sglang/pull/16172) | merged | [NemotronH] PP support | model wrapper, tests/benchmarks | `python/sglang/srt/models/nemotron_h.py`, `test/srt/models/test_nvidia_nemotron_nano_v2.py` |
| 2025-12-31 | [#16227](https://github.com/sgl-project/sglang/pull/16227) | merged | [NemotronH] Add latent MoE support | model wrapper, MoE/router, kernel, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_B200.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_B200.json` |
| 2026-03-04 | [#19903](https://github.com/sgl-project/sglang/pull/19903) | merged | Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models | model wrapper, scheduler/runtime | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-03-12 | [#20407](https://github.com/sgl-project/sglang/pull/20407) | merged | [Model] Support Nemotron 3 Super NVFP4 | quantization, tests/benchmarks, docs/config | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `test/registered/model_loading/test_modelopt_loader.py`, `python/sglang/srt/server_args.py` |
| 2026-03-14 | [#20575](https://github.com/sgl-project/sglang/pull/20575) | merged | [CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4 | model wrapper, quantization, tests/benchmarks | `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` |

### File-level PR diff reading notes

### PR #16172 - [NemotronH] PP support

- Link: https://github.com/sgl-project/sglang/pull/16172
- Status/date: `merged`, created 2025-12-30, merged 2025-12-31; author `roikoren755`.
- Diff scope read: `2` files, `+94/-35`; areas: model wrapper, tests/benchmarks; keywords: cache, attention, config, cuda, fp8, lora, moe, processor, quant, test.
- Code diff details:
  - `python/sglang/srt/models/nemotron_h.py` modified +88/-35 (123 lines); hunks: from sglang.srt.layers.moe.topk import TopK; add_prefix,; symbols: __init__, get_layer, forward, forward
  - `test/srt/models/test_nvidia_nemotron_nano_v2.py` modified +6/-0 (6 lines); hunks: class TestNvidiaNemotronNanoV2BF16(GSM8KMixin, DefaultServerBase):; symbols: TestNvidiaNemotronNanoV2BF16, TestNvidiaNemotronNanoV2BF16PP, TestNvidiaNemotronNanoV2FP8
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/nemotron_h.py`, `test/srt/models/test_nvidia_nemotron_nano_v2.py`; keywords observed in patches: cache, attention, config, cuda, fp8, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/nemotron_h.py`, `test/srt/models/test_nvidia_nemotron_nano_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16227 - [NemotronH] Add latent MoE support

- Link: https://github.com/sgl-project/sglang/pull/16227
- Status/date: `merged`, created 2025-12-31, merged 2026-01-02; author `roikoren755`.
- Diff scope read: `23` files, `+2957/-2`; areas: model wrapper, MoE/router, kernel, tests/benchmarks, docs/config; keywords: config, moe, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_B200.json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_B200.json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_H100_80GB_HBM3.json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=256,N=1344,device_name=NVIDIA_B200.json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_B200.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_B200.json`; keywords observed in patches: config, moe, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_B200.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_B200.json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19903 - Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models

- Link: https://github.com/sgl-project/sglang/pull/19903
- Status/date: `merged`, created 2026-03-04, merged 2026-03-12; author `vedantjh2`.
- Diff scope read: `2` files, `+91/-24`; areas: model wrapper, scheduler/runtime; keywords: attention, cuda, moe, config, expert, triton.
- Code diff details:
  - `python/sglang/srt/models/nemotron_h.py` modified +70/-18 (88 lines); hunks: import torch; is_cuda,; symbols: _forward_core, __init__, _forward_mamba, forward
  - `python/sglang/srt/model_executor/model_runner.py` modified +21/-6 (27 lines); hunks: def init_piecewise_cuda_graphs(self):; def init_piecewise_cuda_graphs(self):; symbols: init_piecewise_cuda_graphs, init_piecewise_cuda_graphs
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/model_executor/model_runner.py`; keywords observed in patches: attention, cuda, moe, config, expert, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/model_executor/model_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20407 - [Model] Support Nemotron 3 Super NVFP4

- Link: https://github.com/sgl-project/sglang/pull/20407
- Status/date: `merged`, created 2026-03-12, merged 2026-03-14; author `mmangkad`.
- Diff scope read: `6` files, `+277/-11`; areas: quantization, tests/benchmarks, docs/config; keywords: config, fp4, fp8, quant, moe, awq, expert, kv, spec, attention.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +177/-0 (177 lines); hunks: def __init__(self, quant_config: ModelOptFp8Config):; symbols: __init__, ModelOptMixedPrecisionConfig, __init__, override_quantization_method
  - `test/registered/model_loading/test_modelopt_loader.py` modified +65/-0 (65 lines); hunks: from sglang.srt.configs.load_config import LoadConfig; def test_non_modelopt_quant_method_unchanged(self):; symbols: test_non_modelopt_quant_method_unchanged, TestModelOptMixedPrecisionConfig, test_nemotron_mixed_precision_uses_modelopt_mixed, test_mixed_precision_override_does_not_hijack_w4afp8
  - `python/sglang/srt/server_args.py` modified +17/-9 (26 lines); hunks: "modelopt",; def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, _handle_model_specific_adjustments, _handle_moe_kernel_config, _handle_moe_kernel_config
  - `python/sglang/srt/configs/model_config.py` modified +12/-0 (12 lines); hunks: def _parse_modelopt_quant_config(self, quant_config_dict: dict) -> Optional[dict; def _get_modelopt_quant_type(self) -> str:; symbols: _parse_modelopt_quant_config, _get_modelopt_quant_type, _validate_quantize_and_serve_config, _verify_quantization
  - `python/sglang/srt/model_loader/loader.py` modified +4/-2 (6 lines); hunks: def get_model_loader(; def get_model_loader(; symbols: get_model_loader, get_model_loader
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/modelopt_quant.py`, `test/registered/model_loading/test_modelopt_loader.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: config, fp4, fp8, quant, moe, awq. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/modelopt_quant.py`, `test/registered/model_loading/test_modelopt_loader.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20575 - [CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4

- Link: https://github.com/sgl-project/sglang/pull/20575
- Status/date: `merged`, created 2026-03-14, merged 2026-03-14; author `mmangkad`.
- Diff scope read: `2` files, `+212/-0`; areas: model wrapper, quantization, tests/benchmarks; keywords: cache, config, cuda, eagle, spec, test, topk, fp4.
- Code diff details:
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` added +106/-0 (106 lines); hunks: +import unittest; symbols: _run_gsm8k, TestNvidiaNemotron3SuperNVFP4, setUpClass, tearDownClass
  - `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` added +106/-0 (106 lines); hunks: +import unittest; symbols: _run_gsm8k, TestNvidiaNemotron3SuperBF16, setUpClass, tearDownClass
- Optimization/support interpretation: The concrete diff surface is `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`; keywords observed in patches: cache, config, cuda, eagle, spec, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 5; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
