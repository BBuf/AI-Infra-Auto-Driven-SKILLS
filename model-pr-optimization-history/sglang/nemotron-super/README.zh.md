# SGLang Nemotron Super / Nano Hybrid 支持与 PR 历史

本文记录 SGLang 中与 Nemotron Super / Nano Hybrid 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Nemotron is a hybrid family: attention, Mamba, MoE, MTP, and VL all intersect.
- Because of that, graph execution, cache dtype, and quantized MoE correctness are the primary risk areas.

## 主要代码面

- `sglang/python/sglang/srt/models/nemotron_h.py`
- `sglang/python/sglang/srt/models/nemotron_h_mtp.py`
- `sglang/python/sglang/srt/models/nano_nemotron_vl.py`

## 已合入 PR

- [#16172](https://github.com/sgl-project/sglang/pull/16172) `NemotronH PP support`：Opened pipeline parallelism on NemotronH.
- [#16227](https://github.com/sgl-project/sglang/pull/16227) `Add latent MoE support`：Added the hybrid latent-MoE path.
- [#19903](https://github.com/sgl-project/sglang/pull/19903) `Enable Piecewise CUDA Graph for NemotronH Hybrid Models`：Improved hybrid serving efficiency.
- [#20407](https://github.com/sgl-project/sglang/pull/20407) `Support Nemotron 3 Super NVFP4`：Added the key quantized Super checkpoint path.
- [#20575](https://github.com/sgl-project/sglang/pull/20575) `Add Nemotron 3 Super CI tests for BF16 and NVFP4`：Added regression coverage for the production checkpoint variants.

## 配套 skill

- `skills/model-optimization/sglang/sglang-nemotron-super-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-nemotron-super-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Nemotron Super / Nano`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-12-30 | [#16172](https://github.com/sgl-project/sglang/pull/16172) | merged | [NemotronH] PP support | model wrapper, tests/benchmarks | `python/sglang/srt/models/nemotron_h.py`, `test/srt/models/test_nvidia_nemotron_nano_v2.py` |
| 2025-12-31 | [#16227](https://github.com/sgl-project/sglang/pull/16227) | merged | [NemotronH] Add latent MoE support | model wrapper, MoE/router, kernel, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_B200.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_B200.json` |
| 2026-03-04 | [#19903](https://github.com/sgl-project/sglang/pull/19903) | merged | Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models | model wrapper, scheduler/runtime | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-03-12 | [#20407](https://github.com/sgl-project/sglang/pull/20407) | merged | [Model] Support Nemotron 3 Super NVFP4 | quantization, tests/benchmarks, docs/config | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `test/registered/model_loading/test_modelopt_loader.py`, `python/sglang/srt/server_args.py` |
| 2026-03-14 | [#20575](https://github.com/sgl-project/sglang/pull/20575) | merged | [CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4 | model wrapper, quantization, tests/benchmarks | `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` |

### 逐 PR 代码 diff 阅读记录

### PR #16172 - [NemotronH] PP support

- 链接：https://github.com/sgl-project/sglang/pull/16172
- 状态/时间：`merged`，created 2025-12-30, merged 2025-12-31；作者 `roikoren755`。
- 代码 diff 已读范围：`2` 个文件，`+94/-35`；代码面：model wrapper, tests/benchmarks；关键词：cache, attention, config, cuda, fp8, lora, moe, processor, quant, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/nemotron_h.py` modified +88/-35 (123 lines); hunk: from sglang.srt.layers.moe.topk import TopK; add_prefix,; 符号: __init__, get_layer, forward, forward
  - `test/srt/models/test_nvidia_nemotron_nano_v2.py` modified +6/-0 (6 lines); hunk: class TestNvidiaNemotronNanoV2BF16(GSM8KMixin, DefaultServerBase):; 符号: TestNvidiaNemotronNanoV2BF16, TestNvidiaNemotronNanoV2BF16PP, TestNvidiaNemotronNanoV2FP8
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/nemotron_h.py`, `test/srt/models/test_nvidia_nemotron_nano_v2.py`；patch 关键词为 cache, attention, config, cuda, fp8, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/nemotron_h.py`, `test/srt/models/test_nvidia_nemotron_nano_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16227 - [NemotronH] Add latent MoE support

- 链接：https://github.com/sgl-project/sglang/pull/16227
- 状态/时间：`merged`，created 2025-12-31, merged 2026-01-02；作者 `roikoren755`。
- 代码 diff 已读范围：`23` 个文件，`+2957/-2`；代码面：model wrapper, MoE/router, kernel, tests/benchmarks, docs/config；关键词：config, moe, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_B200.json` added +146/-0 (146 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json` added +146/-0 (146 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_B200.json` added +146/-0 (146 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_H100_80GB_HBM3.json` added +146/-0 (146 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=256,N=1344,device_name=NVIDIA_B200.json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_B200.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_B200.json`；patch 关键词为 config, moe, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_B200.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_B200.json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19903 - Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models

- 链接：https://github.com/sgl-project/sglang/pull/19903
- 状态/时间：`merged`，created 2026-03-04, merged 2026-03-12；作者 `vedantjh2`。
- 代码 diff 已读范围：`2` 个文件，`+91/-24`；代码面：model wrapper, scheduler/runtime；关键词：attention, cuda, moe, config, expert, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/nemotron_h.py` modified +70/-18 (88 lines); hunk: import torch; is_cuda,; 符号: _forward_core, __init__, _forward_mamba, forward
  - `python/sglang/srt/model_executor/model_runner.py` modified +21/-6 (27 lines); hunk: def init_piecewise_cuda_graphs(self):; def init_piecewise_cuda_graphs(self):; 符号: init_piecewise_cuda_graphs, init_piecewise_cuda_graphs
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/model_executor/model_runner.py`；patch 关键词为 attention, cuda, moe, config, expert, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/model_executor/model_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20407 - [Model] Support Nemotron 3 Super NVFP4

- 链接：https://github.com/sgl-project/sglang/pull/20407
- 状态/时间：`merged`，created 2026-03-12, merged 2026-03-14；作者 `mmangkad`。
- 代码 diff 已读范围：`6` 个文件，`+277/-11`；代码面：quantization, tests/benchmarks, docs/config；关键词：config, fp4, fp8, quant, moe, awq, expert, kv, spec, attention。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +177/-0 (177 lines); hunk: def __init__(self, quant_config: ModelOptFp8Config):; 符号: __init__, ModelOptMixedPrecisionConfig, __init__, override_quantization_method
  - `test/registered/model_loading/test_modelopt_loader.py` modified +65/-0 (65 lines); hunk: from sglang.srt.configs.load_config import LoadConfig; def test_non_modelopt_quant_method_unchanged(self):; 符号: test_non_modelopt_quant_method_unchanged, TestModelOptMixedPrecisionConfig, test_nemotron_mixed_precision_uses_modelopt_mixed, test_mixed_precision_override_does_not_hijack_w4afp8
  - `python/sglang/srt/server_args.py` modified +17/-9 (26 lines); hunk: "modelopt",; def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments, _handle_model_specific_adjustments, _handle_moe_kernel_config, _handle_moe_kernel_config
  - `python/sglang/srt/configs/model_config.py` modified +12/-0 (12 lines); hunk: def _parse_modelopt_quant_config(self, quant_config_dict: dict) -> Optional[dict; def _get_modelopt_quant_type(self) -> str:; 符号: _parse_modelopt_quant_config, _get_modelopt_quant_type, _validate_quantize_and_serve_config, _verify_quantization
  - `python/sglang/srt/model_loader/loader.py` modified +4/-2 (6 lines); hunk: def get_model_loader(; def get_model_loader(; 符号: get_model_loader, get_model_loader
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `test/registered/model_loading/test_modelopt_loader.py`, `python/sglang/srt/server_args.py`；patch 关键词为 config, fp4, fp8, quant, moe, awq。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `test/registered/model_loading/test_modelopt_loader.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20575 - [CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4

- 链接：https://github.com/sgl-project/sglang/pull/20575
- 状态/时间：`merged`，created 2026-03-14, merged 2026-03-14；作者 `mmangkad`。
- 代码 diff 已读范围：`2` 个文件，`+212/-0`；代码面：model wrapper, quantization, tests/benchmarks；关键词：cache, config, cuda, eagle, spec, test, topk, fp4。
- 代码 diff 细节：
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` added +106/-0 (106 lines); hunk: +import unittest; 符号: _run_gsm8k, TestNvidiaNemotron3SuperNVFP4, setUpClass, tearDownClass
  - `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` added +106/-0 (106 lines); hunk: +import unittest; 符号: _run_gsm8k, TestNvidiaNemotron3SuperBF16, setUpClass, tearDownClass
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`；patch 关键词为 cache, config, cuda, eagle, spec, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
