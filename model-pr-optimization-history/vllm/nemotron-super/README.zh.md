# vLLM Nemotron Super / Nano Hybrid 支持与 PR 历史

本文记录 vLLM 中与 Nemotron Super / Nano Hybrid 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Nemotron is a hybrid family: attention, Mamba, MoE, MTP, and VL all intersect.
- Because of that, graph execution, cache dtype, and quantized MoE correctness are the primary risk areas.

## 主要代码面

- `vllm/vllm/model_executor/models/nemotron_h.py`
- `vllm/vllm/model_executor/models/nemotron_h_mtp.py`
- `vllm/vllm/model_executor/models/nano_nemotron_vl.py`
- `vllm/vllm/model_executor/models/nemotron_vl.py`

## 已合入 PR

- [#18863](https://github.com/vllm-project/vllm/pull/18863) `NemotronH support`：Initial NemotronH landing in vLLM.
- [#25863](https://github.com/vllm-project/vllm/pull/25863) `Add MoE support for NemotronH`：Extended the hybrid family to routed experts.
- [#33726](https://github.com/vllm-project/vllm/pull/33726) `Nemotron-H MTP and Mamba Speculative Decoding Support`：Opened the MTP / spec-decode path.
- [#36803](https://github.com/vllm-project/vllm/pull/36803) `E2E Nemotron-3-Super tests`：Added direct Super-family regression coverage.
- [#37803](https://github.com/vllm-project/vllm/pull/37803) `Enable NemotronHPuzzle + NemotronHMTP`：Expanded hybrid and MTP coverage for the family.

## 配套 skill

- `skills/model-optimization/vllm/vllm-nemotron-super-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-nemotron-super-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Nemotron Super / Nano`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-05-28 | [#18863](https://github.com/vllm-project/vllm/pull/18863) | merged | [Model] NemotronH support | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`, `tests/models/registry.py` |
| 2025-09-29 | [#25863](https://github.com/vllm-project/vllm/pull/25863) | merged | [Model] Add MoE support for NemotronH | model wrapper, MoE/router, quantization, scheduler/runtime, docs/config | `vllm/model_executor/models/nemotron_h.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py` |
| 2026-02-03 | [#33726](https://github.com/vllm-project/vllm/pull/33726) | merged | [Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/v1/attention/backends/mamba_attn.py`, `vllm/model_executor/layers/mamba/mamba_mixer2.py` |
| 2026-03-11 | [#36803](https://github.com/vllm-project/vllm/pull/36803) | merged | [Test] E2E Nemotron-3-Super tests | quantization, tests/benchmarks, docs/config | `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` |
| 2026-03-22 | [#37803](https://github.com/vllm-project/vllm/pull/37803) | merged | Enable `NemotronHPuzzle` + `NemotronHMTP` | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/config/speculative.py` |

### 逐 PR 代码 diff 阅读记录

### PR #18863 - [Model] NemotronH support

- 链接：https://github.com/vllm-project/vllm/pull/18863
- 状态/时间：`merged`，created 2025-05-28, merged 2025-06-05；作者 `vegaluisjose`。
- 代码 diff 已读范围：`6` 个文件，`+829/-0`；代码面：model wrapper, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, moe, spec, attention, cache, cuda, doc, fp8, kv, lora。
- 代码 diff 细节：
  - `vllm/model_executor/models/nemotron_h.py` added +565/-0 (565 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: NemotronHMLP, __init__, forward, NemotronHMLPDecoderLayer
  - `vllm/transformers_utils/configs/nemotron_h.py` added +258/-0 (258 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: NemotronHConfig, to, __init__, layers_block_type
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunk: def check_available_online(; 符号: check_available_online
  - `vllm/transformers_utils/configs/__init__.py` modified +2/-0 (2 lines); hunk: from vllm.transformers_utils.configs.moonvit import MoonViTConfig; "MoonViTConfig",
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunk: Specified using `--task generate`.
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`, `tests/models/registry.py`；patch 关键词为 config, moe, spec, attention, cache, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`, `tests/models/registry.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #25863 - [Model] Add MoE support for NemotronH

- 链接：https://github.com/vllm-project/vllm/pull/25863
- 状态/时间：`merged`，created 2025-09-29, merged 2025-10-23；作者 `tomeras91`。
- 代码 diff 已读范围：`7` 个文件，`+413/-39`；代码面：model wrapper, MoE/router, quantization, scheduler/runtime, docs/config；关键词：moe, config, expert, quant, attention, cache, fp8, router, topk, cuda。
- 代码 diff 细节：
  - `vllm/model_executor/models/nemotron_h.py` modified +329/-27 (356 lines); hunk: # limitations under the License.; from vllm.model_executor.models.interfaces import (; 符号: NemotronHMLP, __init__, forward, NemotronHMoE
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +30/-5 (35 lines); hunk: def create_weights(; def create_weights(; 符号: create_weights, create_weights, __init__, __init__
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +26/-5 (31 lines); hunk: def __init__(; def create_weights(; 符号: __init__, create_weights, create_weights, process_weights_after_loading
  - `vllm/transformers_utils/configs/nemotron_h.py` modified +20/-0 (20 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, layers_block_type
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-1 (4 lines); hunk: def fused_experts(; def fused_experts_impl(; 符号: fused_experts, _get_config_quant_dtype, fused_experts_impl
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/nemotron_h.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py`；patch 关键词为 moe, config, expert, quant, attention, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/nemotron_h.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #33726 - [Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support

- 链接：https://github.com/vllm-project/vllm/pull/33726
- 状态/时间：`merged`，created 2026-02-03, merged 2026-02-24；作者 `benchislett`。
- 代码 diff 已读范围：`19` 个文件，`+800/-158`；代码面：model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config；关键词：cache, kv, config, spec, attention, cuda, moe, eagle, expert, processor。
- 代码 diff 细节：
  - `vllm/model_executor/models/nemotron_h_mtp.py` added +503/-0 (503 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: NemotronHMTPAttentionDecoderLayer, __init__, forward, NemotronHMTPMoEDecoderLayer
  - `vllm/v1/attention/backends/mamba_attn.py` modified +193/-85 (278 lines); hunk: # SPDX-FileCopyrightText: Copyright contributors to the vLLM project; class BaseMambaAttentionMetadata:; 符号: BaseMambaAttentionMetadata:, BaseMambaAttentionMetadata:, BaseMambaAttentionMetadataBuilder, __init__
  - `vllm/model_executor/layers/mamba/mamba_mixer2.py` modified +27/-19 (46 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, conv_ssm_forward, conv_ssm_forward
  - `vllm/model_executor/layers/mamba/mamba_mixer.py` modified +2/-18 (20 lines); hunk: def forward_impl(self, hidden_states: torch.Tensor, output: torch.Tensor):; def forward_impl(self, hidden_states: torch.Tensor, output: torch.Tensor):; 符号: forward_impl, forward_impl, PrefillDecodeSplit, split_batch_to_prefill_and_decode
  - `vllm/config/speculative.py` modified +17/-2 (19 lines); hunk: "glm4_moe_lite_mtp",; def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:; 符号: hf_config_override, __post_init__, __post_init__, _verify_args
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/v1/attention/backends/mamba_attn.py`, `vllm/model_executor/layers/mamba/mamba_mixer2.py`；patch 关键词为 cache, kv, config, spec, attention, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/v1/attention/backends/mamba_attn.py`, `vllm/model_executor/layers/mamba/mamba_mixer2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #36803 - [Test] E2E Nemotron-3-Super tests

- 链接：https://github.com/vllm-project/vllm/pull/36803
- 状态/时间：`merged`，created 2026-03-11, merged 2026-03-24；作者 `roikoren755`。
- 代码 diff 已读范围：`6` 个文件，`+37/-0`；代码面：quantization, tests/benchmarks, docs/config；关键词：config, test, expert, fp8, spec, fp4, moe。
- 代码 diff 细节：
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml` added +11/-0 (11 lines); hunk: +model_name: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml` added +11/-0 (11 lines); hunk: +model_name: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` added +11/-0 (11 lines); hunk: +model_name: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
  - `tests/evals/gsm8k/configs/models-h200.txt` modified +2/-0 (2 lines); hunk: DeepSeek-R1-TP.yaml
  - `.buildkite/test_areas/lm_eval.yaml` modified +1/-0 (1 lines); hunk: steps:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml`；patch 关键词为 config, test, expert, fp8, spec, fp4。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #37803 - Enable `NemotronHPuzzle` + `NemotronHMTP`

- 链接：https://github.com/vllm-project/vllm/pull/37803
- 状态/时间：`merged`，created 2026-03-22, merged 2026-03-22；作者 `netanel-haber`。
- 代码 diff 已读范围：`2` 个文件，`+6/-3`；代码面：model wrapper, scheduler/runtime, docs/config；关键词：config, expert, moe, spec。
- 代码 diff 细节：
  - `vllm/model_executor/models/nemotron_h_mtp.py` modified +5/-2 (7 lines); hunk: def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; 符号: load_weights
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunk: def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:; 符号: hf_config_override
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/config/speculative.py`；patch 关键词为 config, expert, moe, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/config/speculative.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
