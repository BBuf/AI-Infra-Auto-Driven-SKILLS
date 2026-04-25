# vLLM MiMo-V2-Flash 支持与 PR 历史

本文记录 vLLM 中与 MiMo-V2-Flash 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- MiMo-V2-Flash is a throughput-oriented MoE serving family in vLLM.
- MTP correctness and the split between older MiMo checkpoints and V2-Flash are the key maintenance points.

## 主要代码面

- `vllm/vllm/model_executor/models/mimo_v2_flash.py`
- `vllm/vllm/model_executor/models/mimo.py`
- `vllm/vllm/model_executor/models/mimo_mtp.py`

## 已合入 PR

- [#17433](https://github.com/vllm-project/vllm/pull/17433) `Support MiMo-7B inference with MTP`：Historical base for the MiMo family.
- [#25136](https://github.com/vllm-project/vllm/pull/25136) `Fix MTP inference path for MiMo-7B model`：Closed a concrete draft-path bug.
- [#30836](https://github.com/vllm-project/vllm/pull/30836) `Add MiMo-V2-Flash support`：Landed the dedicated V2-Flash runtime.

## 配套 skill

- `skills/model-optimization/vllm/vllm-mimo-v2-flash-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-mimo-v2-flash-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `MiMo-V2-Flash`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-04-30 | [#17433](https://github.com/vllm-project/vllm/pull/17433) | merged | [Model] Support MiMo-7B inference with MTP | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`, `vllm/config.py` |
| 2025-09-18 | [#25136](https://github.com/vllm-project/vllm/pull/25136) | merged | [spec decode] Fix MTP inference path for MiMo-7B model | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/mimo_mtp.py`, `examples/offline_inference/spec_decode.py`, `vllm/config/speculative.py` |
| 2025-12-17 | [#30836](https://github.com/vllm-project/vllm/pull/30836) | merged | [Model] Add MiMo-V2-Flash support | model wrapper, quantization, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py` |

### 逐 PR 代码 diff 阅读记录

### PR #17433 - [Model] Support MiMo-7B inference with MTP

- 链接：https://github.com/vllm-project/vllm/pull/17433
- 状态/时间：`merged`，created 2025-04-30, merged 2025-05-12；作者 `bwshen-mi`。
- 代码 diff 已读范围：`7` 个文件，`+507/-4`；代码面：model wrapper, scheduler/runtime, tests/benchmarks, docs/config；关键词：spec, config, eagle, cache, kv, processor, quant, attention, doc, expert。
- 代码 diff 细节：
  - `vllm/model_executor/models/mimo_mtp.py` added +283/-0 (283 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MiMoMultiTokenPredictorLayer, __init__, forward, MiMoMultiTokenPredictor
  - `vllm/model_executor/models/mimo.py` added +190/-0 (190 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MiMoModel, forward, load_weights, MiMoForCausalLM
  - `vllm/config.py` modified +17/-3 (20 lines); hunk: def get_num_attention_heads(self,; def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:; 符号: get_num_attention_heads, get_layers_start_end_indices, hf_config_override, __post_init__
  - `vllm/worker/worker.py` modified +5/-1 (6 lines); hunk: def __init__(; 符号: __init__
  - `docs/source/models/supported_models.md` modified +5/-0 (5 lines); hunk: See this page (#generative-models) for more information on how to use generativ
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`, `vllm/config.py`；patch 关键词为 spec, config, eagle, cache, kv, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`, `vllm/config.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #25136 - [spec decode] Fix MTP inference path for MiMo-7B model

- 链接：https://github.com/vllm-project/vllm/pull/25136
- 状态/时间：`merged`，created 2025-09-18, merged 2025-09-18；作者 `zixi-qi`。
- 代码 diff 已读范围：`3` 个文件，`+20/-6`；代码面：model wrapper, scheduler/runtime, docs/config；关键词：config, spec, eagle。
- 代码 diff 细节：
  - `vllm/model_executor/models/mimo_mtp.py` modified +14/-4 (18 lines); hunk: def load_weights(self, weights: Iterable[tuple[str,; 符号: load_weights, map_model_name_to_mtp_param_name, _rewrite_spec_layer_name
  - `examples/offline_inference/spec_decode.py` modified +5/-1 (6 lines); hunk: def parse_args():; def main():; 符号: parse_args, main
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunk: SpeculativeMethod = Literal["ngram", "eagle", "eagle3", "medusa",
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/mimo_mtp.py`, `examples/offline_inference/spec_decode.py`, `vllm/config/speculative.py`；patch 关键词为 config, spec, eagle。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/mimo_mtp.py`, `examples/offline_inference/spec_decode.py`, `vllm/config/speculative.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #30836 - [Model] Add MiMo-V2-Flash support

- 链接：https://github.com/vllm-project/vllm/pull/30836
- 状态/时间：`merged`，created 2025-12-17, merged 2025-12-19；作者 `Abatom`。
- 代码 diff 已读范围：`8` 个文件，`+789/-13`；代码面：model wrapper, quantization, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, flash, fp8, quant, kv, attention, cache, doc, eagle, expert。
- 代码 diff 细节：
  - `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0 (720 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MiMoV2MLP, __init__, forward, MiMoV2MoE
  - `vllm/model_executor/layers/linear.py` modified +49/-13 (62 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, _maybe_allow_fp8_block_shape_mismatch
  - `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +8/-0 (8 lines); hunk: def validate_fp8_block_shape(; 符号: validate_fp8_block_shape
  - `vllm/config/model.py` modified +5/-0 (5 lines); hunk: def try_match_architecture_defaults(; 符号: try_match_architecture_defaults, str_dtype_to_torch_dtype
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunk: def check_available_online(; 符号: check_available_online
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py`；patch 关键词为 config, flash, fp8, quant, kv, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：3；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
