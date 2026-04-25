# vLLM Qwen3.5 支持与 PR 历史

本文记录 vLLM 中与 Qwen3.5 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- Qwen3.5 builds on the Qwen3-Next era work but has its own model registration and quantization details.
- The hot spots are GDN fusion, FP8/NVFP4 loading, LoRA target naming, and MoE EP precision.

## 主要代码面

- `vllm/vllm/model_executor/models/qwen3_5.py`
- `vllm/vllm/model_executor/models/qwen3_5_mtp.py`

## 已合入 PR

- [#34110](https://github.com/vllm-project/vllm/pull/34110) `Adding Support for Qwen3.5 Models`：Landed the Qwen3.5 runtime family.
- [#34697](https://github.com/vllm-project/vllm/pull/34697) `Redo Qwen3.5/Qwen3-Next GDN projector fusion`：Reworked an earlier fusion that had to be reverted.
- [#35289](https://github.com/vllm-project/vllm/pull/35289) `Fix Qwen3.5 FP8 quantization tuple shard_id weight loading`：Closed a concrete FP8 weight-loading failure.
- [#36658](https://github.com/vllm-project/vllm/pull/36658) `Add Eagle3 support for Qwen3.5`：Enabled the draft-model fast path.
- [#37975](https://github.com/vllm-project/vllm/pull/37975) `Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5`：Reduced duplicated GDN logic across related families.
- [#39181](https://github.com/vllm-project/vllm/pull/39181) `Fix EP precision for Qwen3.5, Qwen3-Next`：Patched a serving-precision bug under expert parallelism.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-qwen35-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen35-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Qwen3.5`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-09 | [#34110](https://github.com/vllm-project/vllm/pull/34110) | merged | [MODEL] Adding Support for Qwen3.5 Models | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `tests/models/registry.py` |
| 2026-02-17 | [#34697](https://github.com/vllm-project/vllm/pull/34697) | merged | [Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion | model wrapper, scheduler/runtime | `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_next.py` |
| 2026-02-25 | [#35289](https://github.com/vllm-project/vllm/pull/35289) | merged | [Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading | scheduler/runtime | `vllm/model_executor/layers/linear.py` |
| 2026-03-10 | [#36658](https://github.com/vllm-project/vllm/pull/36658) | merged | Add: Eagle3 support for Qwen3.5 | model wrapper, scheduler/runtime | `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-24 | [#37975](https://github.com/vllm-project/vllm/pull/37975) | merged | [Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5 | model wrapper, scheduler/runtime | `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py` |
| 2026-04-07 | [#39181](https://github.com/vllm-project/vllm/pull/39181) | merged | [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py` |

### 逐 PR 代码 diff 阅读记录

### PR #34110 - [MODEL] Adding Support for Qwen3.5 Models

- 链接：https://github.com/vllm-project/vllm/pull/34110
- 状态/时间：`merged`，created 2026-02-09, merged 2026-02-09；作者 `JJJYmmm`。
- 代码 diff 已读范围：`11` 个文件，`+1501/-9`；代码面：model wrapper, scheduler/runtime, tests/benchmarks, docs/config；关键词：moe, config, spec, attention, cache, expert, quant, kv, processor, flash。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_5.py` added +993/-0 (993 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Qwen3_5ProcessingInfo, get_hf_config, Qwen3_5MoeProcessingInfo, get_hf_config
  - `vllm/model_executor/models/qwen3_5_mtp.py` added +447/-0 (447 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `tests/models/registry.py` modified +20/-0 (20 lines); hunk: def check_available_online(; 符号: check_available_online
  - `vllm/model_executor/models/qwen3_next.py` modified +6/-6 (12 lines); hunk: class Qwen3NextSparseMoeBlock(nn.Module):; def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; 符号: Qwen3NextSparseMoeBlock, __init__, __init__, Qwen3NextModel
  - `vllm/config/speculative.py` modified +11/-0 (11 lines); hunk: "ernie_mtp",; def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:; 符号: hf_config_override
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `tests/models/registry.py`；patch 关键词为 moe, config, spec, attention, cache, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `tests/models/registry.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #34697 - [Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion

- 链接：https://github.com/vllm-project/vllm/pull/34697
- 状态/时间：`merged`，created 2026-02-17, merged 2026-02-18；作者 `Isotr0py`。
- 代码 diff 已读范围：`3` 个文件，`+102/-192`；代码面：model wrapper, scheduler/runtime；关键词：quant, config, kv, spec, attention, cache, expert, fp8, moe, processor。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_5.py` modified +43/-170 (213 lines); hunk: import torch; ); 符号: get_hf_config, Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering
  - `vllm/model_executor/layers/linear.py` modified +32/-12 (44 lines); hunk: def weight_loader(; def weight_loader(; 符号: weight_loader, weight_loader, weight_loader, _load_fused_module_from_checkpoint
  - `vllm/model_executor/models/qwen3_next.py` modified +27/-10 (37 lines); hunk: from vllm.model_executor.layers.layernorm import RMSNormGated; def __init__(; 符号: __init__, __init__, create_qkvz_proj, fix_query_key_value_ordering
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_next.py`；patch 关键词为 quant, config, kv, spec, attention, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #35289 - [Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading

- 链接：https://github.com/vllm-project/vllm/pull/35289
- 状态/时间：`merged`，created 2026-02-25, merged 2026-02-26；作者 `Alibaba-HZY`。
- 代码 diff 已读范围：`1` 个文件，`+19/-8`；代码面：scheduler/runtime；关键词：quant, spec。
- 代码 diff 细节：
  - `vllm/model_executor/layers/linear.py` modified +19/-8 (27 lines); hunk: def weight_loader(; def weight_loader(; 符号: weight_loader, weight_loader, weight_loader
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/linear.py`；patch 关键词为 quant, spec。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/linear.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #36658 - Add: Eagle3 support for Qwen3.5

- 链接：https://github.com/vllm-project/vllm/pull/36658
- 状态/时间：`merged`，created 2026-03-10, merged 2026-03-11；作者 `rahul-tuli`。
- 代码 diff 已读范围：`2` 个文件，`+25/-2`；代码面：model wrapper, scheduler/runtime；关键词：expert, config, eagle, lora。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_next.py` modified +14/-2 (16 lines); hunk: def get_layer(prefix: str):; def forward(; 符号: get_layer, embed_input_ids, forward, forward
  - `vllm/model_executor/models/qwen3_5.py` modified +11/-0 (11 lines); hunk: IsHybrid,; def get_layer(prefix: str):; 符号: get_layer, load_fused_expert_weights, load_weights, Qwen3_5ForCausalLMBase
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py`；patch 关键词为 expert, config, eagle, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #37975 - [Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5

- 链接：https://github.com/vllm-project/vllm/pull/37975
- 状态/时间：`merged`，created 2026-03-24, merged 2026-03-27；作者 `wxsIcey`。
- 代码 diff 已读范围：`3` 个文件，`+1053/-1126`；代码面：model wrapper, scheduler/runtime；关键词：attention, config, kv, lora, quant, spec, benchmark, cache, cuda, expert。
- 代码 diff 细节：
  - `vllm/model_executor/layers/mamba/gdn_linear_attn.py` added +1046/-0 (1046 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: fi_chunk_gated_delta_rule, ChunkGatedDeltaRule, __init__, forward_cuda
  - `vllm/model_executor/models/qwen3_next.py` modified +3/-975 (978 lines); hunk: from itertools import islice; get_current_vllm_config,; 符号: fi_chunk_gated_delta_rule, ChunkGatedDeltaRule, __init__, forward_cuda
  - `vllm/model_executor/models/qwen3_5.py` modified +4/-151 (155 lines); hunk: from collections.abc import Callable, Iterable; from vllm.model_executor.layers.layernorm import (; 符号: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py`；patch 关键词为 attention, config, kv, lora, quant, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #39181 - [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next

- 链接：https://github.com/vllm-project/vllm/pull/39181
- 状态/时间：`merged`，created 2026-04-07, merged 2026-04-08；作者 `USTCKAY`。
- 代码 diff 已读范围：`2` 个文件，`+4/-0`；代码面：model wrapper, MoE/router, scheduler/runtime；关键词：config, expert, quant, moe。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen2_moe.py` modified +3/-0 (3 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__
  - `vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunk: def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`；patch 关键词为 config, expert, quant, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：6；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
