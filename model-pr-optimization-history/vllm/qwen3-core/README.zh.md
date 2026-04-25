# vLLM Qwen3 Core 支持与 PR 历史

本文记录 vLLM 中与 Qwen3 Core 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- Qwen3 and Qwen3MoE are first-class mainline runtimes in vLLM.
- The highest-risk regressions show up in quantized MoE weight loading, packed-module mappings, and embedding / reranker special cases.

## 主要代码面

- `vllm/vllm/model_executor/models/qwen3.py`
- `vllm/vllm/model_executor/models/qwen3_moe.py`
- `vllm/vllm/model_executor/models/voyage.py`

## 已合入 PR

- [#15289](https://github.com/vllm-project/vllm/pull/15289) `Add Qwen3 and Qwen3MoE`：Initial Qwen3 dense and MoE support landed here.
- [#19260](https://github.com/vllm-project/vllm/pull/19260) `Support Qwen3 Embedding & Reranker`：Extended the family to bidirectional embedding / reranker models.
- [#19598](https://github.com/vllm-project/vllm/pull/19598) `Skip loading extra parameters for modelopt Qwen3 MoE model`：Fixed a concrete ModelOpt launch failure on Qwen3 MoE.
- [#22017](https://github.com/vllm-project/vllm/pull/22017) `KeyError for Qwen3-MoE with GPTQ on ROCm`：Closed a GPTQ loading failure in the Qwen3 MoE path.
- [#22785](https://github.com/vllm-project/vllm/pull/22785) `Fix GGUF loader for Qwen3 MoE`：Made the Qwen3 MoE loader accept GGUF weights again.
- [#23490](https://github.com/vllm-project/vllm/pull/23490) `Fix Qwen3 MoE GPTQ inference`：Patched runtime correctness after GPTQ startup succeeded.
- [#26485](https://github.com/vllm-project/vllm/pull/26485) `Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE`：Enabled the draft-model path on top of the base Qwen3 MoE runtime.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-qwen3-core-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen3-core-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Qwen3 Core`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-03-21 | [#15289](https://github.com/vllm-project/vllm/pull/15289) | merged | [Model] Add Qwen3 and Qwen3MoE | model wrapper, MoE/router, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen2.py` |
| 2025-06-06 | [#19260](https://github.com/vllm-project/vllm/pull/19260) | merged | [New Model]: Support Qwen3 Embedding & Reranker | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3.py`, `tests/models/language/pooling/test_qwen3_reranker.py`, `examples/offline_inference/qwen3_reranker.py` |
| 2025-06-13 | [#19598](https://github.com/vllm-project/vllm/pull/19598) | merged | [Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-07-31 | [#22017](https://github.com/vllm-project/vllm/pull/22017) | merged | [BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-13 | [#22785](https://github.com/vllm-project/vllm/pull/22785) | merged | Fix GGUF loader for Qwen3 MoE. | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/model_loader/gguf_loader.py`, `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-24 | [#23490](https://github.com/vllm-project/vllm/pull/23490) | merged | [Bugfix] Fix Qwen3 MoE GPTQ inference | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-10-09 | [#26485](https://github.com/vllm-project/vllm/pull/26485) | merged | Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen3_moe.py` |

### 逐 PR 代码 diff 阅读记录

### PR #15289 - [Model] Add Qwen3 and Qwen3MoE

- 链接：https://github.com/vllm-project/vllm/pull/15289
- 状态/时间：`merged`，created 2025-03-21, merged 2025-04-07；作者 `YamPengLi`。
- 代码 diff 已读范围：`6` 个文件，`+893/-5`；代码面：model wrapper, MoE/router, scheduler/runtime, tests/benchmarks, docs/config；关键词：moe, cache, config, quant, attention, kv, processor, spec, doc, expert。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_moe.py` added +531/-0 (531 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Qwen3MoeMLP, __init__, forward, Qwen3MoeSparseMoeBlock
  - `vllm/model_executor/models/qwen3.py` added +329/-0 (329 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Qwen3Attention, __init__, forward, Qwen3DecoderLayer
  - `vllm/model_executor/models/qwen2.py` modified +11/-5 (16 lines); hunk: def forward(; def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; 符号: forward, Qwen2Model, __init__, __init__
  - `docs/source/models/supported_models.md` modified +10/-0 (10 lines); hunk: See this page (#generative-models) for more information on how to use generativ
  - `tests/models/registry.py` modified +10/-0 (10 lines); hunk: def check_available_online(; 符号: check_available_online
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen2.py`；patch 关键词为 moe, cache, config, quant, attention, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19260 - [New Model]: Support Qwen3 Embedding & Reranker

- 链接：https://github.com/vllm-project/vllm/pull/19260
- 状态/时间：`merged`，created 2025-06-06, merged 2025-06-11；作者 `noooop`。
- 代码 diff 已读范围：`8` 个文件，`+396/-19`；代码面：model wrapper, scheduler/runtime, tests/benchmarks, docs/config；关键词：test, attention, config, doc, lora, moe, kv, processor, quant, spec。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3.py` modified +123/-2 (125 lines); hunk: from vllm.model_executor.layers.linear import (QKVParallelLinear,; def load_weights(self, weights: Iterable[tuple[str,; 符号: load_weights, Qwen3ForSequenceClassification, __init__, forward
  - `tests/models/language/pooling/test_qwen3_reranker.py` added +87/-0 (87 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: vllm_reranker, hf_reranker, process_inputs, compute_logits
  - `examples/offline_inference/qwen3_reranker.py` added +77/-0 (77 lines); hunk: +# SPDX-License-Identifier: Apache-2.0
  - `tests/models/language/pooling/test_qwen3_reranker_seq_cls.py` added +73/-0 (73 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: vllm_reranker, hf_reranker, process_inputs, compute_logits
  - `docs/models/supported_models.md` modified +25/-17 (42 lines); hunk: See this page (./pooling_models.md) for more information on how to use pooling; If your model is not in the above list, we will try to automatically convert th
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3.py`, `tests/models/language/pooling/test_qwen3_reranker.py`, `examples/offline_inference/qwen3_reranker.py`；patch 关键词为 test, attention, config, doc, lora, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3.py`, `tests/models/language/pooling/test_qwen3_reranker.py`, `examples/offline_inference/qwen3_reranker.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19598 - [Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model

- 链接：https://github.com/vllm-project/vllm/pull/19598
- 状态/时间：`merged`，created 2025-06-13, merged 2025-06-30；作者 `noiji`。
- 代码 diff 已读范围：`1` 个文件，`+15/-9`；代码面：model wrapper, MoE/router, scheduler/runtime；关键词：expert, fp8, moe。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_moe.py` modified +15/-9 (24 lines); hunk: def load_weights(self, weights: Iterable[tuple[str,; def load_weights(self, weights: Iterable[tuple[str,; 符号: load_weights, load_weights, load_weights, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_moe.py`；patch 关键词为 expert, fp8, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22017 - [BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm

- 链接：https://github.com/vllm-project/vllm/pull/22017
- 状态/时间：`merged`，created 2025-07-31, merged 2025-08-11；作者 `JartX`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：model wrapper, MoE/router, scheduler/runtime；关键词：config, expert, moe, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_moe.py`；patch 关键词为 config, expert, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22785 - Fix GGUF loader for Qwen3 MoE.

- 链接：https://github.com/vllm-project/vllm/pull/22785
- 状态/时间：`merged`，created 2025-08-13, merged 2025-08-13；作者 `Gh0u1L5`。
- 代码 diff 已读范围：`2` 个文件，`+12/-0`；代码面：model wrapper, MoE/router, scheduler/runtime；关键词：config, moe, expert, quant。
- 代码 diff 细节：
  - `vllm/model_executor/model_loader/gguf_loader.py` modified +11/-0 (11 lines); hunk: def _get_gguf_weights_map(self, model_config: ModelConfig):; 符号: _get_gguf_weights_map
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-0 (1 lines); hunk: def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/model_loader/gguf_loader.py`, `vllm/model_executor/models/qwen3_moe.py`；patch 关键词为 config, moe, expert, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/model_loader/gguf_loader.py`, `vllm/model_executor/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23490 - [Bugfix] Fix Qwen3 MoE GPTQ inference

- 链接：https://github.com/vllm-project/vllm/pull/23490
- 状态/时间：`merged`，created 2025-08-24, merged 2025-08-25；作者 `Isotr0py`。
- 代码 diff 已读范围：`1` 个文件，`+18/-6`；代码面：model wrapper, MoE/router, scheduler/runtime；关键词：config, expert, marlin, moe, processor, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_moe.py` modified +18/-6 (24 lines); hunk: RowParallelLinear); def __init__(; 符号: __init__, _maybe_ignore_quant_config, forward, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_moe.py`；patch 关键词为 config, expert, marlin, moe, processor, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #26485 - Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE

- 链接：https://github.com/vllm-project/vllm/pull/26485
- 状态/时间：`merged`，created 2025-10-09, merged 2025-10-11；作者 `rahul-tuli`。
- 代码 diff 已读范围：`1` 个文件，`+33/-4`；代码面：model wrapper, MoE/router, scheduler/runtime；关键词：config, eagle, expert, kv, lora, moe, spec。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_moe.py` modified +33/-4 (37 lines); hunk: from vllm.model_executor.models.utils import sequence_parallel_chunk; def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; 符号: __init__, get_input_embeddings, forward, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_moe.py`；patch 关键词为 config, eagle, expert, kv, lora, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：7；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
