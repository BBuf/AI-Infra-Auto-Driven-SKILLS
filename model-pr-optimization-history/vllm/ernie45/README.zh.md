# vLLM Ernie4.5 / Ernie4.5-VL 支持与 PR 历史

本文记录 vLLM 中与 Ernie4.5 / Ernie4.5-VL 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Ernie4.5 spans dense, MoE, and VL paths in vLLM.
- The highest-risk work items are shared-expert behavior, VL rotary/timestamp logic, and long-input stability.

## 主要代码面

- `vllm/vllm/model_executor/models/ernie45.py`
- `vllm/vllm/model_executor/models/ernie45_moe.py`
- `vllm/vllm/model_executor/models/ernie45_vl.py`

## 已合入 PR

- [#20220](https://github.com/vllm-project/vllm/pull/20220) `Add Ernie4.5 and Ernie4.5MoE Model Support`：Landed text and MoE support.
- [#21717](https://github.com/vllm-project/vllm/pull/21717) `Fix Ernie4_5_MoeForCausalLM shared experts`：Fixed shared-expert correctness.
- [#22514](https://github.com/vllm-project/vllm/pull/22514) `Add Ernie4.5 VL Model Support`：Added the multimodal Ernie4.5-VL lane.
- [#24074](https://github.com/vllm-project/vllm/pull/24074) `Fix Ernie4.5-VL hanging on long inputs`：Closed a production long-input stall.
- [#31274](https://github.com/vllm-project/vllm/pull/31274) `Support video metadata for timestamp rendering`：Improved VL video output fidelity.

## 配套 skill

- `skills/model-optimization/vllm/vllm-ernie45-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-ernie45-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `ERNIE 4.5 / ERNIE 4.5 VL`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-06-29 | [#20220](https://github.com/vllm-project/vllm/pull/20220) | merged | [Model] Add Ernie4.5 and Ernie4.5MoE Model Support | model wrapper, MoE/router, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py` |
| 2025-07-28 | [#21717](https://github.com/vllm-project/vllm/pull/21717) | merged | [Bugfix] Fix Ernie4_5_MoeForCausalLM shared experts | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/ernie45_moe.py` |
| 2025-08-08 | [#22514](https://github.com/vllm-project/vllm/pull/22514) | merged | [Model] Add Ernie4.5 VL Model Support | model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py` |
| 2025-09-02 | [#24074](https://github.com/vllm-project/vllm/pull/24074) | merged | [BugFix][Model] Fix Ernie4.5-VL hanging on long inputs | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py` |
| 2025-12-24 | [#31274](https://github.com/vllm-project/vllm/pull/31274) | merged | [Model][Ernie4.5-VL] Support video metadata for timestamp rendering | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks | `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py` |

### 逐 PR 代码 diff 阅读记录

### PR #20220 - [Model] Add Ernie4.5 and Ernie4.5MoE Model Support

- 链接：https://github.com/vllm-project/vllm/pull/20220
- 状态/时间：`merged`，created 2025-06-29, merged 2025-07-02；作者 `CSWYF3634076`。
- 代码 diff 已读范围：`5` 个文件，`+634/-0`；代码面：model wrapper, MoE/router, scheduler/runtime, tests/benchmarks, docs/config；关键词：kv, moe, spec, attention, config, cache, doc, expert, fp8, processor。
- 代码 diff 细节：
  - `vllm/model_executor/models/ernie45_moe.py` added +583/-0 (583 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Ernie4_5_MoeMLP, __init__, forward, Ernie4_5_MoeMoE
  - `vllm/model_executor/models/ernie45.py` added +43/-0 (43 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Ernie4_5_ForCausalLM, __init__
  - `tests/models/registry.py` modified +4/-0 (4 lines); hunk: def check_available_online(; 符号: check_available_online
  - `docs/models/supported_models.md` modified +2/-0 (2 lines); hunk: Specified using `--task generate`.
  - `vllm/model_executor/models/registry.py` modified +2/-0 (2 lines); hunk: "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py`；patch 关键词为 kv, moe, spec, attention, config, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21717 - [Bugfix] Fix Ernie4_5_MoeForCausalLM shared experts

- 链接：https://github.com/vllm-project/vllm/pull/21717
- 状态/时间：`merged`，created 2025-07-28, merged 2025-07-28；作者 `jeejeelee`。
- 代码 diff 已读范围：`1` 个文件，`+6/-5`；代码面：model wrapper, MoE/router, scheduler/runtime；关键词：config, expert, moe, router。
- 代码 diff 细节：
  - `vllm/model_executor/models/ernie45_moe.py` modified +6/-5 (11 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/ernie45_moe.py`；patch 关键词为 config, expert, moe, router。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/ernie45_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22514 - [Model] Add Ernie4.5 VL Model Support

- 链接：https://github.com/vllm-project/vllm/pull/22514
- 状态/时间：`merged`，created 2025-08-08, merged 2025-08-27；作者 `CSWYF3634076`。
- 代码 diff 已读范围：`11` 个文件，`+2463/-0`；代码面：model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：moe, kv, vision, config, attention, cache, cuda, processor, quant, spec。
- 代码 diff 细节：
  - `vllm/model_executor/models/ernie45_vl.py` added +1504/-0 (1504 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: rotate_half, apply_rotary_emb_torch, apply_rotary_pos_emb_vision, all_gather_interleave
  - `vllm/model_executor/models/ernie45_vl_moe.py` added +723/-0 (723 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Ernie4_5_VLMoeMLP, Ernie4_5_VLMoeAttention, __init__, forward
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +123/-0 (123 lines); hunk: def get_input_positions_tensor(; def _glm4v_get_input_positions_tensor(; 符号: get_input_positions_tensor, _glm4v_get_input_positions_tensor, _ernie_get_input_positions_tensor, _vl_get_input_positions_tensor
  - `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` added +72/-0 (72 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Ernie4_5_VLRotaryEmbedding, forward
  - `examples/offline_inference/vision_language.py` modified +32/-0 (32 lines); hunk: def run_deepseek_vl2(questions: list[str], modality: str) -> ModelRequestData:; def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:; 符号: run_deepseek_vl2, run_ernie45_vl, run_florence2, run_tarsier2
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`；patch 关键词为 moe, kv, vision, config, attention, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #24074 - [BugFix][Model] Fix Ernie4.5-VL hanging on long inputs

- 链接：https://github.com/vllm-project/vllm/pull/24074
- 状态/时间：`merged`，created 2025-09-02, merged 2025-09-09；作者 `CSWYF3634076`。
- 代码 diff 已读范围：`2` 个文件，`+18/-7`；代码面：model wrapper, MoE/router, scheduler/runtime；关键词：vision, expert, moe, processor, router。
- 代码 diff 细节：
  - `vllm/model_executor/models/ernie45_vl.py` modified +10/-4 (14 lines); hunk: logger = init_logger(__name__); def get_image_processor(self, **kwargs: object):; 符号: get_image_processor, get_supported_mm_limits, get_mm_max_tokens_per_item, _get_vision_info
  - `vllm/model_executor/models/ernie45_vl_moe.py` modified +8/-3 (11 lines); hunk: def forward(; def forward(; 符号: forward, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`；patch 关键词为 vision, expert, moe, processor, router。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #31274 - [Model][Ernie4.5-VL] Support video metadata for timestamp rendering

- 链接：https://github.com/vllm-project/vllm/pull/31274
- 状态/时间：`merged`，created 2025-12-24, merged 2025-12-25；作者 `Tiiiktak`。
- 代码 diff 已读范围：`2` 个文件，`+82/-5`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks；关键词：attention, config, moe, processor, spec, test。
- 代码 diff 细节：
  - `vllm/model_executor/models/ernie45_vl.py` modified +80/-4 (84 lines); hunk: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.; MMEncoderAttention,; 符号: get_max_video_tokens, Ernie4_5VLMultiModalProcessor, _get_data_parser, _pixel_values_norm
  - `tests/models/multimodal/processing/test_common.py` modified +2/-1 (3 lines); hunk: def create_metadata(frames: np.ndarray):; 符号: create_metadata
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py`；patch 关键词为 attention, config, moe, processor, spec, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
