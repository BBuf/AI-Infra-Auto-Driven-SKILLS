# vLLM GLM-4.5 / 4.5V 支持与 PR 历史

本文记录 vLLM 中与 GLM-4.5 / 4.5V 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- The GLM-4.5 lane is where vLLM reorganized the GLM family around text, MoE, and vision variants.
- Most regressions are in MoE gate behavior, tie-word-embedding policy, and vendor-specific fused MoE tuning.

## 主要代码面

- `vllm/vllm/model_executor/models/glm4.py`
- `vllm/vllm/model_executor/models/glm4_moe.py`
- `vllm/vllm/model_executor/models/glm4v.py`

## 已合入 PR

- [#22171](https://github.com/vllm-project/vllm/pull/22171) `Modify the organization of GLM series`：Reworked the family layout so 4.5-era models reused a cleaner GLM structure.
- [#22460](https://github.com/vllm-project/vllm/pull/22460) `not tie_word_embeddings for glm-4.5 and glm-4.5v`：Aligned the loader with the real 4.5 checkpoint contract instead of forcing tied embeddings.
- [#22832](https://github.com/vllm-project/vllm/pull/22832) `Modify the gate implementation of glm4_moe`：Changed the GLM4.5 MoE gating path used by text and VL variants.
- [#23695](https://github.com/vllm-project/vllm/pull/23695) `Add triton fused moe config for GLM-4.5-Air-FP8 on B200`：Added a production kernel-tuning lane for the 4.5 Air FP8 deployment path.
- [#24589](https://github.com/vllm-project/vllm/pull/24589) `Add documentation for GLM-4.5 series tool-calling and reasoning parser`：Codified the parser choices needed to serve 4.5 reasoning / tool checkpoints correctly.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-glm45-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-glm45-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `GLM-4.5`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-04 | [#22171](https://github.com/vllm-project/vllm/pull/22171) | merged | [Misc] Modify the organization of GLM series | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py` |
| 2025-08-07 | [#22460](https://github.com/vllm-project/vllm/pull/22460) | merged | not tie_word_embeddings for glm-4.5 and glm-4.5v | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-13 | [#22832](https://github.com/vllm-project/vllm/pull/22832) | merged | [Model] Modify the gate implementation of glm4_moe | model wrapper, MoE/router, scheduler/runtime, docs/config | `vllm/model_executor/models/glm4_moe.py`, `docs/models/supported_models.md` |
| 2025-08-26 | [#23695](https://github.com/vllm-project/vllm/pull/23695) | merged | feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200 | MoE/router, quantization, scheduler/runtime, docs/config | `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` |
| 2025-09-10 | [#24589](https://github.com/vllm-project/vllm/pull/24589) | merged | [Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser | docs/config | `docs/features/tool_calling.md`, `docs/features/reasoning_outputs.md` |

### 逐 PR 代码 diff 阅读记录

### PR #22171 - [Misc] Modify the organization of GLM series

- 链接：https://github.com/vllm-project/vllm/pull/22171
- 状态/时间：`merged`，created 2025-08-04, merged 2025-08-04；作者 `jeejeelee`。
- 代码 diff 已读范围：`16` 个文件，`+31/-31`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：test, moe, config, doc, fp8, lora, vision。
- 代码 diff 细节：
  - `docs/models/supported_models.md` modified +5/-5 (10 lines); hunk: th {; th {
  - `tests/models/registry.py` modified +5/-5 (10 lines); hunk: def check_available_online(; def check_available_online(; 符号: check_available_online, check_available_online, check_available_online
  - `tests/models/multimodal/generation/test_common.py` modified +3/-3 (6 lines); hunk: num_logprobs=10,; marks=[large_gpu_mark(min_gb=32)],
  - `vllm/model_executor/models/chatglm.py` modified +3/-3 (6 lines); hunk: # SPDX-License-Identifier: Apache-2.0; def __init__(; 符号: __init__
  - `examples/offline_inference/vision_language.py` modified +2/-2 (4 lines); hunk: def run_gemma3(questions: list[str], modality: str) -> ModelRequestData:; def run_glm4v(questions: list[str], modality: str) -> ModelRequestData:; 符号: run_gemma3, run_glm4v, run_glm4v, run_glm4_1v
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py`；patch 关键词为 test, moe, config, doc, fp8, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22460 - not tie_word_embeddings for glm-4.5 and glm-4.5v

- 链接：https://github.com/vllm-project/vllm/pull/22460
- 状态/时间：`merged`，created 2025-08-07, merged 2025-08-08；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`1` 个文件，`+0/-2`；代码面：model wrapper, MoE/router, scheduler/runtime；关键词：config, moe, processor, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/glm4_moe.py` modified +0/-2 (2 lines); hunk: def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/glm4_moe.py`；patch 关键词为 config, moe, processor, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22832 - [Model] Modify the gate implementation of glm4_moe

- 链接：https://github.com/vllm-project/vllm/pull/22832
- 状态/时间：`merged`，created 2025-08-13, merged 2025-08-14；作者 `jeejeelee`。
- 代码 diff 已读范围：`2` 个文件，`+11/-11`；代码面：model wrapper, MoE/router, scheduler/runtime, docs/config；关键词：moe, config, doc, expert, kv, processor, quant, router。
- 代码 diff 细节：
  - `vllm/model_executor/models/glm4_moe.py` modified +10/-10 (20 lines); hunk: from vllm.model_executor.layers.layernorm import RMSNorm; def __init__(; 符号: __init__, forward
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunk: These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/glm4_moe.py`, `docs/models/supported_models.md`；patch 关键词为 moe, config, doc, expert, kv, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/glm4_moe.py`, `docs/models/supported_models.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23695 - feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200

- 链接：https://github.com/vllm-project/vllm/pull/23695
- 状态/时间：`merged`，created 2025-08-26, merged 2025-08-27；作者 `zixuanzhang226`。
- 代码 diff 已读范围：`1` 个文件，`+146/-0`；代码面：MoE/router, quantization, scheduler/runtime, docs/config；关键词：config, fp8, moe。
- 代码 diff 细节：
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`；patch 关键词为 config, fp8, moe。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #24589 - [Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser

- 链接：https://github.com/vllm-project/vllm/pull/24589
- 状态/时间：`merged`，created 2025-09-10, merged 2025-09-10；作者 `WangErXiao`。
- 代码 diff 已读范围：`2` 个文件，`+10/-0`；代码面：docs/config；关键词：doc。
- 代码 diff 细节：
  - `docs/features/tool_calling.md` modified +9/-0 (9 lines); hunk: Flags:
  - `docs/features/reasoning_outputs.md` modified +1/-0 (1 lines); hunk: vLLM currently supports the following reasoning models:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/features/tool_calling.md`, `docs/features/reasoning_outputs.md`；patch 关键词为 doc。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/features/tool_calling.md`, `docs/features/reasoning_outputs.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
