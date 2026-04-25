# vLLM GLM-4.6 / 4.7 支持与 PR 历史

本文记录 vLLM 中与 GLM-4.6 / 4.7 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- The 4.6/4.7 generation mainly extends the 4.5 base with new tuning tables, parser behavior, and Lite variants.
- AWQ / Marlin compatibility and content-normalization in tool parsing are the recurring pitfalls.

## 主要代码面

- `vllm/vllm/model_executor/models/glm4.py`
- `vllm/vllm/model_executor/models/glm4_moe.py`
- `vllm/vllm/model_executor/models/glm4v.py`

## 已合入 PR

- [#26818](https://github.com/vllm-project/vllm/pull/26818) `Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on B200`：Added fused-MoE tuning configs for the new Blackwell deployment lane.
- [#30210](https://github.com/vllm-project/vllm/pull/30210) `Fix glm46 awq marlin moe compatibility`：Closed an incompatibility between GLM-4.6 AWQ checkpoints and Marlin MoE assumptions.
- [#30876](https://github.com/vllm-project/vllm/pull/30876) `GLM-4.7 Tool Parser and Doc Update`：Brought parser behavior and docs up to date for 4.7 / 4.7-Flash.
- [#31386](https://github.com/vllm-project/vllm/pull/31386) `GLM Model support for GLM-Lite`：Extended the same runtime family to the Lite checkpoint line.
- [#37386](https://github.com/vllm-project/vllm/pull/37386) `Improve tool call parsing and content normalization for glm47`：Fixed concrete parsing errors that surfaced in newer GLM-4.7 outputs.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-glm46-glm47-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-glm46-glm47-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `GLM-4.6 / GLM-4.7`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-10-14 | [#26818](https://github.com/vllm-project/vllm/pull/26818) | merged | [Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200 | MoE/router, quantization, scheduler/runtime, docs/config | `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` |
| 2025-12-07 | [#30210](https://github.com/vllm-project/vllm/pull/30210) | merged | [Bugfix]: Fix glm46 awq marlin moe wna16 compatibility | MoE/router, quantization, scheduler/runtime | `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py` |
| 2025-12-17 | [#30876](https://github.com/vllm-project/vllm/pull/30876) | merged | GLM-4.7 Tool Parser and Doc Update | model wrapper, MoE/router, scheduler/runtime, docs/config | `vllm/tool_parsers/glm47_moe_tool_parser.py`, `docs/features/tool_calling.md`, `vllm/tool_parsers/__init__.py` |
| 2025-12-26 | [#31386](https://github.com/vllm-project/vllm/pull/31386) | merged | [GLM-4.7] GLM Model support for GLM-Lite | model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/config/speculative.py` |
| 2026-03-18 | [#37386](https://github.com/vllm-project/vllm/pull/37386) | merged | fix(glm47): improve tool call parsing and content normalization | MoE/router, tests/benchmarks | `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py` |

### 逐 PR 代码 diff 阅读记录

### PR #26818 - [Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200

- 链接：https://github.com/vllm-project/vllm/pull/26818
- 状态/时间：`merged`，created 2025-10-14, merged 2025-10-14；作者 `zklapow`。
- 代码 diff 已读范围：`3` 个文件，`+441/-0`；代码面：MoE/router, quantization, scheduler/runtime, docs/config；关键词：config, moe, triton, fp8。
- 代码 diff 细节：
  - `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunk: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunk: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json`；patch 关键词为 config, moe, triton, fp8。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #30210 - [Bugfix]: Fix glm46 awq marlin moe wna16 compatibility

- 链接：https://github.com/vllm-project/vllm/pull/30210
- 状态/时间：`merged`，created 2025-12-07, merged 2025-12-09；作者 `baonudesifeizhai`。
- 代码 diff 已读范围：`2` 个文件，`+50/-4`；代码面：MoE/router, quantization, scheduler/runtime；关键词：config, moe, awq, cuda, marlin, quant。
- 代码 diff 细节：
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +45/-0 (45 lines); hunk: def get_moe_configs(; def get_moe_wna16_block_config(; 符号: get_moe_configs, _ensure_block_size_k_divisible, get_moe_wna16_block_config, get_moe_wna16_block_config
  - `vllm/model_executor/layers/quantization/moe_wna16.py` modified +5/-4 (9 lines); hunk: def __init__(; def from_config(cls, config: dict[str, Any]) -> "MoeWNA16Config":; 符号: __init__, from_config, get_quant_method, moe_wna16_weight_loader
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py`；patch 关键词为 config, moe, awq, cuda, marlin, quant。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #30876 - GLM-4.7 Tool Parser and Doc Update

- 链接：https://github.com/vllm-project/vllm/pull/30876
- 状态/时间：`merged`，created 2025-12-17, merged 2025-12-20；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`5` 个文件，`+38/-3`；代码面：model wrapper, MoE/router, scheduler/runtime, docs/config；关键词：moe, doc, spec。
- 代码 diff 细节：
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` added +23/-0 (23 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Glm47MoeModelToolParser, __init__
  - `docs/features/tool_calling.md` modified +8/-1 (9 lines); hunk: Supported models:
  - `vllm/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunk: "glm4_moe_tool_parser",
  - `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunk: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunk: th {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/tool_parsers/glm47_moe_tool_parser.py`, `docs/features/tool_calling.md`, `vllm/tool_parsers/__init__.py`；patch 关键词为 moe, doc, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/tool_parsers/glm47_moe_tool_parser.py`, `docs/features/tool_calling.md`, `vllm/tool_parsers/__init__.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #31386 - [GLM-4.7] GLM Model support for GLM-Lite

- 链接：https://github.com/vllm-project/vllm/pull/31386
- 状态/时间：`merged`，created 2025-12-26, merged 2026-01-19；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`9` 个文件，`+1135/-1`；代码面：model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：moe, config, spec, expert, flash, kv, topk, benchmark, processor, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/glm4_moe_lite.py` added +642/-0 (642 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Glm4MoeLiteMLP, Glm4MoeLite, Glm4LiteMixtureOfExperts, Glm4MoeLiteAttention
  - `vllm/model_executor/models/glm4_moe_lite_mtp.py` added +464/-0 (464 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: SharedHead, __init__, forward, Glm4MoeLiteMultiTokenPredictorLayer
  - `vllm/config/speculative.py` modified +12/-0 (12 lines); hunk: "deepseek_mtp",; def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:; 符号: hf_config_override
  - `tests/models/registry.py` modified +10/-0 (10 lines); hunk: def check_available_online(; def check_available_online(; 符号: check_available_online, check_available_online
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-1 (3 lines); hunk: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/config/speculative.py`；patch 关键词为 moe, config, spec, expert, flash, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/config/speculative.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #37386 - fix(glm47): improve tool call parsing and content normalization

- 链接：https://github.com/vllm-project/vllm/pull/37386
- 状态/时间：`merged`，created 2026-03-18, merged 2026-03-18；作者 `karanb192`。
- 代码 diff 已读范围：`4` 个文件，`+193/-6`；代码面：MoE/router, tests/benchmarks；关键词：moe, test, spec。
- 代码 diff 细节：
  - `tests/tool_parsers/test_glm47_moe_tool_parser.py` added +168/-0 (168 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: glm47_tokenizer, glm47_tool_parser, mock_request, TestGlm47ExtractToolCalls:
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +16/-2 (18 lines); hunk: # SPDX-License-Identifier: Apache-2.0; class Glm47MoeModelToolParser(Glm4MoeModelToolParser):; 符号: Glm47MoeModelToolParser, __init__
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +6/-1 (7 lines); hunk: def extract_tool_calls(; 符号: extract_tool_calls
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +3/-3 (6 lines); hunk: def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, mock_request):; def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, mock_request):; 符号: test_extract_tool_calls_no_tools, test_extract_tool_calls_no_tools, test_extract_tool_calls_no_tools
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`；patch 关键词为 moe, test, spec。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
