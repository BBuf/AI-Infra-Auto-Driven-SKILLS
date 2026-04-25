# vLLM DeepSeek V3 / R1 支持与 PR 历史

本文记录 vLLM 中与 DeepSeek V3 / R1 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- `DeepseekV2ForCausalLM` / `DeepseekV3ForCausalLM` remain the shared runtime for V3 and R1.
- The highest-risk regressions cluster around packed module mapping, quantized MLA/MoE weight loading, LoRA, and MTP draft paths.
- R1 validation should split BF16, FP8/ModelOpt, and compressed-tensors or ROCm lanes.

## 主要代码面

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/models/deepseek_eagle.py`
- `vllm/vllm/model_executor/models/deepseek_eagle3.py`
- `vllm/vllm/model_executor/models/deepseek_mtp.py`

## 已合入 PR

- [#22352](https://github.com/vllm-project/vllm/pull/22352) `Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM``：Fixed quantized and packed-weight loading for DeepSeek V2/V3/R1 style checkpoints.
- [#23971](https://github.com/vllm-project/vllm/pull/23971) `Add LoRA support for DeepSeek models (V2, V3, R1-0528)`：Enabled adapter injection on the DeepSeek family rather than only base dense models.
- [#29545](https://github.com/vllm-project/vllm/pull/29545) `Fix DeepSeek R1 MTP weight loading`：Hardened R1 NextN / MTP draft loading after launch failures on draft weights.
- [#36247](https://github.com/vllm-project/vllm/pull/36247) `Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x`：Closed a production ROCm gap for compressed-tensors DeepSeek-R1 deployment.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-deepseek-v3-r1-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-deepseek-v3-r1-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `DeepSeek V3 / R1`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-06 | [#22352](https://github.com/vllm-project/vllm/pull/22352) | merged | [Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM` | model wrapper, scheduler/runtime | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-08-29 | [#23971](https://github.com/vllm-project/vllm/pull/23971) | merged | Add LoRA support for DeepSeek models (V2, V3, R1-0528) | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/deepseek.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/deepseek_v2.py` |
| 2025-11-26 | [#29545](https://github.com/vllm-project/vllm/pull/29545) | merged | [Bugfix] Fix DeepSeek R1 MTP weight loading | model wrapper, scheduler/runtime | `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-03-06 | [#36247](https://github.com/vllm-project/vllm/pull/36247) | merged | [Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x | model wrapper, scheduler/runtime | `vllm/model_executor/models/deepseek_v2.py` |

### 逐 PR 代码 diff 阅读记录

### PR #22352 - [Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM`

- 链接：https://github.com/vllm-project/vllm/pull/22352
- 状态/时间：`merged`，created 2025-08-06, merged 2025-08-07；作者 `fxmarty-amd`。
- 代码 diff 已读范围：`1` 个文件，`+16/-0`；代码面：model wrapper, scheduler/runtime；关键词：config, expert, kv, lora, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/deepseek_v2.py` modified +16/-0 (16 lines); hunk: def forward(; 符号: forward, DeepseekV2ForCausalLM, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/deepseek_v2.py`；patch 关键词为 config, expert, kv, lora, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23971 - Add LoRA support for DeepSeek models (V2, V3, R1-0528)

- 链接：https://github.com/vllm-project/vllm/pull/23971
- 状态/时间：`merged`，created 2025-08-29, merged 2025-08-30；作者 `sadeghja1070`。
- 代码 diff 已读范围：`3` 个文件，`+12/-7`；代码面：model wrapper, scheduler/runtime, docs/config；关键词：kv, lora, config, doc, expert, moe。
- 代码 diff 细节：
  - `vllm/model_executor/models/deepseek.py` modified +6/-2 (8 lines); hunk: from vllm.model_executor.sampling_metadata import SamplingMetadata; def load_weights(self, weights: Iterable[tuple[str,; 符号: load_weights, DeepseekForCausalLM, DeepseekForCausalLM, __init__
  - `docs/models/supported_models.md` modified +3/-3 (6 lines); hunk: th {
  - `vllm/model_executor/models/deepseek_v2.py` modified +3/-2 (5 lines); hunk: from vllm.model_executor.sampling_metadata import SamplingMetadata; def forward(; 符号: forward, DeepseekV2ForCausalLM, DeepseekV2ForCausalLM
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/deepseek.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/deepseek_v2.py`；patch 关键词为 kv, lora, config, doc, expert, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/deepseek.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #29545 - [Bugfix] Fix DeepSeek R1 MTP weight loading

- 链接：https://github.com/vllm-project/vllm/pull/29545
- 状态/时间：`merged`，created 2025-11-26, merged 2025-12-02；作者 `MatthewBonanni`。
- 代码 diff 已读范围：`1` 个文件，`+11/-0`；代码面：model wrapper, scheduler/runtime；关键词：expert。
- 代码 diff 细节：
  - `vllm/model_executor/models/deepseek_mtp.py` modified +11/-0 (11 lines); hunk: def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str; 符号: load_weights, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/deepseek_mtp.py`；patch 关键词为 expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/deepseek_mtp.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #36247 - [Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x

- 链接：https://github.com/vllm-project/vllm/pull/36247
- 状态/时间：`merged`，created 2026-03-06, merged 2026-03-07；作者 `vllmellm`。
- 代码 diff 已读范围：`1` 个文件，`+2/-2`；代码面：model wrapper, scheduler/runtime；关键词：config, kv, lora, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunk: def _min_latency_fused_qkv_a_proj_fake(; def __init__(; 符号: _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/deepseek_v2.py`；patch 关键词为 config, kv, lora, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：4；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
