# vLLM GLM-5 / 5.1 支持与 PR 历史

本文记录 vLLM 在提交 `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
附近对 GLM-5 / 5.1 的已落地支持。

- 状态: 当前 mainline 仅部分支持

## 核心结论

- GLM-5 现在不是独立的 `glm5.py` 实现，而是通过 DeepSeek-V2/V3 的
  MLA/MoE 运行时做适配。
- 当前最关键的两步是:
  `#34124` 负责架构和配置适配，
  `#34385` 负责修正 MTP 草稿模型的 logits 正确性。

## 主要代码面

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/models/registry.py`
- `vllm/vllm/config/speculative.py`
- `vllm/vllm/transformers_utils/model_arch_config_convertor.py`
- `vllm/vllm/v1/spec_decode/eagle.py`

## 已合入 PR

- [#34124](https://github.com/vllm-project/vllm/pull/34124)
  `GLM adaptation`
  已审 diff: `7` 个文件，`13` 行新增，`3` 行删除。
  它把 `GlmMoeDsaForCausalLM` 接到 DeepSeek-V2 运行时上，并补了
  speculative config 与 `indexer_rope_interleave` 适配。
- [#34385](https://github.com/vllm-project/vllm/pull/34385)
  `Fix MTP accuracy for GLM-5`
  已审 diff: `1` 个文件，`18` 行新增。
  它把目标模型 `lm_head` 显式共享给每个 MTP layer 的
  `shared_head.head`，否则 GLM-5 draft logits 会出现 NaN 或未初始化输出。

## 当前结论

遇到 GLM-5 问题时，优先检查 DeepSeek 复用路径和 speculative decode 基础
设施，而不是先去看旧的 `glm4*` 文件。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `GLM-5 / GLM-5.1`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-09 | [#34124](https://github.com/vllm-project/vllm/pull/34124) | merged | [Model] GLM adaptation | model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py` |
| 2026-02-11 | [#34385](https://github.com/vllm-project/vllm/pull/34385) | merged | [Bugfix] Fix MTP accuracy for GLM-5 | scheduler/runtime | `vllm/v1/spec_decode/eagle.py` |

### 逐 PR 代码 diff 阅读记录

### PR #34124 - [Model] GLM adaptation

- 链接：https://github.com/vllm-project/vllm/pull/34124
- 状态/时间：`merged`，created 2026-02-09, merged 2026-02-09；作者 `jeejeelee`。
- 代码 diff 已读范围：`7` 个文件，`+13/-3`；代码面：model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：moe, config, kv, spec, test, benchmark, cache, flash, mla。
- 代码 diff 细节：
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunk: def __init__(; class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):; 符号: __init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM, get_spec_layer_idx_from_weight_name
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunk: def check_available_online(; 符号: check_available_online
  - `tests/models/test_initialization.py` modified +1/-1 (2 lines); hunk: def _initialize_kv_caches_v1(self, vllm_config):; 符号: _initialize_kv_caches_v1
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunk: def compute_hash(self) -> str:; 符号: compute_hash, hf_config_override
  - `benchmarks/kernels/benchmark_moe.py` modified +1/-0 (1 lines); hunk: def get_model_params(config):; 符号: get_model_params
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`；patch 关键词为 moe, config, kv, spec, test, benchmark。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #34385 - [Bugfix] Fix MTP accuracy for GLM-5

- 链接：https://github.com/vllm-project/vllm/pull/34385
- 状态/时间：`merged`，created 2026-02-11, merged 2026-02-12；作者 `mgoin`。
- 代码 diff 已读范围：`1` 个文件，`+18/-0`；代码面：scheduler/runtime；关键词：eagle, spec。
- 代码 diff 细节：
  - `vllm/v1/spec_decode/eagle.py` modified +18/-0 (18 lines); hunk: def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:; 符号: _maybe_share_lm_head, dummy_run
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/v1/spec_decode/eagle.py`；patch 关键词为 eagle, spec。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/v1/spec_decode/eagle.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：2；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
