# vLLM Qwen3-Next 支持与 PR 历史

本文记录 vLLM 中与 Qwen3-Next 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- Qwen3-Next is its own runtime family because of Gated DeltaNet attention and its MTP path.
- The practical risks are PP, MTP varlen handling, quantized shared-expert naming, and GDN-specific CUDA graph bugs.

## 主要代码面

- `vllm/vllm/model_executor/models/qwen3_next.py`
- `vllm/vllm/model_executor/models/qwen3_next_mtp.py`

## 已合入 PR

- [#24709](https://github.com/vllm-project/vllm/pull/24709) `Fix Qwen3-Next PP`：Corrected pipeline-parallel execution on Qwen3-Next.
- [#24957](https://github.com/vllm-project/vllm/pull/24957) `Fix the varlen issue in qwen3-next MTP implementation`：Removed a concrete MTP correctness bug on variable-length batches.
- [#24960](https://github.com/vllm-project/vllm/pull/24960) `Add prefixes to shared_expert in qwen3-next`：Fixed ignored-parameter and quantized weight loading for shared experts.
- [#25743](https://github.com/vllm-project/vllm/pull/25743) `Fix cuda graph capture bug in GDN metadata and a stride bug`：Stabilized GDN execution under CUDA graphs.
- [#31722](https://github.com/vllm-project/vllm/pull/31722) `Speed-up of GDN attention decode part`：Improved decode throughput on the GDN attention path.
- [#33657](https://github.com/vllm-project/vllm/pull/33657) `Initial support for GDN attention on Qwen3-next/Qwen3.5 (XPU)`：Extended the family beyond CUDA with XPU GDN coverage.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-qwen3-next-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen3-next-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Qwen3 Next`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-09-12 | [#24709](https://github.com/vllm-project/vllm/pull/24709) | merged | [BugFix] Fix Qwen3-Next PP | model wrapper, scheduler/runtime | `vllm/model_executor/models/qwen3_next.py` |
| 2025-09-16 | [#24957](https://github.com/vllm-project/vllm/pull/24957) | merged | [Bugfix][Qwen3-Next] fixes the varlen issue in qwen3-next's MTP implementation. | model wrapper, attention/backend, scheduler/runtime | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`, `vllm/v1/attention/backends/gdn_attn.py`, `vllm/model_executor/models/qwen3_next.py` |
| 2025-09-16 | [#24960](https://github.com/vllm-project/vllm/pull/24960) | merged | [Bugfix][Qwen3-Next] add prefixes to shared_expert in qwen3-next and mlp in qwen2moe to successfully load ignored params in quantized models | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py` |
| 2025-09-26 | [#25743](https://github.com/vllm-project/vllm/pull/25743) | merged | [Qwen3-Next][GDN] fixes cuda graph capturing bug in GDN metadata and a stride bug in causal_conv_1d. | attention/backend, scheduler/runtime | `vllm/v1/attention/backends/gdn_attn.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` |
| 2026-01-05 | [#31722](https://github.com/vllm-project/vllm/pull/31722) | merged | [PERF] Speed-up of GDN attention decode part (Qwen3-Next) | scheduler/runtime | `vllm/model_executor/layers/fla/ops/fused_recurrent.py` |
| 2026-02-03 | [#33657](https://github.com/vllm-project/vllm/pull/33657) | merged | [XPU] Initial support for GDN attention on Qwen3-next/Qwen3.5 | scheduler/runtime | `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/platforms/xpu.py`, `vllm/model_executor/layers/layernorm.py` |

### 逐 PR 代码 diff 阅读记录

### PR #24709 - [BugFix] Fix Qwen3-Next PP

- 链接：https://github.com/vllm-project/vllm/pull/24709
- 状态/时间：`merged`，created 2025-09-12, merged 2025-09-12；作者 `njhill`。
- 代码 diff 已读范围：`1` 个文件，`+7/-3`；代码面：model wrapper, scheduler/runtime；关键词：config。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_next.py` modified +7/-3 (10 lines); hunk: # SPDX-FileCopyrightText: Copyright contributors to the vLLM project; def get_layer(prefix: str):; 符号: get_layer, get_input_embeddings, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_next.py`；patch 关键词为 config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #24957 - [Bugfix][Qwen3-Next] fixes the varlen issue in qwen3-next's MTP implementation.

- 链接：https://github.com/vllm-project/vllm/pull/24957
- 状态/时间：`merged`，created 2025-09-16, merged 2025-09-17；作者 `sighingnow`。
- 代码 diff 已读范围：`3` 个文件，`+139/-34`；代码面：model wrapper, attention/backend, scheduler/runtime；关键词：cache, spec, kv, attention, config, cuda, scheduler。
- 代码 diff 细节：
  - `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` modified +116/-16 (132 lines); hunk: def _causal_conv1d_update_kernel(; def _causal_conv1d_update_kernel(; 符号: _causal_conv1d_update_kernel, _causal_conv1d_update_kernel, _causal_conv1d_update_kernel, _causal_conv1d_update_kernel
  - `vllm/v1/attention/backends/gdn_attn.py` modified +20/-11 (31 lines); hunk: class GDNAttentionMetadata:; def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],; 符号: GDNAttentionMetadata:, __init__, build, build
  - `vllm/model_executor/models/qwen3_next.py` modified +3/-7 (10 lines); hunk: def _forward(; def _forward(; 符号: _forward, _forward, _forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`, `vllm/v1/attention/backends/gdn_attn.py`, `vllm/model_executor/models/qwen3_next.py`；patch 关键词为 cache, spec, kv, attention, config, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`, `vllm/v1/attention/backends/gdn_attn.py`, `vllm/model_executor/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #24960 - [Bugfix][Qwen3-Next] add prefixes to shared_expert in qwen3-next and mlp in qwen2moe to successfully load ignored params in quantized models

- 链接：https://github.com/vllm-project/vllm/pull/24960
- 状态/时间：`merged`，created 2025-09-16, merged 2025-09-18；作者 `toncao`。
- 代码 diff 已读范围：`2` 个文件，`+26/-23`；代码面：model wrapper, MoE/router, scheduler/runtime；关键词：config, expert, quant, attention, kv, moe。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen2_moe.py` modified +25/-23 (48 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, __init__
  - `vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`；patch 关键词为 config, expert, quant, attention, kv, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #25743 - [Qwen3-Next][GDN] fixes cuda graph capturing bug in GDN metadata and a stride bug in causal_conv_1d.

- 链接：https://github.com/vllm-project/vllm/pull/25743
- 状态/时间：`merged`，created 2025-09-26, merged 2025-09-26；作者 `sighingnow`。
- 代码 diff 已读范围：`3` 个文件，`+50/-45`；代码面：attention/backend, scheduler/runtime；关键词：attention, cache, cuda, spec, config, kv, scheduler。
- 代码 diff 细节：
  - `vllm/v1/attention/backends/gdn_attn.py` modified +26/-35 (61 lines); hunk: def build( # type: ignore[override]; def build( # type: ignore[override]; 符号: build, build, build_for_cudagraph_capture
  - `vllm/v1/worker/gpu_model_runner.py` modified +16/-7 (23 lines); hunk: def __init__(; def _prepare_inputs(; 符号: __init__, _prepare_inputs, _prepare_inputs
  - `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` modified +8/-3 (11 lines); hunk: def _causal_conv1d_fwd_kernel( # continuous batching; def _causal_conv1d_fwd_kernel( # continuous batching; 符号: _causal_conv1d_fwd_kernel, _causal_conv1d_fwd_kernel, _causal_conv1d_fwd_kernel, causal_conv1d_fn
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/v1/attention/backends/gdn_attn.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`；patch 关键词为 attention, cache, cuda, spec, config, kv。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/v1/attention/backends/gdn_attn.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #31722 - [PERF] Speed-up of GDN attention decode part (Qwen3-Next)

- 链接：https://github.com/vllm-project/vllm/pull/31722
- 状态/时间：`merged`，created 2026-01-05, merged 2026-01-06；作者 `vadiklyutiy`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：scheduler/runtime；关键词：triton。
- 代码 diff 细节：
  - `vllm/model_executor/layers/fla/ops/fused_recurrent.py` modified +1/-1 (2 lines); hunk: def fused_recurrent_gated_delta_rule_fwd(; 符号: fused_recurrent_gated_delta_rule_fwd
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/fla/ops/fused_recurrent.py`；patch 关键词为 triton。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/fla/ops/fused_recurrent.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #33657 - [XPU] Initial support for GDN attention on Qwen3-next/Qwen3.5

- 链接：https://github.com/vllm-project/vllm/pull/33657
- 状态/时间：`merged`，created 2026-02-03, merged 2026-04-03；作者 `yma11`。
- 代码 diff 已读范围：`3` 个文件，`+150/-0`；代码面：scheduler/runtime；关键词：attention, cache, cuda, kv, spec, config, doc, lora。
- 代码 diff 细节：
  - `vllm/model_executor/layers/mamba/gdn_linear_attn.py` modified +94/-0 (94 lines); hunk: def __init__(; def forward(; 符号: __init__, forward, forward_cuda, forward
  - `vllm/platforms/xpu.py` modified +51/-0 (51 lines); hunk: def check_and_update_config(cls, vllm_config: VllmConfig) -> None:; 符号: check_and_update_config, update_block_size_for_backend, support_hybrid_kv_cache
  - `vllm/model_executor/layers/layernorm.py` modified +5/-0 (5 lines); hunk: def forward_cuda(; 符号: forward_cuda, forward_xpu, LayerNorm
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/platforms/xpu.py`, `vllm/model_executor/layers/layernorm.py`；patch 关键词为 attention, cache, cuda, kv, spec, config。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/platforms/xpu.py`, `vllm/model_executor/layers/layernorm.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：6；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
