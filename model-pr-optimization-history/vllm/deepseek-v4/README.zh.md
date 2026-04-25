# vLLM DeepSeek V4 支持与 PR 历史

本文记录 vLLM 在提交 `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
附近对 DeepSeek V4 的真实状态。

- 状态: 当前 mainline 尚未支持，只有 open PR 证据

## 核心结论

- 当前 mainline 的 `vllm/model_executor/models/registry.py` 里还没有
  `DeepseekV4ForCausalLM`。
- 真正的 bring-up 工作集中在 open PR `#40760`，它不是只加 alias，而是
  一次性改了模型实现、MTP、tokenizer、renderer、tool parser、测试和
  spec-decode 配套。
- 另外两个 open PR 也已经构成 DeepSeek V4 证据链的一部分:
  `#40811` 负责 BF16 persistent top-k，
  `#40806` 负责 DSML 流式解析不泄漏 sentinel。

## 主要代码面

- 当前 mainline 检查点: `vllm/vllm/model_executor/models/registry.py`
- open-radar 代码面:
  `vllm/vllm/model_executor/models/deepseek_v4.py`,
  `vllm/vllm/model_executor/models/deepseek_v4_mtp.py`,
  `vllm/vllm/tokenizers/deepseek_v4.py`,
  `vllm/vllm/renderers/deepseek_v4.py`,
  `vllm/vllm/tool_parsers/deepseekv4_tool_parser.py`,
  `vllm/vllm/v1/spec_decode/eagle.py`,
  `vllm/csrc/persistent_topk.cuh`

## Open PR 雷达

- [#40760](https://github.com/vllm-project/vllm/pull/40760)
  `[New Model] Support DeepseekV4`
  已审 diff: `156` 个文件，`16193` 行新增，`760` 行删除。
  这组改动提出了 DeepSeek V4 主模型、MTP 草稿模型、tokenizer、renderer、
  parser、config 映射和 speculative decode 连接层。
- [#40811](https://github.com/vllm-project/vllm/pull/40811)
  `[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4`
  已审 diff: `3` 个文件，`886` 行新增，`330` 行删除。
  它把稀疏 top-k kernel 从默认 FP32 排序逻辑扩展到 BF16，并补了 BF16
  kernel 测试。
- [#40806](https://github.com/vllm-project/vllm/pull/40806)
  `[Bugfix] Fix the DSML token leakage in DSV4/3.2`
  已审 diff: `2` 个文件，`30` 行新增，`1` 行删除。
  它修掉了流式输出里半截 DSML sentinel 被当作普通文本吐出的 parser 问题。

## 当前结论

在 model alias 真正进入 mainline 之前，不要把 DeepSeek V4 写成 vLLM
已支持。等这些 PR 合入后，要把模型加载、tool calling、spec decode 和
BF16 sparse top-k 作为一整条链路一起回归。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `DeepSeek V4`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-24 | [#40760](https://github.com/vllm-project/vllm/pull/40760) | open | [New Model] Support DeepseekV4 | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `tests/kernels/test_fused_inv_rope_fp8_quant.py` |
| 2026-04-24 | [#40806](https://github.com/vllm-project/vllm/pull/40806) | open | [Bugfix] Fix the DSML token leakage in DSV4/3.2 | tests/benchmarks | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-24 | [#40811](https://github.com/vllm-project/vllm/pull/40811) | open | [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4 | MoE/router, kernel, tests/benchmarks | `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py` |

### 逐 PR 代码 diff 阅读记录

### PR #40760 - [New Model] Support DeepseekV4

- 链接：https://github.com/vllm-project/vllm/pull/40760
- 状态/时间：`open`，created 2026-04-24；作者 `zyongye`。
- 代码 diff 已读范围：`158` 个文件，`+16954/-760`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：kv, attention, cache, cuda, fp8, quant, config, spec, topk, triton。
- 代码 diff 细节：
  - `vllm/model_executor/models/deepseek_v4.py` added +1423/-0 (1423 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: DeepseekV4FP8Config, __init__, get_name, override_quantization_method
  - `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: DeepseekV4MLAModules:, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does
  - `tests/kernels/test_fused_inv_rope_fp8_quant.py` added +998/-0 (998 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: assert_dequant_close, rotate_gptj, make_cos_sin_cache, reference_inv_rope
  - `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `csrc/moe/topk_softplus_sqrt_kernels.cu` added +715/-0 (715 lines); hunk: +/*; 符号: alignas, int, int, int
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `tests/kernels/test_fused_inv_rope_fp8_quant.py`；patch 关键词为 kv, attention, cache, cuda, fp8, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `tests/kernels/test_fused_inv_rope_fp8_quant.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #40806 - [Bugfix] Fix the DSML token leakage in DSV4/3.2

- 链接：https://github.com/vllm-project/vllm/pull/40806
- 状态/时间：`open`，created 2026-04-24；作者 `chaunceyjiang`。
- 代码 diff 已读范围：`2` 个文件，`+76/-23`；代码面：tests/benchmarks；关键词：kv, test。
- 代码 diff 细节：
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunk: def test_no_emission_while_incomplete(self, parser):; 符号: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunk: Tool,; def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] \| None = None):; 符号: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`；patch 关键词为 kv, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #40811 - [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4

- 链接：https://github.com/vllm-project/vllm/pull/40811
- 状态/时间：`open`，created 2026-04-24；作者 `LopezCastroRoberto`。
- 代码 diff 已读范围：`3` 个文件，`+886/-330`；代码面：MoE/router, kernel, tests/benchmarks；关键词：cuda, topk, attention, config, flash, kv, mla, processor, spec, test。
- 代码 diff 细节：
  - `csrc/persistent_topk.cuh` modified +593/-218 (811 lines); hunk: #define PERSISTENT_TOPK_CUH_; __device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {; 符号: TopKDTypeTraits, TopKDTypeTraits, int, int
  - `csrc/topk.cu` modified +156/-112 (268 lines); hunk: -// Persistent TopK kernel for DeepSeek V3 sparse attention indexer.; #include "persistent_topk.cuh"; 符号: int, size_t, bool, size_t
  - `tests/kernels/test_top_k_per_row.py` modified +137/-0 (137 lines); hunk: def run_large_context_topk_test(; def run_large_context_topk_test(; 符号: run_large_context_topk_test, run_large_context_topk_test, run_large_context_topk_test, test_persistent_topk_padded_stride
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py`；patch 关键词为 cuda, topk, attention, config, flash, kv。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：3；open PR 数：3。
- 仍需跟进的 open PR：[#40760](https://github.com/vllm-project/vllm/pull/40760), [#40806](https://github.com/vllm-project/vllm/pull/40806), [#40811](https://github.com/vllm-project/vllm/pull/40811)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
