# SGLang MiMo-V2-Flash 支持与 PR 历史

本文记录 SGLang 中与 MiMo-V2-Flash 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- MiMo-V2-Flash is primarily a throughput-oriented MoE serving family.
- All-reduce fusion, overlap, and reasoning behavior matter more than generic text-only loader work.

## 主要代码面

- `sglang/python/sglang/srt/models/mimo_v2_flash.py`
- `sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py`

## 已合入 PR

- [#15207](https://github.com/sgl-project/sglang/pull/15207) `MiMo-V2-Flash day0 support`：Initial MiMo-V2-Flash landing.
- [#15464](https://github.com/sgl-project/sglang/pull/15464) `Optimize MiMo-V2-Flash by flashinfer fused allreduce`：Targeted decode-side communication cost.
- [#15488](https://github.com/sgl-project/sglang/pull/15488) `Respect `--swa-full-tokens-ratio``：Fixed a concrete runtime flag integration bug.
- [#17634](https://github.com/sgl-project/sglang/pull/17634) `Support two batch overlap`：Added overlap / throughput optimization.
- [#21414](https://github.com/sgl-project/sglang/pull/21414) `Add mimo reasoning parser`：Completed the parser path for thinking outputs.

## 配套 skill

- `skills/model-optimization/sglang/sglang-mimo-v2-flash-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-mimo-v2-flash-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `MiMo-V2-Flash`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-12-15 | [#15207](https://github.com/sgl-project/sglang/pull/15207) | merged | [Feature] Xiaomi `MiMo-V2-Flash` day0 support | model wrapper, attention/backend, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/speculative/mtp_worker.py`, `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/speculative/mtp_worker_v2.py` |
| 2025-12-19 | [#15464](https://github.com/sgl-project/sglang/pull/15464) | merged | Optimize MiMo-V2-Flash by flashinfer fused allreduce | model wrapper | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2025-12-19 | [#15488](https://github.com/sgl-project/sglang/pull/15488) | merged | [MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg | scheduler/runtime | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py` |
| 2026-01-23 | [#17634](https://github.com/sgl-project/sglang/pull/17634) | merged | [MiMoV2Flash] [feat]: support two batch overlap | model wrapper | `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py` |
| 2026-03-25 | [#21414](https://github.com/sgl-project/sglang/pull/21414) | merged | fix(MiMo-V2-Flash): add mimo reasoning parser | misc | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py` |

### 逐 PR 代码 diff 阅读记录

### PR #15207 - [Feature] Xiaomi `MiMo-V2-Flash` day0 support

- 链接：https://github.com/sgl-project/sglang/pull/15207
- 状态/时间：`merged`，created 2025-12-15, merged 2025-12-19；作者 `acelyc111`。
- 代码 diff 已读范围：`38` 个文件，`+5396/-169`；代码面：model wrapper, attention/backend, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：spec, cache, attention, config, cuda, kv, topk, moe, processor, eagle。
- 代码 diff 细节：
  - `python/sglang/srt/speculative/mtp_worker.py` added +989/-0 (989 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: MTPWorker, __init__, init_attention_backend, init_cuda_graphs
  - `python/sglang/srt/models/mimo_v2_flash.py` added +927/-0 (927 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: MiMoV2MLP, __init__, forward, MoEGate
  - `python/sglang/srt/speculative/mtp_worker_v2.py` added +750/-0 (750 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: _get_plan_stream, MTPDraftWorker, __init__, mtp_model_runner
  - `python/sglang/srt/speculative/mtp_draft_extend_cuda_graph_runner.py` added +655/-0 (655 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: MTPDraftExtendCudaGraphRunner:, __init__, init_buffers_and_capture, can_run
  - `test/registered/function_call/test_function_call_parser.py` modified +441/-0 (441 lines); hunk: from sglang.srt.function_call.json_array_parser import JsonArrayParser; def check_single_todos(tool_result, expected):; 符号: check_single_todos, TestMiMoDetector, setUp, test_has_tool_call
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/speculative/mtp_worker.py`, `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/speculative/mtp_worker_v2.py`；patch 关键词为 spec, cache, attention, config, cuda, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/speculative/mtp_worker.py`, `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/speculative/mtp_worker_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15464 - Optimize MiMo-V2-Flash by flashinfer fused allreduce

- 链接：https://github.com/sgl-project/sglang/pull/15464
- 状态/时间：`merged`，created 2025-12-19, merged 2025-12-20；作者 `yuan-luo`。
- 代码 diff 已读范围：`1` 个文件，`+66/-10`；代码面：model wrapper；关键词：attention, config, deepep, eagle, expert, flash, fp4, moe, processor, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +66/-10 (76 lines); hunk: # ==============================================================================; RowParallelLinear,; 符号: __init__, forward, forward, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/mimo_v2_flash.py`；patch 关键词为 attention, config, deepep, eagle, expert, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/mimo_v2_flash.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15488 - [MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg

- 链接：https://github.com/sgl-project/sglang/pull/15488
- 状态/时间：`merged`，created 2025-12-19, merged 2025-12-25；作者 `acelyc111`。
- 代码 diff 已读范围：`2` 个文件，`+16/-16`；代码面：scheduler/runtime；关键词：cache, flash, kv, attention, config, eagle, spec。
- 代码 diff 细节：
  - `python/sglang/srt/model_executor/model_runner.py` modified +10/-12 (22 lines); hunk: def __init__(; def profile_max_num_token(self, total_gpu_memory: int):; 符号: __init__, profile_max_num_token, handle_max_mamba_cache, set_num_token_hybrid
  - `python/sglang/srt/server_args.py` modified +6/-4 (10 lines); hunk: def _handle_model_specific_adjustments(self):; def _handle_cache_compatibility(self):; 符号: _handle_model_specific_adjustments, _handle_cache_compatibility, _handle_deterministic_inference
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`；patch 关键词为 cache, flash, kv, attention, config, eagle。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17634 - [MiMoV2Flash] [feat]: support two batch overlap

- 链接：https://github.com/sgl-project/sglang/pull/17634
- 状态/时间：`merged`，created 2026-01-23, merged 2026-02-02；作者 `TZHelloWorld`。
- 代码 diff 已读范围：`2` 个文件，`+292/-8`；代码面：model wrapper；关键词：config, deepep, expert, moe, attention, cache, cuda, flash, kv, processor。
- 代码 diff 细节：
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8 (216 lines); hunk: import torch.nn.functional as F; kv_cache_scales_loader,; 符号: forward_deepep, op_gate, op_select_experts, op_dispatch_a
  - `python/sglang/srt/batch_overlap/operations_strategy.py` modified +84/-0 (84 lines); hunk: def init_new_tbo(; def _compute_moe_qwen3_decode(layer):; 符号: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_mimov2_layer_operations_strategy_tbo, _compute_moe_mimov2_prefill
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py`；patch 关键词为 config, deepep, expert, moe, attention, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21414 - fix(MiMo-V2-Flash): add mimo reasoning parser

- 链接：https://github.com/sgl-project/sglang/pull/21414
- 状态/时间：`merged`，created 2026-03-25, merged 2026-04-01；作者 `alphabetc1`。
- 代码 diff 已读范围：`2` 个文件，`+7/-0`；代码面：misc；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +6/-0 (6 lines); hunk: def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:; 符号: _get_reasoning_from_request
  - `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunk: class ReasoningParser:; 符号: ReasoningParser:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`；patch 关键词为 n/a。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
