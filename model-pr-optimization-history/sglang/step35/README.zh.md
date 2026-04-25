# SGLang Step3.5 / Step3-VL 支持与 PR 历史

本文记录 SGLang 中与 Step3.5 / Step3-VL 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Step3.5 is split between text/MTP and VL processor work.
- All-reduce efficiency and parser behavior are the main axes to track.

## 主要代码面

- `sglang/python/sglang/srt/models/step3p5.py`
- `sglang/python/sglang/srt/models/step3p5_mtp.py`
- `sglang/python/sglang/srt/models/step3_vl.py`
- `sglang/python/sglang/srt/models/step3_vl_10b.py`

## 已合入 PR

- [#8583](https://github.com/sgl-project/sglang/pull/8583) `Support Step3V`：Initial Step3 visual model support.
- [#8699](https://github.com/sgl-project/sglang/pull/8699) `Support DP Attention for step3_vl`：Enabled multi-GPU VL serving.
- [#9695](https://github.com/sgl-project/sglang/pull/9695) `Add step3 tool parser`：Added tool-call parsing.
- [#18564](https://github.com/sgl-project/sglang/pull/18564) `Implement the standard multi-layer MTP for step3p5`：Added Step3.5 draft-model support.
- [#22773](https://github.com/sgl-project/sglang/pull/22773) `Optimize allreduce in MoE layers`：Targeted the Step3.5 MoE hot path.

## 配套 skill

- `skills/model-optimization/sglang/sglang-step35-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-step35-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Step 3.5`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-30 | [#8583](https://github.com/sgl-project/sglang/pull/8583) | merged | model: support Step3V | model wrapper, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py` |
| 2025-08-02 | [#8699](https://github.com/sgl-project/sglang/pull/8699) | merged | feat: Support DP Attention for step3_vl | model wrapper, attention/backend, multimodal/processor | `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py` |
| 2025-08-27 | [#9695](https://github.com/sgl-project/sglang/pull/9695) | merged | [router] add step3 tool parser | MoE/router, tests/benchmarks | `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs` |
| 2026-02-10 | [#18564](https://github.com/sgl-project/sglang/pull/18564) | merged | [Feature] implement the standard multi-layer MTP for step3p5 | kernel, scheduler/runtime | `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` |
| 2026-04-14 | [#22773](https://github.com/sgl-project/sglang/pull/22773) | merged | [Step3p5] Optimize allreduce in MoE layers | model wrapper | `python/sglang/srt/models/step3p5.py` |

### 逐 PR 代码 diff 阅读记录

### PR #8583 - model: support Step3V

- 链接：https://github.com/sgl-project/sglang/pull/8583
- 状态/时间：`merged`，created 2025-07-30, merged 2025-07-31；作者 `CatherineSue`。
- 代码 diff 已读范围：`16` 个文件，`+2340/-23`；代码面：model wrapper, multimodal/processor, tests/benchmarks, docs/config；关键词：config, spec, vision, attention, cuda, expert, moe, processor, deepep, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/step3_vl.py` added +994/-0 (994 lines); hunk: +import logging; 符号: Step3TextMLP, __init__, forward, Step3TextMoEMLP
  - `python/sglang/srt/multimodal/processors/step3_vl.py` added +515/-0 (515 lines); hunk: +import math; 符号: GPUToTensor, forward, Step3VisionProcessor:, __init__
  - `python/sglang/srt/function_call/step3_detector.py` added +436/-0 (436 lines); hunk: +import ast; 符号: get_argument_type, parse_arguments, Step3Detector, __init__
  - `python/sglang/srt/configs/step3_vl.py` added +172/-0 (172 lines); hunk: +from typing import Any, Optional, Union; 符号: Step3VisionEncoderConfig, __init__, Step3TextConfig, __init__
  - `test/srt/test_reasoning_parser.py` modified +112/-0 (112 lines); hunk: def test_qwen3_thinking_streaming_scenario(self):; 符号: test_qwen3_thinking_streaming_scenario, TestBufferLossBugFix, test_partial_end_tag_buffer_loss_bug, test_partial_start_tag_buffer_preservation
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py`；patch 关键词为 config, spec, vision, attention, cuda, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8699 - feat: Support DP Attention for step3_vl

- 链接：https://github.com/sgl-project/sglang/pull/8699
- 状态/时间：`merged`，created 2025-08-02, merged 2025-08-03；作者 `yhyang201`。
- 代码 diff 已读范围：`3` 个文件，`+25/-6`；代码面：model wrapper, attention/backend, multimodal/processor；关键词：config, attention, quant, vision, cuda, kv, processor。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/vision.py` modified +13/-5 (18 lines); hunk: import torch.nn.functional as F; def __init__(; 符号: __init__, __init__, __init__
  - `python/sglang/srt/models/step3_vl.py` modified +9/-0 (9 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
  - `python/sglang/srt/multimodal/processors/step3_vl.py` modified +3/-1 (4 lines); hunk: from PIL import Image; def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`；patch 关键词为 config, attention, quant, vision, cuda, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9695 - [router] add step3 tool parser

- 链接：https://github.com/sgl-project/sglang/pull/9695
- 状态/时间：`merged`，created 2025-08-27, merged 2025-08-27；作者 `slin1237`。
- 代码 diff 已读范围：`5` 个文件，`+600/-2`；代码面：MoE/router, tests/benchmarks；关键词：router, config, test, spec。
- 代码 diff 细节：
  - `sgl-router/src/tool_parser/parsers/step3_parser.rs` added +348/-0 (348 lines); hunk: +use async_trait::async_trait;; 符号: Step3Parser
  - `sgl-router/tests/tool_parser_step3.rs` added +245/-0 (245 lines); hunk: +//! Step3 Parser Integration Tests
  - `sgl-router/src/tool_parser/registry.rs` modified +3/-1 (4 lines); hunk: use crate::tool_parser::parsers::{; impl ParserRegistry {
  - `sgl-router/src/tool_parser/parsers/mod.rs` modified +3/-0 (3 lines); hunk: pub mod llama_parser;
  - `sgl-router/src/tool_parser/mod.rs` modified +1/-1 (2 lines); hunk: pub use types::{FunctionCall, PartialToolCall, StreamResult, TokenConfig, ToolCa
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs`；patch 关键词为 router, config, test, spec。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18564 - [Feature] implement the standard multi-layer MTP for step3p5

- 链接：https://github.com/sgl-project/sglang/pull/18564
- 状态/时间：`merged`，created 2026-02-10, merged 2026-03-04；作者 `zhaziqwe`。
- 代码 diff 已读范围：`2` 个文件，`+31/-2`；代码面：kernel, scheduler/runtime；关键词：eagle, spec, triton, cache, config, cuda, kv, topk。
- 代码 diff 细节：
  - `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +21/-2 (23 lines); hunk: def __init__(; def _draft_extend_for_prefill(; 符号: __init__, _draft_extend_for_prefill, forward_batch_generation
  - `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` modified +10/-0 (10 lines); hunk: def run_once():; 符号: run_once
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`；patch 关键词为 eagle, spec, triton, cache, config, cuda。影响判断：CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22773 - [Step3p5] Optimize allreduce in MoE layers

- 链接：https://github.com/sgl-project/sglang/pull/22773
- 状态/时间：`merged`，created 2026-04-14, merged 2026-04-16；作者 `yhyang201`。
- 代码 diff 已读范围：`1` 个文件，`+59/-57`；代码面：model wrapper；关键词：attention, config, cuda, expert, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/step3p5.py` modified +59/-57 (116 lines); hunk: -import logging; Step3p5Config = None; 符号: __init__, __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/step3p5.py`；patch 关键词为 attention, config, cuda, expert, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/step3p5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
