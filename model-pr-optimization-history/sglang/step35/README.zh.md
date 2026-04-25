# sglang Step 3.5 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/sglang/srt/configs/step3p5.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084) |
| `python/sglang/srt/models/step3p5.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084), [#22076](https://github.com/sgl-project/sglang/pull/22076), [#22773](https://github.com/sgl-project/sglang/pull/22773) |
| `python/sglang/srt/models/step3p5_mtp.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084) |
| `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 3
- 原文档显式引用补充 PR 数: 4
- 当前文档总 PR 数: 7
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-31 | [#8583](https://github.com/sgl-project/sglang/pull/8583) | merged | model: support Step3V | `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py` |
| 2025-08-03 | [#8699](https://github.com/sgl-project/sglang/pull/8699) | merged | feat: Support DP Attention for step3_vl | `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py` |
| 2025-08-27 | [#9695](https://github.com/sgl-project/sglang/pull/9695) | merged | [router] add step3 tool parser | `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs` |
| 2026-02-02 | [#18084](https://github.com/sgl-project/sglang/pull/18084) | merged | add Step-3.5-Flash model support | `python/sglang/srt/models/step3p5.py`, `python/sglang/srt/models/step3p5_mtp.py`, `python/sglang/srt/configs/step3p5.py` |
| 2026-03-04 | [#18564](https://github.com/sgl-project/sglang/pull/18564) | merged | [Feature] implement the standard multi-layer MTP for step3p5 | `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` |
| 2026-04-04 | [#22076](https://github.com/sgl-project/sglang/pull/22076) | merged | Tiny fix step3.5-flash launch crash | `python/sglang/srt/models/step3p5.py` |
| 2026-04-16 | [#22773](https://github.com/sgl-project/sglang/pull/22773) | merged | [Step3p5] Optimize allreduce in MoE layers | `python/sglang/srt/models/step3p5.py` |

## 逐 PR diff 审计卡

### PR #8583 - model: support Step3V

- 链接: https://github.com/sgl-project/sglang/pull/8583
- 状态/时间: merged / 2025-07-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+2340/-23，可读 patch 2530 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Step 3.5 补齐模型支持入口或运行时能力，标题为「model: support Step3V」，变更集中在 `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py`。PR 描述补充为：## Motivation This PR adds the support for Step3VModel. Co-authored-by: Qiaolin-Yu Co-authored-by: ispobock Co-authored-by: nnnobody-code Co-authored-by: jimpang ## Modification...
- 实现要点: `python/sglang/srt/models/step3_vl.py` added +994/-0 (994 lines); hunks: -0,0 +1,994; symbols: Step3TextMLP, __init__, forward, Step3TextMoEMLP，涉及 `Step3TextMLP, __init__, forward`；`python/sglang/srt/multimodal/processors/step3_vl.py` added +515/-0 (515 lines); hunks: -0,0 +1,515; symbols: GPUToTensor, forward, Step3VisionProcessor, __init__，涉及 `GPUToTensor, forward, Step3VisionProcessor`；`python/sglang/srt/function_call/step3_detector.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: get_argument_type, parse_arguments, Step3Detector, __init__，涉及 `get_argument_type, parse_arguments, Step3Detector`；`python/sglang/srt/configs/step3_vl.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Step3VisionEncoderConfig, __init__, Step3TextConfig, Step3VLConfig，涉及 `Step3VisionEncoderConfig, __init__, Step3TextConfig`。
- 代码 diff 细节:
  - `python/sglang/srt/models/step3_vl.py` added +994/-0 (994 lines); hunks: -0,0 +1,994; symbols: Step3TextMLP, __init__, forward, Step3TextMoEMLP
  - `python/sglang/srt/multimodal/processors/step3_vl.py` added +515/-0 (515 lines); hunks: -0,0 +1,515; symbols: GPUToTensor, forward, Step3VisionProcessor, __init__
  - `python/sglang/srt/function_call/step3_detector.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: get_argument_type, parse_arguments, Step3Detector, __init__
  - `python/sglang/srt/configs/step3_vl.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Step3VisionEncoderConfig, __init__, Step3TextConfig, Step3VLConfig
  - `test/srt/test_reasoning_parser.py` modified +112/-0 (112 lines); hunks: -493,5 +493,117 @@ def test_qwen3_thinking_streaming_scenario(self):; symbols: test_qwen3_thinking_streaming_scenario, TestBufferLossBugFix, test_partial_end_tag_buffer_loss_bug, test_partial_start_tag_buffer_preservation
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/step3_vl.py
@@ -0,0 +1,994 @@
+import logging
+import math
+from collections.abc import Iterable
+from math import sqrt
+from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, TypedDict, Union
+import torch
diff -- python/sglang/srt/multimodal/processors/step3_vl.py
@@ -0,0 +1,515 @@
+import math
+import re
+from itertools import product
+from typing import List, Literal, Optional, TypedDict, Union
+import numpy as np
+import torch
diff -- python/sglang/srt/function_call/step3_detector.py
@@ -0,0 +1,436 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/step3_vl.py` added +994/-0; `python/sglang/srt/multimodal/processors/step3_vl.py` added +515/-0; `python/sglang/srt/function_call/step3_detector.py` added +436/-0; `python/sglang/srt/configs/step3_vl.py` added +172/-0; `python/sglang/srt/configs/__init__.py` modified +8/-0; `python/sglang/srt/configs/model_config.py` modified +3/-0
  - tests: `test/srt/test_reasoning_parser.py` modified +112/-0
- 验证与风险: diff 自带测试面 `test/srt/test_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8699 - feat: Support DP Attention for step3_vl

- 链接: https://github.com/sgl-project/sglang/pull/8699
- 状态/时间: merged / 2025-08-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+25/-6，可读 patch 107 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Step 3.5 补齐模型支持入口或运行时能力，标题为「feat: Support DP Attention for step3_vl」，变更集中在 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`。PR 描述补充为：## Motivation Support DP Attention for step3_vl ## Modifications In the implementation prior to `step3_vl`, DP Attention was already supported for the LLM. This update extends D...
- 实现要点: `python/sglang/srt/layers/attention/vision.py` modified +13/-5 (18 lines); hunks: -11,6 +11,7; -365,19 +366,20 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/step3_vl.py` modified +9/-0 (9 lines); hunks: -531,11 +531,18 @@ def __init__(; -544,6 +551,8 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/multimodal/processors/step3_vl.py` modified +3/-1 (4 lines); hunks: -8,7 +8,7; -276,6 +276,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/vision.py` modified +13/-5 (18 lines); hunks: -11,6 +11,7; -365,19 +366,20 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/step3_vl.py` modified +9/-0 (9 lines); hunks: -531,11 +531,18 @@ def __init__(; -544,6 +551,8 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/multimodal/processors/step3_vl.py` modified +3/-1 (4 lines); hunks: -8,7 +8,7; -276,6 +276,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/vision.py
@@ -11,6 +11,7 @@
+from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
@@ -365,19 +366,20 @@ def __init__(
-        world_size = parallel_state.get_tensor_model_parallel_world_size()
-        self.tp_size = world_size
-        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
+        attn_tp_rank = get_attention_tp_rank()
diff -- python/sglang/srt/models/step3_vl.py
@@ -531,11 +531,18 @@ def __init__(
+        # Since this is a dense model,
+        # the MLP component likewise adopts a DP-MLP approach modeled after DP Attention.
+        # This choice may not represent the optimal solution and remains open to further deliberation.
+        attn_tp_rank = get_attention_tp_rank()
+        attn_tp_size = get_attention_tp_size()
+            tp_rank=attn_tp_rank,
diff -- python/sglang/srt/multimodal/processors/step3_vl.py
@@ -8,7 +8,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/vision.py` modified +13/-5; `python/sglang/srt/models/step3_vl.py` modified +9/-0; `python/sglang/srt/multimodal/processors/step3_vl.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9695 - [router] add step3 tool parser

- 链接: https://github.com/sgl-project/sglang/pull/9695
- 状态/时间: merged / 2025-08-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+600/-2，可读 patch 634 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Step 3.5 补齐模型支持入口或运行时能力，标题为「[router] add step3 tool parser」，变更集中在 `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs`。PR 描述补充为：## Motivation step3 tool parser ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [x] Format your code according to the Format code with pre-commit...
- 实现要点: `sgl-router/src/tool_parser/parsers/step3_parser.rs` added +348/-0 (348 lines); hunks: -0,0 +1,348；`sgl-router/tests/tool_parser_step3.rs` added +245/-0 (245 lines); hunks: -0,0 +1,245；`sgl-router/src/tool_parser/registry.rs` modified +3/-1 (4 lines); hunks: -1,5 +1,5; -113,6 +113,8 @@ impl ParserRegistry {；`sgl-router/src/tool_parser/parsers/mod.rs` modified +3/-0 (3 lines); hunks: -9,12 +9,15 @@ pub mod llama_parser;。
- 代码 diff 细节:
  - `sgl-router/src/tool_parser/parsers/step3_parser.rs` added +348/-0 (348 lines); hunks: -0,0 +1,348
  - `sgl-router/tests/tool_parser_step3.rs` added +245/-0 (245 lines); hunks: -0,0 +1,245
  - `sgl-router/src/tool_parser/registry.rs` modified +3/-1 (4 lines); hunks: -1,5 +1,5; -113,6 +113,8 @@ impl ParserRegistry {
  - `sgl-router/src/tool_parser/parsers/mod.rs` modified +3/-0 (3 lines); hunks: -9,12 +9,15 @@ pub mod llama_parser;
  - `sgl-router/src/tool_parser/mod.rs` modified +1/-1 (2 lines); hunks: -25,5 +25,5 @@ pub use types::{FunctionCall, PartialToolCall, StreamResult, T...
- 关键代码摘录:

```diff
diff -- sgl-router/src/tool_parser/parsers/step3_parser.rs
@@ -0,0 +1,348 @@
+use async_trait::async_trait;
+use regex::Regex;
+use serde_json::Value;
+use crate::tool_parser::{
+    errors::{ToolParserError, ToolParserResult},
+    state::ParseState,
diff -- sgl-router/tests/tool_parser_step3.rs
@@ -0,0 +1,245 @@
+//! Step3 Parser Integration Tests
+use sglang_router_rs::tool_parser::{ParseState, Step3Parser, StreamResult, ToolParser};
+#[tokio::test]
+async fn test_step3_complete_parsing() {
+    let parser = Step3Parser::new();
+    // Test single tool call
diff -- sgl-router/src/tool_parser/registry.rs
@@ -1,5 +1,5 @@
```

- 已读文件:
  - runtime: `sgl-router/src/tool_parser/parsers/step3_parser.rs` added +348/-0; `sgl-router/src/tool_parser/registry.rs` modified +3/-1; `sgl-router/src/tool_parser/parsers/mod.rs` modified +3/-0; `sgl-router/src/tool_parser/mod.rs` modified +1/-1
  - tests: `sgl-router/tests/tool_parser_step3.rs` added +245/-0
- 验证与风险: diff 自带测试面 `sgl-router/tests/tool_parser_step3.rs`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18084 - add Step-3.5-Flash model support

- 链接: https://github.com/sgl-project/sglang/pull/18084
- 状态/时间: merged / 2026-02-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/step3p5.py`, `python/sglang/srt/models/step3p5.py`, `python/sglang/srt/models/step3p5_mtp.py`；关联提交 `980d2936cd9a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+1557/-12，可读 patch 1711 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Step 3.5 补齐模型支持入口或运行时能力，标题为「add Step-3.5-Flash model support」，变更集中在 `python/sglang/srt/models/step3p5.py`, `python/sglang/srt/models/step3p5_mtp.py`, `python/sglang/srt/configs/step3p5.py`。PR 描述补充为：## Motivation add Step-3.5-Flash model support ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code...
- 实现要点: `python/sglang/srt/models/step3p5.py` added +1037/-0 (1037 lines); hunks: -0,0 +1,1037; symbols: Step3p5MLP, __init__, forward, Step3p5MoEMLP，涉及 `Step3p5MLP, __init__, forward`；`python/sglang/srt/models/step3p5_mtp.py` added +336/-0 (336 lines); hunks: -0,0 +1,336; symbols: get_spec_layer_idx_from_weight_name, SharedHead, __init__, forward，涉及 `get_spec_layer_idx_from_weight_name, SharedHead, __init__`；`python/sglang/srt/configs/step3p5.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p5Config, __init__，涉及 `Step3p5Config, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/step3p5.py` added +1037/-0 (1037 lines); hunks: -0,0 +1,1037; symbols: Step3p5MLP, __init__, forward, Step3p5MoEMLP
  - `python/sglang/srt/models/step3p5_mtp.py` added +336/-0 (336 lines); hunks: -0,0 +1,336; symbols: get_spec_layer_idx_from_weight_name, SharedHead, __init__, forward
  - `python/sglang/srt/configs/step3p5.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p5Config, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/step3p5.py
@@ -0,0 +1,1037 @@
+import logging
+import os
+from typing import Any, Dict, Iterable, Optional, Tuple, Union
+import torch
+import torch.nn.functional as F
+from torch import nn
diff -- python/sglang/srt/models/step3p5_mtp.py
@@ -0,0 +1,336 @@
+import logging
+from collections.abc import Iterable
+from typing import Optional
+import torch
+import torch.nn as nn
+from transformers import PretrainedConfig
diff -- python/sglang/srt/configs/step3p5.py
@@ -0,0 +1,97 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/step3p5.py` added +1037/-0; `python/sglang/srt/models/step3p5_mtp.py` added +336/-0; `python/sglang/srt/configs/step3p5.py` added +97/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/step3p5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18564 - [Feature] implement the standard multi-layer MTP for step3p5

- 链接: https://github.com/sgl-project/sglang/pull/18564
- 状态/时间: merged / 2026-03-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+31/-2，可读 patch 61 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Step 3.5 优化关键推理路径或后端选择，标题为「[Feature] implement the standard multi-layer MTP for step3p5」，变更集中在 `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`。PR 描述补充为：## Motivation The current SGL multi-layer MTP behavior can deviate from the standard Step3.5 Flash design, where hidden states should be propagated step-by-step across MTP layer...
- 实现要点: `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +21/-2 (23 lines); hunks: -127,6 +127,11 @@ def __init__(; -382,6 +387,15 @@ def _draft_extend_for_prefill(; symbols: __init__, _draft_extend_for_prefill, forward_batch_generation，涉及 `__init__, _draft_extend_for_prefill, forward_batch_generation`；`python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: -387,6 +387,16 @@ def run_once():; symbols: run_once，涉及 `run_once`。
- 代码 diff 细节:
  - `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +21/-2 (23 lines); hunks: -127,6 +127,11 @@ def __init__(; -382,6 +387,15 @@ def _draft_extend_for_prefill(; symbols: __init__, _draft_extend_for_prefill, forward_batch_generation
  - `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: -387,6 +387,16 @@ def run_once():; symbols: run_once
- 关键代码摘录:

```diff
diff -- python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py
@@ -127,6 +127,11 @@ def __init__(
+        # Chain-style MTP: each step propagates its own output hidden states to the
+        # next step.  Non-chain: each step uses the target model's hidden states.
+        draft_arch = self.draft_worker.model_config.hf_config.architectures[0]
+        self.chain_mtp_hidden_states = draft_arch in ["Step3p5MTP"]
@@ -382,6 +387,15 @@ def _draft_extend_for_prefill(
+            # Chain-style: use this step's output hidden_states as next step's input
diff -- python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py
@@ -387,6 +387,16 @@ def run_once():
+            # Chain-style MTP: overwrite self.hidden_states with the draft model's
+            # output (hidden_states_before_norm) so that assign_new_state_triton
+            # propagates each MTP layer's own output to the next MTP layer,
+            # rather than always feeding the target model's hidden states.
+            if (
+                self.eagle_worker.chain_mtp_hidden_states
```

- 已读文件:
  - runtime: `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +21/-2; `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` modified +10/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`, `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22076 - Tiny fix step3.5-flash launch crash

- 链接: https://github.com/sgl-project/sglang/pull/22076
- 状态/时间: merged / 2026-04-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/step3p5.py`；关联提交 `ef130312434c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-1，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Step 3.5 修复已暴露的启动、加载、解析或数值问题，标题为「Tiny fix step3.5-flash launch crash」，变更集中在 `python/sglang/srt/models/step3p5.py`。PR 描述补充为：## Motivation Before this, launch step3.5-flash will just crash since step3.5-flash config doesn't have `pad_token_id` afaik, `padding_idx` is not used in the model file, so I s...
- 实现要点: `python/sglang/srt/models/step3p5.py` modified +0/-1 (1 lines); hunks: -667,7 +667,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/step3p5.py` modified +0/-1 (1 lines); hunks: -667,7 +667,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/step3p5.py
@@ -667,7 +667,6 @@ def __init__(
-        self.padding_idx = config.pad_token_id
```

- 已读文件:
  - runtime: `python/sglang/srt/models/step3p5.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/step3p5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22773 - [Step3p5] Optimize allreduce in MoE layers

- 链接: https://github.com/sgl-project/sglang/pull/22773
- 状态/时间: merged / 2026-04-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/step3p5.py`；关联提交 `b8794baa6d61`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+59/-57，可读 patch 211 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Step 3.5 补齐模型支持入口或运行时能力，标题为「[Step3p5] Optimize allreduce in MoE layers」，变更集中在 `python/sglang/srt/models/step3p5.py`。PR 描述补充为：## Motivation ## Modifications - Defer o_proj and share_expert all-reduce, combine with MoE output into a single all-reduce per layer (was 3 separate all-reduces) - Enable allre...
- 实现要点: `python/sglang/srt/models/step3p5.py` modified +59/-57 (116 lines); hunks: -1,5 +1,3; -57,7 +55,6; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/step3p5.py` modified +59/-57 (116 lines); hunks: -1,5 +1,3; -57,7 +55,6; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/step3p5.py
@@ -1,5 +1,3 @@
-import logging
-import os
@@ -57,7 +55,6 @@
-logger = logging.getLogger(__name__)
@@ -69,6 +66,9 @@ def __init__(
+        tp_size: Optional[int] = None,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/step3p5.py` modified +59/-57
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/step3p5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
