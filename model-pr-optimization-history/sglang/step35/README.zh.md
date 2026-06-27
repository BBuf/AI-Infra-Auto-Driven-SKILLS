# sglang Step 3.5 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/sglang/srt/configs/step3p5.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084) |
| `python/sglang/srt/models/step3p5.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084), [#22076](https://github.com/sgl-project/sglang/pull/22076), [#22773](https://github.com/sgl-project/sglang/pull/22773) |
| `python/sglang/srt/models/step3p5_mtp.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084) |
| `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 3
- 原文档显式引用补充 PR 数: 12
- 当前文档总 PR 数: 15
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
| 2026-04-29 | [#24105](https://github.com/sgl-project/sglang/pull/24105) | merged | relax the threshold in test_step3p5_flash_chain_mtp | `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` |
| 2026-05-01 | [#24192](https://github.com/sgl-project/sglang/pull/24192) | merged | [spec decoding] add tests for chain-style multi layer eagle + return_logprob | `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` |
| 2026-05-14 | [#25197](https://github.com/sgl-project/sglang/pull/25197) | merged | ci: decouple stage and runner for cuda registry | `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py` |
| 2026-05-14 | [#25236](https://github.com/sgl-project/sglang/pull/25236) | merged | ci: H200 conditional split + dsv4 est_time recalibration (h200 partition 6→2) | `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py` |
| 2026-05-14 | [#24725](https://github.com/sgl-project/sglang/pull/24725) | merged | ci: tag-gated nightly migration — foundation + 40 whole-file moves | `test/registered/models/test_ministral4_models.py`, `test/registered/models/test_compressed_tensors_models.py`, `test/registered/models/test_generation_models.py` |
| 2026-05-28 | [#26610](https://github.com/sgl-project/sglang/pull/26610) | merged | test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit) | `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` |
| 2026-05-29 | [#26565](https://github.com/sgl-project/sglang/pull/26565) | merged | model: Step-3.7-Flash Support | `python/sglang/srt/models/step3p7.py`, `python/sglang/srt/configs/step3p7.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |

## 逐 PR diff 审计卡

### PR #8583 - model: support Step3V

- 链接: https://github.com/sgl-project/sglang/pull/8583
- 状态/时间: merged / 2025-07-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+2340/-23，可读 patch 2530 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: support Step3V」；模型线: Step 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py`；技术摘要: 覆盖「model: support Step3V」；主要实现面是 `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「feat: Support DP Attention for step3_vl」；模型线: Step 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`；技术摘要: 覆盖「feat: Support DP Attention for step3_vl」；主要实现面是 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[router] add step3 tool parser」；模型线: Step 3.5；类别: 文档/测试/CI；主要 diff: `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs`；技术摘要: 覆盖「[router] add step3 tool parser」；主要实现面是 `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/step3p5.py`, `python/sglang/srt/models/step3p5.py`, `python/sglang/srt/models/step3p5_mtp.py`；关联提交 `980d2936cd9a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+1557/-12，可读 patch 1711 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「add Step-3.5-Flash model support」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/step3p5.py`, `python/sglang/srt/models/step3p5_mtp.py`, `python/sglang/srt/configs/step3p5.py`；技术摘要: 覆盖「add Step-3.5-Flash model support」；主要实现面是 `python/sglang/srt/models/step3p5.py`, `python/sglang/srt/models/step3p5_mtp.py`, `python/sglang/srt/configs/step3p5.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Feature] implement the standard multi-layer MTP for step3p5」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`；技术摘要: 覆盖「[Feature] implement the standard multi-layer MTP for step3p5」；主要实现面是 `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/step3p5.py`；关联提交 `ef130312434c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-1，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Tiny fix step3.5-flash launch crash」；模型线: Step 3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/step3p5.py`；技术摘要: 覆盖「Tiny fix step3.5-flash launch crash」；主要实现面是 `python/sglang/srt/models/step3p5.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Step3p5] Optimize allreduce in MoE layers」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/step3p5.py`；技术摘要: 覆盖「[Step3p5] Optimize allreduce in MoE layers」；主要实现面是 `python/sglang/srt/models/step3p5.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #24105 - relax the threshold in test_step3p5_flash_chain_mtp

- 链接: https://github.com/sgl-project/sglang/pull/24105
- 状态/时间: merged / 2026-04-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「relax the threshold in test_step3p5_flash_chain_mtp」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`；技术摘要: 覆盖「relax the threshold in test_step3p5_flash_chain_mtp」；主要实现面是 `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def test_gsm8k(self):; symbols: test_gsm8k，涉及 `test_gsm8k`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def test_gsm8k(self):; symbols: test_gsm8k
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py
@@ -97,7 +97,7 @@ def test_gsm8k(self):
-            self.assertGreater(metrics["score"], 0.84)
+            self.assertGreater(metrics["score"], 0.83)
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24192 - [spec decoding] add tests for chain-style multi layer eagle + return_logprob

- 链接: https://github.com/sgl-project/sglang/pull/24192
- 状态/时间: merged / 2026-05-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+136/-0，可读 patch 150 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[spec decoding] add tests for chain-style multi layer eagle + return_logprob」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`；技术摘要: 覆盖「[spec decoding] add tests for chain-style multi layer eagle + return_logprob」；主要实现面是 `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +136/-0 (136 lines); hunks: -1,6 +1,7; -100,6 +101,141 @@ def test_gsm8k(self):; symbols: test_gsm8k, test_logprob_spec_v2_match，涉及 `test_gsm8k, test_logprob_spec_v2_match`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +136/-0 (136 lines); hunks: -1,6 +1,7; -100,6 +101,141 @@ def test_gsm8k(self):; symbols: test_gsm8k, test_logprob_spec_v2_match
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py
@@ -1,6 +1,7 @@
+import numpy as np
@@ -100,6 +101,141 @@ def test_gsm8k(self):
+    def test_logprob_spec_v2_match(self):
+        """Verify spec v2 decode logprobs match prefill scoring logprobs.
+        Generate tokens with chain MTP spec v2, then score the same sequence
+        via prefill-only (no speculation). The two sets of logprobs should be
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +136/-0
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25197 - ci: decouple stage and runner for cuda registry

- 链接: https://github.com/sgl-project/sglang/pull/25197
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 261 个文件，+388/-293，可读 patch 2625 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: decouple stage and runner for cuda registry」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py`；技术摘要: 覆盖「ci: decouple stage and runner for cuda registry」；主要实现面是 `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1 (3 lines); hunks: -19,7 +19,8；`test/registered/models/test_dummy_grok_models.py` modified +2/-1 (3 lines); hunks: -5,7 +5,8；`test/registered/models/test_ministral3_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8；`test/registered/models/test_ministral4_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8。
- 代码 diff 细节:
  - `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1 (3 lines); hunks: -19,7 +19,8
  - `test/registered/models/test_dummy_grok_models.py` modified +2/-1 (3 lines); hunks: -5,7 +5,8
  - `test/registered/models/test_ministral3_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8
  - `test/registered/models/test_ministral4_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8
  - `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +2/-1 (3 lines); hunks: -6,7 +6,8
- 关键代码摘录:

```diff
diff -- test/registered/layers/test_fla_layernorm_guard.py
@@ -19,7 +19,8 @@
-    suite="stage-b-test-2-gpu-large",
+    stage="stage-b",
+    runner_config="2-gpu-large",
diff -- test/registered/models/test_dummy_grok_models.py
@@ -5,7 +5,8 @@
-    suite="stage-b-test-2-gpu-large",
+    stage="stage-b",
+    runner_config="2-gpu-large",
diff -- test/registered/models/test_ministral3_models.py
@@ -8,7 +8,8 @@
-    suite="stage-b-test-1-gpu-small",
+    stage="stage-b",
+    runner_config="1-gpu-small",
diff -- test/registered/models/test_ministral4_models.py
@@ -8,7 +8,8 @@
-    suite="stage-b-test-2-gpu-large",
```

- 已读文件:
  - tests: `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1; `test/registered/models/test_dummy_grok_models.py` modified +2/-1; `test/registered/models/test_ministral3_models.py` modified +2/-1; `test/registered/models/test_ministral4_models.py` modified +2/-1; `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +2/-1; `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/ci/ci_register.py`, `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py`, `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25236 - ci: H200 conditional split + dsv4 est_time recalibration (h200 partition 6→2)

- 链接: https://github.com/sgl-project/sglang/pull/25236
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+13/-13，可读 patch 117 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: H200 conditional split + dsv4 est_time recalibration (h200 partition 6→2)」；模型线: Step 3.5；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`；技术摘要: 覆盖「ci: H200 conditional split + dsv4 est_time recalibration (h200 partition 6→2)」；主要实现面是 `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7；`test/registered/8-gpu-models/test_deepseek_v3_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7；`test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7；`test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7; symbols: TestMiMoV2Flash，涉及 `TestMiMoV2Flash`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
  - `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7
  - `test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7
  - `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7; symbols: TestMiMoV2Flash
  - `test/registered/8-gpu-models/test_minimax_m25_basic.py` modified +1/-1 (2 lines); hunks: -14,7 +14,7
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_deepseek_v32_indexcache.py
@@ -13,7 +13,7 @@
-register_cuda_ci(est_time=492, stage="stage-c", runner_config="8-gpu-h200")
+register_cuda_ci(est_time=450, suite="nightly-8-gpu-h200", nightly=True)
diff -- test/registered/8-gpu-models/test_deepseek_v3_mtp.py
@@ -17,7 +17,7 @@
-register_cuda_ci(est_time=309, stage="stage-c", runner_config="8-gpu-h200")
+register_cuda_ci(est_time=300, stage="stage-c", runner_config="8-gpu-h200")
diff -- test/registered/8-gpu-models/test_dsa_models_mtp.py
@@ -17,7 +17,7 @@
-    est_time=1048,
+    est_time=1030,
diff -- test/registered/8-gpu-models/test_mimo_models.py
@@ -6,7 +6,7 @@
-register_cuda_ci(est_time=610, stage="stage-c", runner_config="8-gpu-h200")
+register_cuda_ci(est_time=500, stage="stage-c", runner_config="8-gpu-h200")
diff -- test/registered/8-gpu-models/test_minimax_m25_basic.py
@@ -14,7 +14,7 @@
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` modified +1/-1; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` modified +1/-1; `test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +1/-1; `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1; `test/registered/8-gpu-models/test_minimax_m25_basic.py` modified +1/-1; `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`, `test/registered/8-gpu-models/test_mimo_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24725 - ci: tag-gated nightly migration — foundation + 40 whole-file moves

- 链接: https://github.com/sgl-project/sglang/pull/24725
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 78 个文件，+2263/-2140，可读 patch 4964 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: tag-gated nightly migration — foundation + 40 whole-file moves」；模型线: Step 3.5；类别: 文档/测试/CI；主要 diff: `test/registered/models/test_ministral4_models.py`, `test/registered/models/test_compressed_tensors_models.py`, `test/registered/models/test_generation_models.py`；技术摘要: 覆盖「ci: tag-gated nightly migration — foundation + 40 whole-file moves」；主要实现面是 `test/registered/models/test_ministral4_models.py`, `test/registered/models/test_compressed_tensors_models.py`, `test/registered/models/test_generation_models.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models/test_ministral4_models.py` modified +1/-5 (6 lines); hunks: -6,11 +6,7；`test/registered/models/test_compressed_tensors_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7；`test/registered/models/test_generation_models.py` modified +1/-1 (2 lines); hunks: -1,7 +1,7；`test/registered/models/test_vlm_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7。
- 代码 diff 细节:
  - `test/registered/models/test_ministral4_models.py` modified +1/-5 (6 lines); hunks: -6,11 +6,7
  - `test/registered/models/test_compressed_tensors_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
  - `test/registered/models/test_generation_models.py` modified +1/-1 (2 lines); hunks: -1,7 +1,7
  - `test/registered/models/test_vlm_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
  - `test/manual/openai_server/function_call/test_tool_choice.py` renamed +0/-0 (0 lines)
- 关键代码摘录:

```diff
diff -- test/registered/models/test_ministral4_models.py
@@ -6,11 +6,7 @@
-register_cuda_ci(
-    est_time=200,
-    stage="stage-b",
-    runner_config="2-gpu-large",
-)
+register_cuda_ci(est_time=200, stage="extra-a", runner_config="2-gpu-large")
diff -- test/registered/models/test_compressed_tensors_models.py
@@ -13,7 +13,7 @@
-register_cuda_ci(est_time=65, stage="stage-b", runner_config="1-gpu-large")
+register_cuda_ci(est_time=65, stage="extra-a", runner_config="1-gpu-large")
diff -- test/registered/models/test_generation_models.py
@@ -1,7 +1,7 @@
-register_cuda_ci(est_time=150, stage="stage-b", runner_config="1-gpu-large")
+register_cuda_ci(est_time=150, stage="extra-a", runner_config="1-gpu-large")
diff -- test/registered/models/test_vlm_models.py
@@ -13,7 +13,7 @@
```

- 已读文件:
  - tests: `test/registered/models/test_ministral4_models.py` modified +1/-5; `test/registered/models/test_compressed_tensors_models.py` modified +1/-1; `test/registered/models/test_generation_models.py` modified +1/-1; `test/registered/models/test_vlm_models.py` modified +1/-1; `test/manual/openai_server/function_call/test_tool_choice.py` renamed +0/-0; `test/registered/sessions/test_streaming_session.py` modified +62/-1072
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/streaming_session_kit.py`, `python/sglang/test/server_fixtures/hybrid_attn_backend_fixture.py`, `python/sglang/test/server_fixtures/ngram_fixture.py`, `python/sglang/test/server_fixtures/pcg_spec_fixture.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26610 - test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)

- 链接: https://github.com/sgl-project/sglang/pull/26610
- 状态/时间: merged / 2026-05-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+611/-816，可读 patch 1566 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`；技术摘要: 覆盖「test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)」；主要实现面是 `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212 (212 lines); hunks: -1,212 +0,0; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass`；`python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133 (134 lines); hunks: -1,25 +1,12; -28,18 +15,8 @@ def _random_suffixes(n, length, seed):; symbols: _random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching, TestUnifiedFullRadixCache，涉及 `_random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching`；`test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78 (111 lines); hunks: -1,28 +1,20; -31,75 +23,38 @@ class TestStep3p5FlashChainMTP(CustomTestCase):; symbols: TestStep3p5FlashChainMTP, setUpClass, tearDownClass，涉及 `TestStep3p5FlashChainMTP, setUpClass, tearDownClass`；`test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110 (110 lines); hunks: -1,110 +0,0; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV3MTP, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212 (212 lines); hunks: -1,212 +0,0; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133 (134 lines); hunks: -1,25 +1,12; -28,18 +15,8 @@ def _random_suffixes(n, length, seed):; symbols: _random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching, TestUnifiedFullRadixCache
  - `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78 (111 lines); hunks: -1,28 +1,20; -31,75 +23,38 @@ class TestStep3p5FlashChainMTP(CustomTestCase):; symbols: TestStep3p5FlashChainMTP, setUpClass, tearDownClass
  - `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110 (110 lines); hunks: -1,110 +0,0; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-105 (105 lines); hunks: -1,105 +0,0; symbols: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k
- 关键代码摘录:

```diff
diff -- test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py
@@ -1,212 +0,0 @@
-import unittest
-from types import SimpleNamespace
-import requests
-from sglang.srt.utils import kill_process_tree
-from sglang.test.ci.ci_register import register_cuda_ci
-from sglang.test.run_eval import run_eval
diff -- python/sglang/test/kits/unified_radix_cache_kit.py
@@ -1,25 +1,12 @@
-import unittest
-from sglang.srt.utils import kill_process_tree
-from sglang.test.ci.ci_register import register_cuda_ci
-    get_input_ids,
-    make_mamba_decode_assert,
-    make_mamba_prefill_assert,
diff -- test/registered/models_e2e/test_step3p5_flash_chain_mtp.py
@@ -1,28 +1,20 @@
```

- 已读文件:
  - tests: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212; `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133; `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110; `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-105; `test/registered/quant/test_deepseek_v3_fp4_4gpu.py` removed +0/-80
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/manual/core/test_dsv4_hicache_swa_translation_cache.py`, `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26565 - model: Step-3.7-Flash Support

- 链接: https://github.com/sgl-project/sglang/pull/26565
- 状态/时间: merged / 2026-05-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+1094/-7，可读 patch 1284 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: Step-3.7-Flash Support」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/step3p7.py`, `python/sglang/srt/configs/step3p7.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`；技术摘要: 覆盖「model: Step-3.7-Flash Support」；主要实现面是 `python/sglang/srt/models/step3p7.py`, `python/sglang/srt/configs/step3p7.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/step3p7.py` added +200/-0 (200 lines); hunks: -0,0 +1,200; symbols: Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__, _get_vision_model_output，涉及 `Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__`；`python/sglang/srt/configs/step3p7.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p7VisionEncoderConfig, __init__, Step3p7Config，涉及 `Step3p7VisionEncoderConfig, __init__, Step3p7Config`；`python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +18/-2 (20 lines); hunks: -900,6 +900,18 @@ def fused_experts_none_to_flashinfer_trtllm_fp4(; -924,6 +936,10 @@ def fused_experts_none_to_flashinfer_trtllm_fp4(; symbols: fused_experts_none_to_flashinfer_trtllm_fp4，涉及 `fused_experts_none_to_flashinfer_trtllm_fp4`；`python/sglang/srt/models/step3p5.py` modified +17/-1 (18 lines); hunks: -12,6 +12,7; -225,6 +226,8 @@ def forward_normal(; symbols: forward_normal, Step3p5ForCausalLM, get_model_config_for_expert_location, __init__，涉及 `forward_normal, Step3p5ForCausalLM, get_model_config_for_expert_location`。
- 代码 diff 细节:
  - `python/sglang/srt/models/step3p7.py` added +200/-0 (200 lines); hunks: -0,0 +1,200; symbols: Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__, _get_vision_model_output
  - `python/sglang/srt/configs/step3p7.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p7VisionEncoderConfig, __init__, Step3p7Config
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +18/-2 (20 lines); hunks: -900,6 +900,18 @@ def fused_experts_none_to_flashinfer_trtllm_fp4(; -924,6 +936,10 @@ def fused_experts_none_to_flashinfer_trtllm_fp4(; symbols: fused_experts_none_to_flashinfer_trtllm_fp4
  - `python/sglang/srt/models/step3p5.py` modified +17/-1 (18 lines); hunks: -12,6 +12,7; -225,6 +226,8 @@ def forward_normal(; symbols: forward_normal, Step3p5ForCausalLM, get_model_config_for_expert_location, __init__
  - `python/sglang/srt/configs/model_config.py` modified +12/-1 (13 lines); hunks: -452,6 +452,12 @@ def _config_draft_model(self):; -1557,6 +1563,7 @@ def is_generation_model(model_architectures: List[str], is...; symbols: _config_draft_model, is_generation_model, is_hybrid_swa_model, get_hybrid_layer_ids
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/step3p7.py
@@ -0,0 +1,200 @@
+from typing import Iterable, List, Optional, Tuple
+import torch
+from torch import nn
+from transformers.activations import ACT2FN
+from sglang.srt.configs.step3p7 import Step3p7Config
+from sglang.srt.layers.linear import ColumnParallelLinear
diff -- python/sglang/srt/configs/step3p7.py
@@ -0,0 +1,97 @@
+from typing import Optional, Union
+from transformers.configuration_utils import PretrainedConfig
+class Step3p7VisionEncoderConfig(PretrainedConfig):
+    model_type = "perception_encoder"
+    def __init__(
+        self,
diff -- python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
@@ -900,6 +900,18 @@ def fused_experts_none_to_flashinfer_trtllm_fp4(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/step3p7.py` added +200/-0; `python/sglang/srt/configs/step3p7.py` added +97/-0; `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +18/-2; `python/sglang/srt/models/step3p5.py` modified +17/-1; `python/sglang/srt/configs/model_config.py` modified +12/-1; `python/sglang/srt/multimodal/processors/step3_vl.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/step3p5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- 链接: https://github.com/sgl-project/sglang/pull/28567
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 184 个文件，+1865/-1727，可读 patch 8932 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_parallel(): a structured accessor for parallel-topology state」；模型线: Step 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Add get_parallel(): a structured accessor for parallel-topology state」；主要实现面是 `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention，涉及 `ApertusMLP, __init__, forward`；`python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales，涉及 `__init__, forward, load_kv_cache_scales`；`python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__，涉及 `_resolve_moe_input_pad_multiple, __init__`；`python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/apertus.py
@@ -1,687 +1,686 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Copyright 2025 The SwissAI Initiative
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
diff -- python/sglang/srt/models/solar.py
@@ -1,37 +1,14 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
-#
-# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,21 +28,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
