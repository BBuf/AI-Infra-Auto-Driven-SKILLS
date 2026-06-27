# sglang Step 3.5 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/sglang/srt/configs/step3p5.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084) |
| `python/sglang/srt/models/step3p5.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084), [#22076](https://github.com/sgl-project/sglang/pull/22076), [#22773](https://github.com/sgl-project/sglang/pull/22773) |
| `python/sglang/srt/models/step3p5_mtp.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084) |
| `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 3
- Extra PRs preserved from existing docs: 12
- Total PRs in this document: 15
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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

## Per-PR Diff Audit Cards

### PR #8583 - model: support Step3V

- Link: https://github.com/sgl-project/sglang/pull/8583
- Status/date: merged / 2025-07-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +2340/-23, 2530 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: support Step3V"; model line: Step 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py`; technical summary: Covers "model: support Step3V"; the main implementation surface is `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/step3_vl.py` added +994/-0 (994 lines); hunks: -0,0 +1,994; symbols: Step3TextMLP, __init__, forward, Step3TextMoEMLP, touching `Step3TextMLP, __init__, forward`; `python/sglang/srt/multimodal/processors/step3_vl.py` added +515/-0 (515 lines); hunks: -0,0 +1,515; symbols: GPUToTensor, forward, Step3VisionProcessor, __init__, touching `GPUToTensor, forward, Step3VisionProcessor`; `python/sglang/srt/function_call/step3_detector.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: get_argument_type, parse_arguments, Step3Detector, __init__, touching `get_argument_type, parse_arguments, Step3Detector`; `python/sglang/srt/configs/step3_vl.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Step3VisionEncoderConfig, __init__, Step3TextConfig, Step3VLConfig, touching `Step3VisionEncoderConfig, __init__, Step3TextConfig`.
- Code diff details:
  - `python/sglang/srt/models/step3_vl.py` added +994/-0 (994 lines); hunks: -0,0 +1,994; symbols: Step3TextMLP, __init__, forward, Step3TextMoEMLP
  - `python/sglang/srt/multimodal/processors/step3_vl.py` added +515/-0 (515 lines); hunks: -0,0 +1,515; symbols: GPUToTensor, forward, Step3VisionProcessor, __init__
  - `python/sglang/srt/function_call/step3_detector.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: get_argument_type, parse_arguments, Step3Detector, __init__
  - `python/sglang/srt/configs/step3_vl.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Step3VisionEncoderConfig, __init__, Step3TextConfig, Step3VLConfig
  - `test/srt/test_reasoning_parser.py` modified +112/-0 (112 lines); hunks: -493,5 +493,117 @@ def test_qwen3_thinking_streaming_scenario(self):; symbols: test_qwen3_thinking_streaming_scenario, TestBufferLossBugFix, test_partial_end_tag_buffer_loss_bug, test_partial_start_tag_buffer_preservation
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/step3_vl.py` added +994/-0; `python/sglang/srt/multimodal/processors/step3_vl.py` added +515/-0; `python/sglang/srt/function_call/step3_detector.py` added +436/-0; `python/sglang/srt/configs/step3_vl.py` added +172/-0; `python/sglang/srt/configs/__init__.py` modified +8/-0; `python/sglang/srt/configs/model_config.py` modified +3/-0
  - tests: `test/srt/test_reasoning_parser.py` modified +112/-0
- Risk and verification: The diff ships test coverage in `test/srt/test_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8699 - feat: Support DP Attention for step3_vl

- Link: https://github.com/sgl-project/sglang/pull/8699
- Status/date: merged / 2025-08-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +25/-6, 107 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: Support DP Attention for step3_vl"; model line: Step 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`; technical summary: Covers "feat: Support DP Attention for step3_vl"; the main implementation surface is `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/vision.py` modified +13/-5 (18 lines); hunks: -11,6 +11,7; -365,19 +366,20 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/step3_vl.py` modified +9/-0 (9 lines); hunks: -531,11 +531,18 @@ def __init__(; -544,6 +551,8 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/multimodal/processors/step3_vl.py` modified +3/-1 (4 lines); hunks: -8,7 +8,7; -276,6 +276,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/vision.py` modified +13/-5 (18 lines); hunks: -11,6 +11,7; -365,19 +366,20 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/step3_vl.py` modified +9/-0 (9 lines); hunks: -531,11 +531,18 @@ def __init__(; -544,6 +551,8 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/multimodal/processors/step3_vl.py` modified +3/-1 (4 lines); hunks: -8,7 +8,7; -276,6 +276,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/vision.py` modified +13/-5; `python/sglang/srt/models/step3_vl.py` modified +9/-0; `python/sglang/srt/multimodal/processors/step3_vl.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9695 - [router] add step3 tool parser

- Link: https://github.com/sgl-project/sglang/pull/9695
- Status/date: merged / 2025-08-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +600/-2, 634 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[router] add step3 tool parser"; model line: Step 3.5; category: docs/tests/CI; main diff: `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs`; technical summary: Covers "[router] add step3 tool parser"; the main implementation surface is `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `sgl-router/src/tool_parser/parsers/step3_parser.rs` added +348/-0 (348 lines); hunks: -0,0 +1,348; `sgl-router/tests/tool_parser_step3.rs` added +245/-0 (245 lines); hunks: -0,0 +1,245; `sgl-router/src/tool_parser/registry.rs` modified +3/-1 (4 lines); hunks: -1,5 +1,5; -113,6 +113,8 @@ impl ParserRegistry {; `sgl-router/src/tool_parser/parsers/mod.rs` modified +3/-0 (3 lines); hunks: -9,12 +9,15 @@ pub mod llama_parser;.
- Code diff details:
  - `sgl-router/src/tool_parser/parsers/step3_parser.rs` added +348/-0 (348 lines); hunks: -0,0 +1,348
  - `sgl-router/tests/tool_parser_step3.rs` added +245/-0 (245 lines); hunks: -0,0 +1,245
  - `sgl-router/src/tool_parser/registry.rs` modified +3/-1 (4 lines); hunks: -1,5 +1,5; -113,6 +113,8 @@ impl ParserRegistry {
  - `sgl-router/src/tool_parser/parsers/mod.rs` modified +3/-0 (3 lines); hunks: -9,12 +9,15 @@ pub mod llama_parser;
  - `sgl-router/src/tool_parser/mod.rs` modified +1/-1 (2 lines); hunks: -25,5 +25,5 @@ pub use types::{FunctionCall, PartialToolCall, StreamResult, T...
- Key code excerpts:

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

- Reviewed files:
  - runtime: `sgl-router/src/tool_parser/parsers/step3_parser.rs` added +348/-0; `sgl-router/src/tool_parser/registry.rs` modified +3/-1; `sgl-router/src/tool_parser/parsers/mod.rs` modified +3/-0; `sgl-router/src/tool_parser/mod.rs` modified +1/-1
  - tests: `sgl-router/tests/tool_parser_step3.rs` added +245/-0
- Risk and verification: The diff ships test coverage in `sgl-router/tests/tool_parser_step3.rs`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18084 - add Step-3.5-Flash model support

- Link: https://github.com/sgl-project/sglang/pull/18084
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/step3p5.py`, `python/sglang/srt/models/step3p5.py`, `python/sglang/srt/models/step3p5_mtp.py`; associated commits `980d2936cd9a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +1557/-12, 1711 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add Step-3.5-Flash model support"; model line: Step 3.5; category: performance/backend optimization; main diff: `python/sglang/srt/models/step3p5.py`, `python/sglang/srt/models/step3p5_mtp.py`, `python/sglang/srt/configs/step3p5.py`; technical summary: Covers "add Step-3.5-Flash model support"; the main implementation surface is `python/sglang/srt/models/step3p5.py`, `python/sglang/srt/models/step3p5_mtp.py`, `python/sglang/srt/configs/step3p5.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/step3p5.py` added +1037/-0 (1037 lines); hunks: -0,0 +1,1037; symbols: Step3p5MLP, __init__, forward, Step3p5MoEMLP, touching `Step3p5MLP, __init__, forward`; `python/sglang/srt/models/step3p5_mtp.py` added +336/-0 (336 lines); hunks: -0,0 +1,336; symbols: get_spec_layer_idx_from_weight_name, SharedHead, __init__, forward, touching `get_spec_layer_idx_from_weight_name, SharedHead, __init__`; `python/sglang/srt/configs/step3p5.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p5Config, __init__, touching `Step3p5Config, __init__`.
- Code diff details:
  - `python/sglang/srt/models/step3p5.py` added +1037/-0 (1037 lines); hunks: -0,0 +1,1037; symbols: Step3p5MLP, __init__, forward, Step3p5MoEMLP
  - `python/sglang/srt/models/step3p5_mtp.py` added +336/-0 (336 lines); hunks: -0,0 +1,336; symbols: get_spec_layer_idx_from_weight_name, SharedHead, __init__, forward
  - `python/sglang/srt/configs/step3p5.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p5Config, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/step3p5.py` added +1037/-0; `python/sglang/srt/models/step3p5_mtp.py` added +336/-0; `python/sglang/srt/configs/step3p5.py` added +97/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/step3p5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18564 - [Feature] implement the standard multi-layer MTP for step3p5

- Link: https://github.com/sgl-project/sglang/pull/18564
- Status/date: merged / 2026-03-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +31/-2, 61 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] implement the standard multi-layer MTP for step3p5"; model line: Step 3.5; category: performance/backend optimization; main diff: `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`; technical summary: Covers "[Feature] implement the standard multi-layer MTP for step3p5"; the main implementation surface is `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +21/-2 (23 lines); hunks: -127,6 +127,11 @@ def __init__(; -382,6 +387,15 @@ def _draft_extend_for_prefill(; symbols: __init__, _draft_extend_for_prefill, forward_batch_generation, touching `__init__, _draft_extend_for_prefill, forward_batch_generation`; `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: -387,6 +387,16 @@ def run_once():; symbols: run_once, touching `run_once`.
- Code diff details:
  - `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +21/-2 (23 lines); hunks: -127,6 +127,11 @@ def __init__(; -382,6 +387,15 @@ def _draft_extend_for_prefill(; symbols: __init__, _draft_extend_for_prefill, forward_batch_generation
  - `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: -387,6 +387,16 @@ def run_once():; symbols: run_once
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +21/-2; `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` modified +10/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`, `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22076 - Tiny fix step3.5-flash launch crash

- Link: https://github.com/sgl-project/sglang/pull/22076
- Status/date: merged / 2026-04-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/step3p5.py`; associated commits `ef130312434c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-1, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Tiny fix step3.5-flash launch crash"; model line: Step 3.5; category: bug fix; main diff: `python/sglang/srt/models/step3p5.py`; technical summary: Covers "Tiny fix step3.5-flash launch crash"; the main implementation surface is `python/sglang/srt/models/step3p5.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/step3p5.py` modified +0/-1 (1 lines); hunks: -667,7 +667,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/step3p5.py` modified +0/-1 (1 lines); hunks: -667,7 +667,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/step3p5.py
@@ -667,7 +667,6 @@ def __init__(
-        self.padding_idx = config.pad_token_id
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/step3p5.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/step3p5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22773 - [Step3p5] Optimize allreduce in MoE layers

- Link: https://github.com/sgl-project/sglang/pull/22773
- Status/date: merged / 2026-04-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/step3p5.py`; associated commits `b8794baa6d61`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +59/-57, 211 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Step3p5] Optimize allreduce in MoE layers"; model line: Step 3.5; category: performance/backend optimization; main diff: `python/sglang/srt/models/step3p5.py`; technical summary: Covers "[Step3p5] Optimize allreduce in MoE layers"; the main implementation surface is `python/sglang/srt/models/step3p5.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/step3p5.py` modified +59/-57 (116 lines); hunks: -1,5 +1,3; -57,7 +55,6; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/step3p5.py` modified +59/-57 (116 lines); hunks: -1,5 +1,3; -57,7 +55,6; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/step3p5.py` modified +59/-57
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/step3p5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24105 - relax the threshold in test_step3p5_flash_chain_mtp

- Link: https://github.com/sgl-project/sglang/pull/24105
- Status/date: merged / 2026-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "relax the threshold in test_step3p5_flash_chain_mtp"; model line: Step 3.5; category: performance/backend optimization; main diff: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`; technical summary: Covers "relax the threshold in test_step3p5_flash_chain_mtp"; the main implementation surface is `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def test_gsm8k(self):; symbols: test_gsm8k, touching `test_gsm8k`.
- Code diff details:
  - `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def test_gsm8k(self):; symbols: test_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py
@@ -97,7 +97,7 @@ def test_gsm8k(self):
-            self.assertGreater(metrics["score"], 0.84)
+            self.assertGreater(metrics["score"], 0.83)
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24192 - [spec decoding] add tests for chain-style multi layer eagle + return_logprob

- Link: https://github.com/sgl-project/sglang/pull/24192
- Status/date: merged / 2026-05-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +136/-0, 150 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[spec decoding] add tests for chain-style multi layer eagle + return_logprob"; model line: Step 3.5; category: performance/backend optimization; main diff: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`; technical summary: Covers "[spec decoding] add tests for chain-style multi layer eagle + return_logprob"; the main implementation surface is `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +136/-0 (136 lines); hunks: -1,6 +1,7; -100,6 +101,141 @@ def test_gsm8k(self):; symbols: test_gsm8k, test_logprob_spec_v2_match, touching `test_gsm8k, test_logprob_spec_v2_match`.
- Code diff details:
  - `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +136/-0 (136 lines); hunks: -1,6 +1,7; -100,6 +101,141 @@ def test_gsm8k(self):; symbols: test_gsm8k, test_logprob_spec_v2_match
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` modified +136/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25197 - ci: decouple stage and runner for cuda registry

- Link: https://github.com/sgl-project/sglang/pull/25197
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 261 files, +388/-293, 2625 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci: decouple stage and runner for cuda registry"; model line: Step 3.5; category: performance/backend optimization; main diff: `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py`; technical summary: Covers "ci: decouple stage and runner for cuda registry"; the main implementation surface is `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1 (3 lines); hunks: -19,7 +19,8; `test/registered/models/test_dummy_grok_models.py` modified +2/-1 (3 lines); hunks: -5,7 +5,8; `test/registered/models/test_ministral3_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8; `test/registered/models/test_ministral4_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8.
- Code diff details:
  - `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1 (3 lines); hunks: -19,7 +19,8
  - `test/registered/models/test_dummy_grok_models.py` modified +2/-1 (3 lines); hunks: -5,7 +5,8
  - `test/registered/models/test_ministral3_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8
  - `test/registered/models/test_ministral4_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8
  - `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +2/-1 (3 lines); hunks: -6,7 +6,8
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1; `test/registered/models/test_dummy_grok_models.py` modified +2/-1; `test/registered/models/test_ministral3_models.py` modified +2/-1; `test/registered/models/test_ministral4_models.py` modified +2/-1; `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +2/-1; `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/ci/ci_register.py`, `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py`, `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25236 - ci: H200 conditional split + dsv4 est_time recalibration (h200 partition 6→2)

- Link: https://github.com/sgl-project/sglang/pull/25236
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +13/-13, 117 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci: H200 conditional split + dsv4 est_time recalibration (h200 partition 6→2)"; model line: Step 3.5; category: docs/tests/CI; main diff: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`; technical summary: Covers "ci: H200 conditional split + dsv4 est_time recalibration (h200 partition 6→2)"; the main implementation surface is `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7; `test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7; `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7; symbols: TestMiMoV2Flash, touching `TestMiMoV2Flash`.
- Code diff details:
  - `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
  - `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7
  - `test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7
  - `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7; symbols: TestMiMoV2Flash
  - `test/registered/8-gpu-models/test_minimax_m25_basic.py` modified +1/-1 (2 lines); hunks: -14,7 +14,7
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` modified +1/-1; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` modified +1/-1; `test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +1/-1; `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1; `test/registered/8-gpu-models/test_minimax_m25_basic.py` modified +1/-1; `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`, `test/registered/8-gpu-models/test_mimo_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24725 - ci: tag-gated nightly migration — foundation + 40 whole-file moves

- Link: https://github.com/sgl-project/sglang/pull/24725
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 78 files, +2263/-2140, 4964 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci: tag-gated nightly migration — foundation + 40 whole-file moves"; model line: Step 3.5; category: docs/tests/CI; main diff: `test/registered/models/test_ministral4_models.py`, `test/registered/models/test_compressed_tensors_models.py`, `test/registered/models/test_generation_models.py`; technical summary: Covers "ci: tag-gated nightly migration — foundation + 40 whole-file moves"; the main implementation surface is `test/registered/models/test_ministral4_models.py`, `test/registered/models/test_compressed_tensors_models.py`, `test/registered/models/test_generation_models.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models/test_ministral4_models.py` modified +1/-5 (6 lines); hunks: -6,11 +6,7; `test/registered/models/test_compressed_tensors_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7; `test/registered/models/test_generation_models.py` modified +1/-1 (2 lines); hunks: -1,7 +1,7; `test/registered/models/test_vlm_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7.
- Code diff details:
  - `test/registered/models/test_ministral4_models.py` modified +1/-5 (6 lines); hunks: -6,11 +6,7
  - `test/registered/models/test_compressed_tensors_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
  - `test/registered/models/test_generation_models.py` modified +1/-1 (2 lines); hunks: -1,7 +1,7
  - `test/registered/models/test_vlm_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
  - `test/manual/openai_server/function_call/test_tool_choice.py` renamed +0/-0 (0 lines)
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/models/test_ministral4_models.py` modified +1/-5; `test/registered/models/test_compressed_tensors_models.py` modified +1/-1; `test/registered/models/test_generation_models.py` modified +1/-1; `test/registered/models/test_vlm_models.py` modified +1/-1; `test/manual/openai_server/function_call/test_tool_choice.py` renamed +0/-0; `test/registered/sessions/test_streaming_session.py` modified +62/-1072
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/streaming_session_kit.py`, `python/sglang/test/server_fixtures/hybrid_attn_backend_fixture.py`, `python/sglang/test/server_fixtures/ngram_fixture.py`, `python/sglang/test/server_fixtures/pcg_spec_fixture.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26610 - test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)

- Link: https://github.com/sgl-project/sglang/pull/26610
- Status/date: merged / 2026-05-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +611/-816, 1566 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)"; model line: Step 3.5; category: performance/backend optimization; main diff: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`; technical summary: Covers "test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)"; the main implementation surface is `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212 (212 lines); hunks: -1,212 +0,0; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass`; `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133 (134 lines); hunks: -1,25 +1,12; -28,18 +15,8 @@ def _random_suffixes(n, length, seed):; symbols: _random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching, TestUnifiedFullRadixCache, touching `_random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching`; `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78 (111 lines); hunks: -1,28 +1,20; -31,75 +23,38 @@ class TestStep3p5FlashChainMTP(CustomTestCase):; symbols: TestStep3p5FlashChainMTP, setUpClass, tearDownClass, touching `TestStep3p5FlashChainMTP, setUpClass, tearDownClass`; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110 (110 lines); hunks: -1,110 +0,0; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV3MTP, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212 (212 lines); hunks: -1,212 +0,0; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133 (134 lines); hunks: -1,25 +1,12; -28,18 +15,8 @@ def _random_suffixes(n, length, seed):; symbols: _random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching, TestUnifiedFullRadixCache
  - `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78 (111 lines); hunks: -1,28 +1,20; -31,75 +23,38 @@ class TestStep3p5FlashChainMTP(CustomTestCase):; symbols: TestStep3p5FlashChainMTP, setUpClass, tearDownClass
  - `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110 (110 lines); hunks: -1,110 +0,0; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-105 (105 lines); hunks: -1,105 +0,0; symbols: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212; `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133; `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110; `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-105; `test/registered/quant/test_deepseek_v3_fp4_4gpu.py` removed +0/-80
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/manual/core/test_dsv4_hicache_swa_translation_cache.py`, `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26565 - model: Step-3.7-Flash Support

- Link: https://github.com/sgl-project/sglang/pull/26565
- Status/date: merged / 2026-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +1094/-7, 1284 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: Step-3.7-Flash Support"; model line: Step 3.5; category: performance/backend optimization; main diff: `python/sglang/srt/models/step3p7.py`, `python/sglang/srt/configs/step3p7.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`; technical summary: Covers "model: Step-3.7-Flash Support"; the main implementation surface is `python/sglang/srt/models/step3p7.py`, `python/sglang/srt/configs/step3p7.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/step3p7.py` added +200/-0 (200 lines); hunks: -0,0 +1,200; symbols: Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__, _get_vision_model_output, touching `Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__`; `python/sglang/srt/configs/step3p7.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p7VisionEncoderConfig, __init__, Step3p7Config, touching `Step3p7VisionEncoderConfig, __init__, Step3p7Config`; `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +18/-2 (20 lines); hunks: -900,6 +900,18 @@ def fused_experts_none_to_flashinfer_trtllm_fp4(; -924,6 +936,10 @@ def fused_experts_none_to_flashinfer_trtllm_fp4(; symbols: fused_experts_none_to_flashinfer_trtllm_fp4, touching `fused_experts_none_to_flashinfer_trtllm_fp4`; `python/sglang/srt/models/step3p5.py` modified +17/-1 (18 lines); hunks: -12,6 +12,7; -225,6 +226,8 @@ def forward_normal(; symbols: forward_normal, Step3p5ForCausalLM, get_model_config_for_expert_location, __init__, touching `forward_normal, Step3p5ForCausalLM, get_model_config_for_expert_location`.
- Code diff details:
  - `python/sglang/srt/models/step3p7.py` added +200/-0 (200 lines); hunks: -0,0 +1,200; symbols: Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__, _get_vision_model_output
  - `python/sglang/srt/configs/step3p7.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p7VisionEncoderConfig, __init__, Step3p7Config
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +18/-2 (20 lines); hunks: -900,6 +900,18 @@ def fused_experts_none_to_flashinfer_trtllm_fp4(; -924,6 +936,10 @@ def fused_experts_none_to_flashinfer_trtllm_fp4(; symbols: fused_experts_none_to_flashinfer_trtllm_fp4
  - `python/sglang/srt/models/step3p5.py` modified +17/-1 (18 lines); hunks: -12,6 +12,7; -225,6 +226,8 @@ def forward_normal(; symbols: forward_normal, Step3p5ForCausalLM, get_model_config_for_expert_location, __init__
  - `python/sglang/srt/configs/model_config.py` modified +12/-1 (13 lines); hunks: -452,6 +452,12 @@ def _config_draft_model(self):; -1557,6 +1563,7 @@ def is_generation_model(model_architectures: List[str], is...; symbols: _config_draft_model, is_generation_model, is_hybrid_swa_model, get_hybrid_layer_ids
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/step3p7.py` added +200/-0; `python/sglang/srt/configs/step3p7.py` added +97/-0; `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +18/-2; `python/sglang/srt/models/step3p5.py` modified +17/-1; `python/sglang/srt/configs/model_config.py` modified +12/-1; `python/sglang/srt/multimodal/processors/step3_vl.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/step3p5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- Link: https://github.com/sgl-project/sglang/pull/28567
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 184 files, +1865/-1727, 8932 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_parallel(): a structured accessor for parallel-topology state"; model line: Step 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add get_parallel(): a structured accessor for parallel-topology state"; the main implementation surface is `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention, touching `ApertusMLP, __init__, forward`; `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales, touching `__init__, forward, load_kv_cache_scales`; `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__, touching `_resolve_moe_input_pad_multiple, __init__`; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
