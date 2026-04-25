# sglang Step 3.5 Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.
- Diffusion model families have been removed from this history set and are no longer part of model optimization skills.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/sglang/srt/configs/step3p5.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084) |
| `python/sglang/srt/models/step3p5.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084), [#22076](https://github.com/sgl-project/sglang/pull/22076), [#22773](https://github.com/sgl-project/sglang/pull/22773) |
| `python/sglang/srt/models/step3p5_mtp.py` | [#18084](https://github.com/sgl-project/sglang/pull/18084) |
| `test/registered/8-gpu-models/test_step3p5_flash_chain_mtp.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 3
- Extra PRs preserved from existing docs: 4
- Total PRs in this document: 7
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

## Per-PR Diff Audit Cards

### PR #8583 - model: support Step3V

- Link: https://github.com/sgl-project/sglang/pull/8583
- Status/date: merged / 2025-07-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +2340/-23, 2530 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Step 3.5, this PR adds or enables a model support/runtime surface. Title: "model: support Step3V". The diff centers on `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py`. PR body context: ## Motivation This PR adds the support for Step3VModel. Co-authored-by: Qiaolin-Yu Co-authored-by: ispobock Co-authored-by: nnnobody-code Co-authored-by: jimpang ## Modification...
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
- Motivation: For Step 3.5, this PR adds or enables a model support/runtime surface. Title: "feat: Support DP Attention for step3_vl". The diff centers on `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`. PR body context: ## Motivation Support DP Attention for step3_vl ## Modifications In the implementation prior to `step3_vl`, DP Attention was already supported for the LLM. This update extends D...
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
- Motivation: For Step 3.5, this PR adds or enables a model support/runtime surface. Title: "[router] add step3 tool parser". The diff centers on `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs`. PR body context: ## Motivation step3 tool parser ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [x] Format your code according to the Format code with pre-commit...
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/step3p5.py`, `python/sglang/srt/models/step3p5.py`, `python/sglang/srt/models/step3p5_mtp.py`; associated commits `980d2936cd9a`
- Diff scope read: GitHub Pull Request files API returned 15 files, +1557/-12, 1711 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Step 3.5, this PR adds or enables a model support/runtime surface. Title: "add Step-3.5-Flash model support". The diff centers on `python/sglang/srt/models/step3p5.py`, `python/sglang/srt/models/step3p5_mtp.py`, `python/sglang/srt/configs/step3p5.py`. PR body context: ## Motivation add Step-3.5-Flash model support ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code...
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
- Motivation: For Step 3.5, this PR optimizes an inference path or backend selection. Title: "[Feature] implement the standard multi-layer MTP for step3p5". The diff centers on `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`. PR body context: ## Motivation The current SGL multi-layer MTP behavior can deviate from the standard Step3.5 Flash design, where hidden states should be propagated step-by-step across MTP layer...
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/step3p5.py`; associated commits `ef130312434c`
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-1, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Step 3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Tiny fix step3.5-flash launch crash". The diff centers on `python/sglang/srt/models/step3p5.py`. PR body context: ## Motivation Before this, launch step3.5-flash will just crash since step3.5-flash config doesn't have `pad_token_id` afaik, `padding_idx` is not used in the model file, so I s...
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
- Motivation: For Step 3.5, this PR adds or enables a model support/runtime surface. Title: "[Step3p5] Optimize allreduce in MoE layers". The diff centers on `python/sglang/srt/models/step3p5.py`. PR body context: ## Motivation ## Modifications - Defer o_proj and share_expert all-reduce, combine with MoE output into a single all-reduce per layer (was 3 separate all-reduces) - Enable allre...
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

## Gap-Closure Notes

- This version rejects title-only PR lists; every PR must include trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
