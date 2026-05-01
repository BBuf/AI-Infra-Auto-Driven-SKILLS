# sglang MiMo V2 Flash Model PR Optimization History

## Scope

- Rebuilt on: 2026-05-01
- Source baseline: `sgl-project/sglang` trace worktree commit `4197c55968`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2-Flash.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` | [#23851](https://github.com/sgl-project/sglang/pull/23851), [#23936](https://github.com/sgl-project/sglang/pull/23936), [#23945](https://github.com/sgl-project/sglang/pull/23945) |
| `docs_new/src/snippets/autoregressive/mimo-v2-flash-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` | [#23851](https://github.com/sgl-project/sglang/pull/23851), [#23936](https://github.com/sgl-project/sglang/pull/23936), [#23945](https://github.com/sgl-project/sglang/pull/23945) |
| `python/sglang/srt/function_call/mimo_detector.py` | [#15207](https://github.com/sgl-project/sglang/pull/15207) |
| `python/sglang/srt/models/mimo.py` | [#6059](https://github.com/sgl-project/sglang/pull/6059) |
| `python/sglang/srt/models/mimo_audio.py` | [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `python/sglang/srt/models/mimo_mtp.py` | [#6059](https://github.com/sgl-project/sglang/pull/6059), [#7370](https://github.com/sgl-project/sglang/pull/7370) |
| `python/sglang/srt/models/mimo_v2.py` | [#23808](https://github.com/sgl-project/sglang/pull/23808), [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `python/sglang/srt/models/mimo_v2_nextn.py` | [#23808](https://github.com/sgl-project/sglang/pull/23808), [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `python/sglang/srt/models/mimo_vl.py` | [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `python/sglang/srt/multimodal/processors/mimo_v2.py` | [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `test/registered/8-gpu-models/test_mimo_models.py` | [#23811](https://github.com/sgl-project/sglang/pull/23811), [#24118](https://github.com/sgl-project/sglang/pull/24118) |
| `test/registered/ascend/llm_models/test_npu_mimo_7b_rl.py` | no direct PR-number commit |
| `test/registered/ascend/vlm_models/test_npu_mimo_vl_7b_rl.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 9
- Extra PRs preserved from existing docs: 5
- Total PRs in this document: 14
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-05-22 | [#6059](https://github.com/sgl-project/sglang/pull/6059) | merged | Support XiaomiMiMo inference with mtp | `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py` |
| 2025-06-20 | [#7370](https://github.com/sgl-project/sglang/pull/7370) | merged | Clean unused import for mimo mtp model | `python/sglang/srt/models/mimo_mtp.py` |
| 2025-12-19 | [#15207](https://github.com/sgl-project/sglang/pull/15207) | merged | [Feature] Xiaomi `MiMo-V2-Flash` day0 support | `python/sglang/srt/function_call/mimo_detector.py` |
| 2025-12-20 | [#15464](https://github.com/sgl-project/sglang/pull/15464) | merged | Optimize MiMo-V2-Flash by flashinfer fused allreduce | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2025-12-25 | [#15488](https://github.com/sgl-project/sglang/pull/15488) | merged | [MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py` |
| 2026-02-01 | [#18051](https://github.com/sgl-project/sglang/pull/18051) | merged | [Fix] Remove no use code in MiMo-V2-Flash | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2026-02-02 | [#17634](https://github.com/sgl-project/sglang/pull/17634) | merged | [MiMoV2Flash] [feat]: support two batch overlap | `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py` |
| 2026-04-01 | [#21414](https://github.com/sgl-project/sglang/pull/21414) | merged | fix(MiMo-V2-Flash): add mimo reasoning parser | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-04-27 | [#23851](https://github.com/sgl-project/sglang/pull/23851) | merged | [Docs] add cookbook for MiMo-V2.5 family | `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` |
| 2026-04-28 | [#23808](https://github.com/sgl-project/sglang/pull/23808) | merged | [Feature] Xiaomi MiMo-V2.5-Pro day0 support | `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py` |
| 2026-04-28 | [#23945](https://github.com/sgl-project/sglang/pull/23945) | merged | docs: enable MiMo V2.5 MTP cookbook path | `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` |
| 2026-04-29 | [#23936](https://github.com/sgl-project/sglang/pull/23936) | merged | mimo v2.5 pro sglang-jax cookbook | `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`, `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` |
| 2026-04-30 | [#24118](https://github.com/sgl-project/sglang/pull/24118) | merged | fix: rename mimo spec threshold attr to num_accepted_drafts_thres | `test/registered/8-gpu-models/test_mimo_models.py` |
| 2026-04-30 | [#23811](https://github.com/sgl-project/sglang/pull/23811) | merged | [Feature] Xiaomi MiMo-V2.5 day0 support | `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_vl.py` |

## Per-PR Diff Audit Cards

### PR #6059 - Support XiaomiMiMo inference with mtp

- Link: https://github.com/sgl-project/sglang/pull/6059
- Status/date: merged / 2025-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo.py`, `python/sglang/srt/models/mimo_mtp.py`; associated commits `a6ae3af15e84`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +344/-6, 388 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support XiaomiMiMo inference with mtp"; model line: MiMo V2 Flash; category: docs/tests/CI; main diff: `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py`; technical summary: Covers "Support XiaomiMiMo inference with mtp"; the main implementation surface is `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mimo_mtp.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: MiMoMultiTokenPredictorLayer, __init__, forward, MiMoMTP, touching `MiMoMultiTokenPredictorLayer, __init__, forward`; `python/sglang/srt/models/mimo.py` renamed +0/-0 (0 lines).
- Code diff details:
  - `python/sglang/srt/models/mimo_mtp.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: MiMoMultiTokenPredictorLayer, __init__, forward, MiMoMTP
  - `python/sglang/srt/models/mimo.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mimo_mtp.py
@@ -0,0 +1,220 @@
+# Adapted from https://github.com/vllm-project/vllm/pull/17433/files  and deepseek_nextn.py
+from functools import partial
+from typing import Any, Dict, Iterable, Optional, Tuple
+import torch
+from torch import nn
+from transformers import PretrainedConfig
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mimo_mtp.py` added +220/-0; `python/sglang/srt/models/mimo.py` renamed +0/-0
- Risk and verification: The diff ships test coverage in `test/srt/models/test_mtp_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7370 - Clean unused import for mimo mtp model

- Link: https://github.com/sgl-project/sglang/pull/7370
- Status/date: merged / 2025-06-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo_mtp.py`; associated commits `dea8aa7ab8e8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-18, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Clean unused import for mimo mtp model"; model line: MiMo V2 Flash; category: model implementation change; main diff: `python/sglang/srt/models/mimo_mtp.py`; technical summary: Covers "Clean unused import for mimo mtp model"; the main implementation surface is `python/sglang/srt/models/mimo_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mimo_mtp.py` modified +2/-18 (20 lines); hunks: -7,33 +7,17; symbols: MiMoMultiTokenPredictorLayer, touching `MiMoMultiTokenPredictorLayer`.
- Code diff details:
  - `python/sglang/srt/models/mimo_mtp.py` modified +2/-18 (20 lines); hunks: -7,33 +7,17; symbols: MiMoMultiTokenPredictorLayer
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mimo_mtp.py
@@ -7,33 +7,17 @@
-from sglang.srt.distributed import (
-    get_tensor_model_parallel_rank,
-    get_tensor_model_parallel_world_size,
-    split_tensor_along_last_dim,
-    tensor_model_parallel_all_gather,
-)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mimo_mtp.py` modified +2/-18
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/mimo_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15207 - [Feature] Xiaomi `MiMo-V2-Flash` day0 support

- Link: https://github.com/sgl-project/sglang/pull/15207
- Status/date: merged / 2025-12-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/mimo_detector.py`; associated commits `160a06cab23f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 38 files, +5396/-169, 6509 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Xiaomi `MiMo-V2-Flash` day0 support"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `python/sglang/srt/function_call/mimo_detector.py`; technical summary: Covers "[Feature] Xiaomi `MiMo-V2-Flash` day0 support"; the main implementation surface is `python/sglang/srt/function_call/mimo_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/mimo_detector.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: _get_param_type, _convert_param_value, MiMoDetector, __init__, touching `_get_param_type, _convert_param_value, MiMoDetector`.
- Code diff details:
  - `python/sglang/srt/function_call/mimo_detector.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: _get_param_type, _convert_param_value, MiMoDetector, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/mimo_detector.py
@@ -0,0 +1,281 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/mimo_detector.py` added +281/-0
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15464 - Optimize MiMo-V2-Flash by flashinfer fused allreduce

- Link: https://github.com/sgl-project/sglang/pull/15464
- Status/date: merged / 2025-12-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +66/-10, 175 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize MiMo-V2-Flash by flashinfer fused allreduce"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `python/sglang/srt/models/mimo_v2_flash.py`; technical summary: Covers "Optimize MiMo-V2-Flash by flashinfer fused allreduce"; the main implementation surface is `python/sglang/srt/models/mimo_v2_flash.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mimo_v2_flash.py` modified +66/-10 (76 lines); hunks: -13,7 +13,7; -45,7 +45,11; symbols: __init__, forward, forward_normal, touching `__init__, forward, forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +66/-10 (76 lines); hunks: -13,7 +13,7; -45,7 +45,11; symbols: __init__, forward, forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mimo_v2_flash.py
@@ -13,7 +13,7 @@
-from typing import Any, Dict, Iterable, Optional, Tuple, Union
+from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
@@ -45,7 +45,11 @@
-from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend
+from sglang.srt.layers.moe import (
+    get_moe_a2a_backend,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mimo_v2_flash.py` modified +66/-10
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/mimo_v2_flash.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15488 - [MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg

- Link: https://github.com/sgl-project/sglang/pull/15488
- Status/date: merged / 2025-12-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +16/-16, 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg"; model line: MiMo V2 Flash; category: bug fix; main diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`; technical summary: Covers "[MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg"; the main implementation surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +10/-12 (22 lines); hunks: -334,7 +334,6 @@ def __init__(; -1582,10 +1581,9 @@ def profile_max_num_token(self, total_gpu_memory: int):; symbols: __init__, profile_max_num_token, handle_max_mamba_cache, set_num_token_hybrid, touching `__init__, profile_max_num_token, handle_max_mamba_cache`; `python/sglang/srt/server_args.py` modified +6/-4 (10 lines); hunks: -1203,11 +1203,11 @@ def _handle_model_specific_adjustments(self):; -2263,6 +2263,8 @@ def _handle_cache_compatibility(self):; symbols: _handle_model_specific_adjustments, _handle_cache_compatibility, _handle_deterministic_inference, touching `_handle_model_specific_adjustments, _handle_cache_compatibility, _handle_deterministic_inference`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +10/-12 (22 lines); hunks: -334,7 +334,6 @@ def __init__(; -1582,10 +1581,9 @@ def profile_max_num_token(self, total_gpu_memory: int):; symbols: __init__, profile_max_num_token, handle_max_mamba_cache, set_num_token_hybrid
  - `python/sglang/srt/server_args.py` modified +6/-4 (10 lines); hunks: -1203,11 +1203,11 @@ def _handle_model_specific_adjustments(self):; -2263,6 +2263,8 @@ def _handle_cache_compatibility(self):; symbols: _handle_model_specific_adjustments, _handle_cache_compatibility, _handle_deterministic_inference
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -334,7 +334,6 @@ def __init__(
-        self.kv_cache_memory = 0
@@ -1582,10 +1581,9 @@ def profile_max_num_token(self, total_gpu_memory: int):
-        self.kv_cache_memory = int(rest_memory * (1 << 30))
-        max_num_token = int(self.kv_cache_memory // cell_size)
-        return max_num_token
+        return int(rest_memory * (1 << 30)) // cell_size
diff -- python/sglang/srt/server_args.py
@@ -1203,11 +1203,11 @@ def _handle_model_specific_adjustments(self):
-            self.swa_full_tokens_ratio = 1.0
-            logger.warning(
-                "Reset swa_full_tokens_ratio to 1.0 for MiMoV2FlashForCausalLM model"
-            )
+                self.swa_full_tokens_ratio = 1.0
+                logger.warning(
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +10/-12; `python/sglang/srt/server_args.py` modified +6/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18051 - [Fix] Remove no use code in MiMo-V2-Flash

- Link: https://github.com/sgl-project/sglang/pull/18051
- Status/date: merged / 2026-02-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-20, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Remove no use code in MiMo-V2-Flash"; model line: MiMo V2 Flash; category: bug fix; main diff: `python/sglang/srt/models/mimo_v2_flash.py`; technical summary: Covers "[Fix] Remove no use code in MiMo-V2-Flash"; the main implementation surface is `python/sglang/srt/models/mimo_v2_flash.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mimo_v2_flash.py` modified +3/-20 (23 lines); hunks: -13,7 +13,7; -557,16 +557,10 @@ def forward(; symbols: forward, get_input_embedding, get_input_embeddings, set_eagle3_layers_to_capture, touching `forward, get_input_embedding, get_input_embeddings`.
- Code diff details:
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +3/-20 (23 lines); hunks: -13,7 +13,7; -557,16 +557,10 @@ def forward(; symbols: forward, get_input_embedding, get_input_embeddings, set_eagle3_layers_to_capture
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mimo_v2_flash.py
@@ -13,7 +13,7 @@
-from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
+from typing import Any, Dict, Iterable, Optional, Tuple, Union
@@ -557,16 +557,10 @@ def forward(
-        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
-        hidden_states, residual = (
-            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mimo_v2_flash.py` modified +3/-20
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/mimo_v2_flash.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17634 - [MiMoV2Flash] [feat]: support two batch overlap

- Link: https://github.com/sgl-project/sglang/pull/17634
- Status/date: merged / 2026-02-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +292/-8, 366 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MiMoV2Flash] [feat]: support two batch overlap"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py`; technical summary: Covers "[MiMoV2Flash] [feat]: support two batch overlap"; the main implementation surface is `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8 (216 lines); hunks: -19,18 +19,21; -66,7 +69,12; symbols: forward_deepep, op_gate, op_select_experts, op_dispatch_a, touching `forward_deepep, op_gate, op_select_experts`; `python/sglang/srt/batch_overlap/operations_strategy.py` modified +84/-0 (84 lines); hunks: -51,6 +51,15 @@ def init_new_tbo(; -209,3 +218,78 @@ def _compute_moe_qwen3_decode(layer):; symbols: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_mimov2_layer_operations_strategy_tbo, _compute_moe_mimov2_prefill, touching `init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_mimov2_layer_operations_strategy_tbo`.
- Code diff details:
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8 (216 lines); hunks: -19,18 +19,21; -66,7 +69,12; symbols: forward_deepep, op_gate, op_select_experts, op_dispatch_a
  - `python/sglang/srt/batch_overlap/operations_strategy.py` modified +84/-0 (84 lines); hunks: -51,6 +51,15 @@ def init_new_tbo(; -209,3 +218,78 @@ def _compute_moe_qwen3_decode(layer):; symbols: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_mimov2_layer_operations_strategy_tbo, _compute_moe_mimov2_prefill
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mimo_v2_flash.py
@@ -19,18 +19,21 @@
+from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
+from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
+    ScatterMode,
@@ -66,7 +69,12 @@
-from sglang.srt.utils import LazyValue, add_prefix, make_layers
+from sglang.srt.utils import (
diff -- python/sglang/srt/batch_overlap/operations_strategy.py
@@ -51,6 +51,15 @@ def init_new_tbo(
+        elif layer_name == "MiMoV2DecoderLayer":
+            return OperationsStrategy.concat(
+                [
+                    _compute_moe_mimov2_layer_operations_strategy_tbo(
+                        layer, forward_mode
+                    )
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8; `python/sglang/srt/batch_overlap/operations_strategy.py` modified +84/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/batch_overlap/operations_strategy.py`, `python/sglang/srt/models/mimo_v2_flash.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21414 - fix(MiMo-V2-Flash): add mimo reasoning parser

- Link: https://github.com/sgl-project/sglang/pull/21414
- Status/date: merged / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-0, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(MiMo-V2-Flash): add mimo reasoning parser"; model line: MiMo V2 Flash; category: bug fix; main diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`; technical summary: Covers "fix(MiMo-V2-Flash): add mimo reasoning parser"; the main implementation surface is `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +6/-0 (6 lines); hunks: -1268,6 +1268,12 @@ def _get_reasoning_from_request(self, request: ChatComple...; symbols: _get_reasoning_from_request, touching `_get_reasoning_from_request`; `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunks: -495,6 +495,7 @@ class ReasoningParser:; symbols: ReasoningParser, touching `ReasoningParser`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +6/-0 (6 lines); hunks: -1268,6 +1268,12 @@ def _get_reasoning_from_request(self, request: ChatComple...; symbols: _get_reasoning_from_request
  - `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunks: -495,6 +495,7 @@ class ReasoningParser:; symbols: ReasoningParser
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -1268,6 +1268,12 @@ def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:
+        if self.reasoning_parser in ["mimo"]:
+            # Models that require explicit enable thinking (enable_thinking=True)
+            return (
+                request.chat_template_kwargs is not None
+                and request.chat_template_kwargs.get("enable_thinking") is True
+            )
diff -- python/sglang/srt/parser/reasoning_parser.py
@@ -495,6 +495,7 @@ class ReasoningParser:
+        "mimo": Qwen3Detector,
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +6/-0; `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23851 - [Docs] add cookbook for MiMo-V2.5 family

- Link: https://github.com/sgl-project/sglang/pull/23851
- Status/date: merged / 2026-04-27
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; associated commits `f34222da1b22`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +1025/-1, 1042 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] add cookbook for MiMo-V2.5 family"; model line: MiMo V2 Flash; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; technical summary: Covers "[Docs] add cookbook for MiMo-V2.5 family"; the main implementation surface is `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` added +626/-0 (626 lines); hunks: -0,0 +1,626; `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` added +397/-0 (397 lines); hunks: -0,0 +1,397.
- Code diff details:
  - `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` added +626/-0 (626 lines); hunks: -0,0 +1,626
  - `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` added +397/-0 (397 lines); hunks: -0,0 +1,397
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx
@@ -0,0 +1,626 @@
+---
+title: MiMo-V2.5
+metatags:
+    description: "Deploy XiaomiMiMo MiMo-V2.5-Pro (1.02T MoE, text) and MiMo-V2.5 (310B MoE, multimodal) with SGLang — EAGLE speculative decoding, hybrid attention, and 1M-token c
+tag: NEW
+---
diff -- docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx
@@ -0,0 +1,397 @@
+export const MiMoV25Deployment = () => {
+  // MiMo-V2.5 family deployment matrix:
+  //   Variant × Hardware → slug, tp, multinode, blackwell
+  //
+  //   V2.5-Pro (1.02T / 42B active) — text-only:
+  //     H200  → tp=16, 2 nodes,     FP8 (Hopper: fa3 + DeepEP)
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` added +626/-0; `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` added +397/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23808 - [Feature] Xiaomi MiMo-V2.5-Pro day0 support

- Link: https://github.com/sgl-project/sglang/pull/23808
- Status/date: merged / 2026-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py`; associated commits `1a55646dcdf0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +80/-23, 280 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Xiaomi MiMo-V2.5-Pro day0 support"; model line: MiMo V2 Flash; category: model support/runtime entry; main diff: `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py`; technical summary: Covers "[Feature] Xiaomi MiMo-V2.5-Pro day0 support"; the main implementation surface is `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mimo_v2.py` renamed +27/-8 (35 lines); hunks: -76,7 +76,7; -178,7 +178,7 @@ class MiMoV2MoE(nn.Module):; symbols: MiMoV2MoE, __init__, forward, MiMoV2DecoderLayer, touching `MiMoV2MoE, __init__, forward`; `python/sglang/srt/models/mimo_v2_nextn.py` renamed +21/-6 (27 lines); hunks: -28,6 +28,7; -39,23 +40,23; symbols: MiMoV2MTPLayer, __init__, forward, MiMoV2MTP, touching `MiMoV2MTPLayer, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/mimo_v2.py` renamed +27/-8 (35 lines); hunks: -76,7 +76,7; -178,7 +178,7 @@ class MiMoV2MoE(nn.Module):; symbols: MiMoV2MoE, __init__, forward, MiMoV2DecoderLayer
  - `python/sglang/srt/models/mimo_v2_nextn.py` renamed +21/-6 (27 lines); hunks: -28,6 +28,7; -39,23 +40,23; symbols: MiMoV2MTPLayer, __init__, forward, MiMoV2MTP
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mimo_v2.py
@@ -76,7 +76,7 @@
-MiMoV2FlashConfig = None
+MiMoV2Config = None
@@ -178,7 +178,7 @@ class MiMoV2MoE(nn.Module):
-        config: MiMoV2FlashConfig,
+        config: MiMoV2Config,
@@ -562,7 +562,7 @@ def forward(
diff -- python/sglang/srt/models/mimo_v2_nextn.py
@@ -28,6 +28,7 @@
+    get_attention_tp_size,
@@ -39,23 +40,23 @@
-from sglang.srt.models.mimo_v2_flash import (
+from sglang.srt.models.mimo_v2 import (
-    MiMoV2FlashForCausalLM,
+    MiMoV2ForCausalLM,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mimo_v2.py` renamed +27/-8; `python/sglang/srt/models/mimo_v2_nextn.py` renamed +21/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23945 - docs: enable MiMo V2.5 MTP cookbook path

- Link: https://github.com/sgl-project/sglang/pull/23945
- Status/date: merged / 2026-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; associated commits `e458a9248fef`
- Diff scope read: GitHub Pull Request files API returned 2 files, +90/-88, 308 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: enable MiMo V2.5 MTP cookbook path"; model line: MiMo V2 Flash; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; technical summary: Covers "docs: enable MiMo V2.5 MTP cookbook path"; the main implementation surface is `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +84/-80 (164 lines); hunks: -43,7 +43,7 @@ tag: NEW; -84,9 +84,10 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressi...; `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +6/-8 (14 lines); hunks: -15,7 +15,7 @@ export const MiMoV25Deployment = () => {; -44,7 +44,7 @@ export const MiMoV25Deployment = () => {.
- Code diff details:
  - `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +84/-80 (164 lines); hunks: -43,7 +43,7 @@ tag: NEW; -84,9 +84,10 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressi...
  - `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +6/-8 (14 lines); hunks: -15,7 +15,7 @@ export const MiMoV25Deployment = () => {; -44,7 +44,7 @@ export const MiMoV25Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx
@@ -43,7 +43,7 @@ tag: NEW
-- **Multi-Token Prediction (MTP)**: 3-layer MTP module accelerates decoding (329M params on V2.5; V2.5-Pro supports EAGLE speculative decoding on top of MTP).
+- **Multi-Token Prediction (MTP)**: 3-layer MTP module accelerates decoding. Both variants support EAGLE speculative decoding with MTP weights.
@@ -84,9 +84,10 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressive/mimo-v25-deploym
-- The checkpoint has a TP=4-interleaved fused `qkv_proj`; attention-TP per DP group **must** be 4. So DP-attention is always required (`--dp = TP / 4`), and total GPUs must be a m
+- The checkpoint has a TP=4-interleaved fused `qkv_proj`; attention-TP per DP group **must** be 4. Use `--dp = TP / 4`; for TP > 4 this also requires DP-attention. Total GPUs must
+- EAGLE MTP uses the checkpoint's MTP weights. For H100/H200, enable `SGLANG_ENABLE_SPEC_V2=1`, `--speculative-algorithm EAGLE`, and `--enable-multi-layer-eagle`.
diff -- docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx
@@ -15,7 +15,7 @@ export const MiMoV25Deployment = () => {
-  //     EAGLE MTP — Pro only. Adds --speculative-* flags + SGLANG_ENABLE_SPEC_V2=1.
+  //     EAGLE MTP — adds --speculative-* flags + SGLANG_ENABLE_SPEC_V2=1.
@@ -44,7 +44,7 @@ export const MiMoV25Deployment = () => {
-        { id: "enabled",  label: "Enabled",  default: true,  subtitle: "Pro only" },
+        { id: "enabled",  label: "Enabled",  default: true,  subtitle: "EAGLE" },
@@ -68,8 +68,8 @@ export const MiMoV25Deployment = () => {
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +84/-80; `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +6/-8
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23936 - mimo v2.5 pro sglang-jax cookbook

- Link: https://github.com/sgl-project/sglang/pull/23936
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; associated commits `6c7b2421816c`
- Diff scope read: GitHub Pull Request files API returned 2 files, +114/-16, 188 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "mimo v2.5 pro sglang-jax cookbook"; model line: MiMo V2 Flash; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`, `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`; technical summary: Covers "mimo v2.5 pro sglang-jax cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`, `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +78/-16 (94 lines); hunks: -34,10 +34,12 @@ export const MiMoV25Deployment = () => {; -93,15 +95,19 @@ export const MiMoV25Deployment = () => {; `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +36/-0 (36 lines); hunks: -65,6 +65,8 @@ Refer to the [official SGLang installation guide](../../../doc...; -95,6 +97,40 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressi....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +78/-16 (94 lines); hunks: -34,10 +34,12 @@ export const MiMoV25Deployment = () => {; -93,15 +95,19 @@ export const MiMoV25Deployment = () => {
  - `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +36/-0 (36 lines); hunks: -65,6 +65,8 @@ Refer to the [official SGLang installation guide](../../../doc...; -95,6 +97,40 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressi...
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx
@@ -34,10 +34,12 @@ export const MiMoV25Deployment = () => {
-        { id: "h200",  label: "H200",  default: true  },
-        { id: "h100",  label: "H100",  default: false },
-        { id: "b200",  label: "B200",  default: false },
-        { id: "gb300", label: "GB300", default: false },
+        { id: "h200",     label: "H200",     default: true  },
+        { id: "h100",     label: "H100",     default: false },
diff -- docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx
@@ -65,6 +65,8 @@ Refer to the [official SGLang installation guide](../../../docs/get-started/inst
+**TPU (sgl-jax):** MiMo-V2.5-Pro can also be served on TPU via the JAX-based [sgl-jax](https://github.com/sgl-project/sglang-jax) runtime. The container image and `pip install` st
@@ -95,6 +97,40 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressive/mimo-v25-deploym
+### 3.3 TPU Deployment (MiMo-V2.5-Pro, sgl-jax)
+MiMo-V2.5-Pro can also be served on TPU via [sgl-jax](https://github.com/sgl-project/sglang-jax). The runtime is a separate JAX-based stack (`sgl_jax.launch_server`); pick **TPU v
+| TPU Type | Topology | Chips/Node | Nodes | Total Chips | JAX Devices/Chip | Total JAX Devices (= `--tp-size`) |
+| --- | --- | --- | --- | --- | --- | --- |
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +78/-16; `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +36/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #24118 - fix: rename mimo spec threshold attr to num_accepted_drafts_thres

- Link: https://github.com/sgl-project/sglang/pull/24118
- Status/date: merged / 2026-04-30
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_mimo_models.py`; associated commits `c54ada994bd5`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: rename mimo spec threshold attr to num_accepted_drafts_thres"; model line: MiMo V2 Flash; category: bug fix; main diff: `test/registered/8-gpu-models/test_mimo_models.py`; technical summary: Covers "fix: rename mimo spec threshold attr to num_accepted_drafts_thres"; the main implementation surface is `test/registered/8-gpu-models/test_mimo_models.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1 (2 lines); hunks: -45,7 +45,7 @@ class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, DefaultSe...; symbols: TestMiMoV2Flash, touching `TestMiMoV2Flash`.
- Code diff details:
  - `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1 (2 lines); hunks: -45,7 +45,7 @@ class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, DefaultSe...; symbols: TestMiMoV2Flash
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_mimo_models.py
@@ -45,7 +45,7 @@ class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, DefaultServerBase):
-    accept_length_thres = 3.2
+    num_accepted_drafts_thres = 3.2
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_mimo_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23811 - [Feature] Xiaomi MiMo-V2.5 day0 support

- Link: https://github.com/sgl-project/sglang/pull/23811
- Status/date: merged / 2026-04-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py`, `python/sglang/srt/models/mimo_vl.py`, `python/sglang/srt/multimodal/processors/mimo_v2.py` and 6 files; associated commits `651af06a0b5e`
- Diff scope read: GitHub Pull Request files API returned 16 files, +4369/-87, 4885 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Xiaomi MiMo-V2.5 day0 support"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_vl.py`; technical summary: Covers "[Feature] Xiaomi MiMo-V2.5 day0 support"; the main implementation surface is `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/mimo_v2.py` added +2039/-0 (2039 lines); hunks: -0,0 +1,2039; symbols: ImageInput, __post_init__, VideoInput, AudioInput, touching `ImageInput, __post_init__, VideoInput`; `python/sglang/srt/models/mimo_audio.py` added +1350/-0 (1350 lines); hunks: -0,0 +1,1350; symbols: flash_attn_varlen_func, _compute_default_rope_parameters, _dynamic_rope_update, longrope_frequency_update, touching `flash_attn_varlen_func, _compute_default_rope_parameters, _dynamic_rope_update`; `python/sglang/srt/models/mimo_vl.py` added +507/-0 (507 lines); hunks: -0,0 +1,507; symbols: MiMoVLVisionConfig, __init__, MiMoVisionPatchEmbed, sync_proj_weight_linear_format, touching `MiMoVLVisionConfig, __init__, MiMoVisionPatchEmbed`; `python/sglang/srt/models/mimo_v2.py` modified +222/-13 (235 lines); hunks: -13,13 +13,14; -63,11 +64,18; symbols: load_mimo_v2_qkv_proj_weight, MiMoV2MLP, __init__, routed_experts_weights_of_layer, touching `load_mimo_v2_qkv_proj_weight, MiMoV2MLP, __init__`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/mimo_v2.py` added +2039/-0 (2039 lines); hunks: -0,0 +1,2039; symbols: ImageInput, __post_init__, VideoInput, AudioInput
  - `python/sglang/srt/models/mimo_audio.py` added +1350/-0 (1350 lines); hunks: -0,0 +1,1350; symbols: flash_attn_varlen_func, _compute_default_rope_parameters, _dynamic_rope_update, longrope_frequency_update
  - `python/sglang/srt/models/mimo_vl.py` added +507/-0 (507 lines); hunks: -0,0 +1,507; symbols: MiMoVLVisionConfig, __init__, MiMoVisionPatchEmbed, sync_proj_weight_linear_format
  - `python/sglang/srt/models/mimo_v2.py` modified +222/-13 (235 lines); hunks: -13,13 +13,14; -63,11 +64,18; symbols: load_mimo_v2_qkv_proj_weight, MiMoV2MLP, __init__, routed_experts_weights_of_layer
  - `python/sglang/srt/models/mimo_v2_nextn.py` modified +12/-7 (19 lines); hunks: -19,6 +19,7; -28,7 +29,6; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/mimo_v2.py
@@ -0,0 +1,2039 @@
+"""MiMoV2 multimodal processor -- protocol, utilities, and processor."""
+import asyncio
+import base64
+import copy
+import io
+import json
diff -- python/sglang/srt/models/mimo_audio.py
@@ -0,0 +1,1350 @@
+"""MiMo audio: tokenizer, encoding utilities, and audio encoder."""
+# Audio tokenizer adapted from https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer.git
+import logging
+import math
+import os
+import typing as tp
diff -- python/sglang/srt/models/mimo_vl.py
@@ -0,0 +1,507 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/mimo_v2.py` added +2039/-0; `python/sglang/srt/models/mimo_audio.py` added +1350/-0; `python/sglang/srt/models/mimo_vl.py` added +507/-0; `python/sglang/srt/models/mimo_v2.py` modified +222/-13; `python/sglang/srt/models/mimo_v2_nextn.py` modified +12/-7
  - tests: `test/registered/8-gpu-models/test_mimo_models.py` modified +38/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/server_fixtures/mmmu_fixture.py`, `test/registered/8-gpu-models/test_mimo_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
