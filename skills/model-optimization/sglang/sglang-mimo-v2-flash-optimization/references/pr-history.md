# sglang MiMo V2 Flash PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 2
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2-Flash.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/mimo-v2-flash-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/function_call/mimo_detector.py` | [#15207](https://github.com/sgl-project/sglang/pull/15207) |
| `python/sglang/srt/models/mimo.py` | [#6059](https://github.com/sgl-project/sglang/pull/6059) |
| `python/sglang/srt/models/mimo_mtp.py` | [#6059](https://github.com/sgl-project/sglang/pull/6059), [#7370](https://github.com/sgl-project/sglang/pull/7370) |
| `python/sglang/srt/models/mimo_v2_flash.py` | [#15207](https://github.com/sgl-project/sglang/pull/15207), [#15464](https://github.com/sgl-project/sglang/pull/15464), [#17634](https://github.com/sgl-project/sglang/pull/17634), [#18051](https://github.com/sgl-project/sglang/pull/18051) |
| `python/sglang/srt/models/mimo_v2_flash_nextn.py` | [#15207](https://github.com/sgl-project/sglang/pull/15207) |
| `test/registered/8-gpu-models/test_mimo_models.py` | no direct PR-number commit |
| `test/registered/ascend/llm_models/test_npu_mimo_7b_rl.py` | no direct PR-number commit |
| `test/registered/ascend/vlm_models/test_npu_mimo_vl_7b_rl.py` | no direct PR-number commit |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-05-22 | [#6059](https://github.com/sgl-project/sglang/pull/6059) | merged | Support XiaomiMiMo inference with mtp | `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py` |
| 2025-06-20 | [#7370](https://github.com/sgl-project/sglang/pull/7370) | merged | Clean unused import for mimo mtp model | `python/sglang/srt/models/mimo_mtp.py` |
| 2025-12-19 | [#15207](https://github.com/sgl-project/sglang/pull/15207) | merged | [Feature] Xiaomi `MiMo-V2-Flash` day0 support | `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/models/mimo_v2_flash_nextn.py`, `python/sglang/srt/function_call/mimo_detector.py` |
| 2025-12-20 | [#15464](https://github.com/sgl-project/sglang/pull/15464) | merged | Optimize MiMo-V2-Flash by flashinfer fused allreduce | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2025-12-25 | [#15488](https://github.com/sgl-project/sglang/pull/15488) | merged | [MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py` |
| 2026-02-01 | [#18051](https://github.com/sgl-project/sglang/pull/18051) | merged | [Fix] Remove no use code in MiMo-V2-Flash | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2026-02-02 | [#17634](https://github.com/sgl-project/sglang/pull/17634) | merged | [MiMoV2Flash] [feat]: support two batch overlap | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2026-04-01 | [#21414](https://github.com/sgl-project/sglang/pull/21414) | merged | fix(MiMo-V2-Flash): add mimo reasoning parser | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py` |

## Per-PR Diff Audit Cards

### PR #6059 - Support XiaomiMiMo inference with mtp

- Link: https://github.com/sgl-project/sglang/pull/6059
- Status/date: merged / 2025-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo.py`, `python/sglang/srt/models/mimo_mtp.py`; associated commits `a6ae3af15e84`
- Diff scope read: GitHub Pull Request files API returned 6 files, +344/-6, 388 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiMo V2 Flash, this PR adds or enables a model support/runtime surface. Title: "Support XiaomiMiMo inference with mtp". The diff centers on `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py`. PR body context: ## Motivation Support XiaomiMiMo inference with mtp ## Modifications Add new model support. Add corresponding MTP accuracy & latency test ### How to start server ## Test Result...
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo_mtp.py`; associated commits `dea8aa7ab8e8`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-18, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiMo V2 Flash, this PR adds or enables a model support/runtime surface. Title: "Clean unused import for mimo mtp model". The diff centers on `python/sglang/srt/models/mimo_mtp.py`. PR body context: ## Motivation ## Modifications Clean unused import for mimo mtp model. ## Checklist - [x] Format your code according to the Code Formatting with Pre-Commit. - [ ] Add unit tests...
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/mimo_detector.py`, `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/models/mimo_v2_flash_nextn.py`; associated commits `160a06cab23f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 38 files, +5396/-169, 6509 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiMo V2 Flash, this PR adds or enables a model support/runtime surface. Title: "[Feature] Xiaomi `MiMo-V2-Flash` day0 support". The diff centers on `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/models/mimo_v2_flash_nextn.py`, `python/sglang/srt/function_call/mimo_detector.py`. PR body context: ## Motivation MiMo-V2-Flash is a Mixture-of-Experts (MoE) language model with 309B total parameters and 15B active parameters. Designed for high-speed reasoning and agentic work...
- Key implementation: `python/sglang/srt/models/mimo_v2_flash.py` added +927/-0 (927 lines); hunks: -0,0 +1,927; symbols: MiMoV2MLP, __init__, forward, MoEGate, touching `MiMoV2MLP, __init__, forward`; `python/sglang/srt/models/mimo_v2_flash_nextn.py` added +366/-0 (366 lines); hunks: -0,0 +1,366; symbols: MiMoV2MTPLayer, __init__, forward, MiMoV2ModelNextN, touching `MiMoV2MTPLayer, __init__, forward`; `python/sglang/srt/function_call/mimo_detector.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: _get_param_type, _convert_param_value, MiMoDetector, __init__, touching `_get_param_type, _convert_param_value, MiMoDetector`.
- Code diff details:
  - `python/sglang/srt/models/mimo_v2_flash.py` added +927/-0 (927 lines); hunks: -0,0 +1,927; symbols: MiMoV2MLP, __init__, forward, MoEGate
  - `python/sglang/srt/models/mimo_v2_flash_nextn.py` added +366/-0 (366 lines); hunks: -0,0 +1,366; symbols: MiMoV2MTPLayer, __init__, forward, MiMoV2ModelNextN
  - `python/sglang/srt/function_call/mimo_detector.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: _get_param_type, _convert_param_value, MiMoDetector, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mimo_v2_flash.py
@@ -0,0 +1,927 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/mimo_v2_flash_nextn.py
@@ -0,0 +1,366 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/function_call/mimo_detector.py
@@ -0,0 +1,281 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mimo_v2_flash.py` added +927/-0; `python/sglang/srt/models/mimo_v2_flash_nextn.py` added +366/-0; `python/sglang/srt/function_call/mimo_detector.py` added +281/-0
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15464 - Optimize MiMo-V2-Flash by flashinfer fused allreduce

- Link: https://github.com/sgl-project/sglang/pull/15464
- Status/date: merged / 2025-12-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo_v2_flash.py`; associated commits `165f5c04cbc2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +66/-10, 175 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiMo V2 Flash, this PR optimizes an inference path or backend selection. Title: "Optimize MiMo-V2-Flash by flashinfer fused allreduce". The diff centers on `python/sglang/srt/models/mimo_v2_flash.py`. PR body context: ## Motivation This PR is to make MiMo-V2-Flash model leverage FlashInfer fused_allreduce to fuse allreduce+rmsnorm+residual_add. The E2E TTFT reduce 5.1%. ## Modifications ## Ac...
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
- Motivation: For MiMo V2 Flash, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg". The diff centers on `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`. PR body context: ## Motivation MiMoV2Flash uses SWA, it should respect the argument `--swa-full-tokens-ratio` to allocate corresponding size of KV cache. ## Modifications ## Accuracy Tests ## Be...
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo_v2_flash.py`; associated commits `9227d4f74883`
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-20, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiMo V2 Flash, this PR adds or enables a model support/runtime surface. Title: "[Fix] Remove no use code in MiMo-V2-Flash". The diff centers on `python/sglang/srt/models/mimo_v2_flash.py`. PR body context: ## Motivation https://github.com/sgl-project/sglang/pull/15464 introduced some code piece unrelated with flashinfer fused allreduce which had no usage. This PR is to wipe them o...
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo_v2_flash.py`; associated commits `cbf150039037`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +292/-8, 366 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiMo V2 Flash, this PR adds or enables a model support/runtime surface. Title: "[MiMoV2Flash] [feat]: support two batch overlap". The diff centers on `python/sglang/srt/models/mimo_v2_flash.py`. PR body context: ## Motivation support mimo_v2_flash two batch overlap: p: d: lb: ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according t...
- Key implementation: `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8 (216 lines); hunks: -19,18 +19,21; -66,7 +69,12; symbols: forward_deepep, op_gate, op_select_experts, op_dispatch_a, touching `forward_deepep, op_gate, op_select_experts`.
- Code diff details:
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8 (216 lines); hunks: -19,18 +19,21; -66,7 +69,12; symbols: forward_deepep, op_gate, op_select_experts, op_dispatch_a
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
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/batch_overlap/operations_strategy.py`, `python/sglang/srt/models/mimo_v2_flash.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21414 - fix(MiMo-V2-Flash): add mimo reasoning parser

- Link: https://github.com/sgl-project/sglang/pull/21414
- Status/date: merged / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-0, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiMo V2 Flash, this PR adds or enables a model support/runtime surface. Title: "fix(MiMo-V2-Flash): add mimo reasoning parser". The diff centers on `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`. PR body context: ## Motivation MiMo-V2-Flash and similar models default enable_thinking to false in their chat templates, but qwen3-family requests without the flag are currently treated as reas...
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
