# sglang MiniMax M2 Series Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/basic_usage/minimax_m2.md` | [#15538](https://github.com/sgl-project/sglang/pull/15538), [#19443](https://github.com/sgl-project/sglang/pull/19443) |
| `docs_new/cookbook/autoregressive/MiniMax/MiniMax-M2.5.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/MiniMax/MiniMax-M2.7.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/MiniMax/MiniMax-M2.mdx` | no direct PR-number commit |
| `docs_new/docs/basic_usage/minimax_m2.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/minimax-m2-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/minimax-m25-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/minimax-m27-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/function_call/minimax_m2.py` | [#12129](https://github.com/sgl-project/sglang/pull/12129), [#15538](https://github.com/sgl-project/sglang/pull/15538) |
| `python/sglang/srt/models/minimax_m2.py` | [#12129](https://github.com/sgl-project/sglang/pull/12129), [#12798](https://github.com/sgl-project/sglang/pull/12798), [#13297](https://github.com/sgl-project/sglang/pull/13297), [#13659](https://github.com/sgl-project/sglang/pull/13659), [#13892](https://github.com/sgl-project/sglang/pull/13892), [#14047](https://github.com/sgl-project/sglang/pull/14047), [#14416](https://github.com/sgl-project/sglang/pull/14416), [#16483](https://github.com/sgl-project/sglang/pull/16483), [#18217](https://github.com/sgl-project/sglang/pull/18217), [#19577](https://github.com/sgl-project/sglang/pull/19577), [#19995](https://github.com/sgl-project/sglang/pull/19995), [#20067](https://github.com/sgl-project/sglang/pull/20067), ... (20 total) |
| `test/registered/8-gpu-models/test_minimax_m25.py` | [#20067](https://github.com/sgl-project/sglang/pull/20067), [#20083](https://github.com/sgl-project/sglang/pull/20083) |
| `test/registered/8-gpu-models/test_minimax_m25_basic.py` | [#21792](https://github.com/sgl-project/sglang/pull/21792) |
| `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py` | [#19443](https://github.com/sgl-project/sglang/pull/19443) |
| `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py` | [#22722](https://github.com/sgl-project/sglang/pull/22722) |
| `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py` | [#19443](https://github.com/sgl-project/sglang/pull/19443) |
| `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py` | [#22722](https://github.com/sgl-project/sglang/pull/22722) |
| `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py` | [#21524](https://github.com/sgl-project/sglang/pull/21524) |
| `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py` | [#22722](https://github.com/sgl-project/sglang/pull/22722) |
| `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py` | [#21524](https://github.com/sgl-project/sglang/pull/21524) |
| `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py` | [#22722](https://github.com/sgl-project/sglang/pull/22722) |
| `test/registered/ascend/llm_models/test_ascend_minimax_m2.py` | [#17695](https://github.com/sgl-project/sglang/pull/17695) |

## PR Coverage Summary

- Git-traced PRs: 26
- Extra PRs preserved from existing docs: 15
- Total PRs in this document: 41
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-10-26 | [#12129](https://github.com/sgl-project/sglang/pull/12129) | merged | Support MiniMax M2 model | `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py` |
| 2025-10-27 | [#12186](https://github.com/sgl-project/sglang/pull/12186) | merged | improve mimax-m2 rmsnorm precision | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-08 | [#12798](https://github.com/sgl-project/sglang/pull/12798) | merged | Support capturing aux_hidden_states for minimax m2. | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-15 | [#13297](https://github.com/sgl-project/sglang/pull/13297) | merged | Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3 | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-20 | [#13659](https://github.com/sgl-project/sglang/pull/13659) | merged | Super tiny remove unused MiniMaxM2MLP class | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-26 | [#13892](https://github.com/sgl-project/sglang/pull/13892) | merged | fix: correct usage of minimax-m2 deepep moe forward | `python/sglang/srt/models/minimax_m2.py` |
| 2025-12-02 | [#14047](https://github.com/sgl-project/sglang/pull/14047) | merged | Optimize topk sigmoid in minimax_m2 | `python/sglang/srt/models/minimax_m2.py` |
| 2025-12-23 | [#15538](https://github.com/sgl-project/sglang/pull/15538) | merged | Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs | `python/sglang/srt/function_call/minimax_m2.py`, `docs/basic_usage/minimax_m2.md` |
| 2025-12-30 | [#14416](https://github.com/sgl-project/sglang/pull/14416) | merged | Fusing RMSNormTP in minimax_m2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-01-27 | [#17826](https://github.com/sgl-project/sglang/pull/17826) | open | Support Pipeline and Data Parallelism for MiniMax-M2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-02-01 | [#16483](https://github.com/sgl-project/sglang/pull/16483) | merged | Optimizing all_reduce in RMSNormTP in minimax_m2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-02-05 | [#18217](https://github.com/sgl-project/sglang/pull/18217) | merged | [piecewise graph]: support MiniMax-M2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-02-05 | [#18310](https://github.com/sgl-project/sglang/pull/18310) | open | [Fix] MiniMax-M2.1 CUDA Graph + torch.compile crashes due to outplace_all_reduce being traced by Dynamo | `python/sglang/srt/distributed/parallel_state.py` |
| 2026-02-27 | [#19468](https://github.com/sgl-project/sglang/pull/19468) | open | fix[minimax]: support deepep with minimax models | `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh` |
| 2026-02-27 | [#19443](https://github.com/sgl-project/sglang/pull/19443) | merged | [AMD] [MiniMax-M2.5 Day 0] Add MiniMax-M2.5 nightly accuracy test | `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py`, `docs/basic_usage/minimax_m2.md` |
| 2026-03-02 | [#19577](https://github.com/sgl-project/sglang/pull/19577) | merged | [Feat] add PP Support for minimax-m2 series | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-06 | [#20031](https://github.com/sgl-project/sglang/pull/20031) | open | fix(minimax): support loading merged expert weights (w13) for awq | `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-07 | [#20083](https://github.com/sgl-project/sglang/pull/20083) | merged | [Nightly] Replace MiniMax-M2 with MiniMax-M2.5 | `test/registered/8-gpu-models/test_minimax_m25.py` |
| 2026-03-13 | [#20489](https://github.com/sgl-project/sglang/pull/20489) | open | fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general… | `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/rotary_embedding/base.py` |
| 2026-03-18 | [#20873](https://github.com/sgl-project/sglang/pull/20873) | open | docs: add MiniMax-M2.7 and M2.7-highspeed model support | `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md` |
| 2026-03-18 | [#19995](https://github.com/sgl-project/sglang/pull/19995) | merged | Add packed_modules_mapping for MiniMax-M2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-18 | [#20870](https://github.com/sgl-project/sglang/pull/20870) | merged | [MiniMax M2] Fix KV cache scale loading | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-20 | [#20975](https://github.com/sgl-project/sglang/pull/20975) | open | fix(dp-attn): fix issues with dp-attention for MiniMax M2 | `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-03-20 | [#20931](https://github.com/sgl-project/sglang/pull/20931) | merged | [Bugifx] qwen3 rope parameter compatibility | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-23 | [#17695](https://github.com/sgl-project/sglang/pull/17695) | merged | [NPU] enhance accuracy for model minimaxm2 from 16.5% to 95.5% | `test/registered/ascend/llm_models/test_ascend_minimax_m2.py` |
| 2026-03-24 | [#20905](https://github.com/sgl-project/sglang/pull/20905) | merged | [NPU][ModelSlim] adapt w2 quant layer for Minimax2.5 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-31 | [#21241](https://github.com/sgl-project/sglang/pull/21241) | merged | [bugfix] Fix rope theta config for MiniMax after transformers v5 update | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-03 | [#19652](https://github.com/sgl-project/sglang/pull/19652) | merged | [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+) | `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` |
| 2026-04-03 | [#21524](https://github.com/sgl-project/sglang/pull/21524) | merged | [AMD] Add MiniMax-M2.5 nightly perf benchmarks for MI30x and MI35x | `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py` |
| 2026-04-06 | [#21792](https://github.com/sgl-project/sglang/pull/21792) | merged | [CI] Add basic unit test for Minimax-M2.5 | `test/registered/8-gpu-models/test_minimax_m25_basic.py` |
| 2026-04-07 | [#20919](https://github.com/sgl-project/sglang/pull/20919) | merged | [NPU] Support dp-attention for MiniMax2.5 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-08 | [#22300](https://github.com/sgl-project/sglang/pull/22300) | open | [NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5) | `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/model_loader/utils.py` |
| 2026-04-10 | [#20967](https://github.com/sgl-project/sglang/pull/20967) | merged | 【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-10 | [#20067](https://github.com/sgl-project/sglang/pull/20067) | merged | MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn | `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py` |
| 2026-04-13 | [#20673](https://github.com/sgl-project/sglang/pull/20673) | merged | [Feature][JIT Kernel] Fused TP QK norm For Minimax | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-14 | [#22744](https://github.com/sgl-project/sglang/pull/22744) | open | [NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md` |
| 2026-04-14 | [#22722](https://github.com/sgl-project/sglang/pull/22722) | merged | [AMD] Add MiniMax-M2.7 accuracy and performance nightly tests | `python/sglang/srt/models/minimax_m2.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py` |
| 2026-04-16 | [#22934](https://github.com/sgl-project/sglang/pull/22934) | open | Minimax eplb bugfix | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-21 | [#23301](https://github.com/sgl-project/sglang/pull/23301) | open | [sgl] Stream MiniMax M2 string parameters token-by-token | `python/sglang/srt/function_call/minimax_m2.py` |
| 2026-04-27 | [#22432](https://github.com/sgl-project/sglang/pull/22432) | closed | [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-30 | [#23190](https://github.com/sgl-project/sglang/pull/23190) | merged | [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode | `python/sglang/srt/models/minimax_m2.py` |

## Per-PR Diff Audit Cards

### PR #12129 - Support MiniMax M2 model

- Link: https://github.com/sgl-project/sglang/pull/12129
- Status/date: merged / 2025-10-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/models/minimax_m2.py`; associated commits `7ebc28f5d657`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +1320/-1, 1365 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support MiniMax M2 model"; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`; technical summary: Covers "Support MiniMax M2 model"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` added +922/-0 (922 lines); hunks: -0,0 +1,922; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, forward, touching `MiniMaxM2RMSNormTP, __init__, weight_loader`; `python/sglang/srt/function_call/minimax_m2.py` added +367/-0 (367 lines); hunks: -0,0 +1,367; symbols: _safe_val, MinimaxM2Detector, __init__, has_tool_call, touching `_safe_val, MinimaxM2Detector, __init__`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` added +922/-0 (922 lines); hunks: -0,0 +1,922; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, forward
  - `python/sglang/srt/function_call/minimax_m2.py` added +367/-0 (367 lines); hunks: -0,0 +1,367; symbols: _safe_val, MinimaxM2Detector, __init__, has_tool_call
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -0,0 +1,922 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/function_call/minimax_m2.py
@@ -0,0 +1,367 @@
+import ast
+import html
+import json
+import logging
+import re
+from typing import Any, Dict, List, Tuple
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` added +922/-0; `python/sglang/srt/function_call/minimax_m2.py` added +367/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12186 - improve mimax-m2 rmsnorm precision

- Link: https://github.com/sgl-project/sglang/pull/12186
- Status/date: merged / 2025-10-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "improve mimax-m2 rmsnorm precision"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "improve mimax-m2 rmsnorm precision"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -122,7 +122,7 @@ def forward(
-        x = x.to(orig_dtype) * self.weight
+        x = (x * self.weight).to(orig_dtype)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12798 - Support capturing aux_hidden_states for minimax m2.

- Link: https://github.com/sgl-project/sglang/pull/12798
- Status/date: merged / 2025-11-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `f1a9c72de3c1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-3, 90 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support capturing aux_hidden_states for minimax m2."; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "Support capturing aux_hidden_states for minimax m2."; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +34/-3 (37 lines); hunks: -706,6 +706,9 @@ def layer_fn(idx, prefix: str) -> nn.Module:; -716,7 +719,7 @@ def forward(; symbols: layer_fn, get_input_embeddings, forward, touching `layer_fn, get_input_embeddings, forward`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +34/-3 (37 lines); hunks: -706,6 +706,9 @@ def layer_fn(idx, prefix: str) -> nn.Module:; -716,7 +719,7 @@ def forward(; symbols: layer_fn, get_input_embeddings, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -706,6 +706,9 @@ def layer_fn(idx, prefix: str) -> nn.Module:
+        # For EAGLE3 support
+        self.layers_to_capture = []
@@ -716,7 +719,7 @@ def forward(
-    ) -> Union[torch.Tensor, PPProxyTensors]:
+    ) -> Union[torch.Tensor, PPProxyTensors, Tuple[torch.Tensor, list[torch.Tensor]]]:
@@ -728,6 +731,7 @@ def forward(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +34/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13297 - Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3

- Link: https://github.com/sgl-project/sglang/pull/13297
- Status/date: merged / 2025-11-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `b051d76dabb8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +3/-0 (3 lines); hunks: -821,6 +821,9 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[l...; symbols: set_eagle3_layers_to_capture, get_embed_and_head, forward, touching `set_eagle3_layers_to_capture, get_embed_and_head, forward`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-0 (3 lines); hunks: -821,6 +821,9 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[l...; symbols: set_eagle3_layers_to_capture, get_embed_and_head, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -821,6 +821,9 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
+    def get_embed_and_head(self):
+        return self.model.embed_tokens.weight, self.lm_head.weight
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13659 - Super tiny remove unused MiniMaxM2MLP class

- Link: https://github.com/sgl-project/sglang/pull/13659
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `3f1cfd87b6fd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-36, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Super tiny remove unused MiniMaxM2MLP class"; model line: MiniMax M2 Series; category: model implementation change; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "Super tiny remove unused MiniMaxM2MLP class"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +0/-36 (36 lines); hunks: -31,15 +31,13; -127,40 +125,6 @@ def forward(; symbols: forward, MiniMaxM2MLP, __init__, MiniMaxM2MoE, touching `forward, MiniMaxM2MLP, __init__`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +0/-36 (36 lines); hunks: -31,15 +31,13; -127,40 +125,6 @@ def forward(; symbols: forward, MiniMaxM2MLP, __init__, MiniMaxM2MoE
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -31,15 +31,13 @@
-from sglang.srt.layers.activation import SiluAndMul
-    MergedColumnParallelLinear,
@@ -127,40 +125,6 @@ def forward(
-class MiniMaxM2MLP(nn.Module):
-    def __init__(
-        self,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +0/-36
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13892 - fix: correct usage of minimax-m2 deepep moe forward

- Link: https://github.com/sgl-project/sglang/pull/13892
- Status/date: merged / 2025-11-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `e0e8a9963043`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-7, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: correct usage of minimax-m2 deepep moe forward"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "fix: correct usage of minimax-m2 deepep moe forward"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +3/-7 (10 lines); hunks: -222,7 +222,7 @@ def forward_deepep(; -231,14 +231,10 @@ def forward_deepep(; symbols: forward_deepep, touching `forward_deepep`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-7 (10 lines); hunks: -222,7 +222,7 @@ def forward_deepep(; -231,14 +231,10 @@ def forward_deepep(; symbols: forward_deepep
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -222,7 +222,7 @@ def forward_deepep(
-            topk_weights, topk_idx, _ = self.topk(
+            topk_output = self.topk(
@@ -231,14 +231,10 @@ def forward_deepep(
-            topk_weights, topk_idx, _ = self.topk.empty_topk_output(
-                hidden_states.shape[0], self.top_k
-            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +3/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14047 - Optimize topk sigmoid in minimax_m2

- Link: https://github.com/sgl-project/sglang/pull/14047
- Status/date: merged / 2025-12-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `3dabd609fb03`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +38/-13, 149 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize topk sigmoid in minimax_m2"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "Optimize topk sigmoid in minimax_m2"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +0/-3 (3 lines); hunks: -167,9 +167,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +0/-3 (3 lines); hunks: -167,9 +167,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -167,9 +167,6 @@ def __init__(
-            use_grouped_topk=True,  # TODO: Use "grouped top-k" flag only for hardcoded sigmoid scoring
-            num_expert_group=1,
-            topk_group=1,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15538 - Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs

- Link: https://github.com/sgl-project/sglang/pull/15538
- Status/date: merged / 2025-12-23
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/minimax_m2.md`, `python/sglang/srt/function_call/minimax_m2.py`; associated commits `5c64a20da7dd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +254/-19, 345 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `python/sglang/srt/function_call/minimax_m2.py`, `docs/basic_usage/minimax_m2.md`; technical summary: Covers "Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs"; the main implementation surface is `python/sglang/srt/function_call/minimax_m2.py`, `docs/basic_usage/minimax_m2.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/minimax_m2.py` modified +185/-17 (202 lines); hunks: -1,5 +1,3; -16,17 +14,6; symbols: _safe_val, MinimaxM2Detector, detect_and_parse, _convert_param_value, touching `_safe_val, MinimaxM2Detector, detect_and_parse`; `docs/basic_usage/minimax_m2.md` added +66/-0 (66 lines); hunks: -0,0 +1,66.
- Code diff details:
  - `python/sglang/srt/function_call/minimax_m2.py` modified +185/-17 (202 lines); hunks: -1,5 +1,3; -16,17 +14,6; symbols: _safe_val, MinimaxM2Detector, detect_and_parse, _convert_param_value
  - `docs/basic_usage/minimax_m2.md` added +66/-0 (66 lines); hunks: -0,0 +1,66
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/minimax_m2.py
@@ -1,5 +1,3 @@
-import ast
-import html
@@ -16,17 +14,6 @@
-def _safe_val(raw: str) -> Any:
-    raw = html.unescape(raw.strip())
-    try:
diff -- docs/basic_usage/minimax_m2.md
@@ -0,0 +1,66 @@
+# MiniMax M2.1/M2 Usage
+[MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) and [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) are advanced large language models created by [MiniMax]
+MiniMax-M2 series redefines efficiency for agents. It's a compact, fast, and cost-effective MoE model (230 billion total parameters with 10 billion active parameters) built for el
+## Supported Models
+This guide applies to the following models. You only need to update the model name during deployment. The following examples use **MiniMax-M2**:
+- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/minimax_m2.py` modified +185/-17
  - docs: `docs/basic_usage/minimax_m2.md` added +66/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14416 - Fusing RMSNormTP in minimax_m2

- Link: https://github.com/sgl-project/sglang/pull/14416
- Status/date: merged / 2025-12-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `d17b9e639224`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +189/-2, 219 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fusing RMSNormTP in minimax_m2"; model line: MiniMax M2 Series; category: model implementation change; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "Fusing RMSNormTP in minimax_m2"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +189/-2 (191 lines); hunks: -19,6 +19,8; -73,6 +75,164; symbols: rmsnorm_sumsq_kernel_serial, rmsnorm_apply_kernel_serial, rms_sumsq_serial, rms_apply_serial, touching `rmsnorm_sumsq_kernel_serial, rmsnorm_apply_kernel_serial, rms_sumsq_serial`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +189/-2 (191 lines); hunks: -19,6 +19,8; -73,6 +75,164; symbols: rmsnorm_sumsq_kernel_serial, rmsnorm_apply_kernel_serial, rms_sumsq_serial, rms_apply_serial
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -19,6 +19,8 @@
+import triton
+import triton.language as tl
@@ -73,6 +75,164 @@
+@triton.jit
+def rmsnorm_sumsq_kernel_serial(
+    x1_ptr,  # T* [B, D]
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +189/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17826 - Support Pipeline and Data Parallelism for MiniMax-M2

- Link: https://github.com/sgl-project/sglang/pull/17826
- Status/date: open / 2026-01-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +167/-70, 479 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support Pipeline and Data Parallelism for MiniMax-M2"; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "Support Pipeline and Data Parallelism for MiniMax-M2"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +167/-70 (237 lines); hunks: -16,7 +16,8; -28,7 +29,6; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, ebias_weight_loader, touching `MiniMaxM2RMSNormTP, __init__, weight_loader`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +167/-70 (237 lines); hunks: -16,7 +16,8; -28,7 +29,6; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, ebias_weight_loader
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -16,7 +16,8 @@
-from typing import Iterable, Optional, Set, Tuple, Union
+from contextlib import nullcontext
+from typing import Iterable, List, Optional, Set, Tuple, Union
@@ -28,7 +29,6 @@
-    get_tensor_model_parallel_rank,
@@ -39,6 +39,11 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +167/-70
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16483 - Optimizing all_reduce in RMSNormTP in minimax_m2

- Link: https://github.com/sgl-project/sglang/pull/16483
- Status/date: merged / 2026-02-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `486c7de39f5c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-2, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimizing all_reduce in RMSNormTP in minimax_m2"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "Optimizing all_reduce in RMSNormTP in minimax_m2"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +8/-2 (10 lines); hunks: -166,7 +166,14 @@ def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) ->...; -285,7 +292,6 @@ def forward(; symbols: rms_sumsq_serial, forward, forward_qk, touching `rms_sumsq_serial, forward, forward_qk`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +8/-2 (10 lines); hunks: -166,7 +166,14 @@ def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) ->...; -285,7 +292,6 @@ def forward(; symbols: rms_sumsq_serial, forward, forward_qk
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -166,7 +166,14 @@ def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
-    sum_sq = torch.empty(B + B2, device=x1.device, dtype=torch.float32)
+    # We found that custom all-reduce `sglang::cross_device_reduce_1stage`
+    # is much faster than the nccl all-reduce in torch.
+    # However, `should_custom_ar` checks if the reduced buffer is 16-byte aligned.
+    # RMSNormTP reduces a [B, 2] fp32 tensor, so we pad the total element count to
+    # satisfy the alignment requirement.
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +8/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18217 - [piecewise graph]: support MiniMax-M2

- Link: https://github.com/sgl-project/sglang/pull/18217
- Status/date: merged / 2026-02-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `079fc8f3c591`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +28/-7, 70 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[piecewise graph]: support MiniMax-M2"; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "[piecewise graph]: support MiniMax-M2"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +23/-7 (30 lines); hunks: -16,6 +16,7; -442,9 +443,14 @@ def op_select_experts(self, state):; symbols: op_select_experts, op_dispatch_a, op_dispatch_b, forward, touching `op_select_experts, op_dispatch_a, op_dispatch_b`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +23/-7 (30 lines); hunks: -16,6 +16,7; -442,9 +443,14 @@ def op_select_experts(self, state):; symbols: op_select_experts, op_dispatch_a, op_dispatch_b, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -16,6 +16,7 @@
+from contextlib import nullcontext
@@ -442,9 +443,14 @@ def op_select_experts(self, state):
-            with get_global_expert_distribution_recorder().with_current_layer(
-                self.layer_id
-            ):
+            ctx = (
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +23/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18310 - [Fix] MiniMax-M2.1 CUDA Graph + torch.compile crashes due to outplace_all_reduce being traced by Dynamo

- Link: https://github.com/sgl-project/sglang/pull/18310
- Status/date: open / 2026-02-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-0, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] MiniMax-M2.1 CUDA Graph + torch.compile crashes due to outplace_all_reduce being traced by Dynamo"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/distributed/parallel_state.py`; technical summary: Covers "[Fix] MiniMax-M2.1 CUDA Graph + torch.compile crashes due to outplace_all_reduce being traced by Dynamo"; the main implementation surface is `python/sglang/srt/distributed/parallel_state.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/distributed/parallel_state.py` modified +8/-0 (8 lines); hunks: -586,6 +586,14 @@ def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:; symbols: all_reduce, touching `all_reduce`.
- Code diff details:
  - `python/sglang/srt/distributed/parallel_state.py` modified +8/-0 (8 lines); hunks: -586,6 +586,14 @@ def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:; symbols: all_reduce
- Key code excerpts:

```diff
diff -- python/sglang/srt/distributed/parallel_state.py
@@ -586,6 +586,14 @@ def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
+        # IMPORTANT:
+        # Never allow Dynamo/Inductor to trace the out-of-place all-reduce path.
+        # If it gets traced, it will appear in compiled code and break CUDA graph replay.
+        # Reference: https://github.com/sgl-project/sglang/issues/16102
+        if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
+            torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
```

- Reviewed files:
  - runtime: `python/sglang/srt/distributed/parallel_state.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/distributed/parallel_state.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19468 - fix[minimax]: support deepep with minimax models

- Link: https://github.com/sgl-project/sglang/pull/19468
- Status/date: open / 2026-02-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +10/-2, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix[minimax]: support deepep with minimax models"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh`; technical summary: Covers "fix[minimax]: support deepep with minimax models"; the main implementation surface is `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: -2117,6 +2117,12 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`; `docker/Dockerfile` modified +2/-1 (3 lines); hunks: -9,7 +9,8 @@ ARG HOPPER_SBO=0; `scripts/ci/cuda/ci_install_deepep.sh` modified +2/-1 (3 lines); hunks: -88,9 +88,10 @@ if [ "$GRACE_BLACKWELL" = "1" ]; then.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: -2117,6 +2117,12 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `docker/Dockerfile` modified +2/-1 (3 lines); hunks: -9,7 +9,8 @@ ARG HOPPER_SBO=0
  - `scripts/ci/cuda/ci_install_deepep.sh` modified +2/-1 (3 lines); hunks: -88,9 +88,10 @@ if [ "$GRACE_BLACKWELL" = "1" ]; then
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -2117,6 +2117,12 @@ def _handle_model_specific_adjustments(self):
+        elif model_arch in ["MiniMaxM2ForCausalLM"]:
+            if self.moe_a2a_backend == "deepep":
+                # When using DeepEP, we need to make sure activation dtype is bf16 and not float16
+                # otherwise DeepEP will error due to activation dtype mismatch.
+                self.dtype = "bfloat16"
diff -- docker/Dockerfile
@@ -9,7 +9,8 @@ ARG HOPPER_SBO=0
-ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
+# https://github.com/deepseek-ai/DeepEP/pull/458
+ARG DEEPEP_COMMIT=73b6ea4a439ba03a695563f9fd242c8e4b02b37c
diff -- scripts/ci/cuda/ci_install_deepep.sh
@@ -88,9 +88,10 @@ if [ "$GRACE_BLACKWELL" = "1" ]; then
+    DEEPEP_COMMIT=73b6ea4a439ba03a695563f9fd242c8e4b02b37c
-    git checkout 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee && \
+    git checkout ${DEEPEP_COMMIT} && \
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +6/-0
  - other: `docker/Dockerfile` modified +2/-1; `scripts/ci/cuda/ci_install_deepep.sh` modified +2/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19443 - [AMD] [MiniMax-M2.5 Day 0] Add MiniMax-M2.5 nightly accuracy test

- Link: https://github.com/sgl-project/sglang/pull/19443
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/minimax_m2.md`, `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py`; associated commits `403195d59de0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +653/-4, 766 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] [MiniMax-M2.5 Day 0] Add MiniMax-M2.5 nightly accuracy test"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py`, `docs/basic_usage/minimax_m2.md`; technical summary: Covers "[AMD] [MiniMax-M2.5 Day 0] Add MiniMax-M2.5 nightly accuracy test"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py`, `docs/basic_usage/minimax_m2.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, __post_init__, get_display_name, get_one_example, touching `ModelConfig, __post_init__, get_display_name`; `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: ModelConfig, __post_init__, get_display_name, get_one_example, touching `ModelConfig, __post_init__, get_display_name`; `docs/basic_usage/minimax_m2.md` modified +22/-3 (25 lines); hunks: -1,13 +1,14; -49,6 +50,24 @@ python -m sglang.launch_server \.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, __post_init__, get_display_name, get_one_example
  - `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: ModelConfig, __post_init__, get_display_name, get_one_example
  - `docs/basic_usage/minimax_m2.md` modified +22/-3 (25 lines); hunks: -1,13 +1,14; -49,6 +50,24 @@ python -m sglang.launch_server \
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py
@@ -0,0 +1,249 @@
+"""MI35x MiniMax-M2.5 GSM8K Completion Evaluation Test (8-GPU)
+Tests MiniMax-M2.5 with TP=8 + EP=8 configuration using few-shot completion
+benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-minimax-m25 suite
+"""
+import ast
diff -- test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py
@@ -0,0 +1,245 @@
+"""AMD MiniMax-M2.5 GSM8K Completion Evaluation Test (8-GPU)
+Tests MiniMax-M2.5 with TP=8 + EP=8 configuration using few-shot completion
+benchmark on MI325/MI300X.
+Registry: nightly-amd-accuracy-8-gpu-minimax-m25 suite
+"""
+import ast
diff -- docs/basic_usage/minimax_m2.md
@@ -1,13 +1,14 @@
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py` added +249/-0; `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py` added +245/-0
  - docs: `docs/basic_usage/minimax_m2.md` modified +22/-3
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19577 - [Feat] add PP Support for minimax-m2 series

- Link: https://github.com/sgl-project/sglang/pull/19577
- Status/date: merged / 2026-03-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `2d183c4e6d32`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +35/-7, 97 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feat] add PP Support for minimax-m2 series"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "[Feat] add PP Support for minimax-m2 series"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +35/-7 (42 lines); hunks: -54,7 +54,7; -967,6 +967,7 @@ def __init__(; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +35/-7 (42 lines); hunks: -54,7 +54,7; -967,6 +967,7 @@ def __init__(; symbols: __init__, forward, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -54,7 +54,7 @@
-from sglang.srt.layers.utils import PPMissingLayer
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
@@ -967,6 +967,7 @@ def __init__(
+        self.pp_group = get_pp_group()
@@ -999,17 +1000,26 @@ def forward(
+        pp_proxy_tensors: Optional[PPProxyTensors] = None,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +35/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20031 - fix(minimax): support loading merged expert weights (w13) for awq

- Link: https://github.com/sgl-project/sglang/pull/20031
- Status/date: open / 2026-03-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +203/-9, 236 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(minimax): support loading merged expert weights (w13) for awq"; model line: MiniMax M2 Series; category: bug fix; main diff: `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "fix(minimax): support loading merged expert weights (w13) for awq"; the main implementation surface is `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/registered/models/test_minimax_m2_weights.py` added +145/-0 (145 lines); hunks: -0,0 +1,145; symbols: TestMiniMaxM2WeightLoading, setUp, test_load_weights_merged_w13, touching `TestMiniMaxM2WeightLoading, setUp, test_load_weights_merged_w13`; `python/sglang/srt/models/minimax_m2.py` modified +58/-9 (67 lines); hunks: -1058,6 +1058,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; -1112,7 +1120,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `tests/registered/models/test_minimax_m2_weights.py` added +145/-0 (145 lines); hunks: -0,0 +1,145; symbols: TestMiniMaxM2WeightLoading, setUp, test_load_weights_merged_w13
  - `python/sglang/srt/models/minimax_m2.py` modified +58/-9 (67 lines); hunks: -1058,6 +1058,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; -1112,7 +1120,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights
- Key code excerpts:

```diff
diff -- tests/registered/models/test_minimax_m2_weights.py
@@ -0,0 +1,145 @@
+import unittest
+from unittest.mock import MagicMock, patch
+import torch
+from transformers import PretrainedConfig
+from sglang.srt.models.minimax_m2 import MiniMaxM2ForCausalLM
+class TestMiniMaxM2WeightLoading(unittest.TestCase):
diff -- python/sglang/srt/models/minimax_m2.py
@@ -1058,6 +1058,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+        expert_params_mapping_fused = FusedMoE.make_expert_params_mapping_fused(
+            ckpt_gate_up_proj_name="w13",
+            ckpt_down_proj_name="w2",
+            ckpt_gate_up_proj_bias_name="w13_bias",
+            ckpt_down_proj_bias_name="w2_bias",
+            num_experts=self.config.num_local_experts,
```

- Reviewed files:
  - tests: `tests/registered/models/test_minimax_m2_weights.py` added +145/-0
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +58/-9
- Risk and verification: The diff ships test coverage in `tests/registered/models/test_minimax_m2_weights.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20083 - [Nightly] Replace MiniMax-M2 with MiniMax-M2.5

- Link: https://github.com/sgl-project/sglang/pull/20083
- Status/date: merged / 2026-03-07
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_minimax_m25.py`; associated commits `1aa6ab41deb5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-14, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Nightly] Replace MiniMax-M2 with MiniMax-M2.5"; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `test/registered/8-gpu-models/test_minimax_m25.py`; technical summary: Covers "[Nightly] Replace MiniMax-M2 with MiniMax-M2.5"; the main implementation surface is `test/registered/8-gpu-models/test_minimax_m25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_minimax_m25.py` renamed +12/-14 (26 lines); hunks: -9,32 +9,30; -43,10 +41,10 @@ def test_minimax_m2(self):; symbols: TestMiniMaxM2, for, TestMiniMaxM25, test_minimax_m2, touching `TestMiniMaxM2, for, TestMiniMaxM25`.
- Code diff details:
  - `test/registered/8-gpu-models/test_minimax_m25.py` renamed +12/-14 (26 lines); hunks: -9,32 +9,30; -43,10 +41,10 @@ def test_minimax_m2(self):; symbols: TestMiniMaxM2, for, TestMiniMaxM25, test_minimax_m2
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_minimax_m25.py
@@ -9,32 +9,30 @@
-MINIMAX_M2_MODEL_PATH = "MiniMaxAI/MiniMax-M2"
+MINIMAX_M25_MODEL_PATH = "MiniMaxAI/MiniMax-M2.5"
-class TestMiniMaxM2(unittest.TestCase):
-    """Unified test class for MiniMax-M2 performance and accuracy.
+class TestMiniMaxM25(unittest.TestCase):
+    """Unified test class for MiniMax-M2.5 performance and accuracy.
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_minimax_m25.py` renamed +12/-14
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_minimax_m25.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20489 - fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general…

- Link: https://github.com/sgl-project/sglang/pull/20489
- Status/date: open / 2026-03-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +118/-20, 247 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general…"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/rotary_embedding/base.py`; technical summary: Covers "fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general…"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/rotary_embedding/base.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: -27,10 +27,14; -244,10 +248,16 @@ def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader, touching `rms_apply_serial, MiniMaxM2RMSNormTP, __init__`; `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -1976,14 +1976,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run, touching `_dummy_run`; `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0 (2 lines); hunks: -291,6 +291,8 @@ def forward_cuda(; symbols: forward_cuda, touching `forward_cuda`; `PR_DESCRIPTION.md` added +78/-0 (78 lines); hunks: -0,0 +1,78.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: -27,10 +27,14; -244,10 +248,16 @@ def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -1976,14 +1976,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run
  - `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0 (2 lines); hunks: -291,6 +291,8 @@ def forward_cuda(; symbols: forward_cuda
  - `PR_DESCRIPTION.md` added +78/-0 (78 lines); hunks: -0,0 +1,78
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2 (5 lines); hunks: -100,9 +100,10 @@ def _set_kv_buffer_impl(; symbols: _set_kv_buffer_impl
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -27,10 +27,14 @@
+    GroupCoordinator,
+    get_attn_tp_group,
+    get_attn_tensor_model_parallel_world_size,
+    get_tp_group,
@@ -244,10 +248,16 @@ def rms_apply_serial(
-    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -1976,14 +1976,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):
-                    [num_tokens],
+                    [num_tokens] * self.server_args.dp_size,
-                    [num_tokens],
+                    [num_tokens] * self.server_args.dp_size,
diff -- python/sglang/srt/layers/rotary_embedding/base.py
@@ -291,6 +291,8 @@ def forward_cuda(
+            if batch_size == 0:
+                return query, key
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +33/-16; `python/sglang/srt/model_executor/model_runner.py` modified +2/-2; `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0; `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2
  - other: `PR_DESCRIPTION.md` added +78/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/rotary_embedding/base.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20873 - docs: add MiniMax-M2.7 and M2.7-highspeed model support

- Link: https://github.com/sgl-project/sglang/pull/20873
- Status/date: open / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +15/-3, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: add MiniMax-M2.7 and M2.7-highspeed model support"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md`; technical summary: Covers "docs: add MiniMax-M2.7 and M2.7-highspeed model support"; the main implementation surface is `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/basic_usage/minimax_m2.md` modified +14/-2 (16 lines); hunks: -1,13 +1,14; -83,3 +84,14 @@ curl http://localhost:8000/v1/chat/completions \; `docs/supported_models/text_generation/generative_models.md` modified +1/-1 (2 lines); hunks: -37,7 +37,7 @@ in the GitHub search bar..
- Code diff details:
  - `docs/basic_usage/minimax_m2.md` modified +14/-2 (16 lines); hunks: -1,13 +1,14; -83,3 +84,14 @@ curl http://localhost:8000/v1/chat/completions \
  - `docs/supported_models/text_generation/generative_models.md` modified +1/-1 (2 lines); hunks: -37,7 +37,7 @@ in the GitHub search bar.
- Key code excerpts:

```diff
diff -- docs/basic_usage/minimax_m2.md
@@ -1,13 +1,14 @@
-# MiniMax M2.5/M2.1/M2 Usage
+# MiniMax M2.7/M2.5/M2.1/M2 Usage
-[MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5), [MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1), and [MiniMax-M2](https://huggingface.co/MiniMaxAI/Min
+[MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7), [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5), [MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniM
+- [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)
@@ -83,3 +84,14 @@ curl http://localhost:8000/v1/chat/completions \
diff -- docs/supported_models/text_generation/generative_models.md
@@ -37,7 +37,7 @@ in the GitHub search bar.
-| **MiniMax-M2** (M2, M2.1, M2.5)               | `MiniMaxAI/MiniMax-M2.5`, `MiniMaxAI/MiniMax-M2.1`, `MiniMaxAI/MiniMax-M2` | MiniMax's SOTA LLM for coding & agentic workflows. |
+| **MiniMax-M2** (M2, M2.1, M2.5, M2.7)               | `MiniMaxAI/MiniMax-M2.7`, `MiniMaxAI/MiniMax-M2.5`, `MiniMaxAI/MiniMax-M2.1`, `MiniMaxAI/MiniMax-M2` | MiniMax's SOTA LLM f
```

- Reviewed files:
  - docs: `docs/basic_usage/minimax_m2.md` modified +14/-2; `docs/supported_models/text_generation/generative_models.md` modified +1/-1
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #19995 - Add packed_modules_mapping for MiniMax-M2

- Link: https://github.com/sgl-project/sglang/pull/19995
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `df1d046de2a1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-0, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add packed_modules_mapping for MiniMax-M2"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "Add packed_modules_mapping for MiniMax-M2"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +12/-0 (12 lines); hunks: -941,6 +941,18 @@ def forward(; symbols: forward, MiniMaxM2ForCausalLM, __init__, touching `forward, MiniMaxM2ForCausalLM, __init__`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +12/-0 (12 lines); hunks: -941,6 +941,18 @@ def forward(; symbols: forward, MiniMaxM2ForCausalLM, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -941,6 +941,18 @@ def forward(
+    packed_modules_mapping = {
+        "qkv_proj": [
+            "q_proj",
+            "k_proj",
+            "v_proj",
+        ],
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +12/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20870 - [MiniMax M2] Fix KV cache scale loading

- Link: https://github.com/sgl-project/sglang/pull/20870
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `a3196d08b8f6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-0, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MiniMax M2] Fix KV cache scale loading"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "[MiniMax M2] Fix KV cache scale loading"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +8/-0 (8 lines); hunks: -1063,10 +1063,18 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +8/-0 (8 lines); hunks: -1063,10 +1063,18 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -1063,10 +1063,18 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            _is_kv_scale = name.endswith(".k_scale") or name.endswith(".v_scale")
+                # Skip kv cache scales - maybe_remap_kv_scale_name expects the
+                # original checkpoint name (e.g. self_attn.k_proj.k_scale) to
+                # remap it to self_attn.attn.k_scale. Renaming k_proj -> qkv_proj
+                # here would break that pattern match.
+                if _is_kv_scale:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20975 - fix(dp-attn): fix issues with dp-attention for MiniMax M2

- Link: https://github.com/sgl-project/sglang/pull/20975
- Status/date: open / 2026-03-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +122/-20, 258 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(dp-attn): fix issues with dp-attention for MiniMax M2"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/model_executor/model_runner.py`; technical summary: Covers "fix(dp-attn): fix issues with dp-attention for MiniMax M2"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/model_executor/model_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: -28,10 +28,14; -247,10 +251,16 @@ def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader, touching `rms_apply_serial, MiniMaxM2RMSNormTP, __init__`; `python/sglang/srt/layers/dp_attention.py` modified +4/-0 (4 lines); hunks: -328,6 +328,10 @@ def get_attention_tp_size() -> int:; symbols: get_attention_tp_size, get_attention_tp_world_size, get_attention_cp_group, touching `get_attention_tp_size, get_attention_tp_world_size, get_attention_cp_group`; `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -2121,14 +2121,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run, touching `_dummy_run`; `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0 (2 lines); hunks: -291,6 +291,8 @@ def forward_cuda(; symbols: forward_cuda, touching `forward_cuda`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: -28,10 +28,14; -247,10 +251,16 @@ def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader
  - `python/sglang/srt/layers/dp_attention.py` modified +4/-0 (4 lines); hunks: -328,6 +328,10 @@ def get_attention_tp_size() -> int:; symbols: get_attention_tp_size, get_attention_tp_world_size, get_attention_cp_group
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -2121,14 +2121,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run
  - `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0 (2 lines); hunks: -291,6 +291,8 @@ def forward_cuda(; symbols: forward_cuda
  - `PR_DESCRIPTION.md` added +78/-0 (78 lines); hunks: -0,0 +1,78
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -28,10 +28,14 @@
+    GroupCoordinator,
+    get_attention_tp_group,
+    get_attention_tp_world_size,
+    get_tp_group,
@@ -247,10 +251,16 @@ def rms_apply_serial(
-    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
diff -- python/sglang/srt/layers/dp_attention.py
@@ -328,6 +328,10 @@ def get_attention_tp_size() -> int:
+def get_attention_tp_world_size() -> int:
+    return get_attn_tensor_model_parallel_world_size()
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -2121,14 +2121,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):
-                    [num_tokens],
+                    [num_tokens] * self.server_args.dp_size,
-                    [num_tokens],
+                    [num_tokens] * self.server_args.dp_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +33/-16; `python/sglang/srt/layers/dp_attention.py` modified +4/-0; `python/sglang/srt/model_executor/model_runner.py` modified +2/-2; `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0; `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2
  - other: `PR_DESCRIPTION.md` added +78/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/rotary_embedding/base.py`, `python/sglang/srt/mem_cache/memory_pool.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20931 - [Bugifx] qwen3 rope parameter compatibility

- Link: https://github.com/sgl-project/sglang/pull/20931
- Status/date: merged / 2026-03-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-3, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugifx] qwen3 rope parameter compatibility"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[Bugifx] qwen3 rope parameter compatibility"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -78,6 +78,7; -566,7 +567,7 @@ def forward_prepare_native(; symbols: forward_prepare_native, apply_qk_norm_rope, __init__, touching `forward_prepare_native, apply_qk_norm_rope, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -78,6 +78,7; -566,7 +567,7 @@ def forward_prepare_native(; symbols: forward_prepare_native, apply_qk_norm_rope, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -78,6 +78,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -566,7 +567,7 @@ def forward_prepare_native(
-            theta = self.config.rope_parameters["rope_theta"]
+            theta = self.rope_theta
@@ -691,8 +692,8 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17695 - [NPU] enhance accuracy for model minimaxm2 from 16.5% to 95.5%

- Link: https://github.com/sgl-project/sglang/pull/17695
- Status/date: merged / 2026-03-23
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/ascend/llm_models/test_ascend_minimax_m2.py`; associated commits `4641e5a3d2bb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +45/-1, 61 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] enhance accuracy for model minimaxm2 from 16.5% to 95.5%"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `test/registered/ascend/llm_models/test_ascend_minimax_m2.py`; technical summary: Covers "[NPU] enhance accuracy for model minimaxm2 from 16.5% to 95.5%"; the main implementation surface is `test/registered/ascend/llm_models/test_ascend_minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/ascend/llm_models/test_ascend_minimax_m2.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: TestMiniMaxM2, touching `TestMiniMaxM2`.
- Code diff details:
  - `test/registered/ascend/llm_models/test_ascend_minimax_m2.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: TestMiniMaxM2
- Key code excerpts:

```diff
diff -- test/registered/ascend/llm_models/test_ascend_minimax_m2.py
@@ -0,0 +1,43 @@
+import unittest
+from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
+from sglang.test.ascend.test_ascend_utils import MINIMAX_M2_WEIGHTS_PATH
+from sglang.test.ci.ci_register import register_npu_ci
+from sglang.test.test_utils import CustomTestCase
+register_npu_ci(
```

- Reviewed files:
  - tests: `test/registered/ascend/llm_models/test_ascend_minimax_m2.py` added +43/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/ascend/test_ascend_utils.py`, `test/registered/ascend/llm_models/test_ascend_minimax_m2.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20905 - [NPU][ModelSlim] adapt w2 quant layer for Minimax2.5

- Link: https://github.com/sgl-project/sglang/pull/20905
- Status/date: merged / 2026-03-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `1b4933d45d93`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +22/-30, 67 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU][ModelSlim] adapt w2 quant layer for Minimax2.5"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "[NPU][ModelSlim] adapt w2 quant layer for Minimax2.5"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -713,7 +713,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -713,7 +713,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -713,7 +713,7 @@ def __init__(
-            prefix=add_prefix("mlp", prefix),
+            prefix=add_prefix("block_sparse_moe", prefix),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21241 - [bugfix] Fix rope theta config for MiniMax after transformers v5 update

- Link: https://github.com/sgl-project/sglang/pull/21241
- Status/date: merged / 2026-03-31
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `b91f78d255d8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bugfix] Fix rope theta config for MiniMax after transformers v5 update"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "[bugfix] Fix rope theta config for MiniMax after transformers v5 update"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +3/-3 (6 lines); hunks: -73,6 +73,7; -570,7 +571,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-3 (6 lines); hunks: -73,6 +73,7; -570,7 +571,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -73,6 +73,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -570,7 +571,7 @@ def __init__(
-        self.rope_theta = config.rope_theta
+        self.rope_theta, self.rope_scaling = get_rope_config(config)
@@ -600,13 +601,12 @@ def __init__(
-        rope_scaling = getattr(config, "rope_scaling", None)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19652 - [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+)

- Link: https://github.com/sgl-project/sglang/pull/19652
- Status/date: merged / 2026-04-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +1410/-95, 1875 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+)"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`; technical summary: Covers "[Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+)"; the main implementation surface is `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` added +320/-0 (320 lines); hunks: -0,0 +1,320; symbols: is_fp4_marlin_supported, should_use_fp4_marlin_fallback, nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale, touching `is_fp4_marlin_supported, should_use_fp4_marlin_fallback, nvfp4_marlin_process_scales`; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +82/-7 (89 lines); hunks: -40,6 +40,11; -1128,7 +1133,7 @@ def get_supported_act_dtypes(cls) -> List[torch.dtype]:; symbols: get_supported_act_dtypes, get_min_capability, common_group_size, create_weights, touching `get_supported_act_dtypes, get_min_capability, common_group_size`; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` modified +66/-8 (74 lines); hunks: -17,6 +17,10; -38,19 +42,27; symbols: CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability, create_weights, touching `CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability`; `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +33/-10 (43 lines); hunks: -23,6 +23,13 @@ def get_scalar_type(num_bits: int, has_zp: bool):; -46,6 +53,8 @@ def fused_marlin_moe(; symbols: get_scalar_type, _get_fp4_scalar_type, fused_marlin_moe, touching `get_scalar_type, _get_fp4_scalar_type, fused_marlin_moe`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` added +320/-0 (320 lines); hunks: -0,0 +1,320; symbols: is_fp4_marlin_supported, should_use_fp4_marlin_fallback, nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +82/-7 (89 lines); hunks: -40,6 +40,11; -1128,7 +1133,7 @@ def get_supported_act_dtypes(cls) -> List[torch.dtype]:; symbols: get_supported_act_dtypes, get_min_capability, common_group_size, create_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` modified +66/-8 (74 lines); hunks: -17,6 +17,10; -38,19 +42,27; symbols: CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability, create_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +33/-10 (43 lines); hunks: -23,6 +23,13 @@ def get_scalar_type(num_bits: int, has_zp: bool):; -46,6 +53,8 @@ def fused_marlin_moe(; symbols: get_scalar_type, _get_fp4_scalar_type, fused_marlin_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` modified +32/-1 (33 lines); hunks: -16,6 +16,10; -34,7 +38,7 @@ def __init__(self):; symbols: __init__, get_min_capability, create_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/marlin_utils_fp4.py
@@ -0,0 +1,320 @@
+# SPDX-License-Identifier: Apache-2.0
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
+"""NVFP4 Marlin fallback: run FP4-quantized models on non-Blackwell GPUs via Marlin kernel."""
+import logging
+from typing import Optional
+import torch
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -40,6 +40,11 @@
+from sglang.srt.layers.quantization.marlin_utils_fp4 import (
+    prepare_fp4_layer_for_marlin,
+    prepare_moe_fp4_layer_for_marlin,
+    should_use_fp4_marlin_fallback,
+)
@@ -1128,7 +1133,7 @@ def get_supported_act_dtypes(cls) -> List[torch.dtype]:
diff -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py
@@ -17,6 +17,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` added +320/-0; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +82/-7; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` modified +66/-8; `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +33/-10; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` modified +32/-1; `python/sglang/srt/layers/moe/moe_runner/marlin.py` modified +9/-1
  - tests: `test/registered/quant/test_nvfp4_marlin_fallback.py` added +788/-0
- Risk and verification: The diff ships test coverage in `test/registered/quant/test_nvfp4_marlin_fallback.py`, `test/registered/unit/model_loader/test_modelopt_loader.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21524 - [AMD] Add MiniMax-M2.5 nightly perf benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/21524
- Status/date: merged / 2026-04-03
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py`, `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py`; associated commits `d07d0a15ceb8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +338/-4, 400 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add MiniMax-M2.5 nightly perf benchmarks for MI30x and MI35x"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py`; technical summary: Covers "[AMD] Add MiniMax-M2.5 nightly perf benchmarks for MI30x and MI35x"; the main implementation surface is `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM25PerformanceMI35x, setUpClass, test_bench_minimax_m25, touching `generate_simple_markdown_report, TestNightlyMiniMaxM25PerformanceMI35x, setUpClass`; `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM25Performance, setUpClass, test_bench_minimax_m25, touching `generate_simple_markdown_report, TestNightlyMiniMaxM25Performance, setUpClass`.
- Code diff details:
  - `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM25PerformanceMI35x, setUpClass, test_bench_minimax_m25
  - `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM25Performance, setUpClass, test_bench_minimax_m25
- Key code excerpts:

```diff
diff -- test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py
@@ -0,0 +1,146 @@
+"""MI35x Nightly performance benchmark for MiniMax-M2.5 (8-GPU).
+This test benchmarks MiniMax-M2.5 with TP=8 + EP=8 configuration on MI35x.
+The model path can be configured via MINIMAX_M25_MODEL_PATH environment variable.
+Registry: nightly-perf-8-gpu-mi35x-minimax-m25 suite
+Example usage:
+    python -m pytest test_minimax_m25_perf_mi35x.py -v
diff -- test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py
@@ -0,0 +1,140 @@
+"""Nightly performance benchmark for MiniMax-M2.5 on MI325/MI300X (8-GPU).
+This test benchmarks MiniMax-M2.5 with TP=8 + EP=8 configuration.
+The model path can be configured via MINIMAX_M25_MODEL_PATH environment variable.
+Registry: nightly-perf-8-gpu-minimax-m25 suite
+Example usage:
+    python -m pytest test_minimax_m25_perf_amd.py -v
```

- Reviewed files:
  - tests: `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py` added +146/-0; `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py` added +140/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py`, `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21792 - [CI] Add basic unit test for Minimax-M2.5

- Link: https://github.com/sgl-project/sglang/pull/21792
- Status/date: merged / 2026-04-06
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_minimax_m25_basic.py`; associated commits `56266de624f9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +84/-0, 85 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Add basic unit test for Minimax-M2.5"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `test/registered/8-gpu-models/test_minimax_m25_basic.py`; technical summary: Covers "[CI] Add basic unit test for Minimax-M2.5"; the main implementation surface is `test/registered/8-gpu-models/test_minimax_m25_basic.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_minimax_m25_basic.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestMiniMaxM25Basic, setUpClass, tearDownClass, test_a_gsm8k, touching `TestMiniMaxM25Basic, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/8-gpu-models/test_minimax_m25_basic.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestMiniMaxM25Basic, setUpClass, tearDownClass, test_a_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_minimax_m25_basic.py
@@ -0,0 +1,84 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
+from sglang.test.send_one import BenchArgs, send_one_prompt
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_minimax_m25_basic.py` added +84/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_minimax_m25_basic.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20919 - [NPU] Support dp-attention for MiniMax2.5

- Link: https://github.com/sgl-project/sglang/pull/20919
- Status/date: merged / 2026-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `ae38b24cc358`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +104/-40, 298 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Support dp-attention for MiniMax2.5"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "[NPU] Support dp-attention for MiniMax2.5"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +82/-39 (121 lines); hunks: -30,7 +30,6; -41,6 +40,12; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, forward, touching `MiniMaxM2RMSNormTP, __init__, weight_loader`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +82/-39 (121 lines); hunks: -30,7 +30,6; -41,6 +40,12; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -30,7 +30,6 @@
-    get_tensor_model_parallel_rank,
@@ -41,6 +40,12 @@
+from sglang.srt.layers.dp_attention import (
+    attn_tp_all_reduce,
+    get_attention_tp_rank,
+    get_attention_tp_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +82/-39
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22300 - [NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5)

- Link: https://github.com/sgl-project/sglang/pull/22300
- Status/date: open / 2026-04-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +30/-6, 65 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5)"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/model_loader/utils.py`; technical summary: Covers "[NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5)"; the main implementation surface is `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/model_loader/utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +5/-2 (7 lines); hunks: -488,8 +488,11 @@ def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(; symbols: flashinfer_gemm_w8a8_block_fp8_linear_with_fallback, touching `flashinfer_gemm_w8a8_block_fp8_linear_with_fallback`; `python/sglang/srt/layers/quantization/fp8.py` modified +5/-0 (5 lines); hunks: -474,11 +474,16 @@ def process_weights_after_loading_block_quant(self, layer:...; symbols: process_weights_after_loading_block_quant, touching `process_weights_after_loading_block_quant`; `python/sglang/srt/model_loader/utils.py` modified +20/-4 (24 lines); hunks: -259,13 +259,29 @@ def post_load_weights(model: nn.Module, model_config: Mode...; symbols: post_load_weights, should_deepgemm_weight_requant_ue8m0, should_async_load, touching `post_load_weights, should_deepgemm_weight_requant_ue8m0, should_async_load`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +5/-2 (7 lines); hunks: -488,8 +488,11 @@ def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(; symbols: flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
  - `python/sglang/srt/layers/quantization/fp8.py` modified +5/-0 (5 lines); hunks: -474,11 +474,16 @@ def process_weights_after_loading_block_quant(self, layer:...; symbols: process_weights_after_loading_block_quant
  - `python/sglang/srt/model_loader/utils.py` modified +20/-4 (24 lines); hunks: -259,13 +259,29 @@ def post_load_weights(model: nn.Module, model_config: Mode...; symbols: post_load_weights, should_deepgemm_weight_requant_ue8m0, should_async_load
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -488,8 +488,11 @@ def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
-    # TRTLLM backend requires K dimension >= 256.
-    if backend == "trtllm" and input_2d.shape[1] < 256:
+    # TRTLLM backend requires K >= 256 and weight scales in UE8M0/R128c4
+    # packed format. Fall back to triton when scales are plain float32.
+    if backend == "trtllm" and (
+        input_2d.shape[1] < 256 or not getattr(weight_scale, "format_ue8m0", False)
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -474,11 +474,16 @@ def process_weights_after_loading_block_quant(self, layer: Module) -> None:
+            # Only requantize to UE8M0 if DeepGEMM can actually run
+            # this layer. If the dtype or shape is unsupported, the GEMM
+            # falls back to triton at runtime, which needs float32 scales.
+                    output_dtype=getattr(layer, "orig_dtype", None),
+                    weight_shape=layer.weight.shape,
diff -- python/sglang/srt/model_loader/utils.py
@@ -259,13 +259,29 @@ def post_load_weights(model: nn.Module, model_config: ModelConfig):
-def should_deepgemm_weight_requant_ue8m0(weight_block_size):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +5/-2; `python/sglang/srt/layers/quantization/fp8.py` modified +5/-0; `python/sglang/srt/model_loader/utils.py` modified +20/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/model_loader/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20967 - 【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16

- Link: https://github.com/sgl-project/sglang/pull/20967
- Status/date: merged / 2026-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `84194c25c1cd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-10, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +34/-10 (44 lines); hunks: -253,27 +253,47 @@ def rms_apply_serial(; -641,10 +661,14 @@ def __init__(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader, touching `rms_apply_serial, MiniMaxM2RMSNormTP, __init__`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +34/-10 (44 lines); hunks: -253,27 +253,47 @@ def rms_apply_serial(; -641,10 +661,14 @@ def __init__(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -253,27 +253,47 @@ def rms_apply_serial(
-    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
+    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-6) -> None:
+        # Align with QKVParallelLinear pattern
+        if self.attn_tp_size >= num_heads:
+            assert (
+                self.attn_tp_size % num_heads == 0
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +34/-10
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20067 - MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn

- Link: https://github.com/sgl-project/sglang/pull/20067
- Status/date: merged / 2026-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`; associated commits `7dbd0dd9f01a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +39/-6, 106 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`; technical summary: Covers "MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +25/-6 (31 lines); hunks: -53,10 +53,13; -417,12 +420,20 @@ def forward_normal(; symbols: forward_normal, forward_prepare, forward_core, __init__, touching `forward_normal, forward_prepare, forward_core`; `test/registered/8-gpu-models/test_minimax_m25.py` modified +10/-0 (10 lines); hunks: -29,6 +29,10 @@ def test_minimax_m25(self):; -37,6 +41,12 @@ def test_minimax_m25(self):; symbols: test_minimax_m25, touching `test_minimax_m25`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +25/-6 (31 lines); hunks: -53,10 +53,13; -417,12 +420,20 @@ def forward_normal(; symbols: forward_normal, forward_prepare, forward_core, __init__
  - `test/registered/8-gpu-models/test_minimax_m25.py` modified +10/-0 (10 lines); hunks: -29,6 +29,10 @@ def test_minimax_m25(self):; -37,6 +41,12 @@ def test_minimax_m25(self):; symbols: test_minimax_m25
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -53,10 +53,13 @@
+from sglang.srt.layers.moe import (
+    get_moe_a2a_backend,
+    should_use_flashinfer_cutlass_moe_fp4_allgather,
+)
-from sglang.srt.layers.moe.utils import get_moe_a2a_backend
@@ -417,12 +420,20 @@ def forward_normal(
diff -- test/registered/8-gpu-models/test_minimax_m25.py
@@ -29,6 +29,10 @@ def test_minimax_m25(self):
+        dp_attn_args = base_args + [
+            "--enable-dp-attention",
+            "--dp=8",
+        ]
@@ -37,6 +41,12 @@ def test_minimax_m25(self):
+            ModelLaunchSettings(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +25/-6
  - tests: `test/registered/8-gpu-models/test_minimax_m25.py` modified +10/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_minimax_m25.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20673 - [Feature][JIT Kernel] Fused TP QK norm For Minimax

- Link: https://github.com/sgl-project/sglang/pull/20673
- Status/date: merged / 2026-04-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `314d6ecf0880`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +923/-82, 1277 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature][JIT Kernel] Fused TP QK norm For Minimax"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "[Feature][JIT Kernel] Fused TP QK norm For Minimax"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +113/-21 (134 lines); hunks: -17,17 +17,23; -42,6 +48,7; symbols: forward, fused_tp_qknorm, MiniMaxM2QKRMSNorm, __init__, touching `forward, fused_tp_qknorm, MiniMaxM2QKRMSNorm`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +113/-21 (134 lines); hunks: -17,17 +17,23; -42,6 +48,7; symbols: forward, fused_tp_qknorm, MiniMaxM2QKRMSNorm, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -17,17 +17,23 @@
-from typing import Iterable, Optional, Set, Tuple, Union
+from functools import lru_cache
+from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union
+from sglang.jit_kernel.all_reduce import (
+    fused_parallel_qknorm,
+    get_fused_parallel_qknorm_max_occupancy,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +113/-21
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_custom_all_reduce.py`, `python/sglang/jit_kernel/tests/test_tp_qknorm.py`, `python/sglang/jit_kernel/tests/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22744 - [NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance

- Link: https://github.com/sgl-project/sglang/pull/22744
- Status/date: open / 2026-04-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +11/-0, 39 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`; technical summary: Covers "[NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance"; the main implementation surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: -449,6 +449,10 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: -687,6 +687,7 @@ class ServerArgs:; -6074,6 +6075,11 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args, touching `ServerArgs, add_cli_args`; `docs/advanced_features/server_arguments.md` modified +1/-0 (1 lines); hunks: -470,6 +470,7 @@ Please consult the documentation below and [server_args.py](....
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: -449,6 +449,10 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: -687,6 +687,7 @@ class ServerArgs:; -6074,6 +6075,11 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
  - `docs/advanced_features/server_arguments.md` modified +1/-0 (1 lines); hunks: -470,6 +470,7 @@ Please consult the documentation below and [server_args.py](...
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -449,6 +449,10 @@ def __init__(
+        # Set float32 matmul precision
+        if server_args.enable_tf32_matmul:
+            torch.set_float32_matmul_precision("high")
diff -- python/sglang/srt/server_args.py
@@ -687,6 +687,7 @@ class ServerArgs:
+    enable_tf32_matmul: bool = False
@@ -6074,6 +6075,11 @@ def add_cli_args(parser: argparse.ArgumentParser):
+        parser.add_argument(
+            "--enable-tf32-matmul",
+            action="store_true",
+            help="Enable float32 matmuls to use TensorFloat32 precision for better performance.",
diff -- docs/advanced_features/server_arguments.md
@@ -470,6 +470,7 @@ Please consult the documentation below and [server_args.py](https://github.com/s
+| `--enable-tf32-matmul` | Enables float32 matmuls to use TensorFloat32 precision for better performance. | `False` | bool flag (set to enable) |
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +4/-0; `python/sglang/srt/server_args.py` modified +6/-0
  - docs: `docs/advanced_features/server_arguments.md` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22722 - [AMD] Add MiniMax-M2.7 accuracy and performance nightly tests

- Link: https://github.com/sgl-project/sglang/pull/22722
- Status/date: merged / 2026-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py`, `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py`; associated commits `eab045b2b74e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +805/-113, 1069 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add MiniMax-M2.7 accuracy and performance nightly tests"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `python/sglang/srt/models/minimax_m2.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py`; technical summary: Covers "[AMD] Add MiniMax-M2.7 accuracy and performance nightly tests"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +7/-1 (8 lines); hunks: -33,7 +33,6; -81,9 +80,16; `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, __post_init__, get_display_name, get_one_example, touching `ModelConfig, __post_init__, get_display_name`; `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: ModelConfig, __post_init__, get_display_name, get_one_example, touching `ModelConfig, __post_init__, get_display_name`; `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM27PerformanceMI35x, setUpClass, test_bench_minimax_m27, touching `generate_simple_markdown_report, TestNightlyMiniMaxM27PerformanceMI35x, setUpClass`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +7/-1 (8 lines); hunks: -33,7 +33,6; -81,9 +80,16
  - `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, __post_init__, get_display_name, get_one_example
  - `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: ModelConfig, __post_init__, get_display_name, get_one_example
  - `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM27PerformanceMI35x, setUpClass, test_bench_minimax_m27
  - `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM27Performance, setUpClass, test_bench_minimax_m27
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -33,7 +33,6 @@
-    get_bool_env_var,
@@ -81,9 +80,16 @@
+# get_bool_env_var is defined in sglang.srt.utils.common, not sglang.srt.distributed.
+# Importing from the wrong module causes this file to fail import, which prevents the
+# native MiniMaxM2ForCausalLM from registering in ModelRegistry. The fallback to the
+# transformers wrapper then crashes on config.rope_parameters (transformers v5 issue).
diff -- test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py
@@ -0,0 +1,249 @@
+"""MI35x MiniMax-M2.7 GSM8K Completion Evaluation Test (8-GPU)
+Tests MiniMax-M2.7 with TP=8 + EP=8 configuration using few-shot completion
+benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-minimax-m27 suite
+"""
+import ast
diff -- test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py
@@ -0,0 +1,245 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +7/-1
  - tests: `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py` added +249/-0; `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py` added +245/-0; `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py` added +146/-0; `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py` added +140/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py`, `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22934 - Minimax eplb bugfix

- Link: https://github.com/sgl-project/sglang/pull/22934
- Status/date: open / 2026-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +25/-0, 53 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Minimax eplb bugfix"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "Minimax eplb bugfix"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +25/-0 (25 lines); hunks: -66,6 +66,7; -88,6 +89,7; symbols: op_output, get_moe_weights, MiniMaxM2Attention, __init__, touching `op_output, get_moe_weights, MiniMaxM2Attention`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +25/-0 (25 lines); hunks: -66,6 +66,7; -88,6 +89,7; symbols: op_output, get_moe_weights, MiniMaxM2Attention, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -66,6 +66,7 @@
+from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
@@ -88,6 +89,7 @@
+    LazyValue,
@@ -683,6 +685,16 @@ def op_output(self, state):
+    def get_moe_weights(self):
+        return [
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +25/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23301 - [sgl] Stream MiniMax M2 string parameters token-by-token

- Link: https://github.com/sgl-project/sglang/pull/23301
- Status/date: open / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +332/-280, 742 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[sgl] Stream MiniMax M2 string parameters token-by-token"; model line: MiniMax M2 Series; category: model implementation change; main diff: `python/sglang/srt/function_call/minimax_m2.py`; technical summary: Covers "[sgl] Stream MiniMax M2 string parameters token-by-token"; the main implementation surface is `python/sglang/srt/function_call/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/minimax_m2.py` modified +332/-280 (612 lines); hunks: -13,6 +13,11; -24,6 +29,9 @@ class MinimaxM2Detector(BaseFormatDetector):; symbols: MinimaxM2Detector, __init__, touching `MinimaxM2Detector, __init__`.
- Code diff details:
  - `python/sglang/srt/function_call/minimax_m2.py` modified +332/-280 (612 lines); hunks: -13,6 +13,11; -24,6 +29,9 @@ class MinimaxM2Detector(BaseFormatDetector):; symbols: MinimaxM2Detector, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/minimax_m2.py
@@ -13,6 +13,11 @@
+_PARAM_END_TAG = "</parameter>"
+_PARAM_END_TAG_LEN = len(_PARAM_END_TAG)
+# Hold back this many chars while streaming to avoid emitting a partial end tag
+_STREAM_HOLD_BACK = _PARAM_END_TAG_LEN - 1  # 11
@@ -24,6 +29,9 @@ class MinimaxM2Detector(BaseFormatDetector):
+    String-typed parameters are streamed token-by-token.
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/minimax_m2.py` modified +332/-280
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22432 - [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2

- Link: https://github.com/sgl-project/sglang/pull/22432
- Status/date: closed / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +69/-11, 154 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "[NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +69/-11 (80 lines); hunks: -17,7 +17,7; -42,6 +42,7; symbols: forward_prepare, forward_prepare_npu, forward_core, forward, touching `forward_prepare, forward_prepare_npu, forward_core`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +69/-11 (80 lines); hunks: -17,7 +17,7; -42,6 +42,7; symbols: forward_prepare, forward_prepare_npu, forward_core, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -17,7 +17,7 @@
-from typing import Iterable, Optional, Set, Tuple, Union
+from typing import Iterable, List, Optional, Set, Tuple, Union
@@ -42,6 +42,7 @@
+    get_attention_tp_group,
@@ -76,10 +77,16 @@
+    is_npu,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +69/-11
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23190 - [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode

- Link: https://github.com/sgl-project/sglang/pull/23190
- Status/date: merged / 2026-04-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/minimax_m2.py`; associated commits `3553fd032251`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +66/-10, 133 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode"; model line: MiniMax M2 Series; category: bug fix; main diff: `python/sglang/srt/models/minimax_m2.py`; technical summary: Covers "[NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode"; the main implementation surface is `python/sglang/srt/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/minimax_m2.py` modified +66/-10 (76 lines); hunks: -18,7 +18,7; -93,13 +93,18; symbols: forward_prepare, forward_prepare_npu, forward_core, forward, touching `forward_prepare, forward_prepare_npu, forward_core`.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +66/-10 (76 lines); hunks: -18,7 +18,7; -93,13 +93,18; symbols: forward_prepare, forward_prepare_npu, forward_core, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -18,7 +18,7 @@
-from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union
+from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
@@ -93,13 +93,18 @@
+    is_npu,
+_is_npu = is_npu()
+if _is_npu:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +66/-10
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
