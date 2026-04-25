# sglang GPT-OSS PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 2
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `benchmark/gpt_oss/README.md` | [#9728](https://github.com/sgl-project/sglang/pull/9728) |
| `docs/basic_usage/gpt_oss.md` | [#9497](https://github.com/sgl-project/sglang/pull/9497), [#9613](https://github.com/sgl-project/sglang/pull/9613), [#9626](https://github.com/sgl-project/sglang/pull/9626) |
| `docs_new/cookbook/autoregressive/OpenAI/GPT-OSS.mdx` | no direct PR-number commit |
| `docs_new/docs/basic_usage/gpt_oss.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/gpt-oss-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/function_call/gpt_oss_detector.py` | [#9043](https://github.com/sgl-project/sglang/pull/9043), [#9190](https://github.com/sgl-project/sglang/pull/9190), [#9657](https://github.com/sgl-project/sglang/pull/9657) |
| `python/sglang/srt/models/gpt_oss.py` | [#8824](https://github.com/sgl-project/sglang/pull/8824), [#8843](https://github.com/sgl-project/sglang/pull/8843), [#8944](https://github.com/sgl-project/sglang/pull/8944), [#9028](https://github.com/sgl-project/sglang/pull/9028), [#9146](https://github.com/sgl-project/sglang/pull/9146), [#9161](https://github.com/sgl-project/sglang/pull/9161), [#9359](https://github.com/sgl-project/sglang/pull/9359), [#9433](https://github.com/sgl-project/sglang/pull/9433), [#9469](https://github.com/sgl-project/sglang/pull/9469), [#9783](https://github.com/sgl-project/sglang/pull/9783), [#14197](https://github.com/sgl-project/sglang/pull/14197), [#17553](https://github.com/sgl-project/sglang/pull/17553), ... (15 total) |
| `python/sglang/test/gpt_oss_common.py` | [#16426](https://github.com/sgl-project/sglang/pull/16426) |
| `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` | [#18869](https://github.com/sgl-project/sglang/pull/18869), [#22237](https://github.com/sgl-project/sglang/pull/22237) |
| `test/registered/8-gpu-models/test_gpt_oss_120b.py` | [#18134](https://github.com/sgl-project/sglang/pull/18134) |
| `test/registered/amd/accuracy/mi30x/test_gpt_oss_eval_amd.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` | no direct PR-number commit |
| `test/registered/core/test_gpt_oss_1gpu.py` | [#16426](https://github.com/sgl-project/sglang/pull/16426) |
| `test/registered/core/test_gpt_oss_sm120.py` | [#20056](https://github.com/sgl-project/sglang/pull/20056) |
| `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` | [#21570](https://github.com/sgl-project/sglang/pull/21570) |
| `test/registered/perf/test_gpt_oss_4gpu_perf.py` | no direct PR-number commit |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-05 | [#8824](https://github.com/sgl-project/sglang/pull/8824) | merged | Add initial support for gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-06 | [#8843](https://github.com/sgl-project/sglang/pull/8843) | merged | Support mxfp4 for GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-08 | [#8944](https://github.com/sgl-project/sglang/pull/8944) | merged | Expert Parallelism for GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-12 | [#9043](https://github.com/sgl-project/sglang/pull/9043) | merged | (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-08-13 | [#9146](https://github.com/sgl-project/sglang/pull/9146) | merged | Fix gpt-oss ~2x memory consumption issue | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-13 | [#9028](https://github.com/sgl-project/sglang/pull/9028) | merged | Support FA3 backend for gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-13 | [#9161](https://github.com/sgl-project/sglang/pull/9161) | merged | Fix broken trtllm_mha attn backend with gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-20 | [#9359](https://github.com/sgl-project/sglang/pull/9359) | merged | Support DP attention with GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-21 | [#9433](https://github.com/sgl-project/sglang/pull/9433) | merged | [fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-22 | [#9469](https://github.com/sgl-project/sglang/pull/9469) | merged | fix: tmp revert gpt oss tp sharding on hopper | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-22 | [#9497](https://github.com/sgl-project/sglang/pull/9497) | merged | [Docs] Add doc and quick demo for gpt-oss responses api & buildin tools | `docs/basic_usage/gpt_oss.md` |
| 2025-08-25 | [#9190](https://github.com/sgl-project/sglang/pull/9190) | merged | Fix Harmony reasoning parser for and auto-separation for gpt-oss models | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-08-25 | [#9613](https://github.com/sgl-project/sglang/pull/9613) | merged | [docs] Refactor, remove compiled results and add gpt-oss | `docs/basic_usage/gpt_oss.md` |
| 2025-08-28 | [#9728](https://github.com/sgl-project/sglang/pull/9728) | merged | gpt-oss blog reproduction document | `benchmark/gpt_oss/README.md` |
| 2025-09-01 | [#9783](https://github.com/sgl-project/sglang/pull/9783) | merged | support fp8 kvcache for hybrid attn backend on GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-09-15 | [#9626](https://github.com/sgl-project/sglang/pull/9626) | merged | Add reasoning examples for GPT-OSS in Markdown examples | `docs/basic_usage/gpt_oss.md` |
| 2025-09-15 | [#9657](https://github.com/sgl-project/sglang/pull/9657) | merged | fix: gpt-oss streaming dropping normal content when tools are provided but not used | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-12-30 | [#14920](https://github.com/sgl-project/sglang/pull/14920) | merged | Eagle: GPT-OSS Eagle v2 support | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2026-01-07 | [#16426](https://github.com/sgl-project/sglang/pull/16426) | merged | Fix gpt_oss_common import path and migrate core tests | `test/registered/core/test_gpt_oss_1gpu.py`, `python/sglang/test/gpt_oss_common.py` |
| 2026-01-18 | [#14197](https://github.com/sgl-project/sglang/pull/14197) | merged | [NPU]Support GPT-OSS for NPU | `python/sglang/srt/models/gpt_oss.py` |
| 2026-01-22 | [#17553](https://github.com/sgl-project/sglang/pull/17553) | merged | [NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py | `python/sglang/srt/models/gpt_oss.py` |
| 2026-02-03 | [#18134](https://github.com/sgl-project/sglang/pull/18134) | merged | feature: adding gpt-oss 120b nightly test | `test/registered/8-gpu-models/test_gpt_oss_120b.py` |
| 2026-02-12 | [#18405](https://github.com/sgl-project/sglang/pull/18405) | merged | [PCG] GPT OSS Triton Kernel Support | `python/sglang/srt/models/gpt_oss.py` |
| 2026-02-16 | [#18869](https://github.com/sgl-project/sglang/pull/18869) | merged | [CI] Remove `--mem-fraction-static 0.93` from gpt-oss test | `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` |
| 2026-02-20 | [#18988](https://github.com/sgl-project/sglang/pull/18988) | merged | [GPT-OSS] support fp8 online quantization for gpt-oss bf16 | `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py` |
| 2026-03-06 | [#20056](https://github.com/sgl-project/sglang/pull/20056) | merged | [CI] Add GPT-OSS test for SM120 | `test/registered/core/test_gpt_oss_sm120.py` |
| 2026-03-24 | [#20755](https://github.com/sgl-project/sglang/pull/20755) | merged | Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+ | `python/sglang/srt/models/gpt_oss.py` |
| 2026-04-02 | [#21570](https://github.com/sgl-project/sglang/pull/21570) | merged | [4/n] Support gpt oss 20b lora | `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` |
| 2026-04-08 | [#22237](https://github.com/sgl-project/sglang/pull/22237) | merged | [CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58 | `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` |

## Per-PR Diff Audit Cards

### PR #8824 - Add initial support for gpt-oss

- Link: https://github.com/sgl-project/sglang/pull/8824
- Status/date: merged / 2025-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `c1d2061f97ae`
- Diff scope read: GitHub Pull Request files API returned 12 files, +1595/-47, 2185 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "Add initial support for gpt-oss". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: Future progress will be tracked here: https://github.com/sgl-project/sglang/issues/8833 **This PR only works for FP8/BF16 ckpt. The FP8/BF16 ckpt has been uploaded to:** `lmsys/...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` added +923/-0 (923 lines); hunks: -0,0 +1,923; symbols: GptOssConfig, __init__, get_attention_sliding_window_size, GptOssSparseMoeBlock, touching `GptOssConfig, __init__, get_attention_sliding_window_size`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` added +923/-0 (923 lines); hunks: -0,0 +1,923; symbols: GptOssConfig, __init__, get_attention_sliding_window_size, GptOssSparseMoeBlock
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -0,0 +1,923 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` added +923/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8843 - Support mxfp4 for GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/8843
- Status/date: merged / 2025-08-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `168033d5fb1e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +791/-325, 1320 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "Support mxfp4 for GPT-OSS". The diff centers on `python/sglang/srt/models/gpt_oss.py`. The PR body has no extra context, so this assessment comes from the title, file list, and patch.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +209/-9 (218 lines); hunks: -25,6 +25,8; -108,11 +110,15 @@ def __init__(; symbols: __init__, _get_default_weight_mapping, load_weights, touching `__init__, _get_default_weight_mapping, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +209/-9 (218 lines); hunks: -25,6 +25,8; -108,11 +110,15 @@ def __init__(; symbols: __init__, _get_default_weight_mapping, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -25,6 +25,8 @@
+    get_moe_expert_parallel_rank,
+    get_moe_expert_parallel_world_size,
@@ -108,11 +110,15 @@ def __init__(
+            quant_config_name = (
+                quant_config.get_name() if quant_config is not None else None
+            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +209/-9
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/layers/quantization/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8944 - Expert Parallelism for GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/8944
- Status/date: merged / 2025-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `1d24db834803`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +269/-119, 956 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "Expert Parallelism for GPT-OSS". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: ## Motivation - What's in this PR: - Enable GPT-OSS launch without triton-kernels - Support expert parallelism for GPT-OSS Example: - TODO: - [x] Test on Blackwell - [x] Code cl...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +54/-47 (101 lines); hunks: -28,6 +28,7; -96,11 +97,6 @@ def __init__(; symbols: __init__, _load_mxfp4_experts_weights, touching `__init__, _load_mxfp4_experts_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +54/-47 (101 lines); hunks: -28,6 +28,7; -96,11 +97,6 @@ def __init__(; symbols: __init__, _load_mxfp4_experts_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,6 +28,7 @@
+    get_moe_tensor_parallel_world_size,
@@ -96,11 +97,6 @@ def __init__(
-        if self.tp_size > config.num_local_experts:
-            raise ValueError(
-                f"Tensor parallel size {self.tp_size} is greater than "
-                f"the number of experts {config.num_local_experts}."
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +54/-47
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9043 - (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support

- Link: https://github.com/sgl-project/sglang/pull/9043
- Status/date: merged / 2025-08-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/gpt_oss_detector.py`; associated commits `a21849013607`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +717/-409, 1293 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "(gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support". The diff centers on `python/sglang/srt/function_call/gpt_oss_detector.py`. PR body context: ## Motivation ### Why remove Harmony #### Harmony integration was removed due to two critical limitations: 1. Missing output token ID support: Harmony requires output token IDs...
- Key implementation: `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse, touching `GptOssDetector, __init__, has_tool_call`.
- Code diff details:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -0,0 +1,331 @@
+import json
+import logging
+import re
+from typing import List
+from sglang.srt.entrypoints.openai.protocol import Tool
+from sglang.srt.function_call.base_format_detector import BaseFormatDetector
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/harmony_utils.py`, `python/sglang/srt/entrypoints/http_server.py`, `python/sglang/srt/entrypoints/openai/protocol.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9146 - Fix gpt-oss ~2x memory consumption issue

- Link: https://github.com/sgl-project/sglang/pull/9146
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `9394ed63867d`
- Diff scope read: GitHub Pull Request files API returned 1 files, +19/-7, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "Fix gpt-oss ~2x memory consumption issue". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +19/-7 (26 lines); hunks: -64,7 +64,13; -655,6 +661,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, forward, _load_normal_weights, touching `__init__, routed_experts_weights_of_layer, forward`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +19/-7 (26 lines); hunks: -64,7 +64,13; -655,6 +661,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, forward, _load_normal_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -64,7 +64,13 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_flashinfer_available, make_layers
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_cuda,
+    is_flashinfer_available,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +19/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9028 - Support FA3 backend for gpt-oss

- Link: https://github.com/sgl-project/sglang/pull/9028
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `0ff6d1fce122`
- Diff scope read: GitHub Pull Request files API returned 4 files, +24/-6, 121 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "Support FA3 backend for gpt-oss". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: ## Motivation Apply changes of https://github.com/sgl-project/sgl-attn/pull/4. ## Accuracy Test `openai/gpt-oss-20b` mmlu 4k: ## Benchmark & Profiling ### `openai/gpt-oss-20b` T...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -294,7 +294,7 @@ def __init__(
-            torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False
+            torch.empty(self.num_heads, dtype=torch.bfloat16), requires_grad=False
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/pyproject.toml`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9161 - Fix broken trtllm_mha attn backend with gpt-oss

- Link: https://github.com/sgl-project/sglang/pull/9161
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `6b7c24712cda`
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix broken trtllm_mha attn backend with gpt-oss". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: ## Motivation FA3 PR #9028 broke trtllm_mha attn backend. ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the F...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +5/-1 (6 lines); hunks: -293,8 +293,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +5/-1 (6 lines); hunks: -293,8 +293,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -293,8 +293,12 @@ def __init__(
+        # Choose dtype of sinks based on attention backend: trtllm_mha requires float32,
+        # others can use bfloat16
+        attn_backend = global_server_args_dict.get("attention_backend")
+        sinks_dtype = torch.float32 if attn_backend == "trtllm_mha" else torch.bfloat16
-            torch.empty(self.num_heads, dtype=torch.bfloat16), requires_grad=False
+            torch.empty(self.num_heads, dtype=sinks_dtype), requires_grad=False
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9359 - Support DP attention with GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/9359
- Status/date: merged / 2025-08-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `c10b8e6a0f2a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-5, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "Support DP attention with GPT-OSS". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: ## Motivation Tested on 4x B200 with DP4 Attn + EP4 MoE Total output: `66,941.40 tok/s` Total output per GPU: `16,735.35 tok/s/gpu` Cmds: ## Modifications ## Accuracy Tests ## B...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -1123,7 +1123,7 @@ def _load_normal_weights(; symbols: _load_normal_weights, touching `_load_normal_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -1123,7 +1123,7 @@ def _load_normal_weights(; symbols: _load_normal_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -1123,7 +1123,7 @@ def _load_normal_weights(
-                            start = tp_rank * param.numel()
+                            start = get_attention_tp_rank() * param.numel()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9433 - [fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/9433
- Status/date: merged / 2025-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `dae9a80f43e8`
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-3, 46 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: ## Motivation gpqa and mmlu with tp=4 were incorrect. The tp=1 and tep=4 scores are correct though. The bug is due to incorrect TP sharding of mxfp4 MoE weights, which missed lo...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +9/-1 (10 lines); hunks: -16,6 +16,7; -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights, touching `_load_mxfp4_experts_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +9/-1 (10 lines); hunks: -16,6 +16,7; -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -16,6 +16,7 @@
+import math
@@ -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):
+        assert (
+            intermediate_size % mxfp4_block == 0
+        ), f"{intermediate_size=} must be divisible by {mxfp4_block=}"
-        per_rank_intermediate_size_block = intermediate_size_block // moe_tp_size
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +9/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/protocol.py`, `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9469 - fix: tmp revert gpt oss tp sharding on hopper

- Link: https://github.com/sgl-project/sglang/pull/9469
- Status/date: merged / 2025-08-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `849957bc76c3`
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-3, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "fix: tmp revert gpt oss tp sharding on hopper". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights, touching `_load_mxfp4_experts_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):
-        per_rank_intermediate_size_block = math.ceil(
-            intermediate_size_block / moe_tp_size
-        )
+        if _is_sm100_supported:
+            per_rank_intermediate_size_block = math.ceil(
+                intermediate_size_block / moe_tp_size
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9497 - [Docs] Add doc and quick demo for gpt-oss responses api & buildin tools

- Link: https://github.com/sgl-project/sglang/pull/9497
- Status/date: merged / 2025-08-22
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/gpt_oss.md`; associated commits `fedfe91c1a6e`
- Diff scope read: GitHub Pull Request files API returned 1 files, +106/-0, 110 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "[Docs] Add doc and quick demo for gpt-oss responses api & buildin tools". The diff centers on `docs/basic_usage/gpt_oss.md`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `docs/basic_usage/gpt_oss.md` modified +106/-0 (106 lines); hunks: -1,3 +1,109.
- Code diff details:
  - `docs/basic_usage/gpt_oss.md` modified +106/-0 (106 lines); hunks: -1,3 +1,109
- Key code excerpts:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -1,3 +1,109 @@
+## Responses API & Built-in Tools
+### Responses API
+GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use.
+### Built-in Tools
+GPT‑OSS can call built‑in tools for web search and Python execution. You can use the demo tool server or connect to external MCP tool servers.
+#### Python Tool
```

- Reviewed files:
  - docs: `docs/basic_usage/gpt_oss.md` modified +106/-0
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/gpt_oss.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #9190 - Fix Harmony reasoning parser for and auto-separation for gpt-oss models

- Link: https://github.com/sgl-project/sglang/pull/9190
- Status/date: merged / 2025-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/gpt_oss_detector.py`; associated commits `a0a77d937b99`
- Diff scope read: GitHub Pull Request files API returned 8 files, +1681/-556, 2406 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "Fix Harmony reasoning parser for and auto-separation for gpt-oss models". The diff centers on `python/sglang/srt/function_call/gpt_oss_detector.py`. PR body context: ## Motivation This PR fixes critical regressions in Harmony reasoning output parsing that were introduced in PR #9043. After that change, gpt-oss models were incorrectly concate...
- Key implementation: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256 (400 lines); hunks: -1,7 +1,7; -10,60 +10,31; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse, touching `GptOssDetector, __init__, has_tool_call`.
- Code diff details:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256 (400 lines); hunks: -1,7 +1,7; -10,60 +10,31; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -1,7 +1,7 @@
-from typing import List
+from typing import List, Optional
@@ -10,60 +10,31 @@
+from sglang.srt.harmony_parser import HarmonyParser
-    Detector for T4-style function calls with channel format.
+    Detector for T4-style function calls using HarmonyParser.
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_harmony_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9613 - [docs] Refactor, remove compiled results and add gpt-oss

- Link: https://github.com/sgl-project/sglang/pull/9613
- Status/date: merged / 2025-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/gpt_oss.md`; associated commits `9b08d975a0a5`
- Diff scope read: GitHub Pull Request files API returned 5 files, +166/-611, 638 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "[docs] Refactor, remove compiled results and add gpt-oss". The diff centers on `docs/basic_usage/gpt_oss.md`. PR body context: ## Motivation 1. I refactored `docs/advanced_features/function_calling.ipynb`, making sure the servers are turned off as expected. I moved `Tool Choice Mode` to the end of the d...
- Key implementation: `docs/basic_usage/gpt_oss.md` modified +5/-0 (5 lines); hunks: -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python exe....
- Code diff details:
  - `docs/basic_usage/gpt_oss.md` modified +5/-0 (5 lines); hunks: -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python exe...
- Key code excerpts:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python execution. You can
+### Tool & Reasoning Parser
+- We support OpenAI Reasoning and Tool Call parser, as well as our SGLang native api for tool call and reasoning. Refer to [reasoning parser](../advanced_features/separate_reasoni
```

- Reviewed files:
  - docs: `docs/basic_usage/gpt_oss.md` modified +5/-0
- Risk and verification: Runtime changes concentrate in `scripts/playground/frontend_reasoning.ipynb`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9728 - gpt-oss blog reproduction document

- Link: https://github.com/sgl-project/sglang/pull/9728
- Status/date: merged / 2025-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `benchmark/gpt_oss/README.md`; associated commits `d0934a519257`
- Diff scope read: GitHub Pull Request files API returned 1 files, +163/-0, 164 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR extends deployment docs, tests, or CI coverage. Title: "gpt-oss blog reproduction document". The diff centers on `benchmark/gpt_oss/README.md`. The PR body has no extra context, so this assessment comes from the title, file list, and patch.
- Key implementation: `benchmark/gpt_oss/README.md` added +163/-0 (163 lines); hunks: -0,0 +1,163.
- Code diff details:
  - `benchmark/gpt_oss/README.md` added +163/-0 (163 lines); hunks: -0,0 +1,163
- Key code excerpts:

```diff
diff -- benchmark/gpt_oss/README.md
@@ -0,0 +1,163 @@
+# How to reproduce the result of GPT-OSS with SGLang
+### Install the latest SGLang
+'''bash
+git clone https://github.com/sgl-project/sglang.git
+cd sglang
+git checkout v0.5.1.post3
```

- Reviewed files:
  - other: `benchmark/gpt_oss/README.md` added +163/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #9783 - support fp8 kvcache for hybrid attn backend on GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/9783
- Status/date: merged / 2025-09-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `9db8025376b2`
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-4, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "support fp8 kvcache for hybrid attn backend on GPT-OSS". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: ## Motivation #9782 On B200/GB200 kv cache volume actually blocks the batch size which is the bottleneck for the GPT-OSS performance. And the trtllm-mha cuda kernel could not su...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +5/-4 (9 lines); hunks: -193,8 +193,9 @@ def forward_normal(; -341,7 +342,7 @@ def forward_prepare(; symbols: forward_normal, _enable_fused_set_kv_buffer, forward_prepare, forward_core, touching `forward_normal, _enable_fused_set_kv_buffer, forward_prepare`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +5/-4 (9 lines); hunks: -193,8 +193,9 @@ def forward_normal(; -341,7 +342,7 @@ def forward_prepare(; symbols: forward_normal, _enable_fused_set_kv_buffer, forward_prepare, forward_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -193,8 +193,9 @@ def forward_normal(
-def _enable_fused_set_kv_buffer():
-    return _is_cuda
+def _enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
+    """Enable fused set_kv_buffer only on CUDA with bfloat16 KV cache."""
+    return _is_cuda and forward_batch.token_to_kv_pool.dtype == torch.bfloat16
@@ -341,7 +342,7 @@ def forward_prepare(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +5/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9626 - Add reasoning examples for GPT-OSS in Markdown examples

- Link: https://github.com/sgl-project/sglang/pull/9626
- Status/date: merged / 2025-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/gpt_oss.md`; associated commits `0b14159fc4e0`
- Diff scope read: GitHub Pull Request files API returned 2 files, +12/-2, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "Add reasoning examples for GPT-OSS in Markdown examples". The diff centers on `docs/basic_usage/gpt_oss.md`. PR body context: ## Motivation ## Modifications A useful parameter, and it's not mentioned here already. The information is already in Hugging Face, but it's nice to have it in the examples as w...
- Key implementation: `docs/basic_usage/gpt_oss.md` modified +11/-1 (12 lines); hunks: -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/88...; -69,6 +69,16 @@ tools = [.
- Code diff details:
  - `docs/basic_usage/gpt_oss.md` modified +11/-1 (12 lines); hunks: -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/88...; -69,6 +69,16 @@ tools = [
- Key code excerpts:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/8833](https://gith
-GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use.
+GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use. Yo
@@ -69,6 +69,16 @@ tools = [
+# Reasoning level example
+response = client.responses.create(
+    model="openai/gpt-oss-120b",
```

- Reviewed files:
  - docs: `docs/basic_usage/gpt_oss.md` modified +11/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/protocol.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9657 - fix: gpt-oss streaming dropping normal content when tools are provided but not used

- Link: https://github.com/sgl-project/sglang/pull/9657
- Status/date: merged / 2025-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/gpt_oss_detector.py`; associated commits `28c79dc84ab8`
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-0, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR fixes a launch, loading, parsing, or numerical issue. Title: "fix: gpt-oss streaming dropping normal content when tools are provided but not used". The diff centers on `python/sglang/srt/function_call/gpt_oss_detector.py`. PR body context: ## Motivation Streaming chat responses returned empty content when tools were provided but not actually invoked. For gpt-oss, the tool-call streaming path uses the Harmony parse...
- Key implementation: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0 (23 lines); hunks: -81,6 +81,29 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, touching `parse_streaming_increment`.
- Code diff details:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0 (23 lines); hunks: -81,6 +81,29 @@ def parse_streaming_increment(; symbols: parse_streaming_increment
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -81,6 +81,29 @@ def parse_streaming_increment(
+        # If there are no parsed events and the chunk contains no Harmony structural
+        # markers, treat it as plain text and pass it through. This fixes a bug where
+        # normal content was held in the buffer when tools were provided but not used.
+        if not events:
+            has_harmony_markers = any(
+                marker in self._buffer
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/gpt_oss_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14920 - Eagle: GPT-OSS Eagle v2 support

- Link: https://github.com/sgl-project/sglang/pull/14920
- Status/date: merged / 2025-12-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +48/-25, 124 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "Eagle: GPT-OSS Eagle v2 support". The diff centers on `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`. PR body context: ## Motivation EAGLE v2/v3 models from GPT-OSS introduce an optional auxiliary hidden-state mechanism that improves speculative decoding quality by exposing additional per-layer...
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +30/-23 (53 lines); hunks: -345,6 +345,32 @@ def __init__(; -593,30 +619,11 @@ def initialize(self, min_per_gpu_memory: float):; symbols: __init__, initialize, _dummy_run, touching `__init__, initialize, _dummy_run`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, touching `__init__`; `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: -186,6 +186,15 @@ def __init__(; -897,6 +906,7 @@ def forward_draft_extend_after_decode(self, batch: ScheduleB...; symbols: __init__, forward_draft_extend_after_decode, touching `__init__, forward_draft_extend_after_decode`; `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -100,7 +100,10 @@ def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +30/-23 (53 lines); hunks: -345,6 +345,32 @@ def __init__(; -593,30 +619,11 @@ def initialize(self, min_per_gpu_memory: float):; symbols: __init__, initialize, _dummy_run
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__
  - `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: -186,6 +186,15 @@ def __init__(; -897,6 +906,7 @@ def forward_draft_extend_after_decode(self, batch: ScheduleB...; symbols: __init__, forward_draft_extend_after_decode
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -100,7 +100,10 @@ def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -345,6 +345,32 @@ def __init__(
+        # auxiliary hidden capture mode. TODO: expose this to server args?
+        self.eagle_use_aux_hidden_state = False
+        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
+            # load draft config
+            draft_model_config = ModelConfig.from_server_args(
+                server_args,
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):
-        if model_runner.spec_algorithm.is_eagle3():
+        if (
+            model_runner.spec_algorithm.is_eagle3()
+            and model_runner.eagle_use_aux_hidden_state
+        ):
diff -- python/sglang/srt/speculative/eagle_worker.py
@@ -186,6 +186,15 @@ def __init__(
+        self.eagle_use_aux_hidden_state = False
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +30/-23; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1; `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0; `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16426 - Fix gpt_oss_common import path and migrate core tests

- Link: https://github.com/sgl-project/sglang/pull/16426
- Status/date: merged / 2026-01-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/test/gpt_oss_common.py`, `test/registered/core/test_gpt_oss_1gpu.py`; associated commits `0c474273c514`
- Diff scope read: GitHub Pull Request files API returned 15 files, +48/-26, 255 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix gpt_oss_common import path and migrate core tests". The diff centers on `test/registered/core/test_gpt_oss_1gpu.py`, `python/sglang/test/gpt_oss_common.py`. PR body context: Fix sys.path hack by moving gpt_oss_common.py to python/sglang/test/ and migrate core tests to test/registered/core/.
- Key implementation: `test/registered/core/test_gpt_oss_1gpu.py` renamed +5/-1 (6 lines); hunks: -1,6 +1,10; symbols: TestGptOss1Gpu, touching `TestGptOss1Gpu`; `python/sglang/test/gpt_oss_common.py` renamed +0/-0 (0 lines).
- Code diff details:
  - `test/registered/core/test_gpt_oss_1gpu.py` renamed +5/-1 (6 lines); hunks: -1,6 +1,10; symbols: TestGptOss1Gpu
  - `python/sglang/test/gpt_oss_common.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
diff -- test/registered/core/test_gpt_oss_1gpu.py
@@ -1,6 +1,10 @@
-from test_gpt_oss_common import BaseTestGptOss
+from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
+from sglang.test.gpt_oss_common import BaseTestGptOss
+register_cuda_ci(est_time=402, suite="stage-b-test-small-1-gpu")
+register_amd_ci(est_time=750, suite="stage-b-test-small-1-gpu-amd")
```

- Reviewed files:
  - tests: `test/registered/core/test_gpt_oss_1gpu.py` renamed +5/-1; `python/sglang/test/gpt_oss_common.py` renamed +0/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/gpt_oss_common.py`, `test/registered/core/test_deterministic.py`, `test/registered/core/test_gpt_oss_1gpu.py`, `test/registered/core/test_hidden_states.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14197 - [NPU]Support GPT-OSS for NPU

- Link: https://github.com/sgl-project/sglang/pull/14197
- Status/date: merged / 2026-01-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `733de6be31e2`
- Diff scope read: GitHub Pull Request files API returned 4 files, +96/-17, 244 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "[NPU]Support GPT-OSS for NPU". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: Co-author: @mczywu ## Motivation Adapting GPT-OSS model for NPU. ## Modifications 1. Operators capable of handling sinks and sliding windows have been added to the Ascend backen...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +20/-15 (35 lines); hunks: -71,9 +71,10; -129,6 +130,7 @@ def __init__(; symbols: __init__, forward_prepare, touching `__init__, forward_prepare`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +20/-15 (35 lines); hunks: -71,9 +71,10; -129,6 +130,7 @@ def __init__(; symbols: __init__, forward_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -71,9 +71,10 @@
-from sglang.srt.utils import LazyValue, add_prefix, is_cuda, make_layers
+from sglang.srt.utils import LazyValue, add_prefix, is_cuda, is_npu, make_layers
+_is_npu = is_npu()
@@ -129,6 +130,7 @@ def __init__(
@@ -305,20 +307,20 @@ def forward_prepare(
-        q, k = self.rotary_emb(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +20/-15
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17553 - [NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py

- Link: https://github.com/sgl-project/sglang/pull/17553
- Status/date: merged / 2026-01-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `61abff66c150`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "[NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: ## Motivation NPU support for gpt-oss enabled through the following PR https://github.com/sgl-project/sglang/pull/14197 has a small typo in npu device check for activation. That...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -492,7 +492,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -492,7 +492,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -492,7 +492,7 @@ def __init__(
-        if is_npu:
+        if _is_npu:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18134 - feature: adding gpt-oss 120b nightly test

- Link: https://github.com/sgl-project/sglang/pull/18134
- Status/date: merged / 2026-02-03
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_gpt_oss_120b.py`; associated commits `c8da307d7e63`
- Diff scope read: GitHub Pull Request files API returned 2 files, +88/-4, 121 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR extends deployment docs, tests, or CI coverage. Title: "feature: adding gpt-oss 120b nightly test". The diff centers on `test/registered/8-gpu-models/test_gpt_oss_120b.py`. PR body context: ## Motivation Adding gpt-oss nightly tests to h200 and b200 nightly suites, testing key configs as listed on SGL cookbook here: https://cookbook.sglang.io/docs/autoregressive/Op...
- Key implementation: `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestGptOss120B, for, test_gpt_oss_120b_all_variants, touching `TestGptOss120B, for, test_gpt_oss_120b_all_variants`.
- Code diff details:
  - `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestGptOss120B, for, test_gpt_oss_120b_all_variants
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_gpt_oss_120b.py
@@ -0,0 +1,84 @@
+import unittest
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
+# Runs on both H200 and B200 via nightly-8-gpu-common suite
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_gpt_oss_120b.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18405 - [PCG] GPT OSS Triton Kernel Support

- Link: https://github.com/sgl-project/sglang/pull/18405
- Status/date: merged / 2026-02-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `2bd8363486e4`
- Diff scope read: GitHub Pull Request files API returned 6 files, +68/-32, 228 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "[PCG] GPT OSS Triton Kernel Support". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: ## Motivation Support Backend for GPT-OSS ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +21/-4 (25 lines); hunks: -25,6 +25,10; -72,6 +76,7; symbols: forward_normal, moe_impl, GptOssAttention, __init__, touching `forward_normal, moe_impl, GptOssAttention`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +21/-4 (25 lines); hunks: -25,6 +25,10; -72,6 +76,7; symbols: forward_normal, moe_impl, GptOssAttention, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -25,6 +25,10 @@
+from sglang.srt.compilation.piecewise_context_manager import (
+    get_forward_context,
+    is_in_piecewise_cuda_graph,
+)
@@ -72,6 +76,7 @@
+from sglang.srt.utils.custom_op import register_custom_op
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +21/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/compilation/piecewise_context_manager.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18869 - [CI] Remove `--mem-fraction-static 0.93` from gpt-oss test

- Link: https://github.com/sgl-project/sglang/pull/18869
- Status/date: merged / 2026-02-16
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`; associated commits `8290171f5247`
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-2, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR extends deployment docs, tests, or CI coverage. Title: "[CI] Remove `--mem-fraction-static 0.93` from gpt-oss test". The diff centers on `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`. PR body context: ## Motivation Remove `--mem-fraction-static 0.93` from gpt-oss test to avoid OOMs. ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format you...
- Key implementation: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2 (2 lines); hunks: -30,8 +30,6 @@ def test_mxfp4_120b(self):; symbols: test_mxfp4_120b, touching `test_mxfp4_120b`.
- Code diff details:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2 (2 lines); hunks: -30,8 +30,6 @@ def test_mxfp4_120b(self):; symbols: test_mxfp4_120b
- Key code excerpts:

```diff
diff -- test/registered/4-gpu-models/test_gpt_oss_4gpu.py
@@ -30,8 +30,6 @@ def test_mxfp4_120b(self):
-                "--mem-fraction-static",
-                "0.93",
```

- Reviewed files:
  - tests: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18988 - [GPT-OSS] support fp8 online quantization for gpt-oss bf16

- Link: https://github.com/sgl-project/sglang/pull/18988
- Status/date: merged / 2026-02-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +31/-1, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "[GPT-OSS] support fp8 online quantization for gpt-oss bf16". The diff centers on `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`. PR body context: ## Motivation 1. Keep `moe_runner_backend` as `auto` when launch gpt-oss bf16 with online quantization (e.g. fp8) to pick up either `deep_gemm` or `triton` moe backend, since `t...
- Key implementation: `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0 (26 lines); hunks: -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):; -706,8 +707,10 @@ def create_weights(; symbols: __init__, create_weights, apply, touching `__init__, create_weights, apply`; `python/sglang/srt/server_args.py` modified +5/-1 (6 lines); hunks: -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0 (26 lines); hunks: -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):; -706,8 +707,10 @@ def create_weights(; symbols: __init__, create_weights, apply
  - `python/sglang/srt/server_args.py` modified +5/-1 (6 lines); hunks: -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):
+        self.with_bias = False
@@ -706,8 +707,10 @@ def create_weights(
+        with_bias: bool = False,
+        self.with_bias = with_bias
@@ -782,6 +785,27 @@ def create_weights(
+        # BIAS (optional, e.g. GPT-OSS)
diff -- python/sglang/srt/server_args.py
@@ -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):
-                elif self.ep_size == 1 and is_triton_kernels_available():
+                elif (
+                    self.ep_size == 1
+                    and is_triton_kernels_available()
+                    and self.quantization is None
+                ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0; `python/sglang/srt/server_args.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20056 - [CI] Add GPT-OSS test for SM120

- Link: https://github.com/sgl-project/sglang/pull/20056
- Status/date: merged / 2026-03-06
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/core/test_gpt_oss_sm120.py`; associated commits `8cdb7e1fd453`
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-0, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "[CI] Add GPT-OSS test for SM120". The diff centers on `test/registered/core/test_gpt_oss_sm120.py`. PR body context: ## Summary Adds an SM120-specific CI test for GPT-OSS to prevent regressions on consumer Blackwell hardware. This follows up on #20040.
- Key implementation: `test/registered/core/test_gpt_oss_sm120.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: TestGptOssSm120, setUpClass, test_mxfp4_20b, touching `TestGptOssSm120, setUpClass, test_mxfp4_20b`.
- Code diff details:
  - `test/registered/core/test_gpt_oss_sm120.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: TestGptOssSm120, setUpClass, test_mxfp4_20b
- Key code excerpts:

```diff
diff -- test/registered/core/test_gpt_oss_sm120.py
@@ -0,0 +1,34 @@
+import unittest
+import torch
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.gpt_oss_common import BaseTestGptOss
+register_cuda_ci(est_time=500, suite="stage-b-test-small-1-gpu")
+@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
```

- Reviewed files:
  - tests: `test/registered/core/test_gpt_oss_sm120.py` added +34/-0
- Risk and verification: The diff ships test coverage in `test/registered/core/test_gpt_oss_sm120.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20755 - Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+

- Link: https://github.com/sgl-project/sglang/pull/20755
- Status/date: merged / 2026-03-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `bbe25b24126d`
- Diff scope read: GitHub Pull Request files API returned 1 files, +65/-2, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR optimizes an inference path or backend selection. Title: "Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+". The diff centers on `python/sglang/srt/models/gpt_oss.py`. PR body context: ## Summary FlashInfer 0.6.6 (actually 0.6.5) added `tinygemm_bf16`, a faster kernel for small GEMMs. This applies it to the GPT-OSS MoE router on SM90+. ## Accuracy Test (GPQA)...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +65/-2 (67 lines); hunks: -75,10 +75,34; -97,6 +121,45 @@ def get_attention_sliding_window_size(config):; symbols: GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear, __init__, touching `GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +65/-2 (67 lines); hunks: -75,10 +75,34; -97,6 +121,45 @@ def get_attention_sliding_window_size(config):; symbols: GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -75,10 +75,34 @@
-from sglang.srt.utils import LazyValue, add_prefix, is_npu, make_layers
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_blackwell_supported,
+    is_cuda,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +65/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21570 - [4/n] Support gpt oss 20b lora

- Link: https://github.com/sgl-project/sglang/pull/21570
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`; associated commits `566b4a4f1ccc`
- Diff scope read: GitHub Pull Request files API returned 7 files, +195/-24, 328 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR adds or enables a model support/runtime surface. Title: "[4/n] Support gpt oss 20b lora". The diff centers on `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`. PR body context: ## Motivation 1. Support gpt oss 20b lora 2. Tune some ci thresholds - more strictly ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format y...
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +8/-0 (8 lines); hunks: -17,6 +17,7; -651,6 +652,13 @@ def forward(; symbols: forward, GptOssForCausalLM, should_apply_lora, __init__, touching `forward, GptOssForCausalLM, should_apply_lora`; `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff, test_lora_gpt_oss_20b_logprob_accuracy, touching `kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +8/-0 (8 lines); hunks: -17,6 +17,7; -651,6 +652,13 @@ def forward(; symbols: forward, GptOssForCausalLM, should_apply_lora, __init__
  - `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff, test_lora_gpt_oss_20b_logprob_accuracy
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -17,6 +17,7 @@
+import re
@@ -651,6 +652,13 @@ def forward(
+    _lora_pattern_moe = re.compile(
+        r"^(?:model\.layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.experts)|lm_head|model\.embed_tokens)$"
+    )
+    def should_apply_lora(self, module_name: str) -> bool:
diff -- test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py
@@ -0,0 +1,151 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +8/-0
  - tests: `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0
- Risk and verification: The diff ships test coverage in `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_8b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22237 - [CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58

- Link: https://github.com/sgl-project/sglang/pull/22237
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`; associated commits `2ad5e6df12d3`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GPT-OSS, this PR optimizes an inference path or backend selection. Title: "[CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58". The diff centers on `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`. PR body context: ## Summary - Lower `expected_score` for both `test_bf16_120b` and `test_mxfp4_120b` from 0.60 to 0.58 - 21% failure rate on both B200 and H100 runners at the 0.60 threshold ## S...
- Key implementation: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7 @@ def test_bf16_120b(self):; -23,7 +23,7 @@ def test_mxfp4_120b(self):; symbols: test_bf16_120b, test_mxfp4_120b, touching `test_bf16_120b, test_mxfp4_120b`.
- Code diff details:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7 @@ def test_bf16_120b(self):; -23,7 +23,7 @@ def test_mxfp4_120b(self):; symbols: test_bf16_120b, test_mxfp4_120b
- Key code excerpts:

```diff
diff -- test/registered/4-gpu-models/test_gpt_oss_4gpu.py
@@ -13,7 +13,7 @@ def test_bf16_120b(self):
-                "low": 0.60,
+                "low": 0.58,
@@ -23,7 +23,7 @@ def test_mxfp4_120b(self):
-                "low": 0.60,
+                "low": 0.58,
```

- Reviewed files:
  - tests: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.
