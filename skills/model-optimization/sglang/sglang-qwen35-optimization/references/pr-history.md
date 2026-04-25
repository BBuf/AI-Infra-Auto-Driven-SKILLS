# sglang Qwen3.5 PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 16
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/basic_usage/qwen3_5.md` | no direct PR-number commit |
| `docs/platforms/ascend/ascend_npu_qwen3_5_examples.md` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` | no direct PR-number commit |
| `docs_new/docs/basic_usage/qwen3_5.mdx` | no direct PR-number commit |
| `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/configs/qwen3_5.py` | [#18489](https://github.com/sgl-project/sglang/pull/18489) |
| `python/sglang/srt/models/qwen3_5.py` | [#18489](https://github.com/sgl-project/sglang/pull/18489), [#18538](https://github.com/sgl-project/sglang/pull/18538), [#18544](https://github.com/sgl-project/sglang/pull/18544), [#18937](https://github.com/sgl-project/sglang/pull/18937), [#19070](https://github.com/sgl-project/sglang/pull/19070), [#19220](https://github.com/sgl-project/sglang/pull/19220), [#19411](https://github.com/sgl-project/sglang/pull/19411), [#19670](https://github.com/sgl-project/sglang/pull/19670), [#19767](https://github.com/sgl-project/sglang/pull/19767), [#20386](https://github.com/sgl-project/sglang/pull/20386), [#20736](https://github.com/sgl-project/sglang/pull/20736), [#21019](https://github.com/sgl-project/sglang/pull/21019), ... (17 total) |
| `python/sglang/srt/models/qwen3_5_mtp.py` | [#18489](https://github.com/sgl-project/sglang/pull/18489), [#18538](https://github.com/sgl-project/sglang/pull/18538), [#18926](https://github.com/sgl-project/sglang/pull/18926), [#18937](https://github.com/sgl-project/sglang/pull/18937), [#19391](https://github.com/sgl-project/sglang/pull/19391), [#19767](https://github.com/sgl-project/sglang/pull/19767) |
| `test/lm_eval_configs/Qwen3.5-397B-A17B.yaml` | no direct PR-number commit |
| `test/manual/4-gpu-models/test_qwen35_hicache.py` | no direct PR-number commit |
| `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` | [#22913](https://github.com/sgl-project/sglang/pull/22913) |
| `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` | [#22913](https://github.com/sgl-project/sglang/pull/22913) |
| `test/registered/8-gpu-models/test_qwen35.py` | [#19906](https://github.com/sgl-project/sglang/pull/19906), [#22399](https://github.com/sgl-project/sglang/pull/22399) |
| `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` | [#21669](https://github.com/sgl-project/sglang/pull/21669) |
| `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py` | [#21669](https://github.com/sgl-project/sglang/pull/21669) |
| `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py` | [#21669](https://github.com/sgl-project/sglang/pull/21669) |
| `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` | [#21669](https://github.com/sgl-project/sglang/pull/21669) |
| `test/registered/gb300/test_qwen35_fp8.py` | no direct PR-number commit |
| `test/registered/gb300/test_qwen35_nvfp4.py` | no direct PR-number commit |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-02-09 | [#18489](https://github.com/sgl-project/sglang/pull/18489) | merged | [MODEL] Adding Support for Qwen3.5 Models | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py` |
| 2026-02-12 | [#18538](https://github.com/sgl-project/sglang/pull/18538) | merged | [Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation | `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-12 | [#18544](https://github.com/sgl-project/sglang/pull/18544) | merged | [Ascend]Support qwen3.5 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-18 | [#18926](https://github.com/sgl-project/sglang/pull/18926) | merged | feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation | `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-02-19 | [#18937](https://github.com/sgl-project/sglang/pull/18937) | merged | [Qwen3.5] Enable nvfp4 checkpoint | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-02-25 | [#19070](https://github.com/sgl-project/sglang/pull/19070) | merged | fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-26 | [#19220](https://github.com/sgl-project/sglang/pull/19220) | merged | [PCG] fix piecewise cuda graph for Qwen3.5 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-26 | [#19411](https://github.com/sgl-project/sglang/pull/19411) | merged | [Qwen3.5] Qwen3.5-27B inference repeat bug fix | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-04 | [#19391](https://github.com/sgl-project/sglang/pull/19391) | merged | [Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4 | `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-03-06 | [#19906](https://github.com/sgl-project/sglang/pull/19906) | merged | Add Qwen3.5-397B-A17B nightly test (8-GPU) | `test/registered/8-gpu-models/test_qwen35.py` |
| 2026-03-07 | [#19670](https://github.com/sgl-project/sglang/pull/19670) | merged | [Qwen3.5] Support Qwen3.5 Pipeline Parallelism | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-09 | [#19767](https://github.com/sgl-project/sglang/pull/19767) | merged | Fix qwen3.5 mtp eplb related issues | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-03-12 | [#20386](https://github.com/sgl-project/sglang/pull/20386) | merged | perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe… | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-18 | [#19889](https://github.com/sgl-project/sglang/pull/19889) | merged | Use TRTLLM allreduce fusion for Qwen 3.5 | `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-03-18 | [#19961](https://github.com/sgl-project/sglang/pull/19961) | merged | fix: change qwen 3.5 linear attention a_log to fp32 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-21 | [#21070](https://github.com/sgl-project/sglang/pull/21070) | merged | [Qwen3.5] Fix broken pipeline parallelism layer splitting | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-23 | [#21019](https://github.com/sgl-project/sglang/pull/21019) | merged | [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-29 | [#21487](https://github.com/sgl-project/sglang/pull/21487) | merged | feat(ci): add GB300 nightly benchmark test suites | `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py` |
| 2026-03-30 | [#21448](https://github.com/sgl-project/sglang/pull/21448) | merged | [Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-30 | [#21234](https://github.com/sgl-project/sglang/pull/21234) | merged | [AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-31 | [#20864](https://github.com/sgl-project/sglang/pull/20864) | merged | [Perf]Remove H2D for Qwen3.5 SpecV2 | `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py` |
| 2026-04-01 | [#21347](https://github.com/sgl-project/sglang/pull/21347) | merged | [Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model | `python/sglang/srt/models/qwen3_5.py` |
| 2026-04-06 | [#21849](https://github.com/sgl-project/sglang/pull/21849) | merged | [VLM]: allow Qwen3.5 models for encoder disaggregation | `python/sglang/srt/multimodal/processors/qwen_vl.py`, `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py` |
| 2026-04-07 | [#21669](https://github.com/sgl-project/sglang/pull/21669) | merged | [AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x | `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` |
| 2026-04-07 | [#22145](https://github.com/sgl-project/sglang/pull/22145) | merged | [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support) | `python/sglang/srt/disaggregation/nixl/conn.py` |
| 2026-04-07 | [#22240](https://github.com/sgl-project/sglang/pull/22240) | merged | [Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5) | `python/sglang/srt/disaggregation/nixl/conn.py` |
| 2026-04-08 | [#21692](https://github.com/sgl-project/sglang/pull/21692) | merged | [Bugfix] [NPU] Qwen3.5 with quantization fix | `python/sglang/srt/models/qwen3_5.py` |
| 2026-04-09 | [#22399](https://github.com/sgl-project/sglang/pull/22399) | merged | [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model | `test/registered/8-gpu-models/test_qwen35.py` |
| 2026-04-09 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-10 | [#22312](https://github.com/sgl-project/sglang/pull/22312) | merged | Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B | `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `test/registered/attention/test_gdn_noncontiguous_stride.py` |
| 2026-04-15 | [#20736](https://github.com/sgl-project/sglang/pull/20736) | merged | [AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-04-16 | [#22948](https://github.com/sgl-project/sglang/pull/22948) | merged | [AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled | `python/sglang/srt/models/qwen2_moe.py` |
| 2026-04-17 | [#22913](https://github.com/sgl-project/sglang/pull/22913) | merged | test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6 | `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` |
| 2026-04-17 | [#23034](https://github.com/sgl-project/sglang/pull/23034) | merged | docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs | `docs_new/docs/advanced_features/separate_reasoning.mdx`, `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` |
| 2026-04-18 | [#22431](https://github.com/sgl-project/sglang/pull/22431) | merged | Fix Qwen3.5 video processing when passing video_data in "processor_output" format | `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-04-21 | [#22908](https://github.com/sgl-project/sglang/pull/22908) | merged | [AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict. | `python/sglang/srt/server_args.py` |
| 2026-04-22 | [#22493](https://github.com/sgl-project/sglang/pull/22493) | merged | Add MambaPool kvcache offloading during retraction | `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |
| 2026-04-22 | [#23467](https://github.com/sgl-project/sglang/pull/23467) | merged | fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert | `python/sglang/srt/layers/quantization/utils.py` |

## Per-PR Diff Audit Cards

### PR #18489 - [MODEL] Adding Support for Qwen3.5 Models

- Link: https://github.com/sgl-project/sglang/pull/18489
- Status/date: merged / 2026-02-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; associated commits `27c447653d9c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +1923/-9, 2159 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[MODEL] Adding Support for Qwen3.5 Models". The diff centers on `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`. PR body context: ## Purpose This PR adds model support for the upcoming Qwen3.5 models, including both dense and MoE variants. Special thanks to @cao1zhg, @yizhang2077, and @attack204 for their...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` added +1310/-0 (1310 lines); hunks: -0,0 +1,1310; symbols: Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, forward, touching `Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering`; `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0 (415 lines); hunks: -0,0 +1,415; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward, touching `Qwen3_5MultiTokenPredictor, __init__, embed_input_ids`; `python/sglang/srt/configs/qwen3_5.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__, Qwen3_5Config, touching `Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` added +1310/-0 (1310 lines); hunks: -0,0 +1,1310; symbols: Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, forward
  - `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0 (415 lines); hunks: -0,0 +1,415; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/configs/qwen3_5.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__, Qwen3_5Config
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -0,0 +1,1310 @@
+# Copyright 2025 Qwen Team
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -0,0 +1,415 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/qwen3_5.py
@@ -0,0 +1,113 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` added +1310/-0; `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0; `python/sglang/srt/configs/qwen3_5.py` added +113/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18538 - [Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation

- Link: https://github.com/sgl-project/sglang/pull/18538
- Status/date: merged / 2026-02-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; associated commits `4ed2548427a0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +62/-118, 275 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR extends deployment docs, tests, or CI coverage. Title: "[Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation". The diff centers on `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py`. PR body context: Reduce code redundancy by refactoring MTP to reuse Qwen3_5ForCausalLM - Removed `Qwen3_5MultiTokenPredictor` class, replacing it with `Qwen3_5ForCausalLM` as the main model body...
- Key implementation: `python/sglang/srt/models/qwen3_5_mtp.py` modified +44/-112 (156 lines); hunks: -24,114 +24,15; -140,7 +41,7 @@ def __init__(; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward, touching `Qwen3_5MultiTokenPredictor, __init__, embed_input_ids`; `python/sglang/srt/models/qwen3_5.py` modified +18/-6 (24 lines); hunks: -330,6 +330,9 @@ def __init__(; -338,15 +341,18 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +44/-112 (156 lines); hunks: -24,114 +24,15; -140,7 +41,7 @@ def __init__(; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/models/qwen3_5.py` modified +18/-6 (24 lines); hunks: -330,6 +330,9 @@ def __init__(; -338,15 +341,18 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -24,114 +24,15 @@
-from sglang.srt.layers.vocab_parallel_embedding import (
-    ParallelLMHead,
-    VocabParallelEmbedding,
-)
+from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
-from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer
diff -- python/sglang/srt/models/qwen3_5.py
@@ -330,6 +330,9 @@ def __init__(
+            is_layer_sparse = True
+            is_previous_layer_sparse = True
+            is_next_layer_sparse = True
@@ -338,15 +341,18 @@ def __init__(
+            is_layer_sparse = False
+            is_previous_layer_sparse = False
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5_mtp.py` modified +44/-112; `python/sglang/srt/models/qwen3_5.py` modified +18/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18544 - [Ascend]Support qwen3.5

- Link: https://github.com/sgl-project/sglang/pull/18544
- Status/date: merged / 2026-02-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `1edc69be0854`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +23/-4, 75 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[Ascend]Support qwen3.5". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Motivation Ascend adapts qwen3.5 ## Modifications 1、Bugfix with load weights： issue 2、The quantization and import kernel error reporting parts have been modified accordingly....
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: -34,6 +34,7; -328,15 +329,15 @@ def __init__(; symbols: __init__, load_fused_expert_weights, get_model_config_for_expert_location, touching `__init__, load_fused_expert_weights, get_model_config_for_expert_location`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: -34,6 +34,7; -328,15 +329,15 @@ def __init__(; symbols: __init__, load_fused_expert_weights, get_model_config_for_expert_location
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -34,6 +34,7 @@
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
@@ -328,15 +329,15 @@ def __init__(
-                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
+                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
-                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
+                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18926 - feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation

- Link: https://github.com/sgl-project/sglang/pull/18926
- Status/date: merged / 2026-02-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5_mtp.py`; associated commits `fa5698d79164`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +57/-12, 131 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation". The diff centers on `python/sglang/srt/models/qwen3_5_mtp.py`. PR body context: ## Overview This PR introduces support for block-wise FP8 quantization for the Qwen3.5 series and refines model adaptation logic for several architectures (Mistral-3, Qwen3-VL)...
- Key implementation: `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-6 (7 lines); hunks: -64,7 +64,7 @@ def __init__(; -214,16 +214,11 @@ def load_fused_expert_weights(; symbols: __init__, load_fused_expert_weights, touching `__init__, load_fused_expert_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-6 (7 lines); hunks: -64,7 +64,7 @@ def __init__(; -214,16 +214,11 @@ def load_fused_expert_weights(; symbols: __init__, load_fused_expert_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -64,7 +64,7 @@ def __init__(
-            prefix=add_prefix("model", prefix),
+            prefix=add_prefix("mtp", prefix),
@@ -214,16 +214,11 @@ def load_fused_expert_weights(
-            # Some checkpoints use model.language_model.mtp.* prefix
-            if "language_model" in name:
-                name = name.replace(r"model.language_model.", r"model.")
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18937 - [Qwen3.5] Enable nvfp4 checkpoint

- Link: https://github.com/sgl-project/sglang/pull/18937
- Status/date: merged / 2026-02-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; associated commits `bba2fc49a170`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +26/-8, 98 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[Qwen3.5] Enable nvfp4 checkpoint". The diff centers on `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`. PR body context: ## Motivation Enable nvfp4 checkpoint ## Modifications Disable quantization for the Linear, Global attention modules, visual model, and MTP layer. ## Accuracy Tests No MTP: 0.96...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +19/-7 (26 lines); hunks: -318,8 +318,14 @@ def __init__(; -458,13 +464,19 @@ def __init__(; symbols: __init__, load_weights, load_fused_expert_weights, touching `__init__, load_weights, load_fused_expert_weights`; `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0 (4 lines); hunks: -48,6 +48,10 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +19/-7 (26 lines); hunks: -318,8 +318,14 @@ def __init__(; -458,13 +464,19 @@ def __init__(; symbols: __init__, load_weights, load_fused_expert_weights
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0 (4 lines); hunks: -48,6 +48,10 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -318,8 +318,14 @@ def __init__(
+        linear_attn_quant_config = (
+            None
+            if quant_config and quant_config.get_name() == "modelopt_fp4"
+            else quant_config
+        )
-            config, layer_id, quant_config, alt_stream, prefix
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -48,6 +48,10 @@ def __init__(
+        # The MTP model is unquantized in the nvfp4 checkpoint.
+        if quant_config and quant_config.get_name() == "modelopt_fp4":
+            quant_config = None
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +19/-7; `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19070 - fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1

- Link: https://github.com/sgl-project/sglang/pull/19070
- Status/date: merged / 2026-02-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `d38c0e537d95`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +32/-6, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1 - Add conditional `should_allreduce_fusion` check for MLP layers. - Separate calling logic for MoE vs. Dense block...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +32/-6 (38 lines); hunks: -400,11 +400,24 @@ def forward(; -633,11 +646,24 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +32/-6 (38 lines); hunks: -400,11 +400,24 @@ def forward(; -633,11 +646,24 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -400,11 +400,24 @@ def forward(
-        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)
-        hidden_states, residual = self.layer_communicator.postprocess_layer(
-            hidden_states, residual, forward_batch
+        should_allreduce_fusion = (
+            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
+                forward_batch
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +32/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19220 - [PCG] fix piecewise cuda graph for Qwen3.5

- Link: https://github.com/sgl-project/sglang/pull/19220
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `b3202fe6d072`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +9/-46, 115 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[PCG] fix piecewise cuda graph for Qwen3.5". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Motivation fix piecewise cuda graph for Qwen3.5 ## Modifications 1. fix piecewise cuda graph for Qwen3.5 2. clean up legacy code `gdn_with_output` as it's not used anymore. #...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +1/-21 (22 lines); hunks: -22,9 +22,6; -72,7 +69,6; symbols: forward, _forward, touching `forward, _forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-21 (22 lines); hunks: -22,9 +22,6; -72,7 +69,6; symbols: forward, _forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -22,9 +22,6 @@
-# Model Executor
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -72,7 +69,6 @@
-from sglang.srt.models.qwen3_next import gdn_with_output
@@ -253,22 +249,6 @@ def forward(
-    ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +1/-21
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19411 - [Qwen3.5] Qwen3.5-27B inference repeat bug fix

- Link: https://github.com/sgl-project/sglang/pull/19411
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `bdc1e46e5ac9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Qwen3.5] Qwen3.5-27B inference repeat bug fix". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Motivation fix #19393 fix #19322 When deploying the `Qwen3.5-27B` model with `tp=2`, the model produces repetitive (degenerate) outputs, while `tp=1` works correctly. ### Roo...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -352,6 +352,7 @@ def __init__(; -542,6 +543,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -352,6 +352,7 @@ def __init__(; -542,6 +543,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -352,6 +352,7 @@ def __init__(
+            is_last_layer=(layer_id == config.num_hidden_layers - 1),
@@ -542,6 +543,7 @@ def __init__(
+            is_last_layer=(layer_id == config.num_hidden_layers - 1),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19391 - [Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4

- Link: https://github.com/sgl-project/sglang/pull/19391
- Status/date: merged / 2026-03-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5_mtp.py`; associated commits `9457c049e19e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +252/-16, 332 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4". The diff centers on `python/sglang/srt/models/qwen3_5_mtp.py`. PR body context: ## Motivation - Make MTP_v2 work for Qwen3.5 by passing `mm_input_embeds` to the MTP head. - Add MTP_v1/v2 and non-MTP accuracy test for `nvidia/Qwen3.5-397B-A17B-NVFP4` and che...
- Key implementation: `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-1 (2 lines); hunks: -111,7 +111,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-1 (2 lines); hunks: -111,7 +111,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -111,7 +111,7 @@ def forward(
-            and not forward_batch.forward_mode.is_draft_extend()
+            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19906 - Add Qwen3.5-397B-A17B nightly test (8-GPU)

- Link: https://github.com/sgl-project/sglang/pull/19906
- Status/date: merged / 2026-03-06
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_qwen35.py`; associated commits `ac453b253f58`
- Diff scope read: GitHub Pull Request files API returned 1 files, +74/-0, 75 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "Add Qwen3.5-397B-A17B nightly test (8-GPU)". The diff centers on `test/registered/8-gpu-models/test_qwen35.py`. PR body context: ## Summary Add Qwen3.5-397B-A17B as a new 8-GPU nightly test (followup to #19218). Two variants: - **TP8**: base config with `--reasoning-parser=qwen3` - **TP8+MTP**: speculativ...
- Key implementation: `test/registered/8-gpu-models/test_qwen35.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: TestQwen35, for, test_qwen35, touching `TestQwen35, for, test_qwen35`.
- Code diff details:
  - `test/registered/8-gpu-models/test_qwen35.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: TestQwen35, for, test_qwen35
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_qwen35.py
@@ -0,0 +1,74 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_qwen35.py` added +74/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_qwen35.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19670 - [Qwen3.5] Support Qwen3.5 Pipeline Parallelism

- Link: https://github.com/sgl-project/sglang/pull/19670
- Status/date: merged / 2026-03-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `7da590d4d069`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +114/-13, 194 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[Qwen3.5] Support Qwen3.5 Pipeline Parallelism". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Motivation To close https://github.com/sgl-project/sglang/issues/19500 Currently Qwen3.5 PP will crash with error. With this PR it works. Server: gsm8k no drop. ## Modificati...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +60/-13 (73 lines); hunks: -30,7 +30,7; -59,6 +59,7; symbols: __init__, get_layer, get_input_embeddings, touching `__init__, get_layer, get_input_embeddings`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +60/-13 (73 lines); hunks: -30,7 +30,7; -59,6 +59,7; symbols: __init__, get_layer, get_input_embeddings
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -30,7 +30,7 @@
-from sglang.srt.distributed import get_pp_group
+from sglang.srt.distributed import get_pp_group, get_pp_indices
@@ -59,6 +59,7 @@
+from sglang.srt.layers.utils import PPMissingLayer
@@ -680,6 +681,8 @@ def __init__(
+        else:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +60/-13
- Risk and verification: The diff ships test coverage in `test/registered/distributed/test_pp_single_node.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19767 - Fix qwen3.5 mtp eplb related issues

- Link: https://github.com/sgl-project/sglang/pull/19767
- Status/date: merged / 2026-03-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; associated commits `cabe171b6ce3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +79/-16, 272 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "Fix qwen3.5 mtp eplb related issues". The diff centers on `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`. PR body context: ## Motivation support eplb with mtp for Qwen3.5 & Qwen3-Next ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to th...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +34/-1 (35 lines); hunks: -72,7 +72,14; -294,6 +301,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6 (25 lines); hunks: -22,6 +22,8; -69,6 +71,7 @@ def __init__(; symbols: __init__, get_model_config_for_expert_location, get_embed_and_head, forward, touching `__init__, get_model_config_for_expert_location, get_embed_and_head`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-1 (35 lines); hunks: -72,7 +72,14; -294,6 +301,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6 (25 lines); hunks: -22,6 +22,8; -69,6 +71,7 @@ def __init__(; symbols: __init__, get_model_config_for_expert_location, get_embed_and_head, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -72,7 +72,14 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_npu, make_layers, set_weight_attrs
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_cuda,
+    is_npu,
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -22,6 +22,8 @@
+from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
@@ -69,6 +71,7 @@ def __init__(
+            is_nextn=True,
@@ -84,6 +87,15 @@ def __init__(
+    @classmethod
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-1; `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20386 - perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe…

- Link: https://github.com/sgl-project/sglang/pull/20386
- Status/date: merged / 2026-03-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `9b55a98a6705`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR optimizes an inference path or backend selection. Title: "perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe…". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Motivation `einops.rearrange` performs Python-level string parsing and validation on **every call**. In `Qwen3_5GatedDeltaNet.forward()`, the pattern `rearrange(x, "... h d -...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +1/-2 (3 lines); hunks: -20,7 +20,6; -287,7 +286,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-2 (3 lines); hunks: -20,7 +20,6; -287,7 +286,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -20,7 +20,6 @@
-from einops import rearrange
@@ -287,7 +286,7 @@ def forward(
-        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
+        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19889 - Use TRTLLM allreduce fusion for Qwen 3.5

- Link: https://github.com/sgl-project/sglang/pull/19889
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +88/-52, 210 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR changes model-related implementation. Title: "Use TRTLLM allreduce fusion for Qwen 3.5". The diff centers on `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`. PR body context: Before: 21.5 us After 10.4 us This PR is mainly authored by @vincentzed
- Key implementation: `python/sglang/srt/layers/layernorm.py` modified +63/-48 (111 lines); hunks: -86,6 +86,53; -303,53 +350,10 @@ def forward_with_allreduce_fusion(; symbols: _forward_with_allreduce_fusion, RMSNorm, __init__, forward_with_allreduce_fusion, touching `_forward_with_allreduce_fusion, RMSNorm, __init__`; `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: -397,7 +397,12 @@ def forward(; -646,7 +651,12 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/qwen2_moe.py` modified +11/-2 (13 lines); hunks: -54,7 +54,10; -310,6 +313,7 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1978,6 +1978,8 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/layers/layernorm.py` modified +63/-48 (111 lines); hunks: -86,6 +86,53; -303,53 +350,10 @@ def forward_with_allreduce_fusion(; symbols: _forward_with_allreduce_fusion, RMSNorm, __init__, forward_with_allreduce_fusion
  - `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: -397,7 +397,12 @@ def forward(; -646,7 +651,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +11/-2 (13 lines); hunks: -54,7 +54,10; -310,6 +313,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1978,6 +1978,8 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/layernorm.py
@@ -86,6 +86,53 @@
+def _forward_with_allreduce_fusion(
+    norm_module,
+    x: torch.Tensor,
+    residual: Optional[torch.Tensor],
+    post_residual_addition: Optional[torch.Tensor],
+    weight: torch.Tensor,
diff -- python/sglang/srt/models/qwen3_5.py
@@ -397,7 +397,12 @@ def forward(
-            hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)
+            hidden_states = self.mlp(
+                hidden_states,
+                forward_batch,
+                use_reduce_scatter,
+                should_allreduce_fusion,
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -54,7 +54,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/layernorm.py` modified +63/-48; `python/sglang/srt/models/qwen3_5.py` modified +12/-2; `python/sglang/srt/models/qwen2_moe.py` modified +11/-2; `python/sglang/srt/server_args.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19961 - fix: change qwen 3.5 linear attention a_log to fp32

- Link: https://github.com/sgl-project/sglang/pull/19961
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "fix: change qwen 3.5 linear attention a_log to fp32". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Motivation Change the data type of a_log to fp32 in accordance with the original design of Qwen 3.5. ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: -186,7 +186,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: -186,7 +186,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -186,7 +186,7 @@ def __init__(
-            torch.empty(self.num_v_heads // self.attn_tp_size),
+            torch.empty(self.num_v_heads // self.attn_tp_size, dtype=torch.float32),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21070 - [Qwen3.5] Fix broken pipeline parallelism layer splitting

- Link: https://github.com/sgl-project/sglang/pull/21070
- Status/date: merged / 2026-03-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `852e112ebf00`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +15/-20, 94 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Qwen3.5] Fix broken pipeline parallelism layer splitting". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Summary - **Root cause:** `make_layers()` in `Qwen3_5ForCausalLM` (#19670) was called without `pp_rank`/`pp_size`, so all PP stages instantiated every layer and loaded the fu...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +8/-15 (23 lines); hunks: -29,7 +29,7; -721,25 +721,14 @@ def get_layer(idx: int, prefix: str):; symbols: get_layer, load_fused_expert_weights, touching `get_layer, load_fused_expert_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +8/-15 (23 lines); hunks: -29,7 +29,7; -721,25 +721,14 @@ def get_layer(idx: int, prefix: str):; symbols: get_layer, load_fused_expert_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -29,7 +29,7 @@
-from sglang.srt.distributed import get_pp_group, get_pp_indices
+from sglang.srt.distributed import get_pp_group
@@ -721,25 +721,14 @@ def get_layer(idx: int, prefix: str):
-        self.layers = make_layers(
+        self.layers, self._start_layer, self._end_layer = make_layers(
+            pp_rank=self.pp_group.rank_in_group,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +8/-15
- Risk and verification: The diff ships test coverage in `test/registered/distributed/test_pp_single_node.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21019 - [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel

- Link: https://github.com/sgl-project/sglang/pull/21019
- Status/date: merged / 2026-03-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `5bdc07d974f6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +597/-202, 953 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR optimizes an inference path or backend selection. Title: "[Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Motivation In PR https://github.com/sgl-project/sglang/pull/19321 we fused Qwen3-Next GDN's qkvz_proj and ba_proj. This PR is a follow up. The background that Qwen3-Next and...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +285/-65 (350 lines); hunks: -20,6 +20,11; -54,6 +59,10; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +285/-65 (350 lines); hunks: -20,6 +20,11; -54,6 +59,10; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -20,6 +20,11 @@
+import triton
+from sglang.jit_kernel.triton.gdn_fused_proj import (
+    fused_qkvzba_split_reshape_cat_contiguous,
+)
@@ -54,6 +59,10 @@
+from sglang.srt.layers.parameter import (
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +285/-65
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21487 - feat(ci): add GB300 nightly benchmark test suites

- Link: https://github.com/sgl-project/sglang/pull/21487
- Status/date: merged / 2026-03-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +874/-4, 926 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "feat(ci): add GB300 nightly benchmark test suites". The diff centers on `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`. PR body context: ## Summary - Add 8 per-model registered test files for GB300 (4x B200 NVL4, arm64) nightly benchmarks - Register two new nightly suites: `nightly-4-gpu-gb300-nvfp4` and `nightly...
- Key implementation: `python/sglang/test/accuracy_test_runner.py` modified +296/-3 (299 lines); hunks: -150,6 +150,288 @@ def _run_simple_eval(; -224,13 +506,24 @@ def run_accuracy_test(; symbols: _run_simple_eval, _get_nemo_venv, _ensure_nemo_data_prepared, _run_nemo_skills_eval, touching `_run_simple_eval, _get_nemo_venv, _ensure_nemo_data_prepared`; `test/registered/gb300/test_deepseek_v32_nvfp4.py` added +82/-0 (82 lines); hunks: -0,0 +1,82; symbols: TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4, touching `TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4`; `test/registered/gb300/test_deepseek_v32.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: TestDeepseekV32, test_deepseek_v32, touching `TestDeepseekV32, test_deepseek_v32`; `test/registered/gb300/test_qwen35_nvfp4.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: TestQwen35Nvfp4, test_qwen35_nvfp4, touching `TestQwen35Nvfp4, test_qwen35_nvfp4`.
- Code diff details:
  - `python/sglang/test/accuracy_test_runner.py` modified +296/-3 (299 lines); hunks: -150,6 +150,288 @@ def _run_simple_eval(; -224,13 +506,24 @@ def run_accuracy_test(; symbols: _run_simple_eval, _get_nemo_venv, _ensure_nemo_data_prepared, _run_nemo_skills_eval
  - `test/registered/gb300/test_deepseek_v32_nvfp4.py` added +82/-0 (82 lines); hunks: -0,0 +1,82; symbols: TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4
  - `test/registered/gb300/test_deepseek_v32.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: TestDeepseekV32, test_deepseek_v32
  - `test/registered/gb300/test_qwen35_nvfp4.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: TestQwen35Nvfp4, test_qwen35_nvfp4
  - `test/registered/gb300/test_qwen35_fp8.py` added +75/-0 (75 lines); hunks: -0,0 +1,75; symbols: TestQwen35Fp8, test_qwen35_fp8
- Key code excerpts:

```diff
diff -- python/sglang/test/accuracy_test_runner.py
@@ -150,6 +150,288 @@ def _run_simple_eval(
+# Cached uv venv for NeMo Skills (persists across variants within a process).
+_nemo_venv_dir: Optional[str] = None
+_nemo_data_prepared: set = set()
+def _get_nemo_venv() -> Tuple[str, dict]:
+    """Get or create a uv venv with nemo_skills installed.
+    Returns (venv_python_path, env_dict) reusable across calls.
diff -- test/registered/gb300/test_deepseek_v32_nvfp4.py
@@ -0,0 +1,82 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
diff -- test/registered/gb300/test_deepseek_v32.py
@@ -0,0 +1,79 @@
```

- Reviewed files:
  - tests: `python/sglang/test/accuracy_test_runner.py` modified +296/-3; `test/registered/gb300/test_deepseek_v32_nvfp4.py` added +82/-0; `test/registered/gb300/test_deepseek_v32.py` added +79/-0; `test/registered/gb300/test_qwen35_nvfp4.py` added +79/-0; `test/registered/gb300/test_qwen35_fp8.py` added +75/-0; `test/registered/gb300/test_glm5_nvfp4.py` added +71/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/run_combined_tests.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21448 - [Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode

- Link: https://github.com/sgl-project/sglang/pull/21448
- Status/date: merged / 2026-03-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `9b4dd274787c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +78/-8, 262 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: Co-authored-by: zhangxiaolei123456 ## Motivation Although https://github.com/sgl-project/sglang/pull/19670 and https://github.com/sgl-project/sglang/pull/21070 addressed some Qw...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +31/-1 (32 lines); hunks: -67,7 +67,7; -1038,6 +1038,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights, load_fused_expert_weights, touching `load_weights, load_fused_expert_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +31/-1 (32 lines); hunks: -67,7 +67,7; -1038,6 +1038,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights, load_fused_expert_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -67,7 +67,7 @@
-from sglang.srt.layers.utils import PPMissingLayer
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
@@ -1038,6 +1038,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            layer_id = get_layer_id(name)
+            if (
+                layer_id is not None
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +31/-1
- Risk and verification: The diff ships test coverage in `test/registered/unit/mem_cache/test_mamba_unittest.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21234 - [AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model

- Link: https://github.com/sgl-project/sglang/pull/21234
- Status/date: merged / 2026-03-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `e6071e60c097`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-0, 53 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Motivation Enable and validate Qwen3.5 MXFP4 model support on AMD GPUs This PR aims to preserve acceptable accuracy while improving serving performance versus the FP8 baselin...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +18/-0 (18 lines); hunks: -88,6 +88,7; -98,6 +99,7; symbols: forward, Qwen3_5ForCausalLM, __init__, load_fused_expert_weights, touching `forward, Qwen3_5ForCausalLM, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +18/-0 (18 lines); hunks: -88,6 +88,7; -98,6 +99,7; symbols: forward, Qwen3_5ForCausalLM, __init__, load_fused_expert_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -88,6 +88,7 @@
+    is_gfx95_supported,
@@ -98,6 +99,7 @@
+_is_gfx95 = is_gfx95_supported()
@@ -879,6 +881,14 @@ def forward(
+    if _is_gfx95:
+        packed_modules_mapping = {
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +18/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20864 - [Perf]Remove H2D for Qwen3.5 SpecV2

- Link: https://github.com/sgl-project/sglang/pull/20864
- Status/date: merged / 2026-03-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +17/-13, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR optimizes an inference path or backend selection. Title: "[Perf]Remove H2D for Qwen3.5 SpecV2". The diff centers on `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`. PR body context: ## Motivation This PR improves Qwen3.5 specV2 performance by removing unnecessary H2D overhead in the `prepare_v2_verify` path. ## Modifications 1. Use `torch.stack(...).to(torc...
- Key implementation: `python/sglang/srt/model_executor/forward_batch_info.py` modified +14/-8 (22 lines); hunks: -715,15 +715,21 @@ def _compute_spec_mrope_positions(; symbols: _compute_spec_mrope_positions, touching `_compute_spec_mrope_positions`; `python/sglang/srt/speculative/eagle_info_v2.py` modified +3/-5 (8 lines); hunks: -234,14 +234,12 @@ def prepare_for_v2_verify(; symbols: prepare_for_v2_verify, touching `prepare_for_v2_verify`.
- Code diff details:
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +14/-8 (22 lines); hunks: -715,15 +715,21 @@ def _compute_spec_mrope_positions(; symbols: _compute_spec_mrope_positions
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +3/-5 (8 lines); hunks: -234,14 +234,12 @@ def prepare_for_v2_verify(; symbols: prepare_for_v2_verify
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -715,15 +715,21 @@ def _compute_spec_mrope_positions(
-            mrope_deltas = [
-                (
-                    torch.tensor([0], dtype=torch.int64)
-                    if mm_inputs[i] is None
-                    else mm_inputs[i].mrope_position_delta.squeeze(0)
+            # Split text-only and mixed batches here because SpecV2 text-only batches can avoid an extra D2H.
diff -- python/sglang/srt/speculative/eagle_info_v2.py
@@ -234,14 +234,12 @@ def prepare_for_v2_verify(
-                batch.mamba_track_indices = torch.tensor(
+                batch.mamba_track_indices = torch.stack(
-                    ],
-                    dtype=torch.int64,
-                    device=device,
-                )
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/forward_batch_info.py` modified +14/-8; `python/sglang/srt/speculative/eagle_info_v2.py` modified +3/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21347 - [Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model

- Link: https://github.com/sgl-project/sglang/pull/21347
- Status/date: merged / 2026-04-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `2861596fc683`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-0, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Motivation Fixes #21093. Qwen3.5-4B (dense, `tie_word_embeddings=true`) with `--pp-size 2` produces garbage output. **Root cause:** `Qwen3_5ForConditionalGeneration` and `Qwe...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +22/-0 (22 lines); hunks: -1384,6 +1384,17 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; -1549,6 +1560,17 @@ def load_fused_expert_weights(; symbols: load_weights, load_fused_expert_weights, touching `load_weights, load_fused_expert_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +22/-0 (22 lines); hunks: -1384,6 +1384,17 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; -1549,6 +1560,17 @@ def load_fused_expert_weights(; symbols: load_weights, load_fused_expert_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -1384,6 +1384,17 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            if (
+                self.config.tie_word_embeddings
+                and self.pp_group.is_last_rank
+                and "model.embed_tokens.weight" in name
+            ):
+                if "lm_head.weight" in params_dict:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +22/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21849 - [VLM]: allow Qwen3.5 models for encoder disaggregation

- Link: https://github.com/sgl-project/sglang/pull/21849
- Status/date: merged / 2026-04-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +190/-3, 230 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[VLM]: allow Qwen3.5 models for encoder disaggregation". The diff centers on `python/sglang/srt/multimodal/processors/qwen_vl.py`, `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`. PR body context: ## Motivation Fixes #21805. SGLang already supports Qwen3.5 multimodal models in the runtime, but encoder disaggregation rejected them during servervstartup. This blocked valid...
- Key implementation: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -422,7 +422,7 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, touching `get_mm_data`; `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0 (184 lines); hunks: -33,6 +33,7; -813,6 +814,189 @@ def test_mmmu(self):; symbols: test_mmmu, TestEPDDisaggregationQwen35, setUpClass, start_encode, touching `test_mmmu, TestEPDDisaggregationQwen35, setUpClass`; `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2 (5 lines); hunks: -867,10 +867,11 @@ async def _process_mm_items(self, mm_items, modality):; symbols: _process_mm_items, touching `_process_mm_items`; `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -3326,6 +3326,8 @@ def _handle_encoder_disaggregation(self):; symbols: _handle_encoder_disaggregation, touching `_handle_encoder_disaggregation`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -422,7 +422,7 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data
  - `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0 (184 lines); hunks: -33,6 +33,7; -813,6 +814,189 @@ def test_mmmu(self):; symbols: test_mmmu, TestEPDDisaggregationQwen35, setUpClass, start_encode
  - `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2 (5 lines); hunks: -867,10 +867,11 @@ async def _process_mm_items(self, mm_items, modality):; symbols: _process_mm_items
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -3326,6 +3326,8 @@ def _handle_encoder_disaggregation(self):; symbols: _handle_encoder_disaggregation
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -422,7 +422,7 @@ def get_mm_data(self, prompt, embeddings, **kwargs):
-            self.model_type in ["qwen3_vl", "qwen3_vl_moe"]
+            self.model_type in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
diff -- test/registered/distributed/test_epd_disaggregation.py
@@ -33,6 +33,7 @@
+QWEN35_27B_MODEL = "Qwen/Qwen3.5-27B"
@@ -813,6 +814,189 @@ def test_mmmu(self):
+@unittest.skipIf(
+    is_in_ci(),
+    "Qwen3.5 EPD image/video test runs locally only",
+)
diff -- python/sglang/srt/disaggregation/encode_server.py
@@ -867,10 +867,11 @@ async def _process_mm_items(self, mm_items, modality):
-                self.model_type in ["qwen3_vl", "qwen3_vl_moe"]
+                self.model_type
+                in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
-                # For qwen3-vl models, we need to store the video timestamps
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1; `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2; `python/sglang/srt/server_args.py` modified +2/-0
  - tests: `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0
- Risk and verification: The diff ships test coverage in `test/registered/distributed/test_epd_disaggregation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21669 - [AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/21669
- Status/date: merged / 2026-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`; associated commits `ba78f6e0efb9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +408/-8, 538 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x". The diff centers on `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`. PR body context: ## Summary - Add `bench_one_batch` performance tests for **Qwen3.5-397B-A17B-FP8** on both MI325/MI300X and MI35x GPUs - Perf steps run **after** existing Qwen3.5 accuracy tests...
- Key implementation: `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py` added +139/-0 (139 lines); hunks: -0,0 +1,139; symbols: generate_simple_markdown_report, TestNightlyQwen35Fp8Performance, setUpClass, test_bench_qwen35_fp8, touching `generate_simple_markdown_report, TestNightlyQwen35Fp8Performance, setUpClass`; `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` added +139/-0 (139 lines); hunks: -0,0 +1,139; symbols: generate_simple_markdown_report, TestQwen35Fp8PerfMI35x, setUpClass, test_qwen35_fp8_perf, touching `generate_simple_markdown_report, TestQwen35Fp8PerfMI35x, setUpClass`; `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` modified +42/-1 (43 lines); hunks: -8,14 +8,20; -38,7 +44,7 @@ def setUpClass(cls):; symbols: setUpClass, tearDownClass, test_lm_eval, touching `setUpClass, tearDownClass, test_lm_eval`; `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py` modified +36/-3 (39 lines); hunks: -8,16 +8,21; -40,12 +45,12 @@ def setUpClass(cls):; symbols: setUpClass, test_lm_eval, touching `setUpClass, test_lm_eval`.
- Code diff details:
  - `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py` added +139/-0 (139 lines); hunks: -0,0 +1,139; symbols: generate_simple_markdown_report, TestNightlyQwen35Fp8Performance, setUpClass, test_bench_qwen35_fp8
  - `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` added +139/-0 (139 lines); hunks: -0,0 +1,139; symbols: generate_simple_markdown_report, TestQwen35Fp8PerfMI35x, setUpClass, test_qwen35_fp8_perf
  - `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` modified +42/-1 (43 lines); hunks: -8,14 +8,20; -38,7 +44,7 @@ def setUpClass(cls):; symbols: setUpClass, tearDownClass, test_lm_eval
  - `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py` modified +36/-3 (39 lines); hunks: -8,16 +8,21; -40,12 +45,12 @@ def setUpClass(cls):; symbols: setUpClass, test_lm_eval
- Key code excerpts:

```diff
diff -- test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py
@@ -0,0 +1,139 @@
+"""Nightly performance benchmark for Qwen3.5-397B-A17B FP8.
+Tests Qwen3.5-397B-A17B-FP8 (MoE, Hybrid Attention with Gated Delta Networks)
+on 8 GPUs with triton attention backend.
+Model path can be configured via environment variable:
+- QWEN35_FP8_MODEL_PATH: Path to Qwen3.5-FP8 model
+  (default: Qwen/Qwen3.5-397B-A17B-FP8)
diff -- test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py
@@ -0,0 +1,139 @@
+"""MI35x Nightly performance benchmark for Qwen3.5-397B-A17B FP8.
+Tests Qwen3.5-397B-A17B-FP8 (MoE, Hybrid Attention with Gated Delta Networks)
+on 8 GPUs with triton attention backend.
+Registry: nightly-perf-8-gpu-mi35x-qwen35-fp8 suite
+"""
+import os
diff -- test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py
@@ -8,14 +8,20 @@
```

- Reviewed files:
  - tests: `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py` added +139/-0; `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` added +139/-0; `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` modified +42/-1; `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py` modified +36/-3
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22145 - [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support)

- Link: https://github.com/sgl-project/sglang/pull/22145
- Status/date: merged / 2026-04-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +20/-8, 62 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support)". The diff centers on `python/sglang/srt/disaggregation/nixl/conn.py`. PR body context: ## Motivation NIXL disaggregated serving with heterogeneous TP (prefill TP ≠ decode TP) on non-MLA models hangs indefinitely due to two bugs in `nixl/conn.py`: 1. **Notification...
- Key implementation: `python/sglang/srt/disaggregation/nixl/conn.py` modified +20/-8 (28 lines); hunks: -477,25 +477,35 @@ def send_kvcache_slice(; -748,7 +758,9 @@ def add_transfer_request(; symbols: send_kvcache_slice, add_transfer_request, touching `send_kvcache_slice, add_transfer_request`.
- Code diff details:
  - `python/sglang/srt/disaggregation/nixl/conn.py` modified +20/-8 (28 lines); hunks: -477,25 +477,35 @@ def send_kvcache_slice(; -748,7 +758,9 @@ def add_transfer_request(; symbols: send_kvcache_slice, add_transfer_request
- Key code excerpts:

```diff
diff -- python/sglang/srt/disaggregation/nixl/conn.py
@@ -477,25 +477,35 @@ def send_kvcache_slice(
-        num_kv_heads = self.kv_args.kv_head_num
-        # Calculate head distribution
-        src_heads_per_rank = num_kv_heads
-        dst_heads_per_rank = num_kv_heads * prefill_tp_size // decode_tp_size
+        # Use total KV head count (not per-rank) for correct head distribution.
+        # Per-rank kv_head_num is max(1, total//tp) which loses info when total < tp.
```

- Reviewed files:
  - runtime: `python/sglang/srt/disaggregation/nixl/conn.py` modified +20/-8
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/nixl/conn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22240 - [Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5)

- Link: https://github.com/sgl-project/sglang/pull/22240
- Status/date: merged / 2026-04-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +143/-2, 207 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5)". The diff centers on `python/sglang/srt/disaggregation/nixl/conn.py`. PR body context: ## Motivation Depends on #22145. The Mooncake backend already supports Mamba state slice transfer under heterogeneous TP (prefill TP ≠ decode TP), but NIXL lacks this capability...
- Key implementation: `python/sglang/srt/disaggregation/nixl/conn.py` modified +143/-2 (145 lines); hunks: -84,6 +84,8 @@ class KVArgsRegisterInfo:; -93,6 +95,15 @@ def from_zmq(cls, msg: List[bytes]):; symbols: KVArgsRegisterInfo, from_zmq, _send_mamba_state, touching `KVArgsRegisterInfo, from_zmq, _send_mamba_state`.
- Code diff details:
  - `python/sglang/srt/disaggregation/nixl/conn.py` modified +143/-2 (145 lines); hunks: -84,6 +84,8 @@ class KVArgsRegisterInfo:; -93,6 +95,15 @@ def from_zmq(cls, msg: List[bytes]):; symbols: KVArgsRegisterInfo, from_zmq, _send_mamba_state
- Key code excerpts:

```diff
diff -- python/sglang/srt/disaggregation/nixl/conn.py
@@ -84,6 +84,8 @@ class KVArgsRegisterInfo:
+    dst_state_item_lens: list[int] = dataclasses.field(default_factory=list)
+    dst_state_dim_per_tensor: list[int] = dataclasses.field(default_factory=list)
@@ -93,6 +95,15 @@ def from_zmq(cls, msg: List[bytes]):
+        dst_state_item_lens = []
+        dst_state_dim_per_tensor = []
+        if len(msg) > 12 and len(msg[12]) > 0:
```

- Reviewed files:
  - runtime: `python/sglang/srt/disaggregation/nixl/conn.py` modified +143/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/nixl/conn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21692 - [Bugfix] [NPU] Qwen3.5 with quantization fix

- Link: https://github.com/sgl-project/sglang/pull/21692
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `cd373667cdfa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +29/-42, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] [NPU] Qwen3.5 with quantization fix". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Motivation Fix of https://github.com/sgl-project/sglang/issues/21676. After updating the qwen3.5 modeling and changing with to fused and with to fused , quantization no longe...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +3/-3 (6 lines); hunks: -881,7 +881,7 @@ def forward(; -1310,7 +1310,7 @@ def load_fused_expert_weights(; symbols: forward, Qwen3_5ForCausalLM, load_fused_expert_weights, Qwen3_5ForConditionalGeneration, touching `forward, Qwen3_5ForCausalLM, load_fused_expert_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +3/-3 (6 lines); hunks: -881,7 +881,7 @@ def forward(; -1310,7 +1310,7 @@ def load_fused_expert_weights(; symbols: forward, Qwen3_5ForCausalLM, load_fused_expert_weights, Qwen3_5ForConditionalGeneration
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -881,7 +881,7 @@ def forward(
-    if _is_gfx95:
+    if _is_gfx95 or _is_npu:
@@ -1310,7 +1310,7 @@ def load_fused_expert_weights(
-    if _is_gfx95:
+    if _is_gfx95 or _is_npu:
@@ -1447,7 +1447,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22399 - [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model

- Link: https://github.com/sgl-project/sglang/pull/22399
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_qwen35.py`; associated commits `46c2b7762765`
- Diff scope read: GitHub Pull Request files API returned 3 files, +82/-6, 131 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[CI] Add GLM-5.1 nightly tests and update Qwen3.5 model". The diff centers on `test/registered/8-gpu-models/test_qwen35.py`. PR body context: ## Summary - Add GLM-5.1 FP8 nightly test for H200/B200 (`nightly-8-gpu-common` suite) with TP8, TP8+DP8, and TP8+DP8+MTP variants - Update GB300 GLM-5 tests to GLM-5.1 model na...
- Key implementation: `test/registered/8-gpu-models/test_qwen35.py` modified +10/-3 (13 lines); hunks: -9,7 +9,7; -30,6 +30,7 @@ def test_qwen35(self):; symbols: TestQwen35, test_qwen35, touching `TestQwen35, test_qwen35`.
- Code diff details:
  - `test/registered/8-gpu-models/test_qwen35.py` modified +10/-3 (13 lines); hunks: -9,7 +9,7; -30,6 +30,7 @@ def test_qwen35(self):; symbols: TestQwen35, test_qwen35
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_qwen35.py
@@ -9,7 +9,7 @@
-QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"
+QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B-FP8"
@@ -30,6 +30,7 @@ def test_qwen35(self):
+        dp_args = ["--dp=8", "--enable-dp-attention"]
@@ -48,8 +49,14 @@ def test_qwen35(self):
-                extra_args=base_args + mtp_args,
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_qwen35.py` modified +10/-3
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22358 - Enable DFLASH support for additional model backends

- Link: https://github.com/sgl-project/sglang/pull/22358
- Status/date: merged / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +152/-5, 299 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "Enable DFLASH support for additional model backends". The diff centers on `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Summary Enable DFLASH for additional supported models from the z-lab collection: https://huggingface.co/collections/z-lab/dflash Based on #20547, landing this early to enable...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture, touching `forward, get_layer, get_input_embeddings`; `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head, touching `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings`; `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head, touching `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward`; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture, touching `__init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture
  - `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture
  - `python/sglang/srt/models/qwen3_vl.py` modified +16/-0 (16 lines); hunks: -1122,6 +1122,7 @@ def __init__(; -1246,19 +1247,34 @@ def forward(; symbols: __init__, forward, set_dflash_layers_to_capture, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -574,8 +574,15 @@ def forward(
-        hidden_states, residual = self.layer_communicator.prepare_attn(
-            hidden_states, residual, forward_batch
+        hidden_states, residual = (
+            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
+                hidden_states,
+                residual,
diff -- python/sglang/srt/models/kimi_k25.py
@@ -849,6 +849,30 @@ def set_eagle3_layers_to_capture(
+    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
+        """Set the layers to capture for DFLASH draft model training."""
+        if not hasattr(self.language_model, "set_dflash_layers_to_capture"):
+            raise AttributeError(
+                "language_model does not support DFLASH layer capture."
+            )
diff -- python/sglang/srt/models/qwen3_next.py
@@ -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-5; `python/sglang/srt/models/kimi_k25.py` modified +24/-0; `python/sglang/srt/models/qwen3_next.py` modified +20/-0; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0; `python/sglang/srt/models/qwen3_vl.py` modified +16/-0; `python/sglang/srt/models/gpt_oss.py` modified +15/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22312 - Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B

- Link: https://github.com/sgl-project/sglang/pull/22312
- Status/date: merged / 2026-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +272/-8, 346 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B". The diff centers on `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `test/registered/attention/test_gdn_noncontiguous_stride.py`. PR body context: ## Motivation This PR fixes #22311 Qwen3.5-27B takes a fallback BA path (after commit 5bdc07d974f6cf236fa765a685453ea5e587a838) that returns non-contiguous split views for a and...
- Key implementation: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +9/-6 (15 lines); hunks: -30,6 +30,7 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; -81,10 +82,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update, touching `fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update`; `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` modified +8/-2 (10 lines); hunks: -16,6 +16,8 @@ def fused_gdn_gating_kernel(; -26,8 +28,8 @@ def fused_gdn_gating_kernel(; symbols: fused_gdn_gating_kernel, fused_gdn_gating, touching `fused_gdn_gating_kernel, fused_gdn_gating`; `test/registered/attention/test_gdn_noncontiguous_stride.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: _make_noncontiguous_ab, TestFusedGdnGatingNonContiguous, _run_test, test_small, touching `_make_noncontiguous_ab, TestFusedGdnGatingNonContiguous, _run_test`.
- Code diff details:
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +9/-6 (15 lines); hunks: -30,6 +30,7 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; -81,10 +82,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update
  - `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` modified +8/-2 (10 lines); hunks: -16,6 +16,8 @@ def fused_gdn_gating_kernel(; -26,8 +28,8 @@ def fused_gdn_gating_kernel(; symbols: fused_gdn_gating_kernel, fused_gdn_gating
  - `test/registered/attention/test_gdn_noncontiguous_stride.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: _make_noncontiguous_ab, TestFusedGdnGatingNonContiguous, _run_test, test_small
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
@@ -30,6 +30,7 @@ def fused_sigmoid_gating_delta_rule_update_kernel(
+    stride_a,
@@ -81,10 +82,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(
-        p_a = a + (bos * HV + i_hv) * K + o_k
+        p_a = a + bos * stride_a + i_hv * K + o_k
-        p_a = a + bos * HV + i_hv
+        p_a = a + bos * stride_a + i_hv
diff -- python/sglang/srt/layers/attention/fla/fused_gdn_gating.py
@@ -16,6 +16,8 @@ def fused_gdn_gating_kernel(
+    stride_a,
+    stride_b,
@@ -26,8 +28,8 @@ def fused_gdn_gating_kernel(
-    blk_a = tl.load(a + off, mask=mask)
-    blk_b = tl.load(b + off, mask=mask)
+    blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
diff -- test/registered/attention/test_gdn_noncontiguous_stride.py
@@ -0,0 +1,255 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +9/-6; `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` modified +8/-2
  - tests: `test/registered/attention/test_gdn_noncontiguous_stride.py` added +255/-0
- Risk and verification: The diff ships test coverage in `test/registered/attention/test_gdn_noncontiguous_stride.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20736 - [AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8

- Link: https://github.com/sgl-project/sglang/pull/20736
- Status/date: merged / 2026-04-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_5.py`; associated commits `ea05ea5abed1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +218/-8, 383 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8". The diff centers on `python/sglang/srt/models/qwen3_5.py`. PR body context: ## Motivation Qwen2 MoE and Qwen3.5 MoE models use a **shared expert** in addition to routed experts. When `shared_expert_intermediate_size == moe_intermediate_size`, the shared...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +110/-3 (113 lines); hunks: -86,9 +86,11; -100,6 +102,8; symbols: __init__, _get_num_fused_shared_experts, get_embed_and_head, touching `__init__, _get_num_fused_shared_experts, get_embed_and_head`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +110/-3 (113 lines); hunks: -86,9 +86,11; -100,6 +102,8; symbols: __init__, _get_num_fused_shared_experts, get_embed_and_head
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -86,9 +86,11 @@
+    get_bool_env_var,
+    is_hip,
@@ -100,6 +102,8 @@
+_is_hip = is_hip()
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
@@ -528,6 +532,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +110/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22948 - [AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled

- Link: https://github.com/sgl-project/sglang/pull/22948
- Status/date: merged / 2026-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +17/-1, 39 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled". The diff centers on `python/sglang/srt/models/qwen2_moe.py`. PR body context: ## Motivation After shared expert fusion is enabled for Qwen3.5 models (as in #20736 ), MXFP4 model hits an issue: the shared expert in the checkpoint is based on BF16 but curre...
- Key implementation: `python/sglang/srt/models/qwen2_moe.py` modified +17/-1 (18 lines); hunks: -108,6 +108,7; -120,6 +121,20 @@ def can_fuse_shared_expert(; symbols: can_fuse_shared_expert, __init__, touching `can_fuse_shared_expert, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_moe.py` modified +17/-1 (18 lines); hunks: -108,6 +108,7; -120,6 +121,20 @@ def can_fuse_shared_expert(; symbols: can_fuse_shared_expert, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -108,6 +108,7 @@
+    quant_config: Optional[QuantizationConfig],
@@ -120,6 +121,20 @@ def can_fuse_shared_expert(
+    # If the shared expert is excluded from quantization (stored as FP32 in the
+    # checkpoint), fusing it into the quantized MoE weight tensor requires online
+    # quantization which is not supported. Disable fusion in this case.
+    if quant_config is not None:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_moe.py` modified +17/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22913 - test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6

- Link: https://github.com/sgl-project/sglang/pull/22913
- Status/date: merged / 2026-04-17
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`; associated commits `005209317888`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +184/-247, 448 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR extends deployment docs, tests, or CI coverage. Title: "test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6". The diff centers on `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`. PR body context: ## Summary - Splits `test/registered/4-gpu-models/test_qwen35_models.py` into separate files so the CI auto-partitioner can spread the load across separate partitions - Removes...
- Key implementation: `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k, touching `TestQwen35FP4MTPV2, setUpClass, tearDownClass`; `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` added +77/-0 (77 lines); hunks: -0,0 +1,77; symbols: TestQwen35FP4, test_gsm8k, touching `TestQwen35FP4, test_gsm8k`.
- Code diff details:
  - `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k
  - `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` added +77/-0 (77 lines); hunks: -0,0 +1,77; symbols: TestQwen35FP4, test_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py
@@ -0,0 +1,105 @@
+import unittest
+from types import SimpleNamespace
+import requests
+from sglang.srt.environ import envs
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
diff -- test/registered/4-gpu-models/test_qwen35_fp4_triton.py
@@ -0,0 +1,77 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+# This eval harness applies the chat_template, which is critical for qwen3.5
+# to get good accuracy on gsm8k
+from sglang.test.run_combined_tests import run_combined_tests
```

- Reviewed files:
  - tests: `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` added +105/-0; `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` added +77/-0
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`, `test/registered/4-gpu-models/test_qwen35_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23034 - docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs

- Link: https://github.com/sgl-project/sglang/pull/23034
- Status/date: merged / 2026-04-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 73 files, +2214/-215, 3198 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs". The diff centers on `docs_new/docs/advanced_features/separate_reasoning.mdx`, `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`. PR body context: ## Summary - **Add Qwen3.6 documentation**: New full deployment guide for Qwen3.6-35B-A3B (hybrid GDN + sparse MoE architecture) with JSX deployment snippet, covering MTP, tool...
- Key implementation: `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3 (5 lines); hunks: -207,7 +207,7 @@ print_highlight("==== Text ===="); -226,7 +226,7 @@ print_highlight("==== Original Output ===="); `docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2 (3 lines); hunks: -718,7 +718,7 @@ for tool_call in tool_calls:; -738,4 +738,3 @@ terminate_process(server_process); symbols: NewModelDetector, that, touching `NewModelDetector, that`; `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunks: -0,0 +1,509; `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunks: -0,0 +1,471.
- Code diff details:
  - `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3 (5 lines); hunks: -207,7 +207,7 @@ print_highlight("==== Text ===="); -226,7 +226,7 @@ print_highlight("==== Original Output ====")
  - `docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2 (3 lines); hunks: -718,7 +718,7 @@ for tool_call in tool_calls:; -738,4 +738,3 @@ terminate_process(server_process); symbols: NewModelDetector, that
  - `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunks: -0,0 +1,509
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunks: -0,0 +1,471
  - `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0 (299 lines); hunks: -0,0 +1,299; symbols: per_token_group_quant_8bit, add
- Key code excerpts:

```diff
diff -- docs_new/docs/advanced_features/separate_reasoning.mdx
@@ -207,7 +207,7 @@ print_highlight("==== Text ====")
-The reasoning separation is enable by default when specify .
+The reasoning separation is enable by default when specify .
@@ -226,7 +226,7 @@ print_highlight("==== Original Output ====")
-### SGLang Native API
+### SGLang Native API
@@ -315,4 +315,3 @@ llm.shutdown()
diff -- docs_new/docs/advanced_features/tool_parser.mdx
@@ -718,7 +718,7 @@ for tool_call in tool_calls:
-> **Note:**
+> **Note:**
@@ -738,4 +738,3 @@ terminate_process(server_process)
diff -- docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx
@@ -0,0 +1,509 @@
+---
+title: "DP, DPA and SGLang DP Router"
+metatags:
```

- Reviewed files:
  - docs: `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3; `docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2; `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0; `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0; `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0; `docs_new/docs/advanced_features/server_arguments.mdx` modified +241/-45
- Risk and verification: This is mostly docs/examples in `docs_new/.gitignore`, `docs_new/cards/logos/google.png`, `docs_new/cards/logos/mova.png`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22431 - Fix Qwen3.5 video processing when passing video_data in "processor_output" format

- Link: https://github.com/sgl-project/sglang/pull/22431
- Status/date: merged / 2026-04-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix Qwen3.5 video processing when passing video_data in "processor_output" format". The diff centers on `python/sglang/srt/multimodal/processors/qwen_vl.py`. PR body context: ## Motivation Currently, the video preprocessing function of Qwen3.5 requires two return values: https://github.com/sgl-project/sglang/blob/ef6bfc1197ab45290e33941881f23c39fbf30...
- Key implementation: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -162,7 +162,7 @@ async def preprocess_video(; symbols: preprocess_video, touching `preprocess_video`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -162,7 +162,7 @@ async def preprocess_video(; symbols: preprocess_video
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -162,7 +162,7 @@ async def preprocess_video(
-        return vr
+        return vr, None
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/qwen_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22908 - [AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict.

- Link: https://github.com/sgl-project/sglang/pull/22908
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-4, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict.". The diff centers on `python/sglang/srt/server_args.py`. PR body context: ## Motivation When using speculative decoding (e.g., EAGLE) with `Qwen3_5MoeForConditionalGeneration`, SGLang raises a hard `ValueError` in `_handle_mamba_radix_cache()`: This b...
- Key implementation: `python/sglang/srt/server_args.py` modified +14/-4 (18 lines); hunks: -2326,10 +2326,20 @@ def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache, _handle_sampling_backend, touching `_handle_mamba_radix_cache, _handle_sampling_backend`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +14/-4 (18 lines); hunks: -2326,10 +2326,20 @@ def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache, _handle_sampling_backend
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -2326,10 +2326,20 @@ def _handle_mamba_radix_cache(
-                    raise ValueError(
-                        f"Speculative decoding for {model_arch} is not compatible with radix cache when using --mamba-scheduler-strategy no_buffer."
-                        "To use radix cache with speculative decoding, please use --mamba-scheduler-strategy extra_buffer and set SGLANG_ENABLE_SPEC_V2=1."
-                    )
+                    if is_hip():
+                        # On ROCm, extra_buffer is unsupported.
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +14/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22493 - Add MambaPool kvcache offloading during retraction

- Link: https://github.com/sgl-project/sglang/pull/22493
- Status/date: merged / 2026-04-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +193/-16, 311 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "Add MambaPool kvcache offloading during retraction". The diff centers on `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py`. PR body context: ## Motivation Mamba-hybrid models (e.g. Qwen3.5-397B-A17B) maintain SSM state (conv + temporal buffers) in MambaPool separate from the attention KV cache. During request retract...
- Key implementation: `test/registered/unit/mem_cache/test_mamba_unittest.py` modified +123/-0 (123 lines); hunks: -388,6 +388,129 @@ def make_dummy_req():; symbols: make_dummy_req, test_mamba_pool_cpu_offload, test_hybrid_kv_pool_cpu_offload, test_insert_prev_prefix_len, touching `make_dummy_req, test_mamba_pool_cpu_offload, test_hybrid_kv_pool_cpu_offload`; `python/sglang/srt/mem_cache/memory_pool.py` modified +43/-6 (49 lines); hunks: -388,6 +388,28 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[to...; -728,10 +750,10 @@ def set_kv_buffer(; symbols: fork_from, get_cpu_copy, load_cpu_copy, get_contiguous_buf_infos, touching `fork_from, get_cpu_copy, load_cpu_copy`; `python/sglang/srt/mem_cache/allocator.py` modified +8/-8 (16 lines); hunks: -164,11 +164,11 @@ def free(self, free_index: torch.Tensor):; -512,8 +512,8 @@ def clear(self):; symbols: free, get_cpu_copy, load_cpu_copy, touching `free, get_cpu_copy, load_cpu_copy`; `python/sglang/srt/managers/scheduler.py` modified +11/-0 (11 lines); hunks: -2681,11 +2681,20 @@ def update_running_batch(self, batch: ScheduleBatch) ->...; -2715,6 +2724,8 @@ def update_running_batch(self, batch: ScheduleBatch) -> Op...; symbols: update_running_batch, touching `update_running_batch`.
- Code diff details:
  - `test/registered/unit/mem_cache/test_mamba_unittest.py` modified +123/-0 (123 lines); hunks: -388,6 +388,129 @@ def make_dummy_req():; symbols: make_dummy_req, test_mamba_pool_cpu_offload, test_hybrid_kv_pool_cpu_offload, test_insert_prev_prefix_len
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +43/-6 (49 lines); hunks: -388,6 +388,28 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[to...; -728,10 +750,10 @@ def set_kv_buffer(; symbols: fork_from, get_cpu_copy, load_cpu_copy, get_contiguous_buf_infos
  - `python/sglang/srt/mem_cache/allocator.py` modified +8/-8 (16 lines); hunks: -164,11 +164,11 @@ def free(self, free_index: torch.Tensor):; -512,8 +512,8 @@ def clear(self):; symbols: free, get_cpu_copy, load_cpu_copy
  - `python/sglang/srt/managers/scheduler.py` modified +11/-0 (11 lines); hunks: -2681,11 +2681,20 @@ def update_running_batch(self, batch: ScheduleBatch) ->...; -2715,6 +2724,8 @@ def update_running_batch(self, batch: ScheduleBatch) -> Op...; symbols: update_running_batch
  - `python/sglang/srt/managers/schedule_batch.py` modified +8/-2 (10 lines); hunks: -1241,13 +1241,19 @@ def offload_kv_cache(self, req_to_token_pool, token_to_k...; symbols: offload_kv_cache, load_kv_cache, log_time_stats
- Key code excerpts:

```diff
diff -- test/registered/unit/mem_cache/test_mamba_unittest.py
@@ -388,6 +388,129 @@ def make_dummy_req():
+    def test_mamba_pool_cpu_offload(self):
+        """MambaPool.get_cpu_copy / load_cpu_copy round-trips conv and temporal state."""
+        _, _, req_to_token_pool, _ = self._setup_tree_and_allocator()
+        mamba_pool = req_to_token_pool.mamba_pool
+        n = 3
+        indices = mamba_pool.alloc(n)
diff -- python/sglang/srt/mem_cache/memory_pool.py
@@ -388,6 +388,28 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:
+    def get_cpu_copy(self, indices, **kwargs):
+        torch.cuda.synchronize()
+        conv_cpu = [
+            conv[:, indices].to("cpu", non_blocking=True)
+            for conv in self.mamba_cache.conv
+        ]
diff -- python/sglang/srt/mem_cache/allocator.py
@@ -164,11 +164,11 @@ def free(self, free_index: torch.Tensor):
```

- Reviewed files:
  - tests: `test/registered/unit/mem_cache/test_mamba_unittest.py` modified +123/-0
  - runtime: `python/sglang/srt/mem_cache/memory_pool.py` modified +43/-6; `python/sglang/srt/mem_cache/allocator.py` modified +8/-8; `python/sglang/srt/managers/scheduler.py` modified +11/-0; `python/sglang/srt/managers/schedule_batch.py` modified +8/-2
- Risk and verification: The diff ships test coverage in `test/registered/unit/mem_cache/test_mamba_unittest.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- Link: https://github.com/sgl-project/sglang/pull/23474
- Status/date: open / 2026-04-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +284/-8, 330 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models". The diff centers on `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`. PR body context: ## Motivation Fixes #23150. `--cpu-offload-gb > 0` was broken on hybrid linear-attention models (Qwen3-Next, Qwen3.5, Kimi-Linear): the first `/v1/chat/completions` request rais...
- Key implementation: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent, touching `_TiedChild, __init__, forward`; `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward, touching `maybe_offload_to_cpu, forward`.
- Code diff details:
  - `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent
  - `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward
- Key code excerpts:

```diff
diff -- test/registered/unit/utils/test_offloader_tied_params.py
@@ -0,0 +1,199 @@
+"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).
+Two failure modes caused the Qwen3-Next / Qwen3.5 CPU-offload regression:
+1. **Tied parameters**: a single nn.Parameter is registered under both a parent
+   and a child module (Qwen3GatedDeltaNet + RadixLinearAttention share
+   ``A_log`` / ``dt_bias``). state_dict() then lists the same tensor under
+   multiple keys, and functional_call(..., tie_weights=True) rejects it when
diff -- python/sglang/srt/utils/offloader.py
@@ -1,7 +1,7 @@
-from typing import Callable, Generator, List, Optional
+from typing import Callable, Dict, Generator, List, Optional
@@ -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
+        # Record tensor views that alias each parameter's *original* storage
+        # BEFORE we rebind .data to pinned CPU memory. Some hybrid linear-attn
+        # models (e.g. Qwen3-Next) cache such views, which would otherwise point
```

- Reviewed files:
  - tests: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0
  - runtime: `python/sglang/srt/utils/offloader.py` modified +85/-8
- Risk and verification: The diff ships test coverage in `test/registered/unit/utils/test_offloader_tied_params.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23467 - fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert

- Link: https://github.com/sgl-project/sglang/pull/23467
- Status/date: merged / 2026-04-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +31/-4, 63 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert". The diff centers on `python/sglang/srt/layers/quantization/utils.py`. PR body context: ## Summary - `is_layer_skipped` uses naive substring match (`ignored in prefix`) on `modules_to_not_convert` entries, which silently fires when an entry is a prefix-substring of...
- Key implementation: `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunks: -43,6 +43,28 @@ def __getattr__(self, name):; -56,16 +78,19 @@ def is_layer_skipped(; symbols: __getattr__, _module_path_match, is_layer_skipped, touching `__getattr__, _module_path_match, is_layer_skipped`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunks: -43,6 +43,28 @@ def __getattr__(self, name):; -56,16 +78,19 @@ def is_layer_skipped(; symbols: __getattr__, _module_path_match, is_layer_skipped
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/utils.py
@@ -43,6 +43,28 @@ def __getattr__(self, name):
+def _module_path_match(ignored: str, prefix: str) -> bool:
+    # Match on dotted module-path boundaries so that `mlp.gate` does NOT
+    # match `mlp.gate_up_proj`. Needed for quant configs (e.g. Qwen3.6-FP8)
+    # whose `modules_to_not_convert` lists MoE-template names like `mlp.gate`
+    # that collide with fused dense MLP names by plain substring.
+    if ignored == prefix:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/utils.py` modified +31/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.
