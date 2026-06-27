# sglang MiMo V2 Flash Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2-Flash.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` | [#23851](https://github.com/sgl-project/sglang/pull/23851), [#23936](https://github.com/sgl-project/sglang/pull/23936), [#23945](https://github.com/sgl-project/sglang/pull/23945), [#24983](https://github.com/sgl-project/sglang/pull/24983), [#25359](https://github.com/sgl-project/sglang/pull/25359), [#29253](https://github.com/sgl-project/sglang/pull/29253) |
| `docs_new/src/snippets/autoregressive/mimo-v2-flash-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` | [#23851](https://github.com/sgl-project/sglang/pull/23851), [#23936](https://github.com/sgl-project/sglang/pull/23936), [#23945](https://github.com/sgl-project/sglang/pull/23945), [#25359](https://github.com/sgl-project/sglang/pull/25359), [#27668](https://github.com/sgl-project/sglang/pull/27668), [#29253](https://github.com/sgl-project/sglang/pull/29253) |
| `python/sglang/srt/entrypoints/openai/transcription_adapters/mimo_v2_asr.py` | [#26278](https://github.com/sgl-project/sglang/pull/26278) |
| `python/sglang/srt/function_call/mimo_detector.py` | [#15207](https://github.com/sgl-project/sglang/pull/15207) |
| `python/sglang/srt/models/mimo.py` | [#6059](https://github.com/sgl-project/sglang/pull/6059) |
| `python/sglang/srt/models/mimo_audio.py` | [#23811](https://github.com/sgl-project/sglang/pull/23811), [#26278](https://github.com/sgl-project/sglang/pull/26278) |
| `python/sglang/srt/models/mimo_mtp.py` | [#6059](https://github.com/sgl-project/sglang/pull/6059), [#7370](https://github.com/sgl-project/sglang/pull/7370) |
| `python/sglang/srt/models/mimo_v2.py` | [#23808](https://github.com/sgl-project/sglang/pull/23808), [#23811](https://github.com/sgl-project/sglang/pull/23811), [#24931](https://github.com/sgl-project/sglang/pull/24931), [#25455](https://github.com/sgl-project/sglang/pull/25455), [#26278](https://github.com/sgl-project/sglang/pull/26278) |
| `python/sglang/srt/models/mimo_v2_asr.py` | [#26278](https://github.com/sgl-project/sglang/pull/26278) |
| `python/sglang/srt/models/mimo_v2_nextn.py` | [#23808](https://github.com/sgl-project/sglang/pull/23808), [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `python/sglang/srt/models/mimo_vl.py` | [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `python/sglang/srt/multimodal/processors/mimo_audio.py` | [#26278](https://github.com/sgl-project/sglang/pull/26278) |
| `python/sglang/srt/multimodal/processors/mimo_v2.py` | [#23811](https://github.com/sgl-project/sglang/pull/23811), [#24931](https://github.com/sgl-project/sglang/pull/24931), [#25588](https://github.com/sgl-project/sglang/pull/25588), [#26278](https://github.com/sgl-project/sglang/pull/26278) |
| `python/sglang/srt/multimodal/processors/mimo_v2_asr.py` | [#26278](https://github.com/sgl-project/sglang/pull/26278) |
| `test/manual/ascend/llm_models/test_npu_mimo_v2_flash.py` | [#28223](https://github.com/sgl-project/sglang/pull/28223) |
| `test/registered/ascend/llm_models/test_npu_mimo_7b_rl.py` | no direct PR-number commit |
| `test/registered/ascend/vlm_models/test_npu_mimo_vl_7b_rl.py` | no direct PR-number commit |
| `test/registered/models_e2e/test_mimo_v2.py` | [#27378](https://github.com/sgl-project/sglang/pull/27378) |
| `test/registered/models_e2e/test_mimo_v2_flash.py` | [#27378](https://github.com/sgl-project/sglang/pull/27378) |

## PR Coverage Summary

- Git-traced PRs: 18
- Extra PRs preserved from existing docs: 13
- Total PRs in this document: 31
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
| 2026-05-11 | [#24983](https://github.com/sgl-project/sglang/pull/24983) | merged | Update MiMo V2.5 cookbook image to nightly | `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` |
| 2026-05-18 | [#24931](https://github.com/sgl-project/sglang/pull/24931) | merged | feat(mimo-v2): add EPD disaggregation support | `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_v2.py` |
| 2026-05-19 | [#25588](https://github.com/sgl-project/sglang/pull/25588) | merged | perf(mimo-v2-epd): enable GPU image preprocess and parallel video decode | `python/sglang/srt/multimodal/processors/mimo_v2.py` |
| 2026-05-20 | [#25359](https://github.com/sgl-project/sglang/pull/25359) | merged | [Docs] MiMo-V2.5 cookbook: B200 benchmarks + multi-layer EAGLE acceptance profile + long-context reference | `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` |
| 2026-05-22 | [#24751](https://github.com/sgl-project/sglang/pull/24751) | merged | fix(mm): make multimodal data loading non-blocking to prevent health check stalls | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-05-26 | [#25964](https://github.com/sgl-project/sglang/pull/25964) | merged | [EPD] Cross-request batching for image/audio encoder | `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/environ.py` |
| 2026-05-28 | [#26610](https://github.com/sgl-project/sglang/pull/26610) | merged | test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit) | `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` |
| 2026-05-29 | [#26673](https://github.com/sgl-project/sglang/pull/26673) | merged | [refactor] remove unused op_mlp | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-06-08 | [#27512](https://github.com/sgl-project/sglang/pull/27512) | merged | [Spec] Clamp multimodal pad sentinels in spec-v2 draft prefill embedding | `python/sglang/srt/models/mimo_v2_nextn.py`, `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` |
| 2026-06-10 | [#25455](https://github.com/sgl-project/sglang/pull/25455) | merged | [NPU] MiMo-V2-Flash Adaptation | `python/sglang/srt/models/mimo_v2.py` |
| 2026-06-10 | [#27668](https://github.com/sgl-project/sglang/pull/27668) | merged | Fix MiMo-V2.5-Pro DP-attention dp size in cookbook deployment snippet | `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` |
| 2026-06-11 | [#26278](https://github.com/sgl-project/sglang/pull/26278) | merged | Support MiMo v2 ASR | `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/multimodal/processors/mimo_audio.py`, `python/sglang/srt/multimodal/processors/mimo_v2_asr.py` |
| 2026-06-11 | [#27964](https://github.com/sgl-project/sglang/pull/27964) | merged | [Spec] Retire Spec V1 | `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py` |
| 2026-06-13 | [#27378](https://github.com/sgl-project/sglang/pull/27378) | merged | feat: Support HiCache for MiMo-V2 models (1/N) | `test/registered/models_e2e/test_mimo_v2.py`, `test/registered/models_e2e/test_mimo_v2_flash.py` |
| 2026-06-15 | [#28223](https://github.com/sgl-project/sglang/pull/28223) | merged | [NPU] Add MiMo-V2-Flash manual testcases | `test/manual/ascend/llm_models/test_npu_mimo_v2_flash.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-25 | [#29253](https://github.com/sgl-project/sglang/pull/29253) | merged | Add MiMo V2.5 Blackwell vision FA4 recipe | `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`, `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` |

## Per-PR Diff Audit Cards

### PR #6059 - Support XiaomiMiMo inference with mtp

- Link: https://github.com/sgl-project/sglang/pull/6059
- Status/date: merged / 2025-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo.py`, `python/sglang/srt/models/mimo_mtp.py`; associated commits `a6ae3af15e84`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +344/-6, 388 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support XiaomiMiMo inference with mtp"; model line: MiMo V2 Flash; category: model support/runtime entry; main diff: `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py`; technical summary: Covers "Support XiaomiMiMo inference with mtp"; the main implementation surface is `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; associated commits `e458a9248fef`; preserved from an explicit existing history/skill citation
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; associated commits `6c7b2421816c`; preserved from an explicit existing history/skill citation
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
- Trace source: preserved from an explicit existing history/skill citation
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py`, `python/sglang/srt/models/mimo_vl.py`, `python/sglang/srt/multimodal/processors/mimo_v2.py`; associated commits `651af06a0b5e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +4369/-87, 4885 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Xiaomi MiMo-V2.5 day0 support"; model line: MiMo V2 Flash; category: model support/runtime entry; main diff: `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_vl.py`; technical summary: Covers "[Feature] Xiaomi MiMo-V2.5 day0 support"; the main implementation surface is `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Risk and verification: The diff ships test coverage in `python/sglang/test/server_fixtures/mmmu_fixture.py`, `test/registered/8-gpu-models/test_mimo_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24983 - Update MiMo V2.5 cookbook image to nightly

- Link: https://github.com/sgl-project/sglang/pull/24983
- Status/date: merged / 2026-05-11
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`; associated commits `d9cb38012ee7`
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-4, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update MiMo V2.5 cookbook image to nightly"; model line: MiMo V2 Flash; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`; technical summary: Covers "Update MiMo V2.5 cookbook image to nightly"; the main implementation surface is `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +4/-4 (8 lines); hunks: -58,10 +58,10 @@ Refer to the [official SGLang installation guide](../../../d....
- Code diff details:
  - `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +4/-4 (8 lines); hunks: -58,10 +58,10 @@ Refer to the [official SGLang installation guide](../../../d...
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx
@@ -58,10 +58,10 @@ Refer to the [official SGLang installation guide](../../../docs/get-started/inst
-| **MiMo-V2.5 (310B)** | H100 / H200 (Hopper, CUDA 12.9) | `lmsysorg/sglang:dev-mimo-v2.5` |
-| **MiMo-V2.5 (310B)** | B200 / GB300 (Blackwell, CUDA 13.0) | `lmsysorg/sglang:dev-cu13-mimo-v2.5` |
-| **MiMo-V2.5-Pro (1.02T)** | H100 / H200 (Hopper, CUDA 12.9) | `lmsysorg/sglang:dev-mimo-v2.5-pro` |
-| **MiMo-V2.5-Pro (1.02T)** | B200 / GB300 (Blackwell, CUDA 13.0) | `lmsysorg/sglang:dev-cu13-mimo-v2.5-pro` |
+| **MiMo-V2.5 (310B)** | H100 / H200 (Hopper, CUDA 12.9) | `lmsysorg/sglang:nightly-dev-20260511-044bb88a` |
+| **MiMo-V2.5 (310B)** | B200 / GB300 (Blackwell, CUDA 13.0) | `lmsysorg/sglang:nightly-dev-cu13-20260511-044bb88a` |
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +4/-4
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #24931 - feat(mimo-v2): add EPD disaggregation support

- Link: https://github.com/sgl-project/sglang/pull/24931
- Status/date: merged / 2026-05-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/multimodal/processors/mimo_v2.py`; associated commits `784fe7e99b80`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +961/-289, 1710 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(mimo-v2): add EPD disaggregation support"; model line: MiMo V2 Flash; category: model support/runtime entry; main diff: `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_v2.py`; technical summary: Covers "feat(mimo-v2): add EPD disaggregation support"; the main implementation surface is `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +559/-110 (669 lines); hunks: -268,6 +268,49 @@ def __post_init__(self):; -445,6 +488,207 @@ def __init__(; symbols: __post_init__, _decode_frames_and_timestamps, _ffprobe_has_audio, MiMoProcessor, touching `__post_init__, _decode_frames_and_timestamps, _ffprobe_has_audio`; `python/sglang/srt/models/mimo_v2.py` modified +148/-28 (176 lines); hunks: -68,7 +68,11; -1007,6 +1011,11 @@ class MiMoV2ForCausalLM(nn.Module):; symbols: MiMoV2ForCausalLM, __init__, routed_experts_weights_of_layer, touching `MiMoV2ForCausalLM, __init__, routed_experts_weights_of_layer`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +559/-110 (669 lines); hunks: -268,6 +268,49 @@ def __post_init__(self):; -445,6 +488,207 @@ def __init__(; symbols: __post_init__, _decode_frames_and_timestamps, _ffprobe_has_audio, MiMoProcessor
  - `python/sglang/srt/models/mimo_v2.py` modified +148/-28 (176 lines); hunks: -68,7 +68,11; -1007,6 +1011,11 @@ class MiMoV2ForCausalLM(nn.Module):; symbols: MiMoV2ForCausalLM, __init__, routed_experts_weights_of_layer
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/mimo_v2.py
@@ -268,6 +268,49 @@ def __post_init__(self):
+def _decode_frames_and_timestamps(vdw, ele):
+    # Shared E/D frame-sampling recipe: smart_nframes + linspace + permute.
+    total_frames, video_fps = len(vdw), vdw.avg_fps
+    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
+    idx = list(np.unique(np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)))
+    video_tensor = vdw.get_frames_as_tensor(idx).permute(0, 3, 1, 2).float()
diff -- python/sglang/srt/models/mimo_v2.py
@@ -68,7 +68,11 @@
-from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
+    MultimodalInputs,
+)
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +559/-110; `python/sglang/srt/models/mimo_v2.py` modified +148/-28
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/encode_receiver.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/managers/mm_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25588 - perf(mimo-v2-epd): enable GPU image preprocess and parallel video decode

- Link: https://github.com/sgl-project/sglang/pull/25588
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/mimo_v2.py`; associated commits `f0763859edbb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +75/-8, 188 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "perf(mimo-v2-epd): enable GPU image preprocess and parallel video decode"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `python/sglang/srt/multimodal/processors/mimo_v2.py`; technical summary: Covers "perf(mimo-v2-epd): enable GPU image preprocess and parallel video decode"; the main implementation surface is `python/sglang/srt/multimodal/processors/mimo_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +26/-2 (28 lines); hunks: -359,11 +359,13 @@ def __init__(; -546,6 +548,20 @@ def _as_dict(obj):; symbols: __init__, _as_dict, _load_video_for_encoder, preprocess_for_encoder, touching `__init__, _as_dict, _load_video_for_encoder`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +26/-2 (28 lines); hunks: -359,11 +359,13 @@ def __init__(; -546,6 +548,20 @@ def _as_dict(obj):; symbols: __init__, _as_dict, _load_video_for_encoder, preprocess_for_encoder
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/mimo_v2.py
@@ -359,11 +359,13 @@ def __init__(
+        video_decode_num_threads=0,
+        self.video_decode_num_threads = video_decode_num_threads
@@ -546,6 +548,20 @@ def _as_dict(obj):
+        image_cfg = (mm_config or {}).get("image", {})
+        if "device" in image_cfg:
+            kwargs["device"] = image_cfg["device"]
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +26/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/utils/video_decoder.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25359 - [Docs] MiMo-V2.5 cookbook: B200 benchmarks + multi-layer EAGLE acceptance profile + long-context reference

- Link: https://github.com/sgl-project/sglang/pull/25359
- Status/date: merged / 2026-05-20
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; associated commits `52eebc82aed2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +194/-12, 272 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] MiMo-V2.5 cookbook: B200 benchmarks + multi-layer EAGLE acceptance profile + long-context reference"; model line: MiMo V2 Flash; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; technical summary: Covers "[Docs] MiMo-V2.5 cookbook: B200 benchmarks + multi-layer EAGLE acceptance profile + long-context reference"; the main implementation surface is `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +191/-9 (200 lines); hunks: -83,13 +83,13 @@ import { MiMoV25Deployment } from '/src/snippets/autoregress...; -385,9 +385,11 @@ python3 -m sglang.test.run_eval \; symbols: number, touching `number`; `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +3/-3 (6 lines); hunks: -328,15 +328,15 @@ export const MiMoV25Deployment = () => {; -348,7 +348,7 @@ export const MiMoV25Deployment = () => {.
- Code diff details:
  - `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +191/-9 (200 lines); hunks: -83,13 +83,13 @@ import { MiMoV25Deployment } from '/src/snippets/autoregress...; -385,9 +385,11 @@ python3 -m sglang.test.run_eval \; symbols: number
  - `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +3/-3 (6 lines); hunks: -328,15 +328,15 @@ export const MiMoV25Deployment = () => {; -348,7 +348,7 @@ export const MiMoV25Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx
@@ -83,13 +83,13 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressive/mimo-v25-deploym
-- EAGLE speculative decoding (3 steps, topk=1) typically yields a 2–3× decode speedup. Requires `SGLANG_ENABLE_SPEC_V2=1`; on Hopper also pass `--enable-multi-layer-eagle`.
+- EAGLE speculative decoding (3 steps, topk=1) typically yields a 2–3× decode speedup. Requires `SGLANG_ENABLE_SPEC_V2=1` and `--enable-multi-layer-eagle` (both Hopper and Blackwe
-- EAGLE MTP uses the checkpoint's MTP weights. For H100/H200, enable `SGLANG_ENABLE_SPEC_V2=1`, `--speculative-algorithm EAGLE`, and `--enable-multi-layer-eagle`.
+- EAGLE MTP uses the checkpoint's MTP weights. Enable with `SGLANG_ENABLE_SPEC_V2=1`, `--speculative-algorithm EAGLE`, and `--enable-multi-layer-eagle` (both Hopper and Blackwell)
@@ -385,9 +385,11 @@ python3 -m sglang.test.run_eval \
-  - MiMo-V2.5-Pro (FP8)
diff -- docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx
@@ -328,15 +328,15 @@ export const MiMoV25Deployment = () => {
-        flags.push(`  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}'`);
+        flags.push(`  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}'`);
-        flags.push(`  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}'`);
+        flags.push(`  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}'`);
@@ -348,7 +348,7 @@ export const MiMoV25Deployment = () => {
-      if (!blackwell) flags.push("  --enable-multi-layer-eagle");
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +191/-9; `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +3/-3
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #24751 - fix(mm): make multimodal data loading non-blocking to prevent health check stalls

- Link: https://github.com/sgl-project/sglang/pull/24751
- Status/date: merged / 2026-05-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +45/-44, 401 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(mm): make multimodal data loading non-blocking to prevent health check stalls"; model line: MiMo V2 Flash; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`; technical summary: Covers "fix(mm): make multimodal data loading non-blocking to prevent health check stalls"; the main implementation surface is `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data, touching `_process_loaded_mm_data, load_mm_data`; `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async, touching `_process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async`; `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async, touching `_process_special_format, process_mm_data_async`; `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async, touching `__init__, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1 (2 lines); hunks: -29,7 +29,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -1,3 +1,4 @@
+import asyncio
@@ -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):
-    def load_mm_data(
+    async def load_mm_data(
@@ -772,7 +773,7 @@ def load_mm_data(
-            return self.legacy_load_mm_data(
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -310,7 +310,7 @@ async def _process_special_format(
-            base_output = self.load_mm_data(
+            base_output = await self.load_mm_data(
@@ -423,7 +423,7 @@ async def process_qwen_mm_data_async(
-        base_output = self.load_mm_data(
+        base_output = await self.load_mm_data(
@@ -644,7 +644,7 @@ async def process_internlm2_mm_data_async(
diff -- python/sglang/srt/multimodal/processors/minicpm.py
@@ -118,7 +118,7 @@ async def _process_special_format(
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7; `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3; `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2; `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_vl_v2.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/clip.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25964 - [EPD] Cross-request batching for image/audio encoder

- Link: https://github.com/sgl-project/sglang/pull/25964
- Status/date: merged / 2026-05-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +505/-66, 746 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[EPD] Cross-request batching for image/audio encoder"; model line: MiMo V2 Flash; category: model implementation change; main diff: `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/environ.py`; technical summary: Covers "[EPD] Cross-request batching for image/audio encoder"; the main implementation surface is `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/environ.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +2/-3 (5 lines); hunks: -28,6 +28,7; -1815,9 +1816,7 @@ def __init__(self, hf_config, server_args, _processor, *ar...; symbols: __init__, touching `__init__`; `python/sglang/srt/disaggregation/encode_server.py` modified +500/-63 (563 lines); hunks: -1,12 +1,14; -78,9 +80,12; symbols: MMError, __init__, _infer_embedding_dims, _resolve_audio_sr, touching `MMError, __init__, _infer_embedding_dims`; `python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunks: -708,6 +708,9 @@ class Envs:; symbols: Envs, touching `Envs`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +2/-3 (5 lines); hunks: -28,6 +28,7; -1815,9 +1816,7 @@ def __init__(self, hf_config, server_args, _processor, *ar...; symbols: __init__
  - `python/sglang/srt/disaggregation/encode_server.py` modified +500/-63 (563 lines); hunks: -1,12 +1,14; -78,9 +80,12; symbols: MMError, __init__, _infer_embedding_dims, _resolve_audio_sr
  - `python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunks: -708,6 +708,9 @@ class Envs:; symbols: Envs
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/mimo_v2.py
@@ -28,6 +28,7 @@
+from sglang.srt.environ import envs
@@ -1815,9 +1816,7 @@ def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
-        self.use_image_processor_gpu = (
-            int(os.getenv("SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU", "0")) == 1
-        )
+        self.use_image_processor_gpu = envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get()
diff -- python/sglang/srt/disaggregation/encode_server.py
@@ -1,12 +1,14 @@
+import contextlib
+from collections import defaultdict
@@ -78,9 +80,12 @@
-use_image_processor_gpu = (
-    int(os.getenv("SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU", "0")) == 1
-)
diff -- python/sglang/srt/environ.py
@@ -708,6 +708,9 @@ class Envs:
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +2/-3; `python/sglang/srt/disaggregation/encode_server.py` modified +500/-63; `python/sglang/srt/environ.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/environ.py`, `python/sglang/srt/multimodal/processors/mimo_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26610 - test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)

- Link: https://github.com/sgl-project/sglang/pull/26610
- Status/date: merged / 2026-05-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +611/-816, 1566 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`; technical summary: Covers "test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)"; the main implementation surface is `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #26673 - [refactor] remove unused op_mlp

- Link: https://github.com/sgl-project/sglang/pull/26673
- Status/date: merged / 2026-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +0/-53, 95 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[refactor] remove unused op_mlp"; model line: MiMo V2 Flash; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`; technical summary: Covers "[refactor] remove unused op_mlp"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +0/-13 (13 lines); hunks: -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`; `python/sglang/srt/models/glm4_moe.py` modified +0/-13 (13 lines); hunks: -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`; `python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13 (13 lines); hunks: -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`; `python/sglang/srt/models/minimax_m2.py` modified +0/-6 (6 lines); hunks: -1069,12 +1069,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +0/-13 (13 lines); hunks: -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/glm4_moe.py` modified +0/-13 (13 lines); hunks: -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13 (13 lines); hunks: -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/minimax_m2.py` modified +0/-6 (6 lines); hunks: -1069,12 +1069,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/mimo_v2.py` modified +0/-4 (4 lines); hunks: -808,10 +808,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):
-    def op_mlp(self, state):
-        hidden_states = state.pop("hidden_states_mlp_input")
-        if not (
-            enable_moe_dense_fully_dp()
-            and (not self.is_layer_sparse)
-            and hidden_states.shape[0] == 0
diff -- python/sglang/srt/models/glm4_moe.py
@@ -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):
-    def op_mlp(self, state):
-        hidden_states = state.pop("hidden_states_mlp_input")
-        if not (
-            enable_moe_dense_fully_dp()
-            and (not self.is_layer_sparse)
-            and hidden_states.shape[0] == 0
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +0/-13; `python/sglang/srt/models/glm4_moe.py` modified +0/-13; `python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13; `python/sglang/srt/models/minimax_m2.py` modified +0/-6; `python/sglang/srt/models/mimo_v2.py` modified +0/-4; `python/sglang/srt/models/qwen3_moe.py` modified +0/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27512 - [Spec] Clamp multimodal pad sentinels in spec-v2 draft prefill embedding

- Link: https://github.com/sgl-project/sglang/pull/27512
- Status/date: merged / 2026-06-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +16/-1, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Spec] Clamp multimodal pad sentinels in spec-v2 draft prefill embedding"; model line: MiMo V2 Flash; category: model implementation change; main diff: `python/sglang/srt/models/mimo_v2_nextn.py`, `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`; technical summary: Covers "[Spec] Clamp multimodal pad sentinels in spec-v2 draft prefill embedding"; the main implementation surface is `python/sglang/srt/models/mimo_v2_nextn.py`, `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mimo_v2_nextn.py` modified +6/-1 (7 lines); hunks: -201,7 +201,12 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +10/-0 (10 lines); hunks: -383,6 +383,16 @@ def _draft_extend_for_prefill(; symbols: _draft_extend_for_prefill, touching `_draft_extend_for_prefill`.
- Code diff details:
  - `python/sglang/srt/models/mimo_v2_nextn.py` modified +6/-1 (7 lines); hunks: -201,7 +201,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +10/-0 (10 lines); hunks: -383,6 +383,16 @@ def _draft_extend_for_prefill(; symbols: _draft_extend_for_prefill
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mimo_v2_nextn.py
@@ -201,7 +201,12 @@ def forward(
-            hidden_states = self.embed_tokens(input_ids)
+            # Multimodal pad sentinels (MM_PAD_SHIFT_VALUE + hash) sit out of vocab;
+            # clamp to avoid an OOB gather. The draft gets visual semantics from target
+            # hidden_states, so the embedding at these positions is unused anyway.
+            hidden_states = self.embed_tokens(
+                input_ids.clamp(min=0, max=self.vocab_size - 1)
diff -- python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py
@@ -383,6 +383,16 @@ def _draft_extend_for_prefill(
+        # The draft embed clamps unconditionally (to tolerate multimodal pad
+        # sentinels), so probe next_token_ids here first -- otherwise a corrupted id
+        # would be clamped away instead of surfacing.
+        maybe_detect_oob(
+            next_token_ids,
+            0,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mimo_v2_nextn.py` modified +6/-1; `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +10/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/mimo_v2_nextn.py`, `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25455 - [NPU] MiMo-V2-Flash Adaptation

- Link: https://github.com/sgl-project/sglang/pull/25455
- Status/date: merged / 2026-06-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mimo_v2.py`; associated commits `2947781ce6b3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +419/-67, 740 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] MiMo-V2-Flash Adaptation"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `python/sglang/srt/models/mimo_v2.py`; technical summary: Covers "[NPU] MiMo-V2-Flash Adaptation"; the main implementation surface is `python/sglang/srt/models/mimo_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mimo_v2.py` modified +8/-2 (10 lines); hunks: -282,7 +282,11 @@ def __init__(; -299,7 +303,9 @@ def __init__(; symbols: __init__, get_moe_weights, touching `__init__, get_moe_weights`.
- Code diff details:
  - `python/sglang/srt/models/mimo_v2.py` modified +8/-2 (10 lines); hunks: -282,7 +282,11 @@ def __init__(; -299,7 +303,9 @@ def __init__(; symbols: __init__, get_moe_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mimo_v2.py
@@ -282,7 +282,11 @@ def __init__(
-        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
+        if (
+            get_moe_a2a_backend().is_deepep()
+            or get_moe_a2a_backend().is_mooncake()
+            or get_moe_a2a_backend().is_ascend_fuseep()
+        ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mimo_v2.py` modified +8/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/graph_runner/multi_layer_eagle_draft_extend_npu_graph_runner.py`, `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27668 - Fix MiMo-V2.5-Pro DP-attention dp size in cookbook deployment snippet

- Link: https://github.com/sgl-project/sglang/pull/27668
- Status/date: merged / 2026-06-10
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; associated commits `53ed34cb882e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-12, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix MiMo-V2.5-Pro DP-attention dp size in cookbook deployment snippet"; model line: MiMo V2 Flash; category: bug fix; main diff: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; technical summary: Covers "Fix MiMo-V2.5-Pro DP-attention dp size in cookbook deployment snippet"; the main implementation surface is `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +23/-12 (35 lines); hunks: -93,13 +93,17 @@ export const MiMoV25Deployment = () => {; -132,13 +136,18 @@ export const MiMoV25Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +23/-12 (35 lines); hunks: -93,13 +93,17 @@ export const MiMoV25Deployment = () => {; -132,13 +136,18 @@ export const MiMoV25Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx
@@ -93,13 +93,17 @@ export const MiMoV25Deployment = () => {
-  // V2.5 (base) checkpoint has TP=4-interleaved fused qkv_proj, so attention
-  // TP per DP group MUST be 4. Effective TP/DP = 4. With tp=8 → dp=2; tp=4 → dp=1.
+  // The attention qkv_proj is TP-interleaved, so attention-TP per DP group is
+  // fixed: V2.5 (base) is TP=4-interleaved, V2.5-Pro is TP=8-interleaved. When
+  // tp exceeds that factor, DP-attention is required with `dp = tp / factor`
+  // (Pro: tp/8, base: tp/4); when tp equals the factor there is a single
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +23/-12
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #26278 - Support MiMo v2 ASR

- Link: https://github.com/sgl-project/sglang/pull/26278
- Status/date: merged / 2026-06-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/entrypoints/openai/transcription_adapters/mimo_v2_asr.py`, `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_asr.py`, `python/sglang/srt/multimodal/processors/mimo_audio.py` and 7 files; associated commits `ec0eb6cce8a1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +991/-434, 1757 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support MiMo v2 ASR"; model line: MiMo V2 Flash; category: model support/runtime entry; main diff: `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/multimodal/processors/mimo_audio.py`, `python/sglang/srt/multimodal/processors/mimo_v2_asr.py`; technical summary: Covers "Support MiMo v2 ASR"; the main implementation surface is `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/multimodal/processors/mimo_audio.py`, `python/sglang/srt/multimodal/processors/mimo_v2_asr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +51/-276 (327 lines); hunks: -3,21 +3,16; -39,20 +34,14; symbols: ImageInput, __post_init__, AudioInput, VideoAudioInput, touching `ImageInput, __post_init__, AudioInput`; `python/sglang/srt/multimodal/processors/mimo_audio.py` added +318/-0 (318 lines); hunks: -0,0 +1,318; symbols: AudioInput, __post_init__, MiMoAudioPipeline, __init__, touching `AudioInput, __post_init__, MiMoAudioPipeline`; `python/sglang/srt/multimodal/processors/mimo_v2_asr.py` added +286/-0 (286 lines); hunks: -0,0 +1,286; symbols: _Content, MiMoV2ASRProcessor, __init__, __getattr__, touching `_Content, MiMoV2ASRProcessor, __init__`; `python/sglang/srt/models/mimo_audio.py` modified +99/-92 (191 lines); hunks: -8,7 +8,7; -1141,72 +1141,89 @@ def output_local_config(self):; symbols: output_local_config, MiMoAudioEncoder, AudioEncoderMixin, __init__, touching `output_local_config, MiMoAudioEncoder, AudioEncoderMixin`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +51/-276 (327 lines); hunks: -3,21 +3,16; -39,20 +34,14; symbols: ImageInput, __post_init__, AudioInput, VideoAudioInput
  - `python/sglang/srt/multimodal/processors/mimo_audio.py` added +318/-0 (318 lines); hunks: -0,0 +1,318; symbols: AudioInput, __post_init__, MiMoAudioPipeline, __init__
  - `python/sglang/srt/multimodal/processors/mimo_v2_asr.py` added +286/-0 (286 lines); hunks: -0,0 +1,286; symbols: _Content, MiMoV2ASRProcessor, __init__, __getattr__
  - `python/sglang/srt/models/mimo_audio.py` modified +99/-92 (191 lines); hunks: -8,7 +8,7; -1141,72 +1141,89 @@ def output_local_config(self):; symbols: output_local_config, MiMoAudioEncoder, AudioEncoderMixin, __init__
  - `python/sglang/srt/models/mimo_v2_asr.py` added +162/-0 (162 lines); hunks: -0,0 +1,162; symbols: _maybe_override_audio_attn_for_blackwell, MiMoV2ASRForCausalLM, __init__, pad_input_ids
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/mimo_v2.py
@@ -3,21 +3,16 @@
-import io
-import os
-import time
-from collections import OrderedDict
-import pybase64
@@ -39,20 +34,14 @@
diff -- python/sglang/srt/multimodal/processors/mimo_audio.py
@@ -0,0 +1,318 @@
+"""Stateful audio preprocessing pipeline shared by MiMo multimodal and ASR processors."""
+import io
+import math
+import os
+import time
+from collections import OrderedDict
diff -- python/sglang/srt/multimodal/processors/mimo_v2_asr.py
@@ -0,0 +1,286 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +51/-276; `python/sglang/srt/multimodal/processors/mimo_audio.py` added +318/-0; `python/sglang/srt/multimodal/processors/mimo_v2_asr.py` added +286/-0; `python/sglang/srt/models/mimo_audio.py` modified +99/-92; `python/sglang/srt/models/mimo_v2_asr.py` added +162/-0; `python/sglang/srt/models/mimo_v2.py` modified +24/-66
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/__init__.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/mimo_v2_asr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27964 - [Spec] Retire Spec V1

- Link: https://github.com/sgl-project/sglang/pull/27964
- Status/date: merged / 2026-06-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 46 files, +111/-252, 1422 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Spec] Retire Spec V1"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`; technical summary: Covers "[Spec] Retire Spec V1"; the main implementation surface is `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass, touching `TestDeepseekMTP, setUpClass, tearDownClass`; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do; `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family, touching `handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp`; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu....
- Code diff details:
  - `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do
  - `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family
  - `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...
  - `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10 (21 lines); hunks: -1,6 +1,5; -31,7 +30,8 @@ class TestEagleConstrainedDecoding(; symbols: TestEagleConstrainedDecoding, setUpClass, tearDownClass, TestEagleConstrainedDecodingV2
- Key code excerpts:

```diff
diff -- test/registered/ep/test_deepep_large.py
@@ -3,7 +3,6 @@
-from sglang.srt.environ import envs
@@ -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):
-        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
-            cls.process = popen_launch_server(
-                cls.model,
-                cls.base_url,
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx
@@ -1108,7 +1108,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1227,7 +1226,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1351,7 +1349,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1476,7 +1473,6 @@ do
diff -- python/sglang/srt/arg_groups/speculative_hook.py
@@ -1,9 +1,8 @@
```

- Reviewed files:
  - tests: `test/registered/ep/test_deepep_large.py` modified +43/-44; `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10; `python/sglang/test/server_fixtures/standalone_fixture.py` modified +7/-8; `python/sglang/test/server_fixtures/spec_eagle_fixture.py` modified +6/-6
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_optimization.mdx` modified +3/-8
  - runtime: `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26
- Risk and verification: The diff ships test coverage in `python/sglang/test/server_fixtures/spec_eagle_fixture.py`, `python/sglang/test/server_fixtures/standalone_fixture.py`, `test/manual/dsv4/test_dsv4_flash_mtp_tp8.py`, `test/manual/dsv4/test_dsv4_pro_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27378 - feat: Support HiCache for MiMo-V2 models (1/N)

- Link: https://github.com/sgl-project/sglang/pull/27378
- Status/date: merged / 2026-06-13
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/models_e2e/test_mimo_v2.py`, `test/registered/models_e2e/test_mimo_v2_flash.py`; associated commits `806365e778df`
- Diff scope read: GitHub Pull Request files API returned 11 files, +667/-43, 892 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: Support HiCache for MiMo-V2 models (1/N)"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `test/registered/models_e2e/test_mimo_v2.py`, `test/registered/models_e2e/test_mimo_v2_flash.py`; technical summary: Covers "feat: Support HiCache for MiMo-V2 models (1/N)"; the main implementation surface is `test/registered/models_e2e/test_mimo_v2.py`, `test/registered/models_e2e/test_mimo_v2_flash.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models_e2e/test_mimo_v2.py` modified +13/-0 (13 lines); hunks: -1,5 +1,6; -20,6 +21,13; symbols: TestMiMoV2, setUpClass, touching `TestMiMoV2, setUpClass`; `test/registered/models_e2e/test_mimo_v2_flash.py` modified +13/-0 (13 lines); hunks: -1,5 +1,6; -42,11 +43,23 @@ class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, Default...; symbols: TestMiMoV2Flash, setUpClass, touching `TestMiMoV2Flash, setUpClass`.
- Code diff details:
  - `test/registered/models_e2e/test_mimo_v2.py` modified +13/-0 (13 lines); hunks: -1,5 +1,6; -20,6 +21,13; symbols: TestMiMoV2, setUpClass
  - `test/registered/models_e2e/test_mimo_v2_flash.py` modified +13/-0 (13 lines); hunks: -1,5 +1,6; -42,11 +43,23 @@ class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, Default...; symbols: TestMiMoV2Flash, setUpClass
- Key code excerpts:

```diff
diff -- test/registered/models_e2e/test_mimo_v2.py
@@ -1,5 +1,6 @@
+from sglang.srt.environ import envs
@@ -20,6 +21,13 @@
+    "--enable-hierarchical-cache",
+    "--hicache-ratio",
+    "1.5",
+    "--hicache-mem-layout",
diff -- test/registered/models_e2e/test_mimo_v2_flash.py
@@ -1,5 +1,6 @@
+from sglang.srt.environ import envs
@@ -42,11 +43,23 @@ class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, DefaultServerBase):
+        "--enable-hierarchical-cache",
+        "--hicache-ratio",
+        "1.5",
+        "--hicache-mem-layout",
```

- Reviewed files:
  - tests: `test/registered/models_e2e/test_mimo_v2.py` modified +13/-0; `test/registered/models_e2e/test_mimo_v2_flash.py` modified +13/-0
- Risk and verification: The diff ships test coverage in `test/registered/jit/test_kvcacheio_asymmetric.py`, `test/registered/models_e2e/test_mimo_v2.py`, `test/registered/models_e2e/test_mimo_v2_flash.py`, `test/registered/unit/mem_cache/test_asymmetric_mha_pool_host_unit.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28223 - [NPU] Add MiMo-V2-Flash manual testcases

- Link: https://github.com/sgl-project/sglang/pull/28223
- Status/date: merged / 2026-06-15
- Trace source: `git log --name-only -- <model-files>` found it through `test/manual/ascend/llm_models/test_npu_mimo_v2_flash.py`; associated commits `3df6e2f9681a`
- Diff scope read: GitHub Pull Request files API returned 2 files, +40/-0, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Add MiMo-V2-Flash manual testcases"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `test/manual/ascend/llm_models/test_npu_mimo_v2_flash.py`; technical summary: Covers "[NPU] Add MiMo-V2-Flash manual testcases"; the main implementation surface is `test/manual/ascend/llm_models/test_npu_mimo_v2_flash.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/manual/ascend/llm_models/test_npu_mimo_v2_flash.py` added +39/-0 (39 lines); hunks: -0,0 +1,39; symbols: TestMiMoV2FlashGraphWithMTP, touching `TestMiMoV2FlashGraphWithMTP`.
- Code diff details:
  - `test/manual/ascend/llm_models/test_npu_mimo_v2_flash.py` added +39/-0 (39 lines); hunks: -0,0 +1,39; symbols: TestMiMoV2FlashGraphWithMTP
- Key code excerpts:

```diff
diff -- test/manual/ascend/llm_models/test_npu_mimo_v2_flash.py
@@ -0,0 +1,39 @@
+import unittest
+from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
+from sglang.test.ascend.test_ascend_utils import MIMO_V2_FLASH_WEIGHTS_PATH
+from sglang.test.test_utils import CustomTestCase
+class TestMiMoV2FlashGraphWithMTP(GSM8KAscendMixin, CustomTestCase):
+    """Testcase: Verify the inference accuracy of MiMo-V2-Flash on GSM8K with cuda graph and MTP (speculative decoding).
```

- Reviewed files:
  - tests: `test/manual/ascend/llm_models/test_npu_mimo_v2_flash.py` added +39/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/ascend/test_ascend_utils.py`, `test/manual/ascend/llm_models/test_npu_mimo_v2_flash.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- Link: https://github.com/sgl-project/sglang/pull/28567
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 184 files, +1865/-1727, 8932 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_parallel(): a structured accessor for parallel-topology state"; model line: MiMo V2 Flash; category: model support/runtime entry; main diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add get_parallel(): a structured accessor for parallel-topology state"; the main implementation surface is `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #29253 - Add MiMo V2.5 Blackwell vision FA4 recipe

- Link: https://github.com/sgl-project/sglang/pull/29253
- Status/date: merged / 2026-06-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; associated commits `7c9804ef218f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-2, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add MiMo V2.5 Blackwell vision FA4 recipe"; model line: MiMo V2 Flash; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`, `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`; technical summary: Covers "Add MiMo V2.5 Blackwell vision FA4 recipe"; the main implementation surface is `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`, `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +3/-2 (5 lines); hunks: -11,8 +11,8 @@ export const MiMoV25Deployment = () => {; -350,6 +350,7 @@ export const MiMoV25Deployment = () => {; `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +1/-0 (1 lines); hunks: -88,6 +88,7 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressiv....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +3/-2 (5 lines); hunks: -11,8 +11,8 @@ export const MiMoV25Deployment = () => {; -350,6 +350,7 @@ export const MiMoV25Deployment = () => {
  - `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +1/-0 (1 lines); hunks: -88,6 +88,7 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressiv...
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx
@@ -11,8 +11,8 @@ export const MiMoV25Deployment = () => {
-  //     B200  → tp=4, dp=1, single-node, FP8
-  //     GB300 → tp=4, dp=1, single-node, FP8
+  //     B200  → tp=4, dp=1, single-node, FP8 (Blackwell: vision fa4)
+  //     GB300 → tp=4, dp=1, single-node, FP8 (Blackwell: vision fa4)
@@ -350,6 +350,7 @@ export const MiMoV25Deployment = () => {
+      if (blackwell) flags.push("  --mm-attention-backend fa4");
diff -- docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx
@@ -88,6 +88,7 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressive/mimo-v25-deploym
+- On Blackwell, pass `--mm-attention-backend fa4` for the V2.5 vision encoder. The checkpoint config requests FlashAttention-3 internally, but SGLang rejects FA3 on Blackwell and
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +3/-2; `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +1/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
