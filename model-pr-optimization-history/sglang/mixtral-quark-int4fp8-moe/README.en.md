# sglang Mixtral Quark INT4/FP8 MoE Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.
- Diffusion model families have been removed from this history set and are no longer part of model optimization skills.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/sglang/srt/layers/quantization/quark/__init__.py` | no direct PR-number commit |
| `python/sglang/srt/layers/quantization/quark/quark.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#13147](https://github.com/sgl-project/sglang/pull/13147), [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/schemes/__init__.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py` | [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#18252](https://github.com/sgl-project/sglang/pull/18252), [#19422](https://github.com/sgl-project/sglang/pull/19422) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` | [#18252](https://github.com/sgl-project/sglang/pull/18252), [#18684](https://github.com/sgl-project/sglang/pull/18684), [#21040](https://github.com/sgl-project/sglang/pull/21040), [#21067](https://github.com/sgl-project/sglang/pull/21067), [#21097](https://github.com/sgl-project/sglang/pull/21097), [#23585](https://github.com/sgl-project/sglang/pull/23585) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py` | [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/utils.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485) |
| `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` | [#7392](https://github.com/sgl-project/sglang/pull/7392) |
| `python/sglang/srt/models/mixtral.py` | [#460](https://github.com/sgl-project/sglang/pull/460), [#1081](https://github.com/sgl-project/sglang/pull/1081), [#1290](https://github.com/sgl-project/sglang/pull/1290), [#1418](https://github.com/sgl-project/sglang/pull/1418), [#1835](https://github.com/sgl-project/sglang/pull/1835), [#2156](https://github.com/sgl-project/sglang/pull/2156), [#2163](https://github.com/sgl-project/sglang/pull/2163), [#2300](https://github.com/sgl-project/sglang/pull/2300), [#2371](https://github.com/sgl-project/sglang/pull/2371), [#2563](https://github.com/sgl-project/sglang/pull/2563), [#6223](https://github.com/sgl-project/sglang/pull/6223), [#7966](https://github.com/sgl-project/sglang/pull/7966), ... (17 total) |
| `python/sglang/srt/models/mixtral_quant.py` | [#460](https://github.com/sgl-project/sglang/pull/460), [#1081](https://github.com/sgl-project/sglang/pull/1081) |

## PR Coverage Summary

- Git-traced PRs: 27
- Extra PRs preserved from existing docs: 2
- Total PRs in this document: 28
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2024-05-21 | [#460](https://github.com/sgl-project/sglang/pull/460) | merged | port fp8 mixtral | `python/sglang/srt/models/mixtral_quant.py`, `python/sglang/srt/models/mixtral.py` |
| 2024-08-13 | [#1081](https://github.com/sgl-project/sglang/pull/1081) | merged | Update the mixtral to use the better FusedMoE layer | `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py` |
| 2024-09-01 | [#1290](https://github.com/sgl-project/sglang/pull/1290) | merged | fix: resolve fp8 for mixtral | `python/sglang/srt/models/mixtral.py` |
| 2024-09-14 | [#1418](https://github.com/sgl-project/sglang/pull/1418) | merged | Add torchao quant for mixtral and qwen_moe | `python/sglang/srt/models/mixtral.py` |
| 2024-10-29 | [#1835](https://github.com/sgl-project/sglang/pull/1835) | merged | [FP8 KV Cache, Mixtral] Avoid KeyError at loading pre-quantized FP8 m… | `python/sglang/srt/models/mixtral.py` |
| 2024-11-24 | [#2156](https://github.com/sgl-project/sglang/pull/2156) | merged | feat: update other MoE models deps | `python/sglang/srt/models/mixtral.py` |
| 2024-11-24 | [#2163](https://github.com/sgl-project/sglang/pull/2163) | merged | Rename triton_fused_moe -> fused_moe_triton | `python/sglang/srt/models/mixtral.py` |
| 2024-12-03 | [#2300](https://github.com/sgl-project/sglang/pull/2300) | merged | Fix gptq for moe layers | `python/sglang/srt/models/mixtral.py` |
| 2024-12-06 | [#2371](https://github.com/sgl-project/sglang/pull/2371) | merged | MoE Expert Parallel | `python/sglang/srt/models/mixtral.py` |
| 2024-12-23 | [#2563](https://github.com/sgl-project/sglang/pull/2563) | merged | Reorg moe code | `python/sglang/srt/models/mixtral.py` |
| 2025-05-12 | [#6223](https://github.com/sgl-project/sglang/pull/6223) | merged | [PP] Fix init_memory_pool desync & add PP for mixtral | `python/sglang/srt/models/mixtral.py` |
| 2025-07-19 | [#7966](https://github.com/sgl-project/sglang/pull/7966) | merged | [1/N] MoE Refactor: refactor `select_experts` | `python/sglang/srt/models/mixtral.py` |
| 2025-07-29 | [#8448](https://github.com/sgl-project/sglang/pull/8448) | merged | Support EPLB in FusedMoE | `python/sglang/srt/models/mixtral.py` |
| 2025-08-01 | [#8658](https://github.com/sgl-project/sglang/pull/8658) | merged | [5/N] MoE Refactor: Update MoE parallelism arguments | `python/sglang/srt/models/mixtral.py` |
| 2025-08-15 | [#8849](https://github.com/sgl-project/sglang/pull/8849) | merged | [6/N] MoE Refactor: Cleanup MoE-related configs | `python/sglang/srt/models/mixtral.py` |
| 2025-10-08 | [#11211](https://github.com/sgl-project/sglang/pull/11211) | merged | [8/N] MoE Refactor: deprecate `EPMoE` | `python/sglang/srt/models/mixtral.py` |
| 2025-11-13 | [#10485](https://github.com/sgl-project/sglang/pull/10485) | merged | [Quantization] Support Quark Dense + MoE FP8 & FP8 PTPC | `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2025-11-20 | [#13667](https://github.com/sgl-project/sglang/pull/13667) | merged | [Piecewise CUDA Graph] Fix recompile issue for Mixtral and Grok2 | `python/sglang/srt/models/mixtral.py` |
| 2025-12-09 | [#13147](https://github.com/sgl-project/sglang/pull/13147) | merged | Aiter fp8 kv cache | `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-01-14 | [#7392](https://github.com/sgl-project/sglang/pull/7392) | merged | [AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm | `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` |
| 2026-01-19 | [#17116](https://github.com/sgl-project/sglang/pull/17116) | merged | [AMD CI] Migrate and Add More Testcases | `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py` |
| 2026-02-18 | [#18252](https://github.com/sgl-project/sglang/pull/18252) | merged | [4/N] Quantization Refactor: Quark MoE schemes | `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-02-26 | [#19422](https://github.com/sgl-project/sglang/pull/19422) | merged | [AMD] Use fused GEMM with FP8 cast for FP8 prefill | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` |
| 2026-03-20 | [#18684](https://github.com/sgl-project/sglang/pull/18684) | merged | [AMD] Add MoE weights and scales padding | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-03-21 | [#21067](https://github.com/sgl-project/sglang/pull/21067) | merged | Revert "[AMD] Add MoE weights and scales padding" | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-03-25 | [#21040](https://github.com/sgl-project/sglang/pull/21040) | merged | [AMD][MoRI] Auto-select dispatch quantization type from MoE weight dtype. | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-04-13 | [#21097](https://github.com/sgl-project/sglang/pull/21097) | merged | [AMD] Add MoE weights and scales padding | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-04-24 | [#23585](https://github.com/sgl-project/sglang/pull/23585) | merged | Move expert_mask_gpu from FusedMoE layer to StandardDispatcher | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |

## Per-PR Diff Audit Cards

### PR #460 - port fp8 mixtral

- Link: https://github.com/sgl-project/sglang/pull/460
- Status/date: merged / 2024-05-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py`; associated commits `0fafc5606b0d`
- Diff scope read: GitHub Pull Request files API returned 6 files, +636/-121, 921 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR optimizes an inference path or backend selection. Title: "port fp8 mixtral". The diff centers on `python/sglang/srt/models/mixtral_quant.py`, `python/sglang/srt/models/mixtral.py`. The PR body has no extra context, so this assessment comes from the title, file list, and patch.
- Key implementation: `python/sglang/srt/models/mixtral_quant.py` added +371/-0 (371 lines); hunks: -0,0 +1,371; symbols: MixtralMLP, __init__, forward, MixtralMoE, touching `MixtralMLP, __init__, forward`; `python/sglang/srt/models/mixtral.py` modified +240/-101 (341 lines); hunks: -1,5 +1,5; -8,131 +8,226; symbols: MixtralMLP, __init__, forward, MixtralMoE, touching `MixtralMLP, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/mixtral_quant.py` added +371/-0 (371 lines); hunks: -0,0 +1,371; symbols: MixtralMLP, __init__, forward, MixtralMoE
  - `python/sglang/srt/models/mixtral.py` modified +240/-101 (341 lines); hunks: -1,5 +1,5; -8,131 +8,226; symbols: MixtralMLP, __init__, forward, MixtralMoE
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral_quant.py
@@ -0,0 +1,371 @@
+# Adapted from
+# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral_quant.py#L1
+"""Inference-only Mixtral model."""
+from typing import Iterable, Optional, Tuple
+import numpy as np
+import torch
diff -- python/sglang/srt/models/mixtral.py
@@ -1,5 +1,5 @@
-# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral_quant.py#L1
+# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral.py#L1
@@ -8,131 +8,226 @@
+from vllm import _custom_ops as ops
+from vllm.model_executor.layers.fused_moe import fused_moe
+from vllm.model_executor.layers.quantization.fp8 import Fp8Config
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral_quant.py` added +371/-0; `python/sglang/srt/models/mixtral.py` modified +240/-101
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/router/model_rpc.py`, `python/sglang/srt/managers/router/model_runner.py`, `python/sglang/srt/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #1081 - Update the mixtral to use the better FusedMoE layer

- Link: https://github.com/sgl-project/sglang/pull/1081
- Status/date: merged / 2024-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py`; associated commits `ad3e4f16199a`
- Diff scope read: GitHub Pull Request files API returned 4 files, +57/-258, 515 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR optimizes an inference path or backend selection. Title: "Update the mixtral to use the better FusedMoE layer". The diff centers on `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py`. The PR body has no extra context, so this assessment comes from the title, file list, and patch.
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +55/-253 (308 lines); hunks: -18,34 +18,25; -69,216 +60,44 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading, forward, touching `__init__, weight_loader, process_weights_after_loading`; `python/sglang/srt/models/mixtral_quant.py` modified +0/-3 (3 lines); hunks: -160,7 +160,6 @@ def __init__(; -183,7 +182,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +55/-253 (308 lines); hunks: -18,34 +18,25; -69,216 +60,44 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading, forward
  - `python/sglang/srt/models/mixtral_quant.py` modified +0/-3 (3 lines); hunks: -160,7 +160,6 @@ def __init__(; -183,7 +182,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -18,34 +18,25 @@
-import numpy as np
-import torch.nn.functional as F
-from vllm import _custom_ops as ops
-from vllm.distributed import (
-    get_tensor_model_parallel_rank,
-    get_tensor_model_parallel_world_size,
diff -- python/sglang/srt/models/mixtral_quant.py
@@ -160,7 +160,6 @@ def __init__(
-        sliding_window: Optional[int] = None,
@@ -183,7 +182,6 @@ def __init__(
-        self.sliding_window = sliding_window
@@ -246,7 +244,6 @@ def __init__(
-            sliding_window=config.sliding_window,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +55/-253; `python/sglang/srt/models/mixtral_quant.py` modified +0/-3
- Risk and verification: The diff ships test coverage in `test/srt/test_moe_serving_throughput.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1290 - fix: resolve fp8 for mixtral

- Link: https://github.com/sgl-project/sglang/pull/1290
- Status/date: merged / 2024-09-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `9b0805242eea`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "fix: resolve fp8 for mixtral". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: ## Motivation ## Modifications ## Checklist - [ ] Format your code according to the Contributor Guide. - [ ] Add unit tests as outlined in the Contributor Guide. - [ ] Update do...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -362,7 +362,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -362,7 +362,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -362,7 +362,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-                        weight_name,
+                        name,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #1418 - Add torchao quant for mixtral and qwen_moe

- Link: https://github.com/sgl-project/sglang/pull/1418
- Status/date: merged / 2024-09-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `30b404ce72b5`
- Diff scope read: GitHub Pull Request files API returned 4 files, +50/-20, 138 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "Add torchao quant for mixtral and qwen_moe". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: Summary: Similar to https://github.com/sgl-project/sglang/pull/1341 we add torchao quantization to mixtral model Test Plan: Note: compile is not working yet, and I can't install...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +5/-0 (5 lines); hunks: -41,6 +41,8; -296,6 +298,7 @@ def __init__(; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +5/-0 (5 lines); hunks: -41,6 +41,8; -296,6 +298,7 @@ def __init__(; symbols: __init__, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -41,6 +41,8 @@
+from sglang.srt.layers.torchao_utils import apply_torchao_config_
+from sglang.srt.managers.schedule_batch import global_server_args_dict
@@ -296,6 +298,7 @@ def __init__(
+        self.torchao_config = global_server_args_dict["torchao_config"]
@@ -376,5 +379,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+        apply_torchao_config_(self, params_dict, set(["proj.weight"]))
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/torchao_utils.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #1835 - [FP8 KV Cache, Mixtral] Avoid KeyError at loading pre-quantized FP8 m…

- Link: https://github.com/sgl-project/sglang/pull/1835
- Status/date: merged / 2024-10-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `54dd3ea12277`
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[FP8 KV Cache, Mixtral] Avoid KeyError at loading pre-quantized FP8 m…". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: …odel with kv_scale ## Motivation Reuse FP8 quantized Mixtral models, to avoid KeyError, just skip/ignore `kv_scale` for now. E.g. `amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV` has `k...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +3/-0 (3 lines); hunks: -369,6 +369,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +3/-0 (3 lines); hunks: -369,6 +369,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -369,6 +369,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                    # Skip loading kv_scale from ckpts towards new design.
+                    if name.endswith(".kv_scale") and name not in params_dict:
+                        continue
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2156 - feat: update other MoE models deps

- Link: https://github.com/sgl-project/sglang/pull/2156
- Status/date: merged / 2024-11-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `e3938b2f9c96`
- Diff scope read: GitHub Pull Request files API returned 10 files, +28/-14, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "feat: update other MoE models deps". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: ## Motivation ## Modifications ## Checklist - [ ] Format your code according to the Contributor Guide. - [ ] Add unit tests as outlined in the Contributor Guide. - [ ] Update do...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -22,7 +22,6; -36,6 +35,7.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -22,7 +22,6; -36,6 +35,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -22,7 +22,6 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
@@ -36,6 +35,7 @@
+from sglang.srt.layers.triton_fused_moe import FusedMoE
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/fused_moe/layer.py`, `python/sglang/srt/layers/triton_fused_moe/fused_moe.py`, `python/sglang/srt/layers/triton_fused_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2163 - Rename triton_fused_moe -> fused_moe_triton

- Link: https://github.com/sgl-project/sglang/pull/2163
- Status/date: merged / 2024-11-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `be0124bda09d`
- Diff scope read: GitHub Pull Request files API returned 76 files, +19/-19, 199 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR optimizes an inference path or backend selection. Title: "Rename triton_fused_moe -> fused_moe_triton". The diff centers on `python/sglang/srt/models/mixtral.py`. The PR body has no extra context, so this assessment comes from the title, file list, and patch.
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -25,6 +25,7; -35,7 +36,6.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -25,6 +25,7; -35,7 +36,6
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -25,6 +25,7 @@
+from sglang.srt.layers.fused_moe_triton import FusedMoE
@@ -35,7 +36,6 @@
-from sglang.srt.layers.triton_fused_moe import FusedMoE
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/fused_moe/__init__.py`, `python/sglang/srt/layers/fused_moe_grok/__init__.py`, `python/sglang/srt/layers/fused_moe_grok/configs/E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=float8.json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2300 - Fix gptq for moe layers

- Link: https://github.com/sgl-project/sglang/pull/2300
- Status/date: merged / 2024-12-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `1228f7ca69e6`
- Diff scope read: GitHub Pull Request files API returned 2 files, +44/-2, 78 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix gptq for moe layers". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: Try to fix https://github.com/sgl-project/sglang/issues/2117 https://github.com/sgl-project/sglang/issues/2270 We can run `python3 -m sglang.launch_server --model TheBloke/Mixtr...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +10/-2 (12 lines); hunks: -339,7 +339,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; -353,6 +355,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +10/-2 (12 lines); hunks: -339,7 +339,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; -353,6 +355,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -339,7 +339,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-                if name.endswith(".bias") and name not in params_dict:
+                if (
+                    name.endswith(".bias") or name.endswith("_bias")
+                ) and name not in params_dict:
@@ -353,6 +355,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                    if (
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +10/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/__init__.py`, `python/sglang/srt/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2371 - MoE Expert Parallel

- Link: https://github.com/sgl-project/sglang/pull/2371
- Status/date: merged / 2024-12-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `3d32e4a32c4c`
- Diff scope read: GitHub Pull Request files API returned 10 files, +1172/-8, 1320 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR changes model-related implementation. Title: "MoE Expert Parallel". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: Resubmit the PR for MoE-EP. Please refer to the details in the previous PR: https://github.com/sgl-project/sglang/pull/2203.
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +13/-5 (18 lines); hunks: -21,9 +21,13; -38,6 +42,7; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +13/-5 (18 lines); hunks: -21,9 +21,13; -38,6 +42,7; symbols: __init__, forward, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -21,9 +21,13 @@
-from vllm.distributed import get_tensor_model_parallel_world_size
+from vllm.distributed import (
+    get_tensor_model_parallel_world_size,
+    tensor_model_parallel_all_reduce,
+)
+from sglang.srt.layers.ep_moe.layer import EPMoE
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +13/-5
- Risk and verification: The diff ships test coverage in `test/srt/test_moe_ep.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #2563 - Reorg moe code

- Link: https://github.com/sgl-project/sglang/pull/2563
- Status/date: merged / 2024-12-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `e835a50021e0`
- Diff scope read: GitHub Pull Request files API returned 88 files, +338/-344, 1108 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "Reorg moe code". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: ## Motivation Reorg moe code and reuse common part. ## Checklist - [ ] Format your code according to the Contributor Guide. - [ ] Add unit tests as outlined in the Contributor G...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +2/-2 (4 lines); hunks: -27,15 +27,15.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +2/-2 (4 lines); hunks: -27,15 +27,15
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -27,15 +27,15 @@
-from sglang.srt.layers.ep_moe.layer import EPMoE
-from sglang.srt.layers.fused_moe_triton import FusedMoE
+from sglang.srt.layers.moe.ep_moe.layer import EPMoE
+from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/srt/test_fused_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6223 - [PP] Fix init_memory_pool desync & add PP for mixtral

- Link: https://github.com/sgl-project/sglang/pull/6223
- Status/date: merged / 2025-05-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `bad7c26fdc7f`
- Diff scope read: GitHub Pull Request files API returned 8 files, +179/-47, 391 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "[PP] Fix init_memory_pool desync & add PP for mixtral". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: ## Motivation ## Modifications ## Checklist - [x] Format your code according to the Code Formatting with Pre-Commit. - [x] Add unit tests as outlined in the Running Unit Tests....
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +98/-34 (132 lines); hunks: -16,13 +16,15; -38,14 +40,17; symbols: MixtralMoE, __init__, forward, touching `MixtralMoE, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +98/-34 (132 lines); hunks: -16,13 +16,15; -38,14 +40,17; symbols: MixtralMoE, __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -16,13 +16,15 @@
-from typing import Iterable, Optional, Tuple
+import logging
+from typing import Iterable, Optional, Tuple, Union
+    get_pp_group,
@@ -38,14 +40,17 @@
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +98/-34
- Risk and verification: The diff ships test coverage in `test/srt/test_bench_serving.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7966 - [1/N] MoE Refactor: refactor `select_experts`

- Link: https://github.com/sgl-project/sglang/pull/7966
- Status/date: merged / 2025-07-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `15ad6c908670`
- Diff scope read: GitHub Pull Request files API returned 39 files, +557/-872, 2848 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR optimizes an inference path or backend selection. Title: "[1/N] MoE Refactor: refactor `select_experts`". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: ## Motivation This pull request extracts the `select_experts` computation from within `FusedMoE` and `EPMoE`, moving it outside these modules. This refactoring offers three key...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +9/-2 (11 lines); hunks: -37,6 +37,7; -86,14 +87,19 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +9/-2 (11 lines); hunks: -37,6 +37,7; -86,14 +87,19 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -37,6 +37,7 @@
+from sglang.srt.layers.moe.topk import TopK
@@ -86,14 +87,19 @@ def __init__(
+        self.topk = TopK(
+            top_k=top_k,
+            renormalize=True,
+        )
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +9/-2
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_block_fp8.py`, `python/sglang/test/test_block_fp8_ep.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/test/test_fp4_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8448 - Support EPLB in FusedMoE

- Link: https://github.com/sgl-project/sglang/pull/8448
- Status/date: merged / 2025-07-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `9effeb5bddf2`
- Diff scope read: GitHub Pull Request files API returned 15 files, +107/-11, 407 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "Support EPLB in FusedMoE". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: ## Motivation Fix #8398 ## Modifications ## Checklist - [ ] Format your code according to the Code Formatting with Pre-Commit. - [ ] Add unit tests as outlined in the Running Un...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +3/-0 (3 lines); hunks: -69,6 +69,7 @@ def __init__(; -97,6 +98,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +3/-0 (3 lines); hunks: -69,6 +69,7 @@ def __init__(; -97,6 +98,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -69,6 +69,7 @@ def __init__(
+        layer_id: int,
@@ -97,6 +98,7 @@ def __init__(
+            layer_id=layer_id,
@@ -226,6 +228,7 @@ def __init__(
+            layer_id=layer_id,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/eplb/expert_distribution.py`, `python/sglang/srt/eplb/expert_location.py`, `python/sglang/srt/eplb/expert_location_dispatch.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8658 - [5/N] MoE Refactor: Update MoE parallelism arguments

- Link: https://github.com/sgl-project/sglang/pull/8658
- Status/date: merged / 2025-08-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `6c88f6c8d908`
- Diff scope read: GitHub Pull Request files API returned 38 files, +342/-299, 1748 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "[5/N] MoE Refactor: Update MoE parallelism arguments". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: ## Motivation This PR introduces `--moe-a2a-backend` and deprecates `--enable-ep-moe` and `--enable-deepep-moe`. ## Modifications ## Accuracy Test ## Benchmark & Profiling ## Ch...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +3/-3 (6 lines); hunks: -24,6 +24,7; -94,7 +95,7 @@ def __init__(; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +3/-3 (6 lines); hunks: -24,6 +24,7; -94,7 +95,7 @@ def __init__(; symbols: __init__, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -24,6 +24,7 @@
+    get_moe_expert_parallel_world_size,
@@ -94,7 +95,7 @@ def __init__(
-        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
+        MoEImpl = EPMoE if get_moe_expert_parallel_world_size() > 1 else FusedMoE
@@ -398,8 +399,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `python/sglang/test/runners.py`, `test/srt/test_deepep_large.py`, `test/srt/test_deepep_small.py`, `test/srt/test_eplb.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8849 - [6/N] MoE Refactor: Cleanup MoE-related configs

- Link: https://github.com/sgl-project/sglang/pull/8849
- Status/date: merged / 2025-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `295895120df4`
- Diff scope read: GitHub Pull Request files API returned 69 files, +958/-1039, 4640 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "[6/N] MoE Refactor: Cleanup MoE-related configs". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: ## Motivation - Adding `--moe-runner-backend` and deprecating `--enable-triton-kernel-moe`, `--enable-flashinfer-cutlass-moe`, and `--enable-flashinfer-trtllm-moe`. - Adding `To...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +0/-2 (2 lines); hunks: -47,7 +47,6; -104,7 +103,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +0/-2 (2 lines); hunks: -47,7 +47,6; -104,7 +103,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -47,7 +47,6 @@
-from sglang.srt.managers.schedule_batch import global_server_args_dict
@@ -104,7 +103,6 @@ def __init__(
-            tp_size=tp_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +0/-2
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_block_fp8.py`, `python/sglang/test/test_block_fp8_ep.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/test/test_fp4_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11211 - [8/N] MoE Refactor: deprecate `EPMoE`

- Link: https://github.com/sgl-project/sglang/pull/11211
- Status/date: merged / 2025-10-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `3c06b673aff9`
- Diff scope read: GitHub Pull Request files API returned 19 files, +496/-1778, 2897 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "[8/N] MoE Refactor: deprecate `EPMoE`". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: ## Motivation - [x] Deprecate `EPMoE.forward_deepgemm` - [x] Implement `deep_gemm` MoE runner. - [x] Update cutlass EP. - [x] Recover EP unit tests. - [x] Enable DeepGeMM for TP...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +1/-3 (4 lines); hunks: -36,7 +36,6; -94,8 +93,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +1/-3 (4 lines); hunks: -36,7 +36,6; -94,8 +93,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -36,7 +36,6 @@
-from sglang.srt.layers.moe.ep_moe.layer import EPMoE
@@ -94,8 +93,7 @@ def __init__(
-        MoEImpl = EPMoE if get_moe_expert_parallel_world_size() > 1 else FusedMoE
-        self.experts = MoEImpl(
+        self.experts = FusedMoE(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +1/-3
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_block_fp8_ep.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`, `test/srt/ep/test_moe_ep.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10485 - [Quantization] Support Quark Dense + MoE FP8 & FP8 PTPC

- Link: https://github.com/sgl-project/sglang/pull/10485
- Status/date: merged / 2025-11-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/__init__.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `python/sglang/srt/layers/quantization/quark/utils.py`; associated commits `67e9d287eea0`
- Diff scope read: GitHub Pull Request files API returned 10 files, +666/-243, 1041 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "[Quantization] Support Quark Dense + MoE FP8 & FP8 PTPC". The diff centers on `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `python/sglang/srt/layers/quantization/quark/quark.py`. PR body context: Includes #10396 and #10289. Verified with Qwen3-30b-a3b-thinking-2507 quantized by Quark, running gsm8k.
- Key implementation: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +116/-220 (336 lines); hunks: -604,158 +604,16 @@ def apply_fp8_linear(; -783,87 +641,125 @@ def apply_fp8_linear(; symbols: apply_fp8_linear, can_auto_enable_marlin_fp8, touching `apply_fp8_linear, can_auto_enable_marlin_fp8`; `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +186/-0 (186 lines); hunks: -0,0 +1,186; symbols: QuarkW8A8Fp8, __init__, get_min_capability, process_weights_after_loading, touching `QuarkW8A8Fp8, __init__, get_min_capability`; `python/sglang/srt/layers/quantization/quark/quark.py` modified +42/-1 (43 lines); hunks: -14,7 +14,11; -173,6 +177,37 @@ def _check_scheme_supported(self, min_capability: int, erro...; symbols: _check_scheme_supported, _is_fp8_w8a8, _is_mx_fp4, _get_scheme_from_config, touching `_check_scheme_supported, _is_fp8_w8a8, _is_mx_fp4`; `python/sglang/srt/layers/quantization/__init__.py` modified +3/-11 (14 lines); hunks: -35,6 +35,7 @@ def override_quantization_method(self, *args, **kwargs):; -65,23 +66,14 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method, touching `override_quantization_method`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +116/-220 (336 lines); hunks: -604,158 +604,16 @@ def apply_fp8_linear(; -783,87 +641,125 @@ def apply_fp8_linear(; symbols: apply_fp8_linear, can_auto_enable_marlin_fp8
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +186/-0 (186 lines); hunks: -0,0 +1,186; symbols: QuarkW8A8Fp8, __init__, get_min_capability, process_weights_after_loading
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +42/-1 (43 lines); hunks: -14,7 +14,11; -173,6 +177,37 @@ def _check_scheme_supported(self, min_capability: int, erro...; symbols: _check_scheme_supported, _is_fp8_w8a8, _is_mx_fp4, _get_scheme_from_config
  - `python/sglang/srt/layers/quantization/__init__.py` modified +3/-11 (14 lines); hunks: -35,6 +35,7 @@ def override_quantization_method(self, *args, **kwargs):; -65,23 +66,14 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method
  - `python/sglang/srt/layers/quantization/quark/utils.py` modified +11/-1 (12 lines); hunks: -6,7 +6,17; symbols: raise_aiter_import_error
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -604,158 +604,16 @@ def apply_fp8_linear(
-        # cutlass_scaled_mm supports per tensor/channel W and per tensor/token A
-        # for sgl-kernel fp8_scaled_mm, it support per channel W now
+        # Maybe apply padding to output, see comment in __init__
+        num_token_padding = output_padding
-            qinput, x_scale = scaled_fp8_quant(
-                input_2d,
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -0,0 +1,186 @@
+# SPDX-License-Identifier: Apache-2.0
+from typing import Any, Callable, Optional, cast
+import torch
+from torch.nn import Parameter
+from sglang.srt.layers.parameter import (
+    ChannelQuantScaleParameter,
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -14,7 +14,11 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +116/-220; `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +186/-0; `python/sglang/srt/layers/quantization/quark/quark.py` modified +42/-1; `python/sglang/srt/layers/quantization/__init__.py` modified +3/-11; `python/sglang/srt/layers/quantization/quark/utils.py` modified +11/-1; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +8/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/quantization/__init__.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13667 - [Piecewise CUDA Graph] Fix recompile issue for Mixtral and Grok2

- Link: https://github.com/sgl-project/sglang/pull/13667
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `b5344b31b8f1`
- Diff scope read: GitHub Pull Request files API returned 3 files, +24/-160, 315 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Piecewise CUDA Graph] Fix recompile issue for Mixtral and Grok2". The diff centers on `python/sglang/srt/models/mixtral.py`. PR body context: Co-Author: Oasis-Git Co-Author: Minglei Zhu Co-Author: Ke Bao ## Motivation To close https://github.com/sgl-project/sglang/issues/13469 ### Recompile Debugging Note Thanks a lot...
- Key implementation: `python/sglang/srt/models/mixtral.py` modified +1/-0 (1 lines); hunks: -353,6 +353,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/mixtral.py` modified +1/-0 (1 lines); hunks: -353,6 +353,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -353,6 +353,7 @@ def __init__(
+    @torch.no_grad()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/grok.py`, `python/sglang/srt/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13147 - Aiter fp8 kv cache

- Link: https://github.com/sgl-project/sglang/pull/13147
- Status/date: merged / 2025-12-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/quark.py`; associated commits `c106b54b57d8`
- Diff scope read: GitHub Pull Request files API returned 7 files, +594/-96, 1032 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "Aiter fp8 kv cache". The diff centers on `python/sglang/srt/layers/quantization/quark/quark.py`. PR body context: ## Motivation Support fp8 kv cache in aiter-backend of AMD. Aiter backend only support mla decode fp8 computation. Other attention function still do bf16 computation ## Modifica...
- Key implementation: `python/sglang/srt/layers/quantization/quark/quark.py` modified +2/-0 (2 lines); hunks: -71,6 +71,8 @@ def get_quant_method(; symbols: get_quant_method, touching `get_quant_method`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +2/-0 (2 lines); hunks: -71,6 +71,8 @@ def get_quant_method(; symbols: get_quant_method
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -71,6 +71,8 @@ def get_quant_method(
+            elif isinstance(layer, RadixAttention):
+                return QuarkKVCacheMethod(self)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/quark.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/quantization/quark/quark.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7392 - [AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm

- Link: https://github.com/sgl-project/sglang/pull/7392
- Status/date: merged / 2026-01-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`; associated commits `5af84c8af554`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +615/-15, 759 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "[AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm". The diff centers on `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`. PR body context: As per title, this PR supersedes https://github.com/sgl-project/sglang/pull/6238. This PR implements loading MOE models checkpoints in high-precision (fp16, bf16), quantizing on...
- Key implementation: `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` added +443/-0 (443 lines); hunks: -0,0 +1,443; symbols: tqdm_reset_no_print, QuarkInt4Fp8Config, for, __init__, touching `tqdm_reset_no_print, QuarkInt4Fp8Config, for`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` added +443/-0 (443 lines); hunks: -0,0 +1,443; symbols: tqdm_reset_no_print, QuarkInt4Fp8Config, for, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark_int4fp8_moe.py
@@ -0,0 +1,443 @@
+import logging
+from typing import TYPE_CHECKING, Any, Dict, List, Optional
+import torch
+from tqdm import tqdm
+from tqdm.std import EMA
+from sglang.srt.distributed import get_tensor_model_parallel_rank
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` added +443/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_int4fp8_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17116 - [AMD CI] Migrate and Add More Testcases

- Link: https://github.com/sgl-project/sglang/pull/17116
- Status/date: merged / 2026-01-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +310/-66, 596 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "[AMD CI] Migrate and Add More Testcases". The diff centers on `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py`. PR body context: ## Motivation Cleans up and reorganizes AMD CI test infrastructure. ## Modifications - Renamed suite: `stage-a-test-1` → `stage-a-test-1-amd` (5 files) - Update suite name: `sta...
- Key implementation: `.github/workflows/pr-test-amd.yml` modified +81/-47 (128 lines); hunks: -149,7 +149,10 @@ jobs:; -190,7 +193,7 @@ jobs:; `test/registered/amd/test_deepseek_v3_mtp.py` added +116/-0 (116 lines); hunks: -0,0 +1,116; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV3MTP, setUpClass, tearDownClass`; `test/registered/amd/test_deepseek_v3_basic.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestDeepseekV3Basic, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV3Basic, setUpClass, tearDownClass`; `test/srt/run_suite.py` modified +0/-8 (8 lines); hunks: -91,10 +91,7; -103,15 +100,10.
- Code diff details:
  - `.github/workflows/pr-test-amd.yml` modified +81/-47 (128 lines); hunks: -149,7 +149,10 @@ jobs:; -190,7 +193,7 @@ jobs:
  - `test/registered/amd/test_deepseek_v3_mtp.py` added +116/-0 (116 lines); hunks: -0,0 +1,116; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/test_deepseek_v3_basic.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestDeepseekV3Basic, setUpClass, tearDownClass, test_a_gsm8k
  - `test/srt/run_suite.py` modified +0/-8 (8 lines); hunks: -91,10 +91,7; -103,15 +100,10
  - `test/registered/core/test_deterministic.py` modified +5/-1 (6 lines); hunks: -9,15 +9,18; -32,6 +35,7 @@ def get_server_args(cls):; symbols: TestFlashinferDeterministic, get_server_args, TestFa3Deterministic
- Key code excerpts:

```diff
diff -- .github/workflows/pr-test-amd.yml
@@ -149,7 +149,10 @@ jobs:
+          docker exec -w /sglang-checkout/sgl-kernel/tests ci_sglang python3 -m pytest test_moe_topk_sigmoid.py
+          docker exec -w /sglang-checkout/sgl-kernel/tests ci_sglang python3 -m pytest test_torch_defaults_reset.py
+          docker exec -w /sglang-checkout/sgl-kernel/tests ci_sglang python3 -m pytest test_amd_deterministic_custom_allreduce.py
+          docker exec -w /sglang-checkout/sgl-kernel/tests ci_sglang python3 -m pytest test_amd_nccl_allreduce_determinism.py
@@ -190,7 +193,7 @@ jobs:
-          bash scripts/ci/amd_ci_exec.sh -w "/sglang-checkout/test" python3 run_suite.py --hw amd --suite stage-a-test-1
diff -- test/registered/amd/test_deepseek_v3_mtp.py
@@ -0,0 +1,116 @@
+import unittest
+from types import SimpleNamespace
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_amd_ci
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
diff -- test/registered/amd/test_deepseek_v3_basic.py
@@ -0,0 +1,84 @@
```

- Reviewed files:
  - ci: `.github/workflows/pr-test-amd.yml` modified +81/-47
  - tests: `test/registered/amd/test_deepseek_v3_mtp.py` added +116/-0; `test/registered/amd/test_deepseek_v3_basic.py` added +84/-0; `test/srt/run_suite.py` modified +0/-8; `test/registered/core/test_deterministic.py` modified +5/-1; `test/registered/hicache/test_hicache_storage_3fs_backend.py` modified +2/-1; `test/registered/hicache/test_hicache_storage_file_backend.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py`, `test/registered/amd/test_deepseek_v3_basic.py`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/attention/test_wave_attention_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18252 - [4/N] Quantization Refactor: Quark MoE schemes

- Link: https://github.com/sgl-project/sglang/pull/18252
- Status/date: merged / 2026-02-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/__init__.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` and 7 files; associated commits `150ed881be2c`
- Diff scope read: GitHub Pull Request files API returned 8 files, +396/-243, 835 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "[4/N] Quantization Refactor: Quark MoE schemes". The diff centers on `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/quark.py`. PR body context: ## Motivation Add MoE schemes to quark instead of storing all classes in a single file. Follow up to https://github.com/sgl-project/sglang/pull/17503 Images and motivation for t...
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py` renamed +5/-221 (226 lines); hunks: -9,248 +9,32; -479,7 +263,7 @@ def create_moe_runner(; symbols: QuarkMoEMethod, __init__, get_moe_method, QuarkW4A4MXFp4MoEMethod, touching `QuarkMoEMethod, __init__, get_moe_method`; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` added +213/-0 (213 lines); hunks: -0,0 +1,213; symbols: QuarkW4A4MXFp4MoE, __init__, get_min_capability, create_weights, touching `QuarkW4A4MXFp4MoE, __init__, get_min_capability`; `python/sglang/srt/layers/quantization/quark/quark.py` modified +96/-10 (106 lines); hunks: -2,29 +2,36; -77,7 +84,7 @@ def get_quant_method(; symbols: get_quant_method, _find_matched_config, _get_scheme_from_config, touching `get_quant_method, _find_matched_config, _get_scheme_from_config`; `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py` modified +65/-4 (69 lines); hunks: -1,14 +1,20; -30,6 +36,14 @@ def create_weights(self, *args, **kwargs):; symbols: QuarkScheme, QuarkLinearScheme, used, create_weights, touching `QuarkScheme, QuarkLinearScheme, used`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py` renamed +5/-221 (226 lines); hunks: -9,248 +9,32; -479,7 +263,7 @@ def create_moe_runner(; symbols: QuarkMoEMethod, __init__, get_moe_method, QuarkW4A4MXFp4MoEMethod
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` added +213/-0 (213 lines); hunks: -0,0 +1,213; symbols: QuarkW4A4MXFp4MoE, __init__, get_min_capability, create_weights
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +96/-10 (106 lines); hunks: -2,29 +2,36; -77,7 +84,7 @@ def get_quant_method(; symbols: get_quant_method, _find_matched_config, _get_scheme_from_config
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py` modified +65/-4 (69 lines); hunks: -1,14 +1,20; -30,6 +36,14 @@ def create_weights(self, *args, **kwargs):; symbols: QuarkScheme, QuarkLinearScheme, used, create_weights
  - `python/sglang/srt/layers/quantization/quark/schemes/__init__.py` modified +11/-2 (13 lines); hunks: -1,7 +1,16
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py
@@ -9,248 +9,32 @@
-from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
+from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme
-from sglang.srt.utils import (
-    get_bool_env_var,
-    is_gfx95_supported,
-    is_hip,
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -0,0 +1,213 @@
+# SPDX-License-Identifier: Apache-2.0
+from __future__ import annotations
+import logging
+from typing import TYPE_CHECKING, Any
+import torch
+from sglang.srt.layers.moe import MoeRunnerConfig
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -2,29 +2,36 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py` renamed +5/-221; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` added +213/-0; `python/sglang/srt/layers/quantization/quark/quark.py` modified +96/-10; `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py` modified +65/-4; `python/sglang/srt/layers/quantization/quark/schemes/__init__.py` modified +11/-2; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19422 - [AMD] Use fused GEMM with FP8 cast for FP8 prefill

- Link: https://github.com/sgl-project/sglang/pull/19422
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`; associated commits `5172c378456f`
- Diff scope read: GitHub Pull Request files API returned 3 files, +73/-20, 194 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR optimizes an inference path or backend selection. Title: "[AMD] Use fused GEMM with FP8 cast for FP8 prefill". The diff centers on `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`. PR body context: cc @HaiShaw, @kkHuang-amd ## Motivation When using FP8 prefill, q, k, v tensors require a separate element-wise BF16 to FP8 cast before the prefill attention kernel, adding over...
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +30/-5 (35 lines); hunks: -10,6 +10,9; -87,20 +90,29 @@ def apply_weights(; symbols: apply_weights, touching `apply_weights`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +30/-5 (35 lines); hunks: -10,6 +10,9; -87,20 +90,29 @@ def apply_weights(; symbols: apply_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py
@@ -10,6 +10,9 @@
+    from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import (
+        fused_gemm_afp4wfp4_split_cat,
+    )
@@ -87,20 +90,29 @@ def apply_weights(
+        fused_gemm_split_cat = False
-            ], "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +30/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18684 - [AMD] Add MoE weights and scales padding

- Link: https://github.com/sgl-project/sglang/pull/18684
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; associated commits `941945371314`
- Diff scope read: GitHub Pull Request files API returned 8 files, +131/-36, 388 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "[AMD] Add MoE weights and scales padding". The diff centers on `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`. PR body context: ## Motivation Right now, Aiter MoE requires weights and scales to align with a fixed number. Since some models have intermediate sizes that don't fit this rule, we need to add e...
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +23/-5 (28 lines); hunks: -8,6 +8,7; -73,10 +74,20 @@ def create_weights(; symbols: create_weights, touching `create_weights`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +23/-5 (28 lines); hunks: -8,6 +8,7; -73,10 +74,20 @@ def create_weights(; symbols: create_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -8,6 +8,7 @@
+from sglang.srt.layers.moe.utils import get_moe_weight_sizes
@@ -73,10 +74,20 @@ def create_weights(
+        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
+            intermediate_size_per_partition,
+            is_aiter_moe=True,
+            is_concat=True,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +23/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21067 - Revert "[AMD] Add MoE weights and scales padding"

- Link: https://github.com/sgl-project/sglang/pull/21067
- Status/date: merged / 2026-03-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; associated commits `048d90e1651a`
- Diff scope read: GitHub Pull Request files API returned 8 files, +36/-131, 388 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "Revert "[AMD] Add MoE weights and scales padding"". The diff centers on `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`. PR body context: Reverts sgl-project/sglang#18684 Caused CI failure: https://github.com/sgl-project/sglang/actions/runs/23367673750/job/67984876644
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +5/-23 (28 lines); hunks: -8,7 +8,6; -74,20 +73,10 @@ def create_weights(; symbols: create_weights, touching `create_weights`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +5/-23 (28 lines); hunks: -8,7 +8,6; -74,20 +73,10 @@ def create_weights(; symbols: create_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -8,7 +8,6 @@
-from sglang.srt.layers.moe.utils import get_moe_weight_sizes
@@ -74,20 +73,10 @@ def create_weights(
-        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
-            intermediate_size_per_partition,
-            is_aiter_moe=True,
-            is_concat=True,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +5/-23
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21040 - [AMD][MoRI] Auto-select dispatch quantization type from MoE weight dtype.

- Link: https://github.com/sgl-project/sglang/pull/21040
- Status/date: merged / 2026-03-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; associated commits `61a902ce88ea`
- Diff scope read: GitHub Pull Request files API returned 5 files, +90/-54, 331 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR optimizes an inference path or backend selection. Title: "[AMD][MoRI] Auto-select dispatch quantization type from MoE weight dtype.". The diff centers on `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`. PR body context: ## Motivation Previously, MoRI EP dispatch quantization type (BF16/FP8/FP4) was controlled entirely by environment variables (`SGLANG_MORI_FP8_DISP` / `SGLANG_MORI_FP4_DISP`), r...
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +4/-0 (4 lines); hunks: -160,6 +160,10 @@ def process_weights_after_loading(self, layer: torch.nn.Mod...; symbols: process_weights_after_loading, create_moe_runner, touching `process_weights_after_loading, create_moe_runner`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +4/-0 (4 lines); hunks: -160,6 +160,10 @@ def process_weights_after_loading(self, layer: torch.nn.Mod...; symbols: process_weights_after_loading, create_moe_runner
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -160,6 +160,10 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
+        if hasattr(layer, "dispatcher"):
+            # Weights are stored as torch.uint8 but semantically MXFP4
+            layer.dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_moriep_small.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21097 - [AMD] Add MoE weights and scales padding

- Link: https://github.com/sgl-project/sglang/pull/21097
- Status/date: merged / 2026-04-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; associated commits `f4f9e6818916`
- Diff scope read: GitHub Pull Request files API returned 8 files, +153/-46, 432 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR adds or enables a model support/runtime surface. Title: "[AMD] Add MoE weights and scales padding". The diff centers on `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`. PR body context: ## Motivation Right now, Aiter MoE requires weights and scales to align with a fixed number. Since some models have intermediate sizes that don't fit this rule, we need to add e...
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +20/-5 (25 lines); hunks: -8,6 +8,7; -73,10 +74,20 @@ def create_weights(; symbols: create_weights, touching `create_weights`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +20/-5 (25 lines); hunks: -8,6 +8,7; -73,10 +74,20 @@ def create_weights(; symbols: create_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -8,6 +8,7 @@
+from sglang.srt.layers.moe.utils import get_moe_weight_sizes
@@ -73,10 +74,20 @@ def create_weights(
+        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
+            intermediate_size_per_partition,
+            is_aiter_moe=_use_aiter,
+            is_concat=True,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +20/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23585 - Move expert_mask_gpu from FusedMoE layer to StandardDispatcher

- Link: https://github.com/sgl-project/sglang/pull/23585
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; associated commits `000a2525e196`
- Diff scope read: GitHub Pull Request files API returned 6 files, +25/-25, 124 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Mixtral Quark INT4/FP8 MoE, this PR optimizes an inference path or backend selection. Title: "Move expert_mask_gpu from FusedMoE layer to StandardDispatcher". The diff centers on `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`. PR body context: ## Motivation The aiter `expert_mask_gpu` tensor is a pure function of the dispatcher's `local_expert_mapping` and `num_local_experts`, but today it lives on `FusedMoE` and is c...
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +1/-1 (2 lines); hunks: -227,6 +227,6 @@ def apply_weights(; symbols: apply_weights, touching `apply_weights`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +1/-1 (2 lines); hunks: -227,6 +227,6 @@ def apply_weights(; symbols: apply_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -227,6 +227,6 @@ def apply_weights(
-            expert_mask=layer.expert_mask_gpu,
+            expert_mask=layer.dispatcher.expert_mask_gpu,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/token_dispatcher/standard.py`, `python/sglang/srt/layers/quantization/fp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- This version rejects title-only PR lists; every PR must include trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
