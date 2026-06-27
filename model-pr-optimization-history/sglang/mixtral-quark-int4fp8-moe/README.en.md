# sglang Mixtral Quark INT4/FP8 MoE Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/sglang/srt/layers/quantization/quark/__init__.py` | no direct PR-number commit |
| `python/sglang/srt/layers/quantization/quark/quark.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#13147](https://github.com/sgl-project/sglang/pull/13147), [#18005](https://github.com/sgl-project/sglang/pull/18005), [#18182](https://github.com/sgl-project/sglang/pull/18182), [#18252](https://github.com/sgl-project/sglang/pull/18252), [#27057](https://github.com/sgl-project/sglang/pull/27057), [#28213](https://github.com/sgl-project/sglang/pull/28213) |
| `python/sglang/srt/layers/quantization/quark/schemes/__init__.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py` | [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#18005](https://github.com/sgl-project/sglang/pull/18005), [#18182](https://github.com/sgl-project/sglang/pull/18182), [#18252](https://github.com/sgl-project/sglang/pull/18252), [#19422](https://github.com/sgl-project/sglang/pull/19422), [#28213](https://github.com/sgl-project/sglang/pull/28213) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` | [#18005](https://github.com/sgl-project/sglang/pull/18005), [#18182](https://github.com/sgl-project/sglang/pull/18182), [#18252](https://github.com/sgl-project/sglang/pull/18252), [#18684](https://github.com/sgl-project/sglang/pull/18684), [#21040](https://github.com/sgl-project/sglang/pull/21040), [#21067](https://github.com/sgl-project/sglang/pull/21067), [#21097](https://github.com/sgl-project/sglang/pull/21097), [#23585](https://github.com/sgl-project/sglang/pull/23585), [#23597](https://github.com/sgl-project/sglang/pull/23597), [#23760](https://github.com/sgl-project/sglang/pull/23760), [#28213](https://github.com/sgl-project/sglang/pull/28213) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py` | [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/utils.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#18182](https://github.com/sgl-project/sglang/pull/18182), [#28213](https://github.com/sgl-project/sglang/pull/28213) |
| `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` | [#7392](https://github.com/sgl-project/sglang/pull/7392), [#23597](https://github.com/sgl-project/sglang/pull/23597), [#23760](https://github.com/sgl-project/sglang/pull/23760) |
| `python/sglang/srt/models/mixtral.py` | [#460](https://github.com/sgl-project/sglang/pull/460), [#1081](https://github.com/sgl-project/sglang/pull/1081), [#1290](https://github.com/sgl-project/sglang/pull/1290), [#1418](https://github.com/sgl-project/sglang/pull/1418), [#1835](https://github.com/sgl-project/sglang/pull/1835), [#2156](https://github.com/sgl-project/sglang/pull/2156), [#2163](https://github.com/sgl-project/sglang/pull/2163), [#2300](https://github.com/sgl-project/sglang/pull/2300), [#2371](https://github.com/sgl-project/sglang/pull/2371), [#2563](https://github.com/sgl-project/sglang/pull/2563), [#6223](https://github.com/sgl-project/sglang/pull/6223), [#7966](https://github.com/sgl-project/sglang/pull/7966), ... (17 total) |
| `python/sglang/srt/models/mixtral_quant.py` | [#460](https://github.com/sgl-project/sglang/pull/460), [#1081](https://github.com/sgl-project/sglang/pull/1081) |
| `test/registered/quant/test_quark_mxfp4.py` | [#18005](https://github.com/sgl-project/sglang/pull/18005), [#18182](https://github.com/sgl-project/sglang/pull/18182), [#28213](https://github.com/sgl-project/sglang/pull/28213) |

## PR Coverage Summary

- Git-traced PRs: 33
- Extra PRs preserved from existing docs: 10
- Total PRs in this document: 42
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2024-05-21 | [#460](https://github.com/sgl-project/sglang/pull/460) | merged | port fp8 mixtral | `python/sglang/srt/models/mixtral_quant.py`, `python/sglang/srt/models/mixtral.py` |
| 2024-08-13 | [#1081](https://github.com/sgl-project/sglang/pull/1081) | merged | Update the mixtral to use the better FusedMoE layer | `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py` |
| 2024-09-01 | [#1290](https://github.com/sgl-project/sglang/pull/1290) | merged | fix: resolve fp8 for mixtral | `python/sglang/srt/models/mixtral.py` |
| 2024-09-09 | [#1341](https://github.com/sgl-project/sglang/pull/1341) | merged | Add torchao quant (int4/int8/fp8) to llama models | `python/sglang/srt/layers/torchao_utils.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2024-09-14 | [#1418](https://github.com/sgl-project/sglang/pull/1418) | merged | Add torchao quant for mixtral and qwen_moe | `python/sglang/srt/models/mixtral.py` |
| 2024-10-29 | [#1835](https://github.com/sgl-project/sglang/pull/1835) | merged | [FP8 KV Cache, Mixtral] Avoid KeyError at loading pre-quantized FP8 m… | `python/sglang/srt/models/mixtral.py` |
| 2024-11-24 | [#2156](https://github.com/sgl-project/sglang/pull/2156) | merged | feat: update other MoE models deps | `python/sglang/srt/models/mixtral.py` |
| 2024-11-24 | [#2163](https://github.com/sgl-project/sglang/pull/2163) | merged | Rename triton_fused_moe -> fused_moe_triton | `python/sglang/srt/models/mixtral.py` |
| 2024-12-03 | [#2300](https://github.com/sgl-project/sglang/pull/2300) | merged | Fix gptq for moe layers | `python/sglang/srt/models/mixtral.py` |
| 2024-12-05 | [#2203](https://github.com/sgl-project/sglang/pull/2203) | merged | MoE Expert Parallel Impl | `python/sglang/srt/layers/ep_moe/layer.py`, `python/sglang/srt/layers/ep_moe/kernels.py`, `python/sglang/srt/models/mixtral.py` |
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
| 2026-02-16 | [#17503](https://github.com/sgl-project/sglang/pull/17503) | merged | [2/N] Quantization Refactor: Compressed tensors MoE schemes | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` |
| 2026-02-18 | [#18252](https://github.com/sgl-project/sglang/pull/18252) | merged | [4/N] Quantization Refactor: Quark MoE schemes | `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-02-26 | [#19422](https://github.com/sgl-project/sglang/pull/19422) | merged | [AMD] Use fused GEMM with FP8 cast for FP8 prefill | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` |
| 2026-03-20 | [#18684](https://github.com/sgl-project/sglang/pull/18684) | merged | [AMD] Add MoE weights and scales padding | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-03-21 | [#21067](https://github.com/sgl-project/sglang/pull/21067) | merged | Revert "[AMD] Add MoE weights and scales padding" | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-03-25 | [#21040](https://github.com/sgl-project/sglang/pull/21040) | merged | [AMD][MoRI] Auto-select dispatch quantization type from MoE weight dtype. | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-04-13 | [#21097](https://github.com/sgl-project/sglang/pull/21097) | merged | [AMD] Add MoE weights and scales padding | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-04-24 | [#23585](https://github.com/sgl-project/sglang/pull/23585) | merged | Move expert_mask_gpu from FusedMoE layer to StandardDispatcher | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-05-01 | [#23597](https://github.com/sgl-project/sglang/pull/23597) | merged | [MoE] Add Aiter MoE runner backend and purge aiter.fused_moe from quant methods | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` |
| 2026-05-13 | [#25182](https://github.com/sgl-project/sglang/pull/25182) | merged | chore: add vLLM SPDX copyright headers to ported files | `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py` |
| 2026-05-17 | [#23760](https://github.com/sgl-project/sglang/pull/23760) | merged | [MoE] Unify DeepEPMoE+MoriEPMoE through AITER MoeRunner pre/post-permute | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` |
| 2026-05-18 | [#25390](https://github.com/sgl-project/sglang/pull/25390) | merged | [AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model. | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-06-03 | [#18005](https://github.com/sgl-project/sglang/pull/18005) | merged | [AMD][MXFP4] Online MXFP4 quantization 1/N - dense and MOE models w. original BF16 weight | `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` |
| 2026-06-07 | [#22299](https://github.com/sgl-project/sglang/pull/22299) | merged | [AMD] Enable Piecewise CUDA Graph for AMD GPUs | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-06-10 | [#6238](https://github.com/sgl-project/sglang/pull/6238) | closed | [Feature][ROCM] add online int4_fp8_moe quant feature | `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py`, `python/sglang/srt/layers/quark_utils.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-06-13 | [#27057](https://github.com/sgl-project/sglang/pull/27057) | merged | [AMD] move shared expert check function to quark | `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-06-13 | [#18182](https://github.com/sgl-project/sglang/pull/18182) | merged | [AMD][Quantization] Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-06-14 | [#28213](https://github.com/sgl-project/sglang/pull/28213) | merged | Revert "[AMD][Quantization] Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs" | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |

## Per-PR Diff Audit Cards

### PR #460 - port fp8 mixtral

- Link: https://github.com/sgl-project/sglang/pull/460
- Status/date: merged / 2024-05-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py`; associated commits `0fafc5606b0d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +636/-121, 921 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "port fp8 mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/models/mixtral_quant.py`, `python/sglang/srt/models/mixtral.py`; technical summary: Covers "port fp8 mixtral"; the main implementation surface is `python/sglang/srt/models/mixtral_quant.py`, `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py`; associated commits `ad3e4f16199a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +57/-258, 515 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update the mixtral to use the better FusedMoE layer"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py`; technical summary: Covers "Update the mixtral to use the better FusedMoE layer"; the main implementation surface is `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `9b0805242eea`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: resolve fp8 for mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "fix: resolve fp8 for mixtral"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #1341 - Add torchao quant (int4/int8/fp8) to llama models

- Link: https://github.com/sgl-project/sglang/pull/1341
- Status/date: merged / 2024-09-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +151/-12, 275 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add torchao quant (int4/int8/fp8) to llama models"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/torchao_utils.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/model_executor/model_runner.py`; technical summary: Covers "Add torchao quant (int4/int8/fp8) to llama models"; the main implementation surface is `python/sglang/srt/layers/torchao_utils.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/model_executor/model_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/torchao_utils.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: torchao_quantize_param_data, touching `torchao_quantize_param_data`; `python/sglang/srt/models/llama.py` modified +22/-0 (22 lines); hunks: -42,6 +42,8; -299,6 +301,7 @@ def __init__(; symbols: __init__, load_weights, Phi3ForCausalLM, touching `__init__, load_weights, Phi3ForCausalLM`; `python/sglang/srt/model_executor/model_runner.py` modified +1/-0 (1 lines); hunks: -97,6 +97,7 @@ def __init__(; symbols: __init__, touching `__init__`; `test/srt/test_torchao.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: TestTorchCompile, setUpClass, tearDownClass, test_mmlu, touching `TestTorchCompile, setUpClass, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/layers/torchao_utils.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: torchao_quantize_param_data
  - `python/sglang/srt/models/llama.py` modified +22/-0 (22 lines); hunks: -42,6 +42,8; -299,6 +301,7 @@ def __init__(; symbols: __init__, load_weights, Phi3ForCausalLM
  - `python/sglang/srt/model_executor/model_runner.py` modified +1/-0 (1 lines); hunks: -97,6 +97,7 @@ def __init__(; symbols: __init__
  - `test/srt/test_torchao.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: TestTorchCompile, setUpClass, tearDownClass, test_mmlu
  - `python/sglang/srt/server_args.py` modified +8/-1 (9 lines); hunks: -95,6 +95,7 @@ class ServerArgs:; -443,7 +444,13 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/torchao_utils.py
@@ -0,0 +1,36 @@
+"""
+Common utilities for torchao.
+"""
+import torch
+from torchao.quantization import (
+    int4_weight_only,
diff -- python/sglang/srt/models/llama.py
@@ -42,6 +42,8 @@
+from sglang.srt.layers.torchao_utils import torchao_quantize_param_data
+from sglang.srt.managers.schedule_batch import global_server_args_dict
@@ -299,6 +301,7 @@ def __init__(
+        self.torchao_config = global_server_args_dict["torchao_config"]
@@ -361,6 +364,25 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                if self.torchao_config:
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -97,6 +97,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/torchao_utils.py` added +36/-0; `python/sglang/srt/models/llama.py` modified +22/-0; `python/sglang/srt/model_executor/model_runner.py` modified +1/-0; `python/sglang/srt/server_args.py` modified +8/-1
  - tests: `test/srt/test_torchao.py` added +73/-0; `test/srt/test_moe_eval_accuracy_large.py` modified +3/-3; `test/srt/test_torch_compile.py` modified +3/-3; `test/srt/test_eval_accuracy_mini.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/srt/test_eval_accuracy_mini.py`, `test/srt/test_moe_eval_accuracy_large.py`, `test/srt/test_torch_compile.py`, `test/srt/test_torchao.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1418 - Add torchao quant for mixtral and qwen_moe

- Link: https://github.com/sgl-project/sglang/pull/1418
- Status/date: merged / 2024-09-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `30b404ce72b5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +50/-20, 138 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add torchao quant for mixtral and qwen_moe"; model line: Mixtral Quark INT4/FP8 MoE; category: model support/runtime entry; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "Add torchao quant for mixtral and qwen_moe"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `54dd3ea12277`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[FP8 KV Cache, Mixtral] Avoid KeyError at loading pre-quantized FP8 m…"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "[FP8 KV Cache, Mixtral] Avoid KeyError at loading pre-quantized FP8 m…"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `e3938b2f9c96`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +28/-14, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: update other MoE models deps"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "feat: update other MoE models deps"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `be0124bda09d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 76 files, +19/-19, 199 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Rename triton_fused_moe -> fused_moe_triton"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "Rename triton_fused_moe -> fused_moe_triton"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `1228f7ca69e6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +44/-2, 78 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix gptq for moe layers"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "Fix gptq for moe layers"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #2203 - MoE Expert Parallel Impl

- Link: https://github.com/sgl-project/sglang/pull/2203
- Status/date: merged / 2024-12-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +1172/-8, 1320 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "MoE Expert Parallel Impl"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `python/sglang/srt/layers/ep_moe/layer.py`, `python/sglang/srt/layers/ep_moe/kernels.py`, `python/sglang/srt/models/mixtral.py`; technical summary: Covers "MoE Expert Parallel Impl"; the main implementation surface is `python/sglang/srt/layers/ep_moe/layer.py`, `python/sglang/srt/layers/ep_moe/kernels.py`, `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/ep_moe/layer.py` added +661/-0 (661 lines); hunks: -0,0 +1,661; symbols: GroupedGemmRunner, __init__, _init_flashinfer_wrapper, forward, touching `GroupedGemmRunner, __init__, _init_flashinfer_wrapper`; `python/sglang/srt/layers/ep_moe/kernels.py` added +349/-0 (349 lines); hunks: -0,0 +1,349; symbols: compute_seg_indptr_triton_kernel, compute_src2dst_triton_kernel, run_moe_ep_preproess, pre_reorder_triton_kernel, touching `compute_seg_indptr_triton_kernel, compute_src2dst_triton_kernel, run_moe_ep_preproess`; `python/sglang/srt/models/mixtral.py` modified +13/-5 (18 lines); hunks: -21,9 +21,13; -38,6 +42,7; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`; `python/sglang/srt/models/deepseek_v2.py` modified +5/-3 (8 lines); hunks: -31,6 +31,7; -113,12 +114,12 @@ def __init__(; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `python/sglang/srt/layers/ep_moe/layer.py` added +661/-0 (661 lines); hunks: -0,0 +1,661; symbols: GroupedGemmRunner, __init__, _init_flashinfer_wrapper, forward
  - `python/sglang/srt/layers/ep_moe/kernels.py` added +349/-0 (349 lines); hunks: -0,0 +1,349; symbols: compute_seg_indptr_triton_kernel, compute_src2dst_triton_kernel, run_moe_ep_preproess, pre_reorder_triton_kernel
  - `python/sglang/srt/models/mixtral.py` modified +13/-5 (18 lines); hunks: -21,9 +21,13; -38,6 +42,7; symbols: __init__, forward, load_weights
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-3 (8 lines); hunks: -31,6 +31,7; -113,12 +114,12 @@ def __init__(; symbols: __init__, load_weights
  - `python/sglang/srt/model_executor/model_runner.py` modified +1/-0 (1 lines); hunks: -141,6 +141,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/ep_moe/layer.py
@@ -0,0 +1,661 @@
+import logging
+from typing import Callable, List, Optional, Tuple
+import torch
+from torch.nn import Module
+from vllm import _custom_ops as ops
+from vllm.distributed import (
diff -- python/sglang/srt/layers/ep_moe/kernels.py
@@ -0,0 +1,349 @@
+import logging
+from typing import Optional
+import torch
+import triton
+import triton.language as tl
+logger = logging.getLogger(__name__)
diff -- python/sglang/srt/models/mixtral.py
@@ -21,9 +21,13 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/ep_moe/layer.py` added +661/-0; `python/sglang/srt/layers/ep_moe/kernels.py` added +349/-0; `python/sglang/srt/models/mixtral.py` modified +13/-5; `python/sglang/srt/models/deepseek_v2.py` modified +5/-3; `python/sglang/srt/model_executor/model_runner.py` modified +1/-0; `python/sglang/srt/layers/ep_moe/__init__.py` added +0/-0
  - tests: `test/srt/test_moe_ep.py` added +113/-0
- Risk and verification: The diff ships test coverage in `test/srt/test_moe_ep.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #2371 - MoE Expert Parallel

- Link: https://github.com/sgl-project/sglang/pull/2371
- Status/date: merged / 2024-12-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `3d32e4a32c4c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +1172/-8, 1320 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "MoE Expert Parallel"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "MoE Expert Parallel"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `e835a50021e0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 88 files, +338/-344, 1108 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Reorg moe code"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "Reorg moe code"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `bad7c26fdc7f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +179/-47, 391 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[PP] Fix init_memory_pool desync & add PP for mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "[PP] Fix init_memory_pool desync & add PP for mixtral"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `15ad6c908670`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 39 files, +557/-872, 2848 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[1/N] MoE Refactor: refactor `select_experts`"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "[1/N] MoE Refactor: refactor `select_experts`"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `9effeb5bddf2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +107/-11, 407 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support EPLB in FusedMoE"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "Support EPLB in FusedMoE"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `6c88f6c8d908`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 38 files, +342/-299, 1748 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[5/N] MoE Refactor: Update MoE parallelism arguments"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "[5/N] MoE Refactor: Update MoE parallelism arguments"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `295895120df4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 69 files, +958/-1039, 4640 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[6/N] MoE Refactor: Cleanup MoE-related configs"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "[6/N] MoE Refactor: Cleanup MoE-related configs"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `3c06b673aff9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +496/-1778, 2897 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[8/N] MoE Refactor: deprecate `EPMoE`"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "[8/N] MoE Refactor: deprecate `EPMoE`"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/__init__.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `python/sglang/srt/layers/quantization/quark/utils.py`; associated commits `67e9d287eea0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +666/-243, 1041 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] Support Quark Dense + MoE FP8 & FP8 PTPC"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `python/sglang/srt/layers/quantization/quark/quark.py`; technical summary: Covers "[Quantization] Support Quark Dense + MoE FP8 & FP8 PTPC"; the main implementation surface is `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `python/sglang/srt/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mixtral.py`; associated commits `b5344b31b8f1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +24/-160, 315 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Piecewise CUDA Graph] Fix recompile issue for Mixtral and Grok2"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `python/sglang/srt/models/mixtral.py`; technical summary: Covers "[Piecewise CUDA Graph] Fix recompile issue for Mixtral and Grok2"; the main implementation surface is `python/sglang/srt/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/quark.py`; associated commits `c106b54b57d8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +594/-96, 1032 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Aiter fp8 kv cache"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/quark.py`; technical summary: Covers "Aiter fp8 kv cache"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`; technical summary: Covers "[AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm"; the main implementation surface is `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[AMD CI] Migrate and Add More Testcases"; model line: Mixtral Quark INT4/FP8 MoE; category: docs/tests/CI; main diff: `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py`; technical summary: Covers "[AMD CI] Migrate and Add More Testcases"; the main implementation surface is `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #17503 - [2/N] Quantization Refactor: Compressed tensors MoE schemes

- Link: https://github.com/sgl-project/sglang/pull/17503
- Status/date: merged / 2026-02-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +2643/-2237, 5144 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[2/N] Quantization Refactor: Compressed tensors MoE schemes"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`; technical summary: Covers "[2/N] Quantization Refactor: Compressed tensors MoE schemes"; the main implementation surface is `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` removed +0/-2190 (2190 lines); hunks: -1,2190 +0,0; symbols: GPTQMarlinState, CompressedTensorsMoEMethod, __new__, get_moe_method, touching `GPTQMarlinState, CompressedTensorsMoEMethod, __new__`; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` added +621/-0 (621 lines); hunks: -0,0 +1,621; symbols: GPTQMarlinState, CompressedTensorsWNA16MoE, __init__, get_min_capability, touching `GPTQMarlinState, CompressedTensorsWNA16MoE, __init__`; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` added +421/-0 (421 lines); hunks: -0,0 +1,421; symbols: CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability, create_weights, touching `CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability`; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: CompressedTensorsW8A8Fp8MoE, __init__, get_min_capability, create_weights, touching `CompressedTensorsW8A8Fp8MoE, __init__, get_min_capability`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` removed +0/-2190 (2190 lines); hunks: -1,2190 +0,0; symbols: GPTQMarlinState, CompressedTensorsMoEMethod, __new__, get_moe_method
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` added +621/-0 (621 lines); hunks: -0,0 +1,621; symbols: GPTQMarlinState, CompressedTensorsWNA16MoE, __init__, get_min_capability
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` added +421/-0 (421 lines); hunks: -0,0 +1,421; symbols: CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability, create_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: CompressedTensorsW8A8Fp8MoE, __init__, get_min_capability, create_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py` added +358/-0 (358 lines); hunks: -0,0 +1,358; symbols: CompressedTensorsMxInt4MoE, __init__, get_min_capability, create_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -1,2190 +0,0 @@
-# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
-# SPDX-License-Identifier: Apache-2.0
-from __future__ import annotations
-import enum
-import logging
-from enum import Enum
diff -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py
@@ -0,0 +1,621 @@
+from __future__ import annotations
+import enum
+import logging
+from enum import Enum
+from typing import TYPE_CHECKING
+import torch
diff -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py
@@ -0,0 +1,421 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` removed +0/-2190; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` added +621/-0; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` added +421/-0; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py` added +384/-0; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py` added +358/-0; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int8_moe.py` added +293/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18252 - [4/N] Quantization Refactor: Quark MoE schemes

- Link: https://github.com/sgl-project/sglang/pull/18252
- Status/date: merged / 2026-02-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/__init__.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` and 7 files; associated commits `150ed881be2c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +396/-243, 835 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[4/N] Quantization Refactor: Quark MoE schemes"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/quark.py`; technical summary: Covers "[4/N] Quantization Refactor: Quark MoE schemes"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`; associated commits `5172c378456f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +73/-20, 194 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Use fused GEMM with FP8 cast for FP8 prefill"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`; technical summary: Covers "[AMD] Use fused GEMM with FP8 cast for FP8 prefill"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; associated commits `941945371314`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +131/-36, 388 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add MoE weights and scales padding"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; technical summary: Covers "[AMD] Add MoE weights and scales padding"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; associated commits `048d90e1651a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +36/-131, 388 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[AMD] Add MoE weights and scales padding""; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; technical summary: Covers "Revert "[AMD] Add MoE weights and scales padding""; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; associated commits `61a902ce88ea`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +90/-54, 331 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][MoRI] Auto-select dispatch quantization type from MoE weight dtype."; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; technical summary: Covers "[AMD][MoRI] Auto-select dispatch quantization type from MoE weight dtype."; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; associated commits `f4f9e6818916`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +153/-46, 432 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add MoE weights and scales padding"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; technical summary: Covers "[AMD] Add MoE weights and scales padding"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; associated commits `000a2525e196`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +25/-25, 124 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Move expert_mask_gpu from FusedMoE layer to StandardDispatcher"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`; technical summary: Covers "Move expert_mask_gpu from FusedMoE layer to StandardDispatcher"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #23597 - [MoE] Add Aiter MoE runner backend and purge aiter.fused_moe from quant methods

- Link: https://github.com/sgl-project/sglang/pull/23597
- Status/date: merged / 2026-05-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`; associated commits `108bfd8b6a0d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +316/-251, 873 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE] Add Aiter MoE runner backend and purge aiter.fused_moe from quant methods"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`; technical summary: Covers "[MoE] Add Aiter MoE runner backend and purge aiter.fused_moe from quant methods"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +25/-29 (54 lines); hunks: -7,7 +7,7; -32,8 +32,6; symbols: process_weights_after_loading, create_moe_runner, apply_weights, touching `process_weights_after_loading, create_moe_runner, apply_weights`; `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` modified +25/-22 (47 lines); hunks: -11,7 +11,7; -27,8 +27,6; symbols: process_weights_after_loading, create_moe_runner, apply, touching `process_weights_after_loading, create_moe_runner, apply`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +25/-29 (54 lines); hunks: -7,7 +7,7; -32,8 +32,6; symbols: process_weights_after_loading, create_moe_runner, apply_weights
  - `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` modified +25/-22 (47 lines); hunks: -11,7 +11,7; -27,8 +27,6; symbols: process_weights_after_loading, create_moe_runner, apply
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -7,7 +7,7 @@
-from sglang.srt.layers.moe import MoeRunnerConfig
+from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
@@ -32,8 +32,6 @@
-    from aiter import ActivationType, QuantType
-    from aiter.fused_moe import fused_moe
@@ -182,24 +180,31 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
diff -- python/sglang/srt/layers/quantization/quark_int4fp8_moe.py
@@ -11,7 +11,7 @@
-from sglang.srt.layers.moe import MoeRunnerConfig
+from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
@@ -27,8 +27,6 @@
-    from aiter import ActivationType, QuantType
-    from aiter.fused_moe import fused_moe
@@ -405,39 +403,44 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +25/-29; `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` modified +25/-22
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/moe_runner/aiter.py`, `python/sglang/srt/layers/moe/moe_runner/runner.py`, `python/sglang/srt/layers/moe/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25182 - chore: add vLLM SPDX copyright headers to ported files

- Link: https://github.com/sgl-project/sglang/pull/25182
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 136 files, +255/-0, 872 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "chore: add vLLM SPDX copyright headers to ported files"; model line: Mixtral Quark INT4/FP8 MoE; category: model support/runtime entry; main diff: `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`; technical summary: Covers "chore: add vLLM SPDX copyright headers to ported files"; the main implementation surface is `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7; `python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7; `python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6; `python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6.
- Code diff details:
  - `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma2.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/baichuan.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/baichuan.py
diff -- python/sglang/srt/models/commandr.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/commandr.py
diff -- python/sglang/srt/models/dbrx.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
diff -- python/sglang/srt/models/gemma.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/baichuan.py` modified +4/-0; `python/sglang/srt/models/commandr.py` modified +4/-0; `python/sglang/srt/models/dbrx.py` modified +3/-0; `python/sglang/srt/models/gemma.py` modified +3/-0; `python/sglang/srt/models/gemma2.py` modified +3/-0; `python/sglang/srt/models/gpt_bigcode.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_custom_ops.py`, `python/sglang/test/test_marlin_utils.py`, `sgl-kernel/tests/test_causal_conv1d.py`, `test/registered/layers/mamba/test_causal_conv1d.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23760 - [MoE] Unify DeepEPMoE+MoriEPMoE through AITER MoeRunner pre/post-permute

- Link: https://github.com/sgl-project/sglang/pull/23760
- Status/date: merged / 2026-05-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`; associated commits `be3c425788db`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +398/-330, 954 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE] Unify DeepEPMoE+MoriEPMoE through AITER MoeRunner pre/post-permute"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`; technical summary: Covers "[MoE] Unify DeepEPMoE+MoriEPMoE through AITER MoeRunner pre/post-permute"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +1/-1 (2 lines); hunks: -187,7 +187,7 @@ def create_moe_runner(; symbols: create_moe_runner, touching `create_moe_runner`; `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` modified +1/-1 (2 lines); hunks: -410,7 +410,7 @@ def create_moe_runner(; symbols: create_moe_runner, touching `create_moe_runner`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +1/-1 (2 lines); hunks: -187,7 +187,7 @@ def create_moe_runner(; symbols: create_moe_runner
  - `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` modified +1/-1 (2 lines); hunks: -410,7 +410,7 @@ def create_moe_runner(; symbols: create_moe_runner
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -187,7 +187,7 @@ def create_moe_runner(
-        if moe_runner_backend.is_auto() and get_moe_a2a_backend().is_none():
+        if moe_runner_backend.is_auto() and get_moe_a2a_backend().supports_aiter():
diff -- python/sglang/srt/layers/quantization/quark_int4fp8_moe.py
@@ -410,7 +410,7 @@ def create_moe_runner(
-        if moe_runner_backend.is_auto() and get_moe_a2a_backend().is_none():
+        if moe_runner_backend.is_auto() and get_moe_a2a_backend().supports_aiter():
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +1/-1; `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/batch_overlap/two_batch_overlap.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/moe_runner/aiter.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25390 - [AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model.

- Link: https://github.com/sgl-project/sglang/pull/25390
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-2, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model."; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/quark.py`; technical summary: Covers "[AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model."; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -2355,6 +2355,12 @@ def __init__(; -2422,7 +2428,11 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts, touching `__init__, determine_num_fused_shared_experts`; `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-1 (8 lines); hunks: -71,7 +71,13 @@ def get_name(self) -> str:; symbols: get_name, apply_weight_name_mapper, get_quant_method, touching `get_name, apply_weight_name_mapper, get_quant_method`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -2355,6 +2355,12 @@ def __init__(; -2422,7 +2428,11 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-1 (8 lines); hunks: -71,7 +71,13 @@ def get_name(self) -> str:; symbols: get_name, apply_weight_name_mapper, get_quant_method
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2355,6 +2355,12 @@ def __init__(
+        # Quant configs like Quark may rely on the model to provide fused-module
+        # mappings so exclusion checks can unfuse derived names back to the
+        # checkpoint's source layer names.
+        if quant_config is not None and hasattr(quant_config, "packed_modules_mapping"):
+            quant_config.packed_modules_mapping = self.packed_modules_mapping
@@ -2422,7 +2428,11 @@ def determine_num_fused_shared_experts(
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -71,7 +71,13 @@ def get_name(self) -> str:
-        self.exclude_layers = hf_to_sglang_mapper.apply_list(self.exclude_layers)
+        mapped = hf_to_sglang_mapper.apply_list(self.exclude_layers)
+        expanded = []
+        for name in mapped:
+            expanded.append(name)
+            if name.startswith("language_model."):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1; `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18005 - [AMD][MXFP4] Online MXFP4 quantization 1/N - dense and MOE models w. original BF16 weight

- Link: https://github.com/sgl-project/sglang/pull/18005
- Status/date: merged / 2026-06-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `test/registered/quant/test_quark_mxfp4.py`; associated commits `293816ab14af`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +509/-26, 877 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][MXFP4] Online MXFP4 quantization 1/N - dense and MOE models w. original BF16 weight"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`; technical summary: Covers "[AMD][MXFP4] Online MXFP4 quantization 1/N - dense and MOE models w. original BF16 weight"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/quark/quark.py` modified +103/-3 (106 lines); hunks: -29,6 +29,8; -40,21 +42,47 @@ class QuarkConfig(QuantizationConfig):; symbols: QuarkConfig, __init__, quantized_layers, get_linear_method, touching `QuarkConfig, __init__, quantized_layers`; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +77/-10 (87 lines); hunks: -16,6 +16,7; -35,14 +36,25; symbols: QuarkW4A4MXFp4MoE, __init__, get_min_capability, touching `QuarkW4A4MXFp4MoE, __init__, get_min_capability`; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +68/-6 (74 lines); hunks: -1,12 +1,14; -19,26 +21,43; symbols: QuarkW4A4MXFP4, __init__, get_min_capability, process_weights_after_loading, touching `QuarkW4A4MXFP4, __init__, get_min_capability`; `test/registered/quant/test_quark_mxfp4.py` added +188/-0 (188 lines); hunks: -0,0 +1,188; symbols: TestOnlineQuantizationMemoryLoad, setUpClass, _extract_peak_memory_before_load, _extract_memory_increase_load_weights, touching `TestOnlineQuantizationMemoryLoad, setUpClass, _extract_peak_memory_before_load`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +103/-3 (106 lines); hunks: -29,6 +29,8; -40,21 +42,47 @@ class QuarkConfig(QuantizationConfig):; symbols: QuarkConfig, __init__, quantized_layers, get_linear_method
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +77/-10 (87 lines); hunks: -16,6 +16,7; -35,14 +36,25; symbols: QuarkW4A4MXFp4MoE, __init__, get_min_capability
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +68/-6 (74 lines); hunks: -1,12 +1,14; -19,26 +21,43; symbols: QuarkW4A4MXFP4, __init__, get_min_capability, process_weights_after_loading
  - `test/registered/quant/test_quark_mxfp4.py` added +188/-0 (188 lines); hunks: -0,0 +1,188; symbols: TestOnlineQuantizationMemoryLoad, setUpClass, _extract_peak_memory_before_load, _extract_memory_increase_load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -29,6 +29,8 @@
+    from transformers import PretrainedConfig
@@ -40,21 +42,47 @@ class QuarkConfig(QuantizationConfig):
-        quant_config: dict[str, Any],
+        quant_config: Optional[dict[str, Any]] = None,
+        hf_config: "PretrainedConfig | None" = None,
+        is_prequantized: bool = False,
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -16,6 +16,7 @@
+from sglang.srt.utils.common import mxfp_supported
@@ -35,14 +36,25 @@
+if _is_hip:
+    from aiter.ops.triton.quant import dynamic_mxfp4_quant
+else:
+    dynamic_mxfp4_quant = None
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py
@@ -1,12 +1,14 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/quark.py` modified +103/-3; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +77/-10; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +68/-6
  - tests: `test/registered/quant/test_quark_mxfp4.py` added +188/-0
- Risk and verification: The diff ships test coverage in `test/registered/quant/test_quark_mxfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22299 - [AMD] Enable Piecewise CUDA Graph for AMD GPUs

- Link: https://github.com/sgl-project/sglang/pull/22299
- Status/date: merged / 2026-06-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +335/-32, 583 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Enable Piecewise CUDA Graph for AMD GPUs"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`; technical summary: Covers "[AMD] Enable Piecewise CUDA Graph for AMD GPUs"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +134/-5 (139 lines); hunks: -8,16 +8,145; symbols: _aiter_gemm_afp4wfp4, _aiter_gemm_afp4wfp4_fake, gemm_afp4wfp4, _aiter_gemm_afp4wfp4_pre_quant, touching `_aiter_gemm_afp4wfp4, _aiter_gemm_afp4wfp4_fake, gemm_afp4wfp4`; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +66/-12 (78 lines); hunks: -61,12 +61,16; -280,29 +284,75 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, _pre_warm_aiter_chip_info, warmup_compile, replay, touching `__init__, _pre_warm_aiter_chip_info, warmup_compile`; `python/sglang/srt/model_executor/model_runner.py` modified +37/-6 (43 lines); hunks: -35,6 +35,10; -2971,6 +2975,8 @@ def init_piecewise_cuda_graphs(self, force_for_draft_worke...; symbols: init_piecewise_cuda_graphs, forward_extend, forward_idle, touching `init_piecewise_cuda_graphs, forward_extend, forward_idle`; `python/sglang/srt/layers/radix_attention.py` modified +29/-0 (29 lines); hunks: -30,8 +30,11; -179,6 +182,13 @@ def unified_attention_with_output(; symbols: unified_attention_with_output, touching `unified_attention_with_output`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +134/-5 (139 lines); hunks: -8,16 +8,145; symbols: _aiter_gemm_afp4wfp4, _aiter_gemm_afp4wfp4_fake, gemm_afp4wfp4, _aiter_gemm_afp4wfp4_pre_quant
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +66/-12 (78 lines); hunks: -61,12 +61,16; -280,29 +284,75 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, _pre_warm_aiter_chip_info, warmup_compile, replay
  - `python/sglang/srt/model_executor/model_runner.py` modified +37/-6 (43 lines); hunks: -35,6 +35,10; -2971,6 +2975,8 @@ def init_piecewise_cuda_graphs(self, force_for_draft_worke...; symbols: init_piecewise_cuda_graphs, forward_extend, forward_idle
  - `python/sglang/srt/layers/radix_attention.py` modified +29/-0 (29 lines); hunks: -30,8 +30,11; -179,6 +182,13 @@ def unified_attention_with_output(; symbols: unified_attention_with_output
  - `python/sglang/srt/layers/moe/topk.py` modified +22/-0 (22 lines); hunks: -1125,6 +1125,16 @@ def _mask_topk_ids_padded_region(; -1495,6 +1505,12 @@ def _post_process_topk_ids(; symbols: _mask_topk_ids_padded_region, _zero_topk_weights_padded_region, _biased_grouped_topk_postprocess, _post_process_topk_ids
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py
@@ -8,16 +8,145 @@
-from sglang.srt.utils.common import mxfp_supported
+from sglang.srt.utils.common import direct_register_custom_op, mxfp_supported
-        fused_gemm_afp4wfp4_split_cat,
+        fused_gemm_afp4wfp4_split_cat as _fused_gemm_afp4wfp4_split_cat_orig,
-    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
-    from aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic import gemm_afp4wfp4_pre_quant
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -61,12 +61,16 @@
+    get_bool_env_var,
+    is_hip,
+_is_hip = is_hip()
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
@@ -280,29 +284,75 @@ def __init__(self, model_runner: ModelRunner):
-                with enable_piecewise_cuda_graph_compile():
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -35,6 +35,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +134/-5; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +66/-12; `python/sglang/srt/model_executor/model_runner.py` modified +37/-6; `python/sglang/srt/layers/radix_attention.py` modified +29/-0; `python/sglang/srt/layers/moe/topk.py` modified +22/-0; `python/sglang/srt/layers/moe/hash_topk.py` modified +11/-2
  - tests: `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py` modified +7/-2
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py`, `test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6238 - [Feature][ROCM] add online int4_fp8_moe quant feature

- Link: https://github.com/sgl-project/sglang/pull/6238
- Status/date: closed / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +651/-3, 752 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature][ROCM] add online int4_fp8_moe quant feature"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py`, `python/sglang/srt/layers/quark_utils.py`, `python/sglang/srt/model_executor/model_runner.py`; technical summary: Covers "[Feature][ROCM] add online int4_fp8_moe quant feature"; the main implementation surface is `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py`, `python/sglang/srt/layers/quark_utils.py`, `python/sglang/srt/model_executor/model_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py` added +524/-0 (524 lines); hunks: -0,0 +1,524; symbols: dummy_func, QuarkInt4Fp8Config, for, __init__, touching `dummy_func, QuarkInt4Fp8Config, for`; `python/sglang/srt/layers/quark_utils.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: apply_quark_quant_config_to_model, online_quant, quantize_fp8_scale_tensorwise, quantize_int4_scale_columnwise, touching `apply_quark_quant_config_to_model, online_quant, quantize_fp8_scale_tensorwise`; `python/sglang/srt/model_executor/model_runner.py` modified +6/-0 (6 lines); hunks: -49,6 +49,7; -168,6 +169,7 @@ def __init__(; symbols: __init__, initialize, touching `__init__, initialize`; `python/sglang/srt/layers/quantization/__init__.py` modified +2/-0 (2 lines); hunks: -66,6 +66,7 @@ def override_quantization_method(self, *args, **kwargs):; -77,6 +78,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method, touching `override_quantization_method`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py` added +524/-0 (524 lines); hunks: -0,0 +1,524; symbols: dummy_func, QuarkInt4Fp8Config, for, __init__
  - `python/sglang/srt/layers/quark_utils.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: apply_quark_quant_config_to_model, online_quant, quantize_fp8_scale_tensorwise, quantize_int4_scale_columnwise
  - `python/sglang/srt/model_executor/model_runner.py` modified +6/-0 (6 lines); hunks: -49,6 +49,7; -168,6 +169,7 @@ def __init__(; symbols: __init__, initialize
  - `python/sglang/srt/layers/quantization/__init__.py` modified +2/-0 (2 lines); hunks: -66,6 +66,7 @@ def override_quantization_method(self, *args, **kwargs):; -77,6 +78,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method
  - `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: -317,6 +317,7 @@ def _verify_quantization(self) -> None:; symbols: _verify_quantization
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py
@@ -0,0 +1,524 @@
+import logging
+from typing import Any, Callable, Dict, List, Optional
+import torch
+from torch.nn.parameter import Parameter
+try:
+    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
diff -- python/sglang/srt/layers/quark_utils.py
@@ -0,0 +1,104 @@
+"""
+Common utilities for quark.
+"""
+import logging
+from tqdm.auto import tqdm
+import torch
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -49,6 +49,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py` added +524/-0; `python/sglang/srt/layers/quark_utils.py` added +104/-0; `python/sglang/srt/model_executor/model_runner.py` modified +6/-0; `python/sglang/srt/layers/quantization/__init__.py` modified +2/-0; `python/sglang/srt/configs/model_config.py` modified +1/-0; `python/sglang/srt/layers/linear.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27057 - [AMD] move shared expert check function to quark

- Link: https://github.com/sgl-project/sglang/pull/27057
- Status/date: merged / 2026-06-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/quark.py`; associated commits `f288283c07a4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +74/-11, 131 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] move shared expert check function to quark"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `python/sglang/srt/layers/quantization/quark/quark.py`; technical summary: Covers "[AMD] move shared expert check function to quark"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/quark/quark.py` modified +44/-0 (44 lines); hunks: -37,6 +37,18; -492,6 +504,38 @@ def get_moe_scheme(; symbols: QuarkConfig, get_moe_scheme, get_scaled_act_names, can_fuse_shared_expert, touching `QuarkConfig, get_moe_scheme, get_scaled_act_names`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +44/-0 (44 lines); hunks: -37,6 +37,18; -492,6 +504,38 @@ def get_moe_scheme(; symbols: QuarkConfig, get_moe_scheme, get_scaled_act_names, can_fuse_shared_expert
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -37,6 +37,18 @@
+_MOE_SHARED_EXPERT_QUANT_LAYER0_BASES: tuple[str, ...] = (
+    "model.layers.0",
+    "model.language_model.layers.0",
+)
+_SHARED_EXPERT_BODY_PROJ_SUFFIXES: tuple[str, ...] = (
+    "gate_proj",
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/quark.py` modified +44/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18182 - [AMD][Quantization] Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs

- Link: https://github.com/sgl-project/sglang/pull/18182
- Status/date: merged / 2026-06-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/utils.py`, `test/registered/quant/test_quark_mxfp4.py`; associated commits `3f4a338212b2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +1042/-130, 1694 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][Quantization] Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark.py`; technical summary: Covers "[AMD][Quantization] Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs"; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +374/-28 (402 lines); hunks: -3,12 +3,20; -51,10 +59,12 @@ def __init__(; symbols: __init__, create_weights, get_online_weight_loader, touching `__init__, create_weights, get_online_weight_loader`; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +214/-33 (247 lines); hunks: -1,11 +1,19; -162,12 +170,14 @@ def __init__(; symbols: __init__, create_weights, get_online_mxfp4_weight_loader, online_mxfp4_weight_loader, touching `__init__, create_weights, get_online_mxfp4_weight_loader`; `python/sglang/srt/layers/quantization/quark/quark.py` modified +61/-11 (72 lines); hunks: -14,6 +14,7; -54,13 +55,14 @@ class QuarkConfig(QuantizationConfig):; symbols: QuarkConfig, __init__, quantized_layers, touching `QuarkConfig, __init__, quantized_layers`; `python/sglang/srt/layers/quantization/quark/utils.py` modified +4/-0 (4 lines); hunks: -210,5 +210,9 @@ def quark_post_load_weights(self_attn: nn.Module, w: torch.T...; symbols: quark_post_load_weights, touching `quark_post_load_weights`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +374/-28 (402 lines); hunks: -3,12 +3,20; -51,10 +59,12 @@ def __init__(; symbols: __init__, create_weights, get_online_weight_loader
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +214/-33 (247 lines); hunks: -1,11 +1,19; -162,12 +170,14 @@ def __init__(; symbols: __init__, create_weights, get_online_mxfp4_weight_loader, online_mxfp4_weight_loader
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +61/-11 (72 lines); hunks: -14,6 +14,7; -54,13 +55,14 @@ class QuarkConfig(QuantizationConfig):; symbols: QuarkConfig, __init__, quantized_layers
  - `python/sglang/srt/layers/quantization/quark/utils.py` modified +4/-0 (4 lines); hunks: -210,5 +210,9 @@ def quark_post_load_weights(self_attn: nn.Module, w: torch.T...; symbols: quark_post_load_weights
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +2/-0 (2 lines); hunks: -1216,6 +1216,8 @@ def block_quant_dequant(; symbols: block_quant_dequant
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -3,12 +3,20 @@
+import threading
+from sglang.srt.layers.quantization.base_config import QuantizationConfig
+from sglang.srt.layers.quantization.dequantization import (
+    copy_missing_attrs,
+    dequantize_fp8,
+)
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py
@@ -1,11 +1,19 @@
+import threading
+from sglang.srt.layers.quantization import QuantizationConfig
+from sglang.srt.layers.quantization.dequantization import (
+    copy_missing_attrs,
+    dequantize_fp8,
+)
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -14,6 +14,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +374/-28; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +214/-33; `python/sglang/srt/layers/quantization/quark/quark.py` modified +61/-11; `python/sglang/srt/layers/quantization/quark/utils.py` modified +4/-0; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +2/-0; `python/sglang/srt/model_loader/utils.py` modified +15/-1
  - tests: `test/registered/quant/test_quark_mxfp4.py` modified +134/-0
- Risk and verification: The diff ships test coverage in `test/registered/quant/test_quark_mxfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28213 - Revert "[AMD][Quantization] Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs"

- Link: https://github.com/sgl-project/sglang/pull/28213
- Status/date: merged / 2026-06-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/utils.py`, `test/registered/quant/test_quark_mxfp4.py`; associated commits `f18d38d04084`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +130/-1042, 1694 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[AMD][Quantization] Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs""; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark.py`; technical summary: Covers "Revert "[AMD][Quantization] Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs""; the main implementation surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +28/-374 (402 lines); hunks: -3,20 +3,12; -59,12 +51,10 @@ def __init__(; symbols: __init__, create_weights, get_online_weight_loader, touching `__init__, create_weights, get_online_weight_loader`; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +33/-214 (247 lines); hunks: -1,19 +1,11; -170,14 +162,12 @@ def __init__(; symbols: __init__, create_weights, get_online_mxfp4_weight_loader, online_mxfp4_weight_loader, touching `__init__, create_weights, get_online_mxfp4_weight_loader`; `python/sglang/srt/layers/quantization/quark/quark.py` modified +11/-61 (72 lines); hunks: -14,7 +14,6; -55,14 +54,13 @@ class QuarkConfig(QuantizationConfig):; symbols: QuarkConfig, __init__, quantized_layers, touching `QuarkConfig, __init__, quantized_layers`; `python/sglang/srt/layers/quantization/quark/utils.py` modified +0/-4 (4 lines); hunks: -210,9 +210,5 @@ def quark_post_load_weights(self_attn: nn.Module, w: torch.T...; symbols: quark_post_load_weights, touching `quark_post_load_weights`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +28/-374 (402 lines); hunks: -3,20 +3,12; -59,12 +51,10 @@ def __init__(; symbols: __init__, create_weights, get_online_weight_loader
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +33/-214 (247 lines); hunks: -1,19 +1,11; -170,14 +162,12 @@ def __init__(; symbols: __init__, create_weights, get_online_mxfp4_weight_loader, online_mxfp4_weight_loader
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +11/-61 (72 lines); hunks: -14,7 +14,6; -55,14 +54,13 @@ class QuarkConfig(QuantizationConfig):; symbols: QuarkConfig, __init__, quantized_layers
  - `python/sglang/srt/layers/quantization/quark/utils.py` modified +0/-4 (4 lines); hunks: -210,9 +210,5 @@ def quark_post_load_weights(self_attn: nn.Module, w: torch.T...; symbols: quark_post_load_weights
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +0/-2 (2 lines); hunks: -1216,8 +1216,6 @@ def block_quant_dequant(; symbols: block_quant_dequant
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -3,20 +3,12 @@
-import threading
-from sglang.srt.layers.quantization.base_config import QuantizationConfig
-from sglang.srt.layers.quantization.dequantization import (
-    copy_missing_attrs,
-    dequantize_fp8,
-)
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py
@@ -1,19 +1,11 @@
-import threading
-from sglang.srt.layers.quantization import QuantizationConfig
-from sglang.srt.layers.quantization.dequantization import (
-    copy_missing_attrs,
-    dequantize_fp8,
-)
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -14,7 +14,6 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +28/-374; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +33/-214; `python/sglang/srt/layers/quantization/quark/quark.py` modified +11/-61; `python/sglang/srt/layers/quantization/quark/utils.py` modified +0/-4; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +0/-2; `python/sglang/srt/model_loader/utils.py` modified +1/-15
  - tests: `test/registered/quant/test_quark_mxfp4.py` modified +0/-134
- Risk and verification: The diff ships test coverage in `test/registered/quant/test_quark_mxfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- Link: https://github.com/sgl-project/sglang/pull/28567
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 184 files, +1865/-1727, 8932 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_parallel(): a structured accessor for parallel-topology state"; model line: Mixtral Quark INT4/FP8 MoE; category: model support/runtime entry; main diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add get_parallel(): a structured accessor for parallel-topology state"; the main implementation surface is `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
