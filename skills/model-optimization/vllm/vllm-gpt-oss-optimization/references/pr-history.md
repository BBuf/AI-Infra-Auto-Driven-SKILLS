# vllm GPT-OSS PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 4
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/evals/gpt_oss/README.md` | no direct PR-number commit |
| `tests/evals/gpt_oss/__init__.py` | [#24920](https://github.com/vllm-project/vllm/pull/24920) |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-baseline.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-flashinfer-mxfp4-bf16.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-flashinfer-mxfp4-mxfp8-cutlass.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-marlin.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml` | [#36179](https://github.com/vllm-project/vllm/pull/36179) |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-sm100-fi-mxfp4-mxfp8-trtllm.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/models-b200.txt` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/models-gfx942.txt` | [#36179](https://github.com/vllm-project/vllm/pull/36179) |
| `tests/evals/gpt_oss/configs/models-gfx950.txt` | [#36179](https://github.com/vllm-project/vllm/pull/36179), [#38292](https://github.com/vllm-project/vllm/pull/38292) |
| `tests/evals/gpt_oss/configs/models-h100.txt` | no direct PR-number commit |
| `tests/evals/gpt_oss/conftest.py` | [#24920](https://github.com/vllm-project/vllm/pull/24920) |
| `tests/evals/gpt_oss/test_gpqa_correctness.py` | [#24920](https://github.com/vllm-project/vllm/pull/24920), [#26030](https://github.com/vllm-project/vllm/pull/26030) |
| `tests/kernels/moe/test_gpt_oss_triton_kernels.py` | [#22421](https://github.com/vllm-project/vllm/pull/22421), [#29008](https://github.com/vllm-project/vllm/pull/29008), [#37683](https://github.com/vllm-project/vllm/pull/37683), [#39007](https://github.com/vllm-project/vllm/pull/39007) |
| `tests/models/quantization/test_gpt_oss.py` | [#29008](https://github.com/vllm-project/vllm/pull/29008), [#35806](https://github.com/vllm-project/vllm/pull/35806), [#35887](https://github.com/vllm-project/vllm/pull/35887), [#36174](https://github.com/vllm-project/vllm/pull/36174) |
| `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` | [#39007](https://github.com/vllm-project/vllm/pull/39007) |
| `vllm/model_executor/models/gpt_oss.py` | [#22327](https://github.com/vllm-project/vllm/pull/22327), [#22401](https://github.com/vllm-project/vllm/pull/22401), [#22508](https://github.com/vllm-project/vllm/pull/22508), [#22538](https://github.com/vllm-project/vllm/pull/22538), [#22678](https://github.com/vllm-project/vllm/pull/22678), [#22948](https://github.com/vllm-project/vllm/pull/22948), [#22951](https://github.com/vllm-project/vllm/pull/22951), [#23613](https://github.com/vllm-project/vllm/pull/23613), [#23680](https://github.com/vllm-project/vllm/pull/23680), [#23815](https://github.com/vllm-project/vllm/pull/23815), [#24032](https://github.com/vllm-project/vllm/pull/24032), [#25246](https://github.com/vllm-project/vllm/pull/25246), ... (25 total) |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-06 | [#22327](https://github.com/vllm-project/vllm/pull/22327) | merged | Add GPT-OSS model code and config [1/N] | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-07 | [#22401](https://github.com/vllm-project/vllm/pull/22401) | merged | [gpt-oss] fix model config with hf_config | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-08 | [#22421](https://github.com/vllm-project/vllm/pull/22421) | merged | [gpt-oss] triton kernel mxfp4 | `tests/kernels/moe/test_gpt_oss_triton_kernels.py` |
| 2025-08-10 | [#22508](https://github.com/vllm-project/vllm/pull/22508) | merged | [oss] Init gpt-oss bf16 support | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-13 | [#22678](https://github.com/vllm-project/vllm/pull/22678) | merged | Force TRTLLM attention for gpt-oss on SM100 | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-15 | [#22538](https://github.com/vllm-project/vllm/pull/22538) | merged | [Kernel] Add cuda kernel for gpt_oss activation | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-15 | [#22948](https://github.com/vllm-project/vllm/pull/22948) | merged | Revert "[Kernel] Add cuda kernel for gpt_oss activation" | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-17 | [#22951](https://github.com/vllm-project/vllm/pull/22951) | merged | [Kernel] Add cuda kernel for gpt_oss activation | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-27 | [#23613](https://github.com/vllm-project/vllm/pull/23613) | merged | [Bugfix][gpt-oss] passing the cache config in gpt-oss | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-28 | [#23680](https://github.com/vllm-project/vllm/pull/23680) | merged | [Model] Add PP support and VLM backbone compatability for GPT-OSS | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-28 | [#23815](https://github.com/vllm-project/vllm/pull/23815) | merged | [Model] [gpt-oss] fix gpt-oss pp support | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-28 | [#23819](https://github.com/vllm-project/vllm/pull/23819) | merged | [Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE | `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py` |
| 2025-09-17 | [#24920](https://github.com/vllm-project/vllm/pull/24920) | merged | [CI] GPT-OSS GPQA eval test for Blackwell | `tests/evals/gpt_oss/test_gpqa_correctness.py`, `tests/evals/gpt_oss/conftest.py`, `tests/evals/gpt_oss/__init__.py` |
| 2025-09-22 | [#25246](https://github.com/vllm-project/vllm/pull/25246) | merged | Enable Eagle3 speculative decoding for GPT-OSS model | `vllm/model_executor/models/gpt_oss.py` |
| 2025-10-01 | [#26030](https://github.com/vllm-project/vllm/pull/26030) | merged | [CI] Tweaks to GPT-OSS Eval (Blackwell) for stability | `tests/evals/gpt_oss/test_gpqa_correctness.py` |
| 2025-10-18 | [#25515](https://github.com/vllm-project/vllm/pull/25515) | merged | [GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot | `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`, `vllm/reasoning/gptoss_reasoning_parser.py` |
| 2025-10-21 | [#24032](https://github.com/vllm-project/vllm/pull/24032) | merged | [BugFix] GPT-OSS Attention DP + MoE TP weight loading issue | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-05 | [#27786](https://github.com/vllm-project/vllm/pull/27786) | merged | [XPU] Add gpt-oss model support for Intel GPU | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-11 | [#27334](https://github.com/vllm-project/vllm/pull/27334) | merged | [Quantization] fix attention quantization of gpt_oss model | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-12 | [#28536](https://github.com/vllm-project/vllm/pull/28536) | merged | [Bugfix] Fix gpt_oss packed_modules_mapping | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-16 | [#28715](https://github.com/vllm-project/vllm/pull/28715) | merged | Fixed gpt-oss _load_weights_other() parameter position bug | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-16 | [#28765](https://github.com/vllm-project/vllm/pull/28765) | merged | Fix gpt oss weight loading with EP + bf16 | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-20 | [#28244](https://github.com/vllm-project/vllm/pull/28244) | merged | Add truncate arg to yarn to match openai implementation of gpt-oss | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-28 | [#29506](https://github.com/vllm-project/vllm/pull/29506) | merged | Fix parameter order in GPT-OSS weight loading function for non-MXFP4 weights | `vllm/model_executor/models/gpt_oss.py` |
| 2026-01-28 | [#30976](https://github.com/vllm-project/vllm/pull/30976) | merged | Use aiter triton fused_add_rmsnorm_pad for gpt-oss | `vllm/model_executor/models/gpt_oss.py` |
| 2026-02-10 | [#29008](https://github.com/vllm-project/vllm/pull/29008) | merged | [ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations | `vllm/model_executor/models/gpt_oss.py`, `tests/models/quantization/test_gpt_oss.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py` |
| 2026-02-11 | [#34337](https://github.com/vllm-project/vllm/pull/34337) | merged | [GPT-OSS] Remove unnecessary contiguous | `vllm/model_executor/models/gpt_oss.py` |
| 2026-02-27 | [#35404](https://github.com/vllm-project/vllm/pull/35404) | merged | [Bugfix][Model] Fix gpt-oss batch invariance | `vllm/model_executor/models/gpt_oss.py` |
| 2026-03-02 | [#35658](https://github.com/vllm-project/vllm/pull/35658) | merged | [ROCm] add amd-quark package in requirements for rocm to use quantized models | `tests/quantization/test_quark.py`, `requirements/rocm.txt` |
| 2026-03-03 | [#35806](https://github.com/vllm-project/vllm/pull/35806) | merged | [ROCm][CI] Fix Assertion Logic For `test_gpt_oss` | `tests/models/quantization/test_gpt_oss.py` |
| 2026-03-03 | [#35887](https://github.com/vllm-project/vllm/pull/35887) | merged | [ROCm][CI] Fix TP size issue for `test_gpt_oss` | `tests/models/quantization/test_gpt_oss.py` |
| 2026-03-07 | [#36174](https://github.com/vllm-project/vllm/pull/36174) | merged | [ROCm][CI] Enable AITER for failing `test_gpt_oss` test case on MI355 | `tests/models/quantization/test_gpt_oss.py` |
| 2026-03-10 | [#36179](https://github.com/vllm-project/vllm/pull/36179) | merged | [ROCm][CI] Fix ROCm GPT-OSS Eval test group | `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml`, `tests/evals/gpt_oss/configs/models-gfx942.txt`, `tests/evals/gpt_oss/configs/models-gfx950.txt` |
| 2026-03-18 | [#30647](https://github.com/vllm-project/vllm/pull/30647) | merged | [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE | `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`, `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` |
| 2026-03-18 | [#37205](https://github.com/vllm-project/vllm/pull/37205) | merged | [Kernel] Add gpt-oss Router GEMM kernel | `vllm/model_executor/models/gpt_oss.py` |
| 2026-03-20 | [#37683](https://github.com/vllm-project/vllm/pull/37683) | merged | [Perf] Eliminate redundant SparseMatrix creation in gpt_oss_triton_kernels | `tests/kernels/moe/test_gpt_oss_triton_kernels.py` |
| 2026-04-02 | [#38778](https://github.com/vllm-project/vllm/pull/38778) | merged | Revert "[Kernel] Add gpt-oss Router GEMM kernel (#37205)" | `vllm/model_executor/models/gpt_oss.py` |
| 2026-04-02 | [#38292](https://github.com/vllm-project/vllm/pull/38292) | merged | [CI][ROCm] Add gpt-oss w4a8 in CI | `tests/evals/gpt_oss/configs/models-gfx950.txt` |
| 2026-04-14 | [#39007](https://github.com/vllm-project/vllm/pull/39007) | merged | [MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/ | `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py` |

## Per-PR Diff Audit Cards

### PR #22327 - Add GPT-OSS model code and config [1/N]

- Link: https://github.com/vllm-project/vllm/pull/22327
- Status/date: merged / 2025-08-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `de98252f497b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +503/-0, 530 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add GPT-OSS model code and config [1/N]"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: Add model code and config only. Need to add the MXFP4 MoE support for functionality..
- Key implementation: `vllm/model_executor/models/gpt_oss.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: OAIAttention, __init__, forward, MLPBlock, touching `OAIAttention, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: OAIAttention, __init__, forward, MLPBlock
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -0,0 +1,472 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+from typing import Optional
+import torch
+import torch.distributed as dist
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` added +472/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22401 - [gpt-oss] fix model config with hf_config

- Link: https://github.com/vllm-project/vllm/pull/22401
- Status/date: merged / 2025-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `5c7cc33f4daf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[gpt-oss] fix model config with hf_config"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +3/-3 (6 lines); hunks: -61,9 +61,9 @@ def __init__(; -154,7 +154,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +3/-3 (6 lines); hunks: -61,9 +61,9 @@ def __init__(; -154,7 +154,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -61,9 +61,9 @@ def __init__(
-                config.rope_ntk_beta,
+                config.rope_scaling["beta_fast"],
-                config.rope_ntk_alpha,
+                config.rope_scaling["beta_slow"],
@@ -154,7 +154,7 @@ def __init__(
-                                top_k=config.num_experts_per_token,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22421 - [gpt-oss] triton kernel mxfp4

- Link: https://github.com/vllm-project/vllm/pull/22421
- Status/date: merged / 2025-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; associated commits `e789cad6b8b5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +755/-9, 859 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[gpt-oss] triton kernel mxfp4"; model line: GPT-OSS; category: performance/backend optimization; main diff: `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; PR body summary: Need nightly torch and triton main to work. Don't merge. want for accuracy test.
- Key implementation: `tests/kernels/moe/test_gpt_oss_triton_kernels.py` added +375/-0 (375 lines); hunks: -0,0 +1,375; symbols: deshuffle, init_compute_data, ModelConfig, swiglu, touching `deshuffle, init_compute_data, ModelConfig`.
- Code diff details:
  - `tests/kernels/moe/test_gpt_oss_triton_kernels.py` added +375/-0 (375 lines); hunks: -0,0 +1,375; symbols: deshuffle, init_compute_data, ModelConfig, swiglu
- Key code excerpts:

```diff
diff -- tests/kernels/moe/test_gpt_oss_triton_kernels.py
@@ -0,0 +1,375 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from dataclasses import dataclass, fields
+import pytest
+import torch
+import torch.nn.functional as F
```

- Reviewed files:
  - tests: `tests/kernels/moe/test_gpt_oss_triton_kernels.py` added +375/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22508 - [oss] Init gpt-oss bf16 support

- Link: https://github.com/vllm-project/vllm/pull/22508
- Status/date: merged / 2025-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `0c5254b82acc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +342/-125, 726 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[oss] Init gpt-oss bf16 support"; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/model_executor/models/gpt_oss.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +149/-3 (152 lines); hunks: -160,7 +160,9 @@ def __init__(; -262,8 +264,8 @@ def compute_logits(self, hidden_states: torch.Tensor,; symbols: __init__, forward, compute_logits, load_weights, touching `__init__, forward, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +149/-3 (152 lines); hunks: -160,7 +160,9 @@ def __init__(; -262,8 +264,8 @@ def compute_logits(self, hidden_states: torch.Tensor,; symbols: __init__, forward, compute_logits, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -160,7 +160,9 @@ def __init__(
-                                apply_router_weight_on_input=False)
+                                apply_router_weight_on_input=False,
+                                has_bias=True,
+                                activation="swiglu_oai")
@@ -262,8 +264,8 @@ def compute_logits(self, hidden_states: torch.Tensor,
-    def load_weights(self, weights: Iterable[tuple[str,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +149/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22678 - Force TRTLLM attention for gpt-oss on SM100

- Link: https://github.com/vllm-project/vllm/pull/22678
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `c6b928798e96`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +20/-9, 96 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Force TRTLLM attention for gpt-oss on SM100"; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: If attention sinks are being used, then use_trtllm_attention should return True since we only support sinks with that backend. This PR also moves the float32 sinks cast that the....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-4 (5 lines); hunks: -8,7 +8,6; -70,11 +69,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-4 (5 lines); hunks: -8,7 +8,6; -70,11 +69,9 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -8,7 +8,6 @@
-from vllm import envs
@@ -70,11 +69,9 @@ def __init__(
-        attention_sink_dtype = (torch.float32 if envs.VLLM_USE_TRTLLM_ATTENTION
-                                else torch.bfloat16)
-                        dtype=attention_sink_dtype,
+                        dtype=torch.bfloat16,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`, `vllm/utils/flashinfer.py`, `vllm/v1/attention/backends/flashinfer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22538 - [Kernel] Add cuda kernel for gpt_oss activation

- Link: https://github.com/vllm-project/vllm/pull/22538
- Status/date: merged / 2025-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `81f4b9648117`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +150/-24, 290 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Add cuda kernel for gpt_oss activation"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -159,7 +159,7 @@ def __init__(
-                                activation="swiglu_oai")
+                                activation="swigluoai")
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_activation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22948 - Revert "[Kernel] Add cuda kernel for gpt_oss activation"

- Link: https://github.com/vllm-project/vllm/pull/22948
- Status/date: merged / 2025-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `f1f0d2fab8a1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +24/-150, 290 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[Kernel] Add cuda kernel for gpt_oss activation""; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: Reverts vllm-project/vllm#22538.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -159,7 +159,7 @@ def __init__(
-                                activation="swigluoai")
+                                activation="swiglu_oai")
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_activation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22951 - [Kernel] Add cuda kernel for gpt_oss activation

- Link: https://github.com/vllm-project/vllm/pull/22951
- Status/date: merged / 2025-08-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `4d4061b6e73d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +157/-42, 330 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Add cuda kernel for gpt_oss activation"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -159,7 +159,7 @@ def __init__(
-                                activation="swiglu_oai")
+                                activation="swigluoai")
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_activation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23613 - [Bugfix][gpt-oss] passing the cache config in gpt-oss

- Link: https://github.com/vllm-project/vllm/pull/23613
- Status/date: merged / 2025-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `fecbb7c78298`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-1, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][gpt-oss] passing the cache config in gpt-oss"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: Summary: ATT. Otherwise, kv cache fp8 is not supported in 20/120b model Differential Revision: D81004120.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +6/-1 (7 lines); hunks: -174,12 +174,15 @@ class TransformerBlock(torch.nn.Module):; -203,6 +206,7 @@ def __init__(; symbols: TransformerBlock, __init__, touching `TransformerBlock, __init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +6/-1 (7 lines); hunks: -174,12 +174,15 @@ class TransformerBlock(torch.nn.Module):; -203,6 +206,7 @@ def __init__(; symbols: TransformerBlock, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -174,12 +174,15 @@ class TransformerBlock(torch.nn.Module):
+        cache_config: CacheConfig,
-        self.attn = OAIAttention(config, prefix=f"{prefix}.attn")
+        self.attn = OAIAttention(config,
+                                 prefix=f"{prefix}.attn",
+                                 cache_config=cache_config)
@@ -203,6 +206,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23680 - [Model] Add PP support and VLM backbone compatability for GPT-OSS

- Link: https://github.com/vllm-project/vllm/pull/23680
- Status/date: merged / 2025-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `c5d004aaaf3b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +87/-34, 232 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add PP support and VLM backbone compatability for GPT-OSS"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: - OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview can't work out-of-box yet because of missing PP support and VLM backbone compatability for GPT-OSS - This PR fix them to allow `O....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +86/-33 (119 lines); hunks: -11,7 +11,8; -27,7 +28,10; symbols: __init__, forward, MLPBlock, touching `__init__, forward, MLPBlock`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +86/-33 (119 lines); hunks: -11,7 +11,8; -27,7 +28,10; symbols: __init__, forward, MLPBlock
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -11,7 +11,8 @@
-from vllm.distributed import (get_ep_group, get_tensor_model_parallel_rank,
+from vllm.distributed import (get_ep_group, get_pp_group,
+                              get_tensor_model_parallel_rank,
@@ -27,7 +28,10 @@
+from .interfaces import SupportsPP
+                    is_pp_missing_parameter,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +86/-33
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23815 - [Model] [gpt-oss] fix gpt-oss pp support

- Link: https://github.com/vllm-project/vllm/pull/23815
- Status/date: merged / 2025-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `bfab219648fd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-3, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] [gpt-oss] fix gpt-oss pp support"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: when _dummy_run in pp mode, it will trigger assertion fail.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +2/-3 (5 lines); hunks: -668,9 +668,8 @@ def forward(self,; symbols: forward, compute_logits, touching `forward, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +2/-3 (5 lines); hunks: -668,9 +668,8 @@ def forward(self,; symbols: forward, compute_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -668,9 +668,8 @@ def forward(self,
-        assert intermediate_tensors is None
-        assert inputs_embeds is None
-        return self.model(input_ids, positions)
+        return self.model(input_ids, positions, intermediate_tensors,
+                          inputs_embeds)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +2/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23819 - [Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE

- Link: https://github.com/vllm-project/vllm/pull/23819
- Status/date: merged / 2025-08-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +14/-15, 89 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py`; PR body summary: Changes: - Enable EP for GPT-OSS with FlashInfer trtllm-gen MoE - Fix an issue that VLLM_USE_FLASHINFER_MOE_FP4 is checked even when the quant dtype is not nvfp4. Run GPT-OSS-12....
- Key implementation: `vllm/model_executor/layers/fused_moe/config.py` modified +8/-7 (15 lines); hunks: -190,12 +190,6 @@ def use_deepep_ll_kernels(self):; -404,7 +398,14 @@ def use_deepep_ll_kernels(self):; symbols: use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, make, touching `use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, make`; `vllm/model_executor/layers/fused_moe/layer.py` modified +4/-4 (8 lines); hunks: -920,7 +920,7 @@ def __init__(; -974,7 +974,7 @@ def use_deepep_ll_kernels(self):; symbols: __init__, use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, update_expert_map, touching `__init__, use_deepep_ll_kernels, use_flashinfer_cutlass_kernels`; `vllm/model_executor/layers/quantization/mxfp4.py` modified +2/-4 (6 lines); hunks: -623,8 +623,6 @@ def apply(; -650,12 +648,12 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/config.py` modified +8/-7 (15 lines); hunks: -190,12 +190,6 @@ def use_deepep_ll_kernels(self):; -404,7 +398,14 @@ def use_deepep_ll_kernels(self):; symbols: use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, make
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +4/-4 (8 lines); hunks: -920,7 +920,7 @@ def __init__(; -974,7 +974,7 @@ def use_deepep_ll_kernels(self):; symbols: __init__, use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, update_expert_map
  - `vllm/model_executor/layers/quantization/mxfp4.py` modified +2/-4 (6 lines); hunks: -623,8 +623,6 @@ def apply(; -650,12 +648,12 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/config.py
@@ -190,12 +190,6 @@ def use_deepep_ll_kernels(self):
-    @property
-    def use_flashinfer_cutlass_kernels(self):
-        return (envs.VLLM_USE_FLASHINFER_MOE_FP4
-                and has_flashinfer_cutlass_fused_moe()
-                and envs.VLLM_FLASHINFER_MOE_BACKEND == "throughput")
@@ -404,7 +398,14 @@ def use_deepep_ll_kernels(self):
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -920,7 +920,7 @@ def __init__(
-                or self.moe_parallel_config.use_flashinfer_cutlass_kernels):
+                or self.moe_config.use_flashinfer_cutlass_kernels):
@@ -974,7 +974,7 @@ def use_deepep_ll_kernels(self):
-        return self.moe_parallel_config.use_flashinfer_cutlass_kernels
+        return self.moe_config.use_flashinfer_cutlass_kernels
@@ -1665,7 +1665,7 @@ def forward_impl(self, hidden_states: torch.Tensor,
diff -- vllm/model_executor/layers/quantization/mxfp4.py
@@ -623,8 +623,6 @@ def apply(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/config.py` modified +8/-7; `vllm/model_executor/layers/fused_moe/layer.py` modified +4/-4; `vllm/model_executor/layers/quantization/mxfp4.py` modified +2/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24920 - [CI] GPT-OSS GPQA eval test for Blackwell

- Link: https://github.com/vllm-project/vllm/pull/24920
- Status/date: merged / 2025-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gpt_oss/__init__.py`, `tests/evals/gpt_oss/conftest.py`, `tests/evals/gpt_oss/test_gpqa_correctness.py`; associated commits `493b10f8bf38`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +136/-0, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] GPT-OSS GPQA eval test for Blackwell"; model line: GPT-OSS; category: performance/backend optimization; main diff: `tests/evals/gpt_oss/test_gpqa_correctness.py`, `tests/evals/gpt_oss/conftest.py`, `tests/evals/gpt_oss/__init__.py`; PR body summary: Register a gpqa eval test in CI to test the Blackwell FlashInfer integration for `openai/gpt-oss-20b`.
- Key implementation: `tests/evals/gpt_oss/test_gpqa_correctness.py` added +102/-0 (102 lines); hunks: -0,0 +1,102; symbols: run_gpqa_eval, test_gpqa_correctness, touching `run_gpqa_eval, test_gpqa_correctness`; `tests/evals/gpt_oss/conftest.py` added +18/-0 (18 lines); hunks: -0,0 +1,18; symbols: pytest_addoption, touching `pytest_addoption`; `tests/evals/gpt_oss/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2.
- Code diff details:
  - `tests/evals/gpt_oss/test_gpqa_correctness.py` added +102/-0 (102 lines); hunks: -0,0 +1,102; symbols: run_gpqa_eval, test_gpqa_correctness
  - `tests/evals/gpt_oss/conftest.py` added +18/-0 (18 lines); hunks: -0,0 +1,18; symbols: pytest_addoption
  - `tests/evals/gpt_oss/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- Key code excerpts:

```diff
diff -- tests/evals/gpt_oss/test_gpqa_correctness.py
@@ -0,0 +1,102 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+GPQA evaluation using vLLM server and GPT-OSS evaluation package.
+Usage:
+pytest -s -v tests/evals/gpt_oss/test_gpqa_correctness.py \
diff -- tests/evals/gpt_oss/conftest.py
@@ -0,0 +1,18 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Pytest configuration for GPT-OSS evaluation tests.
+"""
+def pytest_addoption(parser):
diff -- tests/evals/gpt_oss/__init__.py
@@ -0,0 +1,2 @@
```

- Reviewed files:
  - tests: `tests/evals/gpt_oss/test_gpqa_correctness.py` added +102/-0; `tests/evals/gpt_oss/conftest.py` added +18/-0; `tests/evals/gpt_oss/__init__.py` added +2/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gpt_oss/__init__.py`, `tests/evals/gpt_oss/conftest.py`, `tests/evals/gpt_oss/test_gpqa_correctness.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25246 - Enable Eagle3 speculative decoding for GPT-OSS model

- Link: https://github.com/vllm-project/vllm/pull/25246
- Status/date: merged / 2025-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `21467f9a1c62`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +41/-12, 111 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable Eagle3 speculative decoding for GPT-OSS model"; model line: GPT-OSS; category: docs/tests/CI; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: This PR adds support for EAGLE3 speculative decoding for GPT-OSS model. Changes tested with a locally trained speculator model, and observed reasonable acceptance rates..
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +17/-2 (19 lines); hunks: -28,7 +28,7; -239,6 +239,7 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, _load_weights_mxfp4, touching `__init__, get_input_embeddings, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +17/-2 (19 lines); hunks: -28,7 +28,7; -239,6 +239,7 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, _load_weights_mxfp4
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -28,7 +28,7 @@
-from .interfaces import SupportsPP
+from .interfaces import SupportsEagle3, SupportsPP
@@ -239,6 +239,7 @@ def __init__(
+        self.aux_hidden_state_layers = tuple[int, ...]()
@@ -262,15 +263,22 @@ def forward(
+        aux_hidden_states = []
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +17/-2
- Risk and verification: Runtime changes concentrate in `vllm/config/speculative.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/v1/spec_decode/eagle.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26030 - [CI] Tweaks to GPT-OSS Eval (Blackwell) for stability

- Link: https://github.com/vllm-project/vllm/pull/26030
- Status/date: merged / 2025-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gpt_oss/test_gpqa_correctness.py`; associated commits `ee04c0cd04cf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +3/-4, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Tweaks to GPT-OSS Eval (Blackwell) for stability"; model line: GPT-OSS; category: docs/tests/CI; main diff: `tests/evals/gpt_oss/test_gpqa_correctness.py`; PR body summary: Using TP was slowing down inference greatly for gpt-oss-20b and reducing thread count helps stability.
- Key implementation: `tests/evals/gpt_oss/test_gpqa_correctness.py` modified +2/-3 (5 lines); hunks: -26,7 +26,8 @@ def run_gpqa_eval(model_name: str, base_url: str) -> float:; -72,8 +73,6 @@ def test_gpqa_correctness(request):; symbols: run_gpqa_eval, test_gpqa_correctness, touching `run_gpqa_eval, test_gpqa_correctness`.
- Code diff details:
  - `tests/evals/gpt_oss/test_gpqa_correctness.py` modified +2/-3 (5 lines); hunks: -26,7 +26,8 @@ def run_gpqa_eval(model_name: str, base_url: str) -> float:; -72,8 +73,6 @@ def test_gpqa_correctness(request):; symbols: run_gpqa_eval, test_gpqa_correctness
- Key code excerpts:

```diff
diff -- tests/evals/gpt_oss/test_gpqa_correctness.py
@@ -26,7 +26,8 @@ def run_gpqa_eval(model_name: str, base_url: str) -> float:
-        model_name, "--reasoning-effort", "low", "--base-url", base_url
+        model_name, "--reasoning-effort", "low", "--base-url", base_url,
+        "--n-threads", "200"
@@ -72,8 +73,6 @@ def test_gpqa_correctness(request):
-        "--max-model-len",
-        "32768",
```

- Reviewed files:
  - tests: `tests/evals/gpt_oss/test_gpqa_correctness.py` modified +2/-3
- Risk and verification: The diff ships test coverage in `tests/evals/gpt_oss/test_gpqa_correctness.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25515 - [GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot

- Link: https://github.com/vllm-project/vllm/pull/25515
- Status/date: merged / 2025-10-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +911/-32, 1107 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot"; model line: GPT-OSS; category: model support/runtime entry; main diff: `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`, `vllm/reasoning/gptoss_reasoning_parser.py`; PR body summary: The model sometimes generates function calls in CoT that have not been provided in tool_server. This PR uses structural tag to force model to 1. do not generate function calls w....
- Key implementation: `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py` added +280/-0 (280 lines); hunks: -0,0 +1,280; symbols: TestGptOssStructuralTagsIntegration, mock_tokenizer, gptoss_parser, tool_server_with_python, touching `TestGptOssStructuralTagsIntegration, mock_tokenizer, gptoss_parser`; `tests/v1/structured_output/test_reasoning_structured_output.py` added +207/-0 (207 lines); hunks: -0,0 +1,207; symbols: TestReasoningStructuredOutput, mock_model_config, mock_scheduler_config, mock_vllm_config, touching `TestReasoningStructuredOutput, mock_model_config, mock_scheduler_config`; `vllm/reasoning/gptoss_reasoning_parser.py` modified +75/-1 (76 lines); hunks: -1,17 +1,61; -81,3 +125,33 @@ def extract_reasoning_content(; symbols: from_builtin_tool_to_tag, tag_with_builtin_funcs, GptOssReasoningParser, extract_reasoning_content, touching `from_builtin_tool_to_tag, tag_with_builtin_funcs, GptOssReasoningParser`; `tests/v1/entrypoints/llm/test_struct_output_generate.py` modified +46/-0 (46 lines); hunks: -864,3 +864,49 @@ def test_structured_output_batched_with_non_structured_outp...; symbols: test_structured_output_batched_with_non_structured_outputs_requests, test_structured_output_with_structural_tag, touching `test_structured_output_batched_with_non_structured_outputs_requests, test_structured_output_with_structural_tag`.
- Code diff details:
  - `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py` added +280/-0 (280 lines); hunks: -0,0 +1,280; symbols: TestGptOssStructuralTagsIntegration, mock_tokenizer, gptoss_parser, tool_server_with_python
  - `tests/v1/structured_output/test_reasoning_structured_output.py` added +207/-0 (207 lines); hunks: -0,0 +1,207; symbols: TestReasoningStructuredOutput, mock_model_config, mock_scheduler_config, mock_vllm_config
  - `vllm/reasoning/gptoss_reasoning_parser.py` modified +75/-1 (76 lines); hunks: -1,17 +1,61; -81,3 +125,33 @@ def extract_reasoning_content(; symbols: from_builtin_tool_to_tag, tag_with_builtin_funcs, GptOssReasoningParser, extract_reasoning_content
  - `tests/v1/entrypoints/llm/test_struct_output_generate.py` modified +46/-0 (46 lines); hunks: -864,3 +864,49 @@ def test_structured_output_batched_with_non_structured_outp...; symbols: test_structured_output_batched_with_non_structured_outputs_requests, test_structured_output_with_structural_tag
  - `vllm/entrypoints/openai/protocol.py` modified +21/-5 (26 lines); hunks: -200,27 +200,39 @@ class JsonSchemaResponseFormat(OpenAIBaseModel):; -823,7 +835,11 @@ def to_sampling_params(; symbols: JsonSchemaResponseFormat, StructuralTag, LegacyStructuralTag, StructuralTagResponseFormat
- Key code excerpts:

```diff
diff -- tests/entrypoints/openai/test_gptoss_structural_tags_integration.py
@@ -0,0 +1,280 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Integration tests for GPT-OSS structural tags functionality (PR #25515)."""
+import json
+from unittest.mock import Mock
+import pytest
diff -- tests/v1/structured_output/test_reasoning_structured_output.py
@@ -0,0 +1,207 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""
+from unittest.mock import Mock
+import pytest
+from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
diff -- vllm/reasoning/gptoss_reasoning_parser.py
@@ -1,17 +1,61 @@
```

- Reviewed files:
  - tests: `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py` added +280/-0; `tests/v1/structured_output/test_reasoning_structured_output.py` added +207/-0; `tests/v1/entrypoints/llm/test_struct_output_generate.py` modified +46/-0; `tests/v1/structured_output/test_gptoss_structural_tags.py` added +172/-0
  - runtime: `vllm/reasoning/gptoss_reasoning_parser.py` modified +75/-1; `vllm/entrypoints/openai/protocol.py` modified +21/-5; `vllm/entrypoints/openai/serving_responses.py` modified +15/-1; `vllm/reasoning/abs_reasoning_parsers.py` modified +12/-0
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/entrypoints/llm/test_struct_output_generate.py`, `tests/v1/structured_output/test_gptoss_structural_tags.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24032 - [BugFix] GPT-OSS Attention DP + MoE TP weight loading issue

- Link: https://github.com/vllm-project/vllm/pull/24032
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `aef368aa0857`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +30/-13, 89 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] GPT-OSS Attention DP + MoE TP weight loading issue"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: Fix GPT-OSS weight loading issue when Attention DP + MoE TP is used. The GPT-OSS weight loading script does not consider the fact that tp needs to be flattened across dp in MoE,....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +16/-4 (20 lines); hunks: -11,13 +11,15; -305,8 +307,13 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4, _load_weights_other, touching `_load_weights_mxfp4, _load_weights_other`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +16/-4 (20 lines); hunks: -11,13 +11,15; -305,8 +307,13 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4, _load_weights_other
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -11,13 +11,15 @@
+    get_dp_group,
+from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
@@ -305,8 +307,13 @@ def _load_weights_mxfp4(
-        tp_rank = get_tensor_model_parallel_rank()
-        tp_size = get_tensor_model_parallel_world_size()
+        # In MoE, we need to flatten the tensor parallel size across the data
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +16/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27786 - [XPU] Add gpt-oss model support for Intel GPU

- Link: https://github.com/vllm-project/vllm/pull/27786
- Status/date: merged / 2025-11-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `18b39828d904`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +101/-6, 160 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU] Add gpt-oss model support for Intel GPU"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: this PR introduce a new `IpexFp4MoeMethod` for xpu, which support Wfp4A16 moe_gemm, implemented in ipex library. With that we can run openai/gpt-oss-20b, openai/gpt-oss-120b on....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +0/-3 (3 lines); hunks: -329,9 +329,6 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4, touching `_load_weights_mxfp4`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-3 (3 lines); hunks: -329,9 +329,6 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -329,9 +329,6 @@ def _load_weights_mxfp4(
-            # FIXME(woosuk): Remove this after testing.
-            weight = weight.cuda()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `vllm/attention/utils/fa_utils.py`, `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27334 - [Quantization] fix attention quantization of gpt_oss model

- Link: https://github.com/vllm-project/vllm/pull/27334
- Status/date: merged / 2025-11-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `5a1271d83a65`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +101/-4, 154 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] fix attention quantization of gpt_oss model"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: There are mainly two aiming purposes of this PR: 1. Enable attention quantization (FP8, MXFP4, UnquantizedLinearMethod, _etc_) by passing `quant_config` to `OAIAttention` and as....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +8/-2 (10 lines); hunks: -198,6 +198,7 @@ class TransformerBlock(torch.nn.Module):; -207,7 +208,10 @@ def __init__(; symbols: TransformerBlock, __init__, touching `TransformerBlock, __init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +8/-2 (10 lines); hunks: -198,6 +198,7 @@ class TransformerBlock(torch.nn.Module):; -207,7 +208,10 @@ def __init__(; symbols: TransformerBlock, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -198,6 +198,7 @@ class TransformerBlock(torch.nn.Module):
+        quant_config: QuantizationConfig,
@@ -207,7 +208,10 @@ def __init__(
-            config, prefix=f"{prefix}.attn", cache_config=cache_config
+            config,
+            prefix=f"{prefix}.attn",
+            quant_config=quant_config,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +8/-2
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_gpt_oss_attn_quantization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28536 - [Bugfix] Fix gpt_oss packed_modules_mapping

- Link: https://github.com/vllm-project/vllm/pull/28536
- Status/date: merged / 2025-11-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `a9d18b51078d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-5, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix gpt_oss packed_modules_mapping"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: https://github.com/vllm-project/vllm/pull/27334 modified the `packed_modules_mapping`, which resulted in test failures for gpt-oss LoRA see: https://buildkite.com/vllm/ci/builds....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +5/-5 (10 lines); hunks: -92,7 +92,7 @@ def __init__(; -129,7 +129,7 @@ def __init__(; symbols: __init__, forward, _load_weights_other, load_weights, touching `__init__, forward, _load_weights_other`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +5/-5 (10 lines); hunks: -92,7 +92,7 @@ def __init__(; -129,7 +129,7 @@ def __init__(; symbols: __init__, forward, _load_weights_other, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -92,7 +92,7 @@ def __init__(
-        self.qkv = QKVParallelLinear(
+        self.qkv_proj = QKVParallelLinear(
@@ -129,7 +129,7 @@ def __init__(
-        qkv, _ = self.qkv(hidden_states)
+        qkv, _ = self.qkv_proj(hidden_states)
@@ -606,9 +606,9 @@ def _load_weights_other(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +5/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28715 - Fixed gpt-oss _load_weights_other() parameter position bug

- Link: https://github.com/vllm-project/vllm/pull/28715
- Status/date: merged / 2025-11-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `af02c409702f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fixed gpt-oss _load_weights_other() parameter position bug"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: Summary: Signed-off-by: Dezhan Tu dezhantu@gmail.com For `_load_weights_other()`, `ep_rank_start` and `ep_rank_end` positions are wrongly placed, leading to the failure of loadi....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -641,8 +641,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -641,8 +641,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -641,8 +641,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-                ep_rank_end,
+                ep_rank_end,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28765 - Fix gpt oss weight loading with EP + bf16

- Link: https://github.com/vllm-project/vllm/pull/28765
- Status/date: merged / 2025-11-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `8d259fad6cd5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix gpt oss weight loading with EP + bf16"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: The signature for `_load_weights_other` is incorrect. The start and end indices are flipped, which casues an indexing issue when attempting to extract the weights on the current....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -494,8 +494,8 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4, _load_weights_other, touching `_load_weights_mxfp4, _load_weights_other`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -494,8 +494,8 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4, _load_weights_other
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -494,8 +494,8 @@ def _load_weights_mxfp4(
-        ep_rank_start: int,
+        ep_rank_start: int,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28244 - Add truncate arg to yarn to match openai implementation of gpt-oss

- Link: https://github.com/vllm-project/vllm/pull/28244
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `6eb745d9bdf5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +12/-7, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add truncate arg to yarn to match openai implementation of gpt-oss"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: Refer to the issue for context: https://github.com/vllm-project/vllm/issues/27722. VLLM's implementation of Yarn does not match OpenAI's for GPT-OSS. This PR provides a fix. I t....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-0 (1 lines); hunks: -78,6 +78,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-0 (1 lines); hunks: -78,6 +78,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -78,6 +78,7 @@ def __init__(
+                "truncate": config.rope_parameters.get("truncate", True),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/model_executor/layers/rotary_embedding/common.py`, `vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29506 - Fix parameter order in GPT-OSS weight loading function for non-MXFP4 weights

- Link: https://github.com/vllm-project/vllm/pull/29506
- Status/date: merged / 2025-11-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `5f5521bd5d7d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix parameter order in GPT-OSS weight loading function for non-MXFP4 weights"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: Current arguments order is wrong as _load_weights_other accepts ep_rank_end as the first argument and ep_rank_start as the second. This only affects non-MXFP4 GPTOSS variants li....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -647,8 +647,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -647,8 +647,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -647,8 +647,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-                ep_rank_start,
+                ep_rank_start,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30976 - Use aiter triton fused_add_rmsnorm_pad for gpt-oss

- Link: https://github.com/vllm-project/vllm/pull/30976
- Status/date: merged / 2026-01-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `59bcc5b6f2e6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +327/-11, 489 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use aiter triton fused_add_rmsnorm_pad for gpt-oss"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: Adds fused padding op before router GEMM on ROCm, eliminating this unfused pad after the GEMM before the fused_moe: https://github.com/ROCm/vllm/blob/main/vllm/model_executor/la....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -187,7 +187,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -187,7 +187,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -187,7 +187,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:
-        x = self.experts(hidden_states=x, router_logits=g)
+        x = self.experts(hidden_states=x, router_logits=g)[:, : self.hidden_size]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/compile/test_fuse_act_padding.py`, `tests/compile/test_fusion.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29008 - [ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations

- Link: https://github.com/vllm-project/vllm/pull/29008
- Status/date: merged / 2026-02-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/models/quantization/test_gpt_oss.py`, `vllm/model_executor/models/gpt_oss.py`; associated commits `b129136c7a73`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1094/-213, 1860 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations"; model line: GPT-OSS; category: model implementation change; main diff: `vllm/model_executor/models/gpt_oss.py`, `tests/models/quantization/test_gpt_oss.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; PR body summary: This PR aims for: - Models: - Quantization schemes: - TP: See results below. (Sub)-tasks.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +491/-18 (509 lines); hunks: -1,6 +1,7; -25,13 +26,17; symbols: __init__, forward, get_expert_mapping, _load_weights_mxfp4, touching `__init__, forward, get_expert_mapping`; `tests/models/quantization/test_gpt_oss.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: has_huggingface_access, ModelCase, EvaluationConfig, get_model_args, touching `has_huggingface_access, ModelCase, EvaluationConfig`; `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +13/-7 (20 lines); hunks: -22,7 +22,7; -298,12 +298,18 @@ def test_equiv(num_token, a_dtype, w_dtype, tp, workspace_...; symbols: test_equiv, touching `test_equiv`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +491/-18 (509 lines); hunks: -1,6 +1,7; -25,13 +26,17; symbols: __init__, forward, get_expert_mapping, _load_weights_mxfp4
  - `tests/models/quantization/test_gpt_oss.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: has_huggingface_access, ModelCase, EvaluationConfig, get_model_args
  - `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +13/-7 (20 lines); hunks: -22,7 +22,7; -298,12 +298,18 @@ def test_equiv(num_token, a_dtype, w_dtype, tp, workspace_...; symbols: test_equiv
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -1,6 +1,7 @@
-from collections.abc import Iterable
+import typing
+from collections.abc import Callable, Iterable
@@ -25,13 +26,17 @@
+from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_BLOCK_SIZE
-from vllm.model_executor.model_loader.weight_utils import default_weight_loader
diff -- tests/models/quantization/test_gpt_oss.py
@@ -0,0 +1,110 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+End-to-end accuracy test for GPT-OSS model quantization.
+Config:
+    Task:   gsm8k_platinum
diff -- tests/kernels/moe/test_gpt_oss_triton_kernels.py
@@ -22,7 +22,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +491/-18
  - tests: `tests/models/quantization/test_gpt_oss.py` added +110/-0; `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +13/-7
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/models/quantization/test_gpt_oss.py`, `tests/models/quantization/test_gpt_oss_attn_quantization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34337 - [GPT-OSS] Remove unnecessary contiguous

- Link: https://github.com/vllm-project/vllm/pull/34337
- Status/date: merged / 2026-02-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `83e26c834ef1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-1, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GPT-OSS] Remove unnecessary contiguous"; model line: GPT-OSS; category: docs/tests/CI; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: Remove unnecessary contiguous for GPT-OSS. Test Result PR: main:.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +0/-1 (1 lines); hunks: -140,7 +140,6 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-1 (1 lines); hunks: -140,7 +140,6 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -140,7 +140,6 @@ def forward(
-        v = v.contiguous()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35404 - [Bugfix][Model] Fix gpt-oss batch invariance

- Link: https://github.com/vllm-project/vllm/pull/35404
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `1f3dbd95fd13`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-8, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Model] Fix gpt-oss batch invariance"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: GPT-OSS is listed as verified in the batch invariance doc, but rerunning the provided tests on an H100 suggests it does not in fact work in all the claimed supported configurati....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +13/-2 (15 lines); hunks: -23,7 +23,11; -165,7 +169,14 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +13/-2 (15 lines); hunks: -23,7 +23,11; -165,7 +169,14 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -23,7 +23,11 @@
-from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
+from vllm.model_executor.layers.linear import (
+    QKVParallelLinear,
+    ReplicatedLinear,
+    RowParallelLinear,
+)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +13/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35658 - [ROCm] add amd-quark package in requirements for rocm to use quantized models

- Link: https://github.com/vllm-project/vllm/pull/35658
- Status/date: merged / 2026-03-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +24/-6, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] add amd-quark package in requirements for rocm to use quantized models"; model line: GPT-OSS; category: bug fix; main diff: `tests/quantization/test_quark.py`, `requirements/rocm.txt`; PR body summary: Fix https://github.com/vllm-project/vllm/issues/35633 - Added amd-quark to requirements/rocm.txt. This way, it can be picked up for building docker, wheel or building from sourc....
- Key implementation: `tests/quantization/test_quark.py` modified +20/-5 (25 lines); hunks: -26,9 +26,12; -200,7 +203,10 @@ def get_model_args(; symbols: get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness, test_mxfp4_fused_qdq_match_quark, touching `get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness`; `requirements/rocm.txt` modified +4/-1 (5 lines); hunks: -19,4 +19,7 @@ setuptools>=77.0.3,<80.0.0.
- Code diff details:
  - `tests/quantization/test_quark.py` modified +20/-5 (25 lines); hunks: -26,9 +26,12; -200,7 +203,10 @@ def get_model_args(; symbols: get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness, test_mxfp4_fused_qdq_match_quark
  - `requirements/rocm.txt` modified +4/-1 (5 lines); hunks: -19,4 +19,7 @@ setuptools>=77.0.3,<80.0.0
- Key code excerpts:

```diff
diff -- tests/quantization/test_quark.py
@@ -26,9 +26,12 @@
+# Minimum amd-quark version for MXFP4/OCP_MX tests (single source of truth).
+QUARK_MXFP4_MIN_VERSION = "0.8.99"
-) >= version.parse("0.8.99")
+) >= version.parse(QUARK_MXFP4_MIN_VERSION)
@@ -200,7 +203,10 @@ def get_model_args(
-@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
diff -- requirements/rocm.txt
@@ -19,4 +19,7 @@ setuptools>=77.0.3,<80.0.0
-timm>=1.0.17
+timm>=1.0.17
+# amd-quark: required for Quark quantization on ROCm
+# To be consistent with test_quark.py
+amd-quark>=0.8.99
```

- Reviewed files:
  - tests: `tests/quantization/test_quark.py` modified +20/-5
  - other: `requirements/rocm.txt` modified +4/-1
- Risk and verification: The diff ships test coverage in `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35806 - [ROCm][CI] Fix Assertion Logic For `test_gpt_oss`

- Link: https://github.com/vllm-project/vllm/pull/35806
- Status/date: merged / 2026-03-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/quantization/test_gpt_oss.py`; associated commits `8b9e8b74541e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-5, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][CI] Fix Assertion Logic For `test_gpt_oss`"; model line: GPT-OSS; category: bug fix; main diff: `tests/models/quantization/test_gpt_oss.py`; PR body summary: After https://github.com/vllm-project/vllm/pull/35658 was merged, we saw `Quantized Models Test` started failing in AMD CI: https://buildkite.com/vllm/amd-ci/builds/5661/steps/c....
- Key implementation: `tests/models/quantization/test_gpt_oss.py` modified +5/-5 (10 lines); hunks: -12,8 +12,8; -104,7 +104,7 @@ def test_gpt_oss_attention_quantization(; symbols: test_gpt_oss_attention_quantization, touching `test_gpt_oss_attention_quantization`.
- Code diff details:
  - `tests/models/quantization/test_gpt_oss.py` modified +5/-5 (10 lines); hunks: -12,8 +12,8; -104,7 +104,7 @@ def test_gpt_oss_attention_quantization(; symbols: test_gpt_oss_attention_quantization
- Key code excerpts:

```diff
diff -- tests/models/quantization/test_gpt_oss.py
@@ -12,8 +12,8 @@
-import importlib
+import importlib.util
@@ -104,7 +104,7 @@ def test_gpt_oss_attention_quantization(
-    assert (
-        measured_accuracy - rtol < expected_accuracy
-        and measured_accuracy + rtol > expected_accuracy
```

- Reviewed files:
  - tests: `tests/models/quantization/test_gpt_oss.py` modified +5/-5
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_gpt_oss.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35887 - [ROCm][CI] Fix TP size issue for `test_gpt_oss`

- Link: https://github.com/vllm-project/vllm/pull/35887
- Status/date: merged / 2026-03-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/quantization/test_gpt_oss.py`; associated commits `e7213003cbf6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-0, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][CI] Fix TP size issue for `test_gpt_oss`"; model line: GPT-OSS; category: bug fix; main diff: `tests/models/quantization/test_gpt_oss.py`; PR body summary: `Quantized Models Test` is allocated to a 1 GPU agent pool in CI, but tries to run multi-GPU tests (example: https://buildkite.com/vllm/amd-ci/builds/5699/steps/canvas?sid=019cb....
- Key implementation: `tests/models/quantization/test_gpt_oss.py` modified +5/-0 (5 lines); hunks: -21,6 +21,8; -83,6 +85,9 @@ def get_model_args(self, tp_size: int):; symbols: get_model_args, test_gpt_oss_attention_quantization, touching `get_model_args, test_gpt_oss_attention_quantization`.
- Code diff details:
  - `tests/models/quantization/test_gpt_oss.py` modified +5/-0 (5 lines); hunks: -21,6 +21,8; -83,6 +85,9 @@ def get_model_args(self, tp_size: int):; symbols: get_model_args, test_gpt_oss_attention_quantization
- Key code excerpts:

```diff
diff -- tests/models/quantization/test_gpt_oss.py
@@ -21,6 +21,8 @@
+from vllm.utils.torch_utils import cuda_device_count_stateless
@@ -83,6 +85,9 @@ def get_model_args(self, tp_size: int):
+    if tp_size > cuda_device_count_stateless():
+        pytest.skip("Not enough GPUs to run this test case")
```

- Reviewed files:
  - tests: `tests/models/quantization/test_gpt_oss.py` modified +5/-0
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_gpt_oss.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36174 - [ROCm][CI] Enable AITER for failing `test_gpt_oss` test case on MI355

- Link: https://github.com/vllm-project/vllm/pull/36174
- Status/date: merged / 2026-03-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/quantization/test_gpt_oss.py`; associated commits `fc4657756ff0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-1, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][CI] Enable AITER for failing `test_gpt_oss` test case on MI355"; model line: GPT-OSS; category: performance/backend optimization; main diff: `tests/models/quantization/test_gpt_oss.py`; PR body summary: This test case is passing on MI325 but failing on MI350: `pytest -v -s tests/models/quantization/test_gpt_oss.py::test_gpt_oss_attention_quantization[amd/gpt-oss-20b-MoE-Quant-W....
- Key implementation: `tests/models/quantization/test_gpt_oss.py` modified +8/-1 (9 lines); hunks: -21,6 +21,7; -83,11 +84,17 @@ def get_model_args(self, tp_size: int):; symbols: get_model_args, test_gpt_oss_attention_quantization, touching `get_model_args, test_gpt_oss_attention_quantization`.
- Code diff details:
  - `tests/models/quantization/test_gpt_oss.py` modified +8/-1 (9 lines); hunks: -21,6 +21,7; -83,11 +84,17 @@ def get_model_args(self, tp_size: int):; symbols: get_model_args, test_gpt_oss_attention_quantization
- Key code excerpts:

```diff
diff -- tests/models/quantization/test_gpt_oss.py
@@ -21,6 +21,7 @@
+from vllm.platforms.rocm import on_gfx950
@@ -83,11 +84,17 @@ def get_model_args(self, tp_size: int):
-    model_name: str, tp_size: int, expected_accuracy: float
+    model_name: str,
+    tp_size: int,
+    expected_accuracy: float,
```

- Reviewed files:
  - tests: `tests/models/quantization/test_gpt_oss.py` modified +8/-1
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_gpt_oss.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36179 - [ROCm][CI] Fix ROCm GPT-OSS Eval test group

- Link: https://github.com/vllm-project/vllm/pull/36179
- Status/date: merged / 2026-03-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml`, `tests/evals/gpt_oss/configs/models-gfx942.txt`, `tests/evals/gpt_oss/configs/models-gfx950.txt`; associated commits `179547d62c73`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +16/-4, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][CI] Fix ROCm GPT-OSS Eval test group"; model line: GPT-OSS; category: bug fix; main diff: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml`, `tests/evals/gpt_oss/configs/models-gfx942.txt`, `tests/evals/gpt_oss/configs/models-gfx950.txt`; PR body summary: Fixes optional test `ROCm GPT-OSS Eval` in AMD-CI external evaluation signal. cc @kenroche.
- Key implementation: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml` added +6/-0 (6 lines); hunks: -0,0 +1,6; `tests/evals/gpt_oss/configs/models-gfx942.txt` added +3/-0 (3 lines); hunks: -0,0 +1,3; `tests/evals/gpt_oss/configs/models-gfx950.txt` added +3/-0 (3 lines); hunks: -0,0 +1,3.
- Code diff details:
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml` added +6/-0 (6 lines); hunks: -0,0 +1,6
  - `tests/evals/gpt_oss/configs/models-gfx942.txt` added +3/-0 (3 lines); hunks: -0,0 +1,3
  - `tests/evals/gpt_oss/configs/models-gfx950.txt` added +3/-0 (3 lines); hunks: -0,0 +1,3
- Key code excerpts:

```diff
diff -- tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml
@@ -0,0 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+model_name: openai/gpt-oss-20b
+metric_threshold: 0.568
+reasoning_effort: low
+server_args: "--attention-backend ROCM_AITER_UNIFIED_ATTN"
diff -- tests/evals/gpt_oss/configs/models-gfx942.txt
@@ -0,0 +1,3 @@
+# GFX942 model configurations for GPQA evaluation
+# Tests different environment variable combinations
+gpt-oss-20b-rocm-baseline.yaml
diff -- tests/evals/gpt_oss/configs/models-gfx950.txt
@@ -0,0 +1,3 @@
+# GFX950 model configurations for GPQA evaluation
+# Tests different environment variable combinations
+gpt-oss-20b-rocm-baseline.yaml
```

- Reviewed files:
  - tests: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml` added +6/-0; `tests/evals/gpt_oss/configs/models-gfx942.txt` added +3/-0; `tests/evals/gpt_oss/configs/models-gfx950.txt` added +3/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml`, `tests/evals/gpt_oss/configs/models-gfx942.txt`, `tests/evals/gpt_oss/configs/models-gfx950.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30647 - [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE

- Link: https://github.com/vllm-project/vllm/pull/30647
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +40/-3, 105 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`, `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py`; PR body summary: - Depends on Flashinfer update #30993 - Eliminated padding op before the MoE: by setting the alignment in flashinfer mxfp8 quant, the output quantized tensor will be padded. - E....
- Key implementation: `vllm/model_executor/layers/quantization/mxfp4.py` modified +16/-1 (17 lines); hunks: -294,6 +294,12 @@ def __init__(self, moe: FusedMoEConfig):; -1130,9 +1136,17 @@ def apply_monolithic(; symbols: __init__, skip_forward_padding, create_weights, apply_monolithic, touching `__init__, skip_forward_padding, create_weights`; `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` modified +5/-0 (5 lines); hunks: -101,6 +101,11 @@ def topk_indices_dtype(self) -> torch.dtype | None:; symbols: topk_indices_dtype, skip_forward_padding, supports_eplb, touching `topk_indices_dtype, skip_forward_padding, supports_eplb`; `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` modified +4/-1 (5 lines); hunks: -415,7 +415,10 @@ def forward(; symbols: forward, touching `forward`; `tests/compile/fusions_e2e/models.py` modified +9/-0 (9 lines); hunks: -162,3 +162,12.
- Code diff details:
  - `vllm/model_executor/layers/quantization/mxfp4.py` modified +16/-1 (17 lines); hunks: -294,6 +294,12 @@ def __init__(self, moe: FusedMoEConfig):; -1130,9 +1136,17 @@ def apply_monolithic(; symbols: __init__, skip_forward_padding, create_weights, apply_monolithic
  - `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` modified +5/-0 (5 lines); hunks: -101,6 +101,11 @@ def topk_indices_dtype(self) -> torch.dtype | None:; symbols: topk_indices_dtype, skip_forward_padding, supports_eplb
  - `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` modified +4/-1 (5 lines); hunks: -415,7 +415,10 @@ def forward(; symbols: forward
  - `tests/compile/fusions_e2e/models.py` modified +9/-0 (9 lines); hunks: -162,3 +162,12
  - `tests/compile/fusions_e2e/conftest.py` modified +4/-0 (4 lines); hunks: -82,6 +82,10 @@ def run(; symbols: run
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/mxfp4.py
@@ -294,6 +294,12 @@ def __init__(self, moe: FusedMoEConfig):
+    @property
+    def skip_forward_padding(self) -> bool:
+        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
+        # so can skip the padding in the forward before applying the moe method
+        return self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
@@ -1130,9 +1136,17 @@ def apply_monolithic(
diff -- vllm/model_executor/layers/fused_moe/fused_moe_method_base.py
@@ -101,6 +101,11 @@ def topk_indices_dtype(self) -> torch.dtype | None:
+    @property
+    def skip_forward_padding(self) -> bool:
+        """Whether to skip the padding in the forward before applying the moe method."""
+        return False
diff -- vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py
@@ -415,7 +415,10 @@ def forward(
-        if self.moe_config.hidden_dim != transformed_hidden_dim:
+        if (
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/mxfp4.py` modified +16/-1; `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` modified +5/-0; `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` modified +4/-1
  - tests: `tests/compile/fusions_e2e/models.py` modified +9/-0; `tests/compile/fusions_e2e/conftest.py` modified +4/-0; `tests/compile/fusions_e2e/test_tp2_ar_rms.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/compile/fusions_e2e/conftest.py`, `tests/compile/fusions_e2e/models.py`, `tests/compile/fusions_e2e/test_tp2_ar_rms.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37205 - [Kernel] Add gpt-oss Router GEMM kernel

- Link: https://github.com/vllm-project/vllm/pull/37205
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `9bd723110689`, `b1169d7be8ad`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +875/-13, 1035 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Add gpt-oss Router GEMM kernel"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: This PR add gpt-oss optimized Router GEMM kernel. 1% - 2% output token throughput improvement at batch size 1. Added unit test. Unit test passed. Micro bench `gpt_oss_router_gem....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +3/-7 (10 lines); hunks: -20,12 +20,11; -175,13 +174,11 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +3/-7 (10 lines); hunks: -20,12 +20,11; -175,13 +174,11 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -20,12 +20,11 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
-    ReplicatedLinear,
@@ -175,13 +174,11 @@ def __init__(
-        self.router = ReplicatedLinear(
+        self.router = GateLinear(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +3/-7
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_router_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37683 - [Perf] Eliminate redundant SparseMatrix creation in gpt_oss_triton_kernels

- Link: https://github.com/vllm-project/vllm/pull/37683
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; associated commits `d0532bf38da5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +73/-4, 108 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Eliminate redundant SparseMatrix creation in gpt_oss_triton_kernels"; model line: GPT-OSS; category: performance/backend optimization; main diff: `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; PR body summary: During profiling I noticed `_sum_bitmatrix_rows` + `_bitmatrix_metadata_compute_stage1` +`_bitmatrix_metadata_compute_stage2` kernels were launched twice. * Creating `SparseMatr....
- Key implementation: `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +44/-0 (44 lines); hunks: -21,12 +21,16; -355,3 +359,43 @@ def test_unit_shuffle():; symbols: test_unit_shuffle, test_legacy_routing, touching `test_unit_shuffle, test_legacy_routing`.
- Code diff details:
  - `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +44/-0 (44 lines); hunks: -21,12 +21,16; -355,3 +359,43 @@ def test_unit_shuffle():; symbols: test_unit_shuffle, test_legacy_routing
- Key code excerpts:

```diff
diff -- tests/kernels/moe/test_gpt_oss_triton_kernels.py
@@ -21,12 +21,16 @@
+from triton_kernels.topk import topk as topk_fn
+    legacy_routing,
+    make_routing_data,
+from vllm.utils.torch_utils import set_random_seed
@@ -355,3 +359,43 @@ def test_unit_shuffle():
+@pytest.mark.parametrize("num_tokens", [2, 8, 64])
```

- Reviewed files:
  - tests: `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +44/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38778 - Revert "[Kernel] Add gpt-oss Router GEMM kernel (#37205)"

- Link: https://github.com/vllm-project/vllm/pull/38778
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `9bd723110689`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +12/-875, 1027 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[Kernel] Add gpt-oss Router GEMM kernel (#37205)""; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/model_executor/models/gpt_oss.py`; PR body summary: PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED. This PR commit b1169d7be8add20ab1db4bc93c2b5c6336ef9754, which is repor....
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -20,11 +20,12; -174,11 +175,13 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -20,11 +20,12; -174,11 +175,13 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -20,11 +20,12 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
+from vllm.model_executor.layers.fused_moe import FusedMoE
+    ReplicatedLinear,
@@ -174,11 +175,13 @@ def __init__(
-        self.router = GateLinear(
+        self.router = ReplicatedLinear(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +6/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_router_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38292 - [CI][ROCm] Add gpt-oss w4a8 in CI

- Link: https://github.com/vllm-project/vllm/pull/38292
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gpt_oss/configs/models-gfx950.txt`; associated commits `82a006beebf0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-1, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI][ROCm] Add gpt-oss w4a8 in CI"; model line: GPT-OSS; category: docs/tests/CI; main diff: `tests/evals/gpt_oss/configs/models-gfx950.txt`; PR body summary: Enable coverage on https://github.com/vllm-project/vllm/blob/f73bcb1c51cfc764f534fcd109f8437e196be2ec/vllm/model_executor/layers/quantization/quark/quark_moe.py#L1095 Next steps....
- Key implementation: `tests/evals/gpt_oss/configs/models-gfx950.txt` modified +2/-1 (3 lines); hunks: -1,3 +1,4.
- Code diff details:
  - `tests/evals/gpt_oss/configs/models-gfx950.txt` modified +2/-1 (3 lines); hunks: -1,3 +1,4
- Key code excerpts:

```diff
diff -- tests/evals/gpt_oss/configs/models-gfx950.txt
@@ -1,3 +1,4 @@
-gpt-oss-20b-rocm-baseline.yaml
+gpt-oss-20b-rocm-baseline.yaml
+gpt-oss-20b-rocm-mxfp4-fp8.yaml
```

- Reviewed files:
  - tests: `tests/evals/gpt_oss/configs/models-gfx950.txt` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-mxfp4-fp8.yaml`, `tests/evals/gpt_oss/configs/models-gfx950.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39007 - [MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/

- Link: https://github.com/vllm-project/vllm/pull/39007
- Status/date: merged / 2026-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`; associated commits `1a9353bb02e6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +16/-12, 100 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; PR body summary: - Moves `gpt_oss_triton_kernels_moe.py` from `fused_moe/` root into `fused_moe/experts/`, consistent with the ongoing migration of expert kernel files (e.g. `trtllm_nvfp4_moe.py....
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` renamed +0/-0 (0 lines); `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +1/-1 (2 lines); hunks: -25,7 +25,7.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` renamed +0/-0 (0 lines)
  - `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +1/-1 (2 lines); hunks: -25,7 +25,7
- Key code excerpts:

```diff
diff -- tests/kernels/moe/test_gpt_oss_triton_kernels.py
@@ -25,7 +25,7 @@
-from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
+from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` renamed +0/-0
  - tests: `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/kernels/moe/test_modular_oai_triton_moe.py`, `tests/kernels/quantization/test_mxfp4_triton_ep.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.
