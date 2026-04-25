# vllm Qwen3 Next Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `vllm/model_executor/models/qwen3_next.py` | [#24526](https://github.com/vllm-project/vllm/pull/24526), [#24709](https://github.com/vllm-project/vllm/pull/24709), [#24957](https://github.com/vllm-project/vllm/pull/24957), [#24960](https://github.com/vllm-project/vllm/pull/24960), [#25079](https://github.com/vllm-project/vllm/pull/25079), [#25243](https://github.com/vllm-project/vllm/pull/25243), [#25268](https://github.com/vllm-project/vllm/pull/25268), [#26437](https://github.com/vllm-project/vllm/pull/26437), [#27030](https://github.com/vllm-project/vllm/pull/27030), [#27578](https://github.com/vllm-project/vllm/pull/27578), [#28202](https://github.com/vllm-project/vllm/pull/28202), [#28267](https://github.com/vllm-project/vllm/pull/28267), ... (22 total) |
| `vllm/model_executor/models/qwen3_next_mtp.py` | [#24526](https://github.com/vllm-project/vllm/pull/24526), [#25079](https://github.com/vllm-project/vllm/pull/25079) |
| `vllm/transformers_utils/configs/qwen3_next.py` | [#24526](https://github.com/vllm-project/vllm/pull/24526) |

## PR Coverage Summary

- Git-traced PRs: 22
- Extra PRs preserved from existing docs: 7
- Total PRs in this document: 29
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-09-01 | [#23994](https://github.com/vllm-project/vllm/pull/23994) | merged | [BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ) | `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/layers/quantization/gptq.py`, `vllm/model_executor/layers/quantization/gptq_marlin.py` |
| 2025-09-11 | [#24526](https://github.com/vllm-project/vllm/pull/24526) | merged | Add the support for the qwen3 next model (a hybrid attention model). | `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_next_mtp.py`, `vllm/transformers_utils/configs/qwen3_next.py` |
| 2025-09-12 | [#24709](https://github.com/vllm-project/vllm/pull/24709) | merged | [BugFix] Fix Qwen3-Next PP | `vllm/model_executor/models/qwen3_next.py` |
| 2025-09-17 | [#24957](https://github.com/vllm-project/vllm/pull/24957) | merged | [Bugfix][Qwen3-Next] fixes the varlen issue in qwen3-next's MTP implementation. | `vllm/model_executor/models/qwen3_next.py` |
| 2025-09-18 | [#24960](https://github.com/vllm-project/vllm/pull/24960) | merged | [Bugfix][Qwen3-Next] add prefixes to shared_expert in qwen3-next and mlp in qwen2moe to successfully load ignored params in quantized models | `vllm/model_executor/models/qwen3_next.py` |
| 2025-09-18 | [#25079](https://github.com/vllm-project/vllm/pull/25079) | merged | [Qwen] Add fp8 checkpoint support for qwen3-next. | `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_next_mtp.py` |
| 2025-09-19 | [#25243](https://github.com/vllm-project/vllm/pull/25243) | merged | [Qwen] Remove cuda hard-code in qwen3 next | `vllm/model_executor/models/qwen3_next.py` |
| 2025-09-20 | [#25268](https://github.com/vllm-project/vllm/pull/25268) | merged | [BUGFIX] GPTQ quantization compatibility for Qwen3 Next MOE models (AutoGPTQ and AutoRound-GPTQ) | `vllm/model_executor/models/qwen3_next.py` |
| 2025-09-26 | [#25743](https://github.com/vllm-project/vllm/pull/25743) | merged | [Qwen3-Next][GDN] fixes cuda graph capturing bug in GDN metadata and a stride bug in causal_conv_1d. | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`, `vllm/v1/attention/backends/gdn_attn.py`, `vllm/v1/worker/gpu_model_runner.py` |
| 2025-10-16 | [#26437](https://github.com/vllm-project/vllm/pull/26437) | merged | [PERF] Qwen3-next MTP speedup (change bool mask indexing to index_select / index_copy to reduce d2h) | `vllm/model_executor/models/qwen3_next.py` |
| 2025-10-17 | [#27030](https://github.com/vllm-project/vllm/pull/27030) | merged | [Bugfix][Qwen] fixes the weights dtype in qwen3_next: it is actually a bfloat16 | `vllm/model_executor/models/qwen3_next.py` |
| 2025-10-21 | [#26440](https://github.com/vllm-project/vllm/pull/26440) | merged | [Performance] Dual stream execution of "shared_experts" and "selected_experts" inside FusedMoE | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` |
| 2025-10-29 | [#27578](https://github.com/vllm-project/vllm/pull/27578) | merged | [perf] Enable concurrent execution of "shared_experts" and "selected_experts" in qwen3-next | `vllm/model_executor/models/qwen3_next.py` |
| 2025-11-06 | [#28182](https://github.com/vllm-project/vllm/pull/28182) | open | [Qwen3-Next][Perf][Attention] Replace torch.zeros with torch.empty to reduce overhead | `vllm/model_executor/models/qwen3_next.py` |
| 2025-11-09 | [#28267](https://github.com/vllm-project/vllm/pull/28267) | merged | [Misc] Add some comments in qwen3-next | `vllm/model_executor/models/qwen3_next.py` |
| 2025-11-11 | [#28202](https://github.com/vllm-project/vllm/pull/28202) | merged | [Bugfix] fix qwen3-next crash | `vllm/model_executor/models/qwen3_next.py` |
| 2025-11-19 | [#28960](https://github.com/vllm-project/vllm/pull/28960) | merged | [Bugfix] Fix typo in Qwen3 Next model executor | `vllm/model_executor/models/qwen3_next.py` |
| 2025-12-13 | [#30433](https://github.com/vllm-project/vllm/pull/30433) | merged | [Bugfix] Qwen3-next with --hf-overrides \{\"num_hidden_layers\":8\} | `vllm/model_executor/models/qwen3_next.py` |
| 2026-01-06 | [#31722](https://github.com/vllm-project/vllm/pull/31722) | merged | [PERF] Speed-up of GDN attention decode part (Qwen3-Next) | `vllm/model_executor/layers/fla/ops/fused_recurrent.py` |
| 2026-01-08 | [#31719](https://github.com/vllm-project/vllm/pull/31719) | merged | [Misc] Support qwen3-next lora | `vllm/model_executor/models/qwen3_next.py` |
| 2026-02-13 | [#34489](https://github.com/vllm-project/vllm/pull/34489) | merged | [Bugfix] Fix mamba state dtype setting for Qwen3-Next and Qwen3.5 | `vllm/model_executor/models/qwen3_next.py` |
| 2026-02-17 | [#34683](https://github.com/vllm-project/vllm/pull/34683) | merged | Revert "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj" | `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/layers/linear.py` |
| 2026-02-18 | [#34697](https://github.com/vllm-project/vllm/pull/34697) | merged | [Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion | `vllm/model_executor/models/qwen3_next.py` |
| 2026-03-09 | [#35777](https://github.com/vllm-project/vllm/pull/35777) | merged | [Kernel] Add fused_sigmoid_gating_delta_rule_update kernel for Qwen3 Next | `vllm/model_executor/models/qwen3_next.py` |
| 2026-03-10 | [#36242](https://github.com/vllm-project/vllm/pull/36242) | merged | [Bugfix] Fix Qwen3-Next in_proj_ba weight sharding with TP > 1 | `vllm/model_executor/models/qwen3_next.py` |
| 2026-03-18 | [#36795](https://github.com/vllm-project/vllm/pull/36795) | merged | [Perf] Enable dual stream execution of input projection for Qwen3 | `vllm/model_executor/models/qwen3_next.py` |
| 2026-03-18 | [#37427](https://github.com/vllm-project/vllm/pull/37427) | merged | [Bugfix] Fix ROCm crash in qwen3_next multi-stream events (#36795) | `vllm/model_executor/models/qwen3_next.py` |
| 2026-04-03 | [#33657](https://github.com/vllm-project/vllm/pull/33657) | merged | [XPU] Initial support for GDN attention on Qwen3-next/Qwen3.5 | `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/layers/layernorm.py`, `vllm/platforms/xpu.py` |
| 2026-04-08 | [#39181](https://github.com/vllm-project/vllm/pull/39181) | merged | [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next | `vllm/model_executor/models/qwen3_next.py` |

## Per-PR Diff Audit Cards

### PR #23994 - [BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ)

- Link: https://github.com/vllm-project/vllm/pull/23994
- Status/date: merged / 2025-09-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +17/-4, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ)"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/layers/quantization/gptq.py`, `vllm/model_executor/layers/quantization/gptq_marlin.py`; PR body summary: The following PR attempts to make the quantized MOE model chart compatible with AutoGPTQ and Autoround-GPTQ. The PR: https://github.com/vllm-project/vllm/issues/23467 attempted....
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +7/-3 (10 lines); hunks: -159,9 +159,13 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, touching `__init__, _maybe_ignore_quant_config`; `vllm/model_executor/layers/quantization/gptq.py` modified +7/-1 (8 lines); hunks: -37,6 +37,7 @@ def __init__(; -74,6 +75,9 @@ def __init__(; symbols: __init__, __repr__, from_config, get_quant_method, touching `__init__, __repr__, from_config`; `vllm/model_executor/layers/quantization/gptq_marlin.py` modified +3/-0 (3 lines); hunks: -119,6 +119,9 @@ def __init__(self, weight_bits: int, group_size: int, desc_a...; symbols: __init__, __repr__, touching `__init__, __repr__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +7/-3 (10 lines); hunks: -159,9 +159,13 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config
  - `vllm/model_executor/layers/quantization/gptq.py` modified +7/-1 (8 lines); hunks: -37,6 +37,7 @@ def __init__(; -74,6 +75,9 @@ def __init__(; symbols: __init__, __repr__, from_config, get_quant_method
  - `vllm/model_executor/layers/quantization/gptq_marlin.py` modified +3/-0 (3 lines); hunks: -119,6 +119,9 @@ def __init__(self, weight_bits: int, group_size: int, desc_a...; symbols: __init__, __repr__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -159,9 +159,13 @@ def __init__(
-        # seems to avoid gate quantization.
-        # See: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4
-        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
+        # seems to avoid gate quantization while AutoRound does.
+        # See: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4,
+        # and https://huggingface.co/jart25/Qwen3-Coder-30B-A3B-Instruct-Int4-gptq
diff -- vllm/model_executor/layers/quantization/gptq.py
@@ -37,6 +37,7 @@ def __init__(
+        autoround_version: str = "",
@@ -74,6 +75,9 @@ def __init__(
+        # used to identify GPTQ model quantized by autoround
+        self.autoround_version = autoround_version
@@ -108,8 +112,10 @@ def from_config(cls, config: dict[str, Any]) -> "GPTQConfig":
+        autoround_version = cls.get_from_keys_or(config, ["autoround_version"],
diff -- vllm/model_executor/layers/quantization/gptq_marlin.py
@@ -119,6 +119,9 @@ def __init__(self, weight_bits: int, group_size: int, desc_act: bool,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +7/-3; `vllm/model_executor/layers/quantization/gptq.py` modified +7/-1; `vllm/model_executor/layers/quantization/gptq_marlin.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/gptq.py`, `vllm/model_executor/layers/quantization/gptq_marlin.py`, `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24526 - Add the support for the qwen3 next model (a hybrid attention model).

- Link: https://github.com/vllm-project/vllm/pull/24526
- Status/date: merged / 2025-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_next_mtp.py`, `vllm/transformers_utils/configs/qwen3_next.py`; associated commits `e93f4cc9e374`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +2476/-61, 3045 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add the support for the qwen3 next model (a hybrid attention model)."; model line: Qwen3 Next; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_next_mtp.py`, `vllm/transformers_utils/configs/qwen3_next.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/qwen3_next.py` added +1294/-0 (1294 lines); hunks: -0,0 +1,1294; symbols: Qwen3NextSparseMoeBlock, __init__, _maybe_ignore_quant_config, forward, touching `Qwen3NextSparseMoeBlock, __init__, _maybe_ignore_quant_config`; `vllm/model_executor/models/qwen3_next_mtp.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: Qwen3NextMultiTokenPredictor, __init__, get_input_embeddings, forward, touching `Qwen3NextMultiTokenPredictor, __init__, get_input_embeddings`; `vllm/transformers_utils/configs/qwen3_next.py` added +275/-0 (275 lines); hunks: -0,0 +1,275; symbols: Qwen3NextConfig, to, __init__, touching `Qwen3NextConfig, to, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` added +1294/-0 (1294 lines); hunks: -0,0 +1,1294; symbols: Qwen3NextSparseMoeBlock, __init__, _maybe_ignore_quant_config, forward
  - `vllm/model_executor/models/qwen3_next_mtp.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: Qwen3NextMultiTokenPredictor, __init__, get_input_embeddings, forward
  - `vllm/transformers_utils/configs/qwen3_next.py` added +275/-0 (275 lines); hunks: -0,0 +1,275; symbols: Qwen3NextConfig, to, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -0,0 +1,1294 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Qwen3Next model."""
+from collections.abc import Iterable
+from typing import Optional
+import torch
diff -- vllm/model_executor/models/qwen3_next_mtp.py
@@ -0,0 +1,285 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Qwen3Next MTP model."""
+from collections.abc import Iterable
+from typing import Optional
+import torch
diff -- vllm/transformers_utils/configs/qwen3_next.py
@@ -0,0 +1,275 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` added +1294/-0; `vllm/model_executor/models/qwen3_next_mtp.py` added +285/-0; `vllm/transformers_utils/configs/qwen3_next.py` added +275/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24709 - [BugFix] Fix Qwen3-Next PP

- Link: https://github.com/vllm-project/vllm/pull/24709
- Status/date: merged / 2025-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `f592b3174b39`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-3, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Fix Qwen3-Next PP"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: Fixes https://github.com/vllm-project/vllm/issues/24703.
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +7/-3 (10 lines); hunks: -2,6 +2,7; -917,8 +918,11 @@ def get_layer(prefix: str):; symbols: get_layer, get_input_embeddings, forward, touching `get_layer, get_input_embeddings, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +7/-3 (10 lines); hunks: -2,6 +2,7; -917,8 +918,11 @@ def get_layer(prefix: str):; symbols: get_layer, get_input_embeddings, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -2,6 +2,7 @@
+from itertools import islice
@@ -917,8 +918,11 @@ def get_layer(prefix: str):
-        self.norm = Qwen3NextRMSNorm(config.hidden_size,
-                                     eps=config.rms_norm_eps)
+        if get_pp_group().is_last_rank:
+            self.norm = Qwen3NextRMSNorm(config.hidden_size,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +7/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24957 - [Bugfix][Qwen3-Next] fixes the varlen issue in qwen3-next's MTP implementation.

- Link: https://github.com/vllm-project/vllm/pull/24957
- Status/date: merged / 2025-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `dd6a910aac65`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +139/-34, 385 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Qwen3-Next] fixes the varlen issue in qwen3-next's MTP implementation."; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: This PR fixes the corner cases where guided decoding backend rollbacks draft tokens, causing unaligned verify batches. Fixes #24730. Fixes #24881..
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +3/-7 (10 lines); hunks: -417,9 +417,7 @@ def _forward(; -458,9 +456,6 @@ def _forward(; symbols: _forward, touching `_forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +3/-7 (10 lines); hunks: -417,9 +417,7 @@ def _forward(; -458,9 +456,6 @@ def _forward(; symbols: _forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -417,9 +417,7 @@ def _forward(
-        num_actual_tokens = (attn_metadata.num_prefill_tokens +
-                             attn_metadata.num_decode_tokens +
-                             attn_metadata.num_spec_decode_tokens)
+        num_actual_tokens = attn_metadata.num_actual_tokens
@@ -458,9 +456,6 @@ def _forward(
-            mixed_qkv_spec = mixed_qkv_spec.view(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +3/-7
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`, `vllm/model_executor/models/qwen3_next.py`, `vllm/v1/attention/backends/gdn_attn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24960 - [Bugfix][Qwen3-Next] add prefixes to shared_expert in qwen3-next and mlp in qwen2moe to successfully load ignored params in quantized models

- Link: https://github.com/vllm-project/vllm/pull/24960
- Status/date: merged / 2025-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `027d37df389b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-23, 101 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Qwen3-Next] add prefixes to shared_expert in qwen3-next and mlp in qwen2moe to successfully load ignored params in quantized models"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: Due to qwen3-next does not pass prefixes as it loads shared_expert modules, and thus, it can not match shared_expert modules with the ignore list in quantization config, and ult....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunks: -138,6 +138,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunks: -138,6 +138,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -138,6 +138,7 @@ def __init__(
+                prefix=f"{prefix}.shared_expert",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25079 - [Qwen] Add fp8 checkpoint support for qwen3-next.

- Link: https://github.com/vllm-project/vllm/pull/25079
- Status/date: merged / 2025-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_next_mtp.py`; associated commits `ef7eefe17a7d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +22/-21, 104 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Qwen] Add fp8 checkpoint support for qwen3-next."; model line: Qwen3 Next; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_next_mtp.py`; PR body summary: Prepare the incoming open-sourced fp8 checkpoints for qwen3-next..
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +17/-18 (35 lines); hunks: -30,7 +30,6; -254,12 +253,20 @@ def __init__(; symbols: __init__, _forward, load_weights, Qwen3NextForCausalLM, touching `__init__, _forward, load_weights`; `vllm/model_executor/models/qwen3_next_mtp.py` modified +5/-3 (8 lines); hunks: -63,7 +63,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; -72,7 +74,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +17/-18 (35 lines); hunks: -30,7 +30,6; -254,12 +253,20 @@ def __init__(; symbols: __init__, _forward, load_weights, Qwen3NextForCausalLM
  - `vllm/model_executor/models/qwen3_next_mtp.py` modified +5/-3 (8 lines); hunks: -63,7 +63,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; -72,7 +74,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -30,7 +30,6 @@
-                                               MergedColumnParallelLinear,
@@ -254,12 +253,20 @@ def __init__(
-        self.in_proj = MergedColumnParallelLinear(
+        self.in_proj_qkvz = ColumnParallelLinear(
-            output_sizes=[self.projection_size_qkvz, self.projection_size_ba],
+            output_size=self.projection_size_qkvz,
diff -- vllm/model_executor/models/qwen3_next_mtp.py
@@ -63,7 +63,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                                       return_bias=False)
+                                       return_bias=False,
+                                       quant_config=quant_config,
+                                       prefix=f'{prefix}.fc')
@@ -72,7 +74,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                prefix=f'{prefix}.layers.{self.mtp_start_layer_idx + idx}',
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +17/-18; `vllm/model_executor/models/qwen3_next_mtp.py` modified +5/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/models/qwen3_next_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25243 - [Qwen] Remove cuda hard-code in qwen3 next

- Link: https://github.com/vllm-project/vllm/pull/25243
- Status/date: merged / 2025-09-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `838d7116ba59`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Qwen] Remove cuda hard-code in qwen3 next"; model line: Qwen3 Next; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: [Qwen] Changed device specification from hard-coded CUDA devices to platform-independent device selectors Test through the exsiting tests.
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +1/-1 (2 lines); hunks: -306,7 +306,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +1/-1 (2 lines); hunks: -306,7 +306,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -306,7 +306,7 @@ def __init__(
-            device=torch.cuda.current_device(),
+            device=current_platform.current_device(),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25268 - [BUGFIX] GPTQ quantization compatibility for Qwen3 Next MOE models (AutoGPTQ and AutoRound-GPTQ)

- Link: https://github.com/vllm-project/vllm/pull/25268
- Status/date: merged / 2025-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `36429096171f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-3, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUGFIX] GPTQ quantization compatibility for Qwen3 Next MOE models (AutoGPTQ and AutoRound-GPTQ)"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: Hi everyone! This PR fixes the same issue as the following PR: https://github.com/vllm-project/vllm/pull/23994 Only for qwen3 next, you need to check if it has been quantized wi....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +5/-3 (8 lines); hunks: -148,9 +148,11 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, touching `__init__, _maybe_ignore_quant_config`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +5/-3 (8 lines); hunks: -148,9 +148,11 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -148,9 +148,11 @@ def __init__(
-        # seems to avoid gate quantization.
-        # See: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4
-        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
+        # seems to avoid gate quantization while AutoRound does.
+        if isinstance(
+                quant_config,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +5/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25743 - [Qwen3-Next][GDN] fixes cuda graph capturing bug in GDN metadata and a stride bug in causal_conv_1d.

- Link: https://github.com/vllm-project/vllm/pull/25743
- Status/date: merged / 2025-09-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +50/-45, 192 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Qwen3-Next][GDN] fixes cuda graph capturing bug in GDN metadata and a stride bug in causal_conv_1d."; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`, `vllm/v1/attention/backends/gdn_attn.py`, `vllm/v1/worker/gpu_model_runner.py`; PR body summary: After #24507 the cuda graph capturing of GDN was broken, see: #25647. To fixes that, this PR refactors the GND's `build_for_cudagraph` part. Along this PR, a potential stride is....
- Key implementation: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` modified +8/-3 (11 lines); hunks: -41,6 +41,7 @@ def _causal_conv1d_fwd_kernel( # continuous batching; -69,7 +70,7 @@ def _causal_conv1d_fwd_kernel( # continuous batching; symbols: _causal_conv1d_fwd_kernel, causal_conv1d_fn, grid, touching `_causal_conv1d_fwd_kernel, causal_conv1d_fn, grid`; `vllm/v1/attention/backends/gdn_attn.py` modified +26/-35 (61 lines); hunks: -125,31 +125,33 @@ def build( # type: ignore[override]; -158,7 +160,6 @@ def build( # type: ignore[override]; symbols: build, build_for_cudagraph_capture, touching `build, build_for_cudagraph_capture`; `vllm/v1/worker/gpu_model_runner.py` modified +16/-7 (23 lines); hunks: -360,8 +360,8 @@ def __init__(; -1099,17 +1099,25 @@ def _prepare_inputs(; symbols: __init__, _prepare_inputs, touching `__init__, _prepare_inputs`.
- Code diff details:
  - `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` modified +8/-3 (11 lines); hunks: -41,6 +41,7 @@ def _causal_conv1d_fwd_kernel( # continuous batching; -69,7 +70,7 @@ def _causal_conv1d_fwd_kernel( # continuous batching; symbols: _causal_conv1d_fwd_kernel, causal_conv1d_fn, grid
  - `vllm/v1/attention/backends/gdn_attn.py` modified +26/-35 (61 lines); hunks: -125,31 +125,33 @@ def build( # type: ignore[override]; -158,7 +160,6 @@ def build( # type: ignore[override]; symbols: build, build_for_cudagraph_capture
  - `vllm/v1/worker/gpu_model_runner.py` modified +16/-7 (23 lines); hunks: -360,8 +360,8 @@ def __init__(; -1099,17 +1099,25 @@ def _prepare_inputs(; symbols: __init__, _prepare_inputs
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mamba/ops/causal_conv1d.py
@@ -41,6 +41,7 @@ def _causal_conv1d_fwd_kernel(  # continuous batching
+    stride_cache_indices: tl.constexpr,
@@ -69,7 +70,7 @@ def _causal_conv1d_fwd_kernel(  # continuous batching
-    idx_seq = tl.load(batch_ptr + tl.program_id(0))
+    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
@@ -91,8 +92,9 @@ def _causal_conv1d_fwd_kernel(  # continuous batching
-        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(
diff -- vllm/v1/attention/backends/gdn_attn.py
@@ -125,31 +125,33 @@ def build(  # type: ignore[override]
-        num_draft_tokens: Optional[torch.Tensor] = None,
+        num_decode_draft_tokens_cpu: Optional[torch.Tensor] = None,
-        seq_lens_tensor = m.seq_lens
-        if (not self.use_spec_decode or num_draft_tokens is None
-                or num_draft_tokens.sum().item() == 0):
+        if (not self.use_spec_decode or num_decode_draft_tokens_cpu is None
diff -- vllm/v1/worker/gpu_model_runner.py
@@ -360,8 +360,8 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` modified +8/-3; `vllm/v1/attention/backends/gdn_attn.py` modified +26/-35; `vllm/v1/worker/gpu_model_runner.py` modified +16/-7
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`, `vllm/v1/attention/backends/gdn_attn.py`, `vllm/v1/worker/gpu_model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26437 - [PERF] Qwen3-next MTP speedup (change bool mask indexing to index_select / index_copy to reduce d2h)

- Link: https://github.com/vllm-project/vllm/pull/26437
- Status/date: merged / 2025-10-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `785d8b6410c3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +56/-36, 203 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[PERF] Qwen3-next MTP speedup (change bool mask indexing to index_select / index_copy to reduce d2h)"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: Qwen3-next MTP suffered from d h memory transfers on prefill phase. That makes MTP slower than STM (at least for some inputs). This PR fixes the following 1. Remove boolean inde....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +12/-12 (24 lines); hunks: -423,7 +423,7 @@ def rearrange_mixed_qkv(self, mixed_qkv):; -455,16 +455,15 @@ def _forward(; symbols: rearrange_mixed_qkv, forward, _forward, touching `rearrange_mixed_qkv, forward, _forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +12/-12 (24 lines); hunks: -423,7 +423,7 @@ def rearrange_mixed_qkv(self, mixed_qkv):; -455,16 +455,15 @@ def _forward(; symbols: rearrange_mixed_qkv, forward, _forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -423,7 +423,7 @@ def rearrange_mixed_qkv(self, mixed_qkv):
-        return query, key, value
+        return query.contiguous(), key.contiguous(), value.contiguous()
@@ -455,16 +455,15 @@ def _forward(
-        spec_token_masks = attn_metadata.spec_token_masks
+        spec_token_indx = attn_metadata.spec_token_indx
+        non_spec_token_indx = attn_metadata.non_spec_token_indx
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +12/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fla/ops/utils.py`, `vllm/model_executor/models/qwen3_next.py`, `vllm/v1/attention/backends/gdn_attn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27030 - [Bugfix][Qwen] fixes the weights dtype in qwen3_next: it is actually a bfloat16

- Link: https://github.com/vllm-project/vllm/pull/27030
- Status/date: merged / 2025-10-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `bde9e2272a28`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-1, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Qwen] fixes the weights dtype in qwen3_next: it is actually a bfloat16"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +0/-1 (1 lines); hunks: -325,7 +325,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +0/-1 (1 lines); hunks: -325,7 +325,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -325,7 +325,6 @@ def __init__(
-                dtype=torch.float32,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26440 - [Performance] Dual stream execution of "shared_experts" and "selected_experts" inside FusedMoE

- Link: https://github.com/vllm-project/vllm/pull/26440
- Status/date: merged / 2025-10-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +122/-22, 271 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Performance] Dual stream execution of "shared_experts" and "selected_experts" inside FusedMoE"; model line: Qwen3 Next; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`; PR body summary: This PR executes the shared_experts part of the FusedMoE on a separate GPU stream, so that the execution is parallelized with the "selected_experts" part. This is possible since....
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +89/-15 (104 lines); hunks: -57,7 +57,7; -1082,6 +1082,17 @@ def __init__(; symbols: __init__, shared_experts, gate, tp_size, touching `__init__, shared_experts, gate`; `vllm/model_executor/models/deepseek_v2.py` modified +12/-6 (18 lines); hunks: -227,6 +227,7 @@ def __init__(; -264,12 +265,17 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` modified +16/-1 (17 lines); hunks: -18,25 +18,40 @@ class SharedFusedMoE(FusedMoE):; symbols: SharedFusedMoE, __init__, shared_experts, gate, touching `SharedFusedMoE, __init__, shared_experts`; `vllm/envs.py` modified +5/-0 (5 lines); hunks: -213,6 +213,7; -1379,6 +1380,10 @@ def get_vllm_port() -> int | None:; symbols: get_default_cache_root, get_vllm_port, touching `get_default_cache_root, get_vllm_port`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +89/-15 (104 lines); hunks: -57,7 +57,7; -1082,6 +1082,17 @@ def __init__(; symbols: __init__, shared_experts, gate, tp_size
  - `vllm/model_executor/models/deepseek_v2.py` modified +12/-6 (18 lines); hunks: -227,6 +227,7 @@ def __init__(; -264,12 +265,17 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: __init__, forward
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` modified +16/-1 (17 lines); hunks: -18,25 +18,40 @@ class SharedFusedMoE(FusedMoE):; symbols: SharedFusedMoE, __init__, shared_experts, gate
  - `vllm/envs.py` modified +5/-0 (5 lines); hunks: -213,6 +213,7; -1379,6 +1380,10 @@ def get_vllm_port() -> int | None:; symbols: get_default_cache_root, get_vllm_port
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -57,7 +57,7 @@
-from vllm.utils.torch_utils import direct_register_custom_op
+from vllm.utils.torch_utils import current_stream, direct_register_custom_op
@@ -1082,6 +1082,17 @@ def __init__(
+        # Allow disabling of the separate shared experts stream for
+        # debug purposes.
+        # TODO: Remove this after more extensive testings with TP/DP
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -227,6 +227,7 @@ def __init__(
+            gate=self.gate,
@@ -264,12 +265,17 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        # router_logits: (num_tokens, n_experts)
-        router_logits, _ = self.gate(hidden_states)
-        fused_moe_out = self.experts(
-            hidden_states=hidden_states, router_logits=router_logits
diff -- vllm/model_executor/layers/fused_moe/shared_fused_moe.py
@@ -18,25 +18,40 @@ class SharedFusedMoE(FusedMoE):
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +89/-15; `vllm/model_executor/models/deepseek_v2.py` modified +12/-6; `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` modified +16/-1; `vllm/envs.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `vllm/envs.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27578 - [perf] Enable concurrent execution of "shared_experts" and "selected_experts" in qwen3-next

- Link: https://github.com/vllm-project/vllm/pull/27578
- Status/date: merged / 2025-10-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `8df98c2161e2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-5, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf] Enable concurrent execution of "shared_experts" and "selected_experts" in qwen3-next"; model line: Qwen3 Next; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: Follow https://github.com/vllm-project/vllm/pull/26440, enable concurrent execution of "shared_experts" and "selected_experts" in qwen3-next Disable multi-stream for shared expe....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +12/-5 (17 lines); hunks: -159,6 +159,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; -181,11 +182,17 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +12/-5 (17 lines); hunks: -159,6 +159,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; -181,11 +182,17 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -159,6 +159,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
+            gate=self.gate,
@@ -181,11 +182,17 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        # router_logits: (num_tokens, n_experts)
-        router_logits, _ = self.gate(hidden_states)
-        final_hidden_states = self.experts(
-            hidden_states=hidden_states, router_logits=router_logits
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +12/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28182 - [Qwen3-Next][Perf][Attention] Replace torch.zeros with torch.empty to reduce overhead

- Link: https://github.com/vllm-project/vllm/pull/28182
- Status/date: open / 2025-11-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Qwen3-Next][Perf][Attention] Replace torch.zeros with torch.empty to reduce overhead"; model line: Qwen3 Next; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: As https://github.com/vllm-project/vllm/pull/26680 and https://github.com/vllm-project/vllm/pull/19784 mentioned, using `torch.zeros` to allocate attention output buffer will in....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +2/-1 (3 lines); hunks: -462,7 +462,7 @@ def forward(; -503,6 +503,7 @@ def _forward_core(; symbols: forward, _forward_core, touching `forward, _forward_core`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +2/-1 (3 lines); hunks: -462,7 +462,7 @@ def forward(; -503,6 +503,7 @@ def _forward_core(; symbols: forward, _forward_core
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -462,7 +462,7 @@ def forward(
-        core_attn_out = torch.zeros(
+        core_attn_out = torch.empty(
@@ -503,6 +503,7 @@ def _forward_core(
+            core_attn_out.fill_(0)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28267 - [Misc] Add some comments in qwen3-next

- Link: https://github.com/vllm-project/vllm/pull/28267
- Status/date: merged / 2025-11-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `7ae5a5fb1115`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Add some comments in qwen3-next"; model line: Qwen3 Next; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: As discussed in https://github.com/vllm-project/vllm/pull/28182, we should not use `torch.empty` to allocate attention output buffer in qwen3 `Qwen3NextGatedDeltaNet`.
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +2/-0 (2 lines); hunks: -462,6 +462,8 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +2/-0 (2 lines); hunks: -462,6 +462,8 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -462,6 +462,8 @@ def forward(
+        # Note: we should not use torch.empty here like other attention backends,
+        # see discussions in https://github.com/vllm-project/vllm/pull/28182
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28202 - [Bugfix] fix qwen3-next crash

- Link: https://github.com/vllm-project/vllm/pull/28202
- Status/date: merged / 2025-11-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `f0359fffa434`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix qwen3-next crash"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: partially fix https://github.com/vllm-project/vllm/issues/27571 In decoding phase with cuda garaph, we will pad for pre-captured cudagraph size. This makes `batch` don't equal t....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +1/-1 (2 lines); hunks: -585,7 +585,7 @@ def _forward_core(; symbols: _forward_core, touching `_forward_core`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +1/-1 (2 lines); hunks: -585,7 +585,7 @@ def _forward_core(; symbols: _forward_core
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -585,7 +585,7 @@ def _forward_core(
-                    : attn_metadata.num_decodes
+                    : attn_metadata.num_actual_tokens
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28960 - [Bugfix] Fix typo in Qwen3 Next model executor

- Link: https://github.com/vllm-project/vllm/pull/28960
- Status/date: merged / 2025-11-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `73ff872db0d4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix typo in Qwen3 Next model executor"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: There is an issue in Qwen3Next model loading because of typo with paddings. Sorry, not sure if PR description is sufficient. But bug is obvious and I fixed it by modifying files....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -1154,8 +1154,8 @@ def set_moe_parameters(self):; symbols: set_moe_parameters, touching `set_moe_parameters`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -1154,8 +1154,8 @@ def set_moe_parameters(self):; symbols: set_moe_parameters
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -1154,8 +1154,8 @@ def set_moe_parameters(self):
-            if example_moe is None:
-                raise RuntimeError("No Qwen3Next layer found in the model.layers.")
+        if example_moe is None:
+            raise RuntimeError("No Qwen3Next layer found in the model.layers.")
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30433 - [Bugfix] Qwen3-next with --hf-overrides \{\"num_hidden_layers\":8\}

- Link: https://github.com/vllm-project/vllm/pull/30433
- Status/date: merged / 2025-12-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `ace34e378320`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-0, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Qwen3-next with --hf-overrides \{\"num_hidden_layers\":8\}"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: I'm testing qwen3 next with `vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --hf-overrides \{\"num_hidden_layers\":8\} --enforce-eager` to fit in one gpu card. But I get this error....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +7/-0 (7 lines); hunks: -1093,6 +1093,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; -1109,6 +1111,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +7/-0 (7 lines); hunks: -1093,6 +1093,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; -1109,6 +1111,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -1093,6 +1093,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+                    if name not in params_dict:
+                        continue
@@ -1109,6 +1111,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+                    if name not in params_dict:
+                        logger.warning_once(
+                            f"Parameter {name} not found in params_dict, skip loading"
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31722 - [PERF] Speed-up of GDN attention decode part (Qwen3-Next)

- Link: https://github.com/vllm-project/vllm/pull/31722
- Status/date: merged / 2026-01-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[PERF] Speed-up of GDN attention decode part (Qwen3-Next)"; model line: Qwen3 Next; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fla/ops/fused_recurrent.py`; PR body summary: Speed Up GDN Attention (Decode) - `fused_recurrent_gated_delta_rule_fwd` Benchmarks H200 — `fused_recurrent_gated_delta_rule_fwd` with shapes from Qwen3-Next Before | Batch Size....
- Key implementation: `vllm/model_executor/layers/fla/ops/fused_recurrent.py` modified +1/-1 (2 lines); hunks: -189,7 +189,7 @@ def fused_recurrent_gated_delta_rule_fwd(; symbols: fused_recurrent_gated_delta_rule_fwd, touching `fused_recurrent_gated_delta_rule_fwd`.
- Code diff details:
  - `vllm/model_executor/layers/fla/ops/fused_recurrent.py` modified +1/-1 (2 lines); hunks: -189,7 +189,7 @@ def fused_recurrent_gated_delta_rule_fwd(; symbols: fused_recurrent_gated_delta_rule_fwd
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fla/ops/fused_recurrent.py
@@ -189,7 +189,7 @@ def fused_recurrent_gated_delta_rule_fwd(
-    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
+    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fla/ops/fused_recurrent.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fla/ops/fused_recurrent.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31719 - [Misc] Support qwen3-next lora

- Link: https://github.com/vllm-project/vllm/pull/31719
- Status/date: merged / 2026-01-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `96fcd3c267a0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-1, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Support qwen3-next lora"; model line: Qwen3 Next; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: Replace the torch.nn.Linear of shared_expert_gate in the qwen3-next model with ReplicatedLinear, so that lora can correctly identify shared_expert_gate..
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +7/-1 (8 lines); hunks: -145,7 +145,13 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +7/-1 (8 lines); hunks: -145,7 +145,13 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -145,7 +145,13 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
-        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
+        self.shared_expert_gate = ReplicatedLinear(
+            config.hidden_size,
+            1,
+            bias=False,
+            quant_config=None,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +7/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34489 - [Bugfix] Fix mamba state dtype setting for Qwen3-Next and Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/34489
- Status/date: merged / 2026-02-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `eea3024f43e0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +42/-6, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix mamba state dtype setting for Qwen3-Next and Qwen3.5"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: Previously conv and ssm state dtypes are coupled for Qwen3-Next, and therefore affected Qwen3.5 which inherits from it. This PR fixes the dtype setting for both models. Note: Fo....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +6/-2 (8 lines); hunks: -341,7 +341,9 @@ def mamba_type(self) -> str:; -1372,7 +1374,9 @@ def get_mamba_state_dtype_from_config(; symbols: mamba_type, get_state_dtype, get_state_shape, get_mamba_state_dtype_from_config, touching `mamba_type, get_state_dtype, get_state_shape`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +6/-2 (8 lines); hunks: -341,7 +341,9 @@ def mamba_type(self) -> str:; -1372,7 +1374,9 @@ def get_mamba_state_dtype_from_config(; symbols: mamba_type, get_state_dtype, get_state_shape, get_mamba_state_dtype_from_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -341,7 +341,9 @@ def mamba_type(self) -> str:
-            self.model_config.dtype, self.cache_config.mamba_cache_dtype
+            self.model_config.dtype,
+            self.cache_config.mamba_cache_dtype,
+            self.cache_config.mamba_ssm_cache_dtype,
@@ -1372,7 +1374,9 @@ def get_mamba_state_dtype_from_config(
-            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +6/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/config.py`, `vllm/model_executor/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34683 - Revert "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj"

- Link: https://github.com/vllm-project/vllm/pull/34683
- Status/date: merged / 2026-02-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +182/-87, 402 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj""; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/layers/linear.py`; PR body summary: FIX https://github.com/vllm-project/vllm/issues/34640 Reverts vllm-project/vllm#34492 It makes the output of qwen3 next as random symbols local-chat-completions (model=Qwen/Qwen....
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +166/-32 (198 lines); hunks: -30,20 +30,36; -57,8 +73,11; symbols: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__, touching `get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering`; `vllm/model_executor/models/qwen3_next.py` modified +10/-27 (37 lines); hunks: -44,7 +44,6; -407,19 +406,19 @@ def __init__(; symbols: __init__, create_qkvz_proj, fix_query_key_value_ordering, touching `__init__, create_qkvz_proj, fix_query_key_value_ordering`; `vllm/model_executor/layers/linear.py` modified +6/-28 (34 lines); hunks: -685,13 +685,8 @@ def weight_loader(; -830,10 +825,7 @@ def weight_loader(; symbols: weight_loader, _load_fused_module_from_checkpoint, weight_loader_v2, touching `weight_loader, _load_fused_module_from_checkpoint, weight_loader_v2`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +166/-32 (198 lines); hunks: -30,20 +30,36; -57,8 +73,11; symbols: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__
  - `vllm/model_executor/models/qwen3_next.py` modified +10/-27 (37 lines); hunks: -44,7 +44,6; -407,19 +406,19 @@ def __init__(; symbols: __init__, create_qkvz_proj, fix_query_key_value_ordering
  - `vllm/model_executor/layers/linear.py` modified +6/-28 (34 lines); hunks: -685,13 +685,8 @@ def weight_loader(; -830,10 +825,7 @@ def weight_loader(; symbols: weight_loader, _load_fused_module_from_checkpoint, weight_loader_v2
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -30,20 +30,36 @@
+from transformers.activations import ACT2FN
+    CacheConfig,
+    ModelConfig,
+    SpeculativeConfig,
+    get_current_vllm_config,
+    divide,
diff -- vllm/model_executor/models/qwen3_next.py
@@ -44,7 +44,6 @@
-    MergedColumnParallelLinear,
@@ -407,19 +406,19 @@ def __init__(
-        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
-        # we need to create qkvz_proj adaptively here.
-        self.in_proj_qkvz = self.create_qkvz_proj(
-            hidden_size=self.hidden_size,
diff -- vllm/model_executor/layers/linear.py
@@ -685,13 +685,8 @@ def weight_loader(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +166/-32; `vllm/model_executor/models/qwen3_next.py` modified +10/-27; `vllm/model_executor/layers/linear.py` modified +6/-28
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34697 - [Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion

- Link: https://github.com/vllm-project/vllm/pull/34697
- Status/date: merged / 2026-02-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `c0bd8b13da36`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +102/-192, 477 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: - Redo https://github.com/vllm-project/vllm/pull/34683 and fix the root issue - Actualy, Qwen3-Next's qkvz_proj output_sizes should be `output_sizes=[sum((key_dim, key_dim, valu....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +27/-10 (37 lines); hunks: -44,6 +44,7; -406,19 +407,19 @@ def __init__(; symbols: __init__, create_qkvz_proj, fix_query_key_value_ordering, touching `__init__, create_qkvz_proj, fix_query_key_value_ordering`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +27/-10 (37 lines); hunks: -44,6 +44,7; -406,19 +407,19 @@ def __init__(; symbols: __init__, create_qkvz_proj, fix_query_key_value_ordering
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -44,6 +44,7 @@
+    MergedColumnParallelLinear,
@@ -406,19 +407,19 @@ def __init__(
-        self.projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
-        self.projection_size_ba = self.num_v_heads * 2
-        self.in_proj_qkvz = ColumnParallelLinear(
-            input_size=self.hidden_size,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +27/-10
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35777 - [Kernel] Add fused_sigmoid_gating_delta_rule_update kernel for Qwen3 Next

- Link: https://github.com/vllm-project/vllm/pull/35777
- Status/date: merged / 2026-03-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `dc6b57846686`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +509/-31, 585 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Add fused_sigmoid_gating_delta_rule_update kernel for Qwen3 Next"; model line: Qwen3 Next; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: This PR adds `fused_sigmoid_gating_delta_rule_update` kernel, to save memory traffic and launch overhead. The idea is inspired by vllm-ascend Summary: * Fuse `fused_gdn_gating`....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +32/-31 (63 lines); hunks: -34,7 +34,7; -731,41 +731,40 @@ def _forward_core(; symbols: _forward_core, touching `_forward_core`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +32/-31 (63 lines); hunks: -34,7 +34,7; -731,41 +731,40 @@ def _forward_core(; symbols: _forward_core
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -34,7 +34,7 @@
-    fused_recurrent_gated_delta_rule,
+    fused_sigmoid_gating_delta_rule_update,
@@ -731,41 +731,40 @@ def _forward_core(
-        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
-        if spec_sequence_masks is not None:
-            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +32/-31
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_sigmoid_gating_delta_rule.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36242 - [Bugfix] Fix Qwen3-Next in_proj_ba weight sharding with TP > 1

- Link: https://github.com/vllm-project/vllm/pull/36242
- Status/date: merged / 2026-03-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `4e95ec111cd1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +45/-6, 72 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3-Next in_proj_ba weight sharding with TP > 1"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: MergedColumnParallelLinear split the interleaved GQA weight at the midpoint, scrambling b/a gating values across TP ranks. Use single-shard output_sizes to preserve the layout.....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +27/-6 (33 lines); hunks: -412,12 +412,11 @@ def __init__(; -497,6 +496,28 @@ def create_qkvz_proj(; symbols: __init__, create_qkvz_proj, create_ba_proj, fix_query_key_value_ordering, touching `__init__, create_qkvz_proj, create_ba_proj`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +27/-6 (33 lines); hunks: -412,12 +412,11 @@ def __init__(; -497,6 +496,28 @@ def create_qkvz_proj(; symbols: __init__, create_qkvz_proj, create_ba_proj, fix_query_key_value_ordering
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -412,12 +412,11 @@ def __init__(
-        # # in_proj_ba is defined as MergedColumnParallelLinear for
-        # compatibility with Qwen3_5.
-        self.in_proj_ba = MergedColumnParallelLinear(
-            input_size=self.hidden_size,
-            output_sizes=[self.num_v_heads] * 2,
-            bias=False,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +27/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36795 - [Perf] Enable dual stream execution of input projection for Qwen3

- Link: https://github.com/vllm-project/vllm/pull/36795
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `a913b612d8a8`, `f1740006e47d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +115/-5, 174 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Enable dual stream execution of input projection for Qwen3"; model line: Qwen3 Next; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: This PR Enable dual stream execution of input projection for Qwen3 Next. * Parallelize the execution of `in_proj_qkvz` and `in_proj_ba` in 2 streams, because their outputs are i....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +61/-3 (64 lines); hunks: -82,7 +82,11; -419,6 +423,12 @@ def __init__(; symbols: __init__, forward, _warmup_prefill_kernels, _forward_in_proj, touching `__init__, forward, _warmup_prefill_kernels`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +61/-3 (64 lines); hunks: -82,7 +82,11; -419,6 +423,12 @@ def __init__(; symbols: __init__, forward, _warmup_prefill_kernels, _forward_in_proj
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -82,7 +82,11 @@
-from vllm.utils.torch_utils import direct_register_custom_op
+from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
+from vllm.utils.torch_utils import (
+    aux_stream,
+    direct_register_custom_op,
+)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +61/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`, `vllm/utils/multi_stream_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37427 - [Bugfix] Fix ROCm crash in qwen3_next multi-stream events (#36795)

- Link: https://github.com/vllm-project/vllm/pull/37427
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `a913b612d8a8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix ROCm crash in qwen3_next multi-stream events (#36795)"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: PR #36795 introduced maybe_execute_in_parallel for qwen3_next but gated CUDA event creation on is_cuda() while the aux stream uses is_cuda_alike(), causing AttributeError: 'None....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +1/-1 (2 lines); hunks: -426,7 +426,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +1/-1 (2 lines); hunks: -426,7 +426,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -426,7 +426,7 @@ def __init__(
-            if current_platform.is_cuda()
+            if current_platform.is_cuda_alike()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33657 - [XPU] Initial support for GDN attention on Qwen3-next/Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/33657
- Status/date: merged / 2026-04-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +150/-0, 185 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU] Initial support for GDN attention on Qwen3-next/Qwen3.5"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/layers/layernorm.py`, `vllm/platforms/xpu.py`; PR body summary: This PR enables Qwen3-next/Qwen3.5 support for XPU path, using triton attention due to k/v in-contiguous not supported. Need mamba cache block size fix like #37467.
- Key implementation: `vllm/model_executor/layers/mamba/gdn_linear_attn.py` modified +94/-0 (94 lines); hunks: -260,6 +260,9 @@ def __init__(; -491,6 +494,13 @@ def forward(; symbols: __init__, forward, forward_cuda, forward_xpu, touching `__init__, forward, forward_cuda`; `vllm/model_executor/layers/layernorm.py` modified +5/-0 (5 lines); hunks: -560,6 +560,11 @@ def forward_cuda(; symbols: forward_cuda, forward_xpu, LayerNorm, touching `forward_cuda, forward_xpu, LayerNorm`; `vllm/platforms/xpu.py` modified +51/-0 (51 lines); hunks: -218,6 +218,57 @@ def check_and_update_config(cls, vllm_config: VllmConfig) -...; symbols: check_and_update_config, update_block_size_for_backend, support_hybrid_kv_cache, touching `check_and_update_config, update_block_size_for_backend, support_hybrid_kv_cache`.
- Code diff details:
  - `vllm/model_executor/layers/mamba/gdn_linear_attn.py` modified +94/-0 (94 lines); hunks: -260,6 +260,9 @@ def __init__(; -491,6 +494,13 @@ def forward(; symbols: __init__, forward, forward_cuda, forward_xpu
  - `vllm/model_executor/layers/layernorm.py` modified +5/-0 (5 lines); hunks: -560,6 +560,11 @@ def forward_cuda(; symbols: forward_cuda, forward_xpu, LayerNorm
  - `vllm/platforms/xpu.py` modified +51/-0 (51 lines); hunks: -218,6 +218,57 @@ def check_and_update_config(cls, vllm_config: VllmConfig) -...; symbols: check_and_update_config, update_block_size_for_backend, support_hybrid_kv_cache
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mamba/gdn_linear_attn.py
@@ -260,6 +260,9 @@ def __init__(
+        self._forward_method = (
+            self.forward_xpu if current_platform.is_xpu() else self.forward_cuda
+        )
@@ -491,6 +494,13 @@ def forward(
+    ):
+        self._forward_method(hidden_states, output)
diff -- vllm/model_executor/layers/layernorm.py
@@ -560,6 +560,11 @@ def forward_cuda(
+    def forward_xpu(
+        self, x: torch.Tensor, z: torch.Tensor | None = None
+    ) -> torch.Tensor:
+        return self.forward_cuda(x, z)
diff -- vllm/platforms/xpu.py
@@ -218,6 +218,57 @@ def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
+    @classmethod
+    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/mamba/gdn_linear_attn.py` modified +94/-0; `vllm/model_executor/layers/layernorm.py` modified +5/-0; `vllm/platforms/xpu.py` modified +51/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/layernorm.py`, `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/platforms/xpu.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39181 - [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next

- Link: https://github.com/vllm-project/vllm/pull/39181
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_next.py`; associated commits `f3c7941ec8d3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-0, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next"; model line: Qwen3 Next; category: bug fix; main diff: `vllm/model_executor/models/qwen3_next.py`; PR body summary: Do not shard shared experts weights when sequence parallel is enabled to fix precision issue for Qwen3.5/Qwen3-Next with EP. At present, when sequence_parallel is enabled, share....
- Key implementation: `vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunks: -140,6 +140,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunks: -140,6 +140,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -140,6 +140,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
+                is_sequence_parallel=self.is_sequence_parallel,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
