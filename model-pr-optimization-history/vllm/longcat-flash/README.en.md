# vLLM LongCat-Flash Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/tool_parsers/test_longcat_tool_parser.py` | no direct PR-number commit |
| `vllm/model_executor/models/longcat_flash.py` | [#23991](https://github.com/vllm-project/vllm/pull/23991), [#28891](https://github.com/vllm-project/vllm/pull/28891), [#41448](https://github.com/vllm-project/vllm/pull/41448), [#47857](https://github.com/vllm-project/vllm/pull/47857) |
| `vllm/model_executor/models/longcat_flash_mtp.py` | [#23991](https://github.com/vllm-project/vllm/pull/23991), [#47857](https://github.com/vllm-project/vllm/pull/47857) |
| `vllm/model_executor/models/longcat_flash_ngram.py` | [#47857](https://github.com/vllm-project/vllm/pull/47857) |
| `vllm/tool_parsers/longcat_tool_parser.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 4
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 4
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-09-25 | [#23991](https://github.com/vllm-project/vllm/pull/23991) | merged | [Model] Add LongCat-Flash | `vllm/model_executor/models/longcat_flash.py`, `vllm/model_executor/models/longcat_flash_mtp.py` |
| 2025-12-20 | [#28891](https://github.com/vllm-project/vllm/pull/28891) | merged | [MoE Refactor][5/N] Isolate zero expert to LongCatFlash | `vllm/model_executor/models/longcat_flash.py` |
| 2026-05-01 | [#41448](https://github.com/vllm-project/vllm/pull/41448) | merged | Refractor longcat loading to use AutoWeightsLoader | `vllm/model_executor/models/longcat_flash.py` |
| 2026-07-10 | [#47857](https://github.com/vllm-project/vllm/pull/47857) | merged | [Model] Add LongCat-Flash-Lite (n-gram embedding) | `vllm/model_executor/models/longcat_flash_ngram.py`, `vllm/model_executor/models/longcat_flash_mtp.py`, `vllm/model_executor/models/longcat_flash.py` |

## Per-PR Diff Audit Cards

### PR #23991 - [Model] Add LongCat-Flash

- Link: https://github.com/vllm-project/vllm/pull/23991
- Status/date: merged / 2025-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/longcat_flash.py`, `vllm/model_executor/models/longcat_flash_mtp.py`; associated commits `845adb3ec6d7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 31 files, +1357/-66, 2009 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/longcat_flash.py` added +712/-0 (712 lines); hunks: -0,0 +1,712; symbols: FlashConfig, __init__, FlashMLP, forward, touching `FlashConfig, __init__, FlashMLP`; `vllm/model_executor/models/longcat_flash_mtp.py` added +352/-0 (352 lines); hunks: -0,0 +1,352; symbols: LongCatMultiTokenPredictorLayer, __init__, forward, LongCatMultiTokenPredictor, touching `LongCatMultiTokenPredictorLayer, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/longcat_flash.py` added +712/-0 (712 lines); hunks: -0,0 +1,712; symbols: FlashConfig, __init__, FlashMLP, forward
  - `vllm/model_executor/models/longcat_flash_mtp.py` added +352/-0 (352 lines); hunks: -0,0 +1,352; symbols: LongCatMultiTokenPredictorLayer, __init__, forward, LongCatMultiTokenPredictor
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/longcat_flash.py
@@ -0,0 +1,712 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Apache License, Version 2.0:
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- vllm/model_executor/models/longcat_flash_mtp.py
@@ -0,0 +1,352 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from
+# https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/deepseek_mtp.py
+from collections.abc import Iterable
+from typing import Optional
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/longcat_flash.py` added +712/-0; `vllm/model_executor/models/longcat_flash_mtp.py` added +352/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_flashinfer.py`, `tests/models/registry.py`, `tests/models/utils.py`, `tests/test_routing_simulator.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28891 - [MoE Refactor][5/N] Isolate zero expert to LongCatFlash

- Link: https://github.com/vllm-project/vllm/pull/28891
- Status/date: merged / 2025-12-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/longcat_flash.py`; associated commits `54c892438479`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 19 files, +263/-108, 709 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/longcat_flash.py` modified +37/-14 (51 lines); hunks: -46,7 +46,7; -179,7 +179,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/longcat_flash.py` modified +37/-14 (51 lines); hunks: -46,7 +46,7; -179,7 +179,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/longcat_flash.py
@@ -46,7 +46,7 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import FusedMoE, ZeroExpertFusedMoE
@@ -179,7 +179,7 @@ def __init__(
-            else self.intermediate_size
+            else intermediate_size
@@ -280,48 +280,69 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/longcat_flash.py` modified +37/-14
- Risk and verification: The diff ships test coverage in `tests/test_routing_simulator.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41448 - Refractor longcat loading to use AutoWeightsLoader

- Link: https://github.com/vllm-project/vllm/pull/41448
- Status/date: merged / 2026-05-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/longcat_flash.py`; associated commits `0dbaf9daad20`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +82/-73, 188 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/longcat_flash.py` modified +82/-73 (155 lines); hunks: -69,6 +69,7; -485,6 +486,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, LongcatFlashForCausalLM, embed_input_ids, touching `__init__, forward, LongcatFlashForCausalLM`.
- Code diff details:
  - `vllm/model_executor/models/longcat_flash.py` modified +82/-73 (155 lines); hunks: -69,6 +69,7; -485,6 +486,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, LongcatFlashForCausalLM, embed_input_ids
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/longcat_flash.py
@@ -69,6 +69,7 @@
+    AutoWeightsLoader,
@@ -485,6 +486,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self.quant_config = quant_config
@@ -551,77 +553,6 @@ def forward(
-class LongcatFlashForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
-    """Flash model for causal language modeling."""
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/longcat_flash.py` modified +82/-73
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/longcat_flash.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #47857 - [Model] Add LongCat-Flash-Lite (n-gram embedding)

- Link: https://github.com/vllm-project/vllm/pull/47857
- Status/date: merged / 2026-07-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/longcat_flash.py`, `vllm/model_executor/models/longcat_flash_mtp.py`, `vllm/model_executor/models/longcat_flash_ngram.py`; associated commits `08dfd68610d2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 16 files, +630/-12, 782 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/longcat_flash_ngram.py` added +405/-0 (405 lines); hunks: -0,0 +1,405; symbols: uses_ngram_embedding, _config_dtype, NgramEmbedding, __init__, touching `uses_ngram_embedding, _config_dtype, NgramEmbedding`; `vllm/model_executor/models/longcat_flash_mtp.py` modified +14/-4 (18 lines); hunks: -126,8 +126,10 @@ def forward(; -287,14 +289,22 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: forward, LongCatFlashMTP, __init__, load_weights, touching `forward, LongCatFlashMTP, __init__`; `vllm/model_executor/models/longcat_flash.py` modified +12/-3 (15 lines); hunks: -318,7 +318,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; -687,14 +687,23 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: forward, load_weights, touching `forward, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/longcat_flash_ngram.py` added +405/-0 (405 lines); hunks: -0,0 +1,405; symbols: uses_ngram_embedding, _config_dtype, NgramEmbedding, __init__
  - `vllm/model_executor/models/longcat_flash_mtp.py` modified +14/-4 (18 lines); hunks: -126,8 +126,10 @@ def forward(; -287,14 +289,22 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: forward, LongCatFlashMTP, __init__, load_weights
  - `vllm/model_executor/models/longcat_flash.py` modified +12/-3 (15 lines); hunks: -318,7 +318,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; -687,14 +687,23 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: forward, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/longcat_flash_ngram.py
@@ -0,0 +1,405 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only LongCat-Flash-Lite (n-gram embedding) model.
+``LongcatFlashNgramForCausalLM`` is LongCat-Flash (MLA dual-attention +
+zero-expert MoE + YaRN) plus an n-gram embedding input layer: each position's
+embedding fuses the token embedding with hashed embeddings of the preceding
diff -- vllm/model_executor/models/longcat_flash_mtp.py
@@ -126,8 +126,10 @@ def forward(
-        # LongCat MTP without MoE layers
-        vllm_config.model_config.hf_config.n_routed_experts = None
+        # LongCat MTP has no MoE layers: clear n_routed_experts so the predictor
+        # builds a dense MLP. object.__setattr__ bypasses the ngram remote
+        # config's strict validation (it rejects setting the int field to None).
+        object.__setattr__(vllm_config.model_config.hf_config, "n_routed_experts", None)
diff -- vllm/model_executor/models/longcat_flash.py
@@ -318,7 +318,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/longcat_flash_ngram.py` added +405/-0; `vllm/model_executor/models/longcat_flash_mtp.py` modified +14/-4; `vllm/model_executor/models/longcat_flash.py` modified +12/-3
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/models/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
