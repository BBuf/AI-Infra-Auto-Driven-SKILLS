# TensorRT-LLM EXAONE 4/4.5/K-EXAONE Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py` | [#12873](https://github.com/NVIDIA/TensorRT-LLM/pull/12873) |
| `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` | [#10796](https://github.com/NVIDIA/TensorRT-LLM/pull/10796), [#11862](https://github.com/NVIDIA/TensorRT-LLM/pull/11862) |
| `tensorrt_llm/_torch/models/modeling_exaone4.py` | [#5696](https://github.com/NVIDIA/TensorRT-LLM/pull/5696), [#6397](https://github.com/NVIDIA/TensorRT-LLM/pull/6397), [#8429](https://github.com/NVIDIA/TensorRT-LLM/pull/8429) |
| `tensorrt_llm/_torch/models/modeling_exaone4_5.py` | [#12873](https://github.com/NVIDIA/TensorRT-LLM/pull/12873), [#16992](https://github.com/NVIDIA/TensorRT-LLM/pull/16992) |
| `tensorrt_llm/_torch/models/modeling_exaone_moe.py` | [#10355](https://github.com/NVIDIA/TensorRT-LLM/pull/10355), [#10796](https://github.com/NVIDIA/TensorRT-LLM/pull/10796) |
| `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` | [#12873](https://github.com/NVIDIA/TensorRT-LLM/pull/12873), [#16992](https://github.com/NVIDIA/TensorRT-LLM/pull/16992), [#19246](https://github.com/NVIDIA/TensorRT-LLM/pull/19246) |
| `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py` | [#10355](https://github.com/NVIDIA/TensorRT-LLM/pull/10355) |

## PR Coverage Summary

- Git-traced PRs: 9
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 9
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-07-14 | [#5696](https://github.com/NVIDIA/TensorRT-LLM/pull/5696) | merged | feat: EXAONE4.0 support | `tensorrt_llm/_torch/models/modeling_exaone4.py` |
| 2025-08-04 | [#6397](https://github.com/NVIDIA/TensorRT-LLM/pull/6397) | merged | chore: add EXAONE4 accuracy test | `tensorrt_llm/_torch/models/modeling_exaone4.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4.py` |
| 2025-10-21 | [#8429](https://github.com/NVIDIA/TensorRT-LLM/pull/8429) | merged | [https://nvbugs/5569713][fix] Disable fp8 deep gemm for EXAONE-4.0-32B-FP8 | `tensorrt_llm/_torch/models/modeling_exaone4.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4.py` |
| 2026-01-11 | [#10355](https://github.com/NVIDIA/TensorRT-LLM/pull/10355) | merged | [TRTLLM-10195][feat] K-EXAONE support | `tensorrt_llm/_torch/models/modeling_exaone_moe.py`, `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py` |
| 2026-01-22 | [#10796](https://github.com/NVIDIA/TensorRT-LLM/pull/10796) | merged | [None][feat] K-EXAONE MTP support | `tensorrt_llm/_torch/models/modeling_exaone_moe.py`, `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` |
| 2026-03-05 | [#11862](https://github.com/NVIDIA/TensorRT-LLM/pull/11862) | merged | [None][fix] Prevent RuntimeError from dict mutation during iteration in EXAONE MoE weight mapper | `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` |
| 2026-05-21 | [#12873](https://github.com/NVIDIA/TensorRT-LLM/pull/12873) | merged | [None][feat] EXAONE-4.5 Support | `tensorrt_llm/_torch/models/modeling_exaone4_5.py`, `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` |
| 2026-08-06 | [#16992](https://github.com/NVIDIA/TensorRT-LLM/pull/16992) | merged | [https://nvbugs/6327149][fix] Handle EXAONE 4.5 33B memory constraints | `tensorrt_llm/_torch/models/modeling_exaone4_5.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` |
| 2026-09-17 | [#19246](https://github.com/NVIDIA/TensorRT-LLM/pull/19246) | merged | [None][test] Add coverage for Exaone4ForCausalLM | `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` |

## Per-PR Diff Audit Cards

### PR #5696 - feat: EXAONE4.0 support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/5696
- Status/date: merged / 2025-07-14
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_exaone4.py`; associated commits `63139fdcff31`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +392/-1, 432 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_exaone4.py` added +322/-0 (322 lines); hunks: -0,0 +1,322; symbols: Exaone4Config, check_is_sliding, Exaone4Attention, __init__, touching `Exaone4Config, check_is_sliding, Exaone4Attention`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_exaone4.py` added +322/-0 (322 lines); hunks: -0,0 +1,322; symbols: Exaone4Config, check_is_sliding, Exaone4Attention, __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone4.py
@@ -0,0 +1,322 @@
+from typing import Optional, Tuple
+import torch
+from torch import nn
+from tensorrt_llm._torch.distributed import AllReduceParams
+from tensorrt_llm.functional import PositionEmbeddingType
+from ..attention_backend import AttentionMetadata
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone4.py` added +322/-0
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/__init__.py`, `tensorrt_llm/_torch/models/modeling_exaone4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6397 - chore: add EXAONE4 accuracy test

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/6397
- Status/date: merged / 2025-08-04
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_exaone4.py`; associated commits `ee6ab5be962b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +412/-16, 529 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +5/-16 (21 lines); hunks: -3,7 +3,6; -55,7 +54,6 @@ class Exaone4Attention(Attention):; symbols: Exaone4Attention, __init__, apply_rope, forward, touching `Exaone4Attention, __init__, apply_rope`; `tests/unittest/_torch/modeling/test_modeling_exaone4.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: Exaone4Config, Scenario, __repr__, TestEXAONE4, touching `Exaone4Config, Scenario, __repr__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +5/-16 (21 lines); hunks: -3,7 +3,6; -55,7 +54,6 @@ class Exaone4Attention(Attention):; symbols: Exaone4Attention, __init__, apply_rope, forward
  - `tests/unittest/_torch/modeling/test_modeling_exaone4.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: Exaone4Config, Scenario, __repr__, TestEXAONE4
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone4.py
@@ -3,7 +3,6 @@
-from tensorrt_llm._torch.distributed import AllReduceParams
@@ -55,7 +54,6 @@ class Exaone4Attention(Attention):
-                 is_sliding: bool,
@@ -64,9 +62,10 @@ def __init__(self,
-        self.is_sliding = is_sliding
+        self.sliding_window = config.sliding_window
diff -- tests/unittest/_torch/modeling/test_modeling_exaone4.py
@@ -0,0 +1,384 @@
+import unittest
+from copy import deepcopy
+from dataclasses import dataclass
+import torch
+from parameterized import parameterized
+try:
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +5/-16
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone4.py` added +384/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/examples_test_list.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8429 - [https://nvbugs/5569713][fix] Disable fp8 deep gemm for EXAONE-4.0-32B-FP8

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/8429
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_exaone4.py`; associated commits `ee6944bfa2be`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +56/-3, 126 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +13/-1 (14 lines); hunks: -5,6 +5,7; -54,7 +55,8 @@ class Exaone4Attention(QKNormRoPEAttention):; symbols: Exaone4Attention, __init__, forward, touching `Exaone4Attention, __init__, forward`; `tests/unittest/_torch/modeling/test_modeling_exaone4.py` modified +42/-2 (44 lines); hunks: -1,3 +1,6; -51,8 +54,9 @@ class Exaone4Config(PretrainedConfig):; symbols: Exaone4Config, Scenario, run_forward, test_llm_load, touching `Exaone4Config, Scenario, run_forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +13/-1 (14 lines); hunks: -5,6 +5,7; -54,7 +55,8 @@ class Exaone4Attention(QKNormRoPEAttention):; symbols: Exaone4Attention, __init__, forward
  - `tests/unittest/_torch/modeling/test_modeling_exaone4.py` modified +42/-2 (44 lines); hunks: -1,3 +1,6; -51,8 +54,9 @@ class Exaone4Config(PretrainedConfig):; symbols: Exaone4Config, Scenario, run_forward, test_llm_load
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone4.py
@@ -5,6 +5,7 @@
+from tensorrt_llm.quantization import QuantAlgo
@@ -54,7 +55,8 @@ class Exaone4Attention(QKNormRoPEAttention):
-                 fuse_qk_norm_rope: bool = False):
+                 fuse_qk_norm_rope: bool = False,
+                 disable_deep_gemm: bool = False):
@@ -88,6 +90,7 @@ def __init__(self,
diff -- tests/unittest/_torch/modeling/test_modeling_exaone4.py
@@ -1,3 +1,6 @@
+import json
+import os
+import shutil
@@ -51,8 +54,9 @@ class Exaone4Config(PretrainedConfig):
-    "num_hidden_layers":
-    4,  #NOTE: For testing, we use 4 instead of 64(all layers)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +13/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone4.py` modified +42/-2
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/test_modeling_exaone4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10355 - [TRTLLM-10195][feat] K-EXAONE support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/10355
- Status/date: merged / 2026-01-11
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_exaone_moe.py`, `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py`; associated commits `8e0d20d90153`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +1330/-63, 1562 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_exaone_moe.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: ExaoneMoEConfig, check_is_moe, enable_attn_allreduce, ExaoneMoeAttention, touching `ExaoneMoEConfig, check_is_moe, enable_attn_allreduce`; `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py` added +406/-0 (406 lines); hunks: -0,0 +1,406; symbols: Scenario, __repr__, TestExaoneMoe, test_exaone_moe_sanity, touching `Scenario, __repr__, TestExaoneMoe`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_exaone_moe.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: ExaoneMoEConfig, check_is_moe, enable_attn_allreduce, ExaoneMoeAttention
  - `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py` added +406/-0 (406 lines); hunks: -0,0 +1,406; symbols: Scenario, __repr__, TestExaoneMoe, test_exaone_moe_sanity
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone_moe.py
@@ -0,0 +1,581 @@
+import math
+import os
+import re
+from typing import Dict, List, Optional, Tuple
+import torch
+from torch import nn
diff -- tests/unittest/_torch/modeling/test_modeling_exaone_moe.py
@@ -0,0 +1,406 @@
+import unittest
+from copy import deepcopy
+from dataclasses import dataclass
+import torch
+from _torch.helpers import create_mock_cuda_graph_runner
+from parameterized import parameterized
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone_moe.py` added +581/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py` added +406/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10796 - [None][feat] K-EXAONE MTP support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/10796
- Status/date: merged / 2026-01-22
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_exaone_moe.py`; associated commits `70caa779a469`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +385/-61, 631 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_exaone_moe.py` modified +181/-44 (225 lines); hunks: -1,6 +1,5; -32,15 +31,15; symbols: ExaoneMoEConfig, check_is_moe, enable_attn_allreduce, __init__, touching `ExaoneMoEConfig, check_is_moe, enable_attn_allreduce`; `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: ExaoneMoeWeightMapper, __init__, preprocess_weights, is_special_instance_module, touching `ExaoneMoeWeightMapper, __init__, preprocess_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_exaone_moe.py` modified +181/-44 (225 lines); hunks: -1,6 +1,5; -32,15 +31,15; symbols: ExaoneMoEConfig, check_is_moe, enable_attn_allreduce, __init__
  - `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: ExaoneMoeWeightMapper, __init__, preprocess_weights, is_special_instance_module
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone_moe.py
@@ -1,6 +1,5 @@
-import re
@@ -32,15 +31,15 @@
-from ..modules.linear import TensorParallelMode
+from ..modules.linear import Linear, TensorParallelMode
+from ..modules.multi_stream_utils import maybe_execute_in_parallel
-from ..utils import AuxStreamType, Fp4QuantizedTensor
diff -- tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py
@@ -0,0 +1,67 @@
+from torch import nn
+from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
+from tensorrt_llm._torch.models.modeling_utils import register_mapper
+from tensorrt_llm._torch.modules.fused_moe.interface import MoE
+@register_mapper("HF", "ExaoneMoEForCausalLM")
+class ExaoneMoeWeightMapper(HfWeightMapper):
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone_moe.py` modified +181/-44; `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` added +67/-0
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/model_config.py`, `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_auto.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11862 - [None][fix] Prevent RuntimeError from dict mutation during iteration in EXAONE MoE weight mapper

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11862
- Status/date: merged / 2026-03-05
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py`; associated commits `5b0e8a9290bd`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` modified +1/-1 (2 lines); hunks: -28,7 +28,7 @@ def __init__(self):; symbols: __init__, preprocess_weights, touching `__init__, preprocess_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` modified +1/-1 (2 lines); hunks: -28,7 +28,7 @@ def __init__(self):; symbols: __init__, preprocess_weights
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py
@@ -28,7 +28,7 @@ def __init__(self):
-        for name in weights.keys():
+        for name in list(weights.keys()):
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12873 - [None][feat] EXAONE-4.5 Support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12873
- Status/date: merged / 2026-05-21
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_exaone4_5.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`; associated commits `7279d6322d72`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 18 files, +1269/-338, 2020 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_exaone4_5.py` added +267/-0 (267 lines); hunks: -0,0 +1,267; symbols: Exaone4_5_VisionConfig, Exaone4_5Config, __init__, Exaone4_5InputProcessor, touching `Exaone4_5_VisionConfig, Exaone4_5Config, __init__`; `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py` added +38/-0 (38 lines); hunks: -0,0 +1,38; symbols: Exaone4_5HfWeightMapper, preprocess_weights, touching `Exaone4_5HfWeightMapper, preprocess_weights`; `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: TestExaone4_5Scenario, when, TestExaone4_5, skip_hf_inference, touching `TestExaone4_5Scenario, when, TestExaone4_5`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_exaone4_5.py` added +267/-0 (267 lines); hunks: -0,0 +1,267; symbols: Exaone4_5_VisionConfig, Exaone4_5Config, __init__, Exaone4_5InputProcessor
  - `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py` added +38/-0 (38 lines); hunks: -0,0 +1,38; symbols: Exaone4_5HfWeightMapper, preprocess_weights
  - `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: TestExaone4_5Scenario, when, TestExaone4_5, skip_hf_inference
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone4_5.py
@@ -0,0 +1,267 @@
+# SPDX-License-Identifier: Apache-2.0
+# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+import copy
+from typing import List, Optional, Tuple, Union
+import torch
+from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedModel
diff -- tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py
@@ -0,0 +1,38 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/unittest/_torch/modeling/test_modeling_exaone4_5.py
@@ -0,0 +1,255 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone4_5.py` added +267/-0; `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py` added +38/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` added +255/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16992 - [https://nvbugs/6327149][fix] Handle EXAONE 4.5 33B memory constraints

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16992
- Status/date: merged / 2026-08-06
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_exaone4_5.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`; associated commits `a99395276420`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +59/-2, 116 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_exaone4_5.py` modified +27/-0 (27 lines); hunks: -10,6 +10,7; -36,6 +37,31; symbols: _normalize_exaone4_5_mtp_layer_types, __init__, touching `_normalize_exaone4_5_mtp_layer_types, __init__`; `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +31/-0 (31 lines); hunks: -1,12 +1,14; -110,6 +112,35; symbols: test_exaone4_5_config_normalizes_trailing_mtp_layer_types, test_exaone4_5_config_preserves_unexpected_layer_type_mismatch, TestExaone4_5Scenario, touching `test_exaone4_5_config_normalizes_trailing_mtp_layer_types, test_exaone4_5_config_preserves_unexpected_layer_type_mismatch, TestExaone4_5Scenario`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_exaone4_5.py` modified +27/-0 (27 lines); hunks: -10,6 +10,7; -36,6 +37,31; symbols: _normalize_exaone4_5_mtp_layer_types, __init__
  - `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +31/-0 (31 lines); hunks: -1,12 +1,14; -110,6 +112,35; symbols: test_exaone4_5_config_normalizes_trailing_mtp_layer_types, test_exaone4_5_config_preserves_unexpected_layer_type_mismatch, TestExaone4_5Scenario
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone4_5.py
@@ -10,6 +10,7 @@
+from tensorrt_llm.logger import logger
@@ -36,6 +37,31 @@
+def _normalize_exaone4_5_mtp_layer_types(text_config: dict) -> None:
+    """Remove MTP-only entries from the base decoder layer layout."""
+    layer_types = text_config.get("layer_types")
+    num_hidden_layers = text_config.get("num_hidden_layers")
diff -- tests/unittest/_torch/modeling/test_modeling_exaone4_5.py
@@ -1,12 +1,14 @@
+import copy
+from huggingface_hub.errors import StrictDataclassClassValidationError
@@ -110,6 +112,35 @@
+def test_exaone4_5_config_normalizes_trailing_mtp_layer_types():
+    config = copy.deepcopy(EXAONE_4_5_TEST_CONFIG)
+    text_config = config["text_config"]
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone4_5.py` modified +27/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +31/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/waives.txt`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19246 - [None][test] Add coverage for Exaone4ForCausalLM

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19246
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`; associated commits `2b0421eae607`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +100/-0, 114 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +97/-0 (97 lines); hunks: -112,6 +112,103; symbols: test_exaone4_construction_and_forward, test_exaone4_5_config_normalizes_trailing_mtp_layer_types, touching `test_exaone4_construction_and_forward, test_exaone4_5_config_normalizes_trailing_mtp_layer_types`.
- Code diff details:
  - `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +97/-0 (97 lines); hunks: -112,6 +112,103; symbols: test_exaone4_construction_and_forward, test_exaone4_5_config_normalizes_trailing_mtp_layer_types
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_exaone4_5.py
@@ -112,6 +112,103 @@
+@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
+@torch.inference_mode()
+def test_exaone4_construction_and_forward() -> None:
+    """Check EXAONE 4 construction, BF16 weights, and forward behavior."""
+    from transformers import Exaone4Config
+    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +97/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_h100.yml`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
