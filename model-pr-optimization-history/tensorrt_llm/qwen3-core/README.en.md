# TensorRT-LLM Qwen3 Core Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md` | [#9488](https://github.com/NVIDIA/TensorRT-LLM/pull/9488) |
| `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py` | [#10962](https://github.com/NVIDIA/TensorRT-LLM/pull/10962) |
| `tensorrt_llm/_torch/models/modeling_qwen3.py` | [#4010](https://github.com/NVIDIA/TensorRT-LLM/pull/4010), [#5879](https://github.com/NVIDIA/TensorRT-LLM/pull/5879), [#6785](https://github.com/NVIDIA/TensorRT-LLM/pull/6785), [#7616](https://github.com/NVIDIA/TensorRT-LLM/pull/7616), [#7618](https://github.com/NVIDIA/TensorRT-LLM/pull/7618), [#7765](https://github.com/NVIDIA/TensorRT-LLM/pull/7765), [#7780](https://github.com/NVIDIA/TensorRT-LLM/pull/7780), [#7892](https://github.com/NVIDIA/TensorRT-LLM/pull/7892), [#8030](https://github.com/NVIDIA/TensorRT-LLM/pull/8030), [#8087](https://github.com/NVIDIA/TensorRT-LLM/pull/8087), [#9060](https://github.com/NVIDIA/TensorRT-LLM/pull/9060), [#9689](https://github.com/NVIDIA/TensorRT-LLM/pull/9689), ... (13 total) |
| `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` | [#4010](https://github.com/NVIDIA/TensorRT-LLM/pull/4010), [#4058](https://github.com/NVIDIA/TensorRT-LLM/pull/4058), [#4141](https://github.com/NVIDIA/TensorRT-LLM/pull/4141), [#4304](https://github.com/NVIDIA/TensorRT-LLM/pull/4304), [#4530](https://github.com/NVIDIA/TensorRT-LLM/pull/4530), [#4575](https://github.com/NVIDIA/TensorRT-LLM/pull/4575), [#5206](https://github.com/NVIDIA/TensorRT-LLM/pull/5206), [#5369](https://github.com/NVIDIA/TensorRT-LLM/pull/5369), [#5459](https://github.com/NVIDIA/TensorRT-LLM/pull/5459), [#6199](https://github.com/NVIDIA/TensorRT-LLM/pull/6199), [#6235](https://github.com/NVIDIA/TensorRT-LLM/pull/6235), [#7443](https://github.com/NVIDIA/TensorRT-LLM/pull/7443), ... (17 total) |

## PR Coverage Summary

- Git-traced PRs: 28
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 28
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-05-01 | [#4010](https://github.com/NVIDIA/TensorRT-LLM/pull/4010) | merged | model: support Qwen3 | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`, `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-05-06 | [#4058](https://github.com/NVIDIA/TensorRT-LLM/pull/4058) | merged | Fix: fix bug of qwen3 moe | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-05-09 | [#4141](https://github.com/NVIDIA/TensorRT-LLM/pull/4141) | merged | [TRTLLM-5147][Qwen3] fix: fix bug of attention dp on qwen3_moe model | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-05-15 | [#4304](https://github.com/NVIDIA/TensorRT-LLM/pull/4304) | merged | Add allreduce and rmsnorm fusion for qwen3 | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-05-23 | [#4575](https://github.com/NVIDIA/TensorRT-LLM/pull/4575) | merged | [Fix][Qwen3] fix bug of qwen3 fp4 workflow with EP | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-05-23 | [#4530](https://github.com/NVIDIA/TensorRT-LLM/pull/4530) | merged | Qwen3 supports TRTLLM FP4 MoE backend | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-06-17 | [#5206](https://github.com/NVIDIA/TensorRT-LLM/pull/5206) | merged | [feat] Add EAGLE3 support for Qwen3 | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-06-25 | [#5369](https://github.com/NVIDIA/TensorRT-LLM/pull/5369) | merged | fix: fix bug of qwen3 + eagle3 + finalize_moe_fusion | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-06-30 | [#5459](https://github.com/NVIDIA/TensorRT-LLM/pull/5459) | merged | feat : support duplicate_kv_weight for qwen3 blockwise scale | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-07-18 | [#5879](https://github.com/NVIDIA/TensorRT-LLM/pull/5879) | merged | feat(eagle3):support qwen3 dense model | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-07-24 | [#6235](https://github.com/NVIDIA/TensorRT-LLM/pull/6235) | merged | [Fix][nvbug 5401163][nvbug 5404726][Qwen3] Fix bug of MoE on tp > 1 with trtllm moe backend | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-08-06 | [#6199](https://github.com/NVIDIA/TensorRT-LLM/pull/6199) | merged | Qwen3: Fix eagle hidden states | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-08-16 | [#6785](https://github.com/NVIDIA/TensorRT-LLM/pull/6785) | merged | [None][feat] Support Yarn on Qwen3 | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-10 | [#7616](https://github.com/NVIDIA/TensorRT-LLM/pull/7616) | merged | [https://nvbugs/5505402] [fix] Disable deep_gemm for Qwen3 QKNormRoPEAttention and Linear layers due to accuracy issues | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-16 | [#7618](https://github.com/NVIDIA/TensorRT-LLM/pull/7618) | merged | [None][feat] support attention dp for qwen3 dense model | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-16 | [#7724](https://github.com/NVIDIA/TensorRT-LLM/pull/7724) | merged | [https://nvbugs/5355219][fix] Fix trtllm moe backend test config and Qwen3 MoE multi node | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-09-16 | [#7765](https://github.com/NVIDIA/TensorRT-LLM/pull/7765) | merged | Revert "[None][feat] support attention dp for qwen3 dense model" | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-18 | [#7780](https://github.com/NVIDIA/TensorRT-LLM/pull/7780) | merged | [None][fix] Revert "Revert "[None][feat] support attention dp for qwen3 dense model"" | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-19 | [#7443](https://github.com/NVIDIA/TensorRT-LLM/pull/7443) | merged | [None][feat] Support EPLB in Qwen3 MoE | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-09-28 | [#8030](https://github.com/NVIDIA/TensorRT-LLM/pull/8030) | merged | [https://nvbugs/5461712] [fix] Use DG for Qwen3 Linear layers | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-29 | [#7892](https://github.com/NVIDIA/TensorRT-LLM/pull/7892) | merged | [None][feat] Support Qwen3 next | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-10-03 | [#8075](https://github.com/NVIDIA/TensorRT-LLM/pull/8075) | merged | [None][fix] Fix Qwen3 FP8 per-tensor when requesting TRTLLM-GEN MoE backend | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-10-14 | [#8087](https://github.com/NVIDIA/TensorRT-LLM/pull/8087) | merged | [None][fix] Disable DeepGEMM for Qwen3 MoE Attention layers | `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-11-27 | [#9488](https://github.com/NVIDIA/TensorRT-LLM/pull/9488) | merged | [TRTLLM-9513][docs] Qwen3 deployment guide | `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md` |
| 2025-12-16 | [#9689](https://github.com/NVIDIA/TensorRT-LLM/pull/9689) | merged | [TRTLLM-8310][feat] Add Qwen3-VL-MoE | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`, `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-12-31 | [#9060](https://github.com/NVIDIA/TensorRT-LLM/pull/9060) | merged | [None][feat] support Qwen3-VL dense model in pytorch backend | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2026-01-28 | [#10962](https://github.com/NVIDIA/TensorRT-LLM/pull/10962) | merged | [https://nvbugs/5835925][fix] Add EPD disagg support for Qwen3 VL MoE | `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py` |
| 2026-04-08 | [#12785](https://github.com/NVIDIA/TensorRT-LLM/pull/12785) | merged | [None][fix] Fix LoRA support for Qwen3 models | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`, `tensorrt_llm/_torch/models/modeling_qwen3.py` |

## Per-PR Diff Audit Cards

### PR #4010 - model: support Qwen3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/4010
- Status/date: merged / 2025-05-01
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `129bf199807a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +596/-5, 657 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` added +312/-0 (312 lines); hunks: -0,0 +1,312; symbols: Qwen3MoE, __init__, forward, Qwen3MoEAttention, touching `Qwen3MoE, __init__, forward`; `tensorrt_llm/_torch/models/modeling_qwen3.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: Qwen3Attention, __init__, Qwen3DecoderLayer, forward, touching `Qwen3Attention, __init__, Qwen3DecoderLayer`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` added +312/-0 (312 lines); hunks: -0,0 +1,312; symbols: Qwen3MoE, __init__, forward, Qwen3MoEAttention
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: Qwen3Attention, __init__, Qwen3DecoderLayer, forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -0,0 +1,312 @@
+from typing import Dict, Optional
+import torch
+from torch import nn
+from tqdm import tqdm
+from transformers import Qwen3MoeConfig
+from tensorrt_llm.functional import PositionEmbeddingType
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -0,0 +1,244 @@
+from typing import Optional, Tuple
+import torch
+from torch import nn
+from transformers import Qwen3Config
+from tensorrt_llm.functional import PositionEmbeddingType
+from ..attention_backend import AttentionMetadata
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` added +312/-0; `tensorrt_llm/_torch/models/modeling_qwen3.py` added +244/-0
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/__init__.py`, `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4058 - Fix: fix bug of qwen3 moe

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/4058
- Status/date: merged / 2025-05-06
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `e053cb651bac`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +9/-11, 57 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7; -48,7 +48,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7; -48,7 +48,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -13,7 +13,7 @@
-from ..modules.fused_moe import DefaultMoeRoutingMethod, FusedMoE
+from ..modules.fused_moe import FusedMoE, RenormalizeMoeRoutingMethod
@@ -48,7 +48,7 @@ def __init__(
-            routing_method=DefaultMoeRoutingMethod(top_k=self.top_k),
+            routing_method=RenormalizeMoeRoutingMethod(top_k=self.top_k),
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/cnn_dailymail.yaml`, `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4141 - [TRTLLM-5147][Qwen3] fix: fix bug of attention dp on qwen3_moe model

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/4141
- Status/date: merged / 2025-05-09
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `700d09ab6540`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +25/-11, 64 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +25/-11 (36 lines); hunks: -73,8 +73,10 @@ def forward(; -183,14 +185,23 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfi...; symbols: forward, __init__, load_weights, filter_weights, touching `forward, __init__, load_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +25/-11 (36 lines); hunks: -73,8 +73,10 @@ def forward(; -183,14 +185,23 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfi...; symbols: forward, __init__, load_weights, filter_weights
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -73,8 +73,10 @@ def forward(
-        final_hidden_states = self.experts(hidden_states, router_logits,
-                                           all_rank_num_tokens)
+        final_hidden_states = self.experts(
+            hidden_states,
+            router_logits,
+            all_rank_num_tokens=all_rank_num_tokens)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +25/-11
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4304 - Add allreduce and rmsnorm fusion for qwen3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/4304
- Status/date: merged / 2025-05-15
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `f0ca60a95da5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +75/-13, 183 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +75/-13 (88 lines); hunks: -1,3 +1,4; -6,15 +7,18; symbols: Qwen3MoE, __init__, forward, touching `Qwen3MoE, __init__, forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +75/-13 (88 lines); hunks: -1,3 +1,4; -6,15 +7,18; symbols: Qwen3MoE, __init__, forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -1,3 +1,4 @@
+import os
@@ -6,15 +7,18 @@
+from ..distributed import AllReduce, AllReduceFusionOp, AllReduceParams
+from ..models.modeling_utils import MissingLayer
-                             duplicate_kv_weight, register_auto_model)
+                             EagerFusionConfig, duplicate_kv_weight,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +75/-13
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4575 - [Fix][Qwen3] fix bug of qwen3 fp4 workflow with EP

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/4575
- Status/date: merged / 2025-05-23
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `d69c6622151f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +48/-4, 98 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +48/-4 (52 lines); hunks: -6,14 +6,18; -32,12 +36,15 @@ def __init__(; symbols: __init__, should_enable_alltoall, forward, touching `__init__, should_enable_alltoall, forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +48/-4 (52 lines); hunks: -6,14 +6,18; -32,12 +36,15 @@ def __init__(; symbols: __init__, should_enable_alltoall, forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -6,14 +6,18 @@
+from tensorrt_llm._mnnvl_utils import MnnvlMemory
-from ..distributed import AllReduce, AllReduceFusionOp, AllReduceParams
+from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
+                           allgather)
+from ..utils import disable_fp4_allgather
@@ -32,12 +36,15 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +48/-4
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4530 - Qwen3 supports TRTLLM FP4 MoE backend

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/4530
- Status/date: merged / 2025-05-23
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `bbea2647b1ba`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 15 files, +1939/-166, 2663 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +62/-11 (73 lines); hunks: -1,5 +1,5; -11,21 +11,69; symbols: Qwen3Gate, __init__, forward, load_weights, touching `Qwen3Gate, __init__, forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +62/-11 (73 lines); hunks: -1,5 +1,5; -11,21 +11,69; symbols: Qwen3Gate, __init__, forward, load_weights
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -1,5 +1,5 @@
-from typing import Dict, Optional
+from typing import Dict, List, Optional
@@ -11,21 +11,69 @@
-from ..modules.fused_moe import FusedMoE, RenormalizeMoeRoutingMethod
-from ..modules.linear import Linear, TensorParallelMode
+from ..modules.fused_moe import (BaseMoeRoutingMethod, FusedMoE,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +62/-11
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/thop/test_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5206 - [feat] Add EAGLE3 support for Qwen3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/5206
- Status/date: merged / 2025-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `498fadceb4eb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +16/-10, 81 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +16/-10 (26 lines); hunks: -18,11 +18,13; -203,6 +205,7 @@ def forward(; symbols: Qwen3Gate, forward, touching `Qwen3Gate, forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +16/-10 (26 lines); hunks: -18,11 +18,13; -203,6 +205,7 @@ def forward(; symbols: Qwen3Gate, forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -18,11 +18,13 @@
+from ..speculative import SpecMetadata
-from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
-                             EagerFusionConfig, duplicate_kv_weight,
-                             filter_weights, register_auto_model)
+from .modeling_speculative import SpecDecOneEngineForCausalLM
+from .modeling_utils import (DecoderModel, EagerFusionConfig,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +16/-10
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5369 - fix: fix bug of qwen3 + eagle3 + finalize_moe_fusion

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/5369
- Status/date: merged / 2025-06-25
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `73ba4fc32057`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +11/-3, 31 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +11/-3 (14 lines); hunks: -263,11 +263,11 @@ def forward(; -296,7 +296,15 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +11/-3 (14 lines); hunks: -263,11 +263,11 @@ def forward(; -296,7 +296,15 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -263,11 +263,11 @@ def forward(
-        if spec_metadata:
-            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
-                                                      hidden_states, residual)
+                if spec_metadata:
+                    spec_metadata.maybe_capture_hidden_states(
+                        self.layer_idx, hidden_states, residual)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +11/-3
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5459 - feat : support duplicate_kv_weight for qwen3 blockwise scale

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/5459
- Status/date: merged / 2025-06-30
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `852b79053d51`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +36/-30, 181 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +5/-4 (9 lines); hunks: -394,9 +394,7 @@ def load_weights(self, weights: Dict):; -419,11 +417,14 @@ def load_weights(self, weights: Dict):; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +5/-4 (9 lines); hunks: -394,9 +394,7 @@ def load_weights(self, weights: Dict):; -419,11 +417,14 @@ def load_weights(self, weights: Dict):; symbols: load_weights
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -394,9 +394,7 @@ def load_weights(self, weights: Dict):
-        head_dim = getattr(
-            self.config, "head_dim",
-            self.config.hidden_size // self.config.num_attention_heads)
+        num_kv_heads = self.config.num_key_value_heads
@@ -419,11 +417,14 @@ def load_weights(self, weights: Dict):
+                        if module.quant_config.quant_mode.has_fp8_block_scales(
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +5/-4
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_gemma3.py`, `tensorrt_llm/_torch/models/modeling_mllama.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5879 - feat(eagle3):support qwen3 dense model

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/5879
- Status/date: merged / 2025-07-18
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`; associated commits `28858c871143`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +39/-32, 139 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +12/-32 (44 lines); hunks: -16,8 +16,9; -148,6 +149,7 @@ def forward(; symbols: Qwen3Attention, forward, touching `Qwen3Attention, forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +12/-32 (44 lines); hunks: -16,8 +16,9; -148,6 +149,7 @@ def forward(; symbols: Qwen3Attention, forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -16,8 +16,9 @@
-from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
-                             register_auto_model)
+from ..speculative import SpecMetadata
+from .modeling_speculative import SpecDecOneEngineForCausalLM
+from .modeling_utils import DecoderModel, register_auto_model
@@ -148,6 +149,7 @@ def forward(
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +12/-32
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_h100.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6235 - [Fix][nvbug 5401163][nvbug 5404726][Qwen3] Fix bug of MoE on tp > 1 with trtllm moe backend

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/6235
- Status/date: merged / 2025-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `7b6aadc80056`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +36/-8, 100 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-0 (8 lines); hunks: -309,6 +309,13 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig]):; -381,6 +388,7 @@ def __init__(; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-0 (8 lines); hunks: -309,6 +309,13 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig]):; -381,6 +388,7 @@ def __init__(; symbols: __init__, load_weights
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -309,6 +309,13 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig]):
+        self.preload_weight_modules = []
+        if config.moe_backend == "TRTLLM":
+            self.preload_weight_modules = [
+                "experts",
+                "routing_method",
+                "all_reduce",
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/waives.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6199 - Qwen3: Fix eagle hidden states

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/6199
- Status/date: merged / 2025-08-06
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `7e0158b58334`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +4/-9, 35 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +4/-9 (13 lines); hunks: -214,7 +214,9 @@ def forward(; -257,9 +259,6 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +4/-9 (13 lines); hunks: -214,7 +214,9 @@ def forward(; -257,9 +259,6 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -214,7 +214,9 @@ def forward(
+        if spec_metadata is not None and spec_metadata.is_layer_capture(
+                self.layer_idx):
+            self.fusion_config.POST_MOE_FUSION = False
@@ -257,9 +259,6 @@ def forward(
-                if spec_metadata:
-                    spec_metadata.maybe_capture_hidden_states(
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +4/-9
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6785 - [None][feat] Support Yarn on Qwen3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/6785
- Status/date: merged / 2025-08-16
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`; associated commits `85cbd0263be9`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +208/-31, 360 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +121/-5 (126 lines); hunks: -1,8 +1,9; -21,6 +22,111; symbols: compute_yarn_parameters, get_mscale, find_correction_dim, find_correction_range, touching `compute_yarn_parameters, get_mscale, find_correction_dim`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +121/-5 (126 lines); hunks: -1,8 +1,9; -21,6 +22,111; symbols: compute_yarn_parameters, get_mscale, find_correction_dim, find_correction_range
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -1,8 +1,9 @@
+import math
-from transformers import Qwen3Config
+from transformers import PretrainedConfig, Qwen3Config
@@ -21,6 +22,111 @@
+# Move out from this class
+def compute_yarn_parameters(
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +121/-5
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/thop/test_fused_qk_norm_rope.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7616 - [https://nvbugs/5505402] [fix] Disable deep_gemm for Qwen3 QKNormRoPEAttention and Linear layers due to accuracy issues

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/7616
- Status/date: merged / 2025-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`; associated commits `fc9d426589ad`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +44/-21, 190 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +9/-2 (11 lines); hunks: -48,8 +48,9 @@ def __init__(; -63,6 +64,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +9/-2 (11 lines); hunks: -48,8 +48,9 @@ def __init__(; -63,6 +64,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -48,8 +48,9 @@ def __init__(
-        # Qwen3 has accuracy issues with deep_gemm (see: https://nvbugspro.nvidia.com/bug/5461712)
-        # TODO: Consider adding disable_deep_gemm support to QKNormRoPEAttention if accuracy still remains
+        # Qwen3 has accuracy issues with deep_gemm (see: https://nvbugspro.nvidia.com/bug/5461712
+        # and https://nvbugspro.nvidia.com/bug/5505402)
+        disable_deep_gemm = True
@@ -63,6 +64,7 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +9/-2
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_b200.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7618 - [None][feat] support attention dp for qwen3 dense model

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/7618
- Status/date: merged / 2025-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`; associated commits `96f11b10ae83`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +16/-1, 57 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +16/-1 (17 lines); hunks: -8,6 +8,7; -80,12 +81,15 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +16/-1 (17 lines); hunks: -8,6 +8,7; -80,12 +81,15 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -8,6 +8,7 @@
+from ..distributed import AllReduceParams
@@ -80,12 +81,15 @@ def __init__(
+        self.mapping = model_config.mapping
+        self.enable_attention_dp = self.mapping.enable_attention_dp
+            overridden_tp_size=1 if self.enable_attention_dp else None,
@@ -95,6 +99,8 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +16/-1
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7724 - [https://nvbugs/5355219][fix] Fix trtllm moe backend test config and Qwen3 MoE multi node

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/7724
- Status/date: merged / 2025-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `f9c9c3f50a64`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +17/-7, 85 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-5 (13 lines); hunks: -5,6 +5,7; -187,6 +188,8 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-5 (13 lines); hunks: -5,6 +5,7; -187,6 +188,8 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -5,6 +5,7 @@
+from tensorrt_llm._ipc_utils import can_access_peer
@@ -187,6 +188,8 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],
+        self.is_p2p_supported = can_access_peer(model_config.mapping)
@@ -242,11 +245,11 @@ def forward(
-        do_finalize = not (hidden_states.shape[0]
-                           <= self.moe_allreduce.max_token
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-5
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_full.txt`, `tests/integration/test_lists/test-db/l0_gb200_multi_nodes.yml`, `tests/integration/test_lists/waives.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7765 - Revert "[None][feat] support attention dp for qwen3 dense model"

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/7765
- Status/date: merged / 2025-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`; associated commits `8226ef23dc20`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-16, 58 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-16 (17 lines); hunks: -8,7 +8,6; -83,8 +82,6 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-16 (17 lines); hunks: -8,7 +8,6; -83,8 +82,6 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -8,7 +8,6 @@
-from ..distributed import AllReduceParams
@@ -83,8 +82,6 @@ def __init__(
-        self.mapping = model_config.mapping
-        self.enable_attention_dp = self.mapping.enable_attention_dp
@@ -95,7 +92,6 @@ def __init__(
-            overridden_tp_size=1 if self.enable_attention_dp else None,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-16
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7780 - [None][fix] Revert "Revert "[None][feat] support attention dp for qwen3 dense model""

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/7780
- Status/date: merged / 2025-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`; associated commits `c65457db8a09`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +15/-1, 57 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +15/-1 (16 lines); hunks: -8,6 +8,7; -82,6 +83,8 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +15/-1 (16 lines); hunks: -8,6 +8,7; -82,6 +83,8 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -8,6 +8,7 @@
+from ..distributed import AllReduceParams
@@ -82,6 +83,8 @@ def __init__(
+        self.mapping = model_config.mapping
+        self.enable_attention_dp = self.mapping.enable_attention_dp
@@ -92,6 +95,7 @@ def __init__(
+            overridden_tp_size=1 if self.enable_attention_dp else None,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +15/-1
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7443 - [None][feat] Support EPLB in Qwen3 MoE

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/7443
- Status/date: merged / 2025-09-19
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `0e72e8f7e655`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +10/-6, 58 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +10/-6 (16 lines); hunks: -78,7 +78,7 @@ class Qwen3MoE(nn.Module):; -108,7 +108,7 @@ def __init__(; symbols: Qwen3MoE, __init__, forward, Qwen3MoEDecoderLayer, touching `Qwen3MoE, __init__, forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +10/-6 (16 lines); hunks: -78,7 +78,7 @@ class Qwen3MoE(nn.Module):; -108,7 +108,7 @@ def __init__(; symbols: Qwen3MoE, __init__, forward, Qwen3MoEDecoderLayer
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -78,7 +78,7 @@ class Qwen3MoE(nn.Module):
-        aux_stream: torch.cuda.Stream,
+        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
@@ -108,7 +108,7 @@ def __init__(
-            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
+            aux_stream_dict=aux_stream_dict,
@@ -160,7 +160,8 @@ def forward(
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +10/-6
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8030 - [https://nvbugs/5461712] [fix] Use DG for Qwen3 Linear layers

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/8030
- Status/date: merged / 2025-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`; associated commits `77b68d9d7d59`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +25/-10, 77 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +25/-10 (35 lines); hunks: -2,9 +2,13; -49,10 +53,6 @@ def __init__(; symbols: __init__, post_load_weights, touching `__init__, post_load_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +25/-10 (35 lines); hunks: -2,9 +2,13; -49,10 +53,6 @@ def __init__(; symbols: __init__, post_load_weights
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -2,9 +2,13 @@
+from tqdm import tqdm
+from tensorrt_llm._utils import is_sm_100f
+from tensorrt_llm.quantization.utils.fp8_utils import (
+    resmooth_to_fp8_e8m0, transform_sf_into_required_layout)
@@ -49,10 +53,6 @@ def __init__(
-        # Qwen3 has accuracy issues with deep_gemm (see: https://nvbugspro.nvidia.com/bug/5461712
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +25/-10
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7892 - [None][feat] Support Qwen3 next

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/7892
- Status/date: merged / 2025-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`; associated commits `38d6e4e60b1a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 30 files, +5286/-39, 5588 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +8/-2 (10 lines); hunks: -32,8 +32,12 @@ def __init__(; -58,13 +62,15 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +8/-2 (10 lines); hunks: -32,8 +32,12 @@ def __init__(; -58,13 +62,15 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -32,8 +32,12 @@ def __init__(
+        attn_output_gate: bool = False,
+        use_gemma_rms_norm: bool = False,
+        self.pretrained_config = config
+        self.attn_output_gate = attn_output_gate
@@ -58,13 +62,15 @@ def __init__(
-            bias=config.attention_bias,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +8/-2
- Risk and verification: Runtime changes concentrate in `cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu`, `tensorrt_llm/_torch/custom_ops/__init__.py`, `tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8075 - [None][fix] Fix Qwen3 FP8 per-tensor when requesting TRTLLM-GEN MoE backend

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/8075
- Status/date: merged / 2025-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `9db436690325`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +15/-19, 93 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +14/-15 (29 lines); hunks: -1,5 +1,5; -15,10 +15,12; symbols: __init__, load_weights, routing_method, touching `__init__, load_weights, routing_method`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +14/-15 (29 lines); hunks: -1,5 +1,5; -15,10 +15,12; symbols: __init__, load_weights, routing_method
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -1,5 +1,5 @@
-from typing import Dict, List, Optional
+from typing import Dict, List, Optional, Type
@@ -15,10 +15,12 @@
-from ..modules.fused_moe import (BaseMoeRoutingMethod,
+from ..modules.fused_moe import (BaseMoeRoutingMethod, CutlassFusedMoE,
-                                 RoutingMethodType, create_moe)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +14/-15
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`, `tensorrt_llm/_torch/modules/fused_moe/create_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8087 - [None][fix] Disable DeepGEMM for Qwen3 MoE Attention layers

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/8087
- Status/date: merged / 2025-10-14
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `9bc055faf1e5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +3/-0, 24 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +2/-0 (2 lines); hunks: -34,6 +34,7 @@ def __init__(; -71,6 +72,7 @@ def __init__(; symbols: __init__, touching `__init__`; `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +1/-0 (1 lines); hunks: -168,6 +168,7 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +2/-0 (2 lines); hunks: -34,6 +34,7 @@ def __init__(; -71,6 +72,7 @@ def __init__(; symbols: __init__
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +1/-0 (1 lines); hunks: -168,6 +168,7 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -34,6 +34,7 @@ def __init__(
+        disable_deep_gemm: bool = False,
@@ -71,6 +72,7 @@ def __init__(
+            disable_deep_gemm=disable_deep_gemm,
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -168,6 +168,7 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],
+            disable_deep_gemm=True,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +2/-0; `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9488 - [TRTLLM-9513][docs] Qwen3 deployment guide

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9488
- Status/date: merged / 2025-11-27
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md`; associated commits `5425d9675738`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +257/-0, 263 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md` added +256/-0 (256 lines); hunks: -0,0 +1,256.
- Code diff details:
  - `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md` added +256/-0 (256 lines); hunks: -0,0 +1,256
- Key code excerpts:

```diff
diff -- docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md
@@ -0,0 +1,256 @@
+# Deployment Guide for Qwen3 on TensorRT LLM - Blackwell & Hopper Hardware
+## Introduction
+This is a functional quick-start guide for running the Qwen3 model on TensorRT LLM. It focuses on a working setup with recommended defaults. Additional performance optimizations a
+## Prerequisites
+* GPU: NVIDIA Blackwell or Hopper Architecture
+* OS: Linux
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md` added +256/-0
- Risk and verification: This is mostly docs/examples in `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md`, `docs/source/deployment-guide/index.rst`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #9689 - [TRTLLM-8310][feat] Add Qwen3-VL-MoE

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9689
- Status/date: merged / 2025-12-16
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `8ba8699f66b6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 31 files, +1630/-160, 2437 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +20/-6 (26 lines); hunks: -18,7 +18,7; -114,6 +114,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +6/-1 (7 lines); hunks: -48,7 +48,11 @@ def __init__(; -64,6 +68,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +20/-6 (26 lines); hunks: -18,7 +18,7; -114,6 +114,7 @@ def __init__(; symbols: __init__, forward
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +6/-1 (7 lines); hunks: -48,7 +48,11 @@ def __init__(; -64,6 +68,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -18,7 +18,7 @@
-from ..modules.fused_moe.interface import MoE
+from ..modules.fused_moe.interface import MoE, MoEWeightLoadingMode
@@ -114,6 +114,7 @@ def __init__(
+        self.weight_loading_mode = MoEWeightLoadingMode.FUSED_GATE_UP_PROJ if config.model_type == "qwen3_vl_moe_text" else MoEWeightLoadingMode.VANILLA
@@ -124,6 +125,7 @@ def __init__(
+            weight_loading_mode=self.weight_loading_mode,
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -48,7 +48,11 @@ def __init__(
-            )
+                mrope_section=config.rope_scaling.get("mrope_section", None),
+                mrope_interleaved=config.rope_scaling.get(
+                    "mrope_interleaved", False))
+            if config.rope_scaling.get("mrope_interleaved", False):
+                fuse_qk_norm_rope = False
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +20/-6; `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +6/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_l40s.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9060 - [None][feat] support Qwen3-VL dense model in pytorch backend

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9060
- Status/date: merged / 2025-12-31
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`; associated commits `73870ae4ad12`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +360/-23, 479 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +11/-2 (13 lines); hunks: -121,6 +121,8 @@ def forward(; -137,6 +139,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +11/-2 (13 lines); hunks: -121,6 +121,8 @@ def forward(; -137,6 +139,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -121,6 +121,8 @@ def forward(
+        mrope_config: Optional[dict] = None,
+        deepstack_embeds: Optional[list[torch.Tensor]] = None,
@@ -137,6 +139,7 @@ def forward(
+            mrope_config=mrope_config,
@@ -150,6 +153,9 @@ def forward(
+        if deepstack_embeds is not None and self.layer_idx in range(
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +11/-2
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_modeling_qwen3vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10962 - [https://nvbugs/5835925][fix] Add EPD disagg support for Qwen3 VL MoE

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/10962
- Status/date: merged / 2026-01-28
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py`; associated commits `abb8106c016a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +134/-13, 268 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py` modified +8/-3 (11 lines); hunks: -16,9 +16,6 @@ def init_model_and_config(self, model: Union[nn.Module,; -49,3 +46,11 @@ def _duplicate_kv_weights(self, module: nn.Module, new_name:...; symbols: init_model_and_config, should_skip_module, _duplicate_kv_weights, _num_kv_heads, touching `init_model_and_config, should_skip_module, _duplicate_kv_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py` modified +8/-3 (11 lines); hunks: -16,9 +16,6 @@ def init_model_and_config(self, model: Union[nn.Module,; -49,3 +46,11 @@ def _duplicate_kv_weights(self, module: nn.Module, new_name:...; symbols: init_model_and_config, should_skip_module, _duplicate_kv_weights, _num_kv_heads
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py
@@ -16,9 +16,6 @@ def init_model_and_config(self, model: Union[nn.Module,
-        self._num_kv_heads = model.config.num_key_value_heads if hasattr(
-            model.config, 'num_key_value_heads'
-        ) and model.config.num_key_value_heads is not None else model.config.num_attention_heads
@@ -49,3 +46,11 @@ def _duplicate_kv_weights(self, module: nn.Module, new_name: str,
+    @property
+    def _num_kv_heads(self) -> int:
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py` modified +8/-3
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/multimodal/test_mm_encoder_standalone.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12785 - [None][fix] Fix LoRA support for Qwen3 models

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12785
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`; associated commits `ae86b91e12b0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +204/-1, 220 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-1 (3 lines); hunks: -398,7 +398,8 @@ def forward(; symbols: forward, touching `forward`; `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-0 (1 lines); hunks: -121,6 +121,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-1 (3 lines); hunks: -398,7 +398,8 @@ def forward(; symbols: forward
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-0 (1 lines); hunks: -121,6 +121,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -398,7 +398,8 @@ def forward(
-                deepstack_embeds=deepstack_embeds)
+                deepstack_embeds=deepstack_embeds,
+                **kwargs)
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -121,6 +121,7 @@ def __init__(
+            layer_idx=layer_idx,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-1; `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modules/tests_lora_modules/test_qwen3_sanity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
