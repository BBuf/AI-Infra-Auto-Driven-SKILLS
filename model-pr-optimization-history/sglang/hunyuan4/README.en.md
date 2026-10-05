# SGLang Hunyuan V4 (Hy4) Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` | [#36804](https://github.com/sgl-project/sglang/pull/36804), [#36808](https://github.com/sgl-project/sglang/pull/36808), [#36823](https://github.com/sgl-project/sglang/pull/36823) |
| `docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx` | [#36804](https://github.com/sgl-project/sglang/pull/36804) |
| `docs/src/snippets/configs/tencent/hy4-preview.jsx` | [#36804](https://github.com/sgl-project/sglang/pull/36804), [#36808](https://github.com/sgl-project/sglang/pull/36808) |
| `python/sglang/kernels/ops/layernorm/hy4_ihc.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `python/sglang/srt/configs/hy_v4.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `python/sglang/srt/models/hunyuan_v4.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805), [#42477](https://github.com/sgl-project/sglang/pull/42477) |
| `python/sglang/srt/models/hunyuan_v4_nextn.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805), [#42477](https://github.com/sgl-project/sglang/pull/42477) |
| `test/registered/kernels/ops/attention/test_hy4_hpc_gated_mla.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `test/registered/kernels/ops/layernorm/test_hy4_decode_kernels.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `test/registered/kernels/ops/layernorm/test_hy4_hpc_ihc.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `test/registered/unit/models/test_hunyuan_v4.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `test/registered/unit/models/test_hunyuan_v4_nextn_weight_loading.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |

## PR Coverage Summary

- Git-traced PRs: 5
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 5
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-08-28 | [#36804](https://github.com/sgl-project/sglang/pull/36804) | merged | [Cookbook] Add the Hy4-Preview model page (Tencent) | `docs/src/snippets/configs/tencent/hy4-preview.jsx`, `docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx`, `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` |
| 2026-08-28 | [#36808](https://github.com/sgl-project/sglang/pull/36808) | merged | [Cookbook] Hy4-Preview follow-ups: runtime-accurate recipes + released-model info | `docs/src/snippets/configs/tencent/hy4-preview.jsx`, `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` |
| 2026-08-28 | [#36823](https://github.com/sgl-project/sglang/pull/36823) | merged | [Docs] Rename Tencent cookbook page titles to "Hy4 preview" / "Hy3 preview" | `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` |
| 2026-09-05 | [#36805](https://github.com/sgl-project/sglang/pull/36805) | merged | Support Hy4-preview | `python/sglang/srt/models/hunyuan_v4.py`, `python/sglang/srt/models/hunyuan_v4_nextn.py`, `python/sglang/srt/configs/hy_v4.py` |
| 2026-10-04 | [#42477](https://github.com/sgl-project/sglang/pull/42477) | merged | [Refactor] Build the Hunyuan V4 decoder from stage boundaries | `python/sglang/srt/models/hunyuan_v4.py`, `python/sglang/srt/models/hunyuan_v4_nextn.py` |

## Per-PR Diff Audit Cards

### PR #36804 - [Cookbook] Add the Hy4-Preview model page (Tencent)

- Link: https://github.com/sgl-project/sglang/pull/36804
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`, `docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx`, `docs/src/snippets/configs/tencent/hy4-preview.jsx`; associated commits `1948b61ad4d7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +888/-2, 914 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/tencent/hy4-preview.jsx` added +574/-0 (574 lines); hunks: -0,0 +1,574; `docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx` added +33/-0 (33 lines); hunks: -0,0 +1,33; `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` added +279/-0 (279 lines); hunks: -0,0 +1,279.
- Code diff details:
  - `docs/src/snippets/configs/tencent/hy4-preview.jsx` added +574/-0 (574 lines); hunks: -0,0 +1,574
  - `docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx` added +33/-0 (33 lines); hunks: -0,0 +1,33
  - `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` added +279/-0 (279 lines); hunks: -0,0 +1,279
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/tencent/hy4-preview.jsx
@@ -0,0 +1,574 @@
+// Hy4-Preview cookbook config. Consumed by _deployment.jsx + _playground.jsx;
+// see _deployment.jsx header for the field contract.
+//
+// Sizing (drives the TP/nodes choices below):
+// ~760B total / ~40B active MoE. BF16 weights ≈ 1.5TB → TP16 on H200/B200
+// (2x8 multi-node) or TP8 on B300 (single 8-GPU node) / GB300 (2x4 multi-node
diff -- docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx
@@ -0,0 +1,33 @@
+// Hy4-Preview benchmark data — one entry per config cell `match` tuple.
+//
+// All entries are bare-match stubs (the card shows "pending"). When a cell's
+// numbers land, fill the entry with the measured data and set
+// `sglang_version` to the exact commit/tag they were measured on (a
+// reproducible anchor — never a moving ref like "main").
diff -- docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx
@@ -0,0 +1,279 @@
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/tencent/hy4-preview.jsx` added +574/-0; `docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx` added +33/-0; `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` added +279/-0
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/Tencent/Hy3.mdx`, `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`, `docs/cookbook/autoregressive/intro.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #36808 - [Cookbook] Hy4-Preview follow-ups: runtime-accurate recipes + released-model info

- Link: https://github.com/sgl-project/sglang/pull/36808
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`, `docs/src/snippets/configs/tencent/hy4-preview.jsx`; associated commits `2960d696228e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +42/-59, 285 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/tencent/hy4-preview.jsx` modified +24/-43 (67 lines); hunks: -2,7 +2,7; -12,10 +12,11; `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +18/-16 (34 lines); hunks: -1,6 +1,6; -10,7 +10,7 @@ tag: NEW.
- Code diff details:
  - `docs/src/snippets/configs/tencent/hy4-preview.jsx` modified +24/-43 (67 lines); hunks: -2,7 +2,7; -12,10 +12,11
  - `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +18/-16 (34 lines); hunks: -1,6 +1,6; -10,7 +10,7 @@ tag: NEW
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/tencent/hy4-preview.jsx
@@ -2,7 +2,7 @@
-// ~760B total / ~40B active MoE. BF16 weights ≈ 1.5TB → TP16 on H200/B200
+// 770B total / 49B active MoE. BF16 weights ≈ 1.5TB → TP16 on H200/B200
@@ -12,10 +12,11 @@
-// Every cell carries `verificationStatus: "in-progress"`. When a recipe's
-// end-to-end verification lands, REPLACE that line with `verified: true` —
-// `verificationStatus` takes precedence over `verified` in the engine, so
diff -- docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx
@@ -1,6 +1,6 @@
-description: "Deploy Tencent Hy4-Preview with SGLang — launch recipes for the 760B-parameter Mixture-of-Experts model with MLA, DeepSeek Sparse Attention (DSA), and MTP speculativ
+description: "Deploy Tencent Hy4-Preview with SGLang — launch recipes for the 770B-parameter Mixture-of-Experts model with MLA, DeepSeek Sparse Attention (DSA), and MTP speculativ
@@ -10,7 +10,7 @@ tag: NEW
-For all methods and hardware platforms, see the [official SGLang installation guide](/docs/get-started/install). The two paths below match the **Python / Docker** toggle in the co
+For all methods and hardware platforms, see the [official SGLang installation guide](/docs/get-started/install). The Docker path below matches the **Docker** framing in the comman
@@ -50,7 +50,7 @@ import { benchmarks } from "/src/snippets/configs/tencent/hy4-preview-benchmarks
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/tencent/hy4-preview.jsx` modified +24/-43; `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +18/-16
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`, `docs/src/snippets/configs/tencent/hy4-preview.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #36823 - [Docs] Rename Tencent cookbook page titles to "Hy4 preview" / "Hy3 preview"

- Link: https://github.com/sgl-project/sglang/pull/36823
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`; associated commits `989e51ba9c4c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +12/-12, 87 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +9/-9 (18 lines); hunks: -1,6 +1,6; -78,7 +78,7 @@ import { Playground } from "/src/snippets/_playground.jsx";.
- Code diff details:
  - `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +9/-9 (18 lines); hunks: -1,6 +1,6; -78,7 +78,7 @@ import { Playground } from "/src/snippets/_playground.jsx";
- Key code excerpts:

```diff
diff -- docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx
@@ -1,6 +1,6 @@
-title: Hy4-Preview
-description: "Deploy Tencent Hy4-Preview with SGLang — launch recipes for the 770B-parameter Mixture-of-Experts model with MLA, DeepSeek Sparse Attention (DSA), and MTP speculativ
+title: Hy4 preview
+description: "Deploy Tencent Hy4 preview with SGLang — launch recipes for the 770B-parameter Mixture-of-Experts model with MLA, DeepSeek Sparse Attention (DSA), and MTP speculativ
@@ -78,7 +78,7 @@ import { Playground } from "/src/snippets/_playground.jsx";
-**Hy4-Preview** is Tencent's next-generation flagship Mixture-of-Experts language model: 770B total parameters with 49B active per token, pairing a DeepSeek-style MLA + sparse-att
```

- Extracted files (not manually reviewed):
  - docs: `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +9/-9
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #36805 - Support Hy4-preview

- Link: https://github.com/sgl-project/sglang/pull/36805
- Status/date: merged / 2026-09-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/ops/layernorm/hy4_ihc.py`, `python/sglang/srt/configs/hy_v4.py`, `python/sglang/srt/models/hunyuan_v4.py`, `python/sglang/srt/models/hunyuan_v4_nextn.py`, `test/registered/kernels/ops/attention/test_hy4_hpc_gated_mla.py` and 9 files; associated commits `55bf3380e073`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 47 files, +3387/-103, 4334 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/hunyuan_v4.py` added +734/-0 (734 lines); hunks: -0,0 +1,734; symbols: _hpc_ihc_available, permute_hyv4_indexer_weight, HYV4HCPreLayer, __init__, touching `_hpc_ihc_available, permute_hyv4_indexer_weight, HYV4HCPreLayer`; `python/sglang/srt/models/hunyuan_v4_nextn.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: _mtp_quant_config, HYV4MTPDecoderLayer, __init__, forward, touching `_mtp_quant_config, HYV4MTPDecoderLayer, __init__`; `python/sglang/srt/configs/hy_v4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: HYV4Config, __init__, _validate_hy_v4, touching `HYV4Config, __init__, _validate_hy_v4`; `test/registered/unit/models/test_hunyuan_v4.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: test_attention_gate_uses_attention_tp, fake_attention_init, FakeColumnParallelLinear, __init__, touching `test_attention_gate_uses_attention_tp, fake_attention_init, FakeColumnParallelLinear`.
- Code diff details:
  - `python/sglang/srt/models/hunyuan_v4.py` added +734/-0 (734 lines); hunks: -0,0 +1,734; symbols: _hpc_ihc_available, permute_hyv4_indexer_weight, HYV4HCPreLayer, __init__
  - `python/sglang/srt/models/hunyuan_v4_nextn.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: _mtp_quant_config, HYV4MTPDecoderLayer, __init__, forward
  - `python/sglang/srt/configs/hy_v4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: HYV4Config, __init__, _validate_hy_v4
  - `test/registered/unit/models/test_hunyuan_v4.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: test_attention_gate_uses_attention_tp, fake_attention_init, FakeColumnParallelLinear, __init__
  - `test/registered/unit/models/test_hunyuan_v4_nextn_weight_loading.py` added +42/-0 (42 lines); hunks: -0,0 +1,42; symbols: TestHunyuanV4NextNWeightLoading, test_indexer_checkpoint_layout_is_permuted
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/hunyuan_v4.py
@@ -0,0 +1,734 @@
+import functools
+import logging
+from typing import Iterable, Tuple
+import torch
+from torch import nn
+from transformers import PretrainedConfig
diff -- python/sglang/srt/models/hunyuan_v4_nextn.py
@@ -0,0 +1,229 @@
+import copy
+from typing import Iterable, Tuple
+import torch
+from torch import nn
+from sglang.srt.distributed import get_pp_group
+from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
diff -- python/sglang/srt/configs/hy_v4.py
@@ -0,0 +1,152 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/hunyuan_v4.py` added +734/-0; `python/sglang/srt/models/hunyuan_v4_nextn.py` added +229/-0; `python/sglang/srt/configs/hy_v4.py` added +152/-0; `python/sglang/kernels/ops/layernorm/hy4_ihc.py` added +443/-0
  - tests: `test/registered/unit/models/test_hunyuan_v4.py` added +119/-0; `test/registered/unit/models/test_hunyuan_v4_nextn_weight_loading.py` added +42/-0; `test/registered/kernels/ops/layernorm/test_hy4_decode_kernels.py` added +229/-0; `test/registered/kernels/ops/attention/test_hy4_hpc_gated_mla.py` added +145/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/attention/test_dsa_sink_pad_cache.py`, `test/registered/kernels/ops/attention/test_hy4_hpc_gated_mla.py`, `test/registered/kernels/ops/layernorm/test_hy4_decode_kernels.py`, `test/registered/kernels/ops/layernorm/test_hy4_hpc_ihc.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42477 - [Refactor] Build the Hunyuan V4 decoder from stage boundaries

- Link: https://github.com/sgl-project/sglang/pull/42477
- Status/date: merged / 2026-10-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/hunyuan_v4.py`, `python/sglang/srt/models/hunyuan_v4_nextn.py`; associated commits `92d60351e2eb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +531/-34, 705 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/hunyuan_v4.py` modified +101/-20 (121 lines); hunks: -7,7 +7,14; -342,7 +349,8 @@ def __init__(; symbols: __init__, dispatch_attn_forward_method, _HeadNorm, __call__, touching `__init__, dispatch_attn_forward_method, _HeadNorm`; `python/sglang/srt/models/hunyuan_v4_nextn.py` modified +29/-14 (43 lines); hunks: -5,7 +5,13; -61,9 +67,19 @@ def __init__(self, config, quant_config=None, prefix="", alt_...; symbols: __init__, forward, HYV4ModelNextN, touching `__init__, forward, HYV4ModelNextN`.
- Code diff details:
  - `python/sglang/srt/models/hunyuan_v4.py` modified +101/-20 (121 lines); hunks: -7,7 +7,14; -342,7 +349,8 @@ def __init__(; symbols: __init__, dispatch_attn_forward_method, _HeadNorm, __call__
  - `python/sglang/srt/models/hunyuan_v4_nextn.py` modified +29/-14 (43 lines); hunks: -5,7 +5,13; -61,9 +67,19 @@ def __init__(self, config, quant_config=None, prefix="", alt_...; symbols: __init__, forward, HYV4ModelNextN
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/hunyuan_v4.py
@@ -7,7 +7,14 @@
-from sglang.srt.layers.layer_boundary import AttentionInputs, get_attn_tp_context
+from sglang.srt.layers.layer_boundary import (
+    IHCState,
+    declare_attn,
+    declare_ffn,
+    get_attn_tp_context,
diff -- python/sglang/srt/models/hunyuan_v4_nextn.py
@@ -5,7 +5,13 @@
-from sglang.srt.layers.layer_boundary import AttentionInputs, get_attn_tp_context
+from sglang.srt.layers.layer_boundary import (
+    declare_attn,
+    declare_ffn,
+    get_attn_tp_context,
+    make_stages,
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/hunyuan_v4.py` modified +101/-20; `python/sglang/srt/models/hunyuan_v4_nextn.py` modified +29/-14
- Risk and verification: The diff ships test coverage in `test/registered/unit/layer_boundary/test_ihc_residual_ops.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
