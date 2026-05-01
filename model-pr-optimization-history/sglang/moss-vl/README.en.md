# sglang MOSS-VL Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/sglang/srt/models/moss_vl.py` | [#23454](https://github.com/sgl-project/sglang/pull/23454), [#23932](https://github.com/sgl-project/sglang/pull/23932) |
| `python/sglang/srt/multimodal/processors/moss_vl.py` | [#23454](https://github.com/sgl-project/sglang/pull/23454) |

## PR Coverage Summary

- Git-traced PRs: 2
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 2
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#23454](https://github.com/sgl-project/sglang/pull/23454) | merged | [srt] Add Moss-VL Python runtime support | `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py` |
| 2026-04-30 | [#23932](https://github.com/sgl-project/sglang/pull/23932) | merged | [moss-vl] use Conv3dLayer and remove no-op flat_encoder_result | `python/sglang/srt/models/moss_vl.py` |

## Per-PR Diff Audit Cards

### PR #23454 - [srt] Add Moss-VL Python runtime support

- Link: https://github.com/sgl-project/sglang/pull/23454
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`; associated commits `59724e90a9b8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +2401/-6, 2611 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[srt] Add Moss-VL Python runtime support"; model line: MOSS-VL; category: model support/runtime entry; main diff: `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`; technical summary: Covers "[srt] Add Moss-VL Python runtime support"; the main implementation surface is `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/moss_vl.py` added +1643/-0 (1643 lines); hunks: -0,0 +1,1643; symbols: MossVLVisionMLP, __init__, forward, MossVLVisionPatchEmbed, touching `MossVLVisionMLP, __init__, forward`; `python/sglang/srt/multimodal/processors/moss_vl.py` added +612/-0 (612 lines); hunks: -0,0 +1,612; symbols: MossVLImageProcessor, __init__, _build_mm_items, _build_vision_token_info, touching `MossVLImageProcessor, __init__, _build_mm_items`.
- Code diff details:
  - `python/sglang/srt/models/moss_vl.py` added +1643/-0 (1643 lines); hunks: -0,0 +1,1643; symbols: MossVLVisionMLP, __init__, forward, MossVLVisionPatchEmbed
  - `python/sglang/srt/multimodal/processors/moss_vl.py` added +612/-0 (612 lines); hunks: -0,0 +1,612; symbols: MossVLImageProcessor, __init__, _build_mm_items, _build_vision_token_info
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/moss_vl.py
@@ -0,0 +1,1643 @@
+"""PyTorch Moss-VL model for SGLang - Qwen3VL Vision + Text with Cross Attention."""
+from __future__ import annotations
+import logging
+from functools import partial
+from typing import Iterable, List, Optional, Tuple
+import torch
diff -- python/sglang/srt/multimodal/processors/moss_vl.py
@@ -0,0 +1,612 @@
+import asyncio
+import os
+import re
+import tempfile
+from typing import Dict, List, Optional, Tuple, Union
+from urllib.parse import unquote, urlparse
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/moss_vl.py` added +1643/-0; `python/sglang/srt/multimodal/processors/moss_vl.py` added +612/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/managers/schedule_batch.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23932 - [moss-vl] use Conv3dLayer and remove no-op flat_encoder_result

- Link: https://github.com/sgl-project/sglang/pull/23932
- Status/date: merged / 2026-04-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/moss_vl.py`; associated commits `4f0b44c5c666`
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-60, 146 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[moss-vl] use Conv3dLayer and remove no-op flat_encoder_result"; model line: MOSS-VL; category: model implementation change; main diff: `python/sglang/srt/models/moss_vl.py`; technical summary: Covers "[moss-vl] use Conv3dLayer and remove no-op flat_encoder_result"; the main implementation surface is `python/sglang/srt/models/moss_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/moss_vl.py` modified +12/-60 (72 lines); hunks: -20,6 +20,7; -96,7 +97,7 @@ def __init__(self, config) -> None:; symbols: __init__, pad_input_ids, _collect_mm_data, _get_vision_features, touching `__init__, pad_input_ids, _collect_mm_data`.
- Code diff details:
  - `python/sglang/srt/models/moss_vl.py` modified +12/-60 (72 lines); hunks: -20,6 +20,7; -96,7 +97,7 @@ def __init__(self, config) -> None:; symbols: __init__, pad_input_ids, _collect_mm_data, _get_vision_features
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/moss_vl.py
@@ -20,6 +20,7 @@
+from sglang.srt.layers.conv import Conv3dLayer
@@ -96,7 +97,7 @@ def __init__(self, config) -> None:
-        self.proj = nn.Conv3d(
+        self.proj = Conv3dLayer(
@@ -1142,11 +1143,10 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
-            return None, None, None, None
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/moss_vl.py` modified +12/-60
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/moss_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
