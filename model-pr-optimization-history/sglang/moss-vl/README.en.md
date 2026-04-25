# sglang MOSS-VL Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.
- Diffusion model families have been removed from this history set and are no longer part of model optimization skills.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/sglang/srt/models/moss_vl.py` | [#23454](https://github.com/sgl-project/sglang/pull/23454) |
| `python/sglang/srt/multimodal/processors/moss_vl.py` | [#23454](https://github.com/sgl-project/sglang/pull/23454) |

## PR Coverage Summary

- Git-traced PRs: 1
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 1
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#23454](https://github.com/sgl-project/sglang/pull/23454) | merged | [srt] Add Moss-VL Python runtime support | `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py` |

## Per-PR Diff Audit Cards

### PR #23454 - [srt] Add Moss-VL Python runtime support

- Link: https://github.com/sgl-project/sglang/pull/23454
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`; associated commits `59724e90a9b8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +2401/-6, 2611 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MOSS-VL, this PR adds or enables a model support/runtime surface. Title: "[srt] Add Moss-VL Python runtime support". The diff centers on `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`. PR body context: ## Summary This PR adds Python-side runtime support for Moss-VL in SRT. The changes include: - add `MossVLForConditionalGeneration` model support - add a Moss-VL multimodal proc...
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

## Gap-Closure Notes

- This version rejects title-only PR lists; every PR must include trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
