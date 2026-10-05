# SGLang dots3 (dots.note) Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx` | [#33829](https://github.com/sgl-project/sglang/pull/33829), [#34797](https://github.com/sgl-project/sglang/pull/34797) |
| `docs/src/snippets/configs/rednote/dots3-note.jsx` | [#33829](https://github.com/sgl-project/sglang/pull/33829), [#34797](https://github.com/sgl-project/sglang/pull/34797) |
| `python/sglang/srt/configs/dots3.py` | [#33829](https://github.com/sgl-project/sglang/pull/33829) |
| `python/sglang/srt/models/dots3.py` | [#33829](https://github.com/sgl-project/sglang/pull/33829) |
| `python/sglang/srt/models/dots3_common/__init__.py` | [#33829](https://github.com/sgl-project/sglang/pull/33829) |
| `python/sglang/srt/models/dots3_common/dots_omni_audio.py` | [#33829](https://github.com/sgl-project/sglang/pull/33829) |
| `python/sglang/srt/models/dots3_common/dots_omni_towers.py` | [#33829](https://github.com/sgl-project/sglang/pull/33829) |
| `python/sglang/srt/models/dots3_common/dots_omni_vision.py` | [#33829](https://github.com/sgl-project/sglang/pull/33829) |
| `python/sglang/srt/models/dots3_common/fp8.py` | [#33829](https://github.com/sgl-project/sglang/pull/33829) |
| `python/sglang/srt/models/dots3_common/modeling.py` | [#33829](https://github.com/sgl-project/sglang/pull/33829), [#41198](https://github.com/sgl-project/sglang/pull/41198) |
| `python/sglang/srt/models/dots3_common/nextn.py` | [#33829](https://github.com/sgl-project/sglang/pull/33829) |
| `python/sglang/srt/models/dots3_nextn.py` | [#33829](https://github.com/sgl-project/sglang/pull/33829) |

## PR Coverage Summary

- Git-traced PRs: 3
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 3
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-08-14 | [#34797](https://github.com/sgl-project/sglang/pull/34797) | merged | docs: link dots3.note checkpoints, add H100 cells | `docs/src/snippets/configs/rednote/dots3-note.jsx`, `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx` |
| 2026-08-22 | [#33829](https://github.com/sgl-project/sglang/pull/33829) | merged | [Model] Complete dots.note.omni support with native encoders, video preprocessing, and MTP decoding | `python/sglang/srt/models/dots3_common/modeling.py`, `python/sglang/srt/models/dots3_common/dots_omni_audio.py`, `python/sglang/srt/models/dots3_common/dots_omni_vision.py` |
| 2026-09-25 | [#41198](https://github.com/sgl-project/sglang/pull/41198) | merged | [Refactor] Move Step-3.5, GLM5-Next, Dots3, MiniMax-M3 and Qwen3.5 onto ffn_exit | `python/sglang/srt/models/dots3_common/modeling.py` |

## Per-PR Diff Audit Cards

### PR #34797 - docs: link dots3.note checkpoints, add H100 cells

- Link: https://github.com/sgl-project/sglang/pull/34797
- Status/date: merged / 2026-08-14
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx`, `docs/src/snippets/configs/rednote/dots3-note.jsx`; associated commits `6ad3f2d8fdc8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +128/-21, 229 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/rednote/dots3-note.jsx` modified +107/-4 (111 lines); hunks: -9,7 +9,7 @@ export const config = {; -24,8 +24,8 @@ export const config = {; `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx` modified +2/-12 (14 lines); hunks: -46,16 +46,12 @@ For how to launch the image, see [Install → Method 3: Using...; -70,13 +66,7 @@ dots3.note is RedNote's native multimodal omni model, built o....
- Code diff details:
  - `docs/src/snippets/configs/rednote/dots3-note.jsx` modified +107/-4 (111 lines); hunks: -9,7 +9,7 @@ export const config = {; -24,8 +24,8 @@ export const config = {
  - `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx` modified +2/-12 (14 lines); hunks: -46,16 +46,12 @@ For how to launch the image, see [Install → Method 3: Using...; -70,13 +66,7 @@ dots3.note is RedNote's native multimodal omni model, built o...
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/rednote/dots3-note.jsx
@@ -9,7 +9,7 @@ export const config = {
-  supportedHardware: ["h200"],
+  supportedHardware: ["h200", "h100"],
@@ -24,8 +24,8 @@ export const config = {
-    // TODO: replace with the public repo id once the checkpoint is released.
-    default: "<dots-note-checkpoint>",
+    bf16: "dots-studio/dots3-note-prev",
diff -- docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx
@@ -46,16 +46,12 @@ For how to launch the image, see [Install → Method 3: Using Docker](../../../d
-Pick the checkpoint precision — the only deployment choice. The recipe runs on a single 8-GPU H200 node with DP8 attention × TP8 × EP8 and DeepEP as the MoE all-to-all transport.
+Pick the hardware and the checkpoint precision. The recipe runs on a single 8-GPU Hopper node with DP8 attention × TP8 × EP8 and DeepEP as the MoE all-to-all transport. Blackwell
-<Note>
-Every cell in the Deploy panel above is currently **unverified**: the recipe runs, but no serving round on public weights has landed (the checkpoint is not yet released). Treat th
-</Note>
@@ -70,13 +66,7 @@ dots3.note is RedNote's native multimodal omni model, built on the dots3 languag
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/rednote/dots3-note.jsx` modified +107/-4; `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx` modified +2/-12
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx`, `docs/src/snippets/_deployment.jsx`, `docs/src/snippets/_playground.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #33829 - [Model] Complete dots.note.omni support with native encoders, video preprocessing, and MTP decoding

- Link: https://github.com/sgl-project/sglang/pull/33829
- Status/date: merged / 2026-08-22
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx`, `docs/src/snippets/configs/rednote/dots3-note.jsx`, `python/sglang/srt/configs/dots3.py`, `python/sglang/srt/models/dots3.py`, `python/sglang/srt/models/dots3_common/__init__.py` and 12 files; associated commits `af39ad93493c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 55 files, +9639/-155, 7772 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/dots3_common/modeling.py` added +2824/-0 (2824 lines); `python/sglang/srt/models/dots3_common/dots_omni_audio.py` added +1027/-0 (1027 lines); hunks: -0,0 +1,1027; symbols: DotsWhisperConfig, __init__, RMSNorm, forward, touching `DotsWhisperConfig, __init__, RMSNorm`; `python/sglang/srt/models/dots3_common/dots_omni_vision.py` added +769/-0 (769 lines); hunks: -0,0 +1,769; symbols: VisionRotaryEmbedding, __init__, _compute_freqs, forward, touching `VisionRotaryEmbedding, __init__, _compute_freqs`; `python/sglang/srt/configs/dots3.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: DotsNoteOmniTokenizerProxy, from_pretrained, Dots3Config, __init__, touching `DotsNoteOmniTokenizerProxy, from_pretrained, Dots3Config`.
- Code diff details:
  - `python/sglang/srt/models/dots3_common/modeling.py` added +2824/-0 (2824 lines)
  - `python/sglang/srt/models/dots3_common/dots_omni_audio.py` added +1027/-0 (1027 lines); hunks: -0,0 +1,1027; symbols: DotsWhisperConfig, __init__, RMSNorm, forward
  - `python/sglang/srt/models/dots3_common/dots_omni_vision.py` added +769/-0 (769 lines); hunks: -0,0 +1,769; symbols: VisionRotaryEmbedding, __init__, _compute_freqs, forward
  - `python/sglang/srt/configs/dots3.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: DotsNoteOmniTokenizerProxy, from_pretrained, Dots3Config, __init__
  - `python/sglang/srt/models/dots3_common/dots_omni_towers.py` added +240/-0 (240 lines); hunks: -0,0 +1,240; symbols: _read_json, load_omni_component_config, DotsNoteOmniVisionEncoder, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/dots3_common/dots_omni_audio.py
@@ -0,0 +1,1027 @@
+"""Dots-path speech encoder for inference only (single GPU).
+Ported from cybertron_alm ``dots_audio_encoder/modeling_whisper.py``.
+Upstream ``WhisperEncoder`` is exposed as :class:`DotsSpeechEncoder`.
+"""
+import math
+from functools import lru_cache
diff -- python/sglang/srt/models/dots3_common/dots_omni_vision.py
@@ -0,0 +1,769 @@
+import math
+from typing import Any
+import torch
+import torch.nn.functional as F
+from torch import nn
+from torch.nn import LayerNorm
diff -- python/sglang/srt/configs/dots3.py
@@ -0,0 +1,243 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/dots3_common/modeling.py` added +2824/-0; `python/sglang/srt/models/dots3_common/dots_omni_audio.py` added +1027/-0; `python/sglang/srt/models/dots3_common/dots_omni_vision.py` added +769/-0; `python/sglang/srt/configs/dots3.py` added +243/-0; `python/sglang/srt/models/dots3_common/dots_omni_towers.py` added +240/-0; `python/sglang/srt/models/dots3_common/nextn.py` added +204/-0
  - docs: `docs/src/snippets/configs/rednote/dots3-note.jsx` modified +112/-60
- Risk and verification: The diff ships test coverage in `test/registered/attention/unittests/swa/test_swa_out_cache_loc.py`, `test/registered/unit/function_call/test_dots_detector.py`, `test/registered/unit/layers/attention/test_dots_hybrid_backend.py`, `test/registered/unit/model_executor/test_mlp_sync_pad_unpad.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41198 - [Refactor] Move Step-3.5, GLM5-Next, Dots3, MiniMax-M3 and Qwen3.5 onto ffn_exit

- Link: https://github.com/sgl-project/sglang/pull/41198
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/dots3_common/modeling.py`; associated commits `b7f6d04a9af1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +50/-219, 402 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/dots3_common/modeling.py` modified +7/-24 (31 lines); hunks: -60,7 +60,6; -1623,30 +1622,14 @@ def forward(; symbols: forward, op_comm_prepare_attn, touching `forward, op_comm_prepare_attn`.
- Code diff details:
  - `python/sglang/srt/models/dots3_common/modeling.py` modified +7/-24 (31 lines); hunks: -60,7 +60,6; -1623,30 +1622,14 @@ def forward(; symbols: forward, op_comm_prepare_attn
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/dots3_common/modeling.py
@@ -60,7 +60,6 @@
-    UnreducedOutput,
@@ -1623,30 +1622,14 @@ def forward(
-        should_allreduce_fusion = (
-            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
-                forward_batch
-            )
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/dots3_common/modeling.py` modified +7/-24
- Risk and verification: The diff ships test coverage in `test/registered/unit/models/test_step3p5_dense_reduce_scatter.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
