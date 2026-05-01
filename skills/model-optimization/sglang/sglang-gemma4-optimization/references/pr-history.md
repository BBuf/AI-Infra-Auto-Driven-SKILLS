# sglang Gemma 4 PR Diff Audit Reference

- Rebuilt on: 2026-05-01
- Source baseline: `sgl-project/sglang` trace worktree commit `4197c55968`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 3
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/function_call/gemma4_detector.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/layers/gemma4_fused_ops.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/models/gemma4_audio.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/models/gemma4_causal.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/models/gemma4_mm.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/models/gemma4_vision.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/multimodal/processors/gemma4.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-07 | [#21952](https://github.com/sgl-project/sglang/pull/21952) | merged | [New Model] Gemma 4 | `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py` |
| 2026-04-10 | [#22079](https://github.com/sgl-project/sglang/pull/22079) | merged | [nvidia] Gemma4 nvfp4 fix | `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` |
| 2026-04-16 | [#21569](https://github.com/sgl-project/sglang/pull/21569) | merged | Upgrade transformers to 5.5.3 and refactor hf_transformers_utils into subpackage | `python/sglang/srt/utils/hf_transformers/tokenizer.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/step3p5.py` |
| 2026-04-17 | [#22408](https://github.com/sgl-project/sglang/pull/22408) | merged | [CI] Adding Gemma 4 to Nightly CI | `test/registered/eval/test_vlms_mmmu_eval.py` |

## Per-PR Diff Audit Cards

### PR #21952 - [New Model] Gemma 4

- Link: https://github.com/sgl-project/sglang/pull/21952
- Status/date: merged / 2026-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/gemma4_detector.py`, `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_audio.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py` and 7 files; associated commits `2813cb6d9a5b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +6007/-70, 6694 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[New Model] Gemma 4"; model line: Gemma 4; category: model support/runtime entry; main diff: `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py`; technical summary: Covers "[New Model] Gemma 4"; the main implementation surface is `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gemma4_causal.py` added +1009/-0 (1009 lines); hunks: -0,0 +1,1009; symbols: get_attention_sliding_window_size, Gemma4Router, __init__, fuse_scale, touching `get_attention_sliding_window_size, Gemma4Router, __init__`; `python/sglang/srt/models/gemma4_mm.py` added +878/-0 (878 lines); hunks: -0,0 +1,878; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4MultimodalEmbedder, __init__, touching `Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4MultimodalEmbedder`; `python/sglang/srt/models/gemma4_audio.py` added +873/-0 (873 lines); hunks: -0,0 +1,873; symbols: Gemma4AudioRelativePositionEmbedding, __init__, _get_timing_signal_1d_pos, _relative_shift, touching `Gemma4AudioRelativePositionEmbedding, __init__, _get_timing_signal_1d_pos`; `python/sglang/srt/models/gemma4_vision.py` added +599/-0 (599 lines); hunks: -0,0 +1,599; symbols: _rotate_half, _apply_rotary, Gemma4VisionRotaryEmbedding, __init__, touching `_rotate_half, _apply_rotary, Gemma4VisionRotaryEmbedding`.
- Code diff details:
  - `python/sglang/srt/models/gemma4_causal.py` added +1009/-0 (1009 lines); hunks: -0,0 +1,1009; symbols: get_attention_sliding_window_size, Gemma4Router, __init__, fuse_scale
  - `python/sglang/srt/models/gemma4_mm.py` added +878/-0 (878 lines); hunks: -0,0 +1,878; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4MultimodalEmbedder, __init__
  - `python/sglang/srt/models/gemma4_audio.py` added +873/-0 (873 lines); hunks: -0,0 +1,873; symbols: Gemma4AudioRelativePositionEmbedding, __init__, _get_timing_signal_1d_pos, _relative_shift
  - `python/sglang/srt/models/gemma4_vision.py` added +599/-0 (599 lines); hunks: -0,0 +1,599; symbols: _rotate_half, _apply_rotary, Gemma4VisionRotaryEmbedding, __init__
  - `python/sglang/srt/function_call/gemma4_detector.py` added +445/-0 (445 lines); hunks: -0,0 +1,445; symbols: _parse_gemma4_value, _parse_gemma4_array, _parse_gemma4_args, _find_matching_brace
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gemma4_causal.py
@@ -0,0 +1,1009 @@
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/gemma4_mm.py
@@ -0,0 +1,878 @@
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/gemma4_audio.py
@@ -0,0 +1,873 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gemma4_causal.py` added +1009/-0; `python/sglang/srt/models/gemma4_mm.py` added +878/-0; `python/sglang/srt/models/gemma4_audio.py` added +873/-0; `python/sglang/srt/models/gemma4_vision.py` added +599/-0; `python/sglang/srt/function_call/gemma4_detector.py` added +445/-0; `python/sglang/srt/multimodal/processors/gemma4.py` added +145/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/unit/parser/test_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22079 - [nvidia] Gemma4 nvfp4 fix

- Link: https://github.com/sgl-project/sglang/pull/22079
- Status/date: merged / 2026-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-0, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[nvidia] Gemma4 nvfp4 fix"; model line: Gemma 4; category: bug fix; main diff: `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`; technical summary: Covers "[nvidia] Gemma4 nvfp4 fix"; the main implementation surface is `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +8/-0 (8 lines); hunks: -72,6 +72,14 @@ def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):; symbols: _get_block_sizes_for_extend_attention, touching `_get_block_sizes_for_extend_attention`.
- Code diff details:
  - `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +8/-0 (8 lines); hunks: -72,6 +72,14 @@ def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):; symbols: _get_block_sizes_for_extend_attention
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/triton_ops/extend_attention.py
@@ -72,6 +72,14 @@ def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):
+        elif _is_cuda and CUDA_CAPABILITY[0] == 10:
+            # Blackwell data-center architecture (GB200, B200, sm_100a)
+            # sm_100a has different register constraints from Hopper; Hopper block sizes
+            # cause PTX register exhaustion (>255 regs) for large head dims (Lq=512).
+            if Lq <= 256:
+                BLOCK_M, BLOCK_N = (64, 64)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21569 - Upgrade transformers to 5.5.3 and refactor hf_transformers_utils into subpackage

- Link: https://github.com/sgl-project/sglang/pull/21569
- Status/date: merged / 2026-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +2838/-1515, 4528 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Upgrade transformers to 5.5.3 and refactor hf_transformers_utils into subpackage"; model line: Gemma 4; category: model implementation change; main diff: `python/sglang/srt/utils/hf_transformers/tokenizer.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/step3p5.py`; technical summary: Covers "Upgrade transformers to 5.5.3 and refactor hf_transformers_utils into subpackage"; the main implementation surface is `python/sglang/srt/utils/hf_transformers/tokenizer.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/step3p5.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/utils/hf_transformers/tokenizer.py` added +551/-0 (551 lines); hunks: -0,0 +1,551; symbols: _load_tokenizer_by_declared_class, declared, mapping, like, touching `_load_tokenizer_by_declared_class, declared, mapping`; `python/sglang/srt/configs/qwen3_5.py` modified +16/-0 (16 lines); hunks: -8,6 +8,9 @@ class Qwen3_5VisionConfig(Qwen3VLVisionConfig):; -109,14 +112,27 @@ def __init__(; symbols: Qwen3_5VisionConfig, __init__, Qwen3_5TextConfig, Qwen3_5MoeVisionConfig, touching `Qwen3_5VisionConfig, __init__, Qwen3_5TextConfig`; `python/sglang/srt/configs/step3p5.py` modified +9/-0 (9 lines); hunks: -94,4 +94,13 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/qwen3_vl.py` modified +7/-1 (8 lines); hunks: -1091,9 +1091,15 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/utils/hf_transformers/tokenizer.py` added +551/-0 (551 lines); hunks: -0,0 +1,551; symbols: _load_tokenizer_by_declared_class, declared, mapping, like
  - `python/sglang/srt/configs/qwen3_5.py` modified +16/-0 (16 lines); hunks: -8,6 +8,9 @@ class Qwen3_5VisionConfig(Qwen3VLVisionConfig):; -109,14 +112,27 @@ def __init__(; symbols: Qwen3_5VisionConfig, __init__, Qwen3_5TextConfig, Qwen3_5MoeVisionConfig
  - `python/sglang/srt/configs/step3p5.py` modified +9/-0 (9 lines); hunks: -94,4 +94,13 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +7/-1 (8 lines); hunks: -1091,9 +1091,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +3/-1480 (1483 lines); hunks: -11,1484 +11,7; symbols: download_from_hf, get_rope_config, _patch_text_config, get_hf_text_config
- Key code excerpts:

```diff
diff -- python/sglang/srt/utils/hf_transformers/tokenizer.py
@@ -0,0 +1,551 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/qwen3_5.py
@@ -8,6 +8,9 @@ class Qwen3_5VisionConfig(Qwen3VLVisionConfig):
+    def __init__(self, **kwargs):
+        super().__init__(**kwargs)
@@ -109,14 +112,27 @@ def __init__(
+    def __init__(self, **kwargs):
+        super().__init__(**kwargs)
+    def __init__(self, **kwargs):
diff -- python/sglang/srt/configs/step3p5.py
@@ -94,4 +94,13 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/utils/hf_transformers/tokenizer.py` added +551/-0; `python/sglang/srt/configs/qwen3_5.py` modified +16/-0; `python/sglang/srt/configs/step3p5.py` modified +9/-0; `python/sglang/srt/models/qwen3_vl.py` modified +7/-1; `python/sglang/srt/utils/hf_transformers_utils.py` modified +3/-1480; `python/sglang/srt/utils/hf_transformers/compat.py` added +458/-0
  - tests: `test/registered/unit/utils/test_hf_transformers.py` added +586/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/utils/test_hf_transformers.py`, `test/registered/vlm/test_vlm_input_format.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22408 - [CI] Adding Gemma 4 to Nightly CI

- Link: https://github.com/sgl-project/sglang/pull/22408
- Status/date: merged / 2026-04-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-3, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Adding Gemma 4 to Nightly CI"; model line: Gemma 4; category: docs/tests/CI; main diff: `test/registered/eval/test_vlms_mmmu_eval.py`; technical summary: Covers "[CI] Adding Gemma 4 to Nightly CI"; the main implementation surface is `test/registered/eval/test_vlms_mmmu_eval.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/eval/test_vlms_mmmu_eval.py` modified +6/-3 (9 lines); hunks: -33,10 +33,13.
- Code diff details:
  - `test/registered/eval/test_vlms_mmmu_eval.py` modified +6/-3 (9 lines); hunks: -33,10 +33,13
- Key code excerpts:

```diff
diff -- test/registered/eval/test_vlms_mmmu_eval.py
@@ -33,10 +33,13 @@
-    ModelLaunchSettings("google/gemma-3-4b-it"): ModelEvalMetrics(0.360, 10.9),
+    ModelLaunchSettings("google/gemma-4-E4B-it"): ModelEvalMetrics(0.26, 15.0),
-        "google/gemma-3n-E4B-it", extra_args=["--tp=2"]
-    ): ModelEvalMetrics(0.270, 17.7),
+        "google/gemma-4-26B-A4B-it", extra_args=["--tp=2"]
+    ): ModelEvalMetrics(0.27, 22.3),
```

- Reviewed files:
  - tests: `test/registered/eval/test_vlms_mmmu_eval.py` modified +6/-3
- Risk and verification: The diff ships test coverage in `test/registered/eval/test_vlms_mmmu_eval.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.
