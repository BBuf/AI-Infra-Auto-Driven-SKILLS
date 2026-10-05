# TokenSpeed Qwen VLM/Omni/ASR Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/tokenspeed/runtime/configs/qwen3_asr_config.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |
| `python/tokenspeed/runtime/configs/qwen3_vision_config.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/models/qwen3_asr.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |
| `python/tokenspeed/runtime/models/qwen3_audio.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |
| `python/tokenspeed/runtime/models/qwen3_omni.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |
| `python/tokenspeed/runtime/models/qwen3_vision.py` | no direct PR-number commit |
| `test/runtime/models/test_qwen3_audio.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |
| `test/runtime/models/test_qwen3_omni.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |

## PR Coverage Summary

- Git-traced PRs: 1
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 1
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-07-12 | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) | merged | feat(multimodal): support ASR and Omni thinker | `python/tokenspeed/runtime/models/qwen3_audio.py`, `python/tokenspeed/runtime/models/qwen3_omni.py`, `test/runtime/models/test_qwen3_omni.py` |

## Per-PR Diff Audit Cards

### PR #654 - feat(multimodal): support ASR and Omni thinker

- Link: https://github.com/lightseekorg/tokenspeed/pull/654
- Status/date: merged / 2026-07-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/configs/qwen3_asr_config.py`, `python/tokenspeed/runtime/models/qwen3_asr.py`, `python/tokenspeed/runtime/models/qwen3_audio.py`, `python/tokenspeed/runtime/models/qwen3_omni.py`, `test/runtime/models/test_qwen3_audio.py` and 6 files; associated commits `5faf5725254f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 13 files, +2505/-46, 2838 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3_audio.py` added +559/-0 (559 lines); hunks: -0,0 +1,559; symbols: _cnn_output_lengths, qwen3_audio_output_lengths, _one_item_feature_length, pack_qwen3_audio_features, touching `_cnn_output_lengths, qwen3_audio_output_lengths, _one_item_feature_length`; `python/tokenspeed/runtime/models/qwen3_omni.py` added +485/-0 (485 lines); hunks: -0,0 +1,485; symbols: _get_thinker_config, _shared_vision_config, Qwen3OmniMoeTextModel, forward, touching `_get_thinker_config, _shared_vision_config, Qwen3OmniMoeTextModel`; `test/runtime/models/test_qwen3_omni.py` added +414/-0 (414 lines); hunks: -0,0 +1,414; symbols: _omni_config, TestQwen3OmniConfig, test_architecture_flags_cover_asr_and_omni, test_model_config_unwraps_omni_thinker_text, touching `_omni_config, TestQwen3OmniConfig, test_architecture_flags_cover_asr_and_omni`; `test/runtime/models/test_qwen3_audio.py` added +287/-0 (287 lines); hunks: -0,0 +1,287; symbols: _upstream_output_lengths, _audio_item, _tiny_config, _tiny_asr_config, touching `_upstream_output_lengths, _audio_item, _tiny_config`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3_audio.py` added +559/-0 (559 lines); hunks: -0,0 +1,559; symbols: _cnn_output_lengths, qwen3_audio_output_lengths, _one_item_feature_length, pack_qwen3_audio_features
  - `python/tokenspeed/runtime/models/qwen3_omni.py` added +485/-0 (485 lines); hunks: -0,0 +1,485; symbols: _get_thinker_config, _shared_vision_config, Qwen3OmniMoeTextModel, forward
  - `test/runtime/models/test_qwen3_omni.py` added +414/-0 (414 lines); hunks: -0,0 +1,414; symbols: _omni_config, TestQwen3OmniConfig, test_architecture_flags_cover_asr_and_omni, test_model_config_unwraps_omni_thinker_text
  - `test/runtime/models/test_qwen3_audio.py` added +287/-0 (287 lines); hunks: -0,0 +1,287; symbols: _upstream_output_lengths, _audio_item, _tiny_config, _tiny_asr_config
  - `python/tokenspeed/runtime/models/qwen3_asr.py` added +274/-0 (274 lines); hunks: -0,0 +1,274; symbols: Qwen3ASRForConditionalGeneration, __init__, get_audio_feature, pad_input_ids
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_audio.py
@@ -0,0 +1,559 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/models/qwen3_omni.py
@@ -0,0 +1,485 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- test/runtime/models/test_qwen3_omni.py
@@ -0,0 +1,414 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3_audio.py` added +559/-0; `python/tokenspeed/runtime/models/qwen3_omni.py` added +485/-0; `python/tokenspeed/runtime/models/qwen3_asr.py` added +274/-0; `python/tokenspeed/runtime/configs/qwen3_asr_config.py` added +204/-0
  - tests: `test/runtime/models/test_qwen3_omni.py` added +414/-0; `test/runtime/models/test_qwen3_audio.py` added +287/-0
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_qwen3_audio.py`, `test/runtime/models/test_qwen3_omni.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
