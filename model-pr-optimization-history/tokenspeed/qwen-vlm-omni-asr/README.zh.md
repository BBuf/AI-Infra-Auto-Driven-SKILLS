# TokenSpeed Qwen VLM/Omni/ASR 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/configs/qwen3_asr_config.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |
| `python/tokenspeed/runtime/configs/qwen3_vision_config.py` | 无直接 PR 号提交 |
| `python/tokenspeed/runtime/models/qwen3_asr.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |
| `python/tokenspeed/runtime/models/qwen3_audio.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |
| `python/tokenspeed/runtime/models/qwen3_omni.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |
| `python/tokenspeed/runtime/models/qwen3_vision.py` | 无直接 PR 号提交 |
| `test/runtime/models/test_qwen3_audio.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |
| `test/runtime/models/test_qwen3_omni.py` | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) |

## PR 覆盖总览

- git 追溯 PR 数: 1
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 1
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-07-12 | [#654](https://github.com/lightseekorg/tokenspeed/pull/654) | merged | feat(multimodal): support ASR and Omni thinker | `python/tokenspeed/runtime/models/qwen3_audio.py`, `python/tokenspeed/runtime/models/qwen3_omni.py`, `test/runtime/models/test_qwen3_omni.py` |

## 逐 PR diff 审计卡

### PR #654 - feat(multimodal): support ASR and Omni thinker

- 链接: https://github.com/lightseekorg/tokenspeed/pull/654
- 状态/时间: merged / 2026-07-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/qwen3_asr_config.py`, `python/tokenspeed/runtime/models/qwen3_asr.py`, `python/tokenspeed/runtime/models/qwen3_audio.py`, `python/tokenspeed/runtime/models/qwen3_omni.py`, `test/runtime/models/test_qwen3_audio.py` 等 6 个文件；关联提交 `5faf5725254f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 13 个文件，+2505/-46，可读 patch 2838 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3_audio.py` added +559/-0 (559 lines); hunks: -0,0 +1,559; symbols: _cnn_output_lengths, qwen3_audio_output_lengths, _one_item_feature_length, pack_qwen3_audio_features，涉及 `_cnn_output_lengths, qwen3_audio_output_lengths, _one_item_feature_length`；`python/tokenspeed/runtime/models/qwen3_omni.py` added +485/-0 (485 lines); hunks: -0,0 +1,485; symbols: _get_thinker_config, _shared_vision_config, Qwen3OmniMoeTextModel, forward，涉及 `_get_thinker_config, _shared_vision_config, Qwen3OmniMoeTextModel`；`test/runtime/models/test_qwen3_omni.py` added +414/-0 (414 lines); hunks: -0,0 +1,414; symbols: _omni_config, TestQwen3OmniConfig, test_architecture_flags_cover_asr_and_omni, test_model_config_unwraps_omni_thinker_text，涉及 `_omni_config, TestQwen3OmniConfig, test_architecture_flags_cover_asr_and_omni`；`test/runtime/models/test_qwen3_audio.py` added +287/-0 (287 lines); hunks: -0,0 +1,287; symbols: _upstream_output_lengths, _audio_item, _tiny_config, _tiny_asr_config，涉及 `_upstream_output_lengths, _audio_item, _tiny_config`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3_audio.py` added +559/-0 (559 lines); hunks: -0,0 +1,559; symbols: _cnn_output_lengths, qwen3_audio_output_lengths, _one_item_feature_length, pack_qwen3_audio_features
  - `python/tokenspeed/runtime/models/qwen3_omni.py` added +485/-0 (485 lines); hunks: -0,0 +1,485; symbols: _get_thinker_config, _shared_vision_config, Qwen3OmniMoeTextModel, forward
  - `test/runtime/models/test_qwen3_omni.py` added +414/-0 (414 lines); hunks: -0,0 +1,414; symbols: _omni_config, TestQwen3OmniConfig, test_architecture_flags_cover_asr_and_omni, test_model_config_unwraps_omni_thinker_text
  - `test/runtime/models/test_qwen3_audio.py` added +287/-0 (287 lines); hunks: -0,0 +1,287; symbols: _upstream_output_lengths, _audio_item, _tiny_config, _tiny_asr_config
  - `python/tokenspeed/runtime/models/qwen3_asr.py` added +274/-0 (274 lines); hunks: -0,0 +1,274; symbols: Qwen3ASRForConditionalGeneration, __init__, get_audio_feature, pad_input_ids
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3_audio.py` added +559/-0; `python/tokenspeed/runtime/models/qwen3_omni.py` added +485/-0; `python/tokenspeed/runtime/models/qwen3_asr.py` added +274/-0; `python/tokenspeed/runtime/configs/qwen3_asr_config.py` added +204/-0
  - tests: `test/runtime/models/test_qwen3_omni.py` added +414/-0; `test/runtime/models/test_qwen3_audio.py` added +287/-0
- 验证与风险: diff 自带测试面 `test/runtime/models/test_qwen3_audio.py`, `test/runtime/models/test_qwen3_omni.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
