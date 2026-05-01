# sglang Gemma 4 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/function_call/gemma4_detector.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/layers/gemma4_fused_ops.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/models/gemma4_audio.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/models/gemma4_causal.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/models/gemma4_mm.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/models/gemma4_vision.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/multimodal/processors/gemma4.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |

## PR 覆盖总览

- git 追溯 PR 数: 1
- 原文档显式引用补充 PR 数: 3
- 当前文档总 PR 数: 4
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-07 | [#21952](https://github.com/sgl-project/sglang/pull/21952) | merged | [New Model] Gemma 4 | `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py` |
| 2026-04-10 | [#22079](https://github.com/sgl-project/sglang/pull/22079) | merged | [nvidia] Gemma4 nvfp4 fix | `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` |
| 2026-04-16 | [#21569](https://github.com/sgl-project/sglang/pull/21569) | merged | Upgrade transformers to 5.5.3 and refactor hf_transformers_utils into subpackage | `python/sglang/srt/utils/hf_transformers/tokenizer.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/step3p5.py` |
| 2026-04-17 | [#22408](https://github.com/sgl-project/sglang/pull/22408) | merged | [CI] Adding Gemma 4 to Nightly CI | `test/registered/eval/test_vlms_mmmu_eval.py` |

## 逐 PR diff 审计卡

### PR #21952 - [New Model] Gemma 4

- 链接: https://github.com/sgl-project/sglang/pull/21952
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/gemma4_detector.py`, `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_audio.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py` 等 7 个文件；关联提交 `2813cb6d9a5b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+6007/-70，可读 patch 6694 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[New Model] Gemma 4」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py`；技术摘要: 覆盖「[New Model] Gemma 4」；主要实现面是 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gemma4_causal.py` added +1009/-0 (1009 lines); hunks: -0,0 +1,1009; symbols: get_attention_sliding_window_size, Gemma4Router, __init__, fuse_scale，涉及 `get_attention_sliding_window_size, Gemma4Router, __init__`；`python/sglang/srt/models/gemma4_mm.py` added +878/-0 (878 lines); hunks: -0,0 +1,878; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4MultimodalEmbedder, __init__，涉及 `Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4MultimodalEmbedder`；`python/sglang/srt/models/gemma4_audio.py` added +873/-0 (873 lines); hunks: -0,0 +1,873; symbols: Gemma4AudioRelativePositionEmbedding, __init__, _get_timing_signal_1d_pos, _relative_shift，涉及 `Gemma4AudioRelativePositionEmbedding, __init__, _get_timing_signal_1d_pos`；`python/sglang/srt/models/gemma4_vision.py` added +599/-0 (599 lines); hunks: -0,0 +1,599; symbols: _rotate_half, _apply_rotary, Gemma4VisionRotaryEmbedding, __init__，涉及 `_rotate_half, _apply_rotary, Gemma4VisionRotaryEmbedding`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma4_causal.py` added +1009/-0 (1009 lines); hunks: -0,0 +1,1009; symbols: get_attention_sliding_window_size, Gemma4Router, __init__, fuse_scale
  - `python/sglang/srt/models/gemma4_mm.py` added +878/-0 (878 lines); hunks: -0,0 +1,878; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4MultimodalEmbedder, __init__
  - `python/sglang/srt/models/gemma4_audio.py` added +873/-0 (873 lines); hunks: -0,0 +1,873; symbols: Gemma4AudioRelativePositionEmbedding, __init__, _get_timing_signal_1d_pos, _relative_shift
  - `python/sglang/srt/models/gemma4_vision.py` added +599/-0 (599 lines); hunks: -0,0 +1,599; symbols: _rotate_half, _apply_rotary, Gemma4VisionRotaryEmbedding, __init__
  - `python/sglang/srt/function_call/gemma4_detector.py` added +445/-0 (445 lines); hunks: -0,0 +1,445; symbols: _parse_gemma4_value, _parse_gemma4_array, _parse_gemma4_args, _find_matching_brace
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma4_causal.py` added +1009/-0; `python/sglang/srt/models/gemma4_mm.py` added +878/-0; `python/sglang/srt/models/gemma4_audio.py` added +873/-0; `python/sglang/srt/models/gemma4_vision.py` added +599/-0; `python/sglang/srt/function_call/gemma4_detector.py` added +445/-0; `python/sglang/srt/multimodal/processors/gemma4.py` added +145/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/unit/parser/test_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22079 - [nvidia] Gemma4 nvfp4 fix

- 链接: https://github.com/sgl-project/sglang/pull/22079
- 状态/时间: merged / 2026-04-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-0，可读 patch 15 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[nvidia] Gemma4 nvfp4 fix」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`；技术摘要: 覆盖「[nvidia] Gemma4 nvfp4 fix」；主要实现面是 `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +8/-0 (8 lines); hunks: -72,6 +72,14 @@ def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):; symbols: _get_block_sizes_for_extend_attention，涉及 `_get_block_sizes_for_extend_attention`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +8/-0 (8 lines); hunks: -72,6 +72,14 @@ def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):; symbols: _get_block_sizes_for_extend_attention
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21569 - Upgrade transformers to 5.5.3 and refactor hf_transformers_utils into subpackage

- 链接: https://github.com/sgl-project/sglang/pull/21569
- 状态/时间: merged / 2026-04-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+2838/-1515，可读 patch 4528 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Upgrade transformers to 5.5.3 and refactor hf_transformers_utils into subpackage」；模型线: Gemma 4；类别: 模型实现调整；主要 diff: `python/sglang/srt/utils/hf_transformers/tokenizer.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/step3p5.py`；技术摘要: 覆盖「Upgrade transformers to 5.5.3 and refactor hf_transformers_utils into subpackage」；主要实现面是 `python/sglang/srt/utils/hf_transformers/tokenizer.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/step3p5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/utils/hf_transformers/tokenizer.py` added +551/-0 (551 lines); hunks: -0,0 +1,551; symbols: _load_tokenizer_by_declared_class, declared, mapping, like，涉及 `_load_tokenizer_by_declared_class, declared, mapping`；`python/sglang/srt/configs/qwen3_5.py` modified +16/-0 (16 lines); hunks: -8,6 +8,9 @@ class Qwen3_5VisionConfig(Qwen3VLVisionConfig):; -109,14 +112,27 @@ def __init__(; symbols: Qwen3_5VisionConfig, __init__, Qwen3_5TextConfig, Qwen3_5MoeVisionConfig，涉及 `Qwen3_5VisionConfig, __init__, Qwen3_5TextConfig`；`python/sglang/srt/configs/step3p5.py` modified +9/-0 (9 lines); hunks: -94,4 +94,13 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen3_vl.py` modified +7/-1 (8 lines); hunks: -1091,9 +1091,15 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/utils/hf_transformers/tokenizer.py` added +551/-0 (551 lines); hunks: -0,0 +1,551; symbols: _load_tokenizer_by_declared_class, declared, mapping, like
  - `python/sglang/srt/configs/qwen3_5.py` modified +16/-0 (16 lines); hunks: -8,6 +8,9 @@ class Qwen3_5VisionConfig(Qwen3VLVisionConfig):; -109,14 +112,27 @@ def __init__(; symbols: Qwen3_5VisionConfig, __init__, Qwen3_5TextConfig, Qwen3_5MoeVisionConfig
  - `python/sglang/srt/configs/step3p5.py` modified +9/-0 (9 lines); hunks: -94,4 +94,13 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +7/-1 (8 lines); hunks: -1091,9 +1091,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +3/-1480 (1483 lines); hunks: -11,1484 +11,7; symbols: download_from_hf, get_rope_config, _patch_text_config, get_hf_text_config
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/utils/hf_transformers/tokenizer.py` added +551/-0; `python/sglang/srt/configs/qwen3_5.py` modified +16/-0; `python/sglang/srt/configs/step3p5.py` modified +9/-0; `python/sglang/srt/models/qwen3_vl.py` modified +7/-1; `python/sglang/srt/utils/hf_transformers_utils.py` modified +3/-1480; `python/sglang/srt/utils/hf_transformers/compat.py` added +458/-0
  - tests: `test/registered/unit/utils/test_hf_transformers.py` added +586/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/utils/test_hf_transformers.py`, `test/registered/vlm/test_vlm_input_format.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22408 - [CI] Adding Gemma 4 to Nightly CI

- 链接: https://github.com/sgl-project/sglang/pull/22408
- 状态/时间: merged / 2026-04-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-3，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Adding Gemma 4 to Nightly CI」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `test/registered/eval/test_vlms_mmmu_eval.py`；技术摘要: 覆盖「[CI] Adding Gemma 4 to Nightly CI」；主要实现面是 `test/registered/eval/test_vlms_mmmu_eval.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/eval/test_vlms_mmmu_eval.py` modified +6/-3 (9 lines); hunks: -33,10 +33,13。
- 代码 diff 细节:
  - `test/registered/eval/test_vlms_mmmu_eval.py` modified +6/-3 (9 lines); hunks: -33,10 +33,13
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/eval/test_vlms_mmmu_eval.py` modified +6/-3
- 验证与风险: diff 自带测试面 `test/registered/eval/test_vlms_mmmu_eval.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
