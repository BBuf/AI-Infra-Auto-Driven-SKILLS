# sglang Gemma 4 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

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
- 原文档显式引用补充 PR 数: 2
- 当前文档总 PR 数: 3
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-07 | [#21952](https://github.com/sgl-project/sglang/pull/21952) | merged | [New Model] Gemma 4 | `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py` |
| 2026-04-10 | [#22079](https://github.com/sgl-project/sglang/pull/22079) | merged | [nvidia] Gemma4 nvfp4 fix | `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` |
| 2026-04-17 | [#22408](https://github.com/sgl-project/sglang/pull/22408) | merged | [CI] Adding Gemma 4 to Nightly CI | `test/registered/eval/test_vlms_mmmu_eval.py` |

## 逐 PR diff 审计卡

### PR #21952 - [New Model] Gemma 4

- 链接: https://github.com/sgl-project/sglang/pull/21952
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/gemma4_detector.py`, `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_audio.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py` 等 7 个文件；关联提交 `2813cb6d9a5b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+6007/-70，可读 patch 6694 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Gemma 4 补齐模型支持入口或运行时能力，标题为「[New Model] Gemma 4」，变更集中在 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py`。PR 描述补充为：## Motivation Add Gemma 4 model support to SGLang. Gemma 4 is Google's next-generation family of open models featuring Dense and MoE architectures, multimodal support (text, ima...
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
- 动机: 该 PR 围绕 Gemma 4 修复已暴露的启动、加载、解析或数值问题，标题为「[nvidia] Gemma4 nvfp4 fix」，变更集中在 `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`。PR 描述补充为：Based on #21952 and depends on https://github.com/flashinfer-ai/flashinfer/pull/2959 ## Motivation Gemma 4 NVFP4 checkpoints does not work on GB200 for the following reasons: Tr...
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

### PR #22408 - [CI] Adding Gemma 4 to Nightly CI

- 链接: https://github.com/sgl-project/sglang/pull/22408
- 状态/时间: merged / 2026-04-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-3，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Gemma 4 补强部署文档、测试或 CI 验证面，标题为「[CI] Adding Gemma 4 to Nightly CI」，变更集中在 `test/registered/eval/test_vlms_mmmu_eval.py`。PR 描述补充为：## Motivation Adding Gemma 4 variants to Nightly CI following https://github.com/sgl-project/sglang/pull/21952 Pending https://github.com/sgl-project/sglang/pull/21569 upgrade t...
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

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
