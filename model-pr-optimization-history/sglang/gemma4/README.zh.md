# sglang Gemma 4 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` | [#24433](https://github.com/sgl-project/sglang/pull/24433), [#27287](https://github.com/sgl-project/sglang/pull/27287), [#27321](https://github.com/sgl-project/sglang/pull/27321), [#29252](https://github.com/sgl-project/sglang/pull/29252), [#29266](https://github.com/sgl-project/sglang/pull/29266) |
| `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` | [#24433](https://github.com/sgl-project/sglang/pull/24433), [#29252](https://github.com/sgl-project/sglang/pull/29252) |
| `python/sglang/srt/function_call/gemma4_detector.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/layers/gemma4_fused_ops.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952), [#23280](https://github.com/sgl-project/sglang/pull/23280), [#24048](https://github.com/sgl-project/sglang/pull/24048), [#24696](https://github.com/sgl-project/sglang/pull/24696), [#26502](https://github.com/sgl-project/sglang/pull/26502), [#26588](https://github.com/sgl-project/sglang/pull/26588) |
| `python/sglang/srt/models/gemma4_audio.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/models/gemma4_causal.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952), [#23280](https://github.com/sgl-project/sglang/pull/23280), [#24048](https://github.com/sgl-project/sglang/pull/24048), [#24436](https://github.com/sgl-project/sglang/pull/24436), [#24696](https://github.com/sgl-project/sglang/pull/24696), [#25054](https://github.com/sgl-project/sglang/pull/25054), [#25284](https://github.com/sgl-project/sglang/pull/25284), [#26026](https://github.com/sgl-project/sglang/pull/26026), [#26147](https://github.com/sgl-project/sglang/pull/26147), [#26502](https://github.com/sgl-project/sglang/pull/26502), [#27471](https://github.com/sgl-project/sglang/pull/27471) |
| `python/sglang/srt/models/gemma4_mm.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952), [#24048](https://github.com/sgl-project/sglang/pull/24048), [#24436](https://github.com/sgl-project/sglang/pull/24436), [#24696](https://github.com/sgl-project/sglang/pull/24696), [#25054](https://github.com/sgl-project/sglang/pull/25054), [#25284](https://github.com/sgl-project/sglang/pull/25284), [#26147](https://github.com/sgl-project/sglang/pull/26147), [#27471](https://github.com/sgl-project/sglang/pull/27471) |
| `python/sglang/srt/models/gemma4_mtp.py` | [#24436](https://github.com/sgl-project/sglang/pull/24436), [#26026](https://github.com/sgl-project/sglang/pull/26026) |
| `python/sglang/srt/models/gemma4_unified.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/gemma4_vision.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952) |
| `python/sglang/srt/multimodal/processors/gemma4.py` | [#21952](https://github.com/sgl-project/sglang/pull/21952), [#26320](https://github.com/sgl-project/sglang/pull/26320) |
| `python/sglang/srt/multimodal/processors/gemma4_unified.py` | 无直接 PR 号提交 |
| `test/registered/attention/test_gemma4_swa_triton_oob_regression.py` | 无直接 PR 号提交 |
| `test/registered/kernels/test_gemma4_fused_routing.py` | [#26502](https://github.com/sgl-project/sglang/pull/26502) |
| `test/registered/models_e2e/test_gemma4_fp8_per_expert_loading.py` | 无直接 PR 号提交 |
| `test/registered/spec/test_gemma4_dflash_31b_extra.py` | [#27471](https://github.com/sgl-project/sglang/pull/27471) |
| `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` | [#24552](https://github.com/sgl-project/sglang/pull/24552), [#26653](https://github.com/sgl-project/sglang/pull/26653), [#27082](https://github.com/sgl-project/sglang/pull/27082) |
| `test/registered/spec/test_gemma4_mtp_31b_extra.py` | [#24552](https://github.com/sgl-project/sglang/pull/24552), [#27101](https://github.com/sgl-project/sglang/pull/27101) |

## PR 覆盖总览

- git 追溯 PR 数: 22
- 原文档显式引用补充 PR 数: 13
- 当前文档总 PR 数: 35
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-07 | [#21952](https://github.com/sgl-project/sglang/pull/21952) | merged | [New Model] Gemma 4 | `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py` |
| 2026-04-10 | [#22079](https://github.com/sgl-project/sglang/pull/22079) | merged | [nvidia] Gemma4 nvfp4 fix | `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` |
| 2026-04-16 | [#21569](https://github.com/sgl-project/sglang/pull/21569) | merged | Upgrade transformers to 5.5.3 and refactor hf_transformers_utils into subpackage | `python/sglang/srt/utils/hf_transformers/tokenizer.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/step3p5.py` |
| 2026-04-17 | [#22408](https://github.com/sgl-project/sglang/pull/22408) | merged | [CI] Adding Gemma 4 to Nightly CI | `test/registered/eval/test_vlms_mmmu_eval.py` |
| 2026-05-04 | [#24048](https://github.com/sgl-project/sglang/pull/24048) | merged | [VLM] Optimize Gemma4 VLM with PCG and fuse RMSNorm + residual add + scalar | `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py` |
| 2026-05-05 | [#24433](https://github.com/sgl-project/sglang/pull/24433) | merged | Gemma4-mtp cookbook | `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`, `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` |
| 2026-05-07 | [#24436](https://github.com/sgl-project/sglang/pull/24436) | merged | [Gemma 4] Adding MTP support | `python/sglang/srt/models/gemma4_mtp.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_causal.py` |
| 2026-05-10 | [#24696](https://github.com/sgl-project/sglang/pull/24696) | merged | [Gemma4] Optimize Gemm4 with fused Q/K/V RMSNorm + per-expert FP8 ckpt loader | `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py` |
| 2026-05-17 | [#25006](https://github.com/sgl-project/sglang/pull/25006) | merged | Enable trtllm_mha as gemma4 default attn backend. | `python/sglang/srt/server_args.py` |
| 2026-05-18 | [#25547](https://github.com/sgl-project/sglang/pull/25547) | merged | Respect user override for Gemma4 attention backend | `python/sglang/srt/server_args.py` |
| 2026-05-19 | [#25284](https://github.com/sgl-project/sglang/pull/25284) | merged | Support Gemma4 Pipeline Parallelism | `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py` |
| 2026-05-21 | [#25054](https://github.com/sgl-project/sglang/pull/25054) | merged | Support Gemma4 MoE NVFP4 | `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_causal.py` |
| 2026-05-21 | [#25983](https://github.com/sgl-project/sglang/pull/25983) | merged | feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-05-22 | [#24751](https://github.com/sgl-project/sglang/pull/24751) | merged | fix(mm): make multimodal data loading non-blocking to prevent health check stalls | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-05-23 | [#26026](https://github.com/sgl-project/sglang/pull/26026) | merged | [bug fix] Fix 3 issues when using Gemma4 MTP | `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mtp.py` |
| 2026-05-28 | [#24552](https://github.com/sgl-project/sglang/pull/24552) | merged | [Gemma4] Add test for MTP models | `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`, `test/registered/spec/test_gemma4_mtp_31b_extra.py` |
| 2026-05-29 | [#26653](https://github.com/sgl-project/sglang/pull/26653) | merged | test: stabilize Gemma4 26B-A4B MTP GSM8K test with deterministic inference + tuned threshold | `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` |
| 2026-05-31 | [#26799](https://github.com/sgl-project/sglang/pull/26799) | merged | Apply gemma's position offset out-of-place instead of in-place | `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/environ.py` |
| 2026-06-02 | [#26502](https://github.com/sgl-project/sglang/pull/26502) | merged | perf(gemma4): single-launch fused router (topk + softmax + scale) | `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `test/registered/kernels/test_gemma4_fused_routing.py` |
| 2026-06-02 | [#27082](https://github.com/sgl-project/sglang/pull/27082) | merged | test: disable test_gemma4_mtp_26b_a4b_extra from CI | `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` |
| 2026-06-03 | [#27101](https://github.com/sgl-project/sglang/pull/27101) | merged | [Gemma4] Use hard GSM8K accuracy floor for 31B MTP test | `test/registered/spec/test_gemma4_mtp_31b_extra.py` |
| 2026-06-03 | [#27167](https://github.com/sgl-project/sglang/pull/27167) | merged | [Model] Support encoder-free unified Text/Vision/Audio model | `python/sglang/srt/models/gemma4_unified.py`, `python/sglang/srt/multimodal/processors/gemma4_unified.py`, `python/sglang/srt/models/gemma4_mtp.py` |
| 2026-06-03 | [#27171](https://github.com/sgl-project/sglang/pull/27171) | merged | [Docs] Update unified Text/Vision/Audio model cookbook: install + sgl-eval accuracy | `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` |
| 2026-06-04 | [#27287](https://github.com/sgl-project/sglang/pull/27287) | merged | docs(cookbook): add Docker install option for Gemma 4 | `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` |
| 2026-06-05 | [#27321](https://github.com/sgl-project/sglang/pull/27321) | merged | docs(cookbook): restore Gemma 4 transformers commit pin | `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` |
| 2026-06-05 | [#23280](https://github.com/sgl-project/sglang/pull/23280) | merged | [XPU] Enable Gemma 4 E2B / E4B / 31B/ 26B-A4B on Intel XPU | `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py` |
| 2026-06-05 | [#27396](https://github.com/sgl-project/sglang/pull/27396) | merged | Cookbook for QAT | `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`, `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` |
| 2026-06-06 | [#26588](https://github.com/sgl-project/sglang/pull/26588) | merged | Optimize Gemma4 H200 MoE and extend attention | `python/sglang/srt/layers/gemma4_fused_ops.py` |
| 2026-06-09 | [#26320](https://github.com/sgl-project/sglang/pull/26320) | merged | fix(gemma4): register image/video/audio token_regex for HF-expanded prompts | `python/sglang/srt/multimodal/processors/gemma4.py` |
| 2026-06-12 | [#26147](https://github.com/sgl-project/sglang/pull/26147) | merged | [NPU] Add Gemma4 Sliding Window Attention support on Ascend backend | `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_causal.py` |
| 2026-06-17 | [#27471](https://github.com/sgl-project/sglang/pull/27471) | merged | add dflash gemma4 support | `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `test/registered/spec/test_gemma4_dflash_31b_extra.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |
| 2026-06-25 | [#29252](https://github.com/sgl-project/sglang/pull/29252) | merged | Tune Gemma4 26B-A4B B200 memory recipe | `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`, `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` |
| 2026-06-25 | [#29266](https://github.com/sgl-project/sglang/pull/29266) | merged | Sync Gemma4 hardware table with Blackwell recipes | `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` |

## 逐 PR diff 审计卡

### PR #21952 - [New Model] Gemma 4

- 链接: https://github.com/sgl-project/sglang/pull/21952
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/gemma4_detector.py`, `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_audio.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py` 等 7 个文件；关联提交 `2813cb6d9a5b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+6007/-70，可读 patch 6694 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[New Model] Gemma 4」；模型线: Gemma 4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py`；技术摘要: 覆盖「[New Model] Gemma 4」；主要实现面是 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #24048 - [VLM] Optimize Gemma4 VLM with PCG and fuse RMSNorm + residual add + scalar

- 链接: https://github.com/sgl-project/sglang/pull/24048
- 状态/时间: merged / 2026-05-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`；关联提交 `e5c58eb9d627`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+158/-6，可读 patch 223 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Optimize Gemma4 VLM with PCG and fuse RMSNorm + residual add + scalar」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`；技术摘要: 覆盖「[VLM] Optimize Gemma4 VLM with PCG and fuse RMSNorm + residual add + scalar」；主要实现面是 `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/gemma4_fused_ops.py` modified +91/-0 (91 lines); hunks: -77,3 +77,94 @@ def gemma_rmsnorm_residual_scalar(; symbols: gemma_rmsnorm_residual_scalar, _gemma_dual_rmsnorm_residual_kernel, gemma_dual_rmsnorm_residual_scalar，涉及 `gemma_rmsnorm_residual_scalar, _gemma_dual_rmsnorm_residual_kernel, gemma_dual_rmsnorm_residual_scalar`；`python/sglang/srt/models/gemma4_causal.py` modified +29/-2 (31 lines); hunks: -27,7 +27,10; -545,12 +548,36 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/gemma4_mm.py` modified +20/-0 (20 lines); hunks: -224,6 +224,26 @@ def __init__(; symbols: __init__, model, satisfies, __setattr__，涉及 `__init__, model, satisfies`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/gemma4_fused_ops.py` modified +91/-0 (91 lines); hunks: -77,3 +77,94 @@ def gemma_rmsnorm_residual_scalar(; symbols: gemma_rmsnorm_residual_scalar, _gemma_dual_rmsnorm_residual_kernel, gemma_dual_rmsnorm_residual_scalar
  - `python/sglang/srt/models/gemma4_causal.py` modified +29/-2 (31 lines); hunks: -27,7 +27,10; -545,12 +548,36 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/gemma4_mm.py` modified +20/-0 (20 lines); hunks: -224,6 +224,26 @@ def __init__(; symbols: __init__, model, satisfies, __setattr__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/gemma4_fused_ops.py
@@ -77,3 +77,94 @@ def gemma_rmsnorm_residual_scalar(
+@triton.jit
+def _gemma_dual_rmsnorm_residual_kernel(
+    X1_ptr,
+    W1_ptr,
+    X2_ptr,
+    W2_ptr,
diff -- python/sglang/srt/models/gemma4_causal.py
@@ -27,7 +27,10 @@
-from sglang.srt.layers.gemma4_fused_ops import gemma_rmsnorm_residual_scalar
+from sglang.srt.layers.gemma4_fused_ops import (
+    gemma_dual_rmsnorm_residual_scalar,
+    gemma_rmsnorm_residual_scalar,
+)
@@ -545,12 +548,36 @@ def forward(
diff -- python/sglang/srt/models/gemma4_mm.py
@@ -224,6 +224,26 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/gemma4_fused_ops.py` modified +91/-0; `python/sglang/srt/models/gemma4_causal.py` modified +29/-2; `python/sglang/srt/models/gemma4_mm.py` modified +20/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24433 - Gemma4-mtp cookbook

- 链接: https://github.com/sgl-project/sglang/pull/24433
- 状态/时间: merged / 2026-05-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`, `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`；关联提交 `932d89690a7f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+84/-7，可读 patch 166 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Gemma4-mtp cookbook」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`, `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`；技术摘要: 覆盖「Gemma4-mtp cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`, `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +61/-6 (67 lines); hunks: -94,6 +94,7 @@ For the full Docker setup and other installation methods, plea...; -159,6 +160,60 @@ sglang serve --model-path google/gemma-4-26B-A4B-it \；`docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` modified +23/-1 (24 lines); hunks: -41,6 +41,15 @@ export const Gemma4Deployment = () => {; -68,7 +77,7 @@ export const Gemma4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +61/-6 (67 lines); hunks: -94,6 +94,7 @@ For the full Docker setup and other installation methods, plea...; -159,6 +160,60 @@ sglang serve --model-path google/gemma-4-26B-A4B-it \
  - `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` modified +23/-1 (24 lines); hunks: -41,6 +41,15 @@ export const Gemma4Deployment = () => {; -68,7 +77,7 @@ export const Gemma4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Google/Gemma4.mdx
@@ -94,6 +94,7 @@ For the full Docker setup and other installation methods, please refer to the [o
+- **Speculative Decoding (MTP)**: Each Gemma 4 variant ships with a paired `*-assistant` draft model that enables NEXTN multi-token prediction. Enable it via the selector above, o
@@ -159,6 +160,60 @@ sglang serve --model-path google/gemma-4-26B-A4B-it \
+#### Speculative Decoding (MTP) Server Commands
+Each Gemma 4 variant ships with a paired `*-assistant` draft model for NEXTN multi-token prediction. Use the commands below to enable MTP for the corresponding target model. These
+'''bash Command
+# Gemma 4 E2B + MTP
diff -- docs_new/src/snippets/autoregressive/gemma4-deployment.jsx
@@ -41,6 +41,15 @@ export const Gemma4Deployment = () => {
+    speculative: {
+      name: 'speculative',
+      title: 'Speculative Decoding (MTP)',
+      condition: (values) => !['mi300x'].includes(values.hardware),
+      items: [
+        { id: 'disabled', label: 'Disabled', subtitle: 'Baseline', default: true },
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +61/-6; `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` modified +23/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`, `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #24436 - [Gemma 4] Adding MTP support

- 链接: https://github.com/sgl-project/sglang/pull/24436
- 状态/时间: merged / 2026-05-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_mtp.py`；关联提交 `d2c1034163cd`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+1949/-7，可读 patch 2060 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Gemma 4] Adding MTP support」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gemma4_mtp.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_causal.py`；技术摘要: 覆盖「[Gemma 4] Adding MTP support」；主要实现面是 `python/sglang/srt/models/gemma4_mtp.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_causal.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gemma4_mtp.py` added +398/-0 (398 lines); hunks: -0,0 +1,398; symbols: _get_text_config, _resolve_target_text_model, Gemma4AssistantForCausalLM, __init__，涉及 `_get_text_config, _resolve_target_text_model, Gemma4AssistantForCausalLM`；`python/sglang/srt/models/gemma4_mm.py` modified +5/-0 (5 lines); hunks: -256,6 +256,11 @@ def pad_input_ids(; symbols: pad_input_ids, get_input_embeddings, get_embed_and_head, get_attention_sliding_window_size，涉及 `pad_input_ids, get_input_embeddings, get_embed_and_head`；`python/sglang/srt/models/gemma4_causal.py` modified +3/-0 (3 lines); hunks: -878,6 +878,9 @@ def __init__(; symbols: __init__, get_input_embeddings, get_embed_and_head, get_attention_sliding_window_size，涉及 `__init__, get_input_embeddings, get_embed_and_head`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma4_mtp.py` added +398/-0 (398 lines); hunks: -0,0 +1,398; symbols: _get_text_config, _resolve_target_text_model, Gemma4AssistantForCausalLM, __init__
  - `python/sglang/srt/models/gemma4_mm.py` modified +5/-0 (5 lines); hunks: -256,6 +256,11 @@ def pad_input_ids(; symbols: pad_input_ids, get_input_embeddings, get_embed_and_head, get_attention_sliding_window_size
  - `python/sglang/srt/models/gemma4_causal.py` modified +3/-0 (3 lines); hunks: -878,6 +878,9 @@ def __init__(; symbols: __init__, get_input_embeddings, get_embed_and_head, get_attention_sliding_window_size
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma4_mtp.py
@@ -0,0 +1,398 @@
+# Copyright 2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/gemma4_mm.py
@@ -256,6 +256,11 @@ def pad_input_ids(
+    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
+        # Gemma 4 multimodal ties its LM head to the text embed_tokens
+        embed = self.language_model.embed_tokens.weight
+        return embed, embed
diff -- python/sglang/srt/models/gemma4_causal.py
@@ -878,6 +878,9 @@ def __init__(
+    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
+        return self.model.embed_tokens.weight, self.lm_head.weight
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma4_mtp.py` added +398/-0; `python/sglang/srt/models/gemma4_mm.py` modified +5/-0; `python/sglang/srt/models/gemma4_causal.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24696 - [Gemma4] Optimize Gemm4 with fused Q/K/V RMSNorm + per-expert FP8 ckpt loader

- 链接: https://github.com/sgl-project/sglang/pull/24696
- 状态/时间: merged / 2026-05-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`；关联提交 `d3fd91ed9726`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+317/-15，可读 patch 369 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Gemma4] Optimize Gemm4 with fused Q/K/V RMSNorm + per-expert FP8 ckpt loader」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`；技术摘要: 覆盖「[Gemma4] Optimize Gemm4 with fused Q/K/V RMSNorm + per-expert FP8 ckpt loader」；主要实现面是 `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/gemma4_fused_ops.py` modified +115/-0 (115 lines); hunks: -4,6 +4,8; -130,6 +132,119 @@ def _gemma_dual_rmsnorm_residual_kernel(; symbols: _gemma_dual_rmsnorm_residual_kernel, _gemma_qkv_rmsnorm_kernel, gemma_qkv_rmsnorm, gemma_dual_rmsnorm_residual_scalar，涉及 `_gemma_dual_rmsnorm_residual_kernel, _gemma_qkv_rmsnorm_kernel, gemma_qkv_rmsnorm`；`python/sglang/srt/models/gemma4_causal.py` modified +58/-15 (73 lines); hunks: -29,6 +29,7; -339,22 +340,64 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/gemma4_mm.py` modified +35/-0 (35 lines); hunks: -787,6 +787,41 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/gemma4_fused_ops.py` modified +115/-0 (115 lines); hunks: -4,6 +4,8; -130,6 +132,119 @@ def _gemma_dual_rmsnorm_residual_kernel(; symbols: _gemma_dual_rmsnorm_residual_kernel, _gemma_qkv_rmsnorm_kernel, gemma_qkv_rmsnorm, gemma_dual_rmsnorm_residual_scalar
  - `python/sglang/srt/models/gemma4_causal.py` modified +58/-15 (73 lines); hunks: -29,6 +29,7; -339,22 +340,64 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/gemma4_mm.py` modified +35/-0 (35 lines); hunks: -787,6 +787,41 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/gemma4_fused_ops.py
@@ -4,6 +4,8 @@
+from typing import Optional
@@ -130,6 +132,119 @@ def _gemma_dual_rmsnorm_residual_kernel(
+@triton.jit
+def _gemma_qkv_rmsnorm_kernel(
+    Q_ptr,
+    K_ptr,
diff -- python/sglang/srt/models/gemma4_causal.py
@@ -29,6 +29,7 @@
+    gemma_qkv_rmsnorm,
@@ -339,22 +340,64 @@ def forward(
-        q = q.unflatten(-1, (self.num_heads, self.head_dim))
-        q = self.q_norm(q)
-        q = q.flatten(-2, -1)
-        # Check if we should use shared KV cache
diff -- python/sglang/srt/models/gemma4_mm.py
@@ -787,6 +787,41 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/gemma4_fused_ops.py` modified +115/-0; `python/sglang/srt/models/gemma4_causal.py` modified +58/-15; `python/sglang/srt/models/gemma4_mm.py` modified +35/-0
- 验证与风险: diff 自带测试面 `test/registered/models/test_gemma4_fp8_per_expert_loading.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25006 - Enable trtllm_mha as gemma4 default attn backend.

- 链接: https://github.com/sgl-project/sglang/pull/25006
- 状态/时间: merged / 2026-05-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-2，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable trtllm_mha as gemma4 default attn backend.」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/server_args.py`；技术摘要: 覆盖「Enable trtllm_mha as gemma4 default attn backend.」；主要实现面是 `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/server_args.py` modified +6/-2 (8 lines); hunks: -2192,9 +2192,13 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments，涉及 `_handle_model_specific_adjustments`。
- 代码 diff 细节:
  - `python/sglang/srt/server_args.py` modified +6/-2 (8 lines); hunks: -2192,9 +2192,13 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- 关键代码摘录:

```diff
diff -- python/sglang/srt/server_args.py
@@ -2192,9 +2192,13 @@ def _handle_model_specific_adjustments(self):
-            if self.is_attention_backend_not_set():
+            if is_sm100_supported():
+                self.attention_backend = "trtllm_mha"
+            else:
-                logger.info("Use triton as default attention backend for Gemma4")
+            logger.info(
```

- 已读文件:
  - runtime: `python/sglang/srt/server_args.py` modified +6/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25547 - Respect user override for Gemma4 attention backend

- 链接: https://github.com/sgl-project/sglang/pull/25547
- 状态/时间: merged / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+22/-5，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Respect user override for Gemma4 attention backend」；模型线: Gemma 4；类别: 模型实现调整；主要 diff: `python/sglang/srt/server_args.py`；技术摘要: 覆盖「Respect user override for Gemma4 attention backend」；主要实现面是 `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/server_args.py` modified +22/-5 (27 lines); hunks: -2192,12 +2192,29 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments，涉及 `_handle_model_specific_adjustments`。
- 代码 diff 细节:
  - `python/sglang/srt/server_args.py` modified +22/-5 (27 lines); hunks: -2192,12 +2192,29 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- 关键代码摘录:

```diff
diff -- python/sglang/srt/server_args.py
@@ -2192,12 +2192,29 @@ def _handle_model_specific_adjustments(self):
-            if is_sm100_supported():
-                self.attention_backend = "trtllm_mha"
+            default_attention_backend = (
+                "trtllm_mha" if is_sm100_supported() else "triton"
+            )
+            if self.is_attention_backend_not_set():
```

- 已读文件:
  - runtime: `python/sglang/srt/server_args.py` modified +22/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25284 - Support Gemma4 Pipeline Parallelism

- 链接: https://github.com/sgl-project/sglang/pull/25284
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`；关联提交 `4c0ce0345d0d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+487/-68，可读 patch 867 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Gemma4 Pipeline Parallelism」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`；技术摘要: 覆盖「Support Gemma4 Pipeline Parallelism」；主要实现面是 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gemma4_causal.py` modified +212/-34 (246 lines); hunks: -14,7 +14,7; -25,6 +25,7; symbols: get_attention_sliding_window_size, pp_filter_load_weight, Gemma4Router, __init__，涉及 `get_attention_sliding_window_size, pp_filter_load_weight, Gemma4Router`；`python/sglang/srt/models/gemma4_mm.py` modified +133/-33 (166 lines); hunks: -28,11 +28,14; -43,13 +46,17; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma4_causal.py` modified +212/-34 (246 lines); hunks: -14,7 +14,7; -25,6 +25,7; symbols: get_attention_sliding_window_size, pp_filter_load_weight, Gemma4Router, __init__
  - `python/sglang/srt/models/gemma4_mm.py` modified +133/-33 (166 lines); hunks: -28,11 +28,14; -43,13 +46,17; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma4_causal.py
@@ -14,7 +14,7 @@
-from typing import Iterable, List, Optional, Set, Tuple
+from typing import Iterable, List, Optional, Set, Tuple, Union
@@ -25,6 +25,7 @@
+    get_pp_group,
@@ -45,8 +46,9 @@
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
diff -- python/sglang/srt/models/gemma4_mm.py
@@ -28,11 +28,14 @@
+from sglang.srt.distributed import get_pp_group
+from sglang.srt.layers.utils import PPMissingLayer
+from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
@@ -43,13 +46,17 @@
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
+from sglang.srt.model_executor.forward_batch_info import (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma4_causal.py` modified +212/-34; `python/sglang/srt/models/gemma4_mm.py` modified +133/-33
- 验证与风险: diff 自带测试面 `python/sglang/test/test_utils.py`, `test/registered/distributed/test_pp_single_node.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25054 - Support Gemma4 MoE NVFP4

- 链接: https://github.com/sgl-project/sglang/pull/25054
- 状态/时间: merged / 2026-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`；关联提交 `847cbada9c42`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+271/-118，可读 patch 605 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Gemma4 MoE NVFP4」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_causal.py`；技术摘要: 覆盖「Support Gemma4 MoE NVFP4」；主要实现面是 `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_causal.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gemma4_mm.py` modified +88/-61 (149 lines); hunks: -33,6 +33,7; -817,6 +818,27 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`；`python/sglang/srt/models/gemma4_causal.py` modified +81/-27 (108 lines); hunks: -42,6 +42,7; -1140,14 +1141,31 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma4_mm.py` modified +88/-61 (149 lines); hunks: -33,6 +33,7; -817,6 +818,27 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
  - `python/sglang/srt/models/gemma4_causal.py` modified +81/-27 (108 lines); hunks: -42,6 +42,7; -1140,14 +1141,31 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma4_mm.py
@@ -33,6 +33,7 @@
+from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
@@ -817,6 +818,27 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+        # Per-expert checkpoint format used by compressed-tensors / FP8
+        # (e.g. RedHatAI/*-FP8-Dynamic) and by ModelOpt NVFP4
+        # (e.g. nvidia/Gemma-4-*-NVFP4). Each expert is stored as a
+        # separate key with shape (out, in):
diff -- python/sglang/srt/models/gemma4_causal.py
@@ -42,6 +42,7 @@
+from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
@@ -1140,14 +1141,31 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        expert_params_mapping = [
+        fused_expert_params_mapping = [
+        # Per-expert checkpoint format used by compressed-tensors / FP8
+        # (e.g. RedHatAI/*-FP8-Dynamic) and by ModelOpt NVFP4
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma4_mm.py` modified +88/-61; `python/sglang/srt/models/gemma4_causal.py` modified +81/-27
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/cutlass_moe.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25983 - feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext

- 链接: https://github.com/sgl-project/sglang/pull/25983
- 状态/时间: merged / 2026-05-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 77 个文件，+1227/-905，可读 patch 5236 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`；技术摘要: 覆盖「feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext」；主要实现面是 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode，涉及 `get_spec_info, run_once, maybe_init_ngram_embedding`；`python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once，涉及 `capture_one_batch_size, run_once`；`python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size，涉及 `warmup_compile, _cache_loc_dtype, capture_one_batch_size`；`python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp，涉及 `_get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp
  - `python/sglang/srt/model_executor/forward_context.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: ForwardContext, set_forward_context, has_forward_context, get_forward_context
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -146,6 +146,11 @@
+from sglang.srt.model_executor.forward_context import (
+    ForwardContext,
+    forward_context,
+    has_forward_context,
+)
@@ -2638,9 +2643,6 @@ def get_spec_info():
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -65,6 +65,7 @@
+from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
@@ -1016,9 +1017,6 @@ def capture_one_batch_size(
-            req_to_token_pool=self.model_runner.req_to_token_pool,
-            token_to_kv_pool=self.model_runner.token_to_kv_pool,
-            attn_backend=attn_backend,
@@ -1040,85 +1038,90 @@ def capture_one_batch_size(
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -58,6 +58,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44; `python/sglang/srt/model_executor/forward_context.py` added +84/-0; `python/sglang/srt/model_executor/cpu_graph_runner.py` modified +39/-38
- 验证与风险: diff 自带测试面 `test/manual/attention/test_flashattn_backend.py`, `test/manual/attention/test_flashattn_mla_backend.py`, `test/manual/attention/test_prefix_chunk_info.py`, `test/manual/attention/test_trtllm_mla_backend.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24751 - fix(mm): make multimodal data loading non-blocking to prevent health check stalls

- 链接: https://github.com/sgl-project/sglang/pull/24751
- 状态/时间: merged / 2026-05-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+45/-44，可读 patch 401 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(mm): make multimodal data loading non-blocking to prevent health check stalls」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`；技术摘要: 覆盖「fix(mm): make multimodal data loading non-blocking to prevent health check stalls」；主要实现面是 `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data，涉及 `_process_loaded_mm_data, load_mm_data`；`python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async，涉及 `_process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async`；`python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async，涉及 `_process_special_format, process_mm_data_async`；`python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async，涉及 `__init__, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1 (2 lines); hunks: -29,7 +29,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -1,3 +1,4 @@
+import asyncio
@@ -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):
-    def load_mm_data(
+    async def load_mm_data(
@@ -772,7 +773,7 @@ def load_mm_data(
-            return self.legacy_load_mm_data(
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -310,7 +310,7 @@ async def _process_special_format(
-            base_output = self.load_mm_data(
+            base_output = await self.load_mm_data(
@@ -423,7 +423,7 @@ async def process_qwen_mm_data_async(
-        base_output = self.load_mm_data(
+        base_output = await self.load_mm_data(
@@ -644,7 +644,7 @@ async def process_internlm2_mm_data_async(
diff -- python/sglang/srt/multimodal/processors/minicpm.py
@@ -118,7 +118,7 @@ async def _process_special_format(
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7; `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3; `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2; `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_vl_v2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/clip.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26026 - [bug fix] Fix 3 issues when using Gemma4 MTP

- 链接: https://github.com/sgl-project/sglang/pull/26026
- 状态/时间: merged / 2026-05-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mtp.py`；关联提交 `89ff2bc1115c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+20/-11，可读 patch 66 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bug fix] Fix 3 issues when using Gemma4 MTP」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mtp.py`；技术摘要: 覆盖「[bug fix] Fix 3 issues when using Gemma4 MTP」；主要实现面是 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gemma4_causal.py` modified +11/-6 (17 lines); hunks: -1147,7 +1147,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -1159,11 +1160,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights，涉及 `load_weights`；`python/sglang/srt/models/gemma4_mtp.py` modified +2/-0 (2 lines); hunks: -21,6 +21,7; -72,6 +73,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma4_causal.py` modified +11/-6 (17 lines); hunks: -1147,7 +1147,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -1159,11 +1160,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights
  - `python/sglang/srt/models/gemma4_mtp.py` modified +2/-0 (2 lines); hunks: -21,6 +21,7; -72,6 +73,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma4_causal.py
@@ -1147,7 +1147,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        num_experts = self.config.num_experts
+        # Dense subclasses (e.g. the Gemma4 MTP assistant) reuse this.
+        num_experts = getattr(self.config, "num_experts", None) or 0
@@ -1159,11 +1160,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        per_expert_params_mapping = FusedMoE.make_expert_params_mapping(
-            ckpt_gate_proj_name="gate_proj",
diff -- python/sglang/srt/models/gemma4_mtp.py
@@ -21,6 +21,7 @@
+from sglang.srt.distributed import get_pp_group
@@ -72,6 +73,7 @@ def __init__(
+        self.pp_group = get_pp_group()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma4_causal.py` modified +11/-6; `python/sglang/srt/models/gemma4_mtp.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mtp.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24552 - [Gemma4] Add test for MTP models

- 链接: https://github.com/sgl-project/sglang/pull/24552
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`, `test/registered/spec/test_gemma4_mtp_31b_extra.py`；关联提交 `9040feebd854`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+535/-0，可读 patch 538 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Gemma4] Add test for MTP models」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`, `test/registered/spec/test_gemma4_mtp_31b_extra.py`；技术摘要: 覆盖「[Gemma4] Add test for MTP models」；主要实现面是 `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`, `test/registered/spec/test_gemma4_mtp_31b_extra.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` added +185/-0 (185 lines); hunks: -0,0 +1,185; symbols: get_server_info, get_avg_spec_accept_length, TestGemma4MTP26BA4B, _server_env，涉及 `get_server_info, get_avg_spec_accept_length, TestGemma4MTP26BA4B`；`test/registered/spec/test_gemma4_mtp_31b_extra.py` added +185/-0 (185 lines); hunks: -0,0 +1,185; symbols: get_server_info, get_avg_spec_accept_length, TestGemma4MTP31B, _server_env，涉及 `get_server_info, get_avg_spec_accept_length, TestGemma4MTP31B`。
- 代码 diff 细节:
  - `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` added +185/-0 (185 lines); hunks: -0,0 +1,185; symbols: get_server_info, get_avg_spec_accept_length, TestGemma4MTP26BA4B, _server_env
  - `test/registered/spec/test_gemma4_mtp_31b_extra.py` added +185/-0 (185 lines); hunks: -0,0 +1,185; symbols: get_server_info, get_avg_spec_accept_length, TestGemma4MTP31B, _server_env
- 关键代码摘录:

```diff
diff -- test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py
@@ -0,0 +1,185 @@
+import os
+import unittest
+from types import SimpleNamespace
+from typing import Optional
+import requests
+from sglang.srt.utils import kill_process_tree
diff -- test/registered/spec/test_gemma4_mtp_31b_extra.py
@@ -0,0 +1,185 @@
+import os
+import unittest
+from types import SimpleNamespace
+from typing import Optional
+import requests
+from sglang.srt.utils import kill_process_tree
```

- 已读文件:
  - tests: `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` added +185/-0; `test/registered/spec/test_gemma4_mtp_31b_extra.py` added +185/-0
- 验证与风险: diff 自带测试面 `test/registered/spec/test_frozen_kv_mtp.py`, `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`, `test/registered/spec/test_gemma4_mtp_31b_extra.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26653 - test: stabilize Gemma4 26B-A4B MTP GSM8K test with deterministic inference + tuned threshold

- 链接: https://github.com/sgl-project/sglang/pull/26653
- 状态/时间: merged / 2026-05-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`；关联提交 `621a79728c40`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-4，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「test: stabilize Gemma4 26B-A4B MTP GSM8K test with deterministic inference + tuned threshold」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`；技术摘要: 覆盖「test: stabilize Gemma4 26B-A4B MTP GSM8K test with deterministic inference + tuned threshold」；主要实现面是 `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` modified +8/-4 (12 lines); hunks: -31,10 +31,11; -84,6 +85,9 @@ def _common_server_args(cls) -> list[str]:; symbols: _common_server_args，涉及 `_common_server_args`。
- 代码 diff 细节:
  - `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` modified +8/-4 (12 lines); hunks: -31,10 +31,11; -84,6 +85,9 @@ def _common_server_args(cls) -> list[str]:; symbols: _common_server_args
- 关键代码摘录:

```diff
diff -- test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py
@@ -31,10 +31,11 @@
-# Initial values are seeded from current Gemma4 GSM8K observations in the
-# cookbook. Replace each top-k entry with exact MTP first-200-sample scores as
-# CI calibration data becomes available.
-OBSERVED_GSM8K_SCORES = {1: 0.450, 3: 0.450}
+# Calibrated from deterministic-inference GSM8K runs (200 examples, 5-shot,
+# greedy, triton, TP=2). With --enable-deterministic-inference the per-topk
```

- 已读文件:
  - tests: `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` modified +8/-4
- 验证与风险: diff 自带测试面 `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26799 - Apply gemma's position offset out-of-place instead of in-place

- 链接: https://github.com/sgl-project/sglang/pull/26799
- 状态/时间: merged / 2026-05-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-1，可读 patch 29 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Apply gemma's position offset out-of-place instead of in-place」；模型线: Gemma 4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/environ.py`；技术摘要: 覆盖「Apply gemma's position offset out-of-place instead of in-place」；主要实现面是 `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/environ.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gemma4_mm.py` modified +6/-1 (7 lines); hunks: -29,6 +29,7; -595,7 +596,11 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -199,6 +199,7 @@ class Envs:; symbols: Envs，涉及 `Envs`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma4_mm.py` modified +6/-1 (7 lines); hunks: -29,6 +29,7; -595,7 +596,11 @@ def forward(; symbols: forward
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -199,6 +199,7 @@ class Envs:; symbols: Envs
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma4_mm.py
@@ -29,6 +29,7 @@
+from sglang.srt.environ import envs
@@ -595,7 +596,11 @@ def forward(
-        positions += 1
+        if envs.SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION.get():
+            positions = positions + 1
+        else:
diff -- python/sglang/srt/environ.py
@@ -199,6 +199,7 @@ class Envs:
+    SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION = EnvBool(False)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma4_mm.py` modified +6/-1; `python/sglang/srt/environ.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/models/gemma4_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26502 - perf(gemma4): single-launch fused router (topk + softmax + scale)

- 链接: https://github.com/sgl-project/sglang/pull/26502
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `test/registered/kernels/test_gemma4_fused_routing.py`；关联提交 `5ae8d286d2b8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+229/-0，可读 patch 248 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「perf(gemma4): single-launch fused router (topk + softmax + scale)」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `test/registered/kernels/test_gemma4_fused_routing.py`；技术摘要: 覆盖「perf(gemma4): single-launch fused router (topk + softmax + scale)」；主要实现面是 `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`, `test/registered/kernels/test_gemma4_fused_routing.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/gemma4_fused_ops.py` modified +114/-0 (114 lines); hunks: -283,3 +283,117 @@ def gemma_dual_rmsnorm_residual_scalar(; symbols: gemma_dual_rmsnorm_residual_scalar, _gemma4_routing_kernel, gemma4_fused_routing，涉及 `gemma_dual_rmsnorm_residual_scalar, _gemma4_routing_kernel, gemma4_fused_routing`；`python/sglang/srt/models/gemma4_causal.py` modified +9/-0 (9 lines); hunks: -30,6 +30,7; -220,6 +221,14 @@ def routing_function(; symbols: routing_function，涉及 `routing_function`；`test/registered/kernels/test_gemma4_fused_routing.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: fused_routing, _reference, test_matches_reference, test_zero_tokens，涉及 `fused_routing, _reference, test_matches_reference`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/gemma4_fused_ops.py` modified +114/-0 (114 lines); hunks: -283,3 +283,117 @@ def gemma_dual_rmsnorm_residual_scalar(; symbols: gemma_dual_rmsnorm_residual_scalar, _gemma4_routing_kernel, gemma4_fused_routing
  - `python/sglang/srt/models/gemma4_causal.py` modified +9/-0 (9 lines); hunks: -30,6 +30,7; -220,6 +221,14 @@ def routing_function(; symbols: routing_function
  - `test/registered/kernels/test_gemma4_fused_routing.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: fused_routing, _reference, test_matches_reference, test_zero_tokens
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/gemma4_fused_ops.py
@@ -283,3 +283,117 @@ def gemma_dual_rmsnorm_residual_scalar(
+@triton.jit
+def _gemma4_routing_kernel(
+    gating_ptr,  # [T, E] router logits, any float dtype
+    per_expert_scale_ptr,  # [E] per-expert scale (any float dtype)
+    topk_weights_ptr,  # [T, K] fp32 out
+    topk_ids_ptr,  # [T, K] int32 out
diff -- python/sglang/srt/models/gemma4_causal.py
@@ -30,6 +30,7 @@
+    gemma4_fused_routing,
@@ -220,6 +221,14 @@ def routing_function(
+            if (
+                gating_output.is_cuda
+                and gating_output.dim() == 2
+                and gating_output.dtype
diff -- test/registered/kernels/test_gemma4_fused_routing.py
@@ -0,0 +1,106 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/gemma4_fused_ops.py` modified +114/-0; `python/sglang/srt/models/gemma4_causal.py` modified +9/-0
  - tests: `test/registered/kernels/test_gemma4_fused_routing.py` added +106/-0
- 验证与风险: diff 自带测试面 `test/registered/kernels/test_gemma4_fused_routing.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27082 - test: disable test_gemma4_mtp_26b_a4b_extra from CI

- 链接: https://github.com/sgl-project/sglang/pull/27082
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`；关联提交 `22bb9a6421e8`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「test: disable test_gemma4_mtp_26b_a4b_extra from CI」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`；技术摘要: 覆盖「test: disable test_gemma4_mtp_26b_a4b_extra from CI」；主要实现面是 `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` modified +6/-1 (7 lines); hunks: -17,7 +17,12。
- 代码 diff 细节:
  - `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` modified +6/-1 (7 lines); hunks: -17,7 +17,12
- 关键代码摘录:

```diff
diff -- test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py
@@ -17,7 +17,12 @@
-register_cuda_ci(est_time=720, stage="extra-a", runner_config="2-gpu-large")
+register_cuda_ci(
+    est_time=720,
+    stage="extra-a",
+    runner_config="2-gpu-large",
+    disabled="FIXME(kpham-sgl): temporary drop due to accuracies issue",
```

- 已读文件:
  - tests: `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py` modified +6/-1
- 验证与风险: diff 自带测试面 `test/registered/spec/test_gemma4_mtp_26b_a4b_extra.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27101 - [Gemma4] Use hard GSM8K accuracy floor for 31B MTP test

- 链接: https://github.com/sgl-project/sglang/pull/27101
- 状态/时间: merged / 2026-06-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/spec/test_gemma4_mtp_31b_extra.py`；关联提交 `6d5361569988`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-6，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Gemma4] Use hard GSM8K accuracy floor for 31B MTP test」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `test/registered/spec/test_gemma4_mtp_31b_extra.py`；技术摘要: 覆盖「[Gemma4] Use hard GSM8K accuracy floor for 31B MTP test」；主要实现面是 `test/registered/spec/test_gemma4_mtp_31b_extra.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/spec/test_gemma4_mtp_31b_extra.py` modified +2/-6 (8 lines); hunks: -28,14 +28,10。
- 代码 diff 细节:
  - `test/registered/spec/test_gemma4_mtp_31b_extra.py` modified +2/-6 (8 lines); hunks: -28,14 +28,10
- 关键代码摘录:

```diff
diff -- test/registered/spec/test_gemma4_mtp_31b_extra.py
@@ -28,14 +28,10 @@
-GSM8K_SCORE_MARGIN = 0.03
-# Initial values are seeded from current Gemma4 GSM8K observations in the
-# cookbook. Replace each top-k entry with exact MTP first-200-sample scores as
-# CI calibration data becomes available.
-OBSERVED_GSM8K_SCORES = {1: 0.805, 3: 0.805}
-GSM8K_SCORE_THRESHOLD = min(OBSERVED_GSM8K_SCORES.values()) - GSM8K_SCORE_MARGIN
```

- 已读文件:
  - tests: `test/registered/spec/test_gemma4_mtp_31b_extra.py` modified +2/-6
- 验证与风险: diff 自带测试面 `test/registered/spec/test_gemma4_mtp_31b_extra.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27167 - [Model] Support encoder-free unified Text/Vision/Audio model

- 链接: https://github.com/sgl-project/sglang/pull/27167
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+816/-5，可读 patch 1022 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support encoder-free unified Text/Vision/Audio model」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gemma4_unified.py`, `python/sglang/srt/multimodal/processors/gemma4_unified.py`, `python/sglang/srt/models/gemma4_mtp.py`；技术摘要: 覆盖「[Model] Support encoder-free unified Text/Vision/Audio model」；主要实现面是 `python/sglang/srt/models/gemma4_unified.py`, `python/sglang/srt/multimodal/processors/gemma4_unified.py`, `python/sglang/srt/models/gemma4_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gemma4_unified.py` added +438/-0 (438 lines); hunks: -0,0 +1,438; symbols: Gemma4UnifiedVisionEmbedder, __init__, forward, Gemma4UnifiedMultimodalEmbedder，涉及 `Gemma4UnifiedVisionEmbedder, __init__, forward`；`python/sglang/srt/multimodal/processors/gemma4_unified.py` added +33/-0 (33 lines); hunks: -0,0 +1,33; symbols: Gemma4UnifiedSGLangProcessor, _get_audio_pad_multiple，涉及 `Gemma4UnifiedSGLangProcessor, _get_audio_pad_multiple`；`python/sglang/srt/models/gemma4_mtp.py` modified +5/-1 (6 lines); hunks: -397,4 +397,8 @@ def _reorder_embedding_to_centroid_order(self) -> None:; symbols: _reorder_embedding_to_centroid_order, Gemma4UnifiedAssistantForCausalLM，涉及 `_reorder_embedding_to_centroid_order, Gemma4UnifiedAssistantForCausalLM`；`python/sglang/srt/configs/model_config.py` modified +4/-0 (4 lines); hunks: -531,6 +531,7 @@ def _derive_hybrid_model(self):; -1515,6 +1516,7 @@ def is_generation_model(model_architectures: List[str], is...; symbols: _derive_hybrid_model, _detect_attention_sinks, is_generation_model, is_hybrid_swa_model，涉及 `_derive_hybrid_model, _detect_attention_sinks, is_generation_model`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma4_unified.py` added +438/-0 (438 lines); hunks: -0,0 +1,438; symbols: Gemma4UnifiedVisionEmbedder, __init__, forward, Gemma4UnifiedMultimodalEmbedder
  - `python/sglang/srt/multimodal/processors/gemma4_unified.py` added +33/-0 (33 lines); hunks: -0,0 +1,33; symbols: Gemma4UnifiedSGLangProcessor, _get_audio_pad_multiple
  - `python/sglang/srt/models/gemma4_mtp.py` modified +5/-1 (6 lines); hunks: -397,4 +397,8 @@ def _reorder_embedding_to_centroid_order(self) -> None:; symbols: _reorder_embedding_to_centroid_order, Gemma4UnifiedAssistantForCausalLM
  - `python/sglang/srt/configs/model_config.py` modified +4/-0 (4 lines); hunks: -531,6 +531,7 @@ def _derive_hybrid_model(self):; -1515,6 +1516,7 @@ def is_generation_model(model_architectures: List[str], is...; symbols: _derive_hybrid_model, _detect_attention_sinks, is_generation_model, is_hybrid_swa_model
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +2/-1 (3 lines); hunks: -174,7 +174,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma4_unified.py
@@ -0,0 +1,438 @@
+# Copyright 2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/multimodal/processors/gemma4_unified.py
@@ -0,0 +1,33 @@
+# Copyright 2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/gemma4_mtp.py
@@ -397,4 +397,8 @@ def _reorder_embedding_to_centroid_order(self) -> None:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma4_unified.py` added +438/-0; `python/sglang/srt/multimodal/processors/gemma4_unified.py` added +33/-0; `python/sglang/srt/models/gemma4_mtp.py` modified +5/-1; `python/sglang/srt/configs/model_config.py` modified +4/-0; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +2/-1; `python/sglang/srt/multimodal/processors/base_processor.py` modified +1/-0
  - docs: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +311/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/arg_groups/speculative_hook.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27171 - [Docs] Update unified Text/Vision/Audio model cookbook: install + sgl-eval accuracy

- 链接: https://github.com/sgl-project/sglang/pull/27171
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+27/-15，可读 patch 62 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Update unified Text/Vision/Audio model cookbook: install + sgl-eval accuracy」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；技术摘要: 覆盖「[Docs] Update unified Text/Vision/Audio model cookbook: install + sgl-eval accuracy」；主要实现面是 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +27/-15 (42 lines); hunks: -67,27 +67,17 @@ Gemma 4 is Google's next-generation family of open models, b...; -1437,6 +1427,28 @@ Median ITL (ms): 15.11。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +27/-15 (42 lines); hunks: -67,27 +67,17 @@ Gemma 4 is Google's next-generation family of open models, b...; -1437,6 +1427,28 @@ Median ITL (ms): 15.11
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Google/Gemma4.mdx
@@ -67,27 +67,17 @@ Gemma 4 is Google's next-generation family of open models, building on the Gemma
-Gemma 4 support requires [sgl-project/sglang#21952](https://github.com/sgl-project/sglang/pull/21952) and a specific transformers commit:
+Gemma 4 (including the encoder-free unified 12B, [sgl-project/sglang#27167](https://github.com/sgl-project/sglang/pull/27167)) is supported on SGLang main. Install it together wit
-# Install SGLang from main branch (after sglang#21952 is merged)
+# Install SGLang from main
-# Install transformers with Gemma 4 support
-pip install 'git+https://github.com/huggingface/transformers.git@91b1ab1fdfa81a552644a92fbe3e8d88de40e167'
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +27/-15
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27287 - docs(cookbook): add Docker install option for Gemma 4

- 链接: https://github.com/sgl-project/sglang/pull/27287
- 状态/时间: merged / 2026-06-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；关联提交 `75be9224519b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+20/-3，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): add Docker install option for Gemma 4」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；技术摘要: 覆盖「docs(cookbook): add Docker install option for Gemma 4」；主要实现面是 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +20/-3 (23 lines); hunks: -67,14 +67,31 @@ Gemma 4 is Google's next-generation family of open models, b...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +20/-3 (23 lines); hunks: -67,14 +67,31 @@ Gemma 4 is Google's next-generation family of open models, b...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Google/Gemma4.mdx
@@ -67,14 +67,31 @@ Gemma 4 is Google's next-generation family of open models, building on the Gemma
-Gemma 4 (including the encoder-free unified 12B, [sgl-project/sglang#27167](https://github.com/sgl-project/sglang/pull/27167)) is supported on SGLang main. Install it together wit
+Gemma 4 (including the encoder-free unified 12B, [sgl-project/sglang#27167](https://github.com/sgl-project/sglang/pull/27167)) is supported on SGLang main:
+'''
+### Docker (prebuilt dev image)
+Prebuilt development images bundle SGLang together with the matching transformers commit preinstalled, so no manual install is needed. All tags are multi-arch (`amd64` + `arm64`):
-# Install transformers with Gemma 4 support (encoder-free unified family included)
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +20/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27321 - docs(cookbook): restore Gemma 4 transformers commit pin

- 链接: https://github.com/sgl-project/sglang/pull/27321
- 状态/时间: merged / 2026-06-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；关联提交 `7425bebb6c5e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): restore Gemma 4 transformers commit pin」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；技术摘要: 覆盖「docs(cookbook): restore Gemma 4 transformers commit pin」；主要实现面是 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +4/-1 (5 lines); hunks: -67,11 +67,14 @@ Gemma 4 is Google's next-generation family of open models, b...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +4/-1 (5 lines); hunks: -67,11 +67,14 @@ Gemma 4 is Google's next-generation family of open models, b...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Google/Gemma4.mdx
@@ -67,11 +67,14 @@ Gemma 4 is Google's next-generation family of open models, building on the Gemma
-Gemma 4 (including the encoder-free unified 12B, [sgl-project/sglang#27167](https://github.com/sgl-project/sglang/pull/27167)) is supported on SGLang main:
+Gemma 4 (including the encoder-free unified 12B, [sgl-project/sglang#27167](https://github.com/sgl-project/sglang/pull/27167)) is supported on SGLang main. Install it together wit
+# Install transformers with Gemma 4 support (encoder-free unified family included)
+pip install 'git+https://github.com/huggingface/transformers.git@1423d22f7a3b62e8c70ad67b58ec25cd9b675897'
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +4/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23280 - [XPU] Enable Gemma 4 E2B / E4B / 31B/ 26B-A4B on Intel XPU

- 链接: https://github.com/sgl-project/sglang/pull/23280
- 状态/时间: merged / 2026-06-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`；关联提交 `2c8357f79471`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+348/-34，可读 patch 641 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU] Enable Gemma 4 E2B / E4B / 31B/ 26B-A4B on Intel XPU」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`；技术摘要: 覆盖「[XPU] Enable Gemma 4 E2B / E4B / 31B/ 26B-A4B on Intel XPU」；主要实现面是 `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/models/gemma4_causal.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/gemma4_fused_ops.py` modified +71/-2 (73 lines); hunks: -215,15 +215,15 @@ def gemma_qkv_rmsnorm(; -245,6 +245,75 @@ def gemma_qkv_rmsnorm(; symbols: gemma_qkv_rmsnorm, _gemma_routing_post_topk_kernel, gemma_routing_post_topk, gemma_dual_rmsnorm_residual_scalar，涉及 `gemma_qkv_rmsnorm, _gemma_routing_post_topk_kernel, gemma_routing_post_topk`；`python/sglang/srt/models/gemma4_causal.py` modified +44/-17 (61 lines); hunks: -34,6 +34,7; -56,6 +57,9; symbols: __init__, fuse_scale, forward, routing_function，涉及 `__init__, fuse_scale, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/gemma4_fused_ops.py` modified +71/-2 (73 lines); hunks: -215,15 +215,15 @@ def gemma_qkv_rmsnorm(; -245,6 +245,75 @@ def gemma_qkv_rmsnorm(; symbols: gemma_qkv_rmsnorm, _gemma_routing_post_topk_kernel, gemma_routing_post_topk, gemma_dual_rmsnorm_residual_scalar
  - `python/sglang/srt/models/gemma4_causal.py` modified +44/-17 (61 lines); hunks: -34,6 +34,7; -56,6 +57,9; symbols: __init__, fuse_scale, forward, routing_function
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/gemma4_fused_ops.py
@@ -215,15 +215,15 @@ def gemma_qkv_rmsnorm(
-    assert q.is_cuda
+    assert q.is_cuda or q.is_xpu
-        assert k.is_cuda and v.is_cuda
+        assert (k.is_cuda and v.is_cuda) or (k.is_xpu and v.is_xpu)
@@ -245,6 +245,75 @@ def gemma_qkv_rmsnorm(
+@triton.jit
diff -- python/sglang/srt/models/gemma4_causal.py
@@ -34,6 +34,7 @@
+    gemma_routing_post_topk,
@@ -56,6 +57,9 @@
+from sglang.srt.models.utils import (
+    create_fused_set_kv_buffer_arg,
+)
@@ -145,7 +149,8 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/gemma4_fused_ops.py` modified +71/-2; `python/sglang/srt/models/gemma4_causal.py` modified +44/-17
- 验证与风险: diff 自带测试面 `test/registered/xpu/test_gemma_4_e2b.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27396 - Cookbook for QAT

- 链接: https://github.com/sgl-project/sglang/pull/27396
- 状态/时间: merged / 2026-06-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+16/-2，可读 patch 51 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Cookbook for QAT」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`, `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；技术摘要: 覆盖「Cookbook for QAT」；主要实现面是 `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`, `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` modified +15/-2 (17 lines); hunks: -11,6 +11,14 @@ export const Gemma4Deployment = () => {; -90,12 +98,17 @@ export const Gemma4Deployment = () => {；`docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +1/-0 (1 lines); hunks: -113,6 +113,7 @@ For other installation methods, please refer to the [officia...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` modified +15/-2 (17 lines); hunks: -11,6 +11,14 @@ export const Gemma4Deployment = () => {; -90,12 +98,17 @@ export const Gemma4Deployment = () => {
  - `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +1/-0 (1 lines); hunks: -113,6 +113,7 @@ For other installation methods, please refer to the [officia...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/gemma4-deployment.jsx
@@ -11,6 +11,14 @@ export const Gemma4Deployment = () => {
+    checkpoint: {
+      name: 'checkpoint',
+      title: 'Checkpoint',
+      items: [
+        { id: 'standard', label: 'Standard', subtitle: 'BF16', default: true },
+        { id: 'qat', label: 'QAT', subtitle: 'q4_0-unquantized', default: false },
diff -- docs_new/cookbook/autoregressive/Google/Gemma4.mdx
@@ -113,6 +113,7 @@ For other installation methods, please refer to the [official SGLang installatio
+- **QAT checkpoints**: Toggle **Checkpoint → QAT** in the selector to target the `qat-q4_0-unquantized` releases. These keep bf16 weights, so memory and TP requirements match the
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` modified +15/-2; `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +1/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`, `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26588 - Optimize Gemma4 H200 MoE and extend attention

- 链接: https://github.com/sgl-project/sglang/pull/26588
- 状态/时间: merged / 2026-06-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/gemma4_fused_ops.py`；关联提交 `f57f8a8afd84`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+337/-35，可读 patch 418 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimize Gemma4 H200 MoE and extend attention」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/gemma4_fused_ops.py`；技术摘要: 覆盖「Optimize Gemma4 H200 MoE and extend attention」；主要实现面是 `python/sglang/srt/layers/gemma4_fused_ops.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/gemma4_fused_ops.py` modified +106/-34 (140 lines); hunks: -132,6 +132,29 @@ def _gemma_dual_rmsnorm_residual_kernel(; -147,48 +170,75 @@ def _gemma_qkv_rmsnorm_kernel(; symbols: _gemma_dual_rmsnorm_residual_kernel, _gemma_qkv_rmsnorm_store, _gemma_qkv_rmsnorm_kernel, gemma_qkv_rmsnorm，涉及 `_gemma_dual_rmsnorm_residual_kernel, _gemma_qkv_rmsnorm_store, _gemma_qkv_rmsnorm_kernel`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/gemma4_fused_ops.py` modified +106/-34 (140 lines); hunks: -132,6 +132,29 @@ def _gemma_dual_rmsnorm_residual_kernel(; -147,48 +170,75 @@ def _gemma_qkv_rmsnorm_kernel(; symbols: _gemma_dual_rmsnorm_residual_kernel, _gemma_qkv_rmsnorm_store, _gemma_qkv_rmsnorm_kernel, gemma_qkv_rmsnorm
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/gemma4_fused_ops.py
@@ -132,6 +132,29 @@ def _gemma_dual_rmsnorm_residual_kernel(
+@triton.jit
+def _gemma_qkv_rmsnorm_store(
+    X_ptr,
+    W_ptr,
+    stride_m,
+    m,
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/gemma4_fused_ops.py` modified +106/-34
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`, `python/sglang/srt/layers/gemma4_fused_ops.py`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=128,N=704,device_name=NVIDIA_H200.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26320 - fix(gemma4): register image/video/audio token_regex for HF-expanded prompts

- 链接: https://github.com/sgl-project/sglang/pull/26320
- 状态/时间: merged / 2026-06-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/gemma4.py`；关联提交 `98fe7e326ed4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+13/-0，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(gemma4): register image/video/audio token_regex for HF-expanded prompts」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/gemma4.py`；技术摘要: 覆盖「fix(gemma4): register image/video/audio token_regex for HF-expanded prompts」；主要实现面是 `python/sglang/srt/multimodal/processors/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/gemma4.py` modified +13/-0 (13 lines); hunks: -12,6 +12,7; -41,9 +42,21 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/gemma4.py` modified +13/-0 (13 lines); hunks: -12,6 +12,7; -41,9 +42,21 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/gemma4.py
@@ -12,6 +12,7 @@
+import re
@@ -41,9 +42,21 @@ def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
+            image_token="<|image|>",
+            image_token_regex=re.compile(
+                r"<\|image>(?:<\|image\|>)+<image\|>|<\|image\|>"
+            ),
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/gemma4.py` modified +13/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/gemma4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26147 - [NPU] Add Gemma4 Sliding Window Attention support on Ascend backend

- 链接: https://github.com/sgl-project/sglang/pull/26147
- 状态/时间: merged / 2026-06-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`；关联提交 `3a3a75946404`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+563/-132，可读 patch 979 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Add Gemma4 Sliding Window Attention support on Ascend backend」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_causal.py`；技术摘要: 覆盖「[NPU] Add Gemma4 Sliding Window Attention support on Ascend backend」；主要实现面是 `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_causal.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gemma4_mm.py` modified +10/-3 (13 lines); hunks: -608,9 +608,16 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/gemma4_causal.py` modified +3/-1 (4 lines); hunks: -295,7 +295,9 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma4_mm.py` modified +10/-3 (13 lines); hunks: -608,9 +608,16 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/gemma4_causal.py` modified +3/-1 (4 lines); hunks: -295,7 +295,9 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma4_mm.py
@@ -608,9 +608,16 @@ def forward(
-            ple_ids[input_ids == self.config.image_token_id] = pad_id
-            ple_ids[input_ids == self.config.video_token_id] = pad_id
-            ple_ids[input_ids == self.config.audio_token_id] = pad_id
+            # Use torch.where instead of boolean indexing for NPU graph compatibility
+            ple_ids = torch.where(
+                input_ids == self.config.image_token_id, pad_id, ple_ids
diff -- python/sglang/srt/models/gemma4_causal.py
@@ -295,7 +295,9 @@ def __init__(
-            config.sliding_window if layer_type == "sliding_attention" else None
+            get_attention_sliding_window_size(config)
+            if layer_type == "sliding_attention"
+            else -1
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma4_mm.py` modified +10/-3; `python/sglang/srt/models/gemma4_causal.py` modified +3/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/ascend/test_ascend_utils.py`, `test/manual/ascend/llm_models/test_npu_gemma_4_26b_a4b_it_llm.py`, `test/manual/ascend/llm_models/test_npu_gemma_4_31b_llm.py`, `test/manual/ascend/llm_models/test_npu_gemma_4_e2b_llm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27471 - add dflash gemma4 support

- 链接: https://github.com/sgl-project/sglang/pull/27471
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `test/registered/spec/test_gemma4_dflash_31b_extra.py`；关联提交 `5ea0d1d09381`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+211/-16，可读 patch 259 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「add dflash gemma4 support」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `test/registered/spec/test_gemma4_dflash_31b_extra.py`；技术摘要: 覆盖「add dflash gemma4 support」；主要实现面是 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `test/registered/spec/test_gemma4_dflash_31b_extra.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gemma4_causal.py` modified +8/-0 (8 lines); hunks: -1127,6 +1127,14 @@ def get_attention_sliding_window_size(self):; symbols: get_attention_sliding_window_size, dtype, set_dflash_layers_to_capture, forward，涉及 `get_attention_sliding_window_size, dtype, set_dflash_layers_to_capture`；`python/sglang/srt/models/gemma4_mm.py` modified +8/-0 (8 lines); hunks: -301,6 +301,14 @@ def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.T...; symbols: get_embed_and_head, get_attention_sliding_window_size, set_dflash_layers_to_capture, prepare_attn_masks，涉及 `get_embed_and_head, get_attention_sliding_window_size, set_dflash_layers_to_capture`；`test/registered/spec/test_gemma4_dflash_31b_extra.py` added +177/-0 (177 lines); hunks: -0,0 +1,177; symbols: get_server_info, get_avg_spec_accept_length, TestGemma4DFlash31B, _common_server_args，涉及 `get_server_info, get_avg_spec_accept_length, TestGemma4DFlash31B`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma4_causal.py` modified +8/-0 (8 lines); hunks: -1127,6 +1127,14 @@ def get_attention_sliding_window_size(self):; symbols: get_attention_sliding_window_size, dtype, set_dflash_layers_to_capture, forward
  - `python/sglang/srt/models/gemma4_mm.py` modified +8/-0 (8 lines); hunks: -301,6 +301,14 @@ def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.T...; symbols: get_embed_and_head, get_attention_sliding_window_size, set_dflash_layers_to_capture, prepare_attn_masks
  - `test/registered/spec/test_gemma4_dflash_31b_extra.py` added +177/-0 (177 lines); hunks: -0,0 +1,177; symbols: get_server_info, get_avg_spec_accept_length, TestGemma4DFlash31B, _common_server_args
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma4_causal.py
@@ -1127,6 +1127,14 @@ def get_attention_sliding_window_size(self):
+    def set_dflash_layers_to_capture(self, layer_ids: list[int]):
+        if layer_ids is None:
+            raise ValueError(
+                "DFLASH requires explicit layer_ids for aux hidden capture."
+            )
+        self.capture_aux_hidden_states = True
diff -- python/sglang/srt/models/gemma4_mm.py
@@ -301,6 +301,14 @@ def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
+    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
+        if layer_ids is None:
+            raise ValueError(
+                "DFLASH requires explicit layer_ids for aux hidden capture."
+            )
+        self.capture_aux_hidden_states = True
diff -- test/registered/spec/test_gemma4_dflash_31b_extra.py
@@ -0,0 +1,177 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma4_causal.py` modified +8/-0; `python/sglang/srt/models/gemma4_mm.py` modified +8/-0
  - tests: `test/registered/spec/test_gemma4_dflash_31b_extra.py` added +177/-0
- 验证与风险: diff 自带测试面 `test/registered/spec/test_gemma4_dflash_31b_extra.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- 链接: https://github.com/sgl-project/sglang/pull/28567
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 184 个文件，+1865/-1727，可读 patch 8932 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_parallel(): a structured accessor for parallel-topology state」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Add get_parallel(): a structured accessor for parallel-topology state」；主要实现面是 `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention，涉及 `ApertusMLP, __init__, forward`；`python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales，涉及 `__init__, forward, load_kv_cache_scales`；`python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__，涉及 `_resolve_moe_input_pad_multiple, __init__`；`python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/apertus.py
@@ -1,687 +1,686 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Copyright 2025 The SwissAI Initiative
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
diff -- python/sglang/srt/models/solar.py
@@ -1,37 +1,14 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
-#
-# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,21 +28,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28697 - [docs] Add B300 cookbook deployment options

- 链接: https://github.com/sgl-project/sglang/pull/28697
- 状态/时间: merged / 2026-06-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+503/-69，可读 patch 1291 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Add B300 cookbook deployment options」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`；技术摘要: 覆盖「[docs] Add B300 cookbook deployment options」；主要实现面是 `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167；`docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {；`docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {；`docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167
  - `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {
  - `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15 (38 lines); hunks: -8,19 +8,19 @@ export const Qwen35Deployment = () => {; -149,7 +149,7 @@ export const Qwen35Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx
@@ -0,0 +1,167 @@
+export const InternS1Deployment = () => {
+  const options = {
+    hardware: {
+      name: 'hardware',
+      title: 'Hardware Platform',
+      items: [
diff -- docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx
@@ -9,6 +9,11 @@ const lookupData = {
+      {
+        "id": "b300",
+        "label": "B300",
+        "default": false
+      },
@@ -182,6 +187,66 @@ const lookupData = {
diff -- docs_new/src/snippets/autoregressive/glm-5-deployment.jsx
@@ -4,6 +4,7 @@ export const GLM5Deployment = () => {
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15; `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` modified +16/-13
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx`, `docs_new/src/snippets/autoregressive/deepseek-math-v2-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #29252 - Tune Gemma4 26B-A4B B200 memory recipe

- 链接: https://github.com/sgl-project/sglang/pull/29252
- 状态/时间: merged / 2026-06-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`, `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`；关联提交 `efbe67d23787`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+2/-1，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Tune Gemma4 26B-A4B B200 memory recipe」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`, `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；技术摘要: 覆盖「Tune Gemma4 26B-A4B B200 memory recipe」；主要实现面是 `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`, `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` modified +1/-1 (2 lines); hunks: -75,7 +75,7 @@ export const Gemma4Deployment = () => {；`docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +1/-0 (1 lines); hunks: -111,6 +111,7 @@ For other installation methods, please refer to the [officia...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` modified +1/-1 (2 lines); hunks: -75,7 +75,7 @@ export const Gemma4Deployment = () => {
  - `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +1/-0 (1 lines); hunks: -111,6 +111,7 @@ For other installation methods, please refer to the [officia...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/gemma4-deployment.jsx
@@ -75,7 +75,7 @@ export const Gemma4Deployment = () => {
-      '26b-a4b': { tp: 1, mem: 0.9 },
+      '26b-a4b': { tp: 1, mem: 0.75 },
diff -- docs_new/cookbook/autoregressive/Google/Gemma4.mdx
@@ -111,6 +111,7 @@ For other installation methods, please refer to the [official SGLang installatio
+- **Gemma 4 26B-A4B on B200**: Use `--mem-fraction-static 0.75` to leave workspace headroom for the Triton MoE path.
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx` modified +1/-1; `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +1/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`, `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #29266 - Sync Gemma4 hardware table with Blackwell recipes

- 链接: https://github.com/sgl-project/sglang/pull/29266
- 状态/时间: merged / 2026-06-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；关联提交 `4d06d4c97f46`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-6，可读 patch 34 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Sync Gemma4 hardware table with Blackwell recipes」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；技术摘要: 覆盖「Sync Gemma4 hardware table with Blackwell recipes」；主要实现面是 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +6/-6 (12 lines); hunks: -133,27 +133,27 @@ For other installation methods, please refer to the [offic...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +6/-6 (12 lines); hunks: -133,27 +133,27 @@ For other installation methods, please refer to the [offic...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Google/Gemma4.mdx
@@ -133,27 +133,27 @@ For other installation methods, please refer to the [official SGLang installatio
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>1x H200 / 1x MI300X / 1x MI325X / 1x MI355X</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>1x H200 / 1x B200 / 1x B300 / 1x MI300X / 1x MI325X / 1x MI355X</td>
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>1x H200 / 1x MI300X / 1x MI325X / 1x MI355X</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>1x H200 / 1x B200 / 1x B300 / 1x MI300X / 1x MI325X / 1x MI355X</td>
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>1x H200 / 1x B200</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>1x H200 / 1x B200 / 1x B300</td>
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` modified +6/-6
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Google/Gemma4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
