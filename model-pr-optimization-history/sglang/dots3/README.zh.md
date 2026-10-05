# SGLang dots3 (dots.note) 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
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

## PR 覆盖总览

- git 追溯 PR 数: 3
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 3
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-08-14 | [#34797](https://github.com/sgl-project/sglang/pull/34797) | merged | docs: link dots3.note checkpoints, add H100 cells | `docs/src/snippets/configs/rednote/dots3-note.jsx`, `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx` |
| 2026-08-22 | [#33829](https://github.com/sgl-project/sglang/pull/33829) | merged | [Model] Complete dots.note.omni support with native encoders, video preprocessing, and MTP decoding | `python/sglang/srt/models/dots3_common/modeling.py`, `python/sglang/srt/models/dots3_common/dots_omni_audio.py`, `python/sglang/srt/models/dots3_common/dots_omni_vision.py` |
| 2026-09-25 | [#41198](https://github.com/sgl-project/sglang/pull/41198) | merged | [Refactor] Move Step-3.5, GLM5-Next, Dots3, MiniMax-M3 and Qwen3.5 onto ffn_exit | `python/sglang/srt/models/dots3_common/modeling.py` |

## 逐 PR diff 审计卡

### PR #34797 - docs: link dots3.note checkpoints, add H100 cells

- 链接: https://github.com/sgl-project/sglang/pull/34797
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx`, `docs/src/snippets/configs/rednote/dots3-note.jsx`；关联提交 `6ad3f2d8fdc8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+128/-21，可读 patch 229 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/src/snippets/configs/rednote/dots3-note.jsx` modified +107/-4 (111 lines); hunks: -9,7 +9,7 @@ export const config = {; -24,8 +24,8 @@ export const config = {；`docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx` modified +2/-12 (14 lines); hunks: -46,16 +46,12 @@ For how to launch the image, see [Install → Method 3: Using...; -70,13 +66,7 @@ dots3.note is RedNote's native multimodal omni model, built o...。
- 代码 diff 细节:
  - `docs/src/snippets/configs/rednote/dots3-note.jsx` modified +107/-4 (111 lines); hunks: -9,7 +9,7 @@ export const config = {; -24,8 +24,8 @@ export const config = {
  - `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx` modified +2/-12 (14 lines); hunks: -46,16 +46,12 @@ For how to launch the image, see [Install → Method 3: Using...; -70,13 +66,7 @@ dots3.note is RedNote's native multimodal omni model, built o...
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - docs: `docs/src/snippets/configs/rednote/dots3-note.jsx` modified +107/-4; `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx` modified +2/-12
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx`, `docs/src/snippets/_deployment.jsx`, `docs/src/snippets/_playground.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #33829 - [Model] Complete dots.note.omni support with native encoders, video preprocessing, and MTP decoding

- 链接: https://github.com/sgl-project/sglang/pull/33829
- 状态/时间: merged / 2026-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx`, `docs/src/snippets/configs/rednote/dots3-note.jsx`, `python/sglang/srt/configs/dots3.py`, `python/sglang/srt/models/dots3.py`, `python/sglang/srt/models/dots3_common/__init__.py` 等 12 个文件；关联提交 `af39ad93493c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 55 个文件，+9639/-155，可读 patch 7772 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/dots3_common/modeling.py` added +2824/-0 (2824 lines)；`python/sglang/srt/models/dots3_common/dots_omni_audio.py` added +1027/-0 (1027 lines); hunks: -0,0 +1,1027; symbols: DotsWhisperConfig, __init__, RMSNorm, forward，涉及 `DotsWhisperConfig, __init__, RMSNorm`；`python/sglang/srt/models/dots3_common/dots_omni_vision.py` added +769/-0 (769 lines); hunks: -0,0 +1,769; symbols: VisionRotaryEmbedding, __init__, _compute_freqs, forward，涉及 `VisionRotaryEmbedding, __init__, _compute_freqs`；`python/sglang/srt/configs/dots3.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: DotsNoteOmniTokenizerProxy, from_pretrained, Dots3Config, __init__，涉及 `DotsNoteOmniTokenizerProxy, from_pretrained, Dots3Config`。
- 代码 diff 细节:
  - `python/sglang/srt/models/dots3_common/modeling.py` added +2824/-0 (2824 lines)
  - `python/sglang/srt/models/dots3_common/dots_omni_audio.py` added +1027/-0 (1027 lines); hunks: -0,0 +1,1027; symbols: DotsWhisperConfig, __init__, RMSNorm, forward
  - `python/sglang/srt/models/dots3_common/dots_omni_vision.py` added +769/-0 (769 lines); hunks: -0,0 +1,769; symbols: VisionRotaryEmbedding, __init__, _compute_freqs, forward
  - `python/sglang/srt/configs/dots3.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: DotsNoteOmniTokenizerProxy, from_pretrained, Dots3Config, __init__
  - `python/sglang/srt/models/dots3_common/dots_omni_towers.py` added +240/-0 (240 lines); hunks: -0,0 +1,240; symbols: _read_json, load_omni_component_config, DotsNoteOmniVisionEncoder, __init__
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/dots3_common/modeling.py` added +2824/-0; `python/sglang/srt/models/dots3_common/dots_omni_audio.py` added +1027/-0; `python/sglang/srt/models/dots3_common/dots_omni_vision.py` added +769/-0; `python/sglang/srt/configs/dots3.py` added +243/-0; `python/sglang/srt/models/dots3_common/dots_omni_towers.py` added +240/-0; `python/sglang/srt/models/dots3_common/nextn.py` added +204/-0
  - docs: `docs/src/snippets/configs/rednote/dots3-note.jsx` modified +112/-60
- 验证与风险: diff 自带测试面 `test/registered/attention/unittests/swa/test_swa_out_cache_loc.py`, `test/registered/unit/function_call/test_dots_detector.py`, `test/registered/unit/layers/attention/test_dots_hybrid_backend.py`, `test/registered/unit/model_executor/test_mlp_sync_pad_unpad.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41198 - [Refactor] Move Step-3.5, GLM5-Next, Dots3, MiniMax-M3 and Qwen3.5 onto ffn_exit

- 链接: https://github.com/sgl-project/sglang/pull/41198
- 状态/时间: merged / 2026-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/dots3_common/modeling.py`；关联提交 `b7f6d04a9af1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+50/-219，可读 patch 402 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/dots3_common/modeling.py` modified +7/-24 (31 lines); hunks: -60,7 +60,6; -1623,30 +1622,14 @@ def forward(; symbols: forward, op_comm_prepare_attn，涉及 `forward, op_comm_prepare_attn`。
- 代码 diff 细节:
  - `python/sglang/srt/models/dots3_common/modeling.py` modified +7/-24 (31 lines); hunks: -60,7 +60,6; -1623,30 +1622,14 @@ def forward(; symbols: forward, op_comm_prepare_attn
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/dots3_common/modeling.py` modified +7/-24
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_step3p5_dense_reduce_scatter.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
