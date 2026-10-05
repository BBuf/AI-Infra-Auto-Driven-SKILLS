# vLLM dots3 (dots.note) 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `vllm/models/dots3_note/__init__.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255) |
| `vllm/models/dots3_note/common/__init__.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255) |
| `vllm/models/dots3_note/common/processor.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255) |
| `vllm/models/dots3_note/common/video.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255) |
| `vllm/models/dots3_note/nvidia/__init__.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255) |
| `vllm/models/dots3_note/nvidia/attention.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255), [#53460](https://github.com/vllm-project/vllm/pull/53460), [#53517](https://github.com/vllm-project/vllm/pull/53517) |
| `vllm/models/dots3_note/nvidia/audio.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255), [#53460](https://github.com/vllm-project/vllm/pull/53460) |
| `vllm/models/dots3_note/nvidia/audio_encoder.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255), [#52134](https://github.com/vllm-project/vllm/pull/52134) |
| `vllm/models/dots3_note/nvidia/model.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255), [#52172](https://github.com/vllm-project/vllm/pull/52172), [#53517](https://github.com/vllm-project/vllm/pull/53517) |
| `vllm/models/dots3_note/nvidia/mtp.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255), [#53517](https://github.com/vllm-project/vllm/pull/53517) |
| `vllm/models/dots3_note/nvidia/multimodal.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255), [#53460](https://github.com/vllm-project/vllm/pull/53460) |
| `vllm/models/dots3_note/nvidia/vision.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255), [#53460](https://github.com/vllm-project/vllm/pull/53460), [#53517](https://github.com/vllm-project/vllm/pull/53517) |
| `vllm/models/dots3_note/nvidia/vision_attention.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255), [#53460](https://github.com/vllm-project/vllm/pull/53460), [#53517](https://github.com/vllm-project/vllm/pull/53517) |
| `vllm/models/dots3_note/nvidia/vision_moe.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255) |
| `vllm/transformers_utils/configs/dots3_note.py` | [#51255](https://github.com/vllm-project/vllm/pull/51255) |

## PR 覆盖总览

- git 追溯 PR 数: 5
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 5
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-08-12 | [#51255](https://github.com/vllm-project/vllm/pull/51255) | merged | [Model] Add native Dots3 NOTE multimodal support | `vllm/models/dots3_note/common/processor.py`, `vllm/models/dots3_note/nvidia/attention.py`, `vllm/models/dots3_note/nvidia/audio_encoder.py` |
| 2026-08-13 | [#52134](https://github.com/vllm-project/vllm/pull/52134) | merged | [Docs] Fix `WhisperEncoderLayer.forward` docstring in `dots3_note` | `vllm/models/dots3_note/nvidia/audio_encoder.py` |
| 2026-08-13 | [#52172](https://github.com/vllm-project/vllm/pull/52172) | merged | [Bugfix] Disable sequence parallelism for Dots3 NOTE | `vllm/models/dots3_note/nvidia/model.py` |
| 2026-08-23 | [#53460](https://github.com/vllm-project/vllm/pull/53460) | merged | [Model] Fix KV cache layout and optimize Dots3 NOTE Omni encoders | `vllm/models/dots3_note/nvidia/vision.py`, `vllm/models/dots3_note/nvidia/audio.py`, `vllm/models/dots3_note/nvidia/vision_attention.py` |
| 2026-08-31 | [#53517](https://github.com/vllm-project/vllm/pull/53517) | merged | [Performance] Optimize Dots3 NOTE runtime | `vllm/models/dots3_note/nvidia/mtp.py`, `vllm/models/dots3_note/nvidia/model.py`, `vllm/models/dots3_note/nvidia/vision.py` |

## 逐 PR diff 审计卡

### PR #51255 - [Model] Add native Dots3 NOTE multimodal support

- 链接: https://github.com/vllm-project/vllm/pull/51255
- 状态/时间: merged / 2026-08-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/dots3_note/__init__.py`, `vllm/models/dots3_note/common/__init__.py`, `vllm/models/dots3_note/common/processor.py`, `vllm/models/dots3_note/common/video.py`, `vllm/models/dots3_note/nvidia/__init__.py` 等 15 个文件；关联提交 `9035151d6c9f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 28 个文件，+6468/-7，可读 patch 6673 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/dots3_note/common/processor.py` added +811/-0 (811 lines); hunks: -0,0 +1,811; symbols: load_note_config_section, Dots3NoteImageProcessor, __init__, factor，涉及 `load_note_config_section, Dots3NoteImageProcessor, __init__`；`vllm/models/dots3_note/nvidia/attention.py` added +807/-0 (807 lines); hunks: -0,0 +1,807; symbols: _gather_swa_kv_kernel, _apply_swa_score_mask_kernel, Dots3NoteDecodeMetadata, _SlidingWindowChunk，涉及 `_gather_swa_kv_kernel, _apply_swa_score_mask_kernel, Dots3NoteDecodeMetadata`；`vllm/models/dots3_note/nvidia/audio_encoder.py` added +736/-0 (736 lines); hunks: -0,0 +1,736; symbols: RMSNorm, __init__, forward, swiglu，涉及 `RMSNorm, __init__, forward`；`vllm/models/dots3_note/nvidia/model.py` added +684/-0 (684 lines); hunks: -0,0 +1,684; symbols: _padded_mlp_size, Dots3NoteMoE, __init__, forward，涉及 `_padded_mlp_size, Dots3NoteMoE, __init__`。
- 代码 diff 细节:
  - `vllm/models/dots3_note/common/processor.py` added +811/-0 (811 lines); hunks: -0,0 +1,811; symbols: load_note_config_section, Dots3NoteImageProcessor, __init__, factor
  - `vllm/models/dots3_note/nvidia/attention.py` added +807/-0 (807 lines); hunks: -0,0 +1,807; symbols: _gather_swa_kv_kernel, _apply_swa_score_mask_kernel, Dots3NoteDecodeMetadata, _SlidingWindowChunk
  - `vllm/models/dots3_note/nvidia/audio_encoder.py` added +736/-0 (736 lines); hunks: -0,0 +1,736; symbols: RMSNorm, __init__, forward, swiglu
  - `vllm/models/dots3_note/nvidia/model.py` added +684/-0 (684 lines); hunks: -0,0 +1,684; symbols: _padded_mlp_size, Dots3NoteMoE, __init__, forward
  - `vllm/models/dots3_note/nvidia/vision.py` added +677/-0 (677 lines); hunks: -0,0 +1,677; symbols: DotsMoEVitConfig, __init__, RMSNorm, forward
- 关键代码摘录:

```diff
diff -- vllm/models/dots3_note/common/processor.py
@@ -0,0 +1,811 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Multimodal preprocessing for Dots3Note checkpoints."""
+import math
+from collections.abc import Mapping, Sequence
+from functools import cached_property
diff -- vllm/models/dots3_note/nvidia/attention.py
@@ -0,0 +1,807 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Dots3 NOTE sliding-window MLA attention backends for Hopper.
+Prefill and mixed batches expand the latent cache and use FlashAttention-3
+varlen MHA. Decode-only batches use the Triton absorbed-MQA kernel.
+"""
diff -- vllm/models/dots3_note/nvidia/audio_encoder.py
@@ -0,0 +1,736 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/dots3_note/common/processor.py` added +811/-0; `vllm/models/dots3_note/nvidia/attention.py` added +807/-0; `vllm/models/dots3_note/nvidia/audio_encoder.py` added +736/-0; `vllm/models/dots3_note/nvidia/model.py` added +684/-0; `vllm/models/dots3_note/nvidia/vision.py` added +677/-0; `vllm/models/dots3_note/common/video.py` added +497/-0
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_rocm_aiter_mla_causal_verify_mask.py`, `tests/models/registry.py`, `tests/models/test_registry.py`, `tests/tool_parsers/test_dots_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #52134 - [Docs] Fix `WhisperEncoderLayer.forward` docstring in `dots3_note`

- 链接: https://github.com/vllm-project/vllm/pull/52134
- 状态/时间: merged / 2026-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/dots3_note/nvidia/audio_encoder.py`；关联提交 `903da602f386`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+18/-9，可读 patch 36 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/dots3_note/nvidia/audio_encoder.py` modified +18/-9 (27 lines); hunks: -345,17 +345,26 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/models/dots3_note/nvidia/audio_encoder.py` modified +18/-9 (27 lines); hunks: -345,17 +345,26 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/models/dots3_note/nvidia/audio_encoder.py
@@ -345,17 +345,26 @@ def forward(
-    ) -> torch.Tensor:
+    ) -> tuple[Any, ...]:
-            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
-            attention_mask (`torch.FloatTensor`): attention mask of size
-                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
-            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/dots3_note/nvidia/audio_encoder.py` modified +18/-9
- 验证与风险: runtime 路径改动集中在 `vllm/models/dots3_note/nvidia/audio_encoder.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #52172 - [Bugfix] Disable sequence parallelism for Dots3 NOTE

- 链接: https://github.com/vllm-project/vllm/pull/52172
- 状态/时间: merged / 2026-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/dots3_note/nvidia/model.py`；关联提交 `170592a931d9`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+3/-5，可读 patch 29 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/dots3_note/nvidia/model.py` modified +3/-5 (8 lines); hunks: -497,6 +497,7 @@ def __init__(; -536,11 +537,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/models/dots3_note/nvidia/model.py` modified +3/-5 (8 lines); hunks: -497,6 +497,7 @@ def __init__(; -536,11 +537,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/models/dots3_note/nvidia/model.py
@@ -497,6 +497,7 @@ def __init__(
+        self.use_sequence_parallel = False
@@ -536,11 +537,7 @@ def __init__(
-        self.use_sequence_parallel_moe = (
-            parallel_config.use_sequence_parallel_moe
-            and parallel_config.pipeline_parallel_size == 1
-            and isinstance(self.mlp, DeepseekV2MoE)
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/dots3_note/nvidia/model.py` modified +3/-5
- 验证与风险: runtime 路径改动集中在 `vllm/models/dots3_note/nvidia/model.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #53460 - [Model] Fix KV cache layout and optimize Dots3 NOTE Omni encoders

- 链接: https://github.com/vllm-project/vllm/pull/53460
- 状态/时间: merged / 2026-08-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/dots3_note/nvidia/attention.py`, `vllm/models/dots3_note/nvidia/audio.py`, `vllm/models/dots3_note/nvidia/multimodal.py`, `vllm/models/dots3_note/nvidia/vision.py`, `vllm/models/dots3_note/nvidia/vision_attention.py`；关联提交 `185cada36bb2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+211/-90，可读 patch 532 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/dots3_note/nvidia/vision.py` modified +56/-62 (118 lines); hunks: -2,6 +2,7; -15,10 +16,11; symbols: forward, VitForwardMeta, compile_block_modules, get_pos_ids_by_grid，涉及 `forward, VitForwardMeta, compile_block_modules`；`vllm/models/dots3_note/nvidia/audio.py` modified +96/-14 (110 lines); hunks: -21,6 +21,7; -126,6 +127,17 @@ def log_mel_spectrogram(audio, n_mels=128):; symbols: Dots3NoteAudioConfig, log_mel_spectrogram, batched_log_mel_spectrogram, compute_audio_token_length，涉及 `Dots3NoteAudioConfig, log_mel_spectrogram, batched_log_mel_spectrogram`；`vllm/models/dots3_note/nvidia/vision_attention.py` modified +29/-12 (41 lines); hunks: -36,15 +36,28 @@ def rotate_half(x: torch.Tensor) -> torch.Tensor:; -149,7 +162,7 @@ def __init__(self, params: VisionAttentionParams) -> None:; symbols: rotate_half, prepare_rotary_pos_emb_vision, apply_rotary_pos_emb_vision, __init__，涉及 `rotate_half, prepare_rotary_pos_emb_vision, apply_rotary_pos_emb_vision`；`vllm/models/dots3_note/nvidia/multimodal.py` modified +22/-2 (24 lines); hunks: -3,6 +3,7; -41,6 +42,21; symbols: _skip_linear_init, _noop_reset_parameters, __init__, _process_image_input，涉及 `_skip_linear_init, _noop_reset_parameters, __init__`。
- 代码 diff 细节:
  - `vllm/models/dots3_note/nvidia/vision.py` modified +56/-62 (118 lines); hunks: -2,6 +2,7; -15,10 +16,11; symbols: forward, VitForwardMeta, compile_block_modules, get_pos_ids_by_grid
  - `vllm/models/dots3_note/nvidia/audio.py` modified +96/-14 (110 lines); hunks: -21,6 +21,7; -126,6 +127,17 @@ def log_mel_spectrogram(audio, n_mels=128):; symbols: Dots3NoteAudioConfig, log_mel_spectrogram, batched_log_mel_spectrogram, compute_audio_token_length
  - `vllm/models/dots3_note/nvidia/vision_attention.py` modified +29/-12 (41 lines); hunks: -36,15 +36,28 @@ def rotate_half(x: torch.Tensor) -> torch.Tensor:; -149,7 +162,7 @@ def __init__(self, params: VisionAttentionParams) -> None:; symbols: rotate_half, prepare_rotary_pos_emb_vision, apply_rotary_pos_emb_vision, __init__
  - `vllm/models/dots3_note/nvidia/multimodal.py` modified +22/-2 (24 lines); hunks: -3,6 +3,7; -41,6 +42,21; symbols: _skip_linear_init, _noop_reset_parameters, __init__, _process_image_input
  - `vllm/models/dots3_note/nvidia/attention.py` modified +8/-0 (8 lines); hunks: -37,6 +37,7; -677,6 +678,13 @@ def forward_mqa(; symbols: forward_mqa, Dots3NotePaddedSparseBackend, supported_kv_cache_layouts, get_name
- 关键代码摘录:

```diff
diff -- vllm/models/dots3_note/nvidia/vision.py
@@ -2,6 +2,7 @@
+from dataclasses import dataclass
@@ -15,10 +16,11 @@
+    VisionRotaryPositionEmbedding,
-    prepare_seqlens_for_attention,
+    prepare_rotary_pos_emb_vision,
@@ -502,6 +504,15 @@ def forward(
diff -- vllm/models/dots3_note/nvidia/audio.py
@@ -21,6 +21,7 @@
+_AUDIO_FORWARD_MAX_SEGMENTS = 16
@@ -126,6 +127,17 @@ def log_mel_spectrogram(audio, n_mels=128):
+def batched_log_mel_spectrogram(audio, n_mels=128):
+    window = _hann_window(audio.device)
+    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
+    magnitudes = stft[..., :-1].abs() ** 2
diff -- vllm/models/dots3_note/nvidia/vision_attention.py
@@ -36,15 +36,28 @@ def rotate_half(x: torch.Tensor) -> torch.Tensor:
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/dots3_note/nvidia/vision.py` modified +56/-62; `vllm/models/dots3_note/nvidia/audio.py` modified +96/-14; `vllm/models/dots3_note/nvidia/vision_attention.py` modified +29/-12; `vllm/models/dots3_note/nvidia/multimodal.py` modified +22/-2; `vllm/models/dots3_note/nvidia/attention.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `vllm/models/dots3_note/nvidia/attention.py`, `vllm/models/dots3_note/nvidia/audio.py`, `vllm/models/dots3_note/nvidia/multimodal.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #53517 - [Performance] Optimize Dots3 NOTE runtime

- 链接: https://github.com/vllm-project/vllm/pull/53517
- 状态/时间: merged / 2026-08-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/dots3_note/nvidia/attention.py`, `vllm/models/dots3_note/nvidia/model.py`, `vllm/models/dots3_note/nvidia/mtp.py`, `vllm/models/dots3_note/nvidia/vision.py`, `vllm/models/dots3_note/nvidia/vision_attention.py`；关联提交 `da0b2d8b17c9`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+94/-85，可读 patch 379 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/dots3_note/nvidia/mtp.py` modified +37/-35 (72 lines); hunks: -9,10 +9,13; -25,7 +28,7; symbols: Dots3NoteMultiTokenPredictorLayer, forward, __init__，涉及 `Dots3NoteMultiTokenPredictorLayer, forward, __init__`；`vllm/models/dots3_note/nvidia/model.py` modified +25/-22 (47 lines); hunks: -19,6 +19,7; -73,6 +74,28 @@ def _padded_mlp_size(; symbols: _padded_mlp_size, _pad_dense_mlp_weight, Dots3NoteMoE, __init__，涉及 `_padded_mlp_size, _pad_dense_mlp_weight, Dots3NoteMoE`；`vllm/models/dots3_note/nvidia/vision.py` modified +7/-21 (28 lines); hunks: -15,6 +15,7; -107,23 +108,6 @@ def __init__(; symbols: __init__, RMSNorm, forward, extra_repr，涉及 `__init__, RMSNorm, forward`；`vllm/models/dots3_note/nvidia/vision_attention.py` modified +16/-6 (22 lines); hunks: -17,8 +17,11; -107,8 +110,8 @@ def forward(self, seqlen: int) -> torch.Tensor:; symbols: forward, _RMSNorm, VisionRMSNorm, __init__，涉及 `forward, _RMSNorm, VisionRMSNorm`。
- 代码 diff 细节:
  - `vllm/models/dots3_note/nvidia/mtp.py` modified +37/-35 (72 lines); hunks: -9,10 +9,13; -25,7 +28,7; symbols: Dots3NoteMultiTokenPredictorLayer, forward, __init__
  - `vllm/models/dots3_note/nvidia/model.py` modified +25/-22 (47 lines); hunks: -19,6 +19,7; -73,6 +74,28 @@ def _padded_mlp_size(; symbols: _padded_mlp_size, _pad_dense_mlp_weight, Dots3NoteMoE, __init__
  - `vllm/models/dots3_note/nvidia/vision.py` modified +7/-21 (28 lines); hunks: -15,6 +15,7; -107,23 +108,6 @@ def __init__(; symbols: __init__, RMSNorm, forward, extra_repr
  - `vllm/models/dots3_note/nvidia/vision_attention.py` modified +16/-6 (22 lines); hunks: -17,8 +17,11; -107,8 +110,8 @@ def forward(self, seqlen: int) -> torch.Tensor:; symbols: forward, _RMSNorm, VisionRMSNorm, __init__
  - `vllm/models/dots3_note/nvidia/attention.py` modified +7/-1 (8 lines); hunks: -7,6 +7,7; -19,7 +20,11; symbols: run_sliding_window, Dots3NoteMLAMetadataBuilder, __init__
- 关键代码摘录:

```diff
diff -- vllm/models/dots3_note/nvidia/mtp.py
@@ -9,10 +9,13 @@
-    get_tensor_model_parallel_world_size,
+from vllm.model_executor.layers.fused_embed_norm import (
+    fused_embed_eh_norm,
+    has_full_vocab_on_rank,
+)
@@ -25,7 +28,7 @@
diff -- vllm/models/dots3_note/nvidia/model.py
@@ -19,6 +19,7 @@
+from vllm.model_executor.layers.fused_embed_norm import has_full_vocab_on_rank
@@ -73,6 +74,28 @@ def _padded_mlp_size(
+def _pad_dense_mlp_weight(
+    name: str,
+    loaded_weight: torch.Tensor,
+    weight_block_size: list[int] | None,
diff -- vllm/models/dots3_note/nvidia/vision.py
@@ -15,6 +15,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/dots3_note/nvidia/mtp.py` modified +37/-35; `vllm/models/dots3_note/nvidia/model.py` modified +25/-22; `vllm/models/dots3_note/nvidia/vision.py` modified +7/-21; `vllm/models/dots3_note/nvidia/vision_attention.py` modified +16/-6; `vllm/models/dots3_note/nvidia/attention.py` modified +7/-1
- 验证与风险: runtime 路径改动集中在 `vllm/config/vllm.py`, `vllm/models/dots3_note/nvidia/attention.py`, `vllm/models/dots3_note/nvidia/model.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
