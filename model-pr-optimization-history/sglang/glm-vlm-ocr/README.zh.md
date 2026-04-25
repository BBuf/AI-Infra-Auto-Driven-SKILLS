# sglang GLM VLM/OCR 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/GLM/GLM-OCR.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/glm-ocr-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/models/glm4v.py` | [#9884](https://github.com/sgl-project/sglang/pull/9884), [#10147](https://github.com/sgl-project/sglang/pull/10147), [#17582](https://github.com/sgl-project/sglang/pull/17582), [#20033](https://github.com/sgl-project/sglang/pull/20033) |
| `python/sglang/srt/models/glm4v_moe.py` | [#20463](https://github.com/sgl-project/sglang/pull/20463), [#20740](https://github.com/sgl-project/sglang/pull/20740), [#21134](https://github.com/sgl-project/sglang/pull/21134) |
| `python/sglang/srt/models/glm_ocr.py` | [#17582](https://github.com/sgl-project/sglang/pull/17582), [#20463](https://github.com/sgl-project/sglang/pull/20463), [#20740](https://github.com/sgl-project/sglang/pull/20740), [#21134](https://github.com/sgl-project/sglang/pull/21134) |
| `python/sglang/srt/models/glm_ocr_nextn.py` | [#17582](https://github.com/sgl-project/sglang/pull/17582) |
| `python/sglang/srt/multimodal/processors/glm4v.py` | [#17582](https://github.com/sgl-project/sglang/pull/17582), [#18885](https://github.com/sgl-project/sglang/pull/18885) |

## PR 覆盖总览

- git 追溯 PR 数: 8
- 原文档显式引用补充 PR 数: 23
- 当前文档总 PR 数: 31
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-08-09 | [#8798](https://github.com/sgl-project/sglang/pull/8798) | merged | Support glm4.1v and glm4.5v | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-08-17 | [#9245](https://github.com/sgl-project/sglang/pull/9245) | merged | Set the default attention backend for GLM-4.5v to fa3 | `python/sglang/srt/utils.py` |
| 2025-08-18 | [#9059](https://github.com/sgl-project/sglang/pull/9059) | merged | [GLM4.1V and GLM4.5V] Add vision transformer num_dummy_head support: max tp=4 -> max tp=8 | `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/internvl.py` |
| 2025-08-19 | [#9349](https://github.com/sgl-project/sglang/pull/9349) | open | Add support for GLM 4.5V FP8 | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` |
| 2025-08-25 | [#9554](https://github.com/sgl-project/sglang/pull/9554) | merged | Fix GLM45v launch server cuda torch compile bug | `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-09-05 | [#9884](https://github.com/sgl-project/sglang/pull/9884) | merged | [Bug Fix] Fix Glm4vVisionBlock norm | `python/sglang/srt/models/glm4v.py` |
| 2025-09-08 | [#10147](https://github.com/sgl-project/sglang/pull/10147) | merged | Fix: (glm4v) Add missing field | `python/sglang/srt/models/glm4v.py` |
| 2025-09-14 | [#10228](https://github.com/sgl-project/sglang/pull/10228) | merged | Add self.capture_aux_hidden_states For GLM-4.5V | `python/sglang/srt/models/glm4v_moe.py` |
| 2025-10-03 | [#11166](https://github.com/sgl-project/sglang/pull/11166) | merged | Tiny move files to utils folder | `test/srt/test_tokenizer_manager.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py` |
| 2025-10-10 | [#11388](https://github.com/sgl-project/sglang/pull/11388) | merged | Replace pad with cat for better performance | `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-10-22 | [#11922](https://github.com/sgl-project/sglang/pull/11922) | merged | [lint] improve ruff check | `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` |
| 2025-10-27 | [#12117](https://github.com/sgl-project/sglang/pull/12117) | merged | GLM-4-0414 and GLM-4.1V Code Refactor | `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-11-16 | [#13228](https://github.com/sgl-project/sglang/pull/13228) | merged | Cleanup vision attention related codes | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2025-12-05 | [#14097](https://github.com/sgl-project/sglang/pull/14097) | merged | support GLM-V vision model dp | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-12-08 | [#14662](https://github.com/sgl-project/sglang/pull/14662) | open | [Glm46v] support ktransformers | `python/sglang/srt/models/glm4v_moe.py` |
| 2025-12-10 | [#14720](https://github.com/sgl-project/sglang/pull/14720) | merged | [GLM-4.6V] Support Pipeline Parallelism for GLM-4.6V & GLM-4.1V | `python/sglang/srt/models/glm4v.py`, `test/srt/test_pp_single_node.py`, `python/sglang/test/test_utils.py` |
| 2025-12-12 | [#14927](https://github.com/sgl-project/sglang/pull/14927) | merged | [CI]add nightly CI for glm4v_moe arch model | `test/nightly/test_vlms_mmmu_eval.py` |
| 2025-12-13 | [#14998](https://github.com/sgl-project/sglang/pull/14998) | merged | add transformers version validation for glm-4.6v moe models | `python/sglang/srt/configs/model_config.py` |
| 2025-12-18 | [#15205](https://github.com/sgl-project/sglang/pull/15205) | merged | [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py` |
| 2026-01-04 | [#15434](https://github.com/sgl-project/sglang/pull/15434) | merged | Convert cu_seqlens to CPU for npu_flash_attention_unpad operator | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/paddleocr_vl.py` |
| 2026-01-27 | [#17582](https://github.com/sgl-project/sglang/pull/17582) | merged | [GLM-OCR] Support GLM-OCR Model | `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/multimodal/processors/glm4v.py` |
| 2026-02-01 | [#17420](https://github.com/sgl-project/sglang/pull/17420) | merged | [VLM] Optimize get_rope_index for GLM4v | `python/sglang/srt/layers/rotary_embedding.py`, `benchmark/bench_rope/benchmark_rope_index.py` |
| 2026-02-16 | [#18885](https://github.com/sgl-project/sglang/pull/18885) | merged | Fix GLM-4V processor registration when glm_ocr is unavailable | `python/sglang/srt/multimodal/processors/glm4v.py` |
| 2026-03-03 | [#19728](https://github.com/sgl-project/sglang/pull/19728) | open | Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `test/registered/moe/test_fused_moe.py` |
| 2026-03-08 | [#20033](https://github.com/sgl-project/sglang/pull/20033) | merged | [VLM] Replace conv3d proj with linear for GLM4V | `python/sglang/srt/models/glm4v.py` |
| 2026-03-14 | [#20463](https://github.com/sgl-project/sglang/pull/20463) | merged | [Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr | `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py` |
| 2026-03-15 | [#20282](https://github.com/sgl-project/sglang/pull/20282) | merged | Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug | `python/sglang/srt/layers/conv.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/pixtral.py` |
| 2026-03-18 | [#20740](https://github.com/sgl-project/sglang/pull/20740) | merged | Revert "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr" | `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py` |
| 2026-03-23 | [#21134](https://github.com/sgl-project/sglang/pull/21134) | merged | [Bug Fix] GLM-V / GLM-OCR: field detection for transformers 5.x and MTP omission fix | `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py` |
| 2026-04-01 | [#17122](https://github.com/sgl-project/sglang/pull/17122) | merged | [bugfix]GLM-4V model | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py` |
| 2026-04-16 | [#22961](https://github.com/sgl-project/sglang/pull/22961) | open | [NPU] Support GLM-4.5V | `python/sglang/srt/models/glm4_moe.py` |

## 逐 PR diff 审计卡

### PR #8798 - Support glm4.1v and glm4.5v

- 链接: https://github.com/sgl-project/sglang/pull/8798
- 状态/时间: merged / 2025-08-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 21 个文件，+1584/-19，可读 patch 1822 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「Support glm4.1v and glm4.5v」，变更集中在 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`。PR 描述补充为：## Motivation Close https://github.com/sgl-project/sglang/issues/7993. Adapted from https://github.com/sgl-project/sglang/pull/8015 and depend on https://github.com/huggingface/...
- 实现要点: `python/sglang/srt/models/glm4v.py` added +589/-0 (589 lines); hunks: -0,0 +1,589; symbols: Glm4vRMSNorm, forward, Glm4vVisionMLP, __init__，涉及 `Glm4vRMSNorm, forward, Glm4vVisionMLP`；`python/sglang/srt/models/glm4v_moe.py` added +400/-0 (400 lines); hunks: -0,0 +1,400; symbols: Glm4vMoeForConditionalGeneration, __init__, determine_num_fused_shared_experts, load_weights，涉及 `Glm4vMoeForConditionalGeneration, __init__, determine_num_fused_shared_experts`；`python/sglang/srt/layers/rotary_embedding.py` modified +230/-1 (231 lines); hunks: -1,6 +1,7; -946,7 +947,37 @@ def __init__(; symbols: __init__, forward, get_rope_index, get_rope_index_glm4v，涉及 `__init__, forward, get_rope_index`；`python/sglang/srt/multimodal/processors/glm4v.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: Glm4vImageProcessor, __init__, preprocess_video, process_mm_data_async，涉及 `Glm4vImageProcessor, __init__, preprocess_video`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v.py` added +589/-0 (589 lines); hunks: -0,0 +1,589; symbols: Glm4vRMSNorm, forward, Glm4vVisionMLP, __init__
  - `python/sglang/srt/models/glm4v_moe.py` added +400/-0 (400 lines); hunks: -0,0 +1,400; symbols: Glm4vMoeForConditionalGeneration, __init__, determine_num_fused_shared_experts, load_weights
  - `python/sglang/srt/layers/rotary_embedding.py` modified +230/-1 (231 lines); hunks: -1,6 +1,7; -946,7 +947,37 @@ def __init__(; symbols: __init__, forward, get_rope_index, get_rope_index_glm4v
  - `python/sglang/srt/multimodal/processors/glm4v.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: Glm4vImageProcessor, __init__, preprocess_video, process_mm_data_async
  - `test/srt/test_function_call_parser.py` modified +18/-2 (20 lines); hunks: -497,6 +497,17 @@ def setUp(self):; -630,16 +641,21 @@ def test_glm45_detector_ebnf(self):; symbols: setUp, test_glm45_detector_ebnf
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -0,0 +1,589 @@
+import logging
+from functools import lru_cache, partial
+from typing import Iterable, List, Optional, Tuple
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -0,0 +1,400 @@
+import logging
+from functools import lru_cache
+from typing import Iterable, Optional, Tuple
+import torch
+import torch.nn as nn
+from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4vMoeConfig
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -1,6 +1,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v.py` added +589/-0; `python/sglang/srt/models/glm4v_moe.py` added +400/-0; `python/sglang/srt/layers/rotary_embedding.py` modified +230/-1; `python/sglang/srt/multimodal/processors/glm4v.py` added +132/-0; `python/sglang/srt/multimodal/processors/base_processor.py` modified +9/-3; `python/sglang/srt/models/glm4.py` modified +6/-0
  - tests: `test/srt/test_function_call_parser.py` modified +18/-2
- 验证与风险: diff 自带测试面 `test/srt/openai_server/function_call/test_openai_function_calling.py`, `test/srt/test_function_call_parser.py`, `test/srt/test_jinja_template_utils.py`, `test/srt/test_vision_openai_server_b.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #9245 - Set the default attention backend for GLM-4.5v to fa3

- 链接: https://github.com/sgl-project/sglang/pull/9245
- 状态/时间: merged / 2025-08-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补强部署文档、测试或 CI 验证面，标题为「Set the default attention backend for GLM-4.5v to fa3」，变更集中在 `python/sglang/srt/utils.py`。PR 描述补充为：As suggested in https://huggingface.co/zai-org/GLM-4.5V#sglang ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code...
- 实现要点: `python/sglang/srt/utils.py` modified +1/-0 (1 lines); hunks: -2343,6 +2343,7 @@ def is_fa3_default_architecture(hf_config):; symbols: is_fa3_default_architecture，涉及 `is_fa3_default_architecture`。
- 代码 diff 细节:
  - `python/sglang/srt/utils.py` modified +1/-0 (1 lines); hunks: -2343,6 +2343,7 @@ def is_fa3_default_architecture(hf_config):; symbols: is_fa3_default_architecture
- 关键代码摘录:

```diff
diff -- python/sglang/srt/utils.py
@@ -2343,6 +2343,7 @@ def is_fa3_default_architecture(hf_config):
+        "Glm4vMoeForConditionalGeneration",
```

- 已读文件:
  - runtime: `python/sglang/srt/utils.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9059 - [GLM4.1V and GLM4.5V] Add vision transformer num_dummy_head support: max tp=4 -> max tp=8

- 链接: https://github.com/sgl-project/sglang/pull/9059
- 状态/时间: merged / 2025-08-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+150/-102，可读 patch 423 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「[GLM4.1V and GLM4.5V] Add vision transformer num_dummy_head support: max tp=4 -> max tp=8」，变更集中在 `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/internvl.py`。PR 描述补充为：## Motivation In some GLM4v models, `num_key_value_heads=8` in text attention and `num_heads=12` in vision attention so `tp=4` is the max supported tp_size today `(num_key_value...
- 实现要点: `python/sglang/srt/layers/attention/vision_utils.py` added +65/-0 (65 lines); hunks: -0,0 +1,65; symbols: update_vit_attn_dummy_heads_config, pad_vit_attn_dummy_heads，涉及 `update_vit_attn_dummy_heads_config, pad_vit_attn_dummy_heads`；`python/sglang/srt/models/glm4v.py` modified +52/-1 (53 lines); hunks: -9,6 +9,7; -91,6 +92,7 @@ def __init__(; symbols: __init__, get_video_feature, _update_hf_config, _pad_vit_attn_dummy_heads，涉及 `__init__, get_video_feature, _update_hf_config`；`python/sglang/srt/models/internvl.py` modified +4/-49 (53 lines); hunks: -10,7 +10,7; -412,7 +412,7 @@ def __init__(; symbols: __init__, _update_vision_config, pixel_shuffle, pad_input_ids，涉及 `__init__, _update_vision_config, pixel_shuffle`；`python/sglang/srt/models/interns1.py` modified +5/-46 (51 lines); hunks: -4,7 +4,7; -35,7 +35,7 @@ def __init__(; symbols: __init__, _update_hf_config, pixel_shuffle, pad_input_ids，涉及 `__init__, _update_hf_config, pixel_shuffle`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/vision_utils.py` added +65/-0 (65 lines); hunks: -0,0 +1,65; symbols: update_vit_attn_dummy_heads_config, pad_vit_attn_dummy_heads
  - `python/sglang/srt/models/glm4v.py` modified +52/-1 (53 lines); hunks: -9,6 +9,7; -91,6 +92,7 @@ def __init__(; symbols: __init__, get_video_feature, _update_hf_config, _pad_vit_attn_dummy_heads
  - `python/sglang/srt/models/internvl.py` modified +4/-49 (53 lines); hunks: -10,7 +10,7; -412,7 +412,7 @@ def __init__(; symbols: __init__, _update_vision_config, pixel_shuffle, pad_input_ids
  - `python/sglang/srt/models/interns1.py` modified +5/-46 (51 lines); hunks: -4,7 +4,7; -35,7 +35,7 @@ def __init__(; symbols: __init__, _update_hf_config, pixel_shuffle, pad_input_ids
  - `python/sglang/srt/models/glm4v_moe.py` modified +6/-0 (6 lines); hunks: -11,6 +11,7; -40,6 +41,7 @@ def __init__(; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/vision_utils.py
@@ -0,0 +1,65 @@
+"""Utility functions for vision attention layers."""
+import torch
+from sglang.srt.layers.dp_attention import get_attention_tp_size
+def update_vit_attn_dummy_heads_config(config):
+    """Update HF config to ensure vision attention num_attention_heads is divisible by tp_size"""
+    tp_size = get_attention_tp_size()
diff -- python/sglang/srt/models/glm4v.py
@@ -9,6 +9,7 @@
+from sglang.srt.layers.attention import vision_utils
@@ -91,6 +92,7 @@ def __init__(
+            num_dummy_heads=config.num_dummy_heads,
@@ -469,7 +471,7 @@ def __init__(
+        vision_utils.update_vit_attn_dummy_heads_config(self.config)
@@ -537,6 +539,51 @@ def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
diff -- python/sglang/srt/models/internvl.py
@@ -10,7 +10,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/vision_utils.py` added +65/-0; `python/sglang/srt/models/glm4v.py` modified +52/-1; `python/sglang/srt/models/internvl.py` modified +4/-49; `python/sglang/srt/models/interns1.py` modified +5/-46; `python/sglang/srt/models/glm4v_moe.py` modified +6/-0; `python/sglang/srt/models/qwen2_5_vl.py` modified +2/-0
  - other: `benchmark/mmmu/bench_hf.py` modified +6/-2; `benchmark/mmmu/bench_sglang.py` modified +6/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9349 - Add support for GLM 4.5V FP8

- 链接: https://github.com/sgl-project/sglang/pull/9349
- 状态/时间: open / 2025-08-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+153/-4，可读 patch 165 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「Add support for GLM 4.5V FP8」，变更集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`。PR 描述补充为：## Motivation Add moe kernel generation support for GLM 4.5V FP8 ## Modifications Write according to the GLM 4.5V FP8 configuration. ## Accuracy Tests N/A ## Benchmarking and Pr...
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146；`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +7/-4 (11 lines); hunks: -441,10 +441,13 @@ def main(args: argparse.Namespace):; symbols: main，涉及 `main`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +7/-4 (11 lines); hunks: -441,10 +441,13 @@ def main(args: argparse.Namespace):; symbols: main
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
diff -- benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py
@@ -441,10 +441,13 @@ def main(args: argparse.Namespace):
-    elif config.architectures[0] in ["Glm4MoeForCausalLM"]:
-        E = config.n_routed_experts
-        topk = config.num_experts_per_tok
-        intermediate_size = config.moe_intermediate_size
+    elif config.architectures[0] in ["Glm4MoeForCausalLM", "Glm4vMoeForConditionalGeneration"]:
+        cfg_source = config
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json` added +146/-0
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +7/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9554 - Fix GLM45v launch server cuda torch compile bug

- 链接: https://github.com/sgl-project/sglang/pull/9554
- 状态/时间: merged / 2025-08-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 修复已暴露的启动、加载、解析或数值问题，标题为「Fix GLM45v launch server cuda torch compile bug」，变更集中在 `python/sglang/srt/models/qwen2_5_vl.py`。PR 描述补充为：## Motivation After recent sglang code changes, glm45v launch server would fail with this error during torch compile ## Modifications - From https://discuss.pytorch.org/t/index-...
- 实现要点: `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0 (1 lines); hunks: -526,6 +526,7 @@ def get_video_feature(self, items: List[MultimodalDataItem])...; symbols: get_video_feature, get_input_embeddings, forward，涉及 `get_video_feature, get_input_embeddings, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0 (1 lines); hunks: -526,6 +526,7 @@ def get_video_feature(self, items: List[MultimodalDataItem])...; symbols: get_video_feature, get_input_embeddings, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -526,6 +526,7 @@ def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
+    @torch.no_grad()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9884 - [Bug Fix] Fix Glm4vVisionBlock norm

- 链接: https://github.com/sgl-project/sglang/pull/9884
- 状态/时间: merged / 2025-09-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4v.py`；关联提交 `0f6ac5e21db0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-4，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bug Fix] Fix Glm4vVisionBlock norm」，变更集中在 `python/sglang/srt/models/glm4v.py`。PR 描述补充为：## Motivation This PR fixes a runtime crash in `Glm4vVisionBlock` when running vision prefill. After #9709, `Qwen2_5_VisionBlock` calls `norm2(x2d, residual=attn2d)` and expects...
- 实现要点: `python/sglang/srt/models/glm4v.py` modified +1/-2 (3 lines); hunks: -93,9 +93,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v.py` modified +1/-2 (3 lines); hunks: -93,9 +93,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -93,9 +93,8 @@ def __init__(
+            rms_norm_eps=config.rms_norm_eps,
-        self.norm1 = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
-        self.norm2 = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10147 - Fix: (glm4v) Add missing field

- 链接: https://github.com/sgl-project/sglang/pull/10147
- 状态/时间: merged / 2025-09-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4v.py`；关联提交 `8116804e4f6e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「Fix: (glm4v) Add missing field」，变更集中在 `python/sglang/srt/models/glm4v.py`。PR 描述补充为：## Motivation Fix: https://github.com/sgl-project/sglang/actions/runs/17539177242/job/49807888401 ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist -...
- 实现要点: `python/sglang/srt/models/glm4v.py` modified +3/-0 (3 lines); hunks: -497,6 +497,9 @@ def __init__(; symbols: __init__, get_image_feature，涉及 `__init__, get_image_feature`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v.py` modified +3/-0 (3 lines); hunks: -497,6 +497,9 @@ def __init__(; symbols: __init__, get_image_feature
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -497,6 +497,9 @@ def __init__(
+        # For EAGLE3 support
+        self.capture_aux_hidden_states = False
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4v.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10228 - Add self.capture_aux_hidden_states For GLM-4.5V

- 链接: https://github.com/sgl-project/sglang/pull/10228
- 状态/时间: merged / 2025-09-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「Add self.capture_aux_hidden_states For GLM-4.5V」，变更集中在 `python/sglang/srt/models/glm4v_moe.py`。PR 正文没有提供额外背景，判断主要来自标题、文件列表和 patch。
- 实现要点: `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunks: -74,6 +74,9 @@ def __init__(; symbols: __init__, determine_num_fused_shared_experts，涉及 `__init__, determine_num_fused_shared_experts`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunks: -74,6 +74,9 @@ def __init__(; symbols: __init__, determine_num_fused_shared_experts
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -74,6 +74,9 @@ def __init__(
+        # For EAGLE3 support
+        self.capture_aux_hidden_states = False
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4v_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11166 - Tiny move files to utils folder

- 链接: https://github.com/sgl-project/sglang/pull/11166
- 状态/时间: merged / 2025-10-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 66 个文件，+91/-79，可读 patch 794 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补强部署文档、测试或 CI 验证面，标题为「Tiny move files to utils folder」，变更集中在 `test/srt/test_tokenizer_manager.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py`。PR 描述补充为：## Motivation cc @merrymercy ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. -...
- 实现要点: `test/srt/test_tokenizer_manager.py` modified +12/-4 (16 lines); hunks: -31,7 +31,9 @@ def setUp(self):; -125,7 +127,9 @@ def setUp(self):; symbols: setUp，涉及 `setUp`；`python/sglang/srt/managers/tokenizer_manager.py` modified +5/-5 (10 lines); hunks: -43,11 +43,6; -99,6 +94,11；`python/sglang/srt/configs/model_config.py` modified +4/-4 (8 lines); hunks: -23,16 +23,16；`python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -29,7 +29,6; -115,7 +114,6。
- 代码 diff 细节:
  - `test/srt/test_tokenizer_manager.py` modified +12/-4 (16 lines); hunks: -31,7 +31,9 @@ def setUp(self):; -125,7 +127,9 @@ def setUp(self):; symbols: setUp
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +5/-5 (10 lines); hunks: -43,11 +43,6; -99,6 +94,11
  - `python/sglang/srt/configs/model_config.py` modified +4/-4 (8 lines); hunks: -23,16 +23,16
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -29,7 +29,6; -115,7 +114,6
  - `test/srt/test_tokenizer_batch_encode.py` modified +3/-1 (4 lines); hunks: -34,7 +34,9 @@ def setUp(self):; symbols: setUp
- 关键代码摘录:

```diff
diff -- test/srt/test_tokenizer_manager.py
@@ -31,7 +31,9 @@ def setUp(self):
-        ), patch("sglang.srt.hf_transformers_utils.get_tokenizer") as mock_tokenizer:
+        ), patch(
+            "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
+        ) as mock_tokenizer:
@@ -125,7 +127,9 @@ def setUp(self):
-        ), patch("sglang.srt.hf_transformers_utils.get_tokenizer") as mock_tokenizer:
diff -- python/sglang/srt/managers/tokenizer_manager.py
@@ -43,11 +43,6 @@
-from sglang.srt.hf_transformers_utils import (
-    get_processor,
-    get_tokenizer,
-    get_tokenizer_from_processor,
-)
@@ -99,6 +94,11 @@
diff -- python/sglang/srt/configs/model_config.py
@@ -23,16 +23,16 @@
```

- 已读文件:
  - tests: `test/srt/test_tokenizer_manager.py` modified +12/-4; `test/srt/test_tokenizer_batch_encode.py` modified +3/-1
  - runtime: `python/sglang/srt/managers/tokenizer_manager.py` modified +5/-5; `python/sglang/srt/configs/model_config.py` modified +4/-4; `python/sglang/srt/model_executor/model_runner.py` modified +2/-2; `python/sglang/srt/managers/detokenizer_manager.py` modified +1/-1; `python/sglang/srt/model_executor/cpu_graph_runner.py` modified +1/-1; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +1/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/runners.py`, `python/sglang/test/test_programs.py`, `python/sglang/test/test_utils.py`, `test/srt/openai_server/basic/test_openai_server.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11388 - Replace pad with cat for better performance

- 链接: https://github.com/sgl-project/sglang/pull/11388
- 状态/时间: merged / 2025-10-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+5/-5，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 优化关键推理路径或后端选择，标题为「Replace pad with cat for better performance」，变更集中在 `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`。PR 描述补充为：## Motivation Using concatenation tends to be faster than padding. ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according...
- 实现要点: `python/sglang/srt/models/dots_vlm_vit.py` modified +1/-1 (2 lines); hunks: -323,7 +323,7 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/glm4v.py` modified +1/-1 (2 lines); hunks: -434,7 +434,7 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -...; symbols: forward，涉及 `forward`；`python/sglang/srt/models/qwen2_5_vl.py` modified +1/-1 (2 lines); hunks: -436,7 +436,7 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -407,7 +407,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/dots_vlm_vit.py` modified +1/-1 (2 lines); hunks: -323,7 +323,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/glm4v.py` modified +1/-1 (2 lines); hunks: -434,7 +434,7 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -...; symbols: forward
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-1 (2 lines); hunks: -436,7 +436,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -407,7 +407,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-1 (2 lines); hunks: -458,7 +458,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/dots_vlm_vit.py
@@ -323,7 +323,7 @@ def forward(
-        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
+        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
diff -- python/sglang/srt/models/glm4v.py
@@ -434,7 +434,7 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
-        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
+        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -436,7 +436,7 @@ def forward(
-        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
+        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
diff -- python/sglang/srt/models/qwen2_vl.py
@@ -407,7 +407,7 @@ def forward(
-        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
+        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -458,7 +458,7 @@ def forward(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/dots_vlm_vit.py` modified +1/-1; `python/sglang/srt/models/glm4v.py` modified +1/-1; `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-1; `python/sglang/srt/models/qwen2_vl.py` modified +1/-1; `python/sglang/srt/models/qwen3_vl.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11922 - [lint] improve ruff check

- 链接: https://github.com/sgl-project/sglang/pull/11922
- 状态/时间: merged / 2025-10-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+73/-31，可读 patch 354 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 修复已暴露的启动、加载、解析或数值问题，标题为「[lint] improve ruff check」，变更集中在 `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`。PR 描述补充为：Prev PR #11685 added `python/sglang` to the Ruff lint scope, but the pre-commit args were written as: These arguments are interpreted as `--select=F401 F821 --fixable=F401`, whi...
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +20/-19 (39 lines); hunks: -13,7 +13,8; -119,14 +120,14 @@ def triton_kernel_fused_experts(; symbols: triton_kernel_fused_experts, triton_kernel_fused_experts_with_bias，涉及 `triton_kernel_fused_experts, triton_kernel_fused_experts_with_bias`；`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +7/-0 (7 lines); hunks: -44,6 +44,13；`python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +4/-1 (5 lines); hunks: -38,6 +38,9; -66,7 +69,7 @@ class PrefillMetadata:; symbols: PrefillMetadata, FlashInferMhaChunkKVRunner, __init__，涉及 `PrefillMetadata, FlashInferMhaChunkKVRunner, __init__`；`python/sglang/srt/models/opt.py` modified +4/-0 (4 lines); hunks: -13,6 +13,7; -46,6 +47,9; symbols: get_activation，涉及 `get_activation`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +20/-19 (39 lines); hunks: -13,7 +13,8; -119,14 +120,14 @@ def triton_kernel_fused_experts(; symbols: triton_kernel_fused_experts, triton_kernel_fused_experts_with_bias
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +7/-0 (7 lines); hunks: -44,6 +44,13
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +4/-1 (5 lines); hunks: -38,6 +38,9; -66,7 +69,7 @@ class PrefillMetadata:; symbols: PrefillMetadata, FlashInferMhaChunkKVRunner, __init__
  - `python/sglang/srt/models/opt.py` modified +4/-0 (4 lines); hunks: -13,6 +13,7; -46,6 +47,9; symbols: get_activation
  - `python/sglang/srt/entrypoints/openai/serving_responses.py` modified +2/-1 (3 lines); hunks: -14,6 +14,7; -1061,7 +1062,7 @@ def _send_event(event):; symbols: _send_event
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py
@@ -13,7 +13,8 @@
-from triton_kernels.numerics import InFlexData
+from triton_kernels.numerics import InFlexData, MicroscalingCtx
+from triton_kernels.quantization import downcast_to_mxfp
@@ -119,14 +120,14 @@ def triton_kernel_fused_experts(
-    assert use_fp8_w8a8 == False, "use_fp8_w8a8 is not supported"
-    assert per_channel_quant == False, "per_channel_quant is not supported"
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
@@ -44,6 +44,13 @@
+    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_24 import (
+        CompressedTensors24,
+    )
+    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_sparse24 import (
+        W4A16SPARSE24_SUPPORTED_BITS,
+        CompressedTensorsW4A16Sparse24,
diff -- python/sglang/srt/layers/attention/flashinfer_mla_backend.py
@@ -38,6 +38,9 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +20/-19; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +7/-0; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +4/-1; `python/sglang/srt/models/opt.py` modified +4/-0; `python/sglang/srt/entrypoints/openai/serving_responses.py` modified +2/-1; `python/sglang/srt/models/deepseek_v2.py` modified +2/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/few_shot_gsm8k_engine.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12117 - GLM-4-0414 and GLM-4.1V Code Refactor

- 链接: https://github.com/sgl-project/sglang/pull/12117
- 状态/时间: merged / 2025-10-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+679/-173，可读 patch 1187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 调整模型相关实现，标题为「GLM-4-0414 and GLM-4.1V Code Refactor」，变更集中在 `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py`。PR 描述补充为：Used new interfaces, including the addition of PPMissingLayer, and discarded some useless old code
- 实现要点: `python/sglang/srt/models/glm4.py` modified +391/-77 (468 lines); hunks: -15,46 +15,119; -63,27 +136,30 @@ def __init__(; symbols: Glm4MLP, __init__, forward, Glm4Attention，涉及 `Glm4MLP, __init__, forward`；`python/sglang/srt/models/glm4v.py` modified +196/-55 (251 lines); hunks: -1,15 +1,35; -20,13 +40,14; symbols: __init__, forward, Glm4vVisionBlock，涉及 `__init__, forward, Glm4vVisionBlock`；`python/sglang/srt/layers/rotary_embedding.py` modified +92/-40 (132 lines); hunks: -1070,6 +1070,7 @@ def _triton_mrope_forward(; -1124,51 +1125,99 @@ def _triton_mrope_forward(; symbols: _triton_mrope_forward, triton_mrope，涉及 `_triton_mrope_forward, triton_mrope`；`python/sglang/srt/models/glm4v_moe.py` modified +0/-1 (1 lines); hunks: -53,7 +53,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4.py` modified +391/-77 (468 lines); hunks: -15,46 +15,119; -63,27 +136,30 @@ def __init__(; symbols: Glm4MLP, __init__, forward, Glm4Attention
  - `python/sglang/srt/models/glm4v.py` modified +196/-55 (251 lines); hunks: -1,15 +1,35; -20,13 +40,14; symbols: __init__, forward, Glm4vVisionBlock
  - `python/sglang/srt/layers/rotary_embedding.py` modified +92/-40 (132 lines); hunks: -1070,6 +1070,7 @@ def _triton_mrope_forward(; -1124,51 +1125,99 @@ def _triton_mrope_forward(; symbols: _triton_mrope_forward, triton_mrope
  - `python/sglang/srt/models/glm4v_moe.py` modified +0/-1 (1 lines); hunks: -53,7 +53,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4.py
@@ -15,46 +15,119 @@
-"""Inference-only GLM4 model compatible with THUDM weights."""
+"""Inference-only GLM-4-0414 model compatible with HuggingFace weights."""
-from typing import Iterable, List, Optional, Tuple, Union
+import logging
+from typing import Any, Dict, Iterable, Optional, Tuple, Union
-from transformers import Glm4Config
diff -- python/sglang/srt/models/glm4v.py
@@ -1,15 +1,35 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -1070,6 +1070,7 @@ def _triton_mrope_forward(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4.py` modified +391/-77; `python/sglang/srt/models/glm4v.py` modified +196/-55; `python/sglang/srt/layers/rotary_embedding.py` modified +92/-40; `python/sglang/srt/models/glm4v_moe.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13228 - Cleanup vision attention related codes

- 链接: https://github.com/sgl-project/sglang/pull/13228
- 状态/时间: merged / 2025-11-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+4/-142，可读 patch 378 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「Cleanup vision attention related codes」，变更集中在 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`。PR 描述补充为：## Motivation Most of the codes for vision attention are hard coded or dead codes. This PR removes them, supports the auto set of `--mm-attention-backend triton_attn` on B200. #...
- 实现要点: `python/sglang/srt/models/glm4v.py` modified +1/-26 (27 lines); hunks: -104,7 +104,6 @@ def __init__(; -114,37 +113,13 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen2_5_vl.py` modified +1/-26 (27 lines); hunks: -111,7 +111,6 @@ def __init__(; -121,37 +120,13 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen3_vl.py` modified +1/-23 (24 lines); hunks: -130,7 +130,6 @@ def __init__(; -140,33 +139,13 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/clip.py` modified +0/-13 (13 lines); hunks: -141,7 +141,6 @@ def __init__(; -150,22 +149,11 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v.py` modified +1/-26 (27 lines); hunks: -104,7 +104,6 @@ def __init__(; -114,37 +113,13 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-26 (27 lines); hunks: -111,7 +111,6 @@ def __init__(; -121,37 +120,13 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-23 (24 lines); hunks: -130,7 +130,6 @@ def __init__(; -140,33 +139,13 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/clip.py` modified +0/-13 (13 lines); hunks: -141,7 +141,6 @@ def __init__(; -150,22 +149,11 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen2_vl.py` modified +0/-13 (13 lines); hunks: -127,7 +127,6 @@ def __init__(; -137,23 +136,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -104,7 +104,6 @@ def __init__(
-        attn_implementation: Optional[str] = None,
@@ -114,37 +113,13 @@ def __init__(
-        if attn_implementation is None:
-            softmax_in_single_precision = False
-            qkv_backend = None
-            flatten_batch = True
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -111,7 +111,6 @@ def __init__(
-        attn_implementation: Optional[str] = None,
@@ -121,37 +120,13 @@ def __init__(
-        if attn_implementation is None:
-            softmax_in_single_precision = False
-            qkv_backend = None
-            flatten_batch = True
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -130,7 +130,6 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +1/-26; `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-26; `python/sglang/srt/models/qwen3_vl.py` modified +1/-23; `python/sglang/srt/models/clip.py` modified +0/-13; `python/sglang/srt/models/qwen2_vl.py` modified +0/-13; `python/sglang/srt/models/siglip.py` modified +0/-13
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/clip.py`, `python/sglang/srt/models/deepseek_janus_pro.py`, `python/sglang/srt/models/dots_vlm_vit.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14097 - support GLM-V vision model dp

- 链接: https://github.com/sgl-project/sglang/pull/14097
- 状态/时间: merged / 2025-12-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+91/-52，可读 patch 329 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「support GLM-V vision model dp」，变更集中在 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py`。PR 描述补充为：Based on https://github.com/sgl-project/sglang/pull/13126 and https://github.com/sgl-project/sglang/pull/13724, add support for the GLM-V vision model DP.
- 实现要点: `python/sglang/srt/models/glm4v.py` modified +84/-50 (134 lines); hunks: -27,18 +27,24; -48,6 +54,8; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/models/glm4.py` modified +3/-1 (4 lines); hunks: -220,7 +220,9 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -682,7 +682,9 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: -24,6 +24,7。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v.py` modified +84/-50 (134 lines); hunks: -27,18 +27,24; -48,6 +54,8; symbols: __init__, forward
  - `python/sglang/srt/models/glm4.py` modified +3/-1 (4 lines); hunks: -220,7 +220,9 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -682,7 +682,9 @@ def __init__(; symbols: __init__
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: -24,6 +24,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -27,18 +27,24 @@
+from sglang.srt.distributed import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+)
+from sglang.srt.distributed.parallel_state import get_pp_group
-from sglang.srt.layers.layernorm import RMSNorm
diff -- python/sglang/srt/models/glm4.py
@@ -220,7 +220,9 @@ def __init__(
-        partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
+        partial_rotary_factor = getattr(
+            getattr(config, "rope_parameters", None), "partial_rotary_factor", None
+        ) or getattr(config, "partial_rotary_factor", 0.5)
diff -- python/sglang/srt/models/glm4_moe.py
@@ -682,7 +682,9 @@ def __init__(
-        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
+        partial_rotary_factor = getattr(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +84/-50; `python/sglang/srt/models/glm4.py` modified +3/-1; `python/sglang/srt/models/glm4_moe.py` modified +3/-1
  - tests: `test/nightly/test_encoder_dp.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/nightly/test_encoder_dp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14662 - [Glm46v] support ktransformers

- 链接: https://github.com/sgl-project/sglang/pull/14662
- 状态/时间: open / 2025-12-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-0，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「[Glm46v] support ktransformers」，变更集中在 `python/sglang/srt/models/glm4v_moe.py`。PR 描述补充为：## Motivation support ktransformers for GLM-4.6V ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [x] Format your code according to the Format cod...
- 实现要点: `python/sglang/srt/models/glm4v_moe.py` modified +8/-0 (8 lines); hunks: -16,6 +16,7; -281,5 +282,12 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, get_model_config_for_expert_location，涉及 `load_weights, get_model_config_for_expert_location`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v_moe.py` modified +8/-0 (8 lines); hunks: -16,6 +16,7; -281,5 +282,12 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, get_model_config_for_expert_location
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -16,6 +16,7 @@
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
@@ -281,5 +282,12 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+    @classmethod
+    def get_model_config_for_expert_location(cls, config):
+        return ModelConfigForExpertLocation(
+            num_layers=config.text_config.num_hidden_layers,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4v_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14720 - [GLM-4.6V] Support Pipeline Parallelism for GLM-4.6V & GLM-4.1V

- 链接: https://github.com/sgl-project/sglang/pull/14720
- 状态/时间: merged / 2025-12-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+66/-2，可读 patch 144 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「[GLM-4.6V] Support Pipeline Parallelism for GLM-4.6V & GLM-4.1V」，变更集中在 `python/sglang/srt/models/glm4v.py`, `test/srt/test_pp_single_node.py`, `python/sglang/test/test_utils.py`。PR 描述补充为：## Motivation This PR is to support GLM-4.6V and GLM-4.1V Pipeline Parallelism. In main branch GLM-4.6V and GLM-4.1V PP don't work correctly. There are two reasons: 1. In pp-siz...
- 实现要点: `python/sglang/srt/models/glm4v.py` modified +24/-1 (25 lines); hunks: -51,7 +51,7; -659,6 +659,7 @@ def forward(; symbols: forward, load_weights，涉及 `forward, load_weights`；`test/srt/test_pp_single_node.py` modified +38/-0 (38 lines); hunks: -19,6 +19,7; -318,5 +319,42 @@ def test_chunked_prefill_with_small_bs(self):; symbols: test_chunked_prefill_with_small_bs, TestGLM41VPPAccuracy, setUpClass, tearDownClass，涉及 `test_chunked_prefill_with_small_bs, TestGLM41VPPAccuracy, setUpClass`；`python/sglang/test/test_utils.py` modified +3/-0 (3 lines); hunks: -57,6 +57,9；`test/srt/run_suite.py` modified +1/-1 (2 lines); hunks: -150,7 +150,7。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v.py` modified +24/-1 (25 lines); hunks: -51,7 +51,7; -659,6 +659,7 @@ def forward(; symbols: forward, load_weights
  - `test/srt/test_pp_single_node.py` modified +38/-0 (38 lines); hunks: -19,6 +19,7; -318,5 +319,42 @@ def test_chunked_prefill_with_small_bs(self):; symbols: test_chunked_prefill_with_small_bs, TestGLM41VPPAccuracy, setUpClass, tearDownClass
  - `python/sglang/test/test_utils.py` modified +3/-0 (3 lines); hunks: -57,6 +57,9
  - `test/srt/run_suite.py` modified +1/-1 (2 lines); hunks: -150,7 +150,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -51,7 +51,7 @@
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch
+from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
@@ -659,6 +659,7 @@ def forward(
+        pp_proxy_tensors: Optional[PPProxyTensors] = None,
@@ -691,6 +692,7 @@ def forward(
+            pp_proxy_tensors=pp_proxy_tensors,
diff -- test/srt/test_pp_single_node.py
@@ -19,6 +19,7 @@
+    DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP,
@@ -318,5 +319,42 @@ def test_chunked_prefill_with_small_bs(self):
+class TestGLM41VPPAccuracy(unittest.TestCase):
+    @classmethod
+    def setUpClass(cls):
+        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP
diff -- python/sglang/test/test_utils.py
@@ -57,6 +57,9 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +24/-1
  - tests: `test/srt/test_pp_single_node.py` modified +38/-0; `python/sglang/test/test_utils.py` modified +3/-0; `test/srt/run_suite.py` modified +1/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/test_utils.py`, `test/srt/run_suite.py`, `test/srt/test_pp_single_node.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14927 - [CI]add nightly CI for glm4v_moe arch model

- 链接: https://github.com/sgl-project/sglang/pull/14927
- 状态/时间: merged / 2025-12-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「[CI]add nightly CI for glm4v_moe arch model」，变更集中在 `test/nightly/test_vlms_mmmu_eval.py`。PR 描述补充为：## Motivation add nightly CI for glm4v_moe arch model to prevent silent regression like this https://github.com/sgl-project/sglang/issues/14582. Note: existing `zai-org/GLM-4.1V...
- 实现要点: `test/nightly/test_vlms_mmmu_eval.py` modified +3/-0 (3 lines); hunks: -46,6 +46,9。
- 代码 diff 细节:
  - `test/nightly/test_vlms_mmmu_eval.py` modified +3/-0 (3 lines); hunks: -46,6 +46,9
- 关键代码摘录:

```diff
diff -- test/nightly/test_vlms_mmmu_eval.py
@@ -46,6 +46,9 @@
+    ModelLaunchSettings(
+        "zai-org/GLM-4.5V-FP8", extra_args=["--tp=2"]
+    ): ModelEvalMetrics(0.26, 32.0),
```

- 已读文件:
  - tests: `test/nightly/test_vlms_mmmu_eval.py` modified +3/-0
- 验证与风险: diff 自带测试面 `test/nightly/test_vlms_mmmu_eval.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14998 - add transformers version validation for glm-4.6v moe models

- 链接: https://github.com/sgl-project/sglang/pull/14998
- 状态/时间: merged / 2025-12-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+37/-0，可读 patch 51 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「add transformers version validation for glm-4.6v moe models」，变更集中在 `python/sglang/srt/configs/model_config.py`。PR 描述补充为：## Motivation This PR adds a version check for the installed transformers library to ensure compatibility with GLM-4.6v MoE vision models. For models requiring Transformers ≥ 5....
- 实现要点: `python/sglang/srt/configs/model_config.py` modified +37/-0 (37 lines); hunks: -205,6 +205,8 @@ def __init__(; -773,6 +775,41 @@ def _verify_dual_chunk_attention_config(self) -> None:; symbols: __init__, _verify_dual_chunk_attention_config, _verify_transformers_version, _get_hf_eos_token_id，涉及 `__init__, _verify_dual_chunk_attention_config, _verify_transformers_version`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/model_config.py` modified +37/-0 (37 lines); hunks: -205,6 +205,8 @@ def __init__(; -773,6 +775,41 @@ def _verify_dual_chunk_attention_config(self) -> None:; symbols: __init__, _verify_dual_chunk_attention_config, _verify_transformers_version, _get_hf_eos_token_id
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/model_config.py
@@ -205,6 +205,8 @@ def __init__(
+        self._verify_transformers_version()
@@ -773,6 +775,41 @@ def _verify_dual_chunk_attention_config(self) -> None:
+    def _verify_transformers_version(self):
+        import transformers
+        from packaging import version
+        tf_version_str = getattr(transformers, "__version__", None)
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/model_config.py` modified +37/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15205 - [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V

- 链接: https://github.com/sgl-project/sglang/pull/15205
- 状态/时间: merged / 2025-12-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+100/-80，可读 patch 345 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「[VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V」，变更集中在 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`。PR 描述补充为：## Motivation Support cos sin cache for Qwen3-VL & GLM-4.1V. This PR refactors the rotary positional embedding (RoPE) implementation to expose an explicit cosine/sine cache inte...
- 实现要点: `python/sglang/srt/models/glm4v.py` modified +34/-50 (84 lines); hunks: -44,6 +44,7; -157,7 +158,8 @@ def forward(; symbols: forward, Glm4vVisionRotaryEmbedding, __init__，涉及 `forward, Glm4vVisionRotaryEmbedding, __init__`；`python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunks: -24,9 +24,6; -39,6 +36,7; symbols: forward, __init__, dtype, device，涉及 `forward, __init__, dtype`；`python/sglang/srt/layers/attention/vision.py` modified +20/-10 (30 lines); hunks: -675,6 +675,8 @@ def forward(; -724,26 +726,34 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/layers/rotary_embedding.py` modified +5/-0 (5 lines); hunks: -219,6 +219,11 @@ def get_cos_sin_with_position(self, positions):; symbols: get_cos_sin_with_position, get_cos_sin, forward_native，涉及 `get_cos_sin_with_position, get_cos_sin, forward_native`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v.py` modified +34/-50 (84 lines); hunks: -44,6 +44,7; -157,7 +158,8 @@ def forward(; symbols: forward, Glm4vVisionRotaryEmbedding, __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunks: -24,9 +24,6; -39,6 +36,7; symbols: forward, __init__, dtype, device
  - `python/sglang/srt/layers/attention/vision.py` modified +20/-10 (30 lines); hunks: -675,6 +675,8 @@ def forward(; -724,26 +726,34 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/rotary_embedding.py` modified +5/-0 (5 lines); hunks: -219,6 +219,11 @@ def get_cos_sin_with_position(self, positions):; symbols: get_cos_sin_with_position, get_cos_sin, forward_native
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -44,6 +44,7 @@
+from sglang.srt.layers.rotary_embedding import get_rope
@@ -157,7 +158,8 @@ def forward(
-        position_embeddings: torch.Tensor,
+        rotary_pos_emb_cos: torch.Tensor,
+        rotary_pos_emb_sin: torch.Tensor,
@@ -169,7 +171,8 @@ def forward(
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -24,9 +24,6 @@
-from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
-    Qwen2_5_VisionRotaryEmbedding,
-)
@@ -39,6 +36,7 @@
+from sglang.srt.layers.rotary_embedding import get_rope
@@ -188,14 +186,16 @@ def forward(
diff -- python/sglang/srt/layers/attention/vision.py
@@ -675,6 +675,8 @@ def forward(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +34/-50; `python/sglang/srt/models/qwen3_vl.py` modified +41/-20; `python/sglang/srt/layers/attention/vision.py` modified +20/-10; `python/sglang/srt/layers/rotary_embedding.py` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/glm4v.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15434 - Convert cu_seqlens to CPU for npu_flash_attention_unpad operator

- 链接: https://github.com/sgl-project/sglang/pull/15434
- 状态/时间: merged / 2026-01-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+36/-13，可读 patch 169 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 优化关键推理路径或后端选择，标题为「Convert cu_seqlens to CPU for npu_flash_attention_unpad operator」，变更集中在 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/paddleocr_vl.py`。PR 描述补充为：## Motivation In order to improve the performance of VisionAscendAttention, we convert cu_seqlens to CPU before the first transformer layer, because converting it to CPU per lay...
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +6/-3 (9 lines); hunks: -63,7 +63,7; -446,9 +446,12 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/glm4v.py` modified +5/-1 (6 lines); hunks: -57,7 +57,7; -514,6 +514,10 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor)...; symbols: forward，涉及 `forward`；`python/sglang/srt/models/paddleocr_vl.py` modified +4/-2 (6 lines); hunks: -36,7 +36,7; -442,7 +442,9 @@ def forward(; symbols: Projector, forward，涉及 `Projector, forward`；`python/sglang/srt/models/qwen2_5_vl.py` modified +4/-2 (6 lines); hunks: -76,7 +76,7; -453,7 +453,9 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +6/-3 (9 lines); hunks: -63,7 +63,7; -446,9 +446,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/glm4v.py` modified +5/-1 (6 lines); hunks: -57,7 +57,7; -514,6 +514,10 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor)...; symbols: forward
  - `python/sglang/srt/models/paddleocr_vl.py` modified +4/-2 (6 lines); hunks: -36,7 +36,7; -442,7 +442,9 @@ def forward(; symbols: Projector, forward
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +4/-2 (6 lines); hunks: -76,7 +76,7; -453,7 +453,9 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/dots_vlm_vit.py` modified +4/-1 (5 lines); hunks: -12,7 +12,7; -315,6 +315,9 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -63,7 +63,7 @@
-from sglang.srt.utils import add_prefix, get_int_env_var
+from sglang.srt.utils import add_prefix, get_int_env_var, is_npu
@@ -446,9 +446,12 @@ def forward(
+        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
+        if not is_npu():
+            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
diff -- python/sglang/srt/models/glm4v.py
@@ -57,7 +57,7 @@
-from sglang.srt.utils import add_prefix
+from sglang.srt.utils import add_prefix, is_npu
@@ -514,6 +514,10 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
+        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
+        if is_npu():
+            cu_seqlens = cu_seqlens.to("cpu")
diff -- python/sglang/srt/models/paddleocr_vl.py
@@ -36,7 +36,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +6/-3; `python/sglang/srt/models/glm4v.py` modified +5/-1; `python/sglang/srt/models/paddleocr_vl.py` modified +4/-2; `python/sglang/srt/models/qwen2_5_vl.py` modified +4/-2; `python/sglang/srt/models/dots_vlm_vit.py` modified +4/-1; `python/sglang/srt/models/idefics2.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17582 - [GLM-OCR] Support GLM-OCR Model

- 链接: https://github.com/sgl-project/sglang/pull/17582
- 状态/时间: merged / 2026-01-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/multimodal/processors/glm4v.py`；关联提交 `7106f6c8e150`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+679/-29，可读 patch 851 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「[GLM-OCR] Support GLM-OCR Model」，变更集中在 `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/multimodal/processors/glm4v.py`。PR 描述补充为：Support for GLM-OCR Model, transformers PR here Need transformer>=5.0.0dev0, but not 5.0.0, so also changed min requirements in glm4v.
- 实现要点: `python/sglang/srt/models/glm_ocr.py` added +435/-0 (435 lines); hunks: -0,0 +1,435; symbols: GlmOcrRMSNorm, GlmOcrVisionMLP, GlmOcrVisionBlock, __init__，涉及 `GlmOcrRMSNorm, GlmOcrVisionMLP, GlmOcrVisionBlock`；`python/sglang/srt/models/glm_ocr_nextn.py` added +162/-0 (162 lines); hunks: -0,0 +1,162; symbols: GlmOcrModelNextN, __init__, forward, GlmOcrForConditionalGenerationNextN，涉及 `GlmOcrModelNextN, __init__, forward`；`python/sglang/srt/multimodal/processors/glm4v.py` modified +6/-1 (7 lines); hunks: -3,14 +3,19; symbols: Glm4vImageProcessor, __init__，涉及 `Glm4vImageProcessor, __init__`；`python/sglang/srt/models/glm4v.py` modified +0/-2 (2 lines); hunks: -758,8 +758,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm_ocr.py` added +435/-0 (435 lines); hunks: -0,0 +1,435; symbols: GlmOcrRMSNorm, GlmOcrVisionMLP, GlmOcrVisionBlock, __init__
  - `python/sglang/srt/models/glm_ocr_nextn.py` added +162/-0 (162 lines); hunks: -0,0 +1,162; symbols: GlmOcrModelNextN, __init__, forward, GlmOcrForConditionalGenerationNextN
  - `python/sglang/srt/multimodal/processors/glm4v.py` modified +6/-1 (7 lines); hunks: -3,14 +3,19; symbols: Glm4vImageProcessor, __init__
  - `python/sglang/srt/models/glm4v.py` modified +0/-2 (2 lines); hunks: -758,8 +758,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm_ocr.py
@@ -0,0 +1,435 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/glm_ocr_nextn.py
@@ -0,0 +1,162 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/multimodal/processors/glm4v.py
@@ -3,14 +3,19 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm_ocr.py` added +435/-0; `python/sglang/srt/models/glm_ocr_nextn.py` added +162/-0; `python/sglang/srt/multimodal/processors/glm4v.py` modified +6/-1; `python/sglang/srt/models/glm4v.py` modified +0/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/glm4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17420 - [VLM] Optimize get_rope_index for GLM4v

- 链接: https://github.com/sgl-project/sglang/pull/17420
- 状态/时间: merged / 2026-02-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+526/-86，可读 patch 758 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 优化关键推理路径或后端选择，标题为「[VLM] Optimize get_rope_index for GLM4v」，变更集中在 `python/sglang/srt/layers/rotary_embedding.py`, `benchmark/bench_rope/benchmark_rope_index.py`。PR 描述补充为：## Motivation Speedup **12%** to **600%**(long token length) for get_rope_index for GLM4v. Benchmark test added. lmms_evals no drop. PR: Main: ## Modifications ## Accuracy Tests...
- 实现要点: `python/sglang/srt/layers/rotary_embedding.py` modified +101/-86 (187 lines); hunks: -1700,11 +1700,12 @@ def get_rope_index(; -1737,24 +1738,24 @@ def get_rope_index(; symbols: get_rope_index, get_rope_index_glm4v，涉及 `get_rope_index, get_rope_index_glm4v`；`benchmark/bench_rope/benchmark_rope_index.py` added +425/-0 (425 lines); hunks: -0,0 +1,425; symbols: DummyVisionConfig, DummyHFConfig, calculate_stats, _sync，涉及 `DummyVisionConfig, DummyHFConfig, calculate_stats`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/rotary_embedding.py` modified +101/-86 (187 lines); hunks: -1700,11 +1700,12 @@ def get_rope_index(; -1737,24 +1738,24 @@ def get_rope_index(; symbols: get_rope_index, get_rope_index_glm4v
  - `benchmark/bench_rope/benchmark_rope_index.py` added +425/-0 (425 lines); hunks: -0,0 +1,425; symbols: DummyVisionConfig, DummyHFConfig, calculate_stats, _sync
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -1700,11 +1700,12 @@ def get_rope_index(
-                    llm_grid_t, llm_grid_h, llm_grid_w = (
-                        t.item(),
-                        h.item() // spatial_merge_size,
-                        w.item() // spatial_merge_size,
-                    )
+                    # Avoid .item() lookups in repeated context
diff -- benchmark/bench_rope/benchmark_rope_index.py
@@ -0,0 +1,425 @@
+# This script benchmarks MRotaryEmbedding.get_rope_index_glm4v (GLM4V mrope index builder).
+# It generates synthetic multimodal input_ids + attention_mask (+ optional image/video grids),
+# runs benchmarks.
+#
+# == Usage Examples ==
+#
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/rotary_embedding.py` modified +101/-86
  - other: `benchmark/bench_rope/benchmark_rope_index.py` added +425/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/rotary_embedding.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18885 - Fix GLM-4V processor registration when glm_ocr is unavailable

- 链接: https://github.com/sgl-project/sglang/pull/18885
- 状态/时间: merged / 2026-02-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/glm4v.py`；关联提交 `206accd15de3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-4，可读 patch 33 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 修复已暴露的启动、加载、解析或数值问题，标题为「Fix GLM-4V processor registration when glm_ocr is unavailable」，变更集中在 `python/sglang/srt/multimodal/processors/glm4v.py`。PR 描述补充为：## Motivation Fix nightly test failures where GLM-4V models fail with "No processor registered for architecture" or show "Model not evaluated". Affected tests: - **4-GPU**: `tes...
- 实现要点: `python/sglang/srt/multimodal/processors/glm4v.py` modified +12/-4 (16 lines); hunks: -3,20 +3,28; symbols: Glm4vImageProcessor, __init__，涉及 `Glm4vImageProcessor, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/glm4v.py` modified +12/-4 (16 lines); hunks: -3,20 +3,28; symbols: Glm4vImageProcessor, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/glm4v.py
@@ -3,20 +3,28 @@
-from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
+try:
+    from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
+except ImportError:
+    GlmOcrForConditionalGeneration = None
-        Glm4vForConditionalGeneration,
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/glm4v.py` modified +12/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/glm4v.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19728 - Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback

- 链接: https://github.com/sgl-project/sglang/pull/19728
- 状态/时间: open / 2026-03-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+104/-4，可读 patch 179 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 修复已暴露的启动、加载、解析或数值问题，标题为「Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback」，变更集中在 `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `test/registered/moe/test_fused_moe.py`。PR 描述补充为：## Summary This fixes two ROCm regressions that blocked `GLM-4.5V-FP8` startup on MI300X when launching with: ## Root Cause There were two independent failures on ROCm: 1. `fuse...
- 实现要点: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4 (25 lines); hunks: -1541,6 +1541,23 @@ def per_token_group_quant_mla_deep_gemm_masked_fp8(; -1549,10 +1566,10 @@ def _native_dynamic_per_token_quant_fp8(output, input, s...; symbols: per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_tensor_quant_fp8，涉及 `per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8`；`python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0 (6 lines); hunks: -351,6 +351,12 @@ def fused_experts_impl(; symbols: fused_experts_impl，涉及 `fused_experts_impl`；`test/registered/moe/test_fused_moe.py` modified +66/-0 (66 lines); hunks: -1,9 +1,11; -239,6 +241,70 @@ def test_various_configurations(self):; symbols: test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding，涉及 `test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding`；`python/sglang/test/test_custom_ops.py` modified +11/-0 (11 lines); hunks: -3,11 +3,14; -141,6 +144,14 @@ def test_scaled_fp8_quant_with_padding(dtype) -> None:; symbols: test_scaled_fp8_quant_with_padding，涉及 `test_scaled_fp8_quant_with_padding`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4 (25 lines); hunks: -1541,6 +1541,23 @@ def per_token_group_quant_mla_deep_gemm_masked_fp8(; -1549,10 +1566,10 @@ def _native_dynamic_per_token_quant_fp8(output, input, s...; symbols: per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_tensor_quant_fp8
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0 (6 lines); hunks: -351,6 +351,12 @@ def fused_experts_impl(; symbols: fused_experts_impl
  - `test/registered/moe/test_fused_moe.py` modified +66/-0 (66 lines); hunks: -1,9 +1,11; -239,6 +241,70 @@ def test_various_configurations(self):; symbols: test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding
  - `python/sglang/test/test_custom_ops.py` modified +11/-0 (11 lines); hunks: -3,11 +3,14; -141,6 +144,14 @@ def test_scaled_fp8_quant_with_padding(dtype) -> None:; symbols: test_scaled_fp8_quant_with_padding
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -1541,6 +1541,23 @@ def per_token_group_quant_mla_deep_gemm_masked_fp8(
+    def _copy_with_optional_row_padding(
+        dst: torch.Tensor,
+        src: torch.Tensor,
+        pad_value: float = 0.0,
+    ) -> None:
+        if dst.shape == src.shape:
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
@@ -351,6 +351,12 @@ def fused_experts_impl(
+    elif hidden_states.shape[1] == w1.shape[2]:
+        # Some ROCm FP8 MoE checkpoints load unpadded expert weights even when
+        # SGLANG_MOE_PADDING is enabled globally. In that case the runtime
+        # shape already matches the true hidden size and subtracting
+        # `padding_size` would reject a valid layout.
+        padded_size = 0
diff -- test/registered/moe/test_fused_moe.py
@@ -1,9 +1,11 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0
  - tests: `test/registered/moe/test_fused_moe.py` modified +66/-0; `python/sglang/test/test_custom_ops.py` modified +11/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/test_custom_ops.py`, `test/registered/moe/test_fused_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20033 - [VLM] Replace conv3d proj with linear for GLM4V

- 链接: https://github.com/sgl-project/sglang/pull/20033
- 状态/时间: merged / 2026-03-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4v.py`；关联提交 `97a2a9be0f45`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+192/-9，可读 patch 228 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 优化关键推理路径或后端选择，标题为「[VLM] Replace conv3d proj with linear for GLM4V」，变更集中在 `python/sglang/srt/models/glm4v.py`。PR 描述补充为：## Motivation Inspired by https://github.com/sgl-project/sglang/pull/19788 with some optimizations: lmms_evals no drops. Main and PR are the same score: More performance test wi...
- 实现要点: `python/sglang/srt/models/glm4v.py` modified +26/-9 (35 lines); hunks: -211,16 +211,26 @@ def __init__(; -446,10 +456,16 @@ def __init__(; symbols: __init__, forward, copy_conv3d_weight_to_linear, Glm4vPatchMerger，涉及 `__init__, forward, copy_conv3d_weight_to_linear`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v.py` modified +26/-9 (35 lines); hunks: -211,16 +211,26 @@ def __init__(; -446,10 +456,16 @@ def __init__(; symbols: __init__, forward, copy_conv3d_weight_to_linear, Glm4vPatchMerger
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -211,16 +211,26 @@ def __init__(
-    def forward(self, x: torch.Tensor) -> torch.Tensor:
-        x = x.view(
-            -1,
-            self.in_channels,
-            self.temporal_patch_size,
-            self.patch_size,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +26/-9
- 验证与风险: diff 自带测试面 `test/registered/vlm/test_patch_embed_perf.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20463 - [Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr

- 链接: https://github.com/sgl-project/sglang/pull/20463
- 状态/时间: merged / 2026-03-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`；关联提交 `c330b687a116`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-0，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr」，变更集中在 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`。PR 描述补充为：## Motivation PR #20033 replaced `Conv3d` with `Linear` in `Glm4vVisionPatchEmbed` and added `copy_conv3d_weight_to_linear()` at the end of `glm4v.py`'s `load_weights()`. Howeve...
- 实现要点: `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunks: -281,5 +281,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`；`python/sglang/srt/models/glm_ocr.py` modified +3/-0 (3 lines); hunks: -431,5 +431,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunks: -281,5 +281,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
  - `python/sglang/srt/models/glm_ocr.py` modified +3/-0 (3 lines); hunks: -431,5 +431,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -281,5 +281,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+        if not is_nextn:
+            self.visual.patch_embed.copy_conv3d_weight_to_linear()
diff -- python/sglang/srt/models/glm_ocr.py
@@ -431,5 +431,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+        if not is_nextn:
+            self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +3/-0; `python/sglang/srt/models/glm_ocr.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20282 - Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug

- 链接: https://github.com/sgl-project/sglang/pull/20282
- 状态/时间: merged / 2026-03-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+704/-90，可读 patch 1053 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug」，变更集中在 `python/sglang/srt/layers/conv.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/pixtral.py`。PR 描述补充为：## Motivation Very thanks for the benchmark data in this PR: https://github.com/sgl-project/sglang/pull/19788 - Add `Conv2dLayer`/`Conv3dLayer` in `sglang/srt/layers/conv.py`. C...
- 实现要点: `python/sglang/srt/layers/conv.py` added +300/-0 (300 lines); hunks: -0,0 +1,300; symbols: _tuplify, _check_enable_linear, _reverse_repeat_tuple, _compute_same_padding_for_pad，涉及 `_tuplify, _check_enable_linear, _reverse_repeat_tuple`；`python/sglang/srt/models/glm4v.py` modified +12/-27 (39 lines); hunks: -35,6 +35,7; -203,34 +204,25 @@ def __init__(; symbols: __init__, copy_conv3d_weight_to_linear, forward, Glm4vPatchMerger，涉及 `__init__, copy_conv3d_weight_to_linear, forward`；`python/sglang/srt/models/pixtral.py` modified +3/-2 (5 lines); hunks: -35,6 +35,7; -328,7 +329,7 @@ class VisionTransformer(nn.Module):; symbols: VisionTransformer, __init__，涉及 `VisionTransformer, __init__`；`python/sglang/srt/models/clip.py` modified +2/-1 (3 lines); hunks: -11,6 +11,7; -32,7 +33,7 @@ def __init__(self, config: CLIPVisionConfig):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/conv.py` added +300/-0 (300 lines); hunks: -0,0 +1,300; symbols: _tuplify, _check_enable_linear, _reverse_repeat_tuple, _compute_same_padding_for_pad
  - `python/sglang/srt/models/glm4v.py` modified +12/-27 (39 lines); hunks: -35,6 +35,7; -203,34 +204,25 @@ def __init__(; symbols: __init__, copy_conv3d_weight_to_linear, forward, Glm4vPatchMerger
  - `python/sglang/srt/models/pixtral.py` modified +3/-2 (5 lines); hunks: -35,6 +35,7; -328,7 +329,7 @@ class VisionTransformer(nn.Module):; symbols: VisionTransformer, __init__
  - `python/sglang/srt/models/clip.py` modified +2/-1 (3 lines); hunks: -11,6 +11,7; -32,7 +33,7 @@ def __init__(self, config: CLIPVisionConfig):; symbols: __init__
  - `python/sglang/srt/models/dots_vlm_vit.py` modified +2/-1 (3 lines); hunks: -11,6 +11,7; -113,7 +114,7 @@ def __init__(self, config, quant_config: Optional[Quantizati...; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/conv.py
@@ -0,0 +1,300 @@
+"""
+Conv2d/Conv3d layers with unfold+linear optimization for patch embeddings.
+When kernel_size == stride, padding == 0, dilation == 1, groups == 1, the conv
+is equivalent to unfold + F.linear, which is significantly faster on CUDA and
+also avoids the PyTorch 2.9.1 + CuDNN < 9.15 Conv3d bug
+(https://github.com/pytorch/pytorch/issues/168167).
diff -- python/sglang/srt/models/glm4v.py
@@ -35,6 +35,7 @@
+from sglang.srt.layers.conv import Conv3dLayer
@@ -203,34 +204,25 @@ def __init__(
-        self.proj = nn.Conv3d(
+        self.proj = Conv3dLayer(
-        k = self.in_channels * self.temporal_patch_size * self.patch_size**2
-        self.linear = nn.Linear(
diff -- python/sglang/srt/models/pixtral.py
@@ -35,6 +35,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/conv.py` added +300/-0; `python/sglang/srt/models/glm4v.py` modified +12/-27; `python/sglang/srt/models/pixtral.py` modified +3/-2; `python/sglang/srt/models/clip.py` modified +2/-1; `python/sglang/srt/models/dots_vlm_vit.py` modified +2/-1; `python/sglang/srt/models/idefics2.py` modified +2/-1
- 验证与风险: diff 自带测试面 `test/unit/test_conv_layer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20740 - Revert "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr"

- 链接: https://github.com/sgl-project/sglang/pull/20740
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`；关联提交 `f15b3338c9f3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+0/-6，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 修复已暴露的启动、加载、解析或数值问题，标题为「Revert "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr"」，变更集中在 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`。PR 描述补充为：Reverts sgl-project/sglang#20463 per request by @JustinTong0323
- 实现要点: `python/sglang/srt/models/glm4v_moe.py` modified +0/-3 (3 lines); hunks: -281,8 +281,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`；`python/sglang/srt/models/glm_ocr.py` modified +0/-3 (3 lines); hunks: -431,8 +431,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v_moe.py` modified +0/-3 (3 lines); hunks: -281,8 +281,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
  - `python/sglang/srt/models/glm_ocr.py` modified +0/-3 (3 lines); hunks: -431,8 +431,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -281,8 +281,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
-        if not is_nextn:
-            self.visual.patch_embed.copy_conv3d_weight_to_linear()
diff -- python/sglang/srt/models/glm_ocr.py
@@ -431,8 +431,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
-        if not is_nextn:
-            self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +0/-3; `python/sglang/srt/models/glm_ocr.py` modified +0/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21134 - [Bug Fix] GLM-V / GLM-OCR: field detection for transformers 5.x and MTP omission fix

- 链接: https://github.com/sgl-project/sglang/pull/21134
- 状态/时间: merged / 2026-03-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`；关联提交 `fcaad42b0038`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+16/-9，可读 patch 74 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bug Fix] GLM-V / GLM-OCR: field detection for transformers 5.x and MTP omission fix」，变更集中在 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`。PR 描述补充为：Mainly modify several issues 1. Modified the position of the replacement model module to ensure that the MTP has a normal acceptance rate after reading, otherwise the accept len...
- 实现要点: `python/sglang/srt/models/glm4v_moe.py` modified +7/-7 (14 lines); hunks: -158,6 +158,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -196,13 +203,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`；`python/sglang/srt/models/glm_ocr.py` modified +4/-1 (5 lines); hunks: -26,6 +26,7; -151,6 +152,7 @@ class GlmOcrVisionModel(Glm4vVisionModel):; symbols: GlmOcrVisionModel, __init__，涉及 `GlmOcrVisionModel, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v_moe.py` modified +7/-7 (14 lines); hunks: -158,6 +158,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -196,13 +203,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
  - `python/sglang/srt/models/glm_ocr.py` modified +4/-1 (5 lines); hunks: -26,6 +26,7; -151,6 +152,7 @@ class GlmOcrVisionModel(Glm4vVisionModel):; symbols: GlmOcrVisionModel, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -158,6 +158,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+            if "language_model." in name:
+                name = name.replace("language_model.", "")
+            if "model.visual." in name:
+                name = name.replace("model.visual.", "visual.")
+            if "rotary_emb.inv_freq" in name:
+                continue
diff -- python/sglang/srt/models/glm_ocr.py
@@ -26,6 +26,7 @@
+    GlmOcrTextConfig,
@@ -151,6 +152,7 @@ class GlmOcrVisionModel(Glm4vVisionModel):
+        text_config: GlmOcrTextConfig,
@@ -203,7 +205,7 @@ def __init__(
-            context_dim=vision_config.out_hidden_size * vision_config.in_channels,
+            context_dim=text_config.intermediate_size,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +7/-7; `python/sglang/srt/models/glm_ocr.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17122 - [bugfix]GLM-4V model

- 链接: https://github.com/sgl-project/sglang/pull/17122
- 状态/时间: merged / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+38/-3，可读 patch 70 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 修复已暴露的启动、加载、解析或数值问题，标题为「[bugfix]GLM-4V model」，变更集中在 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py`。PR 描述补充为：## Motivation GLM-4V model bugfix ## Modifications 1. In VisionAttention, it is necessary to use num_dumy_heads to calculate. If num_dumy_heads=0 and the result of num_dumy_head...
- 实现要点: `python/sglang/srt/models/glm4v.py` modified +2/-2 (4 lines); hunks: -414,6 +414,7 @@ def __init__(; -553,15 +554,14 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/multimodal/processors/base_processor.py` modified +3/-1 (4 lines); hunks: -407,7 +407,9 @@ def process_mm_data(; symbols: process_mm_data，涉及 `process_mm_data`；`test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py` added +33/-0 (33 lines); hunks: -0,0 +1,33; symbols: TestGLM4Models, test_vlm_mmmu_benchmark，涉及 `TestGLM4Models, test_vlm_mmmu_benchmark`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v.py` modified +2/-2 (4 lines); hunks: -414,6 +414,7 @@ def __init__(; -553,15 +554,14 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +3/-1 (4 lines); hunks: -407,7 +407,9 @@ def process_mm_data(; symbols: process_mm_data
  - `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py` added +33/-0 (33 lines); hunks: -0,0 +1,33; symbols: TestGLM4Models, test_vlm_mmmu_benchmark
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -414,6 +414,7 @@ def __init__(
+                    num_dummy_heads=vision_config.num_dummy_heads,
@@ -553,15 +554,14 @@ def __init__(
+        vision_utils.update_vit_attn_dummy_heads_config(self.config)
-        vision_utils.update_vit_attn_dummy_heads_config(self.config)
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -407,7 +407,9 @@ def process_mm_data(
-            else:
+            elif processor.__class__.__name__ not in {
+                "Glm4vProcessor",
+            }:
diff -- test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py
@@ -0,0 +1,33 @@
+import unittest
+from sglang.test.ascend.vlm_utils import TestVLMModels
+from sglang.test.ci.ci_register import register_npu_ci
+register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +2/-2; `python/sglang/srt/multimodal/processors/base_processor.py` modified +3/-1
  - tests: `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py` added +33/-0
- 验证与风险: diff 自带测试面 `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22961 - [NPU] Support GLM-4.5V

- 链接: https://github.com/sgl-project/sglang/pull/22961
- 状态/时间: open / 2026-04-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+17/-5，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 GLM VLM/OCR 补齐模型支持入口或运行时能力，标题为「[NPU] Support GLM-4.5V」，变更集中在 `python/sglang/srt/models/glm4_moe.py`。PR 描述补充为：## Motivation Support GLM-4.5V on NPU. ## Modifications 1. When calling the split_qkv_rmsnorm_rope function, pass the correct arguments based on the use_qk_norm parameter.The `s...
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +17/-5 (22 lines); hunks: -313,18 +313,30 @@ def forward_prepare(; symbols: forward_prepare，涉及 `forward_prepare`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +17/-5 (22 lines); hunks: -313,18 +313,30 @@ def forward_prepare(; symbols: forward_prepare
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -313,18 +313,30 @@ def forward_prepare(
+            if self.use_qk_norm:
+                eps = self.q_norm.variance_epsilon
+                q_weight = self.q_norm.weight
+                k_weight = self.k_norm.weight
+                q_bias = getattr(self.q_norm, "bias", None)
+                k_bias = getattr(self.k_norm, "bias", None)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +17/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
