# vllm InternVL 3.5 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/models/multimodal/pooling/test_intern_vit.py` | 无直接 PR 号提交 |
| `tests/models/multimodal/processing/test_internvl.py` | [#12553](https://github.com/vllm-project/vllm/pull/12553), [#37260](https://github.com/vllm-project/vllm/pull/37260) |
| `vllm/model_executor/models/intern_vit.py` | [#6514](https://github.com/vllm-project/vllm/pull/6514), [#7067](https://github.com/vllm-project/vllm/pull/7067), [#9528](https://github.com/vllm-project/vllm/pull/9528), [#23909](https://github.com/vllm-project/vllm/pull/23909), [#38049](https://github.com/vllm-project/vllm/pull/38049) |
| `vllm/model_executor/models/internvl.py` | [#6514](https://github.com/vllm-project/vllm/pull/6514), [#7067](https://github.com/vllm-project/vllm/pull/7067), [#7164](https://github.com/vllm-project/vllm/pull/7164), [#7860](https://github.com/vllm-project/vllm/pull/7860), [#8201](https://github.com/vllm-project/vllm/pull/8201), [#8250](https://github.com/vllm-project/vllm/pull/8250), [#8299](https://github.com/vllm-project/vllm/pull/8299), [#8375](https://github.com/vllm-project/vllm/pull/8375), [#8614](https://github.com/vllm-project/vllm/pull/8614), [#8946](https://github.com/vllm-project/vllm/pull/8946), [#9351](https://github.com/vllm-project/vllm/pull/9351), [#9528](https://github.com/vllm-project/vllm/pull/9528), ... (27 total) |
| `vllm/transformers_utils/processors/internvl.py` | [#37260](https://github.com/vllm-project/vllm/pull/37260), [#37324](https://github.com/vllm-project/vllm/pull/37324) |

## PR 覆盖总览

- git 追溯 PR 数: 28
- 原文档显式引用补充 PR 数: 1
- 当前文档总 PR 数: 29
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2024-07-29 | [#6514](https://github.com/vllm-project/vllm/pull/6514) | merged | [Model] Initialize support for InternVL2 series models | `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py` |
| 2024-08-03 | [#7067](https://github.com/vllm-project/vllm/pull/7067) | merged | [Model] Refactor and decouple weight loading logic for InternVL2 model | `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py` |
| 2024-08-07 | [#7164](https://github.com/vllm-project/vllm/pull/7164) | merged | [Bugfix] Fix input processor for InternVL2 model | `vllm/model_executor/models/internvl.py`, `tests/models/test_internvl.py` |
| 2024-09-05 | [#7860](https://github.com/vllm-project/vllm/pull/7860) | merged | Inclusion of InternVLChatModel In PP_SUPPORTED_MODELS(Pipeline Parallelism) | `vllm/model_executor/models/internvl.py` |
| 2024-09-07 | [#8201](https://github.com/vllm-project/vllm/pull/8201) | merged | [Model][VLM] Support multi-images inputs for InternVL2 models | `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2024-09-11 | [#8299](https://github.com/vllm-project/vllm/pull/8299) | merged | [Bugfix] Fix InternVL2 vision embeddings process with pipeline parallel | `vllm/model_executor/models/internvl.py` |
| 2024-09-12 | [#8375](https://github.com/vllm-project/vllm/pull/8375) | merged | [Bugfix] Fix InternVL2 inference with various num_patches | `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2024-09-25 | [#8250](https://github.com/vllm-project/vllm/pull/8250) | merged | [BugFix] Propagate 'trust_remote_code' setting in internvl and minicpmv | `vllm/model_executor/models/internvl.py` |
| 2024-09-25 | [#8614](https://github.com/vllm-project/vllm/pull/8614) | merged | [VLM][Bugfix] enable internvl running with num_scheduler_steps > 1 | `vllm/model_executor/models/internvl.py` |
| 2024-09-30 | [#8946](https://github.com/vllm-project/vllm/pull/8946) | merged | [Model] Expose InternVL2 max_dynamic_patch as a mm_processor_kwarg | `vllm/model_executor/models/internvl.py` |
| 2024-10-15 | [#9351](https://github.com/vllm-project/vllm/pull/9351) | merged | [Bugfix] Update InternVL input mapper to support image embeds | `vllm/model_executor/models/internvl.py` |
| 2024-10-22 | [#9528](https://github.com/vllm-project/vllm/pull/9528) | merged | [Model][VLM] Initialize support for Mono-InternVL model | `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/decoder_only/vision_language/test_internvl.py` |
| 2024-11-21 | [#10518](https://github.com/vllm-project/vllm/pull/10518) | merged | [Model] Expose `dynamic_image_size` as mm_processor_kwargs for InternVL2 models | `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2024-12-13 | [#11165](https://github.com/vllm-project/vllm/pull/11165) | merged | [V1][VLM] Fix edge case bug for InternVL2 | `vllm/model_executor/models/internvl.py` |
| 2025-02-04 | [#12553](https://github.com/vllm-project/vllm/pull/12553) | merged | [VLM] Merged multi-modal processor for InternVL-based models | `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py` |
| 2025-03-13 | [#14738](https://github.com/vllm-project/vllm/pull/14738) | merged | [VLM] Support loading InternVideo2.5 models as original InternVLChatModel | `vllm/model_executor/models/internvl.py` |
| 2025-03-20 | [#15086](https://github.com/vllm-project/vllm/pull/15086) | merged | [Bugfix] Fix embedding assignment for InternVL-based models | `vllm/model_executor/models/internvl.py` |
| 2025-05-25 | [#18499](https://github.com/vllm-project/vllm/pull/18499) | merged | [VLM] Initialize video input support for InternVL models | `vllm/model_executor/models/internvl.py` |
| 2025-05-29 | [#18842](https://github.com/vllm-project/vllm/pull/18842) | merged | [LoRA] Add LoRA support for InternVL | `vllm/model_executor/models/internvl.py` |
| 2025-07-29 | [#21684](https://github.com/vllm-project/vllm/pull/21684) | merged | Migrate InternVLImageInputs and InternVLVideoInputs to TensorSchema | `vllm/model_executor/models/internvl.py` |
| 2025-08-26 | [#23658](https://github.com/vllm-project/vllm/pull/23658) | merged | [Model] Enable video support for InternVL3.5 models | `vllm/model_executor/models/internvl.py` |
| 2025-08-27 | [#23742](https://github.com/vllm-project/vllm/pull/23742) | merged | [Model] Enable native HF format InternVL support | `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md` |
| 2025-09-10 | [#24519](https://github.com/vllm-project/vllm/pull/24519) | merged | [Model] Limit CPU threads for image transformations in InternVL to reduce cpu contention. | `vllm/model_executor/models/internvl.py` |
| 2025-09-18 | [#23909](https://github.com/vllm-project/vllm/pull/23909) | merged | [Model] enable data parallel for InternVL vision encoder | `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py` |
| 2025-10-03 | [#26153](https://github.com/vllm-project/vllm/pull/26153) | merged | [Model] Use `merge_by_field_config` for MM models (InternVL family) | `vllm/model_executor/models/internvl.py` |
| 2026-01-23 | [#32397](https://github.com/vllm-project/vllm/pull/32397) | merged | [Model] Enable LoRA support for internvl2 | `vllm/model_executor/models/internvl.py` |
| 2026-03-17 | [#37260](https://github.com/vllm-project/vllm/pull/37260) | merged | [1/2] Move InternVL-based processors | `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py` |
| 2026-03-18 | [#37324](https://github.com/vllm-project/vllm/pull/37324) | merged | [2/3] Refactor InternVL-based processors | `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2026-03-26 | [#38049](https://github.com/vllm-project/vllm/pull/38049) | merged | [Model] Add torch.compile support for InternVL vision encoder | `vllm/model_executor/models/intern_vit.py` |

## 逐 PR diff 审计卡

### PR #6514 - [Model] Initialize support for InternVL2 series models

- 链接: https://github.com/vllm-project/vllm/pull/6514
- 状态/时间: merged / 2024-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；关联提交 `7cbd9ec7a9bf`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+1042/-6，可读 patch 1164 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[Model] Initialize support for InternVL2 series models」，变更集中在 `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE FIX #4393 FIX #6321 (*link existing issues this PR will resolve*) **This PR aims to add support for InternVL2 series models:** - [x] Port InternV...
- 实现要点: `vllm/model_executor/models/internvl.py` added +471/-0 (471 lines); hunks: -0,0 +1,471; symbols: InternVLImagePixelInputs, build_transform, find_closest_aspect_ratio, calculate_num_blocks，涉及 `InternVLImagePixelInputs, build_transform, find_closest_aspect_ratio`；`vllm/model_executor/models/intern_vit.py` added +270/-0 (270 lines); hunks: -0,0 +1,270; symbols: InternVisionEmbeddings, __init__, _get_pos_embed, forward，涉及 `InternVisionEmbeddings, __init__, _get_pos_embed`；`tests/models/test_internvl.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: InternVLProcessor, __init__, __call__, generate，涉及 `InternVLProcessor, __init__, __call__`；`vllm/transformers_utils/configs/internvl.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: InternVLChatConfig, __init__，涉及 `InternVLChatConfig, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` added +471/-0 (471 lines); hunks: -0,0 +1,471; symbols: InternVLImagePixelInputs, build_transform, find_closest_aspect_ratio, calculate_num_blocks
  - `vllm/model_executor/models/intern_vit.py` added +270/-0 (270 lines); hunks: -0,0 +1,270; symbols: InternVisionEmbeddings, __init__, _get_pos_embed, forward
  - `tests/models/test_internvl.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: InternVLProcessor, __init__, __call__, generate
  - `vllm/transformers_utils/configs/internvl.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: InternVLChatConfig, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -0,0 +1,471 @@
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
+# Licensed under The MIT License [see LICENSE for details]
+# --------------------------------------------------------
diff -- vllm/model_executor/models/intern_vit.py
@@ -0,0 +1,270 @@
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
+# Licensed under The MIT License [see LICENSE for details]
+# --------------------------------------------------------
diff -- tests/models/test_internvl.py
@@ -0,0 +1,201 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` added +471/-0; `vllm/model_executor/models/intern_vit.py` added +270/-0; `vllm/transformers_utils/configs/internvl.py` added +51/-0
  - tests: `tests/models/test_internvl.py` added +201/-0
- 验证与风险: diff 自带测试面 `tests/models/test_internvl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7067 - [Model] Refactor and decouple weight loading logic for InternVL2 model

- 链接: https://github.com/vllm-project/vllm/pull/7067
- 状态/时间: merged / 2024-08-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；关联提交 `0c25435daa0a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+38/-55，可读 patch 123 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 调整模型相关实现，标题为「[Model] Refactor and decouple weight loading logic for InternVL2 model」，变更集中在 `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE - The `load_weights` method for `InternVLChatModel` is highly coupled with weight loading logic for `Qwen2`, `Llama` and `InternLM2`, which is di...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +28/-54 (82 lines); hunks: -4,6 +4,7; -414,58 +415,31 @@ def sample(; symbols: sample, load_weights, _filter_weights，涉及 `sample, load_weights, _filter_weights`；`vllm/model_executor/models/intern_vit.py` modified +10/-1 (11 lines); hunks: -4,7 +4,7; -16,6 +16,7; symbols: forward, load_weights，涉及 `forward, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +28/-54 (82 lines); hunks: -4,6 +4,7; -414,58 +415,31 @@ def sample(; symbols: sample, load_weights, _filter_weights
  - `vllm/model_executor/models/intern_vit.py` modified +10/-1 (11 lines); hunks: -4,7 +4,7; -16,6 +16,7; symbols: forward, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -4,6 +4,7 @@
+import itertools
@@ -414,58 +415,31 @@ def sample(
-    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        stacked_params_mapping = [
-            # (param_name, shard_name, shard_id)
-            (".qkv_proj", ".q_proj", "q"),
diff -- vllm/model_executor/models/intern_vit.py
@@ -4,7 +4,7 @@
-from typing import Optional
+from typing import Iterable, Optional, Tuple
@@ -16,6 +16,7 @@
+from vllm.model_executor.model_loader.weight_utils import default_weight_loader
@@ -268,3 +269,11 @@ def forward(
+    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +28/-54; `vllm/model_executor/models/intern_vit.py` modified +10/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7164 - [Bugfix] Fix input processor for InternVL2 model

- 链接: https://github.com/vllm-project/vllm/pull/7164
- 状态/时间: merged / 2024-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `b764547616e6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+73/-34，可读 patch 211 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix input processor for InternVL2 model」，变更集中在 `vllm/model_executor/models/internvl.py`, `tests/models/test_internvl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE FIX #7160 FIX #7272 - This PR also aims to make some small refactor to fix some hidden issues. ~~So I marked it as a draft.~~ - Since most of pro...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +54/-30 (84 lines); hunks: -38,9 +38,6; -84,11 +81,9 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, wi...; symbols: InternVLImagePixelInputs, find_closest_aspect_ratio, calculate_num_blocks，涉及 `InternVLImagePixelInputs, find_closest_aspect_ratio, calculate_num_blocks`；`tests/models/test_internvl.py` modified +19/-4 (23 lines); hunks: -5,6 +5,7; -26,10 +27,15; symbols: __init__, __call__，涉及 `__init__, __call__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +54/-30 (84 lines); hunks: -38,9 +38,6; -84,11 +81,9 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, wi...; symbols: InternVLImagePixelInputs, find_closest_aspect_ratio, calculate_num_blocks
  - `tests/models/test_internvl.py` modified +19/-4 (23 lines); hunks: -5,6 +5,7; -26,10 +27,15; symbols: __init__, __call__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -38,9 +38,6 @@
-MAX_IMAGE_FEATURE_SIZE_WIDTH = 3000
-MAX_IMAGE_FEATURE_SIZE_HEIGHT = 500
@@ -84,11 +81,9 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
-def calculate_num_blocks(orig_width: int,
-                         orig_height: int,
-                         min_num=1,
diff -- tests/models/test_internvl.py
@@ -5,6 +5,7 @@
+from transformers import AutoConfig
@@ -26,10 +27,15 @@
+DOWNLOAD_PATTERN = ["*.json", "*.py", "*.safetensors", "*.txt", "*.model"]
-    snapshot_download("OpenGVLab/InternVL2-1B"),
-    snapshot_download("OpenGVLab/InternVL2-2B"),
-    # snapshot_download("OpenGVLab/InternVL2-4B"),  # broken
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +54/-30
  - tests: `tests/models/test_internvl.py` modified +19/-4
- 验证与风险: diff 自带测试面 `tests/models/test_internvl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7860 - Inclusion of InternVLChatModel In PP_SUPPORTED_MODELS(Pipeline Parallelism)

- 链接: https://github.com/vllm-project/vllm/pull/7860
- 状态/时间: merged / 2024-09-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `8685ba1a1ec0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+90/-35，可读 patch 266 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「Inclusion of InternVLChatModel In PP_SUPPORTED_MODELS(Pipeline Parallelism)」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE Hi Folks, This PR is completed based on the 7168. This @andoorve PR includes the changes needed for the Add remaining model PP support, This PR 7...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +3/-1 (4 lines); hunks: -341,6 +341,8 @@ def __init__(self,; -461,7 +463,7 @@ def forward(; symbols: __init__, pixel_shuffle, forward，涉及 `__init__, pixel_shuffle, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +3/-1 (4 lines); hunks: -341,6 +341,8 @@ def __init__(self,; -461,7 +463,7 @@ def forward(; symbols: __init__, pixel_shuffle, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -341,6 +341,8 @@ def __init__(self,
+        self.make_empty_intermediate_tensors = (
+            self.language_model.make_empty_intermediate_tensors)
@@ -461,7 +463,7 @@ def forward(
-                                                  None,
+                                                  intermediate_tensors,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +3/-1
- 验证与风险: diff 自带测试面 `tests/distributed/test_pipeline_parallel.py`, `tests/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8201 - [Model][VLM] Support multi-images inputs for InternVL2 models

- 链接: https://github.com/vllm-project/vllm/pull/8201
- 状态/时间: merged / 2024-09-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `e807125936a9`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+199/-57，可读 patch 482 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[Model][VLM] Support multi-images inputs for InternVL2 models」，变更集中在 `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE FIX #xxxx (*link existing issues this PR will resolve*) **TODO** - [x] Initialize multi-images inputs support for InternVL2 - [x] ~~Update `_plac...
- 实现要点: `tests/models/test_internvl.py` modified +73/-19 (92 lines); hunks: -1,5 +1,5; -9,7 +9,8; symbols: generate, run_test, __init__, __call__，涉及 `generate, run_test, __init__`；`vllm/model_executor/models/internvl.py` modified +46/-14 (60 lines); hunks: -5,6 +5,7; -26,6 +27,7; symbols: find_closest_aspect_ratio, calculate_num_blocks, dynamic_preprocess, input_processor_for_internvl，涉及 `find_closest_aspect_ratio, calculate_num_blocks, dynamic_preprocess`。
- 代码 diff 细节:
  - `tests/models/test_internvl.py` modified +73/-19 (92 lines); hunks: -1,5 +1,5; -9,7 +9,8; symbols: generate, run_test, __init__, __call__
  - `vllm/model_executor/models/internvl.py` modified +46/-14 (60 lines); hunks: -5,6 +5,7; -26,6 +27,7; symbols: find_closest_aspect_ratio, calculate_num_blocks, dynamic_preprocess, input_processor_for_internvl
- 关键代码摘录:

```diff
diff -- tests/models/test_internvl.py
@@ -1,5 +1,5 @@
-from typing import List, Optional, Tuple, Type
+from typing import List, Optional, Tuple, Type, Union
@@ -9,7 +9,8 @@
-from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
+from ..conftest import (IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner,
+                        _ImageAssets)
diff -- vllm/model_executor/models/internvl.py
@@ -5,6 +5,7 @@
+import re
@@ -26,6 +27,7 @@
+from vllm.utils import is_list_of
@@ -95,8 +97,8 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
-                         max_num: int,
-                         image_size: int) -> Tuple[int, int, int]:
```

- 已读文件:
  - tests: `tests/models/test_internvl.py` modified +73/-19
  - runtime: `vllm/model_executor/models/internvl.py` modified +46/-14
- 验证与风险: diff 自带测试面 `tests/models/test_internvl.py`, `tests/models/test_phi3v.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8299 - [Bugfix] Fix InternVL2 vision embeddings process with pipeline parallel

- 链接: https://github.com/vllm-project/vllm/pull/8299
- 状态/时间: merged / 2024-09-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `1230263e161c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-3，可读 patch 48 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix InternVL2 vision embeddings process with pipeline parallel」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE FIX #8275 - For InternVL2 with PP, we only need to process image input on first rank. - This PR fixed the error raised by image input processing...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +2/-1 (3 lines); hunks: -17,6 +17,7; -480,7 +481,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +2/-1 (3 lines); hunks: -17,6 +17,7; -480,7 +481,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -17,6 +17,7 @@
+from vllm.distributed import get_pp_group
@@ -480,7 +481,7 @@ def forward(
-        if image_input is not None:
+        if image_input is not None and get_pp_group().is_first_rank:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/distributed/test_pipeline_parallel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8375 - [Bugfix] Fix InternVL2 inference with various num_patches

- 链接: https://github.com/vllm-project/vllm/pull/8375
- 状态/时间: merged / 2024-09-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `e56bf2774158`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+39/-3，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[Bugfix] Fix InternVL2 inference with various num_patches」，变更集中在 `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE FIX #8361 (*link existing issues this PR will resolve*) FIX #8369 **TODO** - [x] Add test to cover these cases. **BEFORE SUBMITTING, PLEASE READ...
- 实现要点: `tests/models/test_internvl.py` modified +35/-0 (35 lines); hunks: -331,6 +331,41 @@ def test_multi_images_models(hf_runner, vllm_runner, image_...; symbols: test_multi_images_models, test_different_num_patches，涉及 `test_multi_images_models, test_different_num_patches`；`vllm/model_executor/models/internvl.py` modified +4/-3 (7 lines); hunks: -270,14 +270,14 @@ def input_mapper_for_internvl(ctx: InputContext, data: obj...; -449,11 +449,12 @@ def _parse_and_validate_image_input(; symbols: input_mapper_for_internvl, _parse_and_validate_image_input，涉及 `input_mapper_for_internvl, _parse_and_validate_image_input`。
- 代码 diff 细节:
  - `tests/models/test_internvl.py` modified +35/-0 (35 lines); hunks: -331,6 +331,41 @@ def test_multi_images_models(hf_runner, vllm_runner, image_...; symbols: test_multi_images_models, test_different_num_patches
  - `vllm/model_executor/models/internvl.py` modified +4/-3 (7 lines); hunks: -270,14 +270,14 @@ def input_mapper_for_internvl(ctx: InputContext, data: obj...; -449,11 +449,12 @@ def _parse_and_validate_image_input(; symbols: input_mapper_for_internvl, _parse_and_validate_image_input
- 关键代码摘录:

```diff
diff -- tests/models/test_internvl.py
@@ -331,6 +331,41 @@ def test_multi_images_models(hf_runner, vllm_runner, image_assets, model,
+@pytest.mark.parametrize("model", ["OpenGVLab/InternVL2-2B"])
+@pytest.mark.parametrize("size_factors", [[0.5, 1.0]])
+@pytest.mark.parametrize("dtype", [target_dtype])
+@pytest.mark.parametrize("max_tokens", [128])
+@pytest.mark.parametrize("num_logprobs", [5])
+@torch.inference_mode()
diff -- vllm/model_executor/models/internvl.py
@@ -270,14 +270,14 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):
+        # we can't stack here because the images may have different num_patches
-        data = torch.stack(data)
@@ -449,11 +449,12 @@ def _parse_and_validate_image_input(
+            # We need to flatten (B, N, P) to (B*N*P),
+            # so we call flatten_bn twice.
-                    flatten_bn(pixel_values, concat=True).flatten(0, 1)),
```

- 已读文件:
  - tests: `tests/models/test_internvl.py` modified +35/-0
  - runtime: `vllm/model_executor/models/internvl.py` modified +4/-3
- 验证与风险: diff 自带测试面 `tests/models/test_internvl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8250 - [BugFix] Propagate 'trust_remote_code' setting in internvl and minicpmv

- 链接: https://github.com/vllm-project/vllm/pull/8250
- 状态/时间: merged / 2024-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `e3dd0692fa2c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+126/-41，可读 patch 343 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 修复已暴露的启动、加载、解析或数值问题，标题为「[BugFix] Propagate 'trust_remote_code' setting in internvl and minicpmv」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：For MiniCPM-V model, the token ids needed for compute image bounds are piped through the preprocessing pipeline via `MiniCPMVImageInput` and `MultiModalInputs`. --- section, mar...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +9/-6 (15 lines); hunks: -230,8 +230,9 @@ def input_processor_for_internvl(ctx: InputContext, llm_inpu...; -278,8 +279,9 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):; symbols: input_processor_for_internvl, input_mapper_for_internvl, dummy_data_for_internvl，涉及 `input_processor_for_internvl, input_mapper_for_internvl, dummy_data_for_internvl`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +9/-6 (15 lines); hunks: -230,8 +230,9 @@ def input_processor_for_internvl(ctx: InputContext, llm_inpu...; -278,8 +279,9 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):; symbols: input_processor_for_internvl, input_mapper_for_internvl, dummy_data_for_internvl
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -230,8 +230,9 @@ def input_processor_for_internvl(ctx: InputContext, llm_inputs: LLMInputs):
-    tokenizer = cached_get_tokenizer(model_config.tokenizer,
-                                     trust_remote_code=True)
+    tokenizer = cached_get_tokenizer(
+        model_config.tokenizer,
+        trust_remote_code=model_config.trust_remote_code)
@@ -278,8 +279,9 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +9/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/minicpmv.py`, `vllm/model_executor/models/qwen.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8614 - [VLM][Bugfix] enable internvl running with num_scheduler_steps > 1

- 链接: https://github.com/vllm-project/vllm/pull/8614
- 状态/时间: merged / 2024-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `0c4d2ad5e641`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[VLM][Bugfix] enable internvl running with num_scheduler_steps > 1」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE A small fix for InternVL while running with num_scheduler_steps > 1, before this PR, set `num_scheduler_steps > 1` for InternVL, will raise an er...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +6/-1 (7 lines); hunks: -19,7 +19,7; -376,6 +376,11 @@ def __init__(self,; symbols: __init__, pixel_shuffle，涉及 `__init__, pixel_shuffle`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +6/-1 (7 lines); hunks: -19,7 +19,7; -376,6 +376,11 @@ def __init__(self,; symbols: __init__, pixel_shuffle
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -19,7 +19,7 @@
-from vllm.model_executor.layers.sampler import SamplerOutput
+from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
@@ -376,6 +376,11 @@ def __init__(self,
+        if hasattr(self.language_model, "sampler"):
+            self.sampler = self.language_model.sampler
+        else:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8946 - [Model] Expose InternVL2 max_dynamic_patch as a mm_processor_kwarg

- 链接: https://github.com/vllm-project/vllm/pull/8946
- 状态/时间: merged / 2024-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `2ae25f79cf1e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+90/-61，可读 patch 252 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 修复已暴露的启动、加载、解析或数值问题，标题为「[Model] Expose InternVL2 max_dynamic_patch as a mm_processor_kwarg」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE - Fix the bug mentioned in https://github.com/vllm-project/vllm/pull/6514#issuecomment-2378785414 - Expose InternVL2 max_dynamic_patch as require...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +89/-61 (150 lines); hunks: -5,8 +5,9; -122,6 +123,20 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, dynamic_preprocess, image_to_pixel_values，涉及 `calculate_num_blocks, calculate_num_blocks_wrapper, dynamic_preprocess`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +89/-61 (150 lines); hunks: -5,8 +5,9; -122,6 +123,20 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, dynamic_preprocess, image_to_pixel_values
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -5,8 +5,9 @@
-from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
-                    TypedDict, Union)
+from functools import partial
+from typing import (Any, Dict, Iterable, List, Literal, Mapping, Optional,
+                    Tuple, TypedDict, Union)
@@ -122,6 +123,20 @@ def calculate_num_blocks(orig_width: int, orig_height: int, min_num: int,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +89/-61
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9351 - [Bugfix] Update InternVL input mapper to support image embeds

- 链接: https://github.com/vllm-project/vllm/pull/9351
- 状态/时间: merged / 2024-10-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `55e081fbad29`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[Bugfix] Update InternVL input mapper to support image embeds」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：The InternVL input mapper originally only outputs MultiModalInputs with pixel values. However, the model itself has support for image embedding inputs and this change is needed...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +2/-0 (2 lines); hunks: -342,6 +342,8 @@ def input_mapper(; symbols: input_mapper，涉及 `input_mapper`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +2/-0 (2 lines); hunks: -342,6 +342,8 @@ def input_mapper(; symbols: input_mapper
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -342,6 +342,8 @@ def input_mapper(
+        else:
+            return MultiModalInputs({"image_embeds": data})
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9528 - [Model][VLM] Initialize support for Mono-InternVL model

- 链接: https://github.com/vllm-project/vllm/pull/9528
- 状态/时间: merged / 2024-10-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；关联提交 `bb392ea2d2bf`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+254/-28，可读 patch 387 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[Model][VLM] Initialize support for Mono-InternVL model」，变更集中在 `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/decoder_only/vision_language/test_internvl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE FIX #xxxx (*link existing issues this PR will resolve*) - This PR adds support for Mono-InternVL series models: OpenGVLab/Mono-InternVL-2B, a mon...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +42/-19 (61 lines); hunks: -21,7 +21,8; -427,13 +428,9 @@ def __init__(self,; symbols: __init__, sampler, _init_vision_model, _init_mlp1，涉及 `__init__, sampler, _init_vision_model`；`vllm/model_executor/models/intern_vit.py` modified +31/-0 (31 lines); hunks: -97,6 +97,37 @@ def forward(self, pixel_values: torch.FloatTensor) -> torch.T...; symbols: forward, InternVisionPatchModel, __init__, get_input_embeddings，涉及 `forward, InternVisionPatchModel, __init__`；`tests/models/decoder_only/vision_language/test_internvl.py` modified +13/-8 (21 lines); hunks: -7,7 +7,6; -19,15 +18,20; symbols: generate, run_awq_test，涉及 `generate, run_awq_test`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +42/-19 (61 lines); hunks: -21,7 +21,8; -427,13 +428,9 @@ def __init__(self,; symbols: __init__, sampler, _init_vision_model, _init_mlp1
  - `vllm/model_executor/models/intern_vit.py` modified +31/-0 (31 lines); hunks: -97,6 +97,37 @@ def forward(self, pixel_values: torch.FloatTensor) -> torch.T...; symbols: forward, InternVisionPatchModel, __init__, get_input_embeddings
  - `tests/models/decoder_only/vision_language/test_internvl.py` modified +13/-8 (21 lines); hunks: -7,7 +7,6; -19,15 +18,20; symbols: generate, run_awq_test
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -21,7 +21,8 @@
-from vllm.model_executor.models.intern_vit import InternVisionModel
+from vllm.model_executor.models.intern_vit import (InternVisionModel,
+                                                   InternVisionPatchModel)
@@ -427,13 +428,9 @@ def __init__(self,
-        vision_feature_layer = self.select_layer
-        if vision_feature_layer < 0:
diff -- vllm/model_executor/models/intern_vit.py
@@ -97,6 +97,37 @@ def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
+class InternVisionPatchModel(nn.Module):
+    def __init__(self, config: PretrainedConfig):
+        super().__init__()
+        self.config = config
+        self.embeddings = InternVisionEmbeddings(config)
+    def get_input_embeddings(self):
diff -- tests/models/decoder_only/vision_language/test_internvl.py
@@ -7,7 +7,6 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +42/-19; `vllm/model_executor/models/intern_vit.py` modified +31/-0
  - tests: `tests/models/decoder_only/vision_language/test_internvl.py` modified +13/-8
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_internvl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10518 - [Model] Expose `dynamic_image_size` as mm_processor_kwargs for InternVL2 models

- 链接: https://github.com/vllm-project/vllm/pull/10518
- 状态/时间: merged / 2024-11-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `d5ec121f95f5`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+255/-14，可读 patch 350 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 修复已暴露的启动、加载、解析或数值问题，标题为「[Model] Expose `dynamic_image_size` as mm_processor_kwargs for InternVL2 models」，变更集中在 `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py`, `vllm/model_executor/models/internvl.py`。PR 描述补充为：FIX #9989 - https://github.com/vllm-project/vllm/issues/9989#issuecomment-2488303299 - Note that `dynamic_image_size=False` is equivalent to `max_dynamic_patch=1`, means that im...
- 实现要点: `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py` added +206/-0 (206 lines); hunks: -0,0 +1,206; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override，涉及 `input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens`；`vllm/model_executor/models/internvl.py` modified +49/-14 (63 lines); hunks: -123,8 +123,15 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; -183,10 +190,17 @@ def image_to_pixel_values(image: Image.Image, input_size:...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, image_to_pixel_values, image_to_pixel_values_wrapper，涉及 `calculate_num_blocks, calculate_num_blocks_wrapper, image_to_pixel_values`。
- 代码 diff 细节:
  - `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py` added +206/-0 (206 lines); hunks: -0,0 +1,206; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override
  - `vllm/model_executor/models/internvl.py` modified +49/-14 (63 lines); hunks: -123,8 +123,15 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; -183,10 +190,17 @@ def image_to_pixel_values(image: Image.Image, input_size:...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, image_to_pixel_values, image_to_pixel_values_wrapper
- 关键代码摘录:

```diff
diff -- tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py
@@ -0,0 +1,206 @@
+"""Tests for InternVL's multimodal preprocessing kwargs."""
+from typing import Callable, Optional
+import pytest
+from transformers import AutoTokenizer
+from vllm.inputs import InputContext, token_inputs
+from vllm.multimodal import MultiModalRegistry
diff -- vllm/model_executor/models/internvl.py
@@ -123,8 +123,15 @@ def calculate_num_blocks(orig_width: int, orig_height: int, min_num: int,
-def calculate_num_blocks_wrapper(hf_config: PretrainedConfig,
-                                 max_dynamic_patch: Optional[int] = None):
+def calculate_num_blocks_wrapper(
+    hf_config: PretrainedConfig,
+    max_dynamic_patch: Optional[int] = None,
+    dynamic_image_size: Optional[bool] = None,
```

- 已读文件:
  - tests: `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py` added +206/-0
  - runtime: `vllm/model_executor/models/internvl.py` modified +49/-14
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11165 - [V1][VLM] Fix edge case bug for InternVL2

- 链接: https://github.com/vllm-project/vllm/pull/11165
- 状态/时间: merged / 2024-12-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `969da7d70bc0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 修复已暴露的启动、加载、解析或数值问题，标题为「[V1][VLM] Fix edge case bug for InternVL2」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：When there's only one image in the batch, the `image_embeds` needs to be reshaped to `[feature_size, hidden_size]` from `[num_patches, feature_size_per_patch, hidden_size]` befo...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -669,8 +669,11 @@ def _process_image_input(; symbols: _process_image_input，涉及 `_process_image_input`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -669,8 +669,11 @@ def _process_image_input(; symbols: _process_image_input
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -669,8 +669,11 @@ def _process_image_input(
+        # Only one image in the current batch
-            image_embeds = image_embeds.unsqueeze(0)
+            image_embeds = image_embeds.view(
+                -1, self.config.text_config.hidden_size).unsqueeze(0)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12553 - [VLM] Merged multi-modal processor for InternVL-based models

- 链接: https://github.com/vllm-project/vllm/pull/12553
- 状态/时间: merged / 2025-02-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_internvl.py`, `vllm/model_executor/models/internvl.py`；关联提交 `d1ca7df84d9f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 34 个文件，+1434/-986，可读 patch 3135 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 调整模型相关实现，标题为「[VLM] Merged multi-modal processor for InternVL-based models」，变更集中在 `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py`。PR 描述补充为：Update InternVL (similarly, H2OVL and NVLM-D) to use the merged multi-modal processor. **Note:** `BaseProcessingInfo.get_mm_max_tokens_per_item` now takes a `mm_counts` argument...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +503/-320 (823 lines); hunks: -6,35 +6,37; -75,22 +77,27 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, build_transform, find_closest_aspect_ratio，涉及 `InternVLImageEmbeddingInputs, build_transform, find_closest_aspect_ratio`；`tests/models/multimodal/processing/test_internvl.py` modified +32/-175 (207 lines); hunks: -1,207 +1,64; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override，涉及 `input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +503/-320 (823 lines); hunks: -6,35 +6,37; -75,22 +77,27 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, build_transform, find_closest_aspect_ratio
  - `tests/models/multimodal/processing/test_internvl.py` modified +32/-175 (207 lines); hunks: -1,207 +1,64; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -6,35 +6,37 @@
-import re
-from functools import cached_property, partial
+from abc import ABC, abstractmethod
+from functools import cached_property
-                    TypedDict, Union)
+                    TypedDict, TypeVar, Union)
diff -- tests/models/multimodal/processing/test_internvl.py
@@ -1,207 +1,64 @@
-from typing import Callable, Optional
+from typing import Optional
-from transformers import AutoTokenizer
-from vllm.inputs import InputContext, token_inputs
-from vllm.multimodal import MultiModalRegistry
+from vllm.multimodal import MULTIMODAL_REGISTRY
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +503/-320
  - tests: `tests/models/multimodal/processing/test_internvl.py` modified +32/-175
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_h2ovl.py`, `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14738 - [VLM] Support loading InternVideo2.5 models as original InternVLChatModel

- 链接: https://github.com/vllm-project/vllm/pull/14738
- 状态/时间: merged / 2025-03-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `b1cc4dfef57a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-3，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[VLM] Support loading InternVideo2.5 models as original InternVLChatModel」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：FIX #14180 Separate from #14688 - Note that hierarchical token compression (HiCo) technology newly added in InternVideo hasn't supported yet, so the outputs will be different fr...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +8/-1 (9 lines); hunks: -981,5 +981,12 @@ def sample(; symbols: sample, load_weights，涉及 `sample, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +8/-1 (9 lines); hunks: -981,5 +981,12 @@ def sample(; symbols: sample, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -981,5 +981,12 @@ def sample(
-        loader = AutoWeightsLoader(self)
+        # unused modules appear in OpenGVLab/InternVideo2_5_Chat_8B
+        skip_prefixes = [
+            "action_embed", "temporal_embed", "track_embed",
+            "track_embed_decoder", "box_token", "cg_criterion", "cg_model",
+            "loc_encoder", "loc_decoder", "sam", "temporal_token",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +8/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15086 - [Bugfix] Fix embedding assignment for InternVL-based models

- 链接: https://github.com/vllm-project/vllm/pull/15086
- 状态/时间: merged / 2025-03-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `ffa443afedd3`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+123/-106，可读 patch 488 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix embedding assignment for InternVL-based models」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：Similar to #14980, this PR fixes the mismatch between multimodal embeddings and `PlaceholderRange` indices for InternVL-based models. FIX https://github.com/vllm-project/vllm/is...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +104/-69 (173 lines); hunks: -9,14 +9,13; -36,10 +35,12; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, image_token_id, get_image_repl_features，涉及 `InternVLImagePixelInputs, InternVLImageEmbeddingInputs, image_token_id`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +104/-69 (173 lines); hunks: -9,14 +9,13; -36,10 +35,12; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, image_token_id, get_image_repl_features
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -9,14 +9,13 @@
-from typing import (List, Literal, Optional, Set, Tuple, TypedDict, TypeVar,
-                    Union)
+from typing import Literal, Optional, Set, Tuple, TypedDict, TypeVar, Union
-from transformers import BatchFeature, PretrainedConfig, TensorType
+from transformers import BatchEncoding, PretrainedConfig, TensorType
@@ -36,10 +35,12 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +104/-69
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma3_mm.py`, `vllm/model_executor/models/h2ovl.py`, `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18499 - [VLM] Initialize video input support for InternVL models

- 链接: https://github.com/vllm-project/vllm/pull/18499
- 状态/时间: merged / 2025-05-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `75f81750f3a9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+596/-62，可读 patch 940 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[VLM] Initialize video input support for InternVL models」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：FIX #18381 (*link existing issues this PR will resolve*) - This PR only adds video support to InternVL with Qwen2.5 backbone, because it has unused Qwen2-VL video token in token...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +485/-26 (511 lines); hunks: -8,8 +8,9; -74,6 +75,33 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, InternVLVideoEmbeddingInputs, build_transform，涉及 `InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, InternVLVideoEmbeddingInputs`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +485/-26 (511 lines); hunks: -8,8 +8,9; -74,6 +75,33 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, InternVLVideoEmbeddingInputs, build_transform
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -8,8 +8,9 @@
-from typing import Literal, Optional, TypedDict, TypeVar, Union
+from typing import Any, Literal, Optional, TypedDict, TypeVar, Union
+import numpy.typing as npt
@@ -74,6 +75,33 @@ class InternVLImageEmbeddingInputs(TypedDict):
+class InternVLVideoPixelInputs(TypedDict):
+    type: Literal["pixel_values_videos"]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +485/-26
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18842 - [LoRA] Add LoRA support for InternVL

- 链接: https://github.com/vllm-project/vllm/pull/18842
- 状态/时间: merged / 2025-05-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `34d6c447c4b9`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-2，可读 patch 50 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[LoRA] Add LoRA support for InternVL」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：FIX https://github.com/vllm-project/vllm/issues/18820
- 实现要点: `vllm/model_executor/models/internvl.py` modified +23/-2 (25 lines); hunks: -22,6 +22,7; -36,7 +37,8; symbols: get_video_replacement_internvl, InternVLChatModel, __init__, load_weights，涉及 `get_video_replacement_internvl, InternVLChatModel, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +23/-2 (25 lines); hunks: -22,6 +22,7; -36,7 +37,8; symbols: get_video_replacement_internvl, InternVLChatModel, __init__, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -22,6 +22,7 @@
+from vllm.model_executor.models.module_mapping import MultiModelKeys
@@ -36,7 +37,8 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
+                         SupportsMultiModal, SupportsPP)
@@ -1014,7 +1016,17 @@ def get_video_replacement_internvl(item_idx: int):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +23/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21684 - Migrate InternVLImageInputs and InternVLVideoInputs to TensorSchema

- 链接: https://github.com/vllm-project/vllm/pull/21684
- 状态/时间: merged / 2025-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `f1e2c095ecee`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+49/-62，可读 patch 184 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 调整模型相关实现，标题为「Migrate InternVLImageInputs and InternVLVideoInputs to TensorSchema」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：## Purpose This PR migrates InternVLImageInputs and InternVLVideoInputs from a TypedDict-based definition to a structured TensorSchema model with runtime shape validation. This...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +49/-62 (111 lines); hunks: -9,7 +9,7; -37,6 +37,7; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, InternVLVideoPixelInputs，涉及 `InternVLImagePixelInputs, InternVLImageEmbeddingInputs, InternVLVideoPixelInputs`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +49/-62 (111 lines); hunks: -9,7 +9,7; -37,6 +37,7; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, InternVLVideoPixelInputs
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -9,7 +9,7 @@
-from typing import Any, Literal, Optional, TypedDict, TypeVar, Union
+from typing import Annotated, Any, Literal, Optional, TypeVar, Union
@@ -37,6 +37,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
@@ -51,54 +52,60 @@
-class InternVLImagePixelInputs(TypedDict):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +49/-62
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23658 - [Model] Enable video support for InternVL3.5 models

- 链接: https://github.com/vllm-project/vllm/pull/23658
- 状态/时间: merged / 2025-08-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `9816b81f5f9f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+22/-7，可读 patch 71 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[Model] Enable video support for InternVL3.5 models」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：## Purpose - Fix https://github.com/vllm-project/vllm/issues/23638 - Update InternVL model's video token map to enable video support for InternVL3.5 backbone - Update documentat...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +7/-3 (10 lines); hunks: -855,9 +855,13 @@ def get_supported_mm_limits(self):; symbols: get_supported_mm_limits, get_video_token, get_num_frames_with_most_features，涉及 `get_supported_mm_limits, get_video_token, get_num_frames_with_most_features`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +7/-3 (10 lines); hunks: -855,9 +855,13 @@ def get_supported_mm_limits(self):; symbols: get_supported_mm_limits, get_video_token, get_num_frames_with_most_features
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -855,9 +855,13 @@ def get_supported_mm_limits(self):
-        if text_model_type == "qwen2":
-            return "<|video_pad|>"
-        return None
+        video_token_map = {
+            "qwen2": "<|video_pad|>",
+            "qwen3": "<|video_pad|>",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +7/-3
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23742 - [Model] Enable native HF format InternVL support

- 链接: https://github.com/vllm-project/vllm/pull/23742
- 状态/时间: merged / 2025-08-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+18/-16，可读 patch 76 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[Model] Enable native HF format InternVL support」，变更集中在 `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md`。PR 描述补充为：## Purpose - `InternS1ForConditionalGeneration` shares same implementation with `InternVLForConditionalGeneration`, so we can support it natively. FIX #23714 FIX #23730 ## Test...
- 实现要点: `tests/models/multimodal/generation/test_common.py` modified +14/-15 (29 lines); hunks: -222,21 +222,6; -461,6 +446,20；`tests/models/registry.py` modified +2/-1 (3 lines); hunks: -429,6 +429,7 @@ def check_available_online(; -584,7 +585,7 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -629,6 +629,7 @@ These models primarily accept the [`LLM.generate`](./generat...；`vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -217,6 +217,7。
- 代码 diff 细节:
  - `tests/models/multimodal/generation/test_common.py` modified +14/-15 (29 lines); hunks: -222,21 +222,6; -461,6 +446,20
  - `tests/models/registry.py` modified +2/-1 (3 lines); hunks: -429,6 +429,7 @@ def check_available_online(; -584,7 +585,7 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -629,6 +629,7 @@ These models primarily accept the [`LLM.generate`](./generat...
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -217,6 +217,7
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/generation/test_common.py
@@ -222,21 +222,6 @@
-    # Check "auto" with fallback to transformers
-    "internvl-transformers": VLMTestInfo(
-        models=["OpenGVLab/InternVL3-1B-hf"],
-        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
-        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n", # noqa: E501
-        img_idx_to_prompt=lambda idx: "<IMG_CONTEXT>",
diff -- tests/models/registry.py
@@ -429,6 +429,7 @@ def check_available_online(
+    "InternVLForConditionalGeneration": _HfExamplesInfo("OpenGVLab/InternVL3-1B-hf"),    # noqa: E501
@@ -584,7 +585,7 @@ def check_available_online(
-    "TransformersForMultimodalLM": _HfExamplesInfo("OpenGVLab/InternVL3-1B-hf"),
+    "TransformersForMultimodalLM": _HfExamplesInfo("BAAI/Emu3-Chat-hf"),
diff -- docs/models/supported_models.md
@@ -629,6 +629,7 @@ These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
+| `InternVLForConditionalGeneration` | InternVL 3.0 (HF format) | T + I<sup>E+</sup> + V<sup>E+</sup> | `OpenGVLab/InternVL3-1B-hf`, etc. | ✅︎ | ✅︎ | ✅︎ |
diff -- vllm/model_executor/models/registry.py
```

- 已读文件:
  - tests: `tests/models/multimodal/generation/test_common.py` modified +14/-15; `tests/models/registry.py` modified +2/-1
  - docs: `docs/models/supported_models.md` modified +1/-0
  - runtime: `vllm/model_executor/models/registry.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24519 - [Model] Limit CPU threads for image transformations in InternVL to reduce cpu contention.

- 链接: https://github.com/vllm-project/vllm/pull/24519
- 状态/时间: merged / 2025-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `267c80d31f6b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+16/-1，可读 patch 44 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 调整模型相关实现，标题为「[Model] Limit CPU threads for image transformations in InternVL to reduce cpu contention.」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：# Limit CPU threads for image transformations in InternVL to reduce cpu contention. ## Purpose Image transformation operations in Internvl model (which include tensor computatio...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +16/-1 (17 lines); hunks: -7,6 +7,7; -37,6 +38,7; symbols: InternVLVideoEmbeddingInputs, build_transform, apply，涉及 `InternVLVideoEmbeddingInputs, build_transform, apply`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +16/-1 (17 lines); hunks: -7,6 +7,7; -37,6 +38,7; symbols: InternVLVideoEmbeddingInputs, build_transform, apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -7,6 +7,7 @@
+import os
@@ -37,6 +38,7 @@
+from vllm.utils import set_default_torch_num_threads
@@ -115,13 +117,26 @@ class InternVLVideoEmbeddingInputs(TensorSchema):
-    return T.Compose([
+    transform = T.Compose([
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +16/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23909 - [Model] enable data parallel for InternVL vision encoder

- 链接: https://github.com/vllm-project/vllm/pull/23909
- 状态/时间: merged / 2025-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；关联提交 `52bc9d5b3edb`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+80/-33，可读 patch 262 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[Model] enable data parallel for InternVL vision encoder」，变更集中在 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`。PR 描述补充为：## Purpose This PR enables DP for InternVL vision encoder FIX #23876 ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpos...
- 实现要点: `vllm/model_executor/models/intern_vit.py` modified +75/-32 (107 lines); hunks: -25,9 +25,11; -137,6 +139,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -1020,6 +1020,8 @@ def get_video_replacement_internvl(item_idx: int):; -1038,6 +1040,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: get_video_replacement_internvl, InternVLChatModel, get_placeholder_str, __init__，涉及 `get_video_replacement_internvl, InternVLChatModel, get_placeholder_str`。
- 代码 diff 细节:
  - `vllm/model_executor/models/intern_vit.py` modified +75/-32 (107 lines); hunks: -25,9 +25,11; -137,6 +139,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -1020,6 +1020,8 @@ def get_video_replacement_internvl(item_idx: int):; -1038,6 +1040,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: get_video_replacement_internvl, InternVLChatModel, get_placeholder_str, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/intern_vit.py
@@ -25,9 +25,11 @@
+                                               ReplicatedLinear,
+from vllm.multimodal.utils import run_dp_sharded_vision_model
@@ -137,6 +139,7 @@ def __init__(
+        use_data_parallel: bool = False,
@@ -150,23 +153,34 @@ def __init__(
-        self.tp_size = get_tensor_model_parallel_world_size()
diff -- vllm/model_executor/models/internvl.py
@@ -1020,6 +1020,8 @@ def get_video_replacement_internvl(item_idx: int):
+    supports_encoder_tp_data = True
@@ -1038,6 +1040,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
+        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
@@ -1105,7 +1108,7 @@ def _init_vision_model(
-            )
+                use_data_parallel=self.use_data_parallel)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/intern_vit.py` modified +75/-32; `vllm/model_executor/models/internvl.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26153 - [Model] Use `merge_by_field_config` for MM models (InternVL family)

- 链接: https://github.com/vllm-project/vllm/pull/26153
- 状态/时间: merged / 2025-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `f9a8084e4879`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+84/-182，可读 patch 785 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补强部署文档、测试或 CI 验证面，标题为「[Model] Use `merge_by_field_config` for MM models (InternVL family)」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：## Purpose Part of #26149 Also use Intern-S1-mini for the example scripts. ## Test Plan Model and tensor schema tests should pass. I have also run the example script on all mode...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +27/-54 (81 lines); hunks: -17,7 +17,7; -28,7 +28,7; symbols: _preprocess_image, __call__, InternVLProcessor, _preprocess_video，涉及 `_preprocess_image, __call__, InternVLProcessor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +27/-54 (81 lines); hunks: -17,7 +17,7; -28,7 +28,7; symbols: _preprocess_image, __call__, InternVLProcessor, _preprocess_video
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -17,7 +17,7 @@
-from transformers import BatchEncoding, PretrainedConfig, TensorType
+from transformers import BatchFeature, PretrainedConfig, TensorType
@@ -28,7 +28,7 @@
-                                    MultiModalKwargsItems, NestedTensors)
+                                    MultiModalKwargsItems)
@@ -42,8 +42,7 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +27/-54
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32397 - [Model] Enable LoRA support for internvl2

- 链接: https://github.com/vllm-project/vllm/pull/32397
- 状态/时间: merged / 2026-01-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `fec9da0af48d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+16/-3，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[Model] Enable LoRA support for internvl2」，变更集中在 `vllm/model_executor/models/internvl.py`。PR 描述补充为：## Purpose Enable dynamic LoRA adapters on InternVL2 vision tower + connector (part of #31479) by implementing: - `get_num_mm_encoder_tokens()` - `get_num_mm_connector_tokens()`...
- 实现要点: `vllm/model_executor/models/internvl.py` modified +16/-3 (19 lines); hunks: -1086,9 +1086,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -1430,3 +1429,17 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: __init__, get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens，涉及 `__init__, get_mm_mapping, get_num_mm_encoder_tokens`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +16/-3 (19 lines); hunks: -1086,9 +1086,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -1430,3 +1429,17 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: __init__, get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -1086,9 +1086,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
-        self.num_image_token = int(
-            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
-        )
+        self.patch_tokens = (image_size // patch_size) ** 2
+        self.num_image_token = int(self.patch_tokens * (config.downsample_ratio**2))
@@ -1430,3 +1429,17 @@ def get_mm_mapping(self) -> MultiModelKeys:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +16/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37260 - [1/2] Move InternVL-based processors

- 链接: https://github.com/vllm-project/vllm/pull/37260
- 状态/时间: merged / 2026-03-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_internvl.py`, `vllm/model_executor/models/internvl.py`, `vllm/transformers_utils/processors/internvl.py`；关联提交 `f34032433573`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+3252/-3099，可读 patch 6681 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 调整模型相关实现，标题为「[1/2] Move InternVL-based processors」，变更集中在 `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py`。PR 描述补充为：## Purpose Move the Transformers-like processing code from modeling file to `transformers_utils` in order to enhance readability. They will be further refactored in the next PR...
- 实现要点: `vllm/transformers_utils/processors/internvl.py` added +603/-0 (603 lines); hunks: -0,0 +1,603; symbols: build_transform, find_closest_aspect_ratio, resolve_internvl_min_max_num, get_internvl_target_ratios，涉及 `build_transform, find_closest_aspect_ratio, resolve_internvl_min_max_num`；`vllm/model_executor/models/internvl.py` modified +7/-578 (585 lines); hunks: -7,16 +7,13; -28,7 +25,6; symbols: InternVLImagePixelInputs, InternVLVideoEmbeddingInputs, build_transform, find_closest_aspect_ratio，涉及 `InternVLImagePixelInputs, InternVLVideoEmbeddingInputs, build_transform`；`tests/models/multimodal/processing/test_internvl.py` modified +1/-1 (2 lines); hunks: -23,7 +23,7 @@ def _get_expected_num_patches(; symbols: _get_expected_num_patches，涉及 `_get_expected_num_patches`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/internvl.py` added +603/-0 (603 lines); hunks: -0,0 +1,603; symbols: build_transform, find_closest_aspect_ratio, resolve_internvl_min_max_num, get_internvl_target_ratios
  - `vllm/model_executor/models/internvl.py` modified +7/-578 (585 lines); hunks: -7,16 +7,13; -28,7 +25,6; symbols: InternVLImagePixelInputs, InternVLVideoEmbeddingInputs, build_transform, find_closest_aspect_ratio
  - `tests/models/multimodal/processing/test_internvl.py` modified +1/-1 (2 lines); hunks: -23,7 +23,7 @@ def _get_expected_num_patches(; symbols: _get_expected_num_patches
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/internvl.py
@@ -0,0 +1,603 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
diff -- vllm/model_executor/models/internvl.py
@@ -7,16 +7,13 @@
-from abc import ABC, abstractmethod
+from abc import abstractmethod
-from typing import Annotated, Any, Literal, TypeAlias, TypeVar
+from typing import Annotated, Literal, TypeAlias, TypeVar
-import numpy.typing as npt
-import torchvision.transforms as T
diff -- tests/models/multimodal/processing/test_internvl.py
@@ -23,7 +23,7 @@ def _get_expected_num_patches(
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/internvl.py` added +603/-0; `vllm/model_executor/models/internvl.py` modified +7/-578
  - tests: `tests/models/multimodal/processing/test_internvl.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_h2ovl.py`, `tests/models/multimodal/processing/test_internvl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37324 - [2/3] Refactor InternVL-based processors

- 链接: https://github.com/vllm-project/vllm/pull/37324
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`, `vllm/transformers_utils/processors/internvl.py`；关联提交 `99267c23ca51`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+762/-1146，可读 patch 2597 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 调整模型相关实现，标题为「[2/3] Refactor InternVL-based processors」，变更集中在 `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py`。PR 描述补充为：## Purpose Follow-up to #37289 - Split up processing logic into `*ImageProcessor` and `*VideoProcessor`. - Remove unnecessary processor definitions for Eagle2.5 and SkyworkR1V -...
- 实现要点: `vllm/transformers_utils/processors/internvl.py` modified +233/-273 (506 lines); hunks: -7,24 +7,17; -33,7 +26,7; symbols: build_transform, video_to_pixel_values_internvl, BaseInternVLProcessor, InternVLImageProcessor，涉及 `build_transform, video_to_pixel_values_internvl, BaseInternVLProcessor`；`vllm/model_executor/models/internvl.py` modified +85/-46 (131 lines); hunks: -9,6 +9,7; -45,8 +46,9; symbols: BaseInternVLProcessingInfo, get_hf_processor, get_supported_mm_limits, get_num_image_tokens，涉及 `BaseInternVLProcessingInfo, get_hf_processor, get_supported_mm_limits`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/internvl.py` modified +233/-273 (506 lines); hunks: -7,24 +7,17; -33,7 +26,7; symbols: build_transform, video_to_pixel_values_internvl, BaseInternVLProcessor, InternVLImageProcessor
  - `vllm/model_executor/models/internvl.py` modified +85/-46 (131 lines); hunks: -9,6 +9,7; -45,8 +46,9; symbols: BaseInternVLProcessingInfo, get_hf_processor, get_supported_mm_limits, get_num_image_tokens
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/internvl.py
@@ -7,24 +7,17 @@
-from abc import ABC, abstractmethod
-from typing import Any, TypeVar
-from transformers import BatchFeature, PretrainedConfig, TensorType
+from transformers import BatchFeature, TensorType
+from transformers.processing_utils import ProcessorMixin
-from vllm.tokenizers import TokenizerLike
diff -- vllm/model_executor/models/internvl.py
@@ -9,6 +9,7 @@
+from functools import cached_property
@@ -45,8 +46,9 @@
-    BaseInternVLProcessor,
+    InternVLImageProcessor,
+    InternVLVideoProcessor,
@@ -123,7 +125,7 @@ class BaseInternVLProcessingInfo(BaseProcessingInfo):
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/internvl.py` modified +233/-273; `vllm/model_executor/models/internvl.py` modified +85/-46
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38049 - [Model] Add torch.compile support for InternVL vision encoder

- 链接: https://github.com/vllm-project/vllm/pull/38049
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/intern_vit.py`；关联提交 `38de82231023`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+20/-3，可读 patch 51 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[Model] Add torch.compile support for InternVL vision encoder」，变更集中在 `vllm/model_executor/models/intern_vit.py`。PR 描述补充为：# [Model] Add torch.compile support for InternVL vision encoder ## Summary ## Purpose Add `torch.compile` support for the InternVL/InternViT vision encoder using the `@support_t...
- 实现要点: `vllm/model_executor/models/intern_vit.py` modified +11/-2 (13 lines); hunks: -15,6 +15,10; -280,6 +284,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: forward, InternVisionEncoderLayer, __init__，涉及 `forward, InternVisionEncoderLayer, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/intern_vit.py` modified +11/-2 (13 lines); hunks: -15,6 +15,10; -280,6 +284,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: forward, InternVisionEncoderLayer, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/intern_vit.py
@@ -15,6 +15,10 @@
+from vllm.compilation.decorators import (
+    should_torch_compile_mm_encoder,
+    support_torch_compile,
+)
@@ -280,6 +284,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+@support_torch_compile(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/intern_vit.py` modified +11/-2
- 验证与风险: runtime 路径改动集中在 `vllm/config/utils.py`, `vllm/model_executor/models/intern_vit.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
