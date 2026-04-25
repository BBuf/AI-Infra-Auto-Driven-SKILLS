# SGLang InternVL3.5 支持与 PR 历史

本文记录 SGLang 中与 InternVL3.5 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- InternVL3.5 is mostly a processor / encoder / video problem in SGLang.
- Video handling, data-parallel vision execution, and backend compatibility dominate the risk surface.

## 主要代码面

- `sglang/python/sglang/srt/models/internvl.py`

## 已合入 PR

- [#5350](https://github.com/sgl-project/sglang/pull/5350) `Support InternVL3`：Initial InternVL family support that later carried 3.5.
- [#13640](https://github.com/sgl-project/sglang/pull/13640) `Support Piecewise CUDA Graph for InternVL`：Added graph capture support on the encoder path.
- [#13925](https://github.com/sgl-project/sglang/pull/13925) `Support InternVL Vision Encoder Data Parallelism`：Opened the multi-GPU ViT path.
- [#15942](https://github.com/sgl-project/sglang/pull/15942) `Support Video for InternVL3_5`：Extended support to 3.5 video use cases.
- [#19127](https://github.com/sgl-project/sglang/pull/19127) `Support processor and embedding inputs for InternVL`：Hardened processor / embed input interoperability.

## 配套 skill

- `skills/model-optimization/sglang/sglang-internvl35-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-internvl35-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `InternVL3.5`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-04-13 | [#5350](https://github.com/sgl-project/sglang/pull/5350) | merged | Support InternVL3 | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py` |
| 2025-11-20 | [#13640](https://github.com/sgl-project/sglang/pull/13640) | merged | [VLM] Support Piecewise CUDA Graph for InternVL | model wrapper, kernel, scheduler/runtime, tests/benchmarks | `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/internvl.py` |
| 2025-11-25 | [#13925](https://github.com/sgl-project/sglang/pull/13925) | merged | [VLM] Support InternVL Vision Encoder Data Parallelism | model wrapper, multimodal/processor, tests/benchmarks | `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `test/nightly/test_encoder_dp.py` |
| 2025-12-27 | [#15942](https://github.com/sgl-project/sglang/pull/15942) | merged | [VLM] Support Video for InternVL3_5 | model wrapper, multimodal/processor | `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py` |
| 2026-02-21 | [#19127](https://github.com/sgl-project/sglang/pull/19127) | merged | [vlm][internVL] Support processor and embedding inputs for InternVL | model wrapper, multimodal/processor, tests/benchmarks | `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/base_processor.py` |

### 逐 PR 代码 diff 阅读记录

### PR #5350 - Support InternVL3

- 链接：https://github.com/sgl-project/sglang/pull/5350
- 状态/时间：`merged`，created 2025-04-13, merged 2025-05-02；作者 `xiaomin-D`。
- 代码 diff 已读范围：`12` 个文件，`+1728/-9`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：vision, config, kv, processor, attention, spec, cuda, doc, flash, cache。
- 代码 diff 细节：
  - `python/sglang/srt/configs/internvl.py` added +696/-0 (696 lines); hunk: +import copy; 符号: InternLM2Config, to, __init__, _rope_scaling_validation
  - `python/sglang/srt/models/internvl.py` added +670/-0 (670 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: FlashAttention, __init__, forward, InternAttention
  - `python/sglang/srt/managers/multimodal_processors/internvl.py` added +232/-0 (232 lines); hunk: +# Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py; 符号: InternVLImageProcessor, __init__, build_transform, resize_image
  - `python/sglang/lang/chat_template.py` modified +44/-0 (44 lines); hunk: def get_chat_template_by_model_path(model_path):; def get_chat_template_by_model_path(model_path):; 符号: get_chat_template_by_model_path, get_chat_template_by_model_path, match_gemma3_instruct, match_internvl_chat
  - `python/sglang/srt/conversation.py` modified +29/-2 (31 lines); hunk: class SeparatorStyle(IntEnum):; def get_prompt(self) -> str:; 符号: SeparatorStyle, get_prompt, generate_chat_conv, generate_chat_conv
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py`；patch 关键词为 vision, config, kv, processor, attention, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13640 - [VLM] Support Piecewise CUDA Graph for InternVL

- 链接：https://github.com/sgl-project/sglang/pull/13640
- 状态/时间：`merged`，created 2025-11-20, merged 2025-11-21；作者 `yuan-luo`。
- 代码 diff 已读范围：`5` 个文件，`+103/-13`；代码面：model wrapper, kernel, scheduler/runtime, tests/benchmarks；关键词：cache, cuda, attention, spec, test, config, lora, moe, processor, quant。
- 代码 diff 细节：
  - `test/srt/test_piecewise_cuda_graph.py` modified +41/-0 (41 lines); hunk: def test_gsm8k_accuracy(self):; 符号: test_gsm8k_accuracy, TestPiecewiseCudaGraphInternVL25, setUpClass, tearDownClass
  - `python/sglang/srt/managers/mm_utils.py` modified +37/-2 (39 lines); hunk: def should_use_external_mm_preprocess(multimodal_model: nn.Module) -> bool:; 符号: should_use_external_mm_preprocess, resolve_external_mm_data_embedding_funcs, external_mm_preprocess_routine
  - `python/sglang/srt/models/internvl.py` modified +21/-10 (31 lines); hunk: from sglang.srt.layers.attention.vision import SingletonCache, VisionAttention; def __init__(; 符号: __init__, pixel_shuffle, forward, pad_input_ids
  - `python/sglang/srt/model_executor/model_runner.py` modified +3/-0 (3 lines); hunk: from sglang.srt.lora.lora_registry import LoRARef; def forward_extend(; 符号: forward_extend
  - `test/srt/run_suite.py` modified +1/-1 (2 lines); hunk: TestFile("test_original_logprobs.py", 41),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/internvl.py`；patch 关键词为 cache, cuda, attention, spec, test, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/internvl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13925 - [VLM] Support InternVL Vision Encoder Data Parallelism

- 链接：https://github.com/sgl-project/sglang/pull/13925
- 状态/时间：`merged`，created 2025-11-25, merged 2025-11-26；作者 `yuan-luo`。
- 代码 diff 已读范围：`3` 个文件，`+118/-25`；代码面：model wrapper, multimodal/processor, tests/benchmarks；关键词：vision, attention, cache, config, moe, quant, test, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/internvl.py` modified +83/-25 (108 lines); hunk: import torch.nn.functional as F; from sglang.srt.models.qwen2 import Qwen2ForCausalLM; 符号: __init__, __init__, forward, InternMLP
  - `python/sglang/srt/multimodal/mm_utils.py` modified +34/-0 (34 lines); hunk: def get_dp_encoder_lb_assignment(; 符号: get_dp_encoder_lb_assignment, run_dp_sharded_vision_model, run_dp_sharded_mrope_vision_model
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunk: MODELS = [
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `test/nightly/test_encoder_dp.py`；patch 关键词为 vision, attention, cache, config, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `test/nightly/test_encoder_dp.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15942 - [VLM] Support Video for InternVL3_5

- 链接：https://github.com/sgl-project/sglang/pull/15942
- 状态/时间：`merged`，created 2025-12-27, merged 2025-12-30；作者 `yuan-luo`。
- 代码 diff 已读范围：`2` 个文件，`+426/-118`；代码面：model wrapper, multimodal/processor；关键词：cache, config, cuda, moe, processor, spec, vision。
- 代码 diff 细节：
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +418/-118 (536 lines); hunk: # Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py; MultimodalSpecialTokens,; 符号: InternVLImageProcessor, InternVLProcessor, _get_normalize_tensors, __init__
  - `python/sglang/srt/models/internvl.py` modified +8/-0 (8 lines); hunk: def __init__(; def get_image_feature(self, items: List[MultimodalDataItem]):; 符号: __init__, get_image_feature, get_video_feature, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`；patch 关键词为 cache, config, cuda, moe, processor, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19127 - [vlm][internVL] Support processor and embedding inputs for InternVL

- 链接：https://github.com/sgl-project/sglang/pull/19127
- 状态/时间：`merged`，created 2026-02-21, merged 2026-02-27；作者 `jiangyukunok`。
- 代码 diff 已读范围：`4` 个文件，`+282/-7`；代码面：model wrapper, multimodal/processor, tests/benchmarks；关键词：processor, spec, vision, config, cuda, test。
- 代码 diff 细节：
  - `test/registered/vlm/test_vlm_input_format.py` modified +153/-0 (153 lines); hunk: def _processor_output_image_data(self, processor_output):; 符号: _processor_output_image_data, TestInternVLUnderstandsImage, setUpClass, _init_visual
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +109/-1 (110 lines); hunk: from decord import VideoReader, cpu, gpu; def _resolve_video_num_frames(; 符号: _resolve_video_num_frames, _has_special_format, _process_special_format, process_and_combine_mm_data
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +13/-6 (19 lines); hunk: def __init__(; def build_input_ids(self, prompt, img_grid_thw):; 符号: __init__, build_input_ids, load_mm_data, fast_load_mm_data
  - `python/sglang/srt/models/internvl.py` modified +7/-0 (7 lines); hunk: def get_image_feature(self, items: List[MultimodalDataItem]):; 符号: get_image_feature, get_video_feature
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/base_processor.py`；patch 关键词为 processor, spec, vision, config, cuda, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/base_processor.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
