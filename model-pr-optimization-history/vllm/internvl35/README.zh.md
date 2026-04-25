# vLLM InternVL3.5 支持与 PR 历史

本文记录 vLLM 中与 InternVL3.5 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- InternVL3.5 is mostly a processor / encoder / video problem in vLLM.
- Video handling, native HF loading, and backend compatibility dominate the risk surface.

## 主要代码面

- `vllm/vllm/model_executor/models/internvl.py`

## 已合入 PR

- [#6514](https://github.com/vllm-project/vllm/pull/6514) `Initialize support for InternVL2 series models`：Historical base for current InternVL runtime code.
- [#18499](https://github.com/vllm-project/vllm/pull/18499) `Initialize video input support for InternVL models`：Added video processing to the family.
- [#23658](https://github.com/vllm-project/vllm/pull/23658) `Enable video support for InternVL3.5 models`：Carried video support into the 3.5 checkpoints.
- [#23742](https://github.com/vllm-project/vllm/pull/23742) `Enable native HF format InternVL support`：Removed reliance on ad hoc checkpoint rewrites.
- [#38049](https://github.com/vllm-project/vllm/pull/38049) `Add torch.compile support for InternVL vision encoder`：Modernized the encoder execution path.

## 配套 skill

- `skills/model-optimization/vllm/vllm-internvl35-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-internvl35-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `InternVL3.5`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2024-07-17 | [#6514](https://github.com/vllm-project/vllm/pull/6514) | merged | [Model] Initialize support for InternVL2 series models | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py` |
| 2025-05-21 | [#18499](https://github.com/vllm-project/vllm/pull/18499) | merged | [VLM] Initialize video input support for InternVL models | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `examples/offline_inference/vision_language.py` |
| 2025-08-26 | [#23658](https://github.com/vllm-project/vllm/pull/23658) | merged | [Model] Enable video support for InternVL3.5 models | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py` |
| 2025-08-27 | [#23742](https://github.com/vllm-project/vllm/pull/23742) | merged | [Model] Enable native HF format InternVL support | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md` |
| 2026-03-25 | [#38049](https://github.com/vllm-project/vllm/pull/38049) | merged | [Model] Add torch.compile support for InternVL vision encoder | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/intern_vit.py`, `vllm/config/utils.py` |

### 逐 PR 代码 diff 阅读记录

### PR #6514 - [Model] Initialize support for InternVL2 series models

- 链接：https://github.com/vllm-project/vllm/pull/6514
- 状态/时间：`merged`，created 2024-07-17, merged 2024-07-29；作者 `Isotr0py`。
- 代码 diff 已读范围：`14` 个文件，`+1042/-6`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, attention, vision, cache, kv, spec, processor, quant, cuda, lora。
- 代码 diff 细节：
  - `vllm/model_executor/models/internvl.py` added +471/-0 (471 lines); hunk: +# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py; 符号: InternVLImagePixelInputs, build_transform, find_closest_aspect_ratio, calculate_num_blocks
  - `vllm/model_executor/models/intern_vit.py` added +270/-0 (270 lines); hunk: +# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py; 符号: InternVisionEmbeddings, __init__, _get_pos_embed, forward
  - `tests/models/test_internvl.py` added +201/-0 (201 lines); hunk: +import types; 符号: InternVLProcessor:, __init__, __call__, generate
  - `vllm/transformers_utils/configs/internvl.py` added +51/-0 (51 lines); hunk: +# Adapted from; 符号: InternVLChatConfig, __init__
  - `examples/offline_inference_vision_language.py` modified +15/-0 (15 lines); hunk: def run_minicpmv(question):; def run_blip2(question):; 符号: run_minicpmv, run_internvl, for, run_blip2
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py`；patch 关键词为 config, attention, vision, cache, kv, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18499 - [VLM] Initialize video input support for InternVL models

- 链接：https://github.com/vllm-project/vllm/pull/18499
- 状态/时间：`merged`，created 2025-05-21, merged 2025-05-25；作者 `Isotr0py`。
- 代码 diff 已读范围：`10` 个文件，`+596/-62`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：vision, processor, test, config, spec, attention, cache, doc, fp8。
- 代码 diff 细节：
  - `vllm/model_executor/models/internvl.py` modified +485/-26 (511 lines); hunk: # --------------------------------------------------------; class InternVLImageEmbeddingInputs(TypedDict):; 符号: InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, InternVLVideoEmbeddingInputs, build_transform
  - `tests/models/multimodal/generation/vlm_utils/model_utils.py` modified +66/-20 (86 lines); hunk: from pathlib import PosixPath; def __init__(self, hf_runner: HfRunner):; 符号: __init__, __call__, __call__
  - `examples/offline_inference/vision_language.py` modified +11/-4 (15 lines); hunk: def run_smolvlm(questions: list[str], modality: str) -> ModelRequestData:; def run_internvl(questions: list[str], modality: str) -> ModelRequestData:; 符号: run_smolvlm, run_internvl, run_internvl
  - `vllm/model_executor/models/nvlm_d.py` modified +8/-5 (13 lines); hunk: PromptUpdateDetails); def get_hf_processor(; 符号: get_hf_processor, NVLMDummyInputsBuilder, NVLMDummyInputsBuilder, get_dummy_text
  - `tests/models/multimodal/generation/test_common.py` modified +11/-0 (11 lines); hunk: use_tokenizer_eos=True,
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `examples/offline_inference/vision_language.py`；patch 关键词为 vision, processor, test, config, spec, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `examples/offline_inference/vision_language.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23658 - [Model] Enable video support for InternVL3.5 models

- 链接：https://github.com/vllm-project/vllm/pull/23658
- 状态/时间：`merged`，created 2025-08-26, merged 2025-08-26；作者 `Isotr0py`。
- 代码 diff 已读范围：`5` 个文件，`+22/-7`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：test, fp8, moe, config, doc, spec。
- 代码 diff 细节：
  - `vllm/model_executor/models/internvl.py` modified +7/-3 (10 lines); hunk: def get_supported_mm_limits(self):; 符号: get_supported_mm_limits, get_video_token, get_num_frames_with_most_features
  - `tests/models/multimodal/processing/test_tensor_schema.py` modified +6/-1 (7 lines); hunk: "MiniCPMV",
  - `tests/models/registry.py` modified +4/-1 (5 lines); hunk: def check_available_online(; 符号: check_available_online
  - `docs/models/supported_models.md` modified +2/-2 (4 lines); hunk: These models primarily accept the `LLM.generate` (./generative_models.md#llmgen; Some models are supported only via the [Transformers backend](#transformers).
  - `tests/models/multimodal/processing/test_common.py` modified +3/-0 (3 lines); hunk: def _test_processing_correctness_one(; 符号: _test_processing_correctness_one
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py`；patch 关键词为 test, fp8, moe, config, doc, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23742 - [Model] Enable native HF format InternVL support

- 链接：https://github.com/vllm-project/vllm/pull/23742
- 状态/时间：`merged`，created 2025-08-27, merged 2025-08-27；作者 `Isotr0py`。
- 代码 diff 已读范围：`4` 个文件，`+18/-16`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：test, doc, fp8, moe。
- 代码 diff 细节：
  - `tests/models/multimodal/generation/test_common.py` modified +14/-15 (29 lines); hunk: },; use_tokenizer_eos=True,
  - `tests/models/registry.py` modified +2/-1 (3 lines); hunk: def check_available_online(; def check_available_online(; 符号: check_available_online, check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunk: These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunk: "H2OVLChatModel": ("h2ovl", "H2OVLChatModel"),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md`；patch 关键词为 test, doc, fp8, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #38049 - [Model] Add torch.compile support for InternVL vision encoder

- 链接：https://github.com/vllm-project/vllm/pull/38049
- 状态/时间：`merged`，created 2026-03-25, merged 2026-03-26；作者 `tianrengao`。
- 代码 diff 已读范围：`2` 个文件，`+20/-3`；代码面：model wrapper, scheduler/runtime, docs/config；关键词：config, quant, vision。
- 代码 diff 细节：
  - `vllm/model_executor/models/intern_vit.py` modified +11/-2 (13 lines); hunk: import torch.nn.functional as F; def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; 符号: forward, InternVisionEncoderLayer, __init__, __init__
  - `vllm/config/utils.py` modified +9/-1 (10 lines); hunk: def normalize_value(x):; 符号: normalize_value
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/intern_vit.py`, `vllm/config/utils.py`；patch 关键词为 config, quant, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/intern_vit.py`, `vllm/config/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
