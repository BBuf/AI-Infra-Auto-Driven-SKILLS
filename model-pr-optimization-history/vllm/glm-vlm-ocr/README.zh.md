# vLLM GLM VLM / OCR 支持与 PR 历史

本文记录 vLLM 中与 GLM VLM / OCR 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- GLM visual/OCR support in vLLM spans classic GLM4V, newer GLM4.1V, and GLM-OCR-specific processor paths.
- The main failures are processor-schema drift, MRoPE/video position handling, and OCR-specific weight or patch-merger mismatches.

## 主要代码面

- `vllm/vllm/model_executor/models/glm4v.py`
- `vllm/vllm/model_executor/models/glm4_1v.py`
- `vllm/vllm/model_executor/models/glm_ocr.py`

## 已合入 PR

- [#9242](https://github.com/vllm-project/vllm/pull/9242) `Add GLM-4v support`：Landed the original GLM4V multimodal model path.
- [#19331](https://github.com/vllm-project/vllm/pull/19331) `Add GLM4.1V model`：Extended the family to the newer GLM4.1V checkpoint layout and vision stack.
- [#27860](https://github.com/vllm-project/vllm/pull/27860) `Fix broken MRoPE for GLM-4.1V/GLM-4.5V`：Closed a positional-embedding bug with large practical accuracy impact on vision inputs.
- [#33005](https://github.com/vllm-project/vllm/pull/33005) `GLM-OCR with MTP Support`：Added OCR-specific draft / MTP support rather than text-only OCR loading.
- [#33350](https://github.com/vllm-project/vllm/pull/33350) `Fix broken GLM-OCR initialization`：Fixed startup failures in the GLM-OCR path after the first bring-up.
- [#37962](https://github.com/vllm-project/vllm/pull/37962) `GLM OCR Patch Merger context_dim`：Updated the patch-merger contract for newer OCR checkpoints.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-glm-vlm-ocr-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-glm-vlm-ocr-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `GLM VLM / OCR`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2024-10-10 | [#9242](https://github.com/vllm-project/vllm/pull/9242) | merged | [Model] Add GLM-4v support and meet vllm==0.6.2 | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/chatglm.py`, `vllm/model_executor/models/glm4_vision_encoder.py`, `tests/models/decoder_only/vision_language/test_glm4.py` |
| 2025-06-08 | [#19331](https://github.com/vllm-project/vllm/pull/19331) | merged | Add GLM-4.1V model | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/layers/rotary_embedding.py`, `vllm/multimodal/parse.py` |
| 2025-10-31 | [#27860](https://github.com/vllm-project/vllm/pull/27860) | merged | [Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V | model wrapper, scheduler/runtime | `vllm/model_executor/models/glm4_1v.py` |
| 2026-01-24 | [#33005](https://github.com/vllm-project/vllm/pull/33005) | merged | [GLM-OCR] GLM-OCR with MTP Support | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`, `vllm/model_executor/models/glm4.py` |
| 2026-01-29 | [#33350](https://github.com/vllm-project/vllm/pull/33350) | merged | [Bugfix] Fix broken GLM-OCR initialization | model wrapper, scheduler/runtime | `vllm/model_executor/models/glm_ocr.py` |
| 2026-03-24 | [#37962](https://github.com/vllm-project/vllm/pull/37962) | merged | [bug-fix] GLM OCR Patch Merger context_dim | model wrapper, scheduler/runtime | `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py` |

### 逐 PR 代码 diff 阅读记录

### PR #9242 - [Model] Add GLM-4v support and meet vllm==0.6.2

- 链接：https://github.com/vllm-project/vllm/pull/9242
- 状态/时间：`merged`，created 2024-10-10, merged 2024-10-11；作者 `sixsixcoder`。
- 代码 diff 已读范围：`7` 个文件，`+776/-72`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：vision, attention, config, kv, processor, quant, cache, doc, lora, moe。
- 代码 diff 细节：
  - `vllm/model_executor/models/chatglm.py` modified +298/-52 (350 lines); hunk: # coding=utf-8; class GLMMLP(nn.Module):; 符号: calculate_image_placeholder, mm_input_mapper_for_glmv, merge_glm_vision_embeddings, GLMImagePixelInputs
  - `vllm/model_executor/models/glm4_vision_encoder.py` added +298/-0 (298 lines); hunk: +# coding=utf-8; 符号: PatchEmbedding, __init__, forward, Attention
  - `tests/models/decoder_only/vision_language/test_glm4.py` added +133/-0 (133 lines); hunk: +from typing import List, Optional, Tuple, Type; 符号: run_test, processor, test_models
  - `vllm/transformers_utils/tokenizer.py` modified +21/-18 (39 lines); hunk: def __len__(self):; def get_tokenizer(; 符号: __len__, patch_padding_side, _pad, get_tokenizer
  - `examples/offline_inference_vision_language.py` modified +16/-0 (16 lines); hunk: def run_mllama(question: str, modality: str):; def run_mllama(question: str, modality: str):; 符号: run_mllama, run_glm4v, run_mllama
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/chatglm.py`, `vllm/model_executor/models/glm4_vision_encoder.py`, `tests/models/decoder_only/vision_language/test_glm4.py`；patch 关键词为 vision, attention, config, kv, processor, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/chatglm.py`, `vllm/model_executor/models/glm4_vision_encoder.py`, `tests/models/decoder_only/vision_language/test_glm4.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19331 - Add GLM-4.1V model

- 链接：https://github.com/vllm-project/vllm/pull/19331
- 状态/时间：`merged`，created 2025-06-08, merged 2025-07-01；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`17` 个文件，`+1946/-16`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：processor, config, test, vision, cache, spec, attention, flash, kv, lora。
- 代码 diff 细节：
  - `vllm/model_executor/models/glm4_1v.py` added +1589/-0 (1589 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs, Glm4vVideoEmbeddingInputs
  - `vllm/model_executor/layers/rotary_embedding.py` modified +119/-0 (119 lines); hunk: # See the License for the specific language governing permissions and; def get_input_positions_tensor(; 符号: get_input_positions_tensor, get_input_positions_tensor, _glm4v_get_input_positions_tensor, _vl_get_input_positions_tensor
  - `vllm/multimodal/parse.py` modified +40/-2 (42 lines); hunk: def __init__(self, data: Union[torch.Tensor, list[torch.Tensor]]) -> None:; def __init__(; 符号: __init__, VideoProcessorItems, __init__, __init__
  - `examples/offline_inference/vision_language.py` modified +39/-1 (40 lines); hunk: def run_glm4v(questions: list[str], modality: str) -> ModelRequestData:; def run_skyworkr1v(questions: list[str], modality: str) -> ModelRequestData:; 符号: run_glm4v, run_glm4_1v, run_h2ovl, run_skyworkr1v
  - `tests/models/multimodal/generation/test_common.py` modified +28/-0 (28 lines); hunk: num_logprobs=10,
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/layers/rotary_embedding.py`, `vllm/multimodal/parse.py`；patch 关键词为 processor, config, test, vision, cache, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/layers/rotary_embedding.py`, `vllm/multimodal/parse.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #27860 - [Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V

- 链接：https://github.com/vllm-project/vllm/pull/27860
- 状态/时间：`merged`，created 2025-10-31, merged 2025-10-31；作者 `Isotr0py`。
- 代码 diff 已读范围：`1` 个文件，`+147/-2`；代码面：model wrapper, scheduler/runtime；关键词：config, lora, processor, vision。
- 代码 diff 细节：
  - `vllm/model_executor/models/glm4_1v.py` modified +147/-2 (149 lines); hunk: # limitations under the License.; import torch.nn as nn; 符号: get_video_replacement_glm4v, Glm4vForConditionalGeneration, get_multimodal_embeddings, get_mrope_input_positions
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/glm4_1v.py`；patch 关键词为 config, lora, processor, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/glm4_1v.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #33005 - [GLM-OCR] GLM-OCR with MTP Support

- 链接：https://github.com/vllm-project/vllm/pull/33005
- 状态/时间：`merged`，created 2026-01-24, merged 2026-01-26；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`14` 个文件，`+873/-8`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：spec, config, fp8, processor, test, kv, moe, quant, attention, cache。
- 代码 diff 细节：
  - `vllm/model_executor/models/glm_ocr.py` added +389/-0 (389 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: GlmOcrVisionMLP, GlmOcrVisionAttention, __init__, split_qkv
  - `vllm/model_executor/models/glm_ocr_mtp.py` added +285/-0 (285 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: GlmOcrMultiTokenPredictorLayer, __init__, forward, GlmOcrMultiTokenPredictor
  - `vllm/model_executor/models/glm4.py` modified +99/-2 (101 lines); hunk: from vllm.model_executor.layers.quantization import QuantizationConfig; def __init__(; 符号: Glm4Attention, __init__, __init__, load_weights
  - `examples/offline_inference/vision_language.py` modified +38/-0 (38 lines); hunk: def run_glm4_5v_fp8(questions: list[str], modality: str) -> ModelRequestData:; def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:; 符号: run_glm4_5v_fp8, run_glm_ocr, run_h2ovl, run_tarsier2
  - `tests/models/multimodal/generation/test_common.py` modified +14/-0 (14 lines); hunk: ],
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`, `vllm/model_executor/models/glm4.py`；patch 关键词为 spec, config, fp8, processor, test, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`, `vllm/model_executor/models/glm4.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #33350 - [Bugfix] Fix broken GLM-OCR initialization

- 链接：https://github.com/vllm-project/vllm/pull/33350
- 状态/时间：`merged`，created 2026-01-29, merged 2026-01-29；作者 `Isotr0py`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：model wrapper, scheduler/runtime；关键词：config, quant, vision。
- 代码 diff 细节：
  - `vllm/model_executor/models/glm_ocr.py` modified +1/-1 (2 lines); hunk: class GlmOcrPatchMerger(Glm4vPatchMerger):; 符号: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/glm_ocr.py`；patch 关键词为 config, quant, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/glm_ocr.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #37962 - [bug-fix] GLM OCR Patch Merger context_dim

- 链接：https://github.com/vllm-project/vllm/pull/37962
- 状态/时间：`merged`，created 2026-03-24, merged 2026-03-26；作者 `JaredforReal`。
- 代码 diff 已读范围：`2` 个文件，`+14/-4`；代码面：model wrapper, scheduler/runtime；关键词：config, quant, vision, processor。
- 代码 diff 细节：
  - `vllm/model_executor/models/glm_ocr.py` modified +8/-3 (11 lines); hunk: from einops import rearrange; class GlmOcrPatchMerger(Glm4vPatchMerger):; 符号: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__, __init__
  - `vllm/model_executor/models/glm4_1v.py` modified +6/-1 (7 lines); hunk: import torch.nn.functional as F; def forward(; 符号: forward, Glm4vVisionTransformer, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py`；patch 关键词为 config, quant, vision, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：6；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
