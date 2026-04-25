# vLLM Intern-S1 支持与 PR 历史

本文记录 vLLM 中与 Intern-S1 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Intern-S1 leans heavily on shared InternVL processor code in vLLM.
- Most regressions come from processor compatibility and video-aware serving rather than the text stack alone.

## 主要代码面

- `vllm/vllm/model_executor/models/interns1.py`
- `vllm/vllm/model_executor/models/interns1_pro.py`

## 已合入 PR

- [#21628](https://github.com/vllm-project/vllm/pull/21628) `Support Intern-S1`：Initial Intern-S1 support in vLLM.
- [#21671](https://github.com/vllm-project/vllm/pull/21671) `Add video support for Intern-S1`：Extended the family beyond static images.
- [#22417](https://github.com/vllm-project/vllm/pull/22417) `Fix wrong method name in Intern-S1 image processor`：Patched a processor bug after bring-up.
- [#33636](https://github.com/vllm-project/vllm/pull/33636) `Intern-S1-Pro`：Added the Pro generation / alias path.

## 配套 skill

- `skills/model-optimization/vllm/vllm-intern-s1-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-intern-s1-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Intern-S1`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-25 | [#21628](https://github.com/vllm-project/vllm/pull/21628) | merged | Support Intern-S1 | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py`, `examples/offline_inference/vision_language.py` |
| 2025-07-27 | [#21671](https://github.com/vllm-project/vllm/pull/21671) | merged | [VLM] Add video support for Intern-S1 | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/interns1.py`, `examples/offline_inference/vision_language.py`, `docs/models/supported_models.md` |
| 2025-08-07 | [#22417](https://github.com/vllm-project/vllm/pull/22417) | merged | [Bugfix] Fix wrong method name in Intern-S1 image processor | model wrapper, scheduler/runtime | `vllm/model_executor/models/interns1.py` |
| 2026-02-03 | [#33636](https://github.com/vllm-project/vllm/pull/33636) | merged | [Models] Intern-S1-Pro | model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/interns1_pro.py`, `vllm/model_executor/layers/rotary_embedding/fope.py`, `examples/offline_inference/vision_language.py` |

### 逐 PR 代码 diff 阅读记录

### PR #21628 - Support Intern-S1

- 链接：https://github.com/vllm-project/vllm/pull/21628
- 状态/时间：`merged`，created 2025-07-25, merged 2025-07-26；作者 `lvhan028`。
- 代码 diff 已读范围：`7` 个文件，`+1196/-0`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：vision, config, quant, spec, attention, doc, lora, processor, test。
- 代码 diff 细节：
  - `vllm/model_executor/models/interns1.py` added +711/-0 (711 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: InternS1MultiModalProjector, __init__, forward, InternS1ImagePixelInputs
  - `vllm/model_executor/models/interns1_vit.py` added +421/-0 (421 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: InternS1VisionPatchEmbeddings, __init__, forward, InternS1VisionEmbeddings
  - `examples/offline_inference/vision_language.py` modified +32/-0 (32 lines); hunk: def run_tarsier(questions: list[str], modality: str) -> ModelRequestData:; def run_skyworkr1v(questions: list[str], modality: str) -> ModelRequestData:; 符号: run_tarsier, run_interns1, run_internvl, run_skyworkr1v
  - `examples/offline_inference/vision_language_multi_image.py` modified +28/-0 (28 lines); hunk: def load_smolvlm(question: str, image_urls: list[str]) -> ModelRequestData:; def load_tarsier2(question: str, image_urls: list[str]) -> ModelRequestData:; 符号: load_smolvlm, load_interns1, load_internvl, load_tarsier2
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunk: def check_available_online(; 符号: check_available_online
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py`, `examples/offline_inference/vision_language.py`；patch 关键词为 vision, config, quant, spec, attention, doc。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py`, `examples/offline_inference/vision_language.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21671 - [VLM] Add video support for Intern-S1

- 链接：https://github.com/vllm-project/vllm/pull/21671
- 状态/时间：`merged`，created 2025-07-27, merged 2025-07-27；作者 `Isotr0py`。
- 代码 diff 已读范围：`5` 个文件，`+173/-50`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：spec, vision, config, doc, processor, test。
- 代码 diff 细节：
  - `vllm/model_executor/models/interns1.py` modified +166/-45 (211 lines); hunk: from collections.abc import Iterable, Mapping, Sequence; def get_interns1_target_ratios(; 符号: get_interns1_target_ratios, InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits
  - `examples/offline_inference/vision_language.py` modified +5/-3 (8 lines); hunk: def run_tarsier(questions: list[str], modality: str) -> ModelRequestData:; def run_interns1(questions: list[str], modality: str) -> ModelRequestData:; 符号: run_tarsier, run_interns1, run_interns1
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunk: Specified using `--task generate`.
  - `tests/models/multimodal/processing/test_common.py` modified +1/-0 (1 lines); hunk: def _test_processing_correctness_one(; 符号: _test_processing_correctness_one
  - `vllm/model_executor/models/internvl.py` modified +0/-1 (1 lines); hunk: def get_multimodal_embeddings(self,; 符号: get_multimodal_embeddings
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/interns1.py`, `examples/offline_inference/vision_language.py`, `docs/models/supported_models.md`；patch 关键词为 spec, vision, config, doc, processor, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/interns1.py`, `examples/offline_inference/vision_language.py`, `docs/models/supported_models.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22417 - [Bugfix] Fix wrong method name in Intern-S1 image processor

- 链接：https://github.com/vllm-project/vllm/pull/22417
- 状态/时间：`merged`，created 2025-08-07, merged 2025-08-07；作者 `DarkLight1337`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：model wrapper, scheduler/runtime；关键词：processor。
- 代码 diff 细节：
  - `vllm/model_executor/models/interns1.py` modified +1/-1 (2 lines); hunk: def get_num_image_tokens(; 符号: get_num_image_tokens
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/interns1.py`；patch 关键词为 processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/interns1.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #33636 - [Models] Intern-S1-Pro

- 链接：https://github.com/vllm-project/vllm/pull/33636
- 状态/时间：`merged`，created 2026-02-03, merged 2026-02-03；作者 `CUHKSZzxy`。
- 代码 diff 已读范围：`11` 个文件，`+942/-11`；代码面：model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：cache, vision, config, moe, expert, kv, attention, flash, processor, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/interns1_pro.py` added +633/-0 (633 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: InternS1ProProcessingInfo, get_hf_config, get_hf_processor, InternS1ProMoeMLP
  - `vllm/model_executor/layers/rotary_embedding/fope.py` added +199/-0 (199 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: FourierRotaryEmbedding, __init__, _compute_inv_freq, _compute_cos_sin_cache
  - `examples/offline_inference/vision_language.py` modified +35/-0 (35 lines); hunk: def run_interns1(questions: list[str], modality: str) -> ModelRequestData:; def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:; 符号: run_interns1, run_interns1_pro, run_internvl, run_tarsier2
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +23/-0 (23 lines); hunk: from .dual_chunk_rope import DualChunkRotaryEmbedding; def get_rope(; 符号: get_rope
  - `vllm/model_executor/layers/rotary_embedding/base.py` modified +15/-6 (21 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/interns1_pro.py`, `vllm/model_executor/layers/rotary_embedding/fope.py`, `examples/offline_inference/vision_language.py`；patch 关键词为 cache, vision, config, moe, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/interns1_pro.py`, `vllm/model_executor/layers/rotary_embedding/fope.py`, `examples/offline_inference/vision_language.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：4；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
