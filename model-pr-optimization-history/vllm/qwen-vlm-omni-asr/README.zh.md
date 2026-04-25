# vLLM Qwen2.5-VL / Qwen3-VL / Qwen3-Omni / Qwen3-ASR 支持与 PR 历史

本文记录 vLLM 在提交 `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
附近的多模态 Qwen 家族支持情况。

- 状态: 当前 mainline 已支持

## 核心结论

- 这条家族线已经覆盖 Qwen2.5-VL、Qwen3-VL、Qwen3-VL-MoE、
  Qwen3-Omni thinker、Qwen3-ASR，以及 realtime Qwen3-ASR。
- 真正高风险的地方主要在:
  placeholder 展开、视频时间戳、interleaved MRoPE、
  `use_audio_in_video`、音频特征长度换算，以及 realtime prompt 扩展。

## 主要代码面

- `vllm/vllm/model_executor/models/qwen2_5_vl.py`
- `vllm/vllm/model_executor/models/qwen3_vl.py`
- `vllm/vllm/model_executor/models/qwen3_vl_moe.py`
- `vllm/vllm/model_executor/models/qwen3_omni_moe_thinker.py`
- `vllm/vllm/model_executor/models/qwen3_asr.py`
- `vllm/vllm/model_executor/models/qwen3_asr_realtime.py`
- `vllm/vllm/model_executor/layers/rotary_embedding/mrope.py`

## 已合入 PR

- [#13155](https://github.com/vllm-project/vllm/pull/13155)
  `Qwen2.5-VL Optimization`
  已审 diff: `2` 个文件，`47` 行新增，`51` 行删除。
  主要优化 Qwen2.5-VL 的视觉注意力 fallback 路径，并切到共享 `RMSNorm`。
- [#24727](https://github.com/vllm-project/vllm/pull/24727)
  `Support Qwen3-VL Model Series`
  已审 diff: `13` 个文件，`2084` 行新增，`17` 行删除。
  这是 Qwen3-VL / Qwen3-VL-MoE 的主落地 PR，并补了视频 placeholder 和处理链。
- [#25055](https://github.com/vllm-project/vllm/pull/25055)
  `Add Triton kernel for Qwen3-VL interleaved MRoPE`
  已审 diff: `2` 个文件，`88` 行新增，`46` 行删除。
  它让 Qwen3-VL 的 interleaved MRoPE 真正成为受测的主路径。
- [#25550](https://github.com/vllm-project/vllm/pull/25550)
  `Add Qwen3-Omni moe thinker`
  已审 diff: `6` 个文件，`1795` 行新增，`36` 行删除。
  它加入 thinker 运行时，并单独处理 `use_audio_in_video` 的 placeholder 记账。
- [#33312](https://github.com/vllm-project/vllm/pull/33312)
  `Qwen3-ASR`
  已审 diff: `9` 个文件，`1269` 行新增。
  它补了 Qwen3-ASR 的 config、processor、模型和 transcription 路径。
- [#34613](https://github.com/vllm-project/vllm/pull/34613)
  `Add Qwen3-ASR realtime streaming support`
  已审 diff: `5` 个文件，`256` 行新增，`1` 行删除。
  它加入 realtime 子类、音频 buffer 和 prompt 扩展逻辑。

## 当前结论

这条家族线出问题时，不要笼统地说“多模态坏了”。要先定位是
Qwen2.5-VL attention fallback、Qwen3-VL 视频 prompt replacement、
Qwen3-Omni 的 audio-in-video 记账，还是 Qwen3-ASR 的
prompt / audio length 逻辑出错。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Qwen VLM / Omni / ASR`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-02-12 | [#13155](https://github.com/vllm-project/vllm/pull/13155) | merged | [Misc] Qwen2.5-VL Optimization | model wrapper, scheduler/runtime | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-09-12 | [#24727](https://github.com/vllm-project/vllm/pull/24727) | merged | [Model] Support Qwen3-VL Model Series | model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py` |
| 2025-09-17 | [#25055](https://github.com/vllm-project/vllm/pull/25055) | merged | [Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE | kernel, scheduler/runtime, tests/benchmarks | `tests/kernels/core/test_mrope.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py` |
| 2025-09-24 | [#25550](https://github.com/vllm-project/vllm/pull/25550) | merged | Add Qwen3-Omni moe thinker | model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`, `tests/models/registry.py` |
| 2026-01-29 | [#33312](https://github.com/vllm-project/vllm/pull/33312) | merged | [Models] Qwen3-ASR | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py` |
| 2026-02-16 | [#34613](https://github.com/vllm-project/vllm/pull/34613) | merged | [Realtime] Add Qwen3-ASR realtime streaming support | model wrapper, scheduler/runtime, tests/benchmarks | `vllm/model_executor/models/qwen3_asr_realtime.py`, `tests/models/registry.py`, `vllm/model_executor/models/interfaces.py` |

### 逐 PR 代码 diff 阅读记录

### PR #13155 - [Misc] Qwen2.5-VL Optimization

- 链接：https://github.com/vllm-project/vllm/pull/13155
- 状态/时间：`merged`，created 2025-02-12, merged 2025-02-13；作者 `wulipc`。
- 代码 diff 已读范围：`2` 个文件，`+47/-51`；代码面：model wrapper, scheduler/runtime；关键词：attention, flash, vision, config, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +25/-36 (61 lines); hunk: from vllm.logger import init_logger; def forward(; 符号: forward, forward, forward, Qwen2RMSNorm
  - `vllm/model_executor/models/qwen2_vl.py` modified +22/-15 (37 lines); hunk: def apply_rotary_emb_torch(x: torch.Tensor,; def forward(; 符号: apply_rotary_emb_torch, apply_rotary_pos_emb_vision, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；patch 关键词为 attention, flash, vision, config, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #24727 - [Model] Support Qwen3-VL Model Series

- 链接：https://github.com/vllm-project/vllm/pull/24727
- 状态/时间：`merged`，created 2025-09-12, merged 2025-09-17；作者 `ywang96`。
- 代码 diff 已读范围：`13` 个文件，`+2084/-17`；代码面：model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：moe, vision, config, processor, test, cuda, fp8, kv, quant, scheduler。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_vl.py` added +1478/-0 (1478 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP
  - `vllm/model_executor/models/qwen3_vl_moe.py` added +344/-0 (344 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Qwen3VLMoeProcessingInfo, get_hf_config, Qwen3MoeLLMModel, __init__
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +133/-11 (144 lines); hunk: def triton_mrope(; def __init__(; 符号: triton_mrope, apply_interleaved_rope, MRotaryEmbedding, __init__
  - `examples/offline_inference/vision_language.py` modified +78/-0 (78 lines); hunk: def run_qwen2_5_omni(questions: list[str], modality: str):; def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:; 符号: run_qwen2_5_omni, run_qwen3_vl, run_qwen3_vl_moe, run_r_vl
  - `tests/models/multimodal/processing/test_common.py` modified +34/-1 (35 lines); hunk: def glm4_1v_patch_mm_data(mm_data: MultiModalDataDict) -> MultiModalDataDict:; def glm4_1v_patch_mm_data(mm_data: MultiModalDataDict) -> MultiModalDataDict:; 符号: glm4_1v_patch_mm_data, glm4_1v_patch_mm_data, qwen3_vl_patch_mm_data, create_metadata
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`；patch 关键词为 moe, vision, config, processor, test, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #25055 - [Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE

- 链接：https://github.com/vllm-project/vllm/pull/25055
- 状态/时间：`merged`，created 2025-09-17, merged 2025-09-19；作者 `Isotr0py`。
- 代码 diff 已读范围：`2` 个文件，`+88/-46`；代码面：kernel, scheduler/runtime, tests/benchmarks；关键词：cuda, attention, cache, config, kv, test, triton。
- 代码 diff 细节：
  - `tests/kernels/core/test_mrope.py` modified +66/-32 (98 lines); hunk: # SPDX-License-Identifier: Apache-2.0; def generate_test_data(num_tokens: int, num_q_heads: int, num_kv_heads: int,; 符号: generate_test_data, generate_test_data, unroll_model_tp_dict, MRoPETestInfo
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +22/-14 (36 lines); hunk: @triton.jit; def _triton_qwen2vl_mrope_forward(; 符号: _triton_qwen2vl_mrope_forward, _triton_mrope_forward, _triton_qwen2vl_mrope_forward, _triton_qwen2vl_mrope_forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/kernels/core/test_mrope.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`；patch 关键词为 cuda, attention, cache, config, kv, test。影响判断：CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `tests/kernels/core/test_mrope.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #25550 - Add Qwen3-Omni moe thinker

- 链接：https://github.com/vllm-project/vllm/pull/25550
- 状态/时间：`merged`，created 2025-09-24, merged 2025-10-10；作者 `wangxiongts`。
- 代码 diff 已读范围：`6` 个文件，`+1795/-36`；代码面：model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：moe, config, processor, test, vision, attention, cache, doc, flash, kv。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` added +1409/-0 (1409 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +374/-34 (408 lines); hunk: def split_thw(grid_thw: Union[torch.Tensor, list[int]]) -> list[list[int]]:; def _vl_get_input_positions_tensor(; 符号: split_thw, _vl_get_input_positions_tensor, _vl_get_input_positions_tensor, _vl_get_input_positions_tensor
  - `tests/models/registry.py` modified +5/-0 (5 lines); hunk: def check_available_online(; 符号: check_available_online
  - `docs/models/supported_models.md` modified +2/-2 (4 lines); hunk: These models primarily accept the `LLM.generate` (./generative_models.md#llmgen; Some models are supported only via the [Transformers backend](#transformers).
  - `vllm/model_executor/models/registry.py` modified +4/-0 (4 lines); hunk: "qwen2_5_omni_thinker",
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`, `tests/models/registry.py`；patch 关键词为 moe, config, processor, test, vision, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`, `tests/models/registry.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #33312 - [Models] Qwen3-ASR

- 链接：https://github.com/vllm-project/vllm/pull/33312
- 状态/时间：`merged`，created 2026-01-29, merged 2026-01-29；作者 `ywang96`。
- 代码 diff 已读范围：`9` 个文件，`+1269/-0`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：spec, attention, config, doc, moe, processor, cache, flash, kv, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_asr.py` added +567/-0 (567 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: _get_feat_extract_output_lengths, Qwen3ASRProcessingInfo, get_hf_config, get_hf_processor
  - `vllm/transformers_utils/configs/qwen3_asr.py` added +436/-0 (436 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Qwen3ASRAudioEncoderConfig, to, __init__, Qwen3ASRTextConfig
  - `vllm/transformers_utils/processors/qwen3_asr.py` added +231/-0 (231 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Qwen3ASRProcessorKwargs, _get_feat_extract_output_lengths, Qwen3ASRProcessor, to
  - `examples/offline_inference/audio_language.py` modified +20/-0 (20 lines); hunk: def run_qwen2_5_omni(question: str, audio_count: int):; def run_whisper(question: str, audio_count: int) -> ModelRequestData:; 符号: run_qwen2_5_omni, run_qwen3_asr, run_ultravox, run_whisper
  - `tests/models/registry.py` modified +6/-0 (6 lines); hunk: def check_available_online(; 符号: check_available_online
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`；patch 关键词为 spec, attention, config, doc, moe, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #34613 - [Realtime] Add Qwen3-ASR realtime streaming support

- 链接：https://github.com/vllm-project/vllm/pull/34613
- 状态/时间：`merged`，created 2026-02-16, merged 2026-02-21；作者 `pougetat`。
- 代码 diff 已读范围：`5` 个文件，`+256/-1`；代码面：model wrapper, scheduler/runtime, tests/benchmarks；关键词：cache, config, moe, processor, spec, test。
- 代码 diff 细节：
  - `vllm/model_executor/models/qwen3_asr_realtime.py` added +239/-0 (239 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Qwen3ASRRealtimeBuffer:, __init__, write_audio, read_audio
  - `tests/models/registry.py` modified +8/-0 (8 lines); hunk: def check_available_online(; 符号: check_available_online
  - `vllm/model_executor/models/interfaces.py` modified +4/-0 (4 lines); hunk: class SupportsRealtime(Protocol):; 符号: SupportsRealtime, buffer_realtime_audio
  - `vllm/model_executor/models/registry.py` modified +4/-0 (4 lines); hunk: "qwen3_asr",
  - `vllm/entrypoints/openai/realtime/connection.py` modified +1/-1 (2 lines); hunk: async def _run_generation(; 符号: _run_generation
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/qwen3_asr_realtime.py`, `tests/models/registry.py`, `vllm/model_executor/models/interfaces.py`；patch 关键词为 cache, config, moe, processor, spec, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/qwen3_asr_realtime.py`, `tests/models/registry.py`, `vllm/model_executor/models/interfaces.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：6；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
