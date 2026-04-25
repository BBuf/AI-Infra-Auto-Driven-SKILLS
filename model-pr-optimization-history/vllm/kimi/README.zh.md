# vLLM Kimi K2 / K2.5 / Linear / Audio / VL 支持与 PR 历史

本文记录 vLLM 中与 Kimi K2 / K2.5 / Linear / Audio / VL 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- The Kimi family in vLLM spans vision, linear-attention, K2.5, and audio checkpoints.
- The most fragile areas are MLA plus FP8/NVFP4 loading, processor evolution, and parser alias compatibility between K2 and K2.5.

## 主要代码面

- `vllm/vllm/model_executor/models/kimi_vl.py`
- `vllm/vllm/model_executor/models/kimi_linear.py`
- `vllm/vllm/model_executor/models/kimi_k25.py`
- `vllm/vllm/model_executor/models/kimi_audio.py`

## 已合入 PR

- [#16387](https://github.com/vllm-project/vllm/pull/16387) `Add Kimi-VL model support`：Landed the original Kimi-VL multimodal runtime.
- [#27809](https://github.com/vllm-project/vllm/pull/27809) `Introduce Kimi Linear to vLLM`：Added the linear-attention Kimi family instead of only the VL path.
- [#33131](https://github.com/vllm-project/vllm/pull/33131) `Kimi-K2.5`：Brought the K2.5 generation into mainline.
- [#33876](https://github.com/vllm-project/vllm/pull/33876) `Fix Kimi-K2.5 NVFP4 checkpoints weight loading`：Closed a concrete launch blocker for quantized K2.5 checkpoints.
- [#36127](https://github.com/vllm-project/vllm/pull/36127) `Add support for moonshotai/Kimi-Audio-7B-Instruct`：Extended the family to audio-conditioned serving.
- [#37438](https://github.com/vllm-project/vllm/pull/37438) `Add Kimi-K2.5 reasoning/tool parser aliases`：Aligned parser aliases and tool-call IDs with the newer model outputs.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-kimi-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-kimi-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Kimi K2 / K2.5`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-04-10 | [#16387](https://github.com/vllm-project/vllm/pull/16387) | merged | [Model][VLM] Add Kimi-VL model support | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `examples/offline_inference/vision_language_multi_image.py` |
| 2025-10-30 | [#27809](https://github.com/vllm-project/vllm/pull/27809) | merged | [Model] Introduce Kimi Linear to vLLM | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/layers/kda.py`, `vllm/transformers_utils/configs/kimi_linear.py` |
| 2026-01-27 | [#33131](https://github.com/vllm-project/vllm/pull/33131) | merged | [Models] Kimi-K2.5 | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/entrypoints/chat_utils.py` |
| 2026-02-05 | [#33876](https://github.com/vllm-project/vllm/pull/33876) | merged | [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading | model wrapper, scheduler/runtime | `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/deepseek_v2.py` |
| 2026-03-05 | [#36127](https://github.com/vllm-project/vllm/pull/36127) | merged | [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py` |
| 2026-03-18 | [#37438](https://github.com/vllm-project/vllm/pull/37438) | merged | [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support | tests/benchmarks | `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/entrypoints/chat_utils.py`, `vllm/entrypoints/openai/chat_completion/serving.py` |

### 逐 PR 代码 diff 阅读记录

### PR #16387 - [Model][VLM] Add Kimi-VL model support

- 链接：https://github.com/vllm-project/vllm/pull/16387
- 状态/时间：`merged`，created 2025-04-10, merged 2025-04-14；作者 `courage17340`。
- 代码 diff 已读范围：`18` 个文件，`+1436/-14`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：vision, config, kv, spec, cache, processor, attention, doc, cuda, eagle。
- 代码 diff 细节：
  - `vllm/model_executor/models/moonvit.py` added +628/-0 (628 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope
  - `vllm/model_executor/models/kimi_vl.py` added +608/-0 (608 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MaxImageTokenMeta:, KimiVLMultiModalProjector, __init__, forward
  - `examples/offline_inference/vision_language_multi_image.py` modified +40/-0 (40 lines); hunk: def load_llama4(question: str, image_urls: list[str]) -> ModelRequestData:; def load_qwen2_5_vl(question: str, image_urls: list[str]) -> ModelRequestData:; 符号: load_llama4, load_kimi_vl, load_mistral3, load_qwen2_5_vl
  - `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0 (36 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: KimiVLConfig, __init__
  - `vllm/transformers_utils/configs/moonvit.py` added +32/-0 (32 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MoonViTConfig, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `examples/offline_inference/vision_language_multi_image.py`；patch 关键词为 vision, config, kv, spec, cache, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `examples/offline_inference/vision_language_multi_image.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #27809 - [Model] Introduce Kimi Linear to vLLM

- 链接：https://github.com/vllm-project/vllm/pull/27809
- 状态/时间：`merged`，created 2025-10-30, merged 2025-10-30；作者 `zhiyuan1i`。
- 代码 diff 已读范围：`15` 个文件，`+1326/-49`；代码面：model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config；关键词：attention, cache, kv, config, mla, spec, moe, topk, expert, lora。
- 代码 diff 细节：
  - `vllm/model_executor/models/kimi_linear.py` added +663/-0 (663 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: KimiMLP, __init__, forward, KimiMoE
  - `vllm/model_executor/layers/kda.py` added +426/-0 (426 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: kda_attention, kda_attention_fake, KimiDeltaAttention, mamba_type
  - `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0 (144 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: KimiLinearConfig, __init__, is_mla, is_moe
  - `vllm/model_executor/models/config.py` modified +25/-26 (51 lines); hunk: # SPDX-License-Identifier: Apache-2.0; def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:; 符号: verify_and_update_config, verify_and_update_config, verify_and_update_config, lcm
  - `vllm/model_executor/layers/mamba/mamba_utils.py` modified +41/-0 (41 lines); hunk: def gated_delta_net_state_dtype(; def gated_delta_net_state_shape(; 符号: gated_delta_net_state_dtype, kda_state_dtype, MambaStateShapeCalculator:, gated_delta_net_state_shape
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/layers/kda.py`, `vllm/transformers_utils/configs/kimi_linear.py`；patch 关键词为 attention, cache, kv, config, mla, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/layers/kda.py`, `vllm/transformers_utils/configs/kimi_linear.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #33131 - [Models] Kimi-K2.5

- 链接：https://github.com/vllm-project/vllm/pull/33131
- 状态/时间：`merged`，created 2026-01-27, merged 2026-01-27；作者 `ywang96`。
- 代码 diff 已读范围：`16` 个文件，`+1799/-8`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：kv, vision, config, processor, attention, cache, quant, spec, cuda, expert。
- 代码 diff 细节：
  - `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0 (678 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: _apply_rope_input_validation, get_rope_shape_decorate, wrapper, get_rope_shape
  - `vllm/model_executor/models/kimi_k25.py` added +581/-0 (581 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MaxImageTokenMeta:, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor, __init__
  - `vllm/entrypoints/chat_utils.py` modified +182/-5 (187 lines); hunk: MultiModalBatchedField,; class ConversationMessage(TypedDict, total=False):; 符号: ConversationMessage, _get_embeds_data, rebuild_mm_uuids_from_mm_data, build_video_prompts_from_mm_data
  - `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0 (129 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: KimiK25VisionConfig, __init__, KimiK25Config, __init__
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0 (80 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: KimiK2ReasoningParser, __init__, is_reasoning_end, is_reasoning_end_streaming
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/entrypoints/chat_utils.py`；patch 关键词为 kv, vision, config, processor, attention, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/entrypoints/chat_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #33876 - [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading

- 链接：https://github.com/vllm-project/vllm/pull/33876
- 状态/时间：`merged`，created 2026-02-05, merged 2026-02-05；作者 `Isotr0py`。
- 代码 diff 已读范围：`2` 个文件，`+15/-5`；代码面：model wrapper, scheduler/runtime；关键词：config, expert, fp4, moe, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunk: from vllm.config import VllmConfig; def split_video_chunks(self, video):; 符号: split_video_chunks, KimiK25ForConditionalGeneration, KimiK25ForConditionalGeneration, KimiK25ForConditionalGeneration
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunk: def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/deepseek_v2.py`；patch 关键词为 config, expert, fp4, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #36127 - [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct

- 链接：https://github.com/vllm-project/vllm/pull/36127
- 状态/时间：`merged`，created 2026-03-05, merged 2026-03-11；作者 `tunglinwood`。
- 代码 diff 已读范围：`14` 个文件，`+1446/-29`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：spec, attention, config, processor, cache, kv, test, vision, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/kimi_audio.py` added +725/-0 (725 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, KimiAudioProcessingInfo
  - `vllm/tokenizers/kimi_audio.py` added +410/-0 (410 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: _load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained, __init__
  - `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0 (163 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class
  - `vllm/renderers/kimi_audio.py` added +49/-0 (49 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: KimiAudioRenderer, from_config
  - `vllm/transformers_utils/processors/__init__.py` modified +18/-17 (35 lines); hunk: import importlib; "GLM4VProcessor",; 符号: __getattr__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py`；patch 关键词为 spec, attention, config, processor, cache, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #37438 - [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support

- 链接：https://github.com/vllm-project/vllm/pull/37438
- 状态/时间：`merged`，created 2026-03-18, merged 2026-03-19；作者 `DorBernsohn`。
- 代码 diff 已读范围：`4` 个文件，`+173/-18`；代码面：tests/benchmarks；关键词：config, test, spec。
- 代码 diff 细节：
  - `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0 (155 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags
  - `vllm/entrypoints/chat_utils.py` modified +14/-0 (14 lines); hunk: def get_history_tool_calls_cnt(conversation: list[ConversationMessage]):; 符号: get_history_tool_calls_cnt, get_tool_call_id_type, make_tool_call_id
  - `vllm/entrypoints/openai/chat_completion/serving.py` modified +2/-9 (11 lines); hunk: ChatTemplateContentFormatOption,; def __init__(; 符号: __init__
  - `vllm/entrypoints/openai/responses/serving.py` modified +2/-9 (11 lines); hunk: from vllm.entrypoints.chat_utils import (; def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/entrypoints/chat_utils.py`, `vllm/entrypoints/openai/chat_completion/serving.py`；patch 关键词为 config, test, spec。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/entrypoints/chat_utils.py`, `vllm/entrypoints/openai/chat_completion/serving.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：6；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
