# SGLang Gemma 4 支持与 PR 历史

本文记录 SGLang 中与 Gemma 4 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Gemma 4 is a genuinely multi-surface family: text, MoE, multimodal, tool calling, and speculative decoding all matter.
- Fast prefill, quantized MoE, and tool-parser correctness are the main areas that changed rapidly after bring-up.

## 主要代码面

- `sglang/python/sglang/srt/models/gemma4_causal.py`
- `sglang/python/sglang/srt/models/gemma4_mm.py`
- `sglang/python/sglang/srt/models/gemma4_vision.py`
- `sglang/python/sglang/srt/models/gemma4_audio.py`

## 已合入 PR

- [#21952](https://github.com/sgl-project/sglang/pull/21952) `New Model: Gemma 4`：Initial Gemma 4 support in SGLang.
- [#22079](https://github.com/sgl-project/sglang/pull/22079) `Gemma4 nvfp4 fix`：Fixed the NVFP4 launch path.
- [#22408](https://github.com/sgl-project/sglang/pull/22408) `Adding Gemma 4 to Nightly CI`：Added model-family regression coverage.

## 配套 skill

- `skills/model-optimization/sglang/sglang-gemma4-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-gemma4-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Gemma 4`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-02 | [#21952](https://github.com/sgl-project/sglang/pull/21952) | merged | [New Model] Gemma 4 | model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py` |
| 2026-04-03 | [#22079](https://github.com/sgl-project/sglang/pull/22079) | merged | [nvidia] Gemma4 nvfp4 fix | attention/backend, kernel | `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` |
| 2026-04-09 | [#22408](https://github.com/sgl-project/sglang/pull/22408) | merged | [CI] Adding Gemma 4 to Nightly CI | tests/benchmarks | `test/registered/eval/test_vlms_mmmu_eval.py` |

### 逐 PR 代码 diff 阅读记录

### PR #21952 - [New Model] Gemma 4

- 链接：https://github.com/sgl-project/sglang/pull/21952
- 状态/时间：`merged`，created 2026-04-02, merged 2026-04-07；作者 `JustinTong0323`。
- 代码 diff 已读范围：`35` 个文件，`+6007/-70`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：kv, spec, attention, config, quant, cuda, moe, processor, vision, cache。
- 代码 diff 细节：
  - `python/sglang/srt/models/gemma4_causal.py` added +1009/-0 (1009 lines); hunk: +# Copyright 2025 SGLang Team; 符号: get_attention_sliding_window_size, Gemma4Router, __init__, fuse_scale
  - `python/sglang/srt/models/gemma4_mm.py` added +878/-0 (878 lines); hunk: +# Copyright 2025 SGLang Team; 符号: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4MultimodalEmbedder, __init__
  - `python/sglang/srt/models/gemma4_audio.py` added +873/-0 (873 lines); hunk: +# Copyright 2025 SGLang Team; 符号: Gemma4AudioRelativePositionEmbedding, __init__, _get_timing_signal_1d_pos, _relative_shift
  - `python/sglang/srt/models/gemma4_vision.py` added +599/-0 (599 lines); hunk: +# Copyright 2025 SGLang Team; 符号: _rotate_half, _apply_rotary, Gemma4VisionRotaryEmbedding, __init__
  - `python/sglang/srt/function_call/gemma4_detector.py` added +445/-0 (445 lines); hunk: +import json; 符号: _parse_gemma4_value, _parse_gemma4_array, _parse_gemma4_args, _find_matching_brace
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py`；patch 关键词为 kv, spec, attention, config, quant, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22079 - [nvidia] Gemma4 nvfp4 fix

- 链接：https://github.com/sgl-project/sglang/pull/22079
- 状态/时间：`merged`，created 2026-04-03, merged 2026-04-10；作者 `wenscarl`。
- 代码 diff 已读范围：`1` 个文件，`+8/-0`；代码面：attention/backend, kernel；关键词：attention, cuda, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +8/-0 (8 lines); hunk: def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):; 符号: _get_block_sizes_for_extend_attention
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`；patch 关键词为 attention, cuda, triton。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22408 - [CI] Adding Gemma 4 to Nightly CI

- 链接：https://github.com/sgl-project/sglang/pull/22408
- 状态/时间：`merged`，created 2026-04-09, merged 2026-04-17；作者 `kpham-sgl`。
- 代码 diff 已读范围：`1` 个文件，`+6/-3`；代码面：tests/benchmarks；关键词：test。
- 代码 diff 细节：
  - `test/registered/eval/test_vlms_mmmu_eval.py` modified +6/-3 (9 lines); hunk: ModelLaunchSettings("Efficient-Large-Model/NVILA-Lite-2B-hf"): ModelEvalMetrics(
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/eval/test_vlms_mmmu_eval.py`；patch 关键词为 test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/eval/test_vlms_mmmu_eval.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：3；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
