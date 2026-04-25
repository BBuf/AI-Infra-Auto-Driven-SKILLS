# SGLang Ernie4.5 / Ernie4.5-VL 支持与 PR 历史

本文记录 SGLang 在提交 `c122d343adb969cd9bbd1af2ca86727a11be3845`
附近的 Ernie4.5 多模态支持。

- 状态: 当前 mainline 已支持

## 核心结论

- Ernie4.5-VL 不是薄别名，而是带独立 runtime 和 processor 的原生落地。
- 后续最重要的优化工作基本都集中在 vision rotary 路径:
  先是 fused Triton kernel，再是 cos/sin cache 复用。

## 主要代码面

- `sglang/python/sglang/srt/models/ernie45_vl.py`
- `sglang/python/sglang/srt/models/ernie45_moe_vl.py`
- `sglang/python/sglang/srt/multimodal/processors/ernie45_vl.py`
- `sglang/python/sglang/srt/layers/rotary_embedding.py`

## 已合入 PR

- [#15679](https://github.com/sgl-project/sglang/pull/15679)
  `Add Ernie4.5 VL model support`
  已审 diff: `6` 个文件，`2072` 行新增。
  它加入原生 Ernie4.5-VL / MoE-VL runtime、processor 和多模态注册。
- [#18856](https://github.com/sgl-project/sglang/pull/18856)
  `Optimize Ernie4.5-VL rotary embedding with fused triton kernel`
  已审 diff: `1` 个文件，`268` 行新增，`3` 行删除。
  它为 Ernie4.5 的 `(h, w, t)` 布局加入 fused Triton Q/K rotary kernel。
- [#19743](https://github.com/sgl-project/sglang/pull/19743)
  `Support cos sin cache for Ernie4.5-VL`
  已审 diff: `1` 个文件，`34` 行新增，`12` 行删除。
  它把 vision tower 改成复用 `get_rope(...).get_cos_sin(...)`。

## 当前结论

如果 Ernie4.5-VL 回归了，先确认当前走的是哪条 rotary 路径，再去改整套模型。
这个家族很多 correctness / performance 变化都集中在 vision rotary 处理。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `ERNIE 4.5 / ERNIE 4.5 VL`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-12-23 | [#15679](https://github.com/sgl-project/sglang/pull/15679) | merged | [Model] Add Ernie4.5 VL model support | model wrapper, MoE/router, multimodal/processor, docs/config | `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py` |
| 2026-02-15 | [#18856](https://github.com/sgl-project/sglang/pull/18856) | merged | [VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel | misc | `python/sglang/srt/layers/rotary_embedding.py` |
| 2026-03-03 | [#19743](https://github.com/sgl-project/sglang/pull/19743) | merged | [VLM] Support cos sin cache for Ernie4.5-VL | model wrapper | `python/sglang/srt/models/ernie45_vl.py` |

### 逐 PR 代码 diff 阅读记录

### PR #15679 - [Model] Add Ernie4.5 VL model support

- 链接：https://github.com/sgl-project/sglang/pull/15679
- 状态/时间：`merged`，created 2025-12-23, merged 2026-01-26；作者 `CSWYF3634076`。
- 代码 diff 已读范围：`6` 个文件，`+2072/-0`；代码面：model wrapper, MoE/router, multimodal/processor, docs/config；关键词：config, vision, kv, moe, spec, attention, cache, cuda, expert, processor。
- 代码 diff 细节：
  - `python/sglang/srt/models/ernie45_vl.py` added +845/-0 (845 lines); hunk: +# Copyright 2023-2025 SGLang Team; 符号: Ernie4_5_VisionMLP, __init__, forward, Ernie4_5_VisionBlock
  - `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0 (552 lines); hunk: +# Copyright 2023-2025 SGLang Team; 符号: Ernie4_5_VLMoeAttention, __init__, forward, Ernie4_5_VLMoeMoE
  - `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0 (417 lines); hunk: +import math; 符号: smart_resize, resize_image, round_by_factor, ceil_by_factor
  - `python/sglang/srt/layers/rotary_embedding.py` modified +256/-0 (256 lines); hunk: def get_rope_index_glm4v(; def _get_llm_pos_ids_for_vision(; 符号: get_rope_index_glm4v, get_rope_index_ernie45, _get_feat_extract_output_lengths, _get_llm_pos_ids_for_vision
  - `docs/supported_models/multimodal_language_models.md` modified +1/-0 (1 lines); hunk: in the GitHub search bar.
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py`；patch 关键词为 config, vision, kv, moe, spec, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18856 - [VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel

- 链接：https://github.com/sgl-project/sglang/pull/18856
- 状态/时间：`merged`，created 2026-02-15, merged 2026-02-16；作者 `yuan-luo`。
- 代码 diff 已读范围：`1` 个文件，`+268/-3`；代码面：misc；关键词：cache, cuda, kv, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/rotary_embedding.py` modified +268/-3 (271 lines); hunk: def _compute_cos_sin_cache(self) -> torch.Tensor:; def forward_native( # type: ignore[override]; 符号: _compute_cos_sin_cache, _triton_ernie45_rope_qk_fused, triton_ernie45_rope_fused_inplace, Ernie4_5_VLRotaryEmbedding
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/rotary_embedding.py`；patch 关键词为 cache, cuda, kv, triton。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/rotary_embedding.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19743 - [VLM] Support cos sin cache for Ernie4.5-VL

- 链接：https://github.com/sgl-project/sglang/pull/19743
- 状态/时间：`merged`，created 2026-03-03, merged 2026-03-04；作者 `yuan-luo`。
- 代码 diff 已读范围：`1` 个文件，`+34/-12`；代码面：model wrapper；关键词：cache, config, moe, processor, quant, triton, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/ernie45_vl.py` modified +34/-12 (46 lines); hunk: from sglang.srt.layers.logits_processor import LogitsProcessor; def forward(; 符号: forward, __init__, dtype, device
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/ernie45_vl.py`；patch 关键词为 cache, config, moe, processor, quant, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/ernie45_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：3；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
