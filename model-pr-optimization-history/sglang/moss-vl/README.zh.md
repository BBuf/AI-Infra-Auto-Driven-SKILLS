# SGLang Moss-VL 支持与优化时间线

范围：Moss-VL 原生 SGLang runtime、image/video processor、conversation template、多模态 scheduler metadata、cross-attention custom mask、flashinfer prefill 要求。

证据快照：SGLang `origin/main` `bca3dd958`（`2026-04-24`）。完整卡片见：`skills/model-optimization/sglang/sglang-moss-vl-optimization/references/pr-history.md`。

## 已阅读 Diff 的 PR

#23454 新增 Moss-VL runtime 支持。已完整阅读 `3397` 行 diff、`10` 个文件。该 PR 新增 `moss_vl.py`、`multimodal/processors/moss_vl.py`，在 `schedule_batch.py` 中加入 Moss-VL 多模态字段，注册 `moss-vl` conversation template，并在 `server_args.py` 中要求 flashinfer prefill。

核心契约：Moss-VL vision token 会按 frame 插入 separator；processor 的 frame visibility 会转换成 packed cross-attention custom mask；encoder-prefix placeholder token 在 text extend 前会被剥离。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `MOSS-VL`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-22 | [#23454](https://github.com/sgl-project/sglang/pull/23454) | merged | [srt] Add Moss-VL Python runtime support | model wrapper, attention/backend, multimodal/processor, scheduler/runtime, docs/config | `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`, `python/sglang/srt/managers/schedule_batch.py` |

### 逐 PR 代码 diff 阅读记录

### PR #23454 - [srt] Add Moss-VL Python runtime support

- 链接：https://github.com/sgl-project/sglang/pull/23454
- 状态/时间：`merged`，created 2026-04-22, merged 2026-04-24；作者 `zsj555`。
- 代码 diff 已读范围：`10` 个文件，`+2401/-6`；代码面：model wrapper, attention/backend, multimodal/processor, scheduler/runtime, docs/config；关键词：attention, config, processor, spec, cuda, flash, vision, cache, kv, mla。
- 代码 diff 细节：
  - `python/sglang/srt/models/moss_vl.py` added +1643/-0 (1643 lines); hunk: +"""PyTorch Moss-VL model for SGLang - Qwen3VL Vision + Text with Cross Attention."""; 符号: MossVLVisionMLP, __init__, forward, MossVLVisionPatchEmbed
  - `python/sglang/srt/multimodal/processors/moss_vl.py` added +612/-0 (612 lines); hunk: +import asyncio; 符号: MossVLImageProcessor, __init__, _build_mm_items, _build_vision_token_info
  - `python/sglang/srt/managers/schedule_batch.py` modified +70/-0 (70 lines); hunk: class MultimodalProcessorOutput:; def from_dict(d: dict) -> "MultimodalProcessorOutput":; 符号: MultimodalProcessorOutput:, from_dict, MultimodalInputs:, release_features
  - `python/sglang/srt/parser/conversation.py` modified +29/-2 (31 lines); hunk: def get_prompt(self) -> str:; def generate_chat_conv(; 符号: get_prompt, generate_chat_conv, generate_chat_conv, generate_chat_conv
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +12/-2 (14 lines); hunk: async def _tokenize_one_request(; 符号: _tokenize_one_request
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`, `python/sglang/srt/managers/schedule_batch.py`；patch 关键词为 attention, config, processor, spec, cuda, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`, `python/sglang/srt/managers/schedule_batch.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：1；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
