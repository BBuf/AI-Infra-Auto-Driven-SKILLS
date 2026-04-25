# SGLang Intern-S1 支持与 PR 历史

本文记录 SGLang 中与 Intern-S1 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Intern-S1 leans heavily on shared InternVL processor code in SGLang.
- Most regressions come from processor compatibility, parser behavior, and video-aware serving rather than the text stack alone.

## 主要代码面

- `sglang/python/sglang/srt/models/interns1.py`
- `sglang/python/sglang/srt/models/internvl.py`

## 已合入 PR

- [#9381](https://github.com/sgl-project/sglang/pull/9381) `InternS1 image token updates in InternVL processor`：Aligned the shared processor with Intern-S1 image semantics.
- [#12367](https://github.com/sgl-project/sglang/pull/12367) `Fix Intern-S1 accuracy and `/generate` input_ids support`：Closed early correctness gaps.
- [#14866](https://github.com/sgl-project/sglang/pull/14866) `Add tool calling and reasoning parser support for Intern-S1`：Added parser support that cookbook usage depends on.
- [#17040](https://github.com/sgl-project/sglang/pull/17040) `Support InternS1 text_config in InternVL processor`：Improved sub-config compatibility in shared processors.

## 配套 skill

- `skills/model-optimization/sglang/sglang-intern-s1-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-intern-s1-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Intern-S1`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-20 | [#9381](https://github.com/sgl-project/sglang/pull/9381) | merged | fix: InternS1 don't recognize image, updates image token for InternVL processor | multimodal/processor | `python/sglang/srt/conversation.py`, `python/sglang/srt/multimodal/processors/internvl.py` |
| 2025-10-30 | [#12367](https://github.com/sgl-project/sglang/pull/12367) | merged | [Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids | model wrapper, multimodal/processor | `python/sglang/srt/models/interns1.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py` |
| 2025-12-11 | [#14866](https://github.com/sgl-project/sglang/pull/14866) | merged | Adding tool calling and reasoning parser support for Intern-S1 | misc | `python/sglang/srt/function_call/internlm_detector.py`, `python/sglang/srt/constrained/base_grammar_backend.py`, `python/sglang/srt/constrained/xgrammar_backend.py` |
| 2026-01-13 | [#17040](https://github.com/sgl-project/sglang/pull/17040) | merged | fix(processor): support InternS1 text_config in InternVL processor | multimodal/processor | `python/sglang/srt/multimodal/processors/internvl.py` |

### 逐 PR 代码 diff 阅读记录

### PR #9381 - fix: InternS1 don't recognize image, updates image token for InternVL processor

- 链接：https://github.com/sgl-project/sglang/pull/9381
- 状态/时间：`merged`，created 2025-08-20, merged 2025-08-20；作者 `JustinTong0323`。
- 代码 diff 已读范围：`2` 个文件，`+9/-17`；代码面：multimodal/processor；关键词：config, expert, processor, spec, vision。
- 代码 diff 细节：
  - `python/sglang/srt/conversation.py` modified +2/-15 (17 lines); hunk: def generate_chat_conv(; def generate_chat_conv(; 符号: generate_chat_conv, generate_chat_conv
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2 (9 lines); hunk: def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):; def process_image_internvl(image, input_size=448, max_num=12):; 符号: __init__, process_image_internvl
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/conversation.py`, `python/sglang/srt/multimodal/processors/internvl.py`；patch 关键词为 config, expert, processor, spec, vision。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/conversation.py`, `python/sglang/srt/multimodal/processors/internvl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12367 - [Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids

- 链接：https://github.com/sgl-project/sglang/pull/12367
- 状态/时间：`merged`，created 2025-10-30, merged 2025-11-03；作者 `hhaAndroid`。
- 代码 diff 已读范围：`3` 个文件，`+8/-41`；代码面：model wrapper, multimodal/processor；关键词：attention, config, flash, fp8, processor, quant, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/interns1.py` modified +3/-21 (24 lines); hunk: -from typing import Iterable, List, Optional, Set, Tuple; def __init__(; 符号: __init__, pixel_shuffle, extract_feature, load_weights
  - `python/sglang/srt/models/internvl.py` modified +1/-19 (20 lines); hunk: -from typing import Iterable, List, Optional, Set, Tuple, Union; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights, load_weights
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +4/-1 (5 lines); hunk: async def process_mm_data_async(; 符号: process_mm_data_async
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/interns1.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py`；patch 关键词为 attention, config, flash, fp8, processor, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/interns1.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14866 - Adding tool calling and reasoning parser support for Intern-S1

- 链接：https://github.com/sgl-project/sglang/pull/14866
- 状态/时间：`merged`，created 2025-12-11, merged 2025-12-16；作者 `KennyYao2001`。
- 代码 diff 已读范围：`6` 个文件，`+290/-14`；代码面：misc；关键词：kv, moe, spec。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/internlm_detector.py` added +248/-0 (248 lines); hunk: +# modified from https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/tool_parser/internlm2_parser.py; 符号: InternlmDetector, __init__, has_tool_call, get_arguments
  - `python/sglang/srt/constrained/base_grammar_backend.py` modified +19/-7 (26 lines); hunk: def create_grammar_backend(; 符号: create_grammar_backend
  - `python/sglang/srt/constrained/xgrammar_backend.py` modified +18/-5 (23 lines); hunk: def __repr__(self):; def __init__(; 符号: __repr__, TokenizerNotSupportedError, XGrammarGrammarBackend, __init__
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +2/-2 (4 lines); hunk: def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:; 符号: _get_reasoning_from_request
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunk: from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector; class FunctionCallParser:; 符号: FunctionCallParser:, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/internlm_detector.py`, `python/sglang/srt/constrained/base_grammar_backend.py`, `python/sglang/srt/constrained/xgrammar_backend.py`；patch 关键词为 kv, moe, spec。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/internlm_detector.py`, `python/sglang/srt/constrained/base_grammar_backend.py`, `python/sglang/srt/constrained/xgrammar_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17040 - fix(processor): support InternS1 text_config in InternVL processor

- 链接：https://github.com/sgl-project/sglang/pull/17040
- 状态/时间：`merged`，created 2026-01-13, merged 2026-01-26；作者 `Mahdi-CV`。
- 代码 diff 已读范围：`1` 个文件，`+12/-4`；代码面：multimodal/processor；关键词：config, processor。
- 代码 diff 细节：
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +12/-4 (16 lines); hunk: def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):; def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):; 符号: __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/multimodal/processors/internvl.py`；patch 关键词为 config, processor。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/multimodal/processors/internvl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：4；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
