# SGLang DeepSeek V3.1 支持与优化时间线

本文基于 SGLang `origin/main` 最新快照 `929e00eea`、sgl-cookbook `origin/main` 快照 `8ec4d03`，以及 DeepSeek V3.1 相关 merged 和 open PR 的 patch 阅读结果整理。范围只覆盖 DeepSeek V3.1 / DeepSeek-V3.1-Terminus 的独立差异：tool calling、thinking mode、chat template、streaming parser、结构化标签、加载修复、MTP 验证和 MoE config。DeepSeek V3/R1 的 MLA、MoE、量化、DeepEP 主线不在本文重复，DeepSeek V3.2 的 DSA/NSA 稀疏注意力也单独成文。

结论：截至 `929e00eea`，DeepSeek V3.1 仍复用 `DeepseekV3ForCausalLM` 和 `deepseek_v2.py` 的主模型路径，独立部分是 `deepseekv31` tool parser、`tool_chat_template_deepseekv31.jinja` 和 `thinking` chat-template 参数。V3.1 的 tool-call 格式和 V3 不同，不包含 `function` literal，也不使用 fenced JSON。当前 main 已有基础 tool calling、thinking parser、dict/string argument 类型处理、结构化标签 trigger 修复和 H200 TP8/MTP 验证；新增运行时内容包括 thinking token radix-cache strip，以及继承自 V3/R1 的 adaptive EAGLE、PCG + speculative decoding、spec v2 adaptive spec。后续需要跟进 open PR 里的 streaming 参数丢失、tool 输出后缺 Assistant token、NPU 部署文档、parser CPU 单测和 spec v2 自适应 speculative decoding。

## 1. 时间线总览

| 创建日期 | PR | 状态 | 主线 | 代码区域 | 作用 |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-21 | [#9446](https://github.com/sgl-project/sglang/pull/9446) | merged | tool calling | `deepseekv31_detector.py`、V3.1 template、parser 注册 | 新增 DeepSeek V3.1 tool-call parser 和 chat template。 |
| 2025-08-21 | [#9464](https://github.com/sgl-project/sglang/pull/9464) | merged | thinking parser | `serving_chat.py`、reasoning parser、docs | 增加 DeepSeek V3.1 thinking mode 支持，使用 `--reasoning-parser deepseek-v3`。 |
| 2025-08-23 | [#9544](https://github.com/sgl-project/sglang/pull/9544) | merged | docs | benchmark / basic usage docs | 补充 DeepSeek V3.1 支持文档。 |
| 2025-10-25 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | chat template | V3/V3.1/V3.2 templates、template test | 修复 tool arguments 是 dict 或 string 时的 double-escape 问题。 |
| 2025-11-13 | [#13190](https://github.com/sgl-project/sglang/pull/13190) | merged | nightly test | V3.1/V3.2 perf tests | 移除过时的 `enable_dp_attention`。 |
| 2025-11-17 | [#13394](https://github.com/sgl-project/sglang/pull/13394) | merged | structural tag | `DeepSeekV31Detector.structure_info` | 将 structural trigger 改成泛化的 `<｜tool▁call▁begin｜>`。 |
| 2025-11-26 | [#13954](https://github.com/sgl-project/sglang/pull/13954) | merged | loading | `deepseek_v2.py` | 修复 DeepSeek V3.1 加载问题。 |
| 2026-01-07 | [#16660](https://github.com/sgl-project/sglang/pull/16660) | merged | CI | `test/registered/8-gpu-models/test_deepseek_v31.py` | 开启 DeepSeek V3.1 H200 nightly 测试。 |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | MoE tuning | fused MoE Triton configs | 为 V3.1/V3.2 的 H20/H20-3E FP8 MoE shape 增加 tuning config。 |
| 2026-01-26 | [#17761](https://github.com/sgl-project/sglang/pull/17761) | open | chat template | V3.1/V3.2 templates | 修复 tool output 后缺少 Assistant token 的问题。 |
| 2026-02-04 | [#18236](https://github.com/sgl-project/sglang/pull/18236) | open | streaming parser | `deepseekv31_detector.py` | 修复 streaming 下首 chunk 参数丢失和 normal text 丢失。 |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | MTP/spec | EAGLE runtime、spec workers | 增加 EAGLE top-k=1 自适应 `speculative_num_steps`，V3.1 MTP 继承受益。 |
| 2026-03-31 | [#21739](https://github.com/sgl-project/sglang/pull/21739) | open | NPU docs | Ascend best practice docs | 更新 V3.1/V3.2 NPU 部署说明。 |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | PCG/spec | model runner、PCG runner | 允许 piecewise CUDA graph 和 speculative decoding 同时使用。 |
| 2026-04-09 | [#22433](https://github.com/sgl-project/sglang/pull/22433) | open | parser tests | `test_deepseekv31_detector.py` | 增加 DeepSeekV31Detector CPU 单测。 |
| 2026-04-16 | [#22981](https://github.com/sgl-project/sglang/pull/22981) | open | parser tests | function-call detectors | 给多个缺失的 function-call detector 增加 CPU 单测。 |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | reasoning cache | model config、scheduler、radix cache、reasoning parser | 探索 parser-gated 两阶段 reasoning radix-cache stripping，已关闭。 |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | reasoning cache | `schedule_batch.py`、`mem_cache/common.py`、`server_args.py` | 增加可选的 thinking token radix-cache strip。 |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | spec v2 | scheduler output processor、EAGLE v2 workers | 把 adaptive speculative decoding 扩展到 spec v2。 |

## 1.1 Parser/Template 周边 PR

V3.1 的 parser/template 周边还需要显式记录这些 PR：

- `#9468`：更新 reasoning parser 文档，承接 `#9464` 的 thinking parser 支持。
- `#9895`、`#14837`：两次更新 `tool_chat_template_deepseekv31.jinja`，其中 `#14837` 是 2025-12-10 的 auto-sync。
- `#10550`、`#11223`、`#11589`、`#21593`：tool-choice、tool parser 文档、`tool_choice="auto"` 参数处理、native-format constrained decoding/parser 修复，都会影响 V3.1 tool serving。
- `#10875`、`#11189`、`#17178`：请求级 thinking 开关、eval `--thinking-mode`、以及移除 `deepseek-r1` thinking-mode choice，帮助界定 V3.1 thinking 与 R1 parser 的边界。
- `#17141`、`#17320`、`#17558`：tool 内容后 `finish_reason="stop"` / Assistant token 的 closed 尝试；当前仍以 open `#17761` 作为跟踪入口。
- `#22950`、`#23315`：区分已关闭的 reasoning-cache strip 探索和当前 merged 的 thinking token radix-cache strip。
- `#21599`、`#22128`、`#23336`：V3.1 MTP 继承的 speculative decoding 基础设施，分别对应 adaptive EAGLE、PCG + speculative decoding、spec v2 adaptive spec。

## 2. 为什么 V3.1 要单独写

DeepSeek V3.1 的模型计算主干仍然是 DeepSeek V3/R1 共享路径：`DeepseekV3ForCausalLM`、`DeepseekV2AttentionMLA`、`DeepseekV2MoE`、共享的 DeepSeek weight loader、NextN/MTP 和 server-side backend 选择。因此，如果问题是 MLA backend、FP8/FP4/W4AFP8、shared expert fusion、DeepEP、LoRA、MTP draft loading 或 DP attention，应回到 DeepSeek V3/R1 文档排查。

但 V3.1 的用户可见行为发生了明显变化：

- hybrid thinking：同一个模型可以通过 `thinking` 参数切换思考和非思考。
- tool-call 格式变了：函数名直接跟在 `<｜tool▁call▁begin｜>` 后面，中间用 `<｜tool▁sep｜>` 分隔 JSON 参数。
- chat template 需要同时处理 system prompt、tools、assistant prefix、tool output、thinking 标记和多轮 tool call。
- streaming parser 是自定义的，不等价于 non-streaming parser。
- constrained decoding 需要正确的 `structure_info.trigger`。

所以 V3.1 的核心价值不是新 MLA kernel，而是把 DeepSeek V3 主干包装成可用于 agent 和 hybrid reasoning 的 OpenAI-compatible serving 形态。

## 3. `#9446`：V3.1 tool-call parser 和 template

`#9446` 新增了 `examples/chat_template/tool_chat_template_deepseekv31.jinja`、`python/sglang/srt/function_call/deepseekv31_detector.py`，并把 `deepseekv31` 注册进 function-call parser 表。

V3.1 的 tool-call 格式是：

```text
<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>{tool_name}<｜tool▁sep｜>{json_arguments}<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>
```

和 V3 的区别很关键：

- V3.1 没有 `<｜tool▁call▁begin｜>function<｜tool▁sep｜>` 里的 `function` literal。
- V3.1 参数是直接 JSON 字符串，不包在 ```json fenced block 里。
- 多个 tool call 直接连续拼接，不插入额外分隔符。

`DeepSeekV31Detector.detect_and_parse` 的逻辑是先找到外层 `<｜tool▁calls▁begin｜>`，再用 `func_call_regex` 抓每个 `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>`，最后用 `func_detail_regex` 拆出函数名和参数。参数通过 `json.loads` 解析后，再交给 `parse_base_json` 对齐 OpenAI tool schema。

这一阶段的重点是格式正确，而不是性能。只要 V3/V3.1 parser 互换，就会出现无法解析、函数名不对或参数被当普通文本的症状。

## 4. `#9464`：thinking mode 不是 R1 parser

`#9464` 增加了 DeepSeek V3.1 thinking parser 支持和文档。它明确了 V3.1 使用：

```shell
--reasoning-parser deepseek-v3
```

并在请求里通过：

```json
{"chat_template_kwargs": {"thinking": true}}
```

打开 thinking。这个行为和 R1 不同。R1 使用 `deepseek-r1` parser，且 R1 parser 会处理没有 `<think>` 开头的 reasoning；V3.1 更接近 Qwen3-style hybrid thinking，由 chat template 决定是否注入 `<think>` 或 `</think>`。

当前 `tool_chat_template_deepseekv31.jinja` 中，最后 user 后如果 `add_generation_prompt` 为 true，会输出 `<｜Assistant｜>`，然后根据 `thinking` 决定追加 `<think>` 或 `</think>`。assistant 消息自身如果包含 `</think>`，模板会切掉 reasoning 部分，只保留 content 输出。这是为了让历史消息和下一轮 generation 的格式保持一致。

DeepSeek-V3.1-Speciale 是重要例外。cookbook 当前明确写了 Speciale 不支持 tool calling，应视为 deep reasoning 模型，而不是 V3.1 tool-use 目标。

thinking mode 还会和 radix cache 交叉。`#22950` 是 closed 的 parser-gated reasoning cache strip 方案；当前 main 要看 `#23315`，它在 `server_args.py`、`schedule_batch.py` 和 `mem_cache/common.py` 增加 opt-in strip，把 thinking tokens 从 radix-cache entry 中剥离。对 V3.1 来说，这不是 `deepseekv31` tool parser 变化，而是决定 `<think>` / `</think>` 是否会被 prefix cache 复用的缓存层行为。

## 5. `#12123`：dict/string 参数不 double-escape

多轮 tool calling 里，OpenAI API 对象中的 `tool["function"]["arguments"]` 有时是 dict，有时已经是 JSON 字符串。如果模板一律 `tojson`，已经序列化过的字符串会变成带反斜杠的 JSON string。

`#12123` 对 V3、V3.1、V3.2 三个 DeepSeek template 做了同样修复：

```jinja
{% set formatted_args = tool['function']['arguments'] if tool['function']['arguments'] is string else tool['function']['arguments']|tojson %}
```

同时补了 `test_deepseek_chat_templates.py`，覆盖：

- dict 参数要被正常 JSON 编码。
- string 参数要原样使用。
- 多个 tool call 中 dict 和 string 混合时都不能 double-escape。

这类问题不会体现在模型吞吐上，但会直接破坏 agent 多轮 tool-use，因为下一轮 prompt 里的历史 tool call 会变成错误 JSON。

## 6. `#13394`、`#18236`、`#22433`：结构化标签与 streaming parser

`#13394` 修的是 constrained decoding 触发点。原来 `structure_info.trigger` 包含函数名和 `<｜tool▁sep｜>`，这意味着 parser 只有在知道具体函数名后才触发结构约束。修复后 trigger 是通用的：

```python
trigger="<｜tool▁call▁begin｜>"
```

`begin` 仍然是 name-specific：

```python
begin="<｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>"
```

`#18236` 仍 open，但它指出了当前 streaming parser 的两个风险。第一，如果函数名和 JSON 参数出现在同一个 chunk，当前代码可能只发出 name，不处理首个参数 diff。第二，如果 tool marker 前有普通文本，streaming 路径可能返回空 normal_text，而 non-streaming 是能保留前缀文本的。PR 方案是增加 `_normal_text_sent`，并在第一次看到 `<｜tool▁call▁begin｜>` 时切出 marker 前的 normal text，同时只要 `func_args_raw` 非空就处理参数 diff。

`#22433` 也是 open，但它补齐了长期应保留的 CPU 单测形状：`has_tool_call`、无 tool call 普通文本、单 tool、多 tool、invalid JSON fallback、unknown tool、unicode 参数、streaming chunk、tool index、`structure_info` 和 structural tag support。后续修改 V3.1 parser 时，最小验证面应先覆盖这类 CPU 单测，再运行 8 卡模型验证。

## 7. 加载、MTP 和 MoE config

`#13954` 修复 DeepSeek V3.1 loading issue，落点是 `deepseek_v2.py`。这再次说明 V3.1 的模型计算面是共享 DeepSeek 主干，parser/template 只是独立的 OpenAI serving 面。如果 V3.1 launch 在权重加载、MLA 初始化、MoE 参数映射上失败，不应只看 `deepseekv31_detector.py`。

`#16660` 把 DeepSeek V3.1 纳入 H200 nightly，测试包含：

- TP8 base。
- TP8 + EAGLE MTP，附带 `SGLANG_ENABLE_SPEC_V2=1`。
- accuracy 使用 GSM8K，baseline `0.935`。
- performance profile 输出到 `performance_profiles_deepseek_v31`。

`#13190` 移除 V3.1/V3.2 nightly perf 里的过时 `enable_dp_attention`，避免性能测试还带着旧 server args 时代的配置。

继承的 MTP 基础设施也要一起看：`#21599` 让 EAGLE top-k=1 draft step 数可自适应，`#22128` 让 PCG 可以和 speculative decoding 共存，open `#23336` 把 adaptive spec 推到 spec v2 workers。因为 `#16660` 的 V3.1 MTP lane 明确带 `SGLANG_ENABLE_SPEC_V2=1`，这些 PR 不是 V3.1 parser 工作，但会影响 V3.1 TP8+MTP 的真实运行形态。

`#17133` 则是 MoE config 性能线。它为 V3.1/V3.2 的 DeepSeek-family shape 加入 H20 和 H20-3E fused MoE config，典型文件名包含 `E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]`。这里的 `257` 对应 256 routed experts 加 fused shared expert 的形态。

## 8. 当前验证面与未合入方向

当前验证面：

- `test/manual/test_deepseek_v31.py`：TP8 和 TP8+MTP，GSM8K baseline `0.935`。
- `test/manual/nightly/test_deepseek_v31_perf.py`：V3.1 nightly perf。
- `test/manual/test_deepseek_chat_templates.py`：V3/V3.1/V3.2 template dict/string 参数测试。
- open `#22433` 的 `test/registered/unit/function_call/test_deepseekv31_detector.py`：可作为 parser CPU 单测基线。

需要跟进的 open PR：

- `#17761`：V3.1/V3.2 tool output 后缺 Assistant token。
- `#18236`：V3.1 streaming function-call 参数和 normal text 丢失。
- `#21739`：V3.1/V3.2 NPU 部署文档。
- `#22433`：DeepSeekV31Detector CPU 单测。
- `#22981`：多个 function-call detector 的 CPU 单测补齐。
- `#23336`：spec v2 adaptive speculative decoding。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `DeepSeek V3.1`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-21 | [#9446](https://github.com/sgl-project/sglang/pull/9446) | merged | Support DeepSeek-V3.1 tool call | docs/config | `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `python/sglang/srt/function_call/function_call_parser.py` |
| 2025-08-21 | [#9464](https://github.com/sgl-project/sglang/pull/9464) | merged | Add deepseek v3.1 thinking parser support and update docs | docs/config | `docs/basic_usage/openai_api_completions.ipynb`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py` |
| 2025-08-21 | [#9468](https://github.com/sgl-project/sglang/pull/9468) | merged | Update reasoning parser doc | docs/config | `docs/basic_usage/openai_api_completions.ipynb` |
| 2025-08-23 | [#9544](https://github.com/sgl-project/sglang/pull/9544) | merged | [doc] deepseekv31 support | tests/benchmarks, docs/config | `benchmark/deepseek_v3/README.md`, `docs/basic_usage/deepseek.md` |
| 2025-09-02 | [#9895](https://github.com/sgl-project/sglang/pull/9895) | merged | Update tool_chat_template_deepseekv31.jinja | docs/config | `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-09-17 | [#10550](https://github.com/sgl-project/sglang/pull/10550) | merged | Use jsonschema to constrain required or specific tool choice | tests/benchmarks | `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py` |
| 2025-09-24 | [#10875](https://github.com/sgl-project/sglang/pull/10875) | merged | feat(reasoning): improve enable thinking from request | misc | `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-10-03 | [#11189](https://github.com/sgl-project/sglang/pull/11189) | merged | Add --thinking-mode to run_eval | tests/benchmarks | `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py` |
| 2025-10-05 | [#11223](https://github.com/sgl-project/sglang/pull/11223) | merged | Update tool parser and related documentation | docs/config | `docs/advanced_features/tool_parser.ipynb`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py` |
| 2025-10-14 | [#11589](https://github.com/sgl-project/sglang/pull/11589) | merged | [Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector | misc | `python/sglang/srt/function_call/deepseekv31_detector.py` |
| 2025-10-25 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | Fix DeepSeek chat templates to handle tool call arguments type checking (#11700) | tests/benchmarks, docs/config | `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-11-13 | [#13190](https://github.com/sgl-project/sglang/pull/13190) | merged | Remove enable_dp_attention in deepseek nightly tests | tests/benchmarks | `test/srt/nightly/test_deepseek_v32_perf.py`, `test/srt/nightly/test_deepseek_v31_perf.py` |
| 2025-11-17 | [#13394](https://github.com/sgl-project/sglang/pull/13394) | merged | Fix DeepSeekV31's structural tag trigger | misc | `python/sglang/srt/function_call/deepseekv31_detector.py` |
| 2025-11-26 | [#13954](https://github.com/sgl-project/sglang/pull/13954) | merged | Fix Deepseek v3.1 loading issue | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-10 | [#14837](https://github.com/sgl-project/sglang/pull/14837) | merged | [Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210) | docs/config | `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2026-01-07 | [#16660](https://github.com/sgl-project/sglang/pull/16660) | merged | [CI] Enable dpsk v31 test on nightly H200 | model wrapper, tests/benchmarks | `test/registered/8-gpu-models/test_deepseek_v31.py` |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` |
| 2026-01-15 | [#17141](https://github.com/sgl-project/sglang/pull/17141) | closed | fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content. | docs/config | `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2026-01-16 | [#17178](https://github.com/sgl-project/sglang/pull/17178) | merged | Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py | tests/benchmarks | `python/sglang/test/run_eval.py` |
| 2026-01-19 | [#17320](https://github.com/sgl-project/sglang/pull/17320) | closed | fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content. | docs/config | `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2026-01-22 | [#17558](https://github.com/sgl-project/sglang/pull/17558) | closed | fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content. | docs/config | `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2026-01-26 | [#17761](https://github.com/sgl-project/sglang/pull/17761) | open | fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates | tests/benchmarks, docs/config | `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2026-02-04 | [#18236](https://github.com/sgl-project/sglang/pull/18236) | open | Fix function call arguments missing in streaming mode for DeepSeek V3.1 | misc | `python/sglang/srt/function_call/deepseekv31_detector.py` |
| 2026-03-28 | [#21593](https://github.com/sgl-project/sglang/pull/21593) | merged | Fix tool call constrained decoding and parsing for models with native formats | tests/benchmarks | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py` |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1 | kernel, scheduler/runtime, tests/benchmarks, docs/config | `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py` |
| 2026-03-31 | [#21739](https://github.com/sgl-project/sglang/pull/21739) | open | [NPU] Update DeepSeek-V3.1 and DeepSeek-V3.2 model deployment instructions in documentation | docs/config | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | Allow piecewise CUDA graph with speculative decoding | kernel, scheduler/runtime, tests/benchmarks | `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-04-09 | [#22433](https://github.com/sgl-project/sglang/pull/22433) | open | [Test] Add unit tests for DeepSeekV31Detector | tests/benchmarks | `test/registered/unit/function_call/test_deepseekv31_detector.py` |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373) | scheduler/runtime, tests/benchmarks, docs/config | `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| 2026-04-16 | [#22981](https://github.com/sgl-project/sglang/pull/22981) | open | [Test] Add unit tests for 7 missing function call detectors | tests/benchmarks | `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_kimik2_detector.py` |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | Opt-in strip of thinking tokens from radix cache | scheduler/runtime, tests/benchmarks | `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py` |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | [SPEC V2][2/N] feat: adaptive spec support spec v2 | multimodal/processor, scheduler/runtime | `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` |

### 逐 PR 代码 diff 阅读记录

### PR #9446 - Support DeepSeek-V3.1 tool call

- 链接：https://github.com/sgl-project/sglang/pull/9446
- 状态/时间：`merged`，created 2025-08-21, merged 2025-08-27；作者 `Xu-Wenqing`。
- 代码 diff 已读范围：`3` 个文件，`+315/-0`；代码面：docs/config；关键词：kv, moe, spec。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/deepseekv31_detector.py` added +222/-0 (222 lines); hunk: +import json; 符号: DeepSeekV31Detector, __init__, has_tool_call, detect_and_parse
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunk: +{% if not add_generation_prompt is defined %}
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunk: from sglang.srt.function_call.base_format_detector import BaseFormatDetector; class FunctionCallParser:; 符号: FunctionCallParser:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `python/sglang/srt/function_call/function_call_parser.py`；patch 关键词为 kv, moe, spec。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `python/sglang/srt/function_call/function_call_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9464 - Add deepseek v3.1 thinking parser support and update docs

- 链接：https://github.com/sgl-project/sglang/pull/9464
- 状态/时间：`merged`，created 2025-08-21, merged 2025-08-21；作者 `JustinTong0323`。
- 代码 diff 已读范围：`3` 个文件，`+136/-78`；代码面：docs/config；关键词：config, doc, spec。
- 代码 diff 细节：
  - `docs/basic_usage/openai_api_completions.ipynb` modified +123/-69 (192 lines); hunk: "print_highlight(f\"Response: {response}\")"; " print(chunk.choices[0].delta.content, end=\"\")"
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +9/-6 (15 lines); hunk: def _get_enable_thinking_from_request(self, request: ChatCompletionRequest) -> b; 符号: _get_enable_thinking_from_request, _process_tool_call_stream
  - `python/sglang/srt/reasoning_parser.py` modified +4/-3 (7 lines); hunk: class ReasoningParser:; 符号: ReasoningParser:, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/basic_usage/openai_api_completions.ipynb`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`；patch 关键词为 config, doc, spec。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/basic_usage/openai_api_completions.ipynb`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9468 - Update reasoning parser doc

- 链接：https://github.com/sgl-project/sglang/pull/9468
- 状态/时间：`merged`，created 2025-08-21, merged 2025-08-22；作者 `JustinTong0323`。
- 代码 diff 已读范围：`1` 个文件，`+46/-22`；代码面：docs/config；关键词：doc。
- 代码 diff 细节：
  - `docs/basic_usage/openai_api_completions.ipynb` modified +46/-22 (68 lines); hunk: "
",; ")
",
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/basic_usage/openai_api_completions.ipynb`；patch 关键词为 doc。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/basic_usage/openai_api_completions.ipynb` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9544 - [doc] deepseekv31 support

- 链接：https://github.com/sgl-project/sglang/pull/9544
- 状态/时间：`merged`，created 2025-08-23, merged 2025-08-23；作者 `XiaotongJiang`。
- 代码 diff 已读范围：`2` 个文件，`+82/-4`；代码面：tests/benchmarks, docs/config；关键词：attention, benchmark, config, doc, fp8, mla, spec。
- 代码 diff 细节：
  - `benchmark/deepseek_v3/README.md` modified +80/-2 (82 lines); hunk: -# DeepSeek V3 Support; Add performance optimization options (#performance-optimization-options) as nee
  - `docs/basic_usage/deepseek.md` modified +2/-2 (4 lines); hunk: SGLang provides many optimizations specifically designed for the DeepSeek models
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/deepseek_v3/README.md`, `docs/basic_usage/deepseek.md`；patch 关键词为 attention, benchmark, config, doc, fp8, mla。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `benchmark/deepseek_v3/README.md`, `docs/basic_usage/deepseek.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9895 - Update tool_chat_template_deepseekv31.jinja

- 链接：https://github.com/sgl-project/sglang/pull/9895
- 状态/时间：`merged`，created 2025-09-02, merged 2025-09-03；作者 `WangJianQ-0118`。
- 代码 diff 已读范围：`1` 个文件，`+3/-3`；代码面：docs/config；关键词：kv。
- 代码 diff 细节：
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +3/-3 (6 lines); hunk: {%- for tool in message['tool_calls'] %}
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `examples/chat_template/tool_chat_template_deepseekv31.jinja`；patch 关键词为 kv。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `examples/chat_template/tool_chat_template_deepseekv31.jinja` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10550 - Use jsonschema to constrain required or specific tool choice

- 链接：https://github.com/sgl-project/sglang/pull/10550
- 状态/时间：`merged`，created 2025-09-17, merged 2025-09-27；作者 `TJ5`。
- 代码 diff 已读范围：`12` 个文件，`+1558/-50`；代码面：tests/benchmarks；关键词：spec, test, config, kv, moe。
- 代码 diff 细节：
  - `test/srt/function_call/test_json_schema_constraint.py` added +618/-0 (618 lines); hunk: +"""; 符号: TestJsonSchemaConstraint, setUp, test_required_tool_choice_schema, test_specific_tool_choice_schema
  - `test/srt/openai_server/function_call/test_tool_choice.py` modified +319/-14 (333 lines); hunk: def test_tool_choice_specific_function_streaming(self):; def test_multi_tool_scenario_required(self):; 符号: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming, test_multi_tool_scenario_auto
  - `test/srt/test_function_call_parser.py` modified +319/-0 (319 lines); hunk: from sglang.srt.entrypoints.openai.protocol import Function, Tool; def test_partial_tool_call(self):; 符号: test_partial_tool_call, TestJsonArrayParser, setUp, test_json_detector_ebnf
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +115/-22 (137 lines); hunk: from fastapi import Request; LogProbs,; 符号: _validate_request, _process_messages, if, _build_sampling_params
  - `python/sglang/srt/function_call/utils.py` modified +96/-5 (101 lines); hunk: import json; def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:; 符号: _find_common_prefix, _partial_json_loads, _is_complete_json, _get_tool_schema_defs
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py`；patch 关键词为 spec, test, config, kv, moe。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10875 - feat(reasoning): improve enable thinking from request

- 链接：https://github.com/sgl-project/sglang/pull/10875
- 状态/时间：`merged`，created 2025-09-24, merged 2025-09-29；作者 `jimmy-evo`。
- 代码 diff 已读范围：`1` 个文件，`+8/-10`；代码面：misc；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +8/-10 (18 lines); hunk: def __init__(; async def _generate_chat_stream(; 符号: __init__, _request_id_prefix, _generate_chat_stream, _build_chat_response
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/entrypoints/openai/serving_chat.py`；patch 关键词为 n/a。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/entrypoints/openai/serving_chat.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11189 - Add --thinking-mode to run_eval

- 链接：https://github.com/sgl-project/sglang/pull/11189
- 状态/时间：`merged`，created 2025-10-03, merged 2025-10-03；作者 `hlu1`。
- 代码 diff 已读范围：`2` 个文件，`+29/-1`；代码面：tests/benchmarks；关键词：test。
- 代码 diff 细节：
  - `python/sglang/test/run_eval.py` modified +25/-0 (25 lines); hunk: ); def run_eval(args):; 符号: get_thinking_kwargs, run_eval_once, run_eval, run_eval
  - `python/sglang/test/simple_eval_common.py` modified +4/-1 (5 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, _handle_image, __call__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py`；patch 关键词为 test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11223 - Update tool parser and related documentation

- 链接：https://github.com/sgl-project/sglang/pull/11223
- 状态/时间：`merged`，created 2025-10-05, merged 2025-10-07；作者 `JustinTong0323`。
- 代码 diff 已读范围：`3` 个文件，`+24/-12`；代码面：docs/config；关键词：kv, doc, moe, spec。
- 代码 diff 细节：
  - `docs/advanced_features/tool_parser.ipynb` modified +9/-5 (14 lines); hunk: "
",
  - `python/sglang/srt/function_call/function_call_parser.py` modified +8/-6 (14 lines); hunk: class FunctionCallParser:; 符号: FunctionCallParser:, __init__
  - `python/sglang/srt/server_args.py` modified +7/-1 (8 lines); hunk: def __post_init__(self):; 符号: __post_init__, _handle_deprecated_args, _handle_missing_default_values
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/advanced_features/tool_parser.ipynb`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`；patch 关键词为 kv, doc, moe, spec。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/advanced_features/tool_parser.ipynb`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11589 - [Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector

- 链接：https://github.com/sgl-project/sglang/pull/11589
- 状态/时间：`merged`，created 2025-10-14, merged 2025-11-14；作者 `Muqi1029`。
- 代码 diff 已读范围：`1` 个文件，`+4/-9`；代码面：misc；关键词：kv。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +4/-9 (13 lines); hunk: def parse_streaming_increment(; def parse_streaming_increment(; 符号: parse_streaming_increment, parse_streaming_increment
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/deepseekv31_detector.py`；patch 关键词为 kv。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/deepseekv31_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12123 - Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)

- 链接：https://github.com/sgl-project/sglang/pull/12123
- 状态/时间：`merged`，created 2025-10-25, merged 2025-10-30；作者 `Kangyan-Zhou`。
- 代码 diff 已读范围：`4` 个文件，`+331/-9`；代码面：tests/benchmarks, docs/config；关键词：kv, test。
- 代码 diff 细节：
  - `test/srt/test_deepseek_chat_templates.py` added +319/-0 (319 lines); hunk: +"""; 符号: tool, TestDeepSeekChatTemplateToolCalls, setUpClass, _render_template
  - `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +4/-3 (7 lines); hunk: {%- set ns.is_tool = false -%}
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +4/-3 (7 lines); hunk: {%- set ns.is_first = false %}
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +4/-3 (7 lines); hunk: {%- set ns.is_first = false %}
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`；patch 关键词为 kv, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13190 - Remove enable_dp_attention in deepseek nightly tests

- 链接：https://github.com/sgl-project/sglang/pull/13190
- 状态/时间：`merged`，created 2025-11-13, merged 2025-11-13；作者 `Kangyan-Zhou`。
- 代码 diff 已读范围：`2` 个文件，`+0/-5`；代码面：tests/benchmarks；关键词：attention, eagle, spec, test。
- 代码 diff 细节：
  - `test/srt/nightly/test_deepseek_v32_perf.py` modified +0/-3 (3 lines); hunk: def setUpClass(cls):; def setUpClass(cls):; 符号: setUpClass, setUpClass, setUpClass
  - `test/srt/nightly/test_deepseek_v31_perf.py` modified +0/-2 (2 lines); hunk: def setUpClass(cls):; def setUpClass(cls):; 符号: setUpClass, setUpClass
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/nightly/test_deepseek_v32_perf.py`, `test/srt/nightly/test_deepseek_v31_perf.py`；patch 关键词为 attention, eagle, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/nightly/test_deepseek_v32_perf.py`, `test/srt/nightly/test_deepseek_v31_perf.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13394 - Fix DeepSeekV31's structural tag trigger

- 链接：https://github.com/sgl-project/sglang/pull/13394
- 状态/时间：`merged`，created 2025-11-17, merged 2025-12-31；作者 `Muqi1029`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：misc；关键词：kv。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +1/-1 (2 lines); hunk: def structure_info(self) -> _GetInfoFunc:; 符号: structure_info
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/deepseekv31_detector.py`；patch 关键词为 kv。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/deepseekv31_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13954 - Fix Deepseek v3.1 loading issue

- 链接：https://github.com/sgl-project/sglang/pull/13954
- 状态/时间：`merged`，created 2025-11-26, merged 2025-11-26；作者 `Kangyan-Zhou`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：model wrapper；关键词：config, kv, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunk: def post_load_weights(self, is_nextn=False, weight_names=None):; 符号: post_load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 config, kv, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14837 - [Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210)

- 链接：https://github.com/sgl-project/sglang/pull/14837
- 状态/时间：`merged`，created 2025-12-10, merged 2025-12-10；作者 `zhyncs`。
- 代码 diff 已读范围：`1` 个文件，`+5/-1`；代码面：docs/config；关键词：kv。
- 代码 diff 细节：
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +5/-1 (6 lines); hunk: {% if tools is defined and tools is not none %}
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `examples/chat_template/tool_chat_template_deepseekv31.jinja`；patch 关键词为 kv。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `examples/chat_template/tool_chat_template_deepseekv31.jinja` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16660 - [CI] Enable dpsk v31 test on nightly H200

- 链接：https://github.com/sgl-project/sglang/pull/16660
- 状态/时间：`merged`，created 2026-01-07, merged 2026-01-07；作者 `Fridge003`。
- 代码 diff 已读范围：`1` 个文件，`+1/-2`；代码面：model wrapper, tests/benchmarks；关键词：cuda, kv, test。
- 代码 diff 细节：
  - `test/registered/8-gpu-models/test_deepseek_v31.py` modified +1/-2 (3 lines); hunk: from sglang.test.ci.ci_register import register_cuda_ci; 符号: TestDeepseekV31Unified, for
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/8-gpu-models/test_deepseek_v31.py`；patch 关键词为 cuda, kv, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/8-gpu-models/test_deepseek_v31.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17133 - [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab

- 链接：https://github.com/sgl-project/sglang/pull/17133
- 状态/时间：`merged`，created 2026-01-15, merged 2026-01-16；作者 `xu-yfei`。
- 代码 diff 已读范围：`6` 个文件，`+959/-217`；代码面：MoE/router, quantization, kernel, tests/benchmarks, docs/config；关键词：moe, triton, config, fp8, benchmark, cache, cuda, expert, quant, router。
- 代码 diff 细节：
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +337/-215 (552 lines); hunk: # Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py; sort_config,; 符号: MoeInputs:, KernelWrapper:, __init__, cuda_graph_wrapper
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`；patch 关键词为 moe, triton, config, fp8, benchmark, cache。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17141 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- 链接：https://github.com/sgl-project/sglang/pull/17141
- 状态/时间：`closed`，created 2026-01-15, closed 2026-01-22；作者 `IrisMagicBox`。
- 代码 diff 已读范围：`2` 个文件，`+14/-16`；代码面：docs/config；关键词：kv。
- 代码 diff 细节：
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunk: {% for tool in tools %}; {%- set ns.is_last_user = false -%}
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunk: def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult; 符号: detect_and_parse
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py`；patch 关键词为 kv。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17178 - Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py

- 链接：https://github.com/sgl-project/sglang/pull/17178
- 状态/时间：`merged`，created 2026-01-16, merged 2026-01-16；作者 `hlu1`。
- 代码 diff 已读范围：`1` 个文件，`+3/-2`；代码面：tests/benchmarks；关键词：spec, test。
- 代码 diff 细节：
  - `python/sglang/test/run_eval.py` modified +3/-2 (5 lines); hunk: def get_thinking_kwargs(args):; def run_eval(args):; 符号: get_thinking_kwargs, run_eval, run_eval
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/test/run_eval.py`；patch 关键词为 spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/test/run_eval.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17320 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- 链接：https://github.com/sgl-project/sglang/pull/17320
- 状态/时间：`closed`，created 2026-01-19, closed 2026-01-19；作者 `IrisMagicBox`。
- 代码 diff 已读范围：`2` 个文件，`+14/-16`；代码面：docs/config；关键词：kv。
- 代码 diff 细节：
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunk: {% for tool in tools %}; {%- set ns.is_last_user = false -%}
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunk: def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult; 符号: detect_and_parse
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py`；patch 关键词为 kv。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17558 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- 链接：https://github.com/sgl-project/sglang/pull/17558
- 状态/时间：`closed`，created 2026-01-22, closed 2026-01-24；作者 `IrisMagicBox`。
- 代码 diff 已读范围：`2` 个文件，`+14/-16`；代码面：docs/config；关键词：kv。
- 代码 diff 细节：
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunk: {% for tool in tools %}; {%- set ns.is_last_user = false -%}
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunk: def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult; 符号: detect_and_parse
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py`；patch 关键词为 kv。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17761 - fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates

- 链接：https://github.com/sgl-project/sglang/pull/17761
- 状态/时间：`open`，created 2026-01-26；作者 `ieBoytsov`。
- 代码 diff 已读范围：`3` 个文件，`+79/-2`；代码面：tests/benchmarks, docs/config；关键词：kv, test。
- 代码 diff 细节：
  - `test/manual/test_deepseek_chat_templates.py` modified +77/-0 (77 lines); hunk: def test_tool_call_with_content(self):; 符号: test_tool_call_with_content, test_assistant_marker_after_tool_output
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +1/-1 (2 lines); hunk: {{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +1/-1 (2 lines); hunk: {{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`；patch 关键词为 kv, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18236 - Fix function call arguments missing in streaming mode for DeepSeek V3.1

- 链接：https://github.com/sgl-project/sglang/pull/18236
- 状态/时间：`open`，created 2026-02-04；作者 `momaek`。
- 代码 diff 已读范围：`1` 个文件，`+21/-3`；代码面：misc；关键词：kv。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +21/-3 (24 lines); hunk: def __init__(self):; def parse_streaming_increment(; 符号: __init__, has_tool_call, parse_streaming_increment, parse_streaming_increment
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/deepseekv31_detector.py`；patch 关键词为 kv。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/deepseekv31_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21593 - Fix tool call constrained decoding and parsing for models with native formats

- 链接：https://github.com/sgl-project/sglang/pull/21593
- 状态/时间：`merged`，created 2026-03-28, merged 2026-04-11；作者 `JustinTong0323`。
- 代码 diff 已读范围：`9` 个文件，`+306/-61`；代码面：tests/benchmarks；关键词：spec, test, kv, config。
- 代码 diff 细节：
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +113/-0 (113 lines); hunk: def test_streaming_function_call_marker_json_split_at_quotes(self):; 符号: test_streaming_function_call_marker_json_split_at_quotes, TestGetStructureConstraint, _make_tools, _make_parser
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +67/-43 (110 lines); hunk: def _process_messages(; def _process_tool_calls(; 符号: _process_messages, _process_tool_calls, _process_tool_calls, _process_tool_call_stream
  - `test/registered/openai_server/basic/test_serving_chat.py` modified +72/-0 (72 lines); hunk: def test_extract_routed_dp_rank_from_header_invalid(self):; 符号: test_extract_routed_dp_rank_from_header_invalid, TestProcessToolCallsWithRequiredToolChoice, setUp, test_required_with_parser_uses_function_call_parser
  - `python/sglang/srt/function_call/function_call_parser.py` modified +35/-11 (46 lines); hunk: from sglang.srt.entrypoints.openai.protocol import (; from sglang.srt.function_call.qwen25_detector import Qwen25Detector; 符号: parse_stream_chunk, get_structure_tag, get_structure_tag, get_structure_tag
  - `test/registered/openai_server/function_call/test_tool_choice.py` modified +8/-2 (10 lines); hunk: def test_tool_choice_specific_function_streaming(self):; def test_required_streaming_arguments_chunks_json(self):; 符号: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py`；patch 关键词为 spec, test, kv, config。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21599 - [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1

- 链接：https://github.com/sgl-project/sglang/pull/21599
- 状态/时间：`merged`，created 2026-03-28, merged 2026-04-20；作者 `alphabetc1`。
- 代码 diff 已读范围：`13` 个文件，`+1296/-33`；代码面：kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：spec, config, cuda, eagle, attention, topk, cache, kv, quant, test。
- 代码 diff 细节：
  - `benchmark/bench_adaptive_speculative.py` added +263/-0 (263 lines); hunk: +"""Benchmark adaptive speculative decoding against static baselines.; 符号: build_phase_plan, send_request, run_phase, summarize_phases
  - `test/registered/unit/spec/test_adaptive_spec_params.py` added +195/-0 (195 lines); hunk: +import unittest; 符号: TestAdaptiveSpeculativeParams, test_initial_steps_snap_to_nearest_candidate_preferring_larger_step, test_update_respects_warmup_and_interval, test_empty_batches_do_not_consume_warmup_or_shift_steps
  - `test/registered/spec/eagle/test_adaptive_speculative.py` added +170/-0 (170 lines); hunk: +import json; 符号: TestAdaptiveSpeculativeServer, setUpClass, tearDownClass, _get_internal_state
  - `python/sglang/srt/speculative/eagle_worker.py` modified +162/-4 (166 lines); hunk: import logging; alloc_token_slots,; 符号: __init__, __init__, init_cuda_graphs, apply_runtime_state
  - `docs/advanced_features/adaptive_speculative_decoding.md` added +156/-0 (156 lines); hunk: +# Adaptive Speculative Decoding
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`；patch 关键词为 spec, config, cuda, eagle, attention, topk。影响判断：CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21739 - [NPU] Update DeepSeek-V3.1 and DeepSeek-V3.2 model deployment instructions in documentation

- 链接：https://github.com/sgl-project/sglang/pull/21739
- 状态/时间：`open`，created 2026-03-31；作者 `MichelleWu351`。
- 代码 diff 已读范围：`1` 个文件，`+163/-19`；代码面：docs/config；关键词：attention, benchmark, cache, config, cuda, deepep, doc, eagle, expert, lora。
- 代码 diff 细节：
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +163/-19 (182 lines); hunk: you encounter issues or have any questions, please [open an issue](https://githu; We tested it based on the `RANDOM` dataset.
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/platforms/ascend/ascend_npu_best_practice.md`；patch 关键词为 attention, benchmark, cache, config, cuda, deepep。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/platforms/ascend/ascend_npu_best_practice.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22128 - Allow piecewise CUDA graph with speculative decoding

- 链接：https://github.com/sgl-project/sglang/pull/22128
- 状态/时间：`merged`，created 2026-04-05, merged 2026-04-17；作者 `narutolhy`。
- 代码 diff 已读范围：`4` 个文件，`+272/-18`；代码面：kernel, scheduler/runtime, tests/benchmarks；关键词：cuda, spec, quant, attention, cache, config, eagle, expert, fp8, lora。
- 代码 diff 细节：
  - `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` added +243/-0 (243 lines); hunk: +"""Test piecewise CUDA graph coexisting with speculative decoding.; 符号: TestPCGWithMTP, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/server_args.py` modified +15/-18 (33 lines); hunk: def _handle_piecewise_cuda_graph(self):; 符号: _handle_piecewise_cuda_graph
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +10/-0 (10 lines); hunk: def can_run(self, forward_batch: ForwardBatch):; 符号: can_run
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunk: def init_piecewise_cuda_graphs(self):; 符号: init_piecewise_cuda_graphs
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`；patch 关键词为 cuda, spec, quant, attention, cache, config。影响判断：CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22433 - [Test] Add unit tests for DeepSeekV31Detector

- 链接：https://github.com/sgl-project/sglang/pull/22433
- 状态/时间：`open`，created 2026-04-09；作者 `vedantagarwal-web`。
- 代码 diff 已读范围：`1` 个文件，`+314/-0`；代码面：tests/benchmarks；关键词：kv, test。
- 代码 diff 细节：
  - `test/registered/unit/function_call/test_deepseekv31_detector.py` added +314/-0 (314 lines); hunk: +"""Unit tests for DeepSeekV31Detector — no server, no model loading.; 符号: _wrap_single, _make_tools, TestDeepSeekV31DetectorHasToolCall, setUp
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/function_call/test_deepseekv31_detector.py`；patch 关键词为 kv, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/function_call/test_deepseekv31_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22950 - [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)

- 链接：https://github.com/sgl-project/sglang/pull/22950
- 状态/时间：`closed`，created 2026-04-16, closed 2026-04-21；作者 `Wen-xuan-Xu`。
- 代码 diff 已读范围：`11` 个文件，`+597/-64`；代码面：scheduler/runtime, tests/benchmarks, docs/config；关键词：cache, kv, cuda, eagle, test, attention, spec。
- 代码 diff 细节：
  - `test/registered/unit/mem_cache/test_radix_cache_thinking.py` added +238/-0 (238 lines); hunk: +import unittest; 符号: _MockReqToTokenPool:, __init__, write, _MockAllocator:
  - `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py` added +220/-0 (220 lines); hunk: +import unittest; 符号: _MockReqToTokenPool:, __init__, write, _MockAllocator:
  - `python/sglang/srt/mem_cache/mamba_radix_cache.py` modified +62/-50 (112 lines); hunk: from numpy import float64; MatchPrefixParams,; 符号: cache_finished_req, _skip_cache_unfinished_req, _skip_cache_unfinished_req
  - `python/sglang/srt/mem_cache/radix_cache_cpp.py` modified +27/-14 (41 lines); hunk: MatchPrefixParams,; def cache_finished_req(self, req: Req, is_insert: bool = True):; 符号: cache_finished_req, cache_unfinished_req, cache_unfinished_req, pretty_print
  - `python/sglang/srt/mem_cache/common.py` modified +22/-0 (22 lines); hunk: def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:; 符号: alloc_for_decode, maybe_strip_thinking_tokens, release_kv_cache
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py`；patch 关键词为 cache, kv, cuda, eagle, test, attention。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22981 - [Test] Add unit tests for 7 missing function call detectors

- 链接：https://github.com/sgl-project/sglang/pull/22981
- 状态/时间：`open`，created 2026-04-16；作者 `JustinTong0323`。
- 代码 diff 已读范围：`3` 个文件，`+1017/-1`；代码面：tests/benchmarks；关键词：test, doc, kv, moe。
- 代码 diff 细节：
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +960/-1 (961 lines); hunk: import unittest; from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector; 符号: TestPythonicDetector, setUp, test_has_tool_call, test_detect_and_parse_single
  - `test/registered/openai_server/function_call/test_tool_choice.py` modified +57/-0 (57 lines); hunk: def setUpClass(cls):; 符号: setUpClass, TestToolChoiceWithConstrainedDecoding, setUpClass, test_tool_choice_required_strict_finish_reason
  - `test/registered/unit/function_call/test_kimik2_detector.py` renamed +0/-0 (0 lines)
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_kimik2_detector.py`；patch 关键词为 test, doc, kv, moe。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_kimik2_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23315 - Opt-in strip of thinking tokens from radix cache

- 链接：https://github.com/sgl-project/sglang/pull/23315
- 状态/时间：`merged`，created 2026-04-21, merged 2026-04-21；作者 `hnyls2002`。
- 代码 diff 已读范围：`4` 个文件，`+72/-4`；代码面：scheduler/runtime, tests/benchmarks；关键词：cache, kv, spec, cuda, scheduler, test。
- 代码 diff 细节：
  - `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` modified +52/-1 (53 lines); hunk: from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType; def test_cache_finished_req_insert(self):; 符号: test_cache_finished_req_insert, test_cache_finished_req_strips_thinking, test_cache_finished_req_no_insert
  - `python/sglang/srt/managers/schedule_batch.py` modified +9/-2 (11 lines); hunk: def output_ids_through_stop(self) -> List[int]:; def pop_overallocated_kv_cache(self) -> Tuple[int, int]:; 符号: output_ids_through_stop, _cache_commit_len, pop_committed_kv_cache, pop_overallocated_kv_cache
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunk: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; 符号: ServerArgs:, add_cli_args
  - `python/sglang/srt/mem_cache/common.py` modified +3/-1 (4 lines); hunk: def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = Tr; 符号: release_kv_cache
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`；patch 关键词为 cache, kv, spec, cuda, scheduler, test。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23336 - [SPEC V2][2/N] feat: adaptive spec support spec v2

- 链接：https://github.com/sgl-project/sglang/pull/23336
- 状态/时间：`open`，created 2026-04-21；作者 `alphabetc1`。
- 代码 diff 已读范围：`6` 个文件，`+193/-10`；代码面：multimodal/processor, scheduler/runtime；关键词：spec, eagle, cuda, scheduler, attention, processor, config, kv, moe, topk。
- 代码 diff 细节：
  - `python/sglang/srt/speculative/eagle_worker_v2.py` modified +173/-0 (173 lines); hunk: from sglang.srt.managers.schedule_batch import ModelWorkerBatch; def __init__(; 符号: __init__, __init__, target_worker, forward_batch_generation
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +8/-4 (12 lines); hunk: def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):; def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):; 符号: prepare_for_decode, prepare_for_decode, prepare_for_v2_draft
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +10/-1 (11 lines); hunk: def _resolve_spec_overlap_token_ids(; 符号: _resolve_spec_overlap_token_ids
  - `python/sglang/srt/speculative/adaptive_spec_params.py` modified +0/-5 (5 lines); hunk: def adaptive_unsupported_reason(server_args: ServerArgs) -> str \| None:; 符号: adaptive_unsupported_reason
  - `python/sglang/srt/managers/utils.py` modified +1/-0 (1 lines); hunk: class GenerationBatchResult:; 符号: GenerationBatchResult:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`；patch 关键词为 spec, eagle, cuda, scheduler, attention, processor。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：32；open PR 数：6。
- 仍需跟进的 open PR：[#17761](https://github.com/sgl-project/sglang/pull/17761), [#18236](https://github.com/sgl-project/sglang/pull/18236), [#21739](https://github.com/sgl-project/sglang/pull/21739), [#22433](https://github.com/sgl-project/sglang/pull/22433), [#22981](https://github.com/sgl-project/sglang/pull/22981), [#23336](https://github.com/sgl-project/sglang/pull/23336)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
