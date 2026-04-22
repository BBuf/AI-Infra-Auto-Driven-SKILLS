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
