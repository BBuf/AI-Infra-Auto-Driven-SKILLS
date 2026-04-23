# SGLang DeepSeek V3.1 Support and Optimization Timeline

This document is based on SGLang `origin/main` snapshot `929e00eea`, sgl-cookbook `origin/main` snapshot `8ec4d03`, and patch-level reading of DeepSeek V3.1 merged and open PRs. The scope only covers the independent DeepSeek V3.1 / DeepSeek-V3.1-Terminus differences: tool calling, thinking mode, chat template, streaming parser, structural tags, loading fixes, MTP validation, and MoE configs. The DeepSeek V3/R1 MLA, MoE, quantization, and DeepEP mainline is not repeated here, and DeepSeek V3.2 DSA/NSA sparse attention is documented separately.

Conclusion: as of `929e00eea`, DeepSeek V3.1 still reuses `DeepseekV3ForCausalLM` and the `deepseek_v2.py` main model path. The independent pieces are the `deepseekv31` tool parser, `tool_chat_template_deepseekv31.jinja`, and the `thinking` chat-template parameter. V3.1's tool-call format differs from V3: it does not contain the `function` literal and does not use fenced JSON. Current main already has basic tool calling, thinking parser support, dict/string argument handling, structural trigger fix, and H200 TP8/MTP validation. Additional runtime items include thinking-token radix-cache stripping and inherited V3/R1 adaptive EAGLE, PCG plus speculative decoding, and spec-v2 adaptive spec. The open items are streaming argument loss, missing Assistant token after tool output, NPU deployment docs, parser CPU tests, and spec-v2 adaptive speculative decoding.

## 1. Timeline Overview

| Created | PR | State | Track | Code Area | Effect |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-21 | [#9446](https://github.com/sgl-project/sglang/pull/9446) | merged | tool calling | `deepseekv31_detector.py`, V3.1 template, parser registration | Added the DeepSeek V3.1 tool-call parser and chat template. |
| 2025-08-21 | [#9464](https://github.com/sgl-project/sglang/pull/9464) | merged | thinking parser | `serving_chat.py`, reasoning parser, docs | Added DeepSeek V3.1 thinking-mode support with `--reasoning-parser deepseek-v3`. |
| 2025-08-23 | [#9544](https://github.com/sgl-project/sglang/pull/9544) | merged | docs | benchmark / basic usage docs | Added DeepSeek V3.1 support documentation. |
| 2025-10-25 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | chat template | V3/V3.1/V3.2 templates, template test | Fixed double escaping when tool arguments are dicts or strings. |
| 2025-11-13 | [#13190](https://github.com/sgl-project/sglang/pull/13190) | merged | nightly test | V3.1/V3.2 perf tests | Removed stale `enable_dp_attention`. |
| 2025-11-17 | [#13394](https://github.com/sgl-project/sglang/pull/13394) | merged | structural tag | `DeepSeekV31Detector.structure_info` | Changed the structural trigger to generic `<｜tool▁call▁begin｜>`. |
| 2025-11-26 | [#13954](https://github.com/sgl-project/sglang/pull/13954) | merged | loading | `deepseek_v2.py` | Fixed a DeepSeek V3.1 loading issue. |
| 2026-01-07 | [#16660](https://github.com/sgl-project/sglang/pull/16660) | merged | CI | `test/registered/8-gpu-models/test_deepseek_v31.py` | Enabled DeepSeek V3.1 H200 nightly testing. |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | MoE tuning | fused MoE Triton configs | Added H20/H20-3E FP8 MoE tuning configs for V3.1/V3.2 shapes. |
| 2026-01-26 | [#17761](https://github.com/sgl-project/sglang/pull/17761) | open | chat template | V3.1/V3.2 templates | Fixes missing Assistant token after tool output. |
| 2026-02-04 | [#18236](https://github.com/sgl-project/sglang/pull/18236) | open | streaming parser | `deepseekv31_detector.py` | Fixes missing first-chunk arguments and dropped normal text in streaming. |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | MTP/spec | EAGLE runtime, spec workers | Added adaptive `speculative_num_steps` for EAGLE top-k=1, inherited by V3.1 MTP. |
| 2026-03-31 | [#21739](https://github.com/sgl-project/sglang/pull/21739) | open | NPU docs | Ascend best-practice docs | Updates V3.1/V3.2 NPU deployment instructions. |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | PCG/spec | model runner, PCG runner | Allowed piecewise CUDA graph to run with speculative decoding. |
| 2026-04-09 | [#22433](https://github.com/sgl-project/sglang/pull/22433) | open | parser tests | `test_deepseekv31_detector.py` | Adds CPU unit tests for `DeepSeekV31Detector`. |
| 2026-04-16 | [#22981](https://github.com/sgl-project/sglang/pull/22981) | open | parser tests | function-call detectors | Adds CPU tests for several missing function-call detectors. |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | reasoning cache | model config, scheduler, radix cache, reasoning parser | Explored parser-gated two-phase reasoning radix-cache stripping and closed. |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | reasoning cache | `schedule_batch.py`, `mem_cache/common.py`, `server_args.py` | Added opt-in thinking-token stripping from radix cache. |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | spec v2 | scheduler output processor, EAGLE v2 workers | Extends adaptive speculative decoding to spec v2. |

## 1.1 Parser/Template Adjacent PRs

The V3.1 parser/template coverage also includes these adjacent PRs:

- `#9468`: updated reasoning-parser docs after `#9464` landed thinking-parser support.
- `#9895` and `#14837`: updated `tool_chat_template_deepseekv31.jinja`; `#14837` is the 2025-12-10 auto-sync.
- `#10550`, `#11223`, `#11589`, and `#21593`: tool-choice, tool-parser docs, `tool_choice="auto"` argument handling, and native-format constrained decoding/parser fixes that affect V3.1 tool serving.
- `#10875`, `#11189`, and `#17178`: request-level thinking enablement, eval `--thinking-mode`, and removal of `deepseek-r1` from thinking-mode choices, which clarify the V3.1 thinking vs R1 parser boundary.
- `#17141`, `#17320`, and `#17558`: closed attempts around `finish_reason="stop"` / Assistant-token behavior after tool content; open `#17761` remains the current tracker.
- `#22950` and `#23315`: distinguish the closed reasoning-cache strip exploration from the current merged thinking-token radix-cache strip.
- `#21599`, `#22128`, and `#23336`: speculative-decoding infrastructure inherited by V3.1 MTP, covering adaptive EAGLE, PCG plus speculative decoding, and spec-v2 adaptive spec.

## 2. Why V3.1 Needs Its Own Document

DeepSeek V3.1 still uses the shared DeepSeek V3/R1 compute backbone: `DeepseekV3ForCausalLM`, `DeepseekV2AttentionMLA`, `DeepseekV2MoE`, the shared DeepSeek weight loader, NextN/MTP, and server-side backend selection. If the issue is MLA backend, FP8/FP4/W4AFP8, shared-expert fusion, DeepEP, LoRA, MTP draft loading, or DP attention, use the DeepSeek V3/R1 document.

However, V3.1 changes the user-visible serving behavior:

- hybrid thinking: the same model can switch between thinking and non-thinking through the `thinking` parameter.
- changed tool-call format: the function name comes immediately after `<｜tool▁call▁begin｜>`, with `<｜tool▁sep｜>` separating JSON arguments.
- the chat template must handle system prompts, tools, assistant prefixes, tool outputs, thinking markers, and multi-turn tool calls.
- the streaming parser is custom and is not equivalent to non-streaming parsing.
- constrained decoding requires the correct `structure_info.trigger`.

The main V3.1 value is not a new MLA kernel. It is the serving layer that turns the DeepSeek V3 backbone into an OpenAI-compatible agent and hybrid-reasoning model.

## 3. `#9446`: V3.1 Tool-Call Parser and Template

`#9446` added `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `python/sglang/srt/function_call/deepseekv31_detector.py`, and the `deepseekv31` parser registration.

The V3.1 tool-call format is:

```text
<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>{tool_name}<｜tool▁sep｜>{json_arguments}<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>
```

The differences from V3 are important:

- V3.1 does not have the `function` literal in `<｜tool▁call▁begin｜>function<｜tool▁sep｜>`.
- V3.1 arguments are direct JSON strings, not fenced in a Markdown `json` block.
- Multiple tool calls are concatenated directly without extra separators.

`DeepSeekV31Detector.detect_and_parse` first finds the outer `<｜tool▁calls▁begin｜>`, then uses `func_call_regex` to collect each `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>`, and finally uses `func_detail_regex` to split the function name and arguments. Arguments are parsed with `json.loads` and then passed to `parse_base_json` to match the OpenAI tool schema.

This stage is about format correctness, not throughput. If V3 and V3.1 parsers are swapped, symptoms include parse failure, wrong function names, or arguments being returned as plain text.

## 4. `#9464`: Thinking Mode Is Not the R1 Parser

`#9464` added DeepSeek V3.1 thinking-parser support and docs. It established that V3.1 uses:

```shell
--reasoning-parser deepseek-v3
```

and enables thinking in requests with:

```json
{"chat_template_kwargs": {"thinking": true}}
```

This differs from R1. R1 uses the `deepseek-r1` parser, and that parser handles reasoning without an opening `<think>` tag. V3.1 is closer to Qwen3-style hybrid thinking: the chat template decides whether to inject `<think>` or `</think>`.

In current `tool_chat_template_deepseekv31.jinja`, after the last user message, `add_generation_prompt` emits `<｜Assistant｜>` and then appends `<think>` or `</think>` based on `thinking`. If a previous assistant message contains `</think>`, the template strips the reasoning portion and preserves only content. This keeps historical messages and the next generation prompt in the same format.

DeepSeek-V3.1-Speciale is an important exception. Current cookbook docs state that Speciale does not support tool calling; it should be treated as a deep-reasoning model, not a V3.1 tool-use target.

Thinking mode also intersects with radix cache. `#22950` is the closed parser-gated reasoning-cache strip design; current main should be read from `#23315`, which adds an opt-in strip in `server_args.py`, `schedule_batch.py`, and `mem_cache/common.py` so thinking tokens can be removed from radix-cache entries. For V3.1, this is not a `deepseekv31` tool-parser change; it is cache-layer behavior that decides whether `<think>` / `</think>` can be reused through prefix cache.

## 5. `#12123`: Avoid Double-Escaping Dict/String Arguments

In multi-turn tool calling, `tool["function"]["arguments"]` can be a dict or an already serialized JSON string. If the template always applies `tojson`, an already serialized string becomes an escaped JSON string.

`#12123` made the same fix across the V3, V3.1, and V3.2 DeepSeek templates:

```jinja
{% set formatted_args = tool['function']['arguments'] if tool['function']['arguments'] is string else tool['function']['arguments']|tojson %}
```

It also added `test_deepseek_chat_templates.py`, covering:

- dict arguments must be JSON-encoded normally.
- string arguments must be used as-is.
- mixed dict and string arguments across multiple tool calls must not double-escape.

This does not affect model throughput, but it directly affects agent multi-turn tool use because malformed historical tool calls poison the next prompt.

## 6. `#13394`, `#18236`, and `#22433`: Structural Tags and Streaming Parser

`#13394` fixed the constrained-decoding trigger. Previously `structure_info.trigger` included the concrete function name and `<｜tool▁sep｜>`, which meant structural constraints could only trigger after the function name was known. The fix made the trigger generic:

```python
trigger="<｜tool▁call▁begin｜>"
```

`begin` remains name-specific:

```python
begin="<｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>"
```

`#18236` is still open, but it documents two current streaming-parser risks. First, when the function name and JSON arguments arrive in the same chunk, current code may emit only the name and skip the first argument diff. Second, when plain text appears before the tool marker, streaming can return empty normal text even though non-streaming preserves the prefix. The PR adds `_normal_text_sent`, extracts normal text before the first `<｜tool▁call▁begin｜>`, and processes argument diffs whenever `func_args_raw` is non-empty.

`#22433` is also open, but it defines the CPU tests that should remain long term: `has_tool_call`, no-tool plain text, single tool, multiple tools, invalid JSON fallback, unknown tool, unicode arguments, streaming chunks, tool index, `structure_info`, and structural tag support. Future V3.1 parser changes should start with this kind of CPU validation before running 8-GPU model tests.

## 7. Loading, MTP, and MoE Configs

`#13954` fixed a DeepSeek V3.1 loading issue in `deepseek_v2.py`. This reinforces that V3.1's compute path is shared DeepSeek infrastructure, while parser/template is the independent OpenAI-serving surface. If V3.1 launch fails during weight loading, MLA initialization, or MoE parameter mapping, do not focus only on `deepseekv31_detector.py`.

`#16660` added DeepSeek V3.1 to H200 nightly testing with:

- TP8 base.
- TP8 plus EAGLE MTP with `SGLANG_ENABLE_SPEC_V2=1`.
- GSM8K accuracy baseline `0.935`.
- performance profiles written to `performance_profiles_deepseek_v31`.

`#13190` removed stale `enable_dp_attention` from V3.1/V3.2 nightly perf tests so benchmark shapes match current server args.

Inherited MTP infrastructure must also be checked: `#21599` makes EAGLE top-k=1 draft steps adaptive, `#22128` lets PCG coexist with speculative decoding, and open `#23336` carries adaptive spec into spec-v2 workers. Because the `#16660` V3.1 MTP lane explicitly uses `SGLANG_ENABLE_SPEC_V2=1`, these PRs are not V3.1 parser work, but they affect the real V3.1 TP8+MTP serving shape.

`#17133` is the MoE-config performance line. It added H20 and H20-3E fused MoE configs for DeepSeek-family V3.1/V3.2 shapes. Typical filenames contain `E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]`. The `257` corresponds to 256 routed experts plus one fused shared expert.

## 8. Current Validation Surface and Open PRs

Current validation surface:

- `test/manual/test_deepseek_v31.py`: TP8 and TP8+MTP, GSM8K baseline `0.935`.
- `test/manual/nightly/test_deepseek_v31_perf.py`: V3.1 nightly performance.
- `test/manual/test_deepseek_chat_templates.py`: V3/V3.1/V3.2 template dict/string argument tests.
- open `#22433`'s `test/registered/unit/function_call/test_deepseekv31_detector.py`: recommended parser CPU test baseline.

Open PRs to track:

- `#17761`: missing Assistant token after V3.1/V3.2 tool output.
- `#18236`: missing V3.1 streaming function-call arguments and normal text.
- `#21739`: V3.1/V3.2 NPU deployment docs.
- `#22433`: DeepSeekV31Detector CPU tests.
- `#22981`: CPU tests for several missing function-call detectors.
- `#23336`: spec-v2 adaptive speculative decoding.
