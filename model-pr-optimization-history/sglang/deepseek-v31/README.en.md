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

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `DeepSeek V3.1` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
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

### File-level PR diff reading notes

### PR #9446 - Support DeepSeek-V3.1 tool call

- Link: https://github.com/sgl-project/sglang/pull/9446
- Status/date: `merged`, created 2025-08-21, merged 2025-08-27; author `Xu-Wenqing`.
- Diff scope read: `3` files, `+315/-0`; areas: docs/config; keywords: kv, moe, spec.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` added +222/-0 (222 lines); hunks: +import json; symbols: DeepSeekV31Detector, __init__, has_tool_call, detect_and_parse
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: +{% if not add_generation_prompt is defined %}
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunks: from sglang.srt.function_call.base_format_detector import BaseFormatDetector; class FunctionCallParser:; symbols: FunctionCallParser:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `python/sglang/srt/function_call/function_call_parser.py`; keywords observed in patches: kv, moe, spec. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `python/sglang/srt/function_call/function_call_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9464 - Add deepseek v3.1 thinking parser support and update docs

- Link: https://github.com/sgl-project/sglang/pull/9464
- Status/date: `merged`, created 2025-08-21, merged 2025-08-21; author `JustinTong0323`.
- Diff scope read: `3` files, `+136/-78`; areas: docs/config; keywords: config, doc, spec.
- Code diff details:
  - `docs/basic_usage/openai_api_completions.ipynb` modified +123/-69 (192 lines); hunks: "print_highlight(f\"Response: {response}\")"; " print(chunk.choices[0].delta.content, end=\"\")"
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +9/-6 (15 lines); hunks: def _get_enable_thinking_from_request(self, request: ChatCompletionRequest) -> b; symbols: _get_enable_thinking_from_request, _process_tool_call_stream
  - `python/sglang/srt/reasoning_parser.py` modified +4/-3 (7 lines); hunks: class ReasoningParser:; symbols: ReasoningParser:, __init__
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/openai_api_completions.ipynb`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`; keywords observed in patches: config, doc, spec. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/openai_api_completions.ipynb`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9468 - Update reasoning parser doc

- Link: https://github.com/sgl-project/sglang/pull/9468
- Status/date: `merged`, created 2025-08-21, merged 2025-08-22; author `JustinTong0323`.
- Diff scope read: `1` files, `+46/-22`; areas: docs/config; keywords: doc.
- Code diff details:
  - `docs/basic_usage/openai_api_completions.ipynb` modified +46/-22 (68 lines); hunks: "
",; ")
",
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/openai_api_completions.ipynb`; keywords observed in patches: doc. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/openai_api_completions.ipynb`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9544 - [doc] deepseekv31 support

- Link: https://github.com/sgl-project/sglang/pull/9544
- Status/date: `merged`, created 2025-08-23, merged 2025-08-23; author `XiaotongJiang`.
- Diff scope read: `2` files, `+82/-4`; areas: tests/benchmarks, docs/config; keywords: attention, benchmark, config, doc, fp8, mla, spec.
- Code diff details:
  - `benchmark/deepseek_v3/README.md` modified +80/-2 (82 lines); hunks: -# DeepSeek V3 Support; Add performance optimization options (#performance-optimization-options) as nee
  - `docs/basic_usage/deepseek.md` modified +2/-2 (4 lines); hunks: SGLang provides many optimizations specifically designed for the DeepSeek models
- Optimization/support interpretation: The concrete diff surface is `benchmark/deepseek_v3/README.md`, `docs/basic_usage/deepseek.md`; keywords observed in patches: attention, benchmark, config, doc, fp8, mla. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `benchmark/deepseek_v3/README.md`, `docs/basic_usage/deepseek.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9895 - Update tool_chat_template_deepseekv31.jinja

- Link: https://github.com/sgl-project/sglang/pull/9895
- Status/date: `merged`, created 2025-09-02, merged 2025-09-03; author `WangJianQ-0118`.
- Diff scope read: `1` files, `+3/-3`; areas: docs/config; keywords: kv.
- Code diff details:
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +3/-3 (6 lines); hunks: {%- for tool in message['tool_calls'] %}
- Optimization/support interpretation: The concrete diff surface is `examples/chat_template/tool_chat_template_deepseekv31.jinja`; keywords observed in patches: kv. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `examples/chat_template/tool_chat_template_deepseekv31.jinja`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10550 - Use jsonschema to constrain required or specific tool choice

- Link: https://github.com/sgl-project/sglang/pull/10550
- Status/date: `merged`, created 2025-09-17, merged 2025-09-27; author `TJ5`.
- Diff scope read: `12` files, `+1558/-50`; areas: tests/benchmarks; keywords: spec, test, config, kv, moe.
- Code diff details:
  - `test/srt/function_call/test_json_schema_constraint.py` added +618/-0 (618 lines); hunks: +"""; symbols: TestJsonSchemaConstraint, setUp, test_required_tool_choice_schema, test_specific_tool_choice_schema
  - `test/srt/openai_server/function_call/test_tool_choice.py` modified +319/-14 (333 lines); hunks: def test_tool_choice_specific_function_streaming(self):; def test_multi_tool_scenario_required(self):; symbols: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming, test_multi_tool_scenario_auto
  - `test/srt/test_function_call_parser.py` modified +319/-0 (319 lines); hunks: from sglang.srt.entrypoints.openai.protocol import Function, Tool; def test_partial_tool_call(self):; symbols: test_partial_tool_call, TestJsonArrayParser, setUp, test_json_detector_ebnf
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +115/-22 (137 lines); hunks: from fastapi import Request; LogProbs,; symbols: _validate_request, _process_messages, if, _build_sampling_params
  - `python/sglang/srt/function_call/utils.py` modified +96/-5 (101 lines); hunks: import json; def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:; symbols: _find_common_prefix, _partial_json_loads, _is_complete_json, _get_tool_schema_defs
- Optimization/support interpretation: The concrete diff surface is `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py`; keywords observed in patches: spec, test, config, kv, moe. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10875 - feat(reasoning): improve enable thinking from request

- Link: https://github.com/sgl-project/sglang/pull/10875
- Status/date: `merged`, created 2025-09-24, merged 2025-09-29; author `jimmy-evo`.
- Diff scope read: `1` files, `+8/-10`; areas: misc; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +8/-10 (18 lines); hunks: def __init__(; async def _generate_chat_stream(; symbols: __init__, _request_id_prefix, _generate_chat_stream, _build_chat_response
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/entrypoints/openai/serving_chat.py`; keywords observed in patches: n/a. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/entrypoints/openai/serving_chat.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11189 - Add --thinking-mode to run_eval

- Link: https://github.com/sgl-project/sglang/pull/11189
- Status/date: `merged`, created 2025-10-03, merged 2025-10-03; author `hlu1`.
- Diff scope read: `2` files, `+29/-1`; areas: tests/benchmarks; keywords: test.
- Code diff details:
  - `python/sglang/test/run_eval.py` modified +25/-0 (25 lines); hunks: ); def run_eval(args):; symbols: get_thinking_kwargs, run_eval_once, run_eval, run_eval
  - `python/sglang/test/simple_eval_common.py` modified +4/-1 (5 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, _handle_image, __call__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py`; keywords observed in patches: test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11223 - Update tool parser and related documentation

- Link: https://github.com/sgl-project/sglang/pull/11223
- Status/date: `merged`, created 2025-10-05, merged 2025-10-07; author `JustinTong0323`.
- Diff scope read: `3` files, `+24/-12`; areas: docs/config; keywords: kv, doc, moe, spec.
- Code diff details:
  - `docs/advanced_features/tool_parser.ipynb` modified +9/-5 (14 lines); hunks: "
",
  - `python/sglang/srt/function_call/function_call_parser.py` modified +8/-6 (14 lines); hunks: class FunctionCallParser:; symbols: FunctionCallParser:, __init__
  - `python/sglang/srt/server_args.py` modified +7/-1 (8 lines); hunks: def __post_init__(self):; symbols: __post_init__, _handle_deprecated_args, _handle_missing_default_values
- Optimization/support interpretation: The concrete diff surface is `docs/advanced_features/tool_parser.ipynb`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: kv, doc, moe, spec. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/advanced_features/tool_parser.ipynb`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11589 - [Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector

- Link: https://github.com/sgl-project/sglang/pull/11589
- Status/date: `merged`, created 2025-10-14, merged 2025-11-14; author `Muqi1029`.
- Diff scope read: `1` files, `+4/-9`; areas: misc; keywords: kv.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +4/-9 (13 lines); hunks: def parse_streaming_increment(; def parse_streaming_increment(; symbols: parse_streaming_increment, parse_streaming_increment
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/deepseekv31_detector.py`; keywords observed in patches: kv. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/deepseekv31_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12123 - Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)

- Link: https://github.com/sgl-project/sglang/pull/12123
- Status/date: `merged`, created 2025-10-25, merged 2025-10-30; author `Kangyan-Zhou`.
- Diff scope read: `4` files, `+331/-9`; areas: tests/benchmarks, docs/config; keywords: kv, test.
- Code diff details:
  - `test/srt/test_deepseek_chat_templates.py` added +319/-0 (319 lines); hunks: +"""; symbols: tool, TestDeepSeekChatTemplateToolCalls, setUpClass, _render_template
  - `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +4/-3 (7 lines); hunks: {%- set ns.is_tool = false -%}
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +4/-3 (7 lines); hunks: {%- set ns.is_first = false %}
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +4/-3 (7 lines); hunks: {%- set ns.is_first = false %}
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`; keywords observed in patches: kv, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13190 - Remove enable_dp_attention in deepseek nightly tests

- Link: https://github.com/sgl-project/sglang/pull/13190
- Status/date: `merged`, created 2025-11-13, merged 2025-11-13; author `Kangyan-Zhou`.
- Diff scope read: `2` files, `+0/-5`; areas: tests/benchmarks; keywords: attention, eagle, spec, test.
- Code diff details:
  - `test/srt/nightly/test_deepseek_v32_perf.py` modified +0/-3 (3 lines); hunks: def setUpClass(cls):; def setUpClass(cls):; symbols: setUpClass, setUpClass, setUpClass
  - `test/srt/nightly/test_deepseek_v31_perf.py` modified +0/-2 (2 lines); hunks: def setUpClass(cls):; def setUpClass(cls):; symbols: setUpClass, setUpClass
- Optimization/support interpretation: The concrete diff surface is `test/srt/nightly/test_deepseek_v32_perf.py`, `test/srt/nightly/test_deepseek_v31_perf.py`; keywords observed in patches: attention, eagle, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/nightly/test_deepseek_v32_perf.py`, `test/srt/nightly/test_deepseek_v31_perf.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13394 - Fix DeepSeekV31's structural tag trigger

- Link: https://github.com/sgl-project/sglang/pull/13394
- Status/date: `merged`, created 2025-11-17, merged 2025-12-31; author `Muqi1029`.
- Diff scope read: `1` files, `+1/-1`; areas: misc; keywords: kv.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +1/-1 (2 lines); hunks: def structure_info(self) -> _GetInfoFunc:; symbols: structure_info
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/deepseekv31_detector.py`; keywords observed in patches: kv. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/deepseekv31_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13954 - Fix Deepseek v3.1 loading issue

- Link: https://github.com/sgl-project/sglang/pull/13954
- Status/date: `merged`, created 2025-11-26, merged 2025-11-26; author `Kangyan-Zhou`.
- Diff scope read: `1` files, `+1/-1`; areas: model wrapper; keywords: config, kv, quant.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: def post_load_weights(self, is_nextn=False, weight_names=None):; symbols: post_load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: config, kv, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14837 - [Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210)

- Link: https://github.com/sgl-project/sglang/pull/14837
- Status/date: `merged`, created 2025-12-10, merged 2025-12-10; author `zhyncs`.
- Diff scope read: `1` files, `+5/-1`; areas: docs/config; keywords: kv.
- Code diff details:
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +5/-1 (6 lines); hunks: {% if tools is defined and tools is not none %}
- Optimization/support interpretation: The concrete diff surface is `examples/chat_template/tool_chat_template_deepseekv31.jinja`; keywords observed in patches: kv. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `examples/chat_template/tool_chat_template_deepseekv31.jinja`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16660 - [CI] Enable dpsk v31 test on nightly H200

- Link: https://github.com/sgl-project/sglang/pull/16660
- Status/date: `merged`, created 2026-01-07, merged 2026-01-07; author `Fridge003`.
- Diff scope read: `1` files, `+1/-2`; areas: model wrapper, tests/benchmarks; keywords: cuda, kv, test.
- Code diff details:
  - `test/registered/8-gpu-models/test_deepseek_v31.py` modified +1/-2 (3 lines); hunks: from sglang.test.ci.ci_register import register_cuda_ci; symbols: TestDeepseekV31Unified, for
- Optimization/support interpretation: The concrete diff surface is `test/registered/8-gpu-models/test_deepseek_v31.py`; keywords observed in patches: cuda, kv, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/8-gpu-models/test_deepseek_v31.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17133 - [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab

- Link: https://github.com/sgl-project/sglang/pull/17133
- Status/date: `merged`, created 2026-01-15, merged 2026-01-16; author `xu-yfei`.
- Diff scope read: `6` files, `+959/-217`; areas: MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: moe, triton, config, fp8, benchmark, cache, cuda, expert, quant, router.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +337/-215 (552 lines); hunks: # Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py; sort_config,; symbols: MoeInputs:, KernelWrapper:, __init__, cuda_graph_wrapper
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`; keywords observed in patches: moe, triton, config, fp8, benchmark, cache. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17141 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- Link: https://github.com/sgl-project/sglang/pull/17141
- Status/date: `closed`, created 2026-01-15, closed 2026-01-22; author `IrisMagicBox`.
- Diff scope read: `2` files, `+14/-16`; areas: docs/config; keywords: kv.
- Code diff details:
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: {% for tool in tools %}; {%- set ns.is_last_user = false -%}
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult; symbols: detect_and_parse
- Optimization/support interpretation: The concrete diff surface is `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py`; keywords observed in patches: kv. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17178 - Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py

- Link: https://github.com/sgl-project/sglang/pull/17178
- Status/date: `merged`, created 2026-01-16, merged 2026-01-16; author `hlu1`.
- Diff scope read: `1` files, `+3/-2`; areas: tests/benchmarks; keywords: spec, test.
- Code diff details:
  - `python/sglang/test/run_eval.py` modified +3/-2 (5 lines); hunks: def get_thinking_kwargs(args):; def run_eval(args):; symbols: get_thinking_kwargs, run_eval, run_eval
- Optimization/support interpretation: The concrete diff surface is `python/sglang/test/run_eval.py`; keywords observed in patches: spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/test/run_eval.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17320 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- Link: https://github.com/sgl-project/sglang/pull/17320
- Status/date: `closed`, created 2026-01-19, closed 2026-01-19; author `IrisMagicBox`.
- Diff scope read: `2` files, `+14/-16`; areas: docs/config; keywords: kv.
- Code diff details:
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: {% for tool in tools %}; {%- set ns.is_last_user = false -%}
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult; symbols: detect_and_parse
- Optimization/support interpretation: The concrete diff surface is `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py`; keywords observed in patches: kv. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17558 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- Link: https://github.com/sgl-project/sglang/pull/17558
- Status/date: `closed`, created 2026-01-22, closed 2026-01-24; author `IrisMagicBox`.
- Diff scope read: `2` files, `+14/-16`; areas: docs/config; keywords: kv.
- Code diff details:
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: {% for tool in tools %}; {%- set ns.is_last_user = false -%}
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult; symbols: detect_and_parse
- Optimization/support interpretation: The concrete diff surface is `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py`; keywords observed in patches: kv. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `examples/chat_template/tool_chat_template_deepseekv32.jinja`, `python/sglang/srt/function_call/deepseekv32_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17761 - fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates

- Link: https://github.com/sgl-project/sglang/pull/17761
- Status/date: `open`, created 2026-01-26; author `ieBoytsov`.
- Diff scope read: `3` files, `+79/-2`; areas: tests/benchmarks, docs/config; keywords: kv, test.
- Code diff details:
  - `test/manual/test_deepseek_chat_templates.py` modified +77/-0 (77 lines); hunks: def test_tool_call_with_content(self):; symbols: test_tool_call_with_content, test_assistant_marker_after_tool_output
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +1/-1 (2 lines); hunks: {{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +1/-1 (2 lines); hunks: {{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
- Optimization/support interpretation: The concrete diff surface is `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`; keywords observed in patches: kv, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18236 - Fix function call arguments missing in streaming mode for DeepSeek V3.1

- Link: https://github.com/sgl-project/sglang/pull/18236
- Status/date: `open`, created 2026-02-04; author `momaek`.
- Diff scope read: `1` files, `+21/-3`; areas: misc; keywords: kv.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +21/-3 (24 lines); hunks: def __init__(self):; def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment, parse_streaming_increment
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/deepseekv31_detector.py`; keywords observed in patches: kv. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/deepseekv31_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21593 - Fix tool call constrained decoding and parsing for models with native formats

- Link: https://github.com/sgl-project/sglang/pull/21593
- Status/date: `merged`, created 2026-03-28, merged 2026-04-11; author `JustinTong0323`.
- Diff scope read: `9` files, `+306/-61`; areas: tests/benchmarks; keywords: spec, test, kv, config.
- Code diff details:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +113/-0 (113 lines); hunks: def test_streaming_function_call_marker_json_split_at_quotes(self):; symbols: test_streaming_function_call_marker_json_split_at_quotes, TestGetStructureConstraint, _make_tools, _make_parser
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +67/-43 (110 lines); hunks: def _process_messages(; def _process_tool_calls(; symbols: _process_messages, _process_tool_calls, _process_tool_calls, _process_tool_call_stream
  - `test/registered/openai_server/basic/test_serving_chat.py` modified +72/-0 (72 lines); hunks: def test_extract_routed_dp_rank_from_header_invalid(self):; symbols: test_extract_routed_dp_rank_from_header_invalid, TestProcessToolCallsWithRequiredToolChoice, setUp, test_required_with_parser_uses_function_call_parser
  - `python/sglang/srt/function_call/function_call_parser.py` modified +35/-11 (46 lines); hunks: from sglang.srt.entrypoints.openai.protocol import (; from sglang.srt.function_call.qwen25_detector import Qwen25Detector; symbols: parse_stream_chunk, get_structure_tag, get_structure_tag, get_structure_tag
  - `test/registered/openai_server/function_call/test_tool_choice.py` modified +8/-2 (10 lines); hunks: def test_tool_choice_specific_function_streaming(self):; def test_required_streaming_arguments_chunks_json(self):; symbols: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py`; keywords observed in patches: spec, test, kv, config. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21599 - [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1

- Link: https://github.com/sgl-project/sglang/pull/21599
- Status/date: `merged`, created 2026-03-28, merged 2026-04-20; author `alphabetc1`.
- Diff scope read: `13` files, `+1296/-33`; areas: kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: spec, config, cuda, eagle, attention, topk, cache, kv, quant, test.
- Code diff details:
  - `benchmark/bench_adaptive_speculative.py` added +263/-0 (263 lines); hunks: +"""Benchmark adaptive speculative decoding against static baselines.; symbols: build_phase_plan, send_request, run_phase, summarize_phases
  - `test/registered/unit/spec/test_adaptive_spec_params.py` added +195/-0 (195 lines); hunks: +import unittest; symbols: TestAdaptiveSpeculativeParams, test_initial_steps_snap_to_nearest_candidate_preferring_larger_step, test_update_respects_warmup_and_interval, test_empty_batches_do_not_consume_warmup_or_shift_steps
  - `test/registered/spec/eagle/test_adaptive_speculative.py` added +170/-0 (170 lines); hunks: +import json; symbols: TestAdaptiveSpeculativeServer, setUpClass, tearDownClass, _get_internal_state
  - `python/sglang/srt/speculative/eagle_worker.py` modified +162/-4 (166 lines); hunks: import logging; alloc_token_slots,; symbols: __init__, __init__, init_cuda_graphs, apply_runtime_state
  - `docs/advanced_features/adaptive_speculative_decoding.md` added +156/-0 (156 lines); hunks: +# Adaptive Speculative Decoding
- Optimization/support interpretation: The concrete diff surface is `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`; keywords observed in patches: spec, config, cuda, eagle, attention, topk. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21739 - [NPU] Update DeepSeek-V3.1 and DeepSeek-V3.2 model deployment instructions in documentation

- Link: https://github.com/sgl-project/sglang/pull/21739
- Status/date: `open`, created 2026-03-31; author `MichelleWu351`.
- Diff scope read: `1` files, `+163/-19`; areas: docs/config; keywords: attention, benchmark, cache, config, cuda, deepep, doc, eagle, expert, lora.
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +163/-19 (182 lines); hunks: you encounter issues or have any questions, please [open an issue](https://githu; We tested it based on the `RANDOM` dataset.
- Optimization/support interpretation: The concrete diff surface is `docs/platforms/ascend/ascend_npu_best_practice.md`; keywords observed in patches: attention, benchmark, cache, config, cuda, deepep. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/platforms/ascend/ascend_npu_best_practice.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22128 - Allow piecewise CUDA graph with speculative decoding

- Link: https://github.com/sgl-project/sglang/pull/22128
- Status/date: `merged`, created 2026-04-05, merged 2026-04-17; author `narutolhy`.
- Diff scope read: `4` files, `+272/-18`; areas: kernel, scheduler/runtime, tests/benchmarks; keywords: cuda, spec, quant, attention, cache, config, eagle, expert, fp8, lora.
- Code diff details:
  - `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` added +243/-0 (243 lines); hunks: +"""Test piecewise CUDA graph coexisting with speculative decoding.; symbols: TestPCGWithMTP, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/server_args.py` modified +15/-18 (33 lines); hunks: def _handle_piecewise_cuda_graph(self):; symbols: _handle_piecewise_cuda_graph
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: def can_run(self, forward_batch: ForwardBatch):; symbols: can_run
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: def init_piecewise_cuda_graphs(self):; symbols: init_piecewise_cuda_graphs
- Optimization/support interpretation: The concrete diff surface is `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; keywords observed in patches: cuda, spec, quant, attention, cache, config. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22433 - [Test] Add unit tests for DeepSeekV31Detector

- Link: https://github.com/sgl-project/sglang/pull/22433
- Status/date: `open`, created 2026-04-09; author `vedantagarwal-web`.
- Diff scope read: `1` files, `+314/-0`; areas: tests/benchmarks; keywords: kv, test.
- Code diff details:
  - `test/registered/unit/function_call/test_deepseekv31_detector.py` added +314/-0 (314 lines); hunks: +"""Unit tests for DeepSeekV31Detector — no server, no model loading.; symbols: _wrap_single, _make_tools, TestDeepSeekV31DetectorHasToolCall, setUp
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/function_call/test_deepseekv31_detector.py`; keywords observed in patches: kv, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/function_call/test_deepseekv31_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22950 - [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)

- Link: https://github.com/sgl-project/sglang/pull/22950
- Status/date: `closed`, created 2026-04-16, closed 2026-04-21; author `Wen-xuan-Xu`.
- Diff scope read: `11` files, `+597/-64`; areas: scheduler/runtime, tests/benchmarks, docs/config; keywords: cache, kv, cuda, eagle, test, attention, spec.
- Code diff details:
  - `test/registered/unit/mem_cache/test_radix_cache_thinking.py` added +238/-0 (238 lines); hunks: +import unittest; symbols: _MockReqToTokenPool:, __init__, write, _MockAllocator:
  - `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py` added +220/-0 (220 lines); hunks: +import unittest; symbols: _MockReqToTokenPool:, __init__, write, _MockAllocator:
  - `python/sglang/srt/mem_cache/mamba_radix_cache.py` modified +62/-50 (112 lines); hunks: from numpy import float64; MatchPrefixParams,; symbols: cache_finished_req, _skip_cache_unfinished_req, _skip_cache_unfinished_req
  - `python/sglang/srt/mem_cache/radix_cache_cpp.py` modified +27/-14 (41 lines); hunks: MatchPrefixParams,; def cache_finished_req(self, req: Req, is_insert: bool = True):; symbols: cache_finished_req, cache_unfinished_req, cache_unfinished_req, pretty_print
  - `python/sglang/srt/mem_cache/common.py` modified +22/-0 (22 lines); hunks: def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:; symbols: alloc_for_decode, maybe_strip_thinking_tokens, release_kv_cache
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py`; keywords observed in patches: cache, kv, cuda, eagle, test, attention. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22981 - [Test] Add unit tests for 7 missing function call detectors

- Link: https://github.com/sgl-project/sglang/pull/22981
- Status/date: `open`, created 2026-04-16; author `JustinTong0323`.
- Diff scope read: `3` files, `+1017/-1`; areas: tests/benchmarks; keywords: test, doc, kv, moe.
- Code diff details:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +960/-1 (961 lines); hunks: import unittest; from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector; symbols: TestPythonicDetector, setUp, test_has_tool_call, test_detect_and_parse_single
  - `test/registered/openai_server/function_call/test_tool_choice.py` modified +57/-0 (57 lines); hunks: def setUpClass(cls):; symbols: setUpClass, TestToolChoiceWithConstrainedDecoding, setUpClass, test_tool_choice_required_strict_finish_reason
  - `test/registered/unit/function_call/test_kimik2_detector.py` renamed +0/-0 (0 lines)
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_kimik2_detector.py`; keywords observed in patches: test, doc, kv, moe. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_kimik2_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23315 - Opt-in strip of thinking tokens from radix cache

- Link: https://github.com/sgl-project/sglang/pull/23315
- Status/date: `merged`, created 2026-04-21, merged 2026-04-21; author `hnyls2002`.
- Diff scope read: `4` files, `+72/-4`; areas: scheduler/runtime, tests/benchmarks; keywords: cache, kv, spec, cuda, scheduler, test.
- Code diff details:
  - `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` modified +52/-1 (53 lines); hunks: from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType; def test_cache_finished_req_insert(self):; symbols: test_cache_finished_req_insert, test_cache_finished_req_strips_thinking, test_cache_finished_req_no_insert
  - `python/sglang/srt/managers/schedule_batch.py` modified +9/-2 (11 lines); hunks: def output_ids_through_stop(self) -> List[int]:; def pop_overallocated_kv_cache(self) -> Tuple[int, int]:; symbols: output_ids_through_stop, _cache_commit_len, pop_committed_kv_cache, pop_overallocated_kv_cache
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs:, add_cli_args
  - `python/sglang/srt/mem_cache/common.py` modified +3/-1 (4 lines); hunks: def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = Tr; symbols: release_kv_cache
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: cache, kv, spec, cuda, scheduler, test. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23336 - [SPEC V2][2/N] feat: adaptive spec support spec v2

- Link: https://github.com/sgl-project/sglang/pull/23336
- Status/date: `open`, created 2026-04-21; author `alphabetc1`.
- Diff scope read: `6` files, `+193/-10`; areas: multimodal/processor, scheduler/runtime; keywords: spec, eagle, cuda, scheduler, attention, processor, config, kv, moe, topk.
- Code diff details:
  - `python/sglang/srt/speculative/eagle_worker_v2.py` modified +173/-0 (173 lines); hunks: from sglang.srt.managers.schedule_batch import ModelWorkerBatch; def __init__(; symbols: __init__, __init__, target_worker, forward_batch_generation
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +8/-4 (12 lines); hunks: def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):; def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):; symbols: prepare_for_decode, prepare_for_decode, prepare_for_v2_draft
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +10/-1 (11 lines); hunks: def _resolve_spec_overlap_token_ids(; symbols: _resolve_spec_overlap_token_ids
  - `python/sglang/srt/speculative/adaptive_spec_params.py` modified +0/-5 (5 lines); hunks: def adaptive_unsupported_reason(server_args: ServerArgs) -> str \| None:; symbols: adaptive_unsupported_reason
  - `python/sglang/srt/managers/utils.py` modified +1/-0 (1 lines); hunks: class GenerationBatchResult:; symbols: GenerationBatchResult:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`; keywords observed in patches: spec, eagle, cuda, scheduler, attention, processor. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 32; open PRs: 6.
- Open PRs to keep tracking: [#17761](https://github.com/sgl-project/sglang/pull/17761), [#18236](https://github.com/sgl-project/sglang/pull/18236), [#21739](https://github.com/sgl-project/sglang/pull/21739), [#22433](https://github.com/sgl-project/sglang/pull/22433), [#22981](https://github.com/sgl-project/sglang/pull/22981), [#23336](https://github.com/sgl-project/sglang/pull/23336)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
