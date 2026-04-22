# DeepSeek V3.1 PR History

Snapshot:

- SGLang `origin/main`: `929e00eea`
- sgl-cookbook `origin/main`: `8ec4d03`
- Date: `2026-04-21`

This history covers DeepSeek V3.1 only. DeepSeek V3/R1 runtime internals and DeepSeek V3.2 DSA/NSA are tracked by separate skills.

## Chronological Timeline

| Date | PR | State | Area | Main effect |
| --- | ---: | --- | --- | --- |
| 2025-08-21 | [#9446](https://github.com/sgl-project/sglang/pull/9446) | merged | tool calling | Added `tool_chat_template_deepseekv31.jinja`, `DeepSeekV31Detector`, and parser registration. |
| 2025-08-21 | [#9464](https://github.com/sgl-project/sglang/pull/9464) | merged | thinking parser | Added V3.1 thinking-mode docs and `deepseek-v3` reasoning-parser support. |
| 2025-08-23 | [#9544](https://github.com/sgl-project/sglang/pull/9544) | merged | docs | Added DeepSeek V3.1 support docs and examples. |
| 2025-10-25 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | chat template | Fixed V3/V3.1/V3.2 templates so dict and string tool arguments do not double-escape. |
| 2025-11-13 | [#13190](https://github.com/sgl-project/sglang/pull/13190) | merged | nightly tests | Removed stale `enable_dp_attention` from V3.1/V3.2 nightly tests. |
| 2025-11-17 | [#13394](https://github.com/sgl-project/sglang/pull/13394) | merged | structural tags | Changed V3.1 structural trigger to generic `<｜tool▁call▁begin｜>`. |
| 2025-11-26 | [#13954](https://github.com/sgl-project/sglang/pull/13954) | merged | loading | Fixed DeepSeek V3.1 loading issue in `deepseek_v2.py`. |
| 2026-01-07 | [#16660](https://github.com/sgl-project/sglang/pull/16660) | merged | CI | Enabled DeepSeek V3.1 registered nightly H200 test. |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | MoE tuning | Added H20/H20-3E fused MoE configs for V3.1/V3.2 shapes. |
| 2026-01-26 | [#17761](https://github.com/sgl-project/sglang/pull/17761) | open | chat template | Tracks missing Assistant token after tool output in V3.1/V3.2 templates. |
| 2026-02-04 | [#18236](https://github.com/sgl-project/sglang/pull/18236) | open | streaming parser | Fixes missing arguments and dropped normal text in V3.1 streaming function calls. |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | MTP/spec | Added adaptive `speculative_num_steps` for EAGLE top-k=1, inherited by V3.1 MTP. |
| 2026-03-31 | [#21739](https://github.com/sgl-project/sglang/pull/21739) | open | NPU docs | Updates V3.1/V3.2 Ascend NPU deployment instructions. |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | PCG/spec | Allowed piecewise CUDA graph with speculative decoding. |
| 2026-04-09 | [#22433](https://github.com/sgl-project/sglang/pull/22433) | open | parser tests | Adds CPU unit tests for `DeepSeekV31Detector`. |
| 2026-04-16 | [#22981](https://github.com/sgl-project/sglang/pull/22981) | open | parser tests | Adds broader CPU tests for missing function-call detectors. |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | reasoning cache | Explored parser-gated two-phase radix-cache stripping for reasoning tokens. |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | reasoning cache | Added opt-in stripping of thinking tokens from radix cache. |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | spec v2 | Extends adaptive speculative decoding to spec v2 EAGLE workers. |

## Additional PR Coverage

Additional all-state PR coverage also includes V3.1-adjacent parser and template changes that should not be omitted from an audit:

- 2025-08-21 [#9468](https://github.com/sgl-project/sglang/pull/9468): updated reasoning-parser docs after the V3.1 thinking parser landed.
- 2025-09-02 [#9895](https://github.com/sgl-project/sglang/pull/9895): updated `tool_chat_template_deepseekv31.jinja`.
- 2025-09-17 [#10550](https://github.com/sgl-project/sglang/pull/10550): used `jsonschema` for required or specific tool choice.
- 2025-09-24 [#10875](https://github.com/sgl-project/sglang/pull/10875): improved request-level thinking enablement.
- 2025-10-03 [#11189](https://github.com/sgl-project/sglang/pull/11189): added `--thinking-mode` to `run_eval`.
- 2025-10-05 [#11223](https://github.com/sgl-project/sglang/pull/11223): updated tool parser and related docs.
- 2025-10-14 [#11589](https://github.com/sgl-project/sglang/pull/11589): streamlined function arguments when `tool_choice="auto"` for `deepseekv31_detector`.
- 2025-12-10 [#14837](https://github.com/sgl-project/sglang/pull/14837): auto-synced `tool_chat_template_deepseekv31.jinja`.
- 2026-01-15 [#17141](https://github.com/sgl-project/sglang/pull/17141), 2026-01-19 [#17320](https://github.com/sgl-project/sglang/pull/17320), and 2026-01-22 [#17558](https://github.com/sgl-project/sglang/pull/17558): closed attempts around `finish_reason="stop"` after tool content; open [#17761](https://github.com/sgl-project/sglang/pull/17761) remains the current tracker.
- 2026-01-16 [#17178](https://github.com/sgl-project/sglang/pull/17178): removed `deepseek-r1` from thinking-mode choices, clarifying V3.1 vs R1 parser semantics.
- 2026-03-28 [#21593](https://github.com/sgl-project/sglang/pull/21593): fixed tool-call constrained decoding and parsing for native-format models.
- 2026-04-16 [#22950](https://github.com/sgl-project/sglang/pull/22950) and 2026-04-21 [#23315](https://github.com/sgl-project/sglang/pull/23315): distinguish closed reasoning-cache exploration from the merged opt-in strip of thinking tokens from radix cache.
- 2026-03-28 [#21599](https://github.com/sgl-project/sglang/pull/21599), 2026-04-05 [#22128](https://github.com/sgl-project/sglang/pull/22128), and open 2026-04-21 [#23336](https://github.com/sgl-project/sglang/pull/23336): inherited MTP/spec-v2 infrastructure to check when V3.1 uses EAGLE MTP.

## Code-Level Narrative

### 1. Tool-call parser and template

[#9446](https://github.com/sgl-project/sglang/pull/9446) created the V3.1 tool-call surface. It added:

- `examples/chat_template/tool_chat_template_deepseekv31.jinja`
- `python/sglang/srt/function_call/deepseekv31_detector.py`
- `deepseekv31` registration in `function_call_parser.py`

The V3.1 format is:

```text
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>{json_arguments}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

This differs from V3 because V3.1 does not insert a literal `function` token and does not fence arguments in a Markdown `json` block. The detector therefore searches for `<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>` and extracts name/arguments with `<｜tool▁sep｜>`.

### 2. Thinking parser

[#9464](https://github.com/sgl-project/sglang/pull/9464) established that V3.1 thinking mode is controlled by `chat_template_kwargs.thinking` and parsed with `--reasoning-parser deepseek-v3`. It also updated serving logic so model-specific thinking flags can be read from chat-template kwargs.

Current cookbook docs launch Terminus with:

```shell
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.1-Terminus \
  --reasoning-parser deepseek-v3 \
  --tool-call-parser deepseekv31 \
  --chat-template ./examples/chat_template/tool_chat_template_deepseekv31.jinja
```

The important distinction is that R1's parser is not used here. V3.1 is hybrid thinking/non-thinking and the template decides whether to emit `<think>` or `</think>`.

Thinking mode also intersects with radix-cache reuse. [#22950](https://github.com/sgl-project/sglang/pull/22950) is a closed attempt at parser-gated two-phase reasoning cache stripping. The merged implementation is [#23315](https://github.com/sgl-project/sglang/pull/23315), which adds an opt-in path in `server_args.py`, `schedule_batch.py`, and `mem_cache/common.py` to strip thinking tokens from radix-cache entries. For V3.1 this does not change the `deepseekv31` tool parser; it changes whether thinking tokens can be reused as prefix-cache material.

### 3. Template argument type checking

[#12123](https://github.com/sgl-project/sglang/pull/12123) fixed issue `#11700` across DeepSeek V3/V3.1/V3.2 templates. Tool arguments can arrive as:

- a dict from an OpenAI API object
- an already serialized JSON string from a previous model/tool round

The template now does:

```jinja
{% set formatted_args = tool['function']['arguments'] if tool['function']['arguments'] is string else tool['function']['arguments']|tojson %}
```

This prevents already serialized strings from becoming escaped JSON strings such as `"{\"symbol\":\"NVDA\"}"`.

### 4. Structural trigger and streaming parser

[#13394](https://github.com/sgl-project/sglang/pull/13394) changed:

```python
trigger="<｜tool▁call▁begin｜>"
```

instead of a trigger that included the concrete function name and separator. This matters for constrained decoding because the trigger has to activate before the function name is known.

[#18236](https://github.com/sgl-project/sglang/pull/18236) is still open, but its patch explains the current streaming risk. The existing streaming parser can skip arguments when the function name and JSON arguments arrive in the first chunk because argument processing is inside the `else` branch after `current_tool_name_sent`. The PR changes this to process `func_args_raw` whenever it exists. It also adds `_normal_text_sent` so normal text before the first tool marker is emitted instead of dropped.

[#22433](https://github.com/sgl-project/sglang/pull/22433) adds the missing CPU tests. The proposed tests cover `has_tool_call`, non-streaming parse, invalid JSON fallback, unknown tools, unicode arguments, multi-call ordering, streaming diffs, `structure_info`, and structural tag support.

### 5. Loading, MTP, and MoE tuning

[#13954](https://github.com/sgl-project/sglang/pull/13954) fixed a V3.1 loading issue in `deepseek_v2.py`. Because V3.1 reuses the V3/R1 model class, loading fixes usually belong to shared DeepSeek code rather than the V3.1 parser.

[#16660](https://github.com/sgl-project/sglang/pull/16660) added a registered H200 V3.1 lane with two variants:

- TP8 base
- TP8 plus EAGLE MTP with `SGLANG_ENABLE_SPEC_V2=1`

[#13190](https://github.com/sgl-project/sglang/pull/13190) removed stale `enable_dp_attention` from V3.1/V3.2 nightly perf tests, keeping the benchmark shape aligned with the current server defaults.

[#21599](https://github.com/sgl-project/sglang/pull/21599) adds adaptive EAGLE top-k=1 draft steps; [#22128](https://github.com/sgl-project/sglang/pull/22128) allows piecewise CUDA graph with speculative decoding; open [#23336](https://github.com/sgl-project/sglang/pull/23336) applies adaptive spec to spec v2 workers. These PRs are not V3.1 parser work, but they matter for the TP8+MTP validation shape because V3.1 reuses the DeepSeek V3/R1 NextN stack.

[#17133](https://github.com/sgl-project/sglang/pull/17133) added H20/H20-3E fused MoE configs for DeepSeek-family `E=257,N=256,fp8_w8a8` shapes. This is not parser work, but it affects V3.1 throughput on those devices.
