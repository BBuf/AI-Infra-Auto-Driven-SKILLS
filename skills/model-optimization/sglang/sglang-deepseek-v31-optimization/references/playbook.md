# DeepSeek V3.1 Playbook

Use this playbook when a V3.1 optimization or regression report is ambiguous. The goal is to decide whether the bug belongs to V3.1 parser/template behavior or to the inherited DeepSeek V3/R1 runtime.

## 1. Identify The Serving Contract

Collect these first:

- model path: V3.1, V3.1-Terminus, or V3.1-Speciale
- `--tool-call-parser`
- `--reasoning-parser`
- chat template path
- `extra_body.chat_template_kwargs.thinking`
- streaming or non-streaming request
- whether request includes tools
- TP / DP / EP / PP
- MTP and `SGLANG_ENABLE_SPEC_V2`
- hardware and quantization

If the issue is MLA, MoE, W4AFP8, DeepEP, LoRA, or backend selection, switch to `sglang-deepseek-v3-r1-optimization`.
If the issue is DSA/NSA sparse attention, switch to `sglang-deepseek-v32-optimization`.

## 2. Tool-Call Format

V3.1 format:

```text
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{json_args}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

V3 format differs:

- V3 has a literal `function` marker after `<｜tool▁call▁begin｜>`.
- V3 wraps arguments in a fenced JSON block.
- V3.1 does not.

If `deepseekv3` and `deepseekv31` are accidentally swapped, non-streaming may fail to parse or streaming may emit malformed deltas.

## 3. Thinking Mode

Expected launch and request:

```shell
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.1-Terminus \
  --reasoning-parser deepseek-v3 \
  --tool-call-parser deepseekv31 \
  --chat-template examples/chat_template/tool_chat_template_deepseekv31.jinja
```

```json
{"chat_template_kwargs": {"thinking": true}}
```

Common mistakes:

- using `--reasoning-parser deepseek-r1`
- sending `enable_thinking` instead of `thinking`
- using DeepSeek-V3.1-Speciale as a tool-call target
- forgetting the V3.1 tool template when tool calling is enabled

## 4. Streaming Parser

Read `DeepSeekV31Detector.parse_streaming_increment`.

Check:

- `_buffer` contains the outer or inner tool marker
- `current_tool_id` starts at `0` only after the first tool call is detected
- `current_tool_name_sent` is reset after each complete call
- `_last_arguments` is cleared between calls
- first-chunk arguments are not skipped when the function name and JSON arrive together
- normal text before `<｜tool▁call▁begin｜>` is preserved

Open [#18236](https://github.com/sgl-project/sglang/pull/18236) is the current reference for missing streaming arguments and dropped normal text.

## 5. Structural Tag Constrained Decoding

`structure_info()` should return:

- `begin`: `<｜tool▁call▁begin｜>` + name + `<｜tool▁sep｜>`
- `end`: `<｜tool▁call▁end｜>`
- `trigger`: `<｜tool▁call▁begin｜>`

The trigger must not include the function name. That was fixed in [#13394](https://github.com/sgl-project/sglang/pull/13394).

## 6. Chat Template Regressions

Read `examples/chat_template/tool_chat_template_deepseekv31.jinja`.

Check:

- dict arguments use `tojson`
- string arguments are emitted as-is
- tool output blocks are wrapped with `<｜tool▁output▁begin｜>` and `<｜tool▁output▁end｜>`
- assistant generation after tool output has the right `<｜Assistant｜>` marker
- `thinking=false` emits `</think>` after `<｜Assistant｜>`
- `thinking=true` emits `<think>`

Open [#17761](https://github.com/sgl-project/sglang/pull/17761) tracks missing Assistant token after tool output in V3.1/V3.2 templates.

## 7. Loading And MTP

For V3.1 model-backed failures:

- inspect `deepseek_v2.py`
- inspect `deepseek_common/deepseek_weight_loader.py`
- inspect `deepseek_nextn.py` when MTP is enabled
- compare against V3/R1 lanes before changing parser code

Expected V3.1 manual variants:

- TP8 base
- TP8 + EAGLE MTP with `SGLANG_ENABLE_SPEC_V2=1`

## 8. Validation Order

Pick the smallest lane:

1. parser-only: `DeepSeekV31Detector` unit tests following open [#22433](https://github.com/sgl-project/sglang/pull/22433)
2. template-only: `test/manual/test_deepseek_chat_templates.py`
3. model-backed V3.1: `test/manual/test_deepseek_v31.py`
4. nightly performance: `test/manual/nightly/test_deepseek_v31_perf.py`
5. inherited runtime regression: V3/R1 validation lanes

## 9. Common False Conclusions

- "V3.1 is only V3 with a new checkpoint." Not for tool calling or thinking.
- "R1 parser should be used because V3.1 thinks." V3.1 uses `deepseek-v3`.
- "Tool arguments are always dicts." Multi-turn paths may pass already serialized JSON strings.
- "A streaming parser bug is fixed if non-streaming works." V3.1 has a custom streaming override and needs separate validation.
- "Speciale supports tools." Current cookbook docs say DeepSeek-V3.1-Speciale does not support tool calling.
