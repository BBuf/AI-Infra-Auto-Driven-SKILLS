---
name: sglang-deepseek-v31-optimization
description: PR-backed and current-main optimization manual for DeepSeek V3.1 and DeepSeek-V3.1-Terminus in SGLang. Use when Codex needs to recover, extend, or audit DeepSeek V3.1 tool calling, thinking mode, chat templates, streaming parser behavior, loading fixes, MTP validation, fused MoE configs, or backend-specific tests.
---

# SGLang DeepSeek V3.1 Optimization

## Overview

This skill covers the DeepSeek V3.1 support and optimization ladder that is active in SGLang main. V3.1 shares the DeepSeek V3/R1 model backbone, but its tool-call format, chat template, thinking flag, streaming parser, and validation lanes are separate enough to require an independent skill.

Current-main snapshot:

- SGLang `origin/main`: `929e00eea` on `2026-04-21`
- sgl-cookbook `origin/main`: `8ec4d03` on `2026-04-21`
- core runtime entry: `python/sglang/srt/models/deepseek_v2.py`
- V3.1 tool parser: `python/sglang/srt/function_call/deepseekv31_detector.py`
- V3.1 tool template: `examples/chat_template/tool_chat_template_deepseekv31.jinja`

The historical evidence lives in:

- [references/pr-history.md](references/pr-history.md): chronological PR evidence and code-level notes
- [references/playbook.md](references/playbook.md): investigation order, symptom mapping, validation commands

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Record the exact serving shape first:

- model: `deepseek-ai/DeepSeek-V3.1`, `DeepSeek-V3.1-Terminus`, or `DeepSeek-V3.1-Speciale`
- whether thinking mode is enabled with `chat_template_kwargs.thinking`
- whether tool calling is enabled with `--tool-call-parser deepseekv31`
- whether the V3.1 tool template is used: `examples/chat_template/tool_chat_template_deepseekv31.jinja`
- `--reasoning-parser deepseek-v3`
- TP / DP / EP / PP topology
- MTP enabled or not
- backend and quantization inherited from the DeepSeek V3/R1 backbone
- streaming or non-streaming OpenAI API path
- exact test lane: manual, nightly, parser unit, chat-template unit, or model-backed accuracy/perf

## Core Principle

Do not debug V3.1 as ordinary V3.

- The model class is shared with V3/R1, so MLA, MoE, quantization, DeepEP, and MTP bugs usually belong to `sglang-deepseek-v3-r1-optimization`.
- The user-visible V3.1 delta is parser and template behavior: hybrid thinking, tool calling, streaming deltas, and structural-tag constrained decoding.
- V3.1 tool calls do not use V3's literal `function` marker or fenced JSON block.
- V3.1 uses `chat_template_kwargs: {"thinking": true}` with `--reasoning-parser deepseek-v3`, not R1's parser.
- DeepSeek-V3.1-Speciale should not be treated as a tool-calling target.

The optimization order matters:

1. confirm the parser/template contract
2. confirm streaming and non-streaming parity
3. confirm thinking mode and `</think>` template behavior
4. confirm model loading and MTP inherited from V3/R1
5. only then tune MoE/backend performance
6. add CPU parser tests for parser-only changes before running model-backed tests

## Main Runtime Surfaces

Start from these files before changing behavior:

- `python/sglang/srt/function_call/deepseekv31_detector.py`
- `python/sglang/srt/function_call/function_call_parser.py`
- `examples/chat_template/tool_chat_template_deepseekv31.jinja`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `python/sglang/srt/parser/reasoning_parser.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/models/deepseek_v2.py`
- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- `test/manual/test_deepseek_v31.py`
- `test/manual/nightly/test_deepseek_v31_perf.py`
- `test/manual/test_deepseek_chat_templates.py`

## Open PRs to Track

Check these before declaring a V3.1 gap:

- [#17761](https://github.com/sgl-project/sglang/pull/17761): missing Assistant token after tool output in V3.1/V3.2 chat templates, open.
- [#18236](https://github.com/sgl-project/sglang/pull/18236): function-call arguments missing in V3.1 streaming mode, open.
- [#21739](https://github.com/sgl-project/sglang/pull/21739): NPU deployment docs for V3.1/V3.2, open.
- [#22433](https://github.com/sgl-project/sglang/pull/22433): CPU unit tests for `DeepSeekV31Detector`, open.
- [#22981](https://github.com/sgl-project/sglang/pull/22981): broader CPU tests for missing function-call detectors, open.
- [#23336](https://github.com/sgl-project/sglang/pull/23336): adaptive speculative-num-steps support for spec v2 EAGLE workers, open and relevant when V3.1 MTP runs with spec v2.

## Additional PR Coverage

Additional all-state PR coverage includes parser/template PRs that are relevant to V3.1 even though they are not the core bring-up PRs:

- [#9468](https://github.com/sgl-project/sglang/pull/9468), [#10875](https://github.com/sgl-project/sglang/pull/10875), and [#11189](https://github.com/sgl-project/sglang/pull/11189) refine reasoning/thinking docs, request handling, and eval flags.
- [#9895](https://github.com/sgl-project/sglang/pull/9895) and [#14837](https://github.com/sgl-project/sglang/pull/14837) update `tool_chat_template_deepseekv31.jinja`.
- [#10550](https://github.com/sgl-project/sglang/pull/10550), [#11223](https://github.com/sgl-project/sglang/pull/11223), [#11589](https://github.com/sgl-project/sglang/pull/11589), and [#21593](https://github.com/sgl-project/sglang/pull/21593) are general tool-choice / constrained-decoding / parser fixes that affect V3.1 serving behavior.
- [#17141](https://github.com/sgl-project/sglang/pull/17141), [#17320](https://github.com/sgl-project/sglang/pull/17320), and [#17558](https://github.com/sgl-project/sglang/pull/17558) are closed attempts around the missing-Assistant-token-after-tool-output issue that remains tracked by open [#17761](https://github.com/sgl-project/sglang/pull/17761).
- [#22950](https://github.com/sgl-project/sglang/pull/22950) closed the first parser-gated reasoning radix-cache stripping attempt, while merged [#23315](https://github.com/sgl-project/sglang/pull/23315) adds the current opt-in strip of thinking tokens from radix cache. For V3.1 thinking mode, this is cache-behavior work rather than a new tool parser.
- [#21599](https://github.com/sgl-project/sglang/pull/21599) and [#22128](https://github.com/sgl-project/sglang/pull/22128) are inherited MTP/speculative-decoding infrastructure updates: adaptive EAGLE draft steps and piecewise CUDA graph with speculative decoding.

## Evolution Path

### Stage V31-0: Split V3.1 tool calling from V3

DeepSeek V3.1 has a distinct tool-call wire format:

```text
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{json_args}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

It does not use V3's `function` literal or fenced `json` block.

Key PR:

- [#9446](https://github.com/sgl-project/sglang/pull/9446)

Success check:

- `--tool-call-parser deepseekv31` resolves to `DeepSeekV31Detector`
- the template emits V3.1 markers, not V3 markers
- multiple tool calls can be chained without separators

### Stage V31-1: Enable hybrid thinking correctly

V3.1 thinking mode is toggled through the chat template, not through the R1 parser.

Key PR:

- [#9464](https://github.com/sgl-project/sglang/pull/9464)

Rules:

- launch with `--reasoning-parser deepseek-v3`
- send `extra_body: {"chat_template_kwargs": {"thinking": true}}`
- the template emits `<think>` when thinking is enabled and `</think>` when non-thinking is desired

### Stage V31-2: Keep chat template argument types stable

V3.1 multi-turn tool calls can pass `tool["function"]["arguments"]` as either a dict or an already serialized JSON string. The template must not double-escape JSON strings.

Key PR:

- [#12123](https://github.com/sgl-project/sglang/pull/12123)

Success check:

- dict arguments are rendered through `tojson`
- string arguments are used as-is
- mixed multi-tool calls keep both forms intact

### Stage V31-3: Harden structural tags and streaming deltas

The structural trigger should be the generic per-call begin token, not the full name-specific begin string. Streaming must also preserve arguments that arrive in the first chunk and normal text before the tool marker.

Key PRs:

- [#13394](https://github.com/sgl-project/sglang/pull/13394)
- [#18236](https://github.com/sgl-project/sglang/pull/18236), open
- [#22433](https://github.com/sgl-project/sglang/pull/22433), open

Success check:

- `structure_info().trigger` is `<｜tool▁call▁begin｜>`
- streaming emits the function name and the first argument diff when they appear together
- normal text before the first tool marker is not dropped
- CPU parser tests cover invalid JSON, unknown tools, unicode args, multiple calls, and streaming chunks

### Stage V31-4: Keep inherited model loading and MTP healthy

V3.1 still depends on DeepSeek V3/R1 loader, MLA, MoE, and MTP surfaces.

Key PRs:

- [#13954](https://github.com/sgl-project/sglang/pull/13954)
- [#16660](https://github.com/sgl-project/sglang/pull/16660)
- [#13190](https://github.com/sgl-project/sglang/pull/13190)

Success check:

- `test/manual/test_deepseek_v31.py` covers TP8 and TP8+MTP
- nightly perf no longer carries stale `enable_dp_attention` flags
- loading fixes are checked in the shared DeepSeek V2/V3 loader and in parser code

### Stage V31-5: Tune MoE configs as a DeepSeek-family shape

V3.1 shares the DeepSeek MoE shape with V3/V3.2-style fused MoE config work.

Key PR:

- [#17133](https://github.com/sgl-project/sglang/pull/17133)

Success check:

- H20 and H20-3E configs exist for `E=257,N=256,fp8_w8a8`
- config selection is validated on the same hardware lane that motivated the tuning

## Validation Surface

Use the narrowest lane that matches the change:

- parser-only: open [#22433](https://github.com/sgl-project/sglang/pull/22433) test pattern, or equivalent `DeepSeekV31Detector` CPU unit tests
- template-only: `test/manual/test_deepseek_chat_templates.py`
- V3.1 base/MTP: `test/manual/test_deepseek_v31.py`
- nightly performance: `test/manual/nightly/test_deepseek_v31_perf.py`
- inherited MLA/MoE/quant bug: switch to `sglang-deepseek-v3-r1-optimization`
