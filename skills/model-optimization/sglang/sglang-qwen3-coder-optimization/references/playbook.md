# Qwen3-Coder Playbook

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| XML-ish tool calls malformed or missing parameters | `qwen3_coder_detector.py`, parser tests | `#8357`, `#8371`, `#8445`, `#16744` |
| Unknown tool call disappears | `base_format_detector.py`, `qwen3_coder_detector.py`, `environ.py` | `#12226` |
| Required or named `tool_choice` constraint fails | `function_call_parser.py`, JSON schema constraint tests | `#8357`, `#13163` |
| String arguments converted to wrong type | `qwen3_coder_detector.py` | `#16744`, open `#13411` |
| Streaming arguments arrive as one huge chunk | `qwen3_coder_detector.py`, OpenAI streaming deltas | open `#21829` |
| Qwen3-Coder-Next NVFP4 load fails | `qwen3_next.py`, ModelOpt packed mapping | `#18224` |
| AMD Qwen3-Coder-Next attention or MTP failure | `aiter_backend.py`, AMD registered tests | `#18355`, `#18608`, `#19736` |
| NPU fused MoE shape mismatch | `fused_moe_method_npu.py`, hybrid attention import | `#18700` |
| MoE perf regression | Triton fused MoE configs and tuning scripts | `#17965`, `#18195` |
| Missing production perf signal for 480B | nightly benchmark tests and workflows | open `#13979` |

## Investigation Order

1. Reproduce parser-only behavior with synthetic chunks and no model server.
2. If parsing succeeds, reproduce raw model output without `--tool-call-parser`.
3. Enable `--tool-call-parser qwen3_coder` and test non-streaming complete JSON.
4. Test streaming deltas with tiny chunks, including tool name, parameter prefix, partial content, closing tag, and multiple parameters.
5. For Qwen3-Coder-Next, verify checkpoint loading without quantization before enabling FP8 or NVFP4.
6. Enable MTP/EAGLE, FP8 KV, EP, or backend-specific flags one at a time.
7. Only after correctness is stable, run MoE/GDN performance benchmarks.

## Parser Rules

- Do not revive `build_ebnf()` for Qwen3-Coder; PR #13163 removed detector-owned EBNF in favor of JSON Schema constraints.
- Keep XML-ish parser state explicit. Hidden state bugs have already caused streaming regressions.
- Unknown-tool forwarding is opt-in through `SGLANG_FORWARD_UNKNOWN_TOOLS`; test both false and true.
- Treat schema-aware string preservation as a required future direction, but mark PR #13411 as open until it lands.
- For incremental argument streaming, only stream string-like schema types. Non-string values need the full text for conversion.

## Runtime Rules

- Qwen3-Coder-480B-A35B MoE tuning should check TP and EP together; EP remaps top-k expert ids.
- Qwen3-Coder-Next belongs to the Qwen3-Next hybrid runtime path. Runtime patches should cross-link Qwen3-Next validation.
- ModelOpt NVFP4 depends on packed-module mapping and KV-scale key remapping. Test both.
- AMD validation must cover AITER basic and MTP paths. NPU validation must inspect MoE weight layout after load.

## Validation Checklist

- Parser one-shot: single tool, multiple tools, empty argument object, nested JSON-like string, string-looking number, unknown tool.
- Parser streaming: tool name delta, parameter diff, multi-parameter ordering, invalid function name, exact JSON argument reconstruction.
- Constraint path: `tool_choice=required` and specific function name through JSON Schema.
- Qwen3-Coder-480B: TP=8, EP mode, FP8 checkpoint load, multithreaded load if using the nightly recipe.
- Qwen3-Coder-Next: BF16 baseline, FP8/NVFP4 load, MTP/EAGLE, FP8 KV where backend supports it.
- AMD: `--attention-backend aiter`, MI35x basic and MTP registered tests.
- NPU: fused MoE `w13` and `w2` shapes after load.

## Documentation Rules

- Every cited PR must have motivation, key implementation, reviewed files, validation impact, and a real snippet from the diff.
- Open PRs can be included only if their diff was read and their state is clearly marked open.
- Do not use title-only "radar" lists for this model family.
- When a parser change affects Qwen3.6 docs through `--tool-call-parser qwen3_coder`, cross-link the Qwen3.6 skill.
