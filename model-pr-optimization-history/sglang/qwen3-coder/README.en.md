# SGLang Qwen3-Coder Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: Qwen3-Coder-480B-A35B, Qwen3-Coder-Next, the `qwen3_coder` parser, streaming tool arguments, NVFP4/FP8, and AMD/NPU/Blackwell recipes.

Summary: keep parser correctness separate from model performance. `qwen3_coder_detector.py` is its own high-risk surface and is reused by Qwen3.6 docs. Qwen3-Coder-Next runtime mostly shares the Qwen3-Next hybrid path across GDN/Mamba/MTP/cache, MoE, ModelOpt, AMD, and NPU.

## Code Surfaces

- `python/sglang/srt/function_call/qwen3_coder_detector.py`
- `python/sglang/srt/function_call/base_format_detector.py`
- `python/sglang/srt/function_call/function_call_parser.py`
- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `python/sglang/srt/layers/attention/aiter_backend.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`
- `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`
- `test/registered/amd/test_qwen3_coder_next_8gpu.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder-Next.mdx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-next-deployment.jsx`

## Manually Reviewed PR Cards

### PR #8357 - XML-ish grammar and Qwen3-Coder detector fixes

- Link: https://github.com/sgl-project/sglang/pull/8357
- State: merged, `2025-07-25T05:08:06Z`
- Diff coverage: full patch reviewed, `6` files, `+305/-58`.
- Motivation: after PR #8260, Qwen3-Coder required and named `tool_choice` still failed because `EBNFComposer` lacked XML-like grammar. The detector also had streaming index, structural-tag, and registry-name issues.
- Key implementation: added XML format support to `EBNFComposer`, renamed the detector registration to `qwen3_coder`, and added `supports_structural_tag()` so XML-like tool-call streams are not wrapped again.
- Key snippet:

```python
FORMAT_TYPE_OVERRIDES = {
    "pythonic": {"boolean": '"True" | "False"', "null": '"None"'},
    "xml": {"string": "xml_text"},
}
```

```python
def supports_structural_tag(self) -> bool:
    return False
```

- Reviewed files: `base_format_detector.py`, `ebnf_composer.py`, `function_call_parser.py`, `pythonic_detector.py`, `qwen3_coder_detector.py`, `test/srt/test_function_call_parser.py`.
- Validation impact: cover XML parameter tags, `tool_choice=required`, named function selection, streaming indices, and disabled structural tags.

### PR #8371 - Streaming parser update

- Link: https://github.com/sgl-project/sglang/pull/8371
- State: merged, `2025-08-08T06:42:29Z`
- Diff coverage: full patch reviewed, `2` files, `+304/-54`.
- Motivation: the old parser buffered too much output and could raise `AttributeError` during streaming. Clients also needed the function name as soon as the `<function=...>` header completed.
- Key implementation: added explicit current-function, current-parameter, streamed-argument, and in-tool-call state. Streaming now emits the tool name first with empty parameters, then emits JSON argument diffs when parameter blocks become parseable.
- Key snippet:

```python
self._current_function_name: str = ""
self._current_parameters: Dict[str, Any] = {}
self._streamed_parameters: Dict[str, str] = {}
self._in_tool_call: bool = False
self._function_name_sent: bool = False
```

```python
argument_diff = current_args_json[sent_length:]
calls.append(ToolCallItem(tool_index=self.current_tool_id, name=None, parameters=argument_diff))
```

- Reviewed files: `qwen3_coder_detector.py`, `test/srt/test_function_call_parser.py`.
- Validation impact: use tiny-chunk parser-only tests; model smoke tests will not catch state machine bugs reliably.

### PR #8445 - Qwen3-Coder EBNF separator fix in a GLM-4.5 follow-up

- Link: https://github.com/sgl-project/sglang/pull/8445
- State: merged, `2025-07-28T06:35:20Z`
- Diff coverage: full patch reviewed, `6` files, `+44/-15`.
- Motivation: the PR primarily addressed GLM-4.5 review feedback, but it also fixed Qwen3-Coder's XML parameter grammar by adding a newline separator between `<parameter=...>` blocks.
- Key implementation: `Qwen3CoderDetector.build_ebnf()` passed `key_value_separator="\n"` into the XML grammar builder; tool-choice tests also used larger `max_tokens` to avoid truncation.
- Key snippet:

```python
return EBNFComposer.build_ebnf(
    tools,
    function_format="xml",
    call_rule_fmt='"<function={name}>\\n" {arguments_rule} "\\n</function>"',
    key_value_rule_fmt='"<parameter={key}>\\n" {valrule} "\\n</parameter>"',
    key_value_separator="\\n",
)
```

- Reviewed files: `glm4_moe_detector.py`, `qwen3_coder_detector.py`, `test_tool_choice.py`, `test_function_call_parser.py`.
- Validation impact: multi-parameter constrained generation must check parameter separators, not only single-parameter parsing.

### PR #12226 - Forward unknown tool calls instead of dropping

- Link: https://github.com/sgl-project/sglang/pull/12226
- State: merged, `2025-11-01T02:10:35Z`
- Diff coverage: full patch reviewed, `7` files, `+145/-60`.
- Motivation: models can emit a function name not present in the request's `tools`; silently dropping it hides model intent from client orchestrators.
- Key implementation: added `SGLANG_FORWARD_UNKNOWN_TOOLS`; base, GPT-OSS, Pythonic, and Qwen3-Coder detectors now forward unknown calls when the env is enabled. Qwen3-Coder streaming only resets and flushes invalid names in legacy drop mode.
- Key snippet:

```python
if not (name and name in tool_indices):
    logger.warning(f"Model attempted to call undefined function: {name}")
    if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
        continue
```

```python
if not is_valid:
    logger.warning(f"Invalid function name: {function_name}")
    if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
        self._reset_streaming_state()
        normal += self._buf
        self._buf = ""
        break
```

- Reviewed files: `environ.py`, `base_format_detector.py`, `gpt_oss_detector.py`, `pythonic_detector.py`, `qwen3_coder_detector.py`, `test_unknown_tool_name.py`, `environment_variables.md`.
- Validation impact: test default drop and opt-in forward behavior, including Qwen3-Coder invalid-function streaming state.

### PR #13163 - Remove EBNF Composer

- Link: https://github.com/sgl-project/sglang/pull/13163
- State: merged, `2025-11-13T01:55:31Z`
- Diff coverage: full patch reviewed, `18` files, `+6/-1081`.
- Motivation: SGLang moved required and named `tool_choice` constraints to JSON Schema. Keeping detector-owned EBNF created duplicate constraint systems and made Qwen3-Coder XML grammar brittle.
- Key implementation: deleted `ebnf_composer.py`, removed `build_ebnf()` from all detectors including `Qwen3CoderDetector`, and made JSON Schema constraints the maintained path.
- Key snippet:

```python
elif tool_choice == "required" or isinstance(tool_choice, ToolChoice):
    json_schema = get_json_schema_constraint(self.tools, tool_choice)
    return ("json_schema", json_schema)
```

```diff
-    def build_ebnf(self, tools: List[Tool]):
-        return EBNFComposer.build_ebnf(...)
```

- Reviewed files: `base_format_detector.py`, `function_call_parser.py`, `qwen3_coder_detector.py`, `glm4_moe_detector.py`, `json_array_parser.py`, `test_json_schema_constraint.py`, `test_function_call_parser.py`.
- Validation impact: do not restore Qwen3-Coder EBNF. Test JSON Schema constraints and detector parsing separately.

### PR #13411 - Schema-aware parameter type conversion

- Link: https://github.com/sgl-project/sglang/pull/13411
- State: open as of `2026-04-23`
- Diff coverage: full open patch reviewed, `2` files, `+155/-10`.
- Motivation: `_safe_val()` guessed types with `json.loads()` and `ast.literal_eval()`, so string values such as zip codes or `"42"` could be converted incorrectly. The declared tool schema should decide the type.
- Key implementation: replaces `_safe_val(raw)` with `_convert_param_value(param_value, param_name, param_config, func_name)`, builds tool-parameter schema maps in both streaming and non-streaming paths, and preserves string-typed values even when content looks numeric, boolean, or JSON-like.
- Key snippet:

```python
def _convert_param_value(
    param_value: str, param_name: str, param_config: dict, func_name: str
) -> Any:
    param_value = html.unescape(param_value.strip())
    if param_value.lower() == "null":
        return None
```

```python
self._tool_parameter_configs = {
    tool.function.name: tool.function.parameters.get("properties", {})
    for tool in tools
    if tool.function.name
}
```

- Reviewed files: `qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py`.
- Validation impact: this is open design evidence, not current main behavior. A future merge needs streaming and non-streaming schema-aware tests.

### PR #16744 - New Qwen3-Coder detector

- Link: https://github.com/sgl-project/sglang/pull/16744
- State: merged, `2026-01-19T02:22:41Z`
- Diff coverage: full patch reviewed, `2` files, `+637/-667`.
- Motivation: SGLang needed a production parser for the official Qwen3-Coder XML-like format, with behavior confirmed by the Qwen team.
- Key implementation: rewrote `qwen3_coder_detector.py` around sentinel tokens, function and parameter regexes, conversion helpers, and a cursor-based streaming parser. Parsed calls are emitted as stable-index `ToolCallItem` objects with JSON-serialized parameters.
- Key snippet:

```python
self.tool_call_start_token = "<tool_call>"
self.tool_call_end_token = "</tool_call>"
self.tool_call_prefix = "<function="
self.function_end_token = "</function>"
self.parameter_prefix = "<parameter="
self.parameter_end_token = "</parameter>"
```

```python
calls.append(
    ToolCallItem(
        tool_index=tool_idx,
        name=func_name,
        parameters=json.dumps(parsed_params, ensure_ascii=False),
    )
)
```

- Reviewed files: `qwen3_coder_detector.py`, `test/registered/function_call/test_function_call_parser.py`.
- Validation impact: this is the baseline for Qwen3-Coder parser work. Regressions need one-shot, streaming, multi-parameter, type-conversion, and tool-index coverage.

### PR #21829 - Incremental streaming for tool-call arguments

- Link: https://github.com/sgl-project/sglang/pull/21829
- State: open as of `2026-04-23`
- Diff coverage: full open patch reviewed, `1` file, `+140/-0`.
- Motivation: with `--tool-call-parser qwen3_coder` and streaming enabled, long code or text arguments are buffered until `</parameter>` and then emitted as one huge delta.
- Key implementation: adds active-parameter streaming state, emitted cursor, leading-newline handling, string-like schema gating, and a safe cut-point helper that avoids splitting closing tags.
- Key snippet:

```python
self._streaming_param_active: bool = False
self._streaming_param_emitted: int = 0
self._streaming_param_leading_checked: bool = False
```

```python
return p_type in ("string", "str", "text", "varchar", "char", "enum")
```

- Reviewed files: `qwen3_coder_detector.py`.
- Validation impact: this is open design evidence. If merged, exact OpenAI delta sequence tests are required.

### PR #17965 - Triton TP MoE SwapAB tuning for Qwen3-Coder

- Link: https://github.com/sgl-project/sglang/pull/17965
- State: merged, `2026-01-31T21:57:39Z`
- Diff coverage: full patch reviewed, `12` files, `+765/-13`.
- Motivation: enable SwapAB on H200 and retune TP MoE for `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` with TP=8, EP=2, and EAGLE3.
- Key implementation: added `ep_size` to tuning and remapped global top-k expert ids to local ids when EP is enabled; added H200 FP8 Triton configs for large MoE shapes.
- Key snippet:

```python
if ep_size > 1:
    topk_ids = (topk_ids // ep_size).to(
        device=moe_inputs[k].topk_ids.device,
        dtype=moe_inputs[k].topk_ids.dtype,
    )
```

- Reviewed files: `tuning_fused_moe_triton_sep.py`, H200 Triton MoE config JSONs, `fused_moe_triton_kernels.py`.
- Validation impact: triage MoE regressions with TP and EP together; bad EP remapping can masquerade as a kernel issue.

### PR #18195 - H100 TP=2 fused MoE config for Qwen3-Coder-Next FP8

- Link: https://github.com/sgl-project/sglang/pull/18195
- State: merged, `2026-02-04T19:38:25Z`
- Diff coverage: full patch reviewed, `1` file, `+70/-0`.
- Motivation: Qwen3-Coder-Next-FP8 needed a tuned Triton MoE config on H100 TP=2. The PR reported output throughput `+2.2%`, peak `+7.3%`, median TTFT `-40.8%`, p99 E2E `-9.6%`, and median ITL `-8.2%`.
- Key implementation: added an `E=512,N=256` H100 FP8 config keyed by dtype and block shape, specializing `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, and `num_stages`.
- Key snippet:

```json
"2048": {
  "BLOCK_SIZE_M": 64,
  "BLOCK_SIZE_N": 128,
  "BLOCK_SIZE_K": 128,
  "GROUP_SIZE_M": 16,
  "num_warps": 4,
  "num_stages": 4
}
```

- Reviewed files: the H100 FP8 fused MoE config JSON.
- Validation impact: compare with the same hardware, TP, dtype, and block shape.

### PR #18224 - ModelOpt NVFP4 support for Qwen3-Coder-Next

- Link: https://github.com/sgl-project/sglang/pull/18224
- State: merged, `2026-02-08T06:38:39Z`
- Diff coverage: full patch reviewed, `1` file, `+23/-12`.
- Motivation: `vincentzed-hf/Qwen3-Coder-Next-NVFP4` needed to load through `--quantization modelopt_fp4`. The PR reported GSM8K Platinum accuracy `0.969` and throughput `4610.959 tok/s` on B300.
- Key implementation: passed `quant_config` into Qwen3-Next attention, registered packed-module mapping for `qkv_proj` and `gate_up_proj`, and remapped ModelOpt KV scale names to SGLang fused attention names.
- Key snippet:

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

```python
if name.endswith(".k_proj.k_scale"):
    name = name.replace(".k_proj.k_scale", ".attn.k_scale")
elif name.endswith(".v_proj.v_scale"):
    name = name.replace(".v_proj.v_scale", ".attn.v_scale")
```

- Reviewed files: `python/sglang/srt/models/qwen3_next.py`.
- Validation impact: test fused projection packing and KV-scale remapping together.

### PR #18355 - AMD support for Qwen3-Coder-Next

- Link: https://github.com/sgl-project/sglang/pull/18355
- State: merged, `2026-02-25T00:29:30Z`
- Diff coverage: full patch reviewed, `2` files, `+72/-12`.
- Motivation: enable Qwen3-Coder-Next on AMD GPUs, including non-MTP FP8 KV and MTP flows. AITER needed correct value-head dimension handling for hybrid linear-attention models.
- Key implementation: AITER now derives `v_head_dim` from MLA config, hybrid KV pool metadata, or standard KV buffers. Qwen3-Next dual stream remains CUDA-only, and CuTe DSL GDN import is optional unless explicitly enabled.
- Key snippet:

```python
if self.use_mla:
    self.v_head_dim = model_runner.model_config.v_head_dim
elif model_runner.hybrid_gdn_config is not None or model_runner.kimi_linear_config is not None:
    self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
else:
    self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]
```

- Reviewed files: `aiter_backend.py`, `qwen3_next.py`.
- Validation impact: AMD validation must cover AITER, hybrid GDN state, FP8 KV, and both MTP and non-MTP modes.

### PR #18608 - AMD MI35x tests

- Link: https://github.com/sgl-project/sglang/pull/18608
- State: merged, `2026-03-02T21:52:04Z`
- Diff coverage: full patch reviewed, `2` files, `+246/-0`.
- Motivation: Qwen3-Coder-Next needed registered MI35x accuracy and functionality tests after AMD runtime support landed.
- Key implementation: added nightly MI35x accuracy and stage-c functionality tests. The basic lane uses AITER, large chunked prefill, disabled radix cache, FP8 KV, and TP=8. The MTP lane uses EAGLE flags and skips FP8 KV because gfx950 Triton extend attention lacked that support at the time.
- Key snippet:

```python
other_args=[
    "--attention-backend", "aiter",
    "--chunked-prefill-size", "131072",
    "--disable-radix-cache",
    "--kv-cache-dtype", "fp8_e4m3",
]
```

- Reviewed files: `test_qwen3_coder_next_eval_mi35x.py`, `test_qwen3_coder_next_8gpu.py`.
- Validation impact: run both AMD basic and MTP lanes; keep the FP8-KV skip rationale current.

### PR #18700 - NPU weight transpose fix

- Link: https://github.com/sgl-project/sglang/pull/18700
- State: merged, `2026-02-25T06:02:41Z`
- Diff coverage: full patch reviewed, `2` files, `+7/-9`.
- Motivation: Qwen3-Coder-Next on NPU had duplicate weight transposes between load postprocessing and fused MoE runtime. Hybrid attention also imported CuTe DSL on NPU unless guarded.
- Key implementation: removed redundant `.permute(0, 2, 1)` from NPU fused MoE weight handoff and guarded CuTe DSL import by `is_npu()` plus the explicit CuTe env toggle.
- Key snippet:

```python
weight=[layer.w13_weight],
...
weight=[layer.w2_weight],
```

- Reviewed files: `fused_moe_method_npu.py`, `hybrid_linear_attn_backend.py`.
- Validation impact: inspect actual `w13` and `w2` shapes after load, not only server startup.

### PR #19736 - AITER k/v scale signature fix

- Link: https://github.com/sgl-project/sglang/pull/19736
- State: merged, `2026-03-04T17:20:38Z`
- Diff coverage: full patch reviewed, `1` file, `+4/-0`.
- Motivation: after `extend_attention_fwd()` gained required `k_scale` and `v_scale` parameters, AITER non-MLA target-verify and draft-extend paths were missed, breaking Qwen3-Coder-Next MTP on AMD.
- Key implementation: added default `1.0` k/v scale arguments at the AITER call sites.
- Key snippet:

```python
1.0,  # k_scale
1.0,  # v_scale
layer.scaling,
```

- Reviewed files: `aiter_backend.py`.
- Validation impact: any attention signature change needs AITER MTP target-verify and draft-extend smoke coverage.

### PR #13979 - Qwen3-Coder-480B nightly performance tests

- Link: https://github.com/sgl-project/sglang/pull/13979
- State: open as of `2026-04-23`
- Diff coverage: full open patch reviewed, `3` files, `+288/-171`.
- Motivation: add 8-GPU H200/B200 nightly performance coverage for `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`.
- Key implementation: adds a `NightlyBenchmarkRunner` test with TP=8, EP=8, multithreaded load, batch sizes `[1, 1, 8, 16, 64]`, input length `4096`, output length `512`, and `server_start_timeout=3600`. The workflow hunk temporarily comments unrelated nightly jobs, so treat it as recipe evidence rather than merge-ready CI structure.
- Key snippet:

```python
QWEN3_CODER_480B_MODEL_PATH = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
cls.other_args = [
    "--tp",
    "8",
    "--ep",
    "8",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
]
```

- Reviewed files: `.github/workflows/nightly-test-nvidia.yml`, `test_qwen3_coder_480b_perf.py`, `nightly_utils.py`.
- Validation impact: reuse the benchmark test and timeout hook, but do not disable unrelated nightly jobs.

## Cookbook Evidence

- `sgl-cookbook#86`: Qwen3-Coder-480B-A35B on AMD MI300X.
- `sgl-cookbook#112`: MI325X and MI355X.
- `sgl-cookbook#143`: Qwen3-Coder-Next.
- `sgl-cookbook#174`: NVIDIA B200/GB200.

## Next Work

1. Add parser-only tests for complex schemas, empty names, multiple tools, incremental streaming, unknown tools, and string-looking numeric values.
2. Run Qwen3-Next MTP/cache tests for Qwen3-Coder-Next runtime changes.
3. Cover AMD AITER basic and MTP lanes; inspect NPU MoE weight shapes after load.
4. Cookbook commands should include `--tool-call-parser qwen3_coder` and document parser correctness separately from model performance.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Qwen3 Coder` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-22 | [#8260](https://github.com/sgl-project/sglang/pull/8260) | merged | Preliminary Support for Qwen3XMLDetector | misc | `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py` |
| 2025-07-25 | [#8357](https://github.com/sgl-project/sglang/pull/8357) | merged | [Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector | tests/benchmarks | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py` |
| 2025-07-26 | [#8371](https://github.com/sgl-project/sglang/pull/8371) | merged | Update qwen3_coder_detector.py for streaming | tests/benchmarks | `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/test_function_call_parser.py` |
| 2025-07-28 | [#8445](https://github.com/sgl-project/sglang/pull/8445) | merged | GLM-4.5 Model Support Follow-up | MoE/router, tests/benchmarks | `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/features/test_enable_thinking.py` |
| 2025-10-27 | [#12226](https://github.com/sgl-project/sglang/pull/12226) | merged | Forward unknown tool calls instead of dropping | tests/benchmarks, docs/config | `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py` |
| 2025-11-12 | [#13163](https://github.com/sgl-project/sglang/pull/13163) | merged | Remove EBNF Composer | MoE/router, tests/benchmarks | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py` |
| 2025-11-17 | [#13411](https://github.com/sgl-project/sglang/pull/13411) | open | Improve Qwen3CoderDetector with schema-aware parameter type conversion | tests/benchmarks | `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py` |
| 2025-11-26 | [#13979](https://github.com/sgl-project/sglang/pull/13979) | open | Add Qwen3-Coder-480B to nightly tests | tests/benchmarks | `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py` |
| 2026-01-08 | [#16744](https://github.com/sgl-project/sglang/pull/16744) | merged | support new qwen3_coder_detector | tests/benchmarks | `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/registered/function_call/test_function_call_parser.py` |
| 2026-01-30 | [#17965](https://github.com/sgl-project/sglang/pull/17965) | merged | [Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` |
| 2026-02-03 | [#18195](https://github.com/sgl-project/sglang/pull/18195) | merged | Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2 | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2026-02-04 | [#18224](https://github.com/sgl-project/sglang/pull/18224) | merged | [ModelOPT] Support Qwen 3 Next Coder NVFP4 | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-06 | [#18355](https://github.com/sgl-project/sglang/pull/18355) | merged | [AMD] Support Qwen3-Coder-Next on AMD platform | model wrapper, attention/backend | `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-11 | [#18608](https://github.com/sgl-project/sglang/pull/18608) | merged | [AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU | tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py` |
| 2026-02-12 | [#18700](https://github.com/sgl-project/sglang/pull/18700) | merged | [NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu. | attention/backend, MoE/router, quantization | `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2026-02-16 | [#18882](https://github.com/sgl-project/sglang/pull/18882) | merged | feat: Add FP8 KV cache support for Triton attention backend | attention/backend, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/attention/triton_backend.py`, `test/registered/quant/test_fp8kv_triton.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` |
| 2026-03-03 | [#19736](https://github.com/sgl-project/sglang/pull/19736) | merged | [AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend | attention/backend | `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-04-01 | [#21829](https://github.com/sgl-project/sglang/pull/21829) | open | [Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector | misc | `python/sglang/srt/function_call/qwen3_coder_detector.py` |

### File-level PR diff reading notes

### PR #8260 - Preliminary Support for Qwen3XMLDetector

- Link: https://github.com/sgl-project/sglang/pull/8260
- Status/date: `merged`, created 2025-07-22, merged 2025-07-22; author `yhyang201`.
- Diff scope read: `3` files, `+153/-0`; areas: misc; keywords: kv, spec.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_detector.py` added +150/-0 (150 lines); hunks: +import ast; symbols: _safe_val, Qwen3XMLDetector, __init__, has_tool_call
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunks: from sglang.srt.function_call.llama32_detector import Llama32Detector; class FunctionCallParser:; symbols: FunctionCallParser:, __init__
  - `python/sglang/srt/server_args.py` modified +1/-0 (1 lines); hunks: def add_cli_args(parser: argparse.ArgumentParser):; symbols: add_cli_args
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: kv, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8357 - [Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector

- Link: https://github.com/sgl-project/sglang/pull/8357
- Status/date: `merged`, created 2025-07-25, merged 2025-07-25; author `CatherineSue`.
- Diff scope read: `7` files, `+574/-83`; areas: tests/benchmarks; keywords: spec, kv, doc, test.
- Code diff details:
  - `test/srt/test_function_call_parser.py` modified +455/-0 (455 lines); hunks: from sglang.srt.function_call.llama32_detector import Llama32Detector; def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_qwen3_coder_detector_ebnf
  - `python/sglang/srt/function_call/ebnf_composer.py` modified +95/-63 (158 lines); hunks: -from typing import Literal, Optional; class EBNFComposer:; symbols: EBNFComposer:, EBNFComposer:, get_value_rule, _handle_enum
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` renamed +10/-9 (19 lines); hunks: from sglang.srt.function_call.base_format_detector import BaseFormatDetector; def _safe_val(raw: str) -> Any:; symbols: _safe_val, Qwen3XMLDetector, Qwen3CoderDetector, _parse_block
  - `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-5 (9 lines); hunks: from sglang.srt.function_call.base_format_detector import BaseFormatDetector; def _get_parameter_value(self, val):; symbols: _get_parameter_value, structure_info, info, supports_structural_tag
  - `python/sglang/srt/function_call/function_call_parser.py` modified +4/-4 (8 lines); hunks: from sglang.srt.function_call.llama32_detector import Llama32Detector; class FunctionCallParser:; symbols: FunctionCallParser:, __init__, get_structure_constraint
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py`; keywords observed in patches: spec, kv, doc, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8371 - Update qwen3_coder_detector.py for streaming

- Link: https://github.com/sgl-project/sglang/pull/8371
- Status/date: `merged`, created 2025-07-26, merged 2025-08-08; author `maocheng23`.
- Diff scope read: `2` files, `+348/-67`; areas: tests/benchmarks; keywords: moe, test.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +219/-9 (228 lines); hunks: def __init__(self):; def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment, _parse_and_stream_parameters
  - `test/srt/test_function_call_parser.py` modified +129/-58 (187 lines); hunks: def test_parse_streaming_simple(self):; def test_parse_streaming_incomplete(self):; symbols: test_parse_streaming_simple, test_parse_streaming_incomplete, test_edge_case_no_parameters, test_extract_tool_calls_type_conversion
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/test_function_call_parser.py`; keywords observed in patches: moe, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/test_function_call_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8445 - GLM-4.5 Model Support Follow-up

- Link: https://github.com/sgl-project/sglang/pull/8445
- Status/date: `merged`, created 2025-07-28, merged 2025-07-28; author `byjiang1996`.
- Diff scope read: `6` files, `+44/-15`; areas: MoE/router, tests/benchmarks; keywords: test, moe, spec.
- Code diff details:
  - `test/srt/openai_server/function_call/test_tool_choice.py` modified +39/-10 (49 lines); hunks: def get_test_messages(self):; def test_tool_choice_auto_non_streaming(self):; symbols: get_test_messages, test_tool_choice_auto_non_streaming, test_tool_choice_auto_streaming, test_tool_choice_required_non_streaming
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +1/-2 (3 lines); hunks: def build_ebnf(self, tools: List[Tool]):; symbols: build_ebnf
  - `test/srt/openai_server/features/test_enable_thinking.py` modified +1/-1 (2 lines); hunks: def test_stream_chat_completion_without_reasoning(self):; symbols: test_stream_chat_completion_without_reasoning, TestGLM45EnableThinking, setUpClass
  - `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +1/-1 (2 lines); hunks: def test_pythonic_tool_call_streaming(self):; symbols: test_pythonic_tool_call_streaming, TestGLM45ServerFunctionCalling, setUpClass
  - `test/srt/test_function_call_parser.py` modified +1/-1 (2 lines); hunks: def test_streaming_multiple_tool_calls(self):; symbols: test_streaming_multiple_tool_calls, test_tool_call_completion, test_tool_call_id
- Optimization/support interpretation: The concrete diff surface is `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/features/test_enable_thinking.py`; keywords observed in patches: test, moe, spec. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/features/test_enable_thinking.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12226 - Forward unknown tool calls instead of dropping

- Link: https://github.com/sgl-project/sglang/pull/12226
- Status/date: `merged`, created 2025-10-27, merged 2025-11-01; author `Surya-Gunukula`.
- Diff scope read: `7` files, `+145/-60`; areas: tests/benchmarks, docs/config; keywords: config, cache, doc, expert, test.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +41/-37 (78 lines); hunks: from typing import Any, Dict, List, Tuple; def parse_streaming_increment(; symbols: parse_streaming_increment
  - `test/srt/function_call/test_unknown_tool_name.py` added +69/-0 (69 lines); hunks: +import json; symbols: DummyDetector, has_tool_call, detect_and_parse, test_unknown_tool_name_dropped_default
  - `python/sglang/srt/function_call/base_format_detector.py` modified +15/-12 (27 lines); hunks: from partial_json_parser.core.options import Allow; def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:; symbols: parse_base_json
  - `docs/references/environment_variables.md` modified +10/-9 (19 lines); hunks: SGLang supports various environment variables that can be used to configure its
  - `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-1 (5 lines); hunks: from typing import List, Optional; def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult; symbols: detect_and_parse
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py`; keywords observed in patches: config, cache, doc, expert, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13163 - Remove EBNF Composer

- Link: https://github.com/sgl-project/sglang/pull/13163
- Status/date: `merged`, created 2025-11-12, merged 2025-11-13; author `TJ5`.
- Diff scope read: `18` files, `+6/-1081`; areas: MoE/router, tests/benchmarks; keywords: spec, kv, moe, test, doc.
- Code diff details:
  - `test/srt/test_function_call_parser.py` modified +5/-459 (464 lines); hunks: import json; def test_detect_and_parse_with_text_before_tool_call(self):; symbols: test_detect_and_parse_with_text_before_tool_call, TestEBNFGeneration, setUp, test_pythonic_detector_ebnf
  - `python/sglang/srt/function_call/ebnf_composer.py` removed +0/-344 (344 lines); hunks: -from typing import Any, Dict, Literal, Optional; symbols: EBNFComposer:, get_value_rule, _handle_enum, format_enum_val
  - `test/srt/function_call/test_json_schema_constraint.py` modified +0/-52 (52 lines); hunks: def test_tools_without_parameters(self):; symbols: test_tools_without_parameters, test_json_schema_vs_ebnf_constraint_generation, test_conflicting_defs_raises_valueerror
  - `python/sglang/srt/function_call/function_call_parser.py` modified +0/-38 (38 lines); hunks: def get_structure_constraint(; symbols: get_structure_constraint, get_ebnf
  - `python/sglang/srt/function_call/step3_detector.py` modified +0/-29 (29 lines); hunks: ToolCallItem,; def supports_structural_tag(self) -> bool:; symbols: supports_structural_tag, structure_info, build_ebnf
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py`; keywords observed in patches: spec, kv, moe, test, doc. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13411 - Improve Qwen3CoderDetector with schema-aware parameter type conversion

- Link: https://github.com/sgl-project/sglang/pull/13411
- Status/date: `open`, created 2025-11-17; author `00INDEX`.
- Diff scope read: `2` files, `+155/-10`; areas: tests/benchmarks; keywords: config, test.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +135/-10 (145 lines); hunks: logger = logging.getLogger(__name__); def parse_streaming_increment(; symbols: _safe_val, _convert_param_value, Qwen3CoderDetector, parse_streaming_increment
  - `test/per_commit/function_call/test_function_call_parser.py` modified +20/-0 (20 lines); hunks: def test_extract_tool_calls_type_conversion(self):; def test_extract_tool_calls_type_conversion(self):; symbols: test_extract_tool_calls_type_conversion, test_extract_tool_calls_type_conversion, test_extract_tool_calls_type_conversion, test_parse_streaming_incremental
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py`; keywords observed in patches: config, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13979 - Add Qwen3-Coder-480B to nightly tests

- Link: https://github.com/sgl-project/sglang/pull/13979
- Status/date: `open`, created 2025-11-26; author `Kangyan-Zhou`.
- Diff scope read: `3` files, `+288/-171`; areas: tests/benchmarks; keywords: test, benchmark, config, fp8, moe.
- Code diff details:
  - `.github/workflows/nightly-test-nvidia.yml` modified +232/-170 (402 lines); hunks: jobs:; jobs:
  - `test/nightly/test_qwen3_coder_480b_perf.py` added +53/-0 (53 lines); hunks: +import unittest; symbols: TestNightlyQwen3Coder480BPerformance, setUpClass, test_bench_one_batch
  - `test/nightly/nightly_utils.py` modified +3/-1 (4 lines); hunks: def run_benchmark_for_model(; def run_benchmark_for_model(; symbols: run_benchmark_for_model, run_benchmark_for_model, run_benchmark_for_model
- Optimization/support interpretation: The concrete diff surface is `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py`; keywords observed in patches: test, benchmark, config, fp8, moe. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16744 - support new qwen3_coder_detector

- Link: https://github.com/sgl-project/sglang/pull/16744
- Status/date: `merged`, created 2026-01-08, merged 2026-01-19; author `attack204`.
- Diff scope read: `2` files, `+637/-667`; areas: tests/benchmarks; keywords: config, moe, spec, test.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +392/-271 (663 lines); hunks: import ast; logger = logging.getLogger(__name__); symbols: _safe_val, Qwen3CoderDetector, __init__, already
  - `test/registered/function_call/test_function_call_parser.py` modified +245/-396 (641 lines); hunks: def test_streaming_no_parameters_with_whitespace(self):; symbols: test_streaming_no_parameters_with_whitespace, TestQwen3CoderDetector, setUp, test_has_tool_call
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/registered/function_call/test_function_call_parser.py`; keywords observed in patches: config, moe, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/registered/function_call/test_function_call_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17965 - [Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB

- Link: https://github.com/sgl-project/sglang/pull/17965
- Status/date: `merged`, created 2026-01-30, merged 2026-01-31; author `b8zhong`.
- Diff scope read: `6` files, `+573/-16`; areas: MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: moe, triton, config, fp8, cuda, benchmark, cache, expert, quant, topk.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +128/-0 (128 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +114/-0 (114 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` modified +4/-16 (20 lines); hunks: import triton; from sglang.srt.utils import (; symbols: support_tensor_descriptor, should_enable_swap_ab, is_h20_device_and_sm90_supported
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`; keywords observed in patches: moe, triton, config, fp8, cuda, benchmark. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18195 - Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2

- Link: https://github.com/sgl-project/sglang/pull/18195
- Status/date: `merged`, created 2026-02-03, merged 2026-02-04; author `mmangkad`.
- Diff scope read: `1` files, `+146/-0`; areas: MoE/router, quantization, kernel, docs/config; keywords: config, fp8, moe, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`; keywords observed in patches: config, fp8, moe, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18224 - [ModelOPT] Support Qwen 3 Next Coder NVFP4

- Link: https://github.com/sgl-project/sglang/pull/18224
- Status/date: `merged`, created 2026-02-04, merged 2026-02-08; author `vincentzed`.
- Diff scope read: `1` files, `+35/-6`; areas: model wrapper; keywords: cache, config, expert, fp8, kv, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunks: def __init__(; class HybridLayerType(enum.Enum):; symbols: __init__, HybridLayerType, Qwen3NextForCausalLM, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: cache, config, expert, fp8, kv, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18355 - [AMD] Support Qwen3-Coder-Next on AMD platform

- Link: https://github.com/sgl-project/sglang/pull/18355
- Status/date: `merged`, created 2026-02-06, merged 2026-02-25; author `yichiche`.
- Diff scope read: `2` files, `+213/-74`; areas: model wrapper, attention/backend; keywords: cuda, attention, cache, config, flash, kv, mla, spec, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunks: class ForwardMetadata:; def __init__(; symbols: ForwardMetadata:, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: cuda, attention, cache, config, flash, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18608 - [AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU

- Link: https://github.com/sgl-project/sglang/pull/18608
- Status/date: `merged`, created 2026-02-11, merged 2026-03-02; author `yichiche`.
- Diff scope read: `2` files, `+486/-0`; areas: tests/benchmarks; keywords: attention, cache, config, eagle, fp8, kv, spec, test, topk, triton.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` added +302/-0 (302 lines); hunks: +"""MI35x Qwen3-Coder-Next GSM8K Completion Evaluation Test (8-GPU); symbols: get_model_path, ModelConfig:, __post_init__, get_display_name
  - `test/registered/amd/test_qwen3_coder_next_8gpu.py` added +184/-0 (184 lines); hunks: +"""MI35x Qwen3-Coder-Next Functionality Test (8-GPU); symbols: TestQwen3CoderNext, setUpClass, tearDownClass, test_a_gsm8k
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`; keywords observed in patches: attention, cache, config, eagle, fp8, kv. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18700 - [NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu.

- Link: https://github.com/sgl-project/sglang/pull/18700
- Status/date: `merged`, created 2026-02-12, merged 2026-02-25; author `Hexq0210`.
- Diff scope read: `2` files, `+3/-3`; areas: attention/backend, MoE/router, quantization; keywords: attention, cuda, moe, quant.
- Code diff details:
  - `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py` modified +2/-2 (4 lines); hunks: def npu_fused_moe_without_routing_weights_bf16(; def npu_fused_moe_without_routing_weights_bf16(; symbols: npu_fused_moe_without_routing_weights_bf16, npu_fused_moe_without_routing_weights_bf16
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +1/-1 (2 lines); hunks: from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_npu
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; keywords observed in patches: attention, cuda, moe, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18882 - feat: Add FP8 KV cache support for Triton attention backend

- Link: https://github.com/sgl-project/sglang/pull/18882
- Status/date: `merged`, created 2026-02-16, merged 2026-03-03; author `zack041`.
- Diff scope read: `6` files, `+180/-27`; areas: attention/backend, quantization, kernel, tests/benchmarks; keywords: attention, kv, triton, test, cache, quant, config, cuda, flash, fp8.
- Code diff details:
  - `python/sglang/srt/layers/attention/triton_backend.py` modified +63/-6 (69 lines); hunks: import triton; def __init__(; symbols: __init__, forward_extend, forward_extend, forward_extend
  - `test/registered/quant/test_fp8kv_triton.py` added +58/-0 (58 lines); hunks: +import unittest; symbols: TestFP8KVCacheTritonBackend, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +26/-15 (41 lines); hunks: def _fwd_kernel_stage1(; def _fwd_kernel_stage1(; symbols: _fwd_kernel_stage1, _fwd_kernel_stage1, _decode_att_m_fwd, _decode_att_m_fwd
  - `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +16/-6 (22 lines); hunks: def _fwd_kernel(; def _fwd_kernel(; symbols: _fwd_kernel, _fwd_kernel, _fwd_kernel, extend_attention_fwd
  - `test/registered/attention/test_triton_attention_kernels.py` modified +14/-0 (14 lines); hunks: def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):; def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):; symbols: _test_extend_attention_once, _test_extend_attention_once, _test_extend_attention_sliding_window_once, _test_decode_attention_once
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/triton_backend.py`, `test/registered/quant/test_fp8kv_triton.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`; keywords observed in patches: attention, kv, triton, test, cache, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/triton_backend.py`, `test/registered/quant/test_fp8kv_triton.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19736 - [AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend

- Link: https://github.com/sgl-project/sglang/pull/19736
- Status/date: `merged`, created 2026-03-03, merged 2026-03-04; author `michaelzhang-ai`.
- Diff scope read: `1` files, `+2/-0`; areas: attention/backend; keywords: attention.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-0 (2 lines); hunks: def forward_extend(; symbols: forward_extend
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/aiter_backend.py`; keywords observed in patches: attention. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/aiter_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21829 - [Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector

- Link: https://github.com/sgl-project/sglang/pull/21829
- Status/date: `open`, created 2026-04-01; author `yunkchen`.
- Diff scope read: `1` files, `+140/-0`; areas: misc; keywords: config.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +140/-0 (140 lines); hunks: def __init__(self):; def _convert_param_value(; symbols: __init__, has_tool_call, _convert_param_value, _should_stream_param
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/qwen3_coder_detector.py`; keywords observed in patches: config. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/qwen3_coder_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 18; open PRs: 3.
- Open PRs to keep tracking: [#13411](https://github.com/sgl-project/sglang/pull/13411), [#13979](https://github.com/sgl-project/sglang/pull/13979), [#21829](https://github.com/sgl-project/sglang/pull/21829)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
