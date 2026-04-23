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
