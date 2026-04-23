# Qwen3-Coder PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched paths: `qwen3_coder_detector.py`, Qwen3-Coder docs/snippets, Qwen3-Next runtime files, AMD/NPU Coder tests.
- Searched PR terms: `Qwen3-Coder`, `Qwen3 Coder`, `qwen3_coder`, `Qwen3-Coder-Next`, `qwen3-coder-next`, `qwen3 coder detector`.
- Diff review rule: every PR below was manually opened with GitHub metadata and `gh pr diff --patch`; snippets are taken from reviewed hunks, not generated from title lists.

## Runtime Surfaces

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

## Parser and Tool-Call Dossiers

### PR #8357 - XML-ish grammar and Qwen3-Coder detector fixes

- Link: https://github.com/sgl-project/sglang/pull/8357
- State: merged, `2025-07-25T05:08:06Z`
- Diff coverage: reviewed full patch, `6` files, `+305/-58`.
- Motivation: PR #8260 had added Qwen3-style function calling, but required/specific `tool_choice` still failed because `EBNFComposer` had no XML-like grammar for Qwen3-Coder. The detector also had streaming index bugs, structural-tag incompatibility, and an inconsistent registry name.
- Key implementation: added XML format support to `EBNFComposer`, renamed the detector path from a generic Qwen3 name to `qwen3_coder`, and made detectors advertise whether structural tags are safe. Qwen3-Coder and Pythonic formats explicitly return `False`, so server-side structural tag logic does not wrap an already XML-like tool-call stream.
- Key snippet:

```python
BASE_TYPE_MAPPING = {
    "string": "basic_string",
    "number": "basic_number",
    "integer": "basic_number",
    "boolean": "basic_boolean",
    "null": "basic_null",
    "array": "basic_array",
    "object": "basic_object",
}
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
- Validation impact: parser tests must cover XML parameter tags, `tool_choice=required`, specific function selection, streaming index increments, and structural-tag disabled behavior.

### PR #8371 - Qwen3-Coder streaming parser update

- Link: https://github.com/sgl-project/sglang/pull/8371
- State: merged, `2025-08-08T06:42:29Z`
- Diff coverage: reviewed full patch, `2` files, `+304/-54`.
- Motivation: the old Qwen3-Coder parser buffered too much output and could raise `AttributeError` during streaming, dropping OpenAI-compatible streaming responses. Clients also needed to see the tool name as soon as the `<function=...>` header was complete, not only after all parameters had arrived.
- Key implementation: introduced explicit streaming state for current function name, current parameters, streamed argument lengths, and in-tool-call status. `parse_streaming_increment()` now emits an initial tool-call delta with `name` and empty `parameters`, then emits JSON argument diffs after parameter blocks become parseable.
- Key snippet:

```python
self._current_function_name: str = ""
self._current_parameters: Dict[str, Any] = {}
self._streamed_parameters: Dict[str, str] = {}
self._in_tool_call: bool = False
self._function_name_sent: bool = False
```

```python
calls.append(
    ToolCallItem(
        tool_index=self.current_tool_id,
        name=function_name,
        parameters="",
    )
)
```

```python
argument_diff = current_args_json[sent_length:]
calls.append(
    ToolCallItem(
        tool_index=self.current_tool_id,
        name=None,
        parameters=argument_diff,
    )
)
```

- Reviewed files: `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/test_function_call_parser.py`.
- Validation impact: tiny-chunk streaming tests are mandatory. A model smoke test is not enough because the failure is in parser state transitions, not model logits.

### PR #8445 - GLM-4.5 follow-up that also fixed Qwen3-Coder EBNF separators

- Link: https://github.com/sgl-project/sglang/pull/8445
- State: merged, `2025-07-28T06:35:20Z`
- Diff coverage: reviewed full patch, `6` files, `+44/-15`.
- Motivation: the PR primarily addressed GLM-4.5 review feedback, but it also fixed Qwen3-Coder's XML argument grammar by adding a newline separator between `<parameter=...>` blocks. Without that separator, constrained generation for multiple parameters could force malformed adjacent XML-ish tags.
- Key implementation: `Qwen3CoderDetector.build_ebnf()` passed `key_value_separator="\n"` to the shared XML EBNF builder. GLM-4.5 also re-enabled multi-tool separators and widened tool-choice tests to avoid premature truncation.
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
- Validation impact: when multiple Qwen3-Coder parameters exist, constrained generation tests must assert the newline between parameters and not only check that a single parameter parses.

### PR #12226 - Forward unknown tool calls instead of dropping

- Link: https://github.com/sgl-project/sglang/pull/12226
- State: merged, `2025-11-01T02:10:35Z`
- Diff coverage: reviewed full patch, `7` files, `+145/-60`.
- Motivation: models can emit a function name not present in the request's `tools`. The previous parser behavior silently dropped those calls, causing client orchestrators to lose model intent. The PR made forwarding opt-in so legacy drop behavior stayed stable.
- Key implementation: added `SGLANG_FORWARD_UNKNOWN_TOOLS` to `environ.py`, then taught base, GPT-OSS, Pythonic, and Qwen3-Coder detectors to continue emitting the tool call when the env is true. Qwen3-Coder's streaming parser now logs invalid names but does not reset and flush the whole buffer when forwarding is enabled.
- Key snippet:

```python
if not (name and name in tool_indices):
    logger.warning(f"Model attempted to call undefined function: {name}")
    if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
        continue
```

```python
is_valid = function_name in self._tool_indices
if not is_valid:
    logger.warning(f"Invalid function name: {function_name}")
    if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
        self._reset_streaming_state()
        normal += self._buf
        self._buf = ""
        break
```

- Reviewed files: `environ.py`, `base_format_detector.py`, `gpt_oss_detector.py`, `pythonic_detector.py`, `qwen3_coder_detector.py`, `test_unknown_tool_name.py`, `environment_variables.md`.
- Validation impact: test both default drop behavior and opt-in forwarding. Qwen3-Coder streaming tests need one invalid function name case because the state reset path differs from base JSON parsing.

### PR #13163 - Remove EBNF Composer

- Link: https://github.com/sgl-project/sglang/pull/13163
- State: merged, `2025-11-13T01:55:31Z`
- Diff coverage: reviewed full patch, `18` files, `+6/-1081`.
- Motivation: SGLang moved required and named `tool_choice` constraints to JSON Schema, resolving issue #11078. Keeping per-detector `build_ebnf()` created two independent constraint systems and made Qwen3-Coder XML grammar maintenance brittle.
- Key implementation: deleted `python/sglang/srt/function_call/ebnf_composer.py`, removed `build_ebnf()` from all function-call detectors including `Qwen3CoderDetector`, and deleted EBNF-focused tests. `FunctionCallParser.get_structure_constraint()` remains the entry point and returns JSON Schema constraints for required or named tool choices.
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
- Validation impact: do not add new `build_ebnf()` logic for Qwen3-Coder. New parser work should test JSON Schema constraints and detector parsing separately.

### PR #13411 - Schema-aware Qwen3-Coder parameter conversion

- Link: https://github.com/sgl-project/sglang/pull/13411
- State: open as of `2026-04-23`
- Diff coverage: reviewed full open patch, `2` files, `+155/-10`.
- Motivation: Qwen3-Coder tool-call values arrive as XML-ish text. The previous `_safe_val()` guessed types with `json.loads()` and `ast.literal_eval()`, so a string such as zip code `03106` could be misread or degraded. The PR proposes using the declared tool schema as the source of truth.
- Key implementation: replaces `_safe_val(raw)` with `_convert_param_value(param_value, param_name, param_config, func_name)`, builds a tool-to-parameter-schema map in both streaming and non-streaming paths, and preserves string-typed values even when their content looks like ints, floats, booleans, or JSON objects.
- Key snippet:

```python
def _convert_param_value(
    param_value: str, param_name: str, param_config: dict, func_name: str
) -> Any:
    param_value = html.unescape(param_value.strip())
    if param_value.lower() == "null":
        return None
    if param_name not in param_config:
        return param_value
```

```python
self._tool_parameter_configs = {
    tool.function.name: tool.function.parameters.get("properties", {})
    for tool in tools
    if tool.function.name
}
```

```python
self.assertEqual(params["str_param_int_content"], "42")
self.assertEqual(params["str_param_float_content"], "3.14")
self.assertEqual(params["str_param_bool_content"], "true")
self.assertEqual(params["str_param_obj_content"], '{"key": "value"}')
```

- Reviewed files: `qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py`.
- Validation impact: this is not merged. Treat it as design evidence for future work, not current SGLang behavior. Any follow-up must add schema-aware tests for string-looking numeric/JSON values in both streaming and non-streaming parser paths.

### PR #16744 - New Qwen3-Coder detector implementation

- Link: https://github.com/sgl-project/sglang/pull/16744
- State: merged, `2026-01-19T02:22:41Z`
- Diff coverage: reviewed full patch, `2` files, `+637/-667`.
- Motivation: SGLang needed a production parser for the official Qwen3-Coder XML-ish tool-call format, and the Qwen team confirmed the behavior. Existing parser coverage did not robustly parse `<tool_call>`, `<function=...>`, and nested `<parameter=...>` blocks under streaming.
- Key implementation: rewrote `qwen3_coder_detector.py` around explicit sentinel tokens, function/parameter regexes, schema-assisted conversion helpers, and a cursor-based streaming parser ported from vLLM semantics. It emits `ToolCallItem` objects with stable tool indices and JSON-serialized parameters after parsing each function block.
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
def _convert_param_value(
    self, param_value: str, param_name: str, param_config: dict, func_name: str
) -> Any:
    if param_value.lower() == "null":
        return None
    ...
    elif param_type in ["boolean", "bool", "binary"]:
        param_value = param_value.lower()
        return param_value == "true"
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

- Reviewed files: `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/registered/function_call/test_function_call_parser.py`.
- Validation impact: this PR is the baseline for all later Qwen3-Coder parser work. Regression tests should include one-shot parsing, streaming incremental parsing, multiple parameters, parameter type conversion, and tool index progression.

### PR #21829 - Incremental streaming for Qwen3-Coder tool-call arguments

- Link: https://github.com/sgl-project/sglang/pull/21829
- State: open as of `2026-04-23`
- Diff coverage: reviewed full open patch, `1` file, `+140/-0`.
- Motivation: with `--tool-call-parser qwen3_coder` and streaming enabled, long code or text arguments are buffered until `</parameter>` arrives, then emitted as one massive delta. This hurts clients that expect incremental tool-call argument chunks.
- Key implementation: adds state for active parameter streaming, emitted cursor, and leading newline handling. It streams only string-like schema types because ints, booleans, arrays, and objects need a complete value for conversion. `_find_safe_emit_end()` avoids cutting the stream in the middle of `</parameter>`, `<parameter=`, or `</function>`.
- Key snippet:

```python
self._streaming_param_active: bool = False
self._streaming_param_emitted: int = 0
self._streaming_param_leading_checked: bool = False
```

```python
def _should_stream_param(self, param_name: str, tools: List[Tool]) -> bool:
    param_config = self._get_arguments_config(self.current_func_name, tools)
    if param_name not in param_config:
        return True
    p_type = str(param_config[param_name]["type"]).strip().lower()
    return p_type in ("string", "str", "text", "varchar", "char", "enum")
```

```python
for tag in [self.parameter_end_token, self.parameter_prefix, self.function_end_token]:
    if tag.startswith(suffix):
        return last_angle
```

- Reviewed files: `python/sglang/srt/function_call/qwen3_coder_detector.py`.
- Validation impact: this is not merged. If adopted, add chunk-level tests that assert exact OpenAI delta sequence: JSON key prefix, escaped partial content, final remaining content, and closing quote.

## Runtime, Quantization, Hardware, and CI Dossiers

### PR #17965 - Triton TP MoE SwapAB tuning for Qwen3-Coder

- Link: https://github.com/sgl-project/sglang/pull/17965
- State: merged, `2026-01-31T21:57:39Z`
- Diff coverage: reviewed full patch, `12` files, `+765/-13`.
- Motivation: enable `SwapAB` on H200 and retune TP MoE performance for DeepSeek V3 and Qwen3-Coder. The Qwen3-Coder target was `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` with TP=8, EP=2, and an EAGLE3 draft model.
- Key implementation: the tuning script gained `ep_size` and remaps global top-k expert ids into local expert ids when EP is active. New H200 FP8 Triton configs were added for large MoE shapes, including Qwen3-Coder's `E=80,N=640` style shape.
- Key snippet:

```python
if ep_size > 1:
    topk_ids = (topk_ids // ep_size).to(
        device=moe_inputs[k].topk_ids.device,
        dtype=moe_inputs[k].topk_ids.dtype,
    )
```

```bash
--model-path Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
--tp-size 8 \
--enable-ep-moe \
--ep-size 2 \
--speculative-algorithm EAGLE3
```

- Reviewed files: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, H200 Triton MoE config JSONs, `fused_moe_triton_kernels.py`.
- Validation impact: Qwen3-Coder MoE perf regressions should be triaged with both TP-only and EP-enabled top-k remapping. Wrong remapping can look like a kernel performance issue but is actually routing to the wrong local expert id.

### PR #18195 - H100 TP=2 fused MoE config for Qwen3-Coder-Next FP8

- Link: https://github.com/sgl-project/sglang/pull/18195
- State: merged, `2026-02-04T19:38:25Z`
- Diff coverage: reviewed full patch, `1` file, `+70/-0`.
- Motivation: Qwen3-Coder-Next-FP8 needed a tuned Triton MoE config on H100 TP=2. The PR body reported output throughput `+2.2%`, peak `+7.3%`, median TTFT `-40.8%`, p99 E2E `-9.6%`, and median ITL `-8.2%`.
- Key implementation: added a device/dtype/block-shape keyed config file for `E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128]`. Config entries specialize `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, and `num_stages` by token count.
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

- Reviewed files: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`.
- Validation impact: performance validation should keep the same hardware, TP, dtype, and block shape; otherwise config wins or losses are not comparable.

### PR #18224 - ModelOpt NVFP4 support for Qwen3-Coder-Next

- Link: https://github.com/sgl-project/sglang/pull/18224
- State: merged, `2026-02-08T06:38:39Z`
- Diff coverage: reviewed full patch, `1` file, `+23/-12`.
- Motivation: `vincentzed-hf/Qwen3-Coder-Next-NVFP4` needed to load through `--quantization modelopt_fp4`. The PR body reported GSM8K Platinum accuracy `0.969` and throughput `4610.959 tok/s` on B300 after the fix.
- Key implementation: passed `quant_config` into Qwen3-Next attention construction, registered packed-module mapping for fused `qkv_proj` and `gate_up_proj`, and remapped ModelOpt FP8 KV scale names from split `k_proj/v_proj` keys to SGLang's fused `attn.k_scale/v_scale` keys.
- Key snippet:

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

```python
if quant_config is not None and hasattr(quant_config, "packed_modules_mapping"):
    quant_config.packed_modules_mapping = self.packed_modules_mapping
```

```python
if name.endswith(".k_proj.k_scale"):
    name = name.replace(".k_proj.k_scale", ".attn.k_scale")
elif name.endswith(".v_proj.v_scale"):
    name = name.replace(".v_proj.v_scale", ".attn.v_scale")
```

- Reviewed files: `python/sglang/srt/models/qwen3_next.py`.
- Validation impact: NVFP4 loading must test fused projection packing and FP8 KV scale remapping together. A checkpoint can load weights yet still produce wrong output if KV scale keys are silently ignored.

### PR #18355 - AMD support for Qwen3-Coder-Next

- Link: https://github.com/sgl-project/sglang/pull/18355
- State: merged, `2026-02-25T00:29:30Z`
- Diff coverage: reviewed full patch, `2` files, `+72/-12`.
- Motivation: enable Qwen3-Coder-Next on AMD GPUs, including non-MTP with FP8 KV cache and MTP flows. The AITER backend needed value-head dimension handling for hybrid linear-attention models.
- Key implementation: AITER computes `v_head_dim` from the right source for MLA, hybrid GDN/Kimi-linear, or standard KV pools. Qwen3-Next dual-stream logic stayed CUDA-only, and CuTe DSL GDN import became optional unless `SGLANG_USE_CUTEDSL_GDN_DECODE=1` explicitly requires it.
- Key snippet:

```python
if self.use_mla:
    self.v_head_dim = model_runner.model_config.v_head_dim
elif model_runner.hybrid_gdn_config is not None or model_runner.kimi_linear_config is not None:
    self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
else:
    self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]
```

```python
alt_stream = torch.cuda.Stream() if _is_cuda else None
```

```python
if use_cutedsl and cutedsl_fused_sigmoid_gating_delta_rule_update is None:
    raise ImportError(...)
```

- Reviewed files: `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`.
- Validation impact: AMD Qwen3-Coder-Next validation must include AITER attention, hybrid GDN state, FP8 KV, and MTP/non-MTP modes. Dual stream should not be assumed available on AMD.

### PR #18608 - AMD MI35x Qwen3-Coder-Next tests

- Link: https://github.com/sgl-project/sglang/pull/18608
- State: merged, `2026-03-02T21:52:04Z`
- Diff coverage: reviewed full patch, `2` files, `+246/-0`.
- Motivation: after AMD runtime support, Qwen3-Coder-Next needed registered MI35x accuracy and functionality tests. The model combines hybrid full attention, GDN, 512-expert MoE, FP8 KV, chunked prefill, and optional MTP, so generic AMD smoke tests were too thin.
- Key implementation: added a nightly MI35x accuracy suite and a stage-c functionality suite. The basic path uses `--attention-backend aiter`, large `--chunked-prefill-size`, `--disable-radix-cache`, `--kv-cache-dtype fp8_e4m3`, and TP=8. The MTP path uses EAGLE flags but skips FP8 KV because Triton `extend_attention` did not support FP8 KV on gfx950 at that time.
- Key snippet:

```python
register_amd_ci(est_time=3600, suite="nightly-amd-8-gpu-mi35x", nightly=True)
```

```python
other_args=[
    "--attention-backend", "aiter",
    "--chunked-prefill-size", "131072",
    "--disable-radix-cache",
    "--kv-cache-dtype", "fp8_e4m3",
]
```

```python
"--speculative-algorithm", "EAGLE",
"--speculative-num-steps", "3",
"--speculative-eagle-topk", "1",
"--speculative-num-draft-tokens", "4",
```

- Reviewed files: `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`.
- Validation impact: if touching AMD Qwen3-Coder-Next, run both basic and MTP lanes. Keep the FP8-KV skip rationale updated as backend support changes.

### PR #18700 - NPU Qwen3-Coder-Next weight transpose fix

- Link: https://github.com/sgl-project/sglang/pull/18700
- State: merged, `2026-02-25T06:02:41Z`
- Diff coverage: reviewed full patch, `2` files, `+7/-9`.
- Motivation: Qwen3-Coder-Next on NPU hit wrong weight shapes because weight-load postprocessing and fused MoE runtime both tried to transpose the same tensors. The hybrid attention file also imported CuTe DSL paths that were not valid on NPU unless explicitly requested.
- Key implementation: removed redundant `.permute(0, 2, 1)` from NPU fused MoE weight handoff and guarded CuTe DSL imports with `is_npu()` plus the explicit CuTe env toggle.
- Key snippet:

```python
weight=[layer.w13_weight],
```

```python
weight=[layer.w2_weight],
```

```python
if not is_npu() or use_cutedsl:
    from sglang.jit_kernel.cutedsl_gdn import ...
```

- Reviewed files: `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`.
- Validation impact: NPU tests must inspect actual loaded tensor shapes for `w13` and `w2`, not only check that the server launches.

### PR #19736 - AITER extend_attention k/v scale fix for Qwen3-Coder-Next MTP

- Link: https://github.com/sgl-project/sglang/pull/19736
- State: merged, `2026-03-04T17:20:38Z`
- Diff coverage: reviewed full patch, `1` file, `+4/-0`.
- Motivation: PR #18882 added required `k_scale` and `v_scale` parameters to `extend_attention_fwd()` and updated Triton backend call sites, but missed AITER backend non-MLA `target_verify` and `draft_extend` paths. Qwen3-Coder-Next MTP on AMD then failed with a missing `v_scale` positional argument.
- Key implementation: added default `1.0` k/v scale arguments at the AITER call sites so non-quantized and default-scale paths match the updated function signature.
- Key snippet:

```python
1.0,  # k_scale
1.0,  # v_scale
layer.scaling,
```

- Reviewed files: `python/sglang/srt/layers/attention/aiter_backend.py`.
- Validation impact: AITER MTP target-verify and draft-extend need a direct smoke test after any `extend_attention_fwd()` signature change. This is a signature compatibility bug, so unit coverage can catch it before a full model run.

### PR #13979 - Qwen3-Coder-480B nightly performance tests

- Link: https://github.com/sgl-project/sglang/pull/13979
- State: open as of `2026-04-23`
- Diff coverage: reviewed full open patch, `3` files, `+288/-171`.
- Motivation: add nightly performance coverage for `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` on 8-GPU H200 and 8-GPU B200 lanes. This is needed because Coder-480B has a distinct MoE/EP footprint from smaller Qwen models.
- Key implementation: adds `test/nightly/test_qwen3_coder_480b_perf.py` using `NightlyBenchmarkRunner`, TP=8, EP=8, multithreaded model loading, batch sizes `[1, 1, 8, 16, 64]`, input len `4096`, output len `512`, and a `server_start_timeout=3600`. The workflow patch temporarily comments other nightly perf jobs while adding Qwen3-Coder runs, so it is not merge-ready as a general CI shape.
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

```python
self.runner.run_benchmark_for_model(
    model_path=self.model,
    batch_sizes=self.batch_sizes,
    input_lens=self.input_lens,
    output_lens=self.output_lens,
    other_args=self.other_args,
    server_start_timeout=3600,
)
```

- Reviewed files: `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py`.
- Validation impact: good benchmark recipe evidence, but do not copy the workflow hunk blindly because it comments out other nightly jobs. Keep the test file pattern and timeout hook, then integrate into CI without disabling unrelated coverage.

## Cookbook Evidence

- sgl-cookbook [#86](https://github.com/sgl-project/sgl-cookbook/pull/86): Qwen3-Coder-480B-A35B AMD MI300X deployment guide.
- sgl-cookbook [#112](https://github.com/sgl-project/sgl-cookbook/pull/112): MI325X and MI355X support.
- sgl-cookbook [#143](https://github.com/sgl-project/sgl-cookbook/pull/143): Qwen3-Coder-Next cookbook.
- sgl-cookbook [#174](https://github.com/sgl-project/sgl-cookbook/pull/174): NVIDIA B200/GB200 support.

## Validation Notes

- Parser-only tests are mandatory; model smoke tests alone miss incremental streaming and unknown-tool forwarding bugs.
- Coder-Next model changes should also run Qwen3-Next MTP/cache tests because the runtime path is shared.
- AMD changes must cover AITER basic and MTP paths; NPU changes must inspect weight layout after load.
- Open PRs are included only as reviewed design evidence and must be labeled open/未合入 in downstream docs.
