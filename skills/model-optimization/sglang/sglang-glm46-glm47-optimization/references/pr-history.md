# GLM-4.6/4.7 PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Manual diff review date: `2026-04-23`
- Searched paths: `glm4_moe.py`, `glm4_moe_lite.py`, `glm4_moe_nextn.py`, `glm4_moe_detector.py`, `glm47_moe_detector.py`, `reasoning_parser.py`, GLM docs/snippets, NPU/AMD backend files, MoE dispatchers, quant loaders, model-gateway parsers.
- Searched PR terms: `GLM-4.6`, `GLM46`, `glm46`, `GLM-4.7`, `GLM47`, `glm47`, `GLM-4.7-Flash`, `glm4_moe_lite`, `Glm4MoeLite`, `Glm4MoeForCausalLMNextN`.
- Public blog evidence: LMSYS / Novita, "Optimizing GLM4-MoE for Production: 65% Faster TTFT with SGLang" (`https://www.lmsys.org/blog/2026-01-21-novita-glm4/`) ties this lane to shared-expert fusion, QK-Norm-RoPE fusion, async transfer, suffix decoding, and EAGLE/NEXTN serving flags.

## Runtime and Docs Surfaces

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/function_call/glm4_moe_detector.py`
- `python/sglang/srt/function_call/glm47_moe_detector.py`
- `python/sglang/srt/parser/reasoning_parser.py`
- `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`
- `sgl-router/src/tool_parser/parsers/glm47_moe_parser.rs`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7-Flash.mdx`
- `docs_new/src/snippets/autoregressive/glm-46-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-47-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-47-flash-deployment.jsx`
- `test/registered/8-gpu-models/test_glm_46.py`
- `test/registered/amd/*glm47*`
- `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`

## Diff-Reviewed PR Cards

### PR #12456 - Handle escaped characters in GLM tool call parser

- Link: https://github.com/sgl-project/sglang/pull/12456
- State: merged at `2025-11-05T00:48:15Z`, merge commit `44da737770e4bcd9bfa27751f0a0751c9b5c06e1`
- Diff stats: `2` files, `+127/-13`
- Diff coverage: full diff reviewed.
- Motivation: GLM-4.x tool calls can contain literal escaped `\n`, escaped quotes, and JSON strings inside `<arg_value>`. The original GLM parser expected a real newline after `<tool_call>` and often fell back to treating array/object values as plain strings, which caused double serialization and broken tool calls.
- Key implementation: `parse_arguments()` first tries direct `json.loads`, then wraps the raw value in a temporary JSON object to unescape JSON escape sequences before parsing again, then falls back to `ast.literal_eval`. The parser also compiles regexes that accept either real newlines or literal `\\n` around function name and argument tags.
- Key code excerpts:

```python
parsed_value = json.loads(json_value)
...
wrapped = json.loads('{"tmp": "' + json_value + '"}')
parsed_value = json.loads(wrapped["tmp"])
```

```python
self.func_detail_regex = re.compile(
    r"<tool_call>(.*?)(?:\\n|\n)(.*)</tool_call>", re.DOTALL
)
self.func_arg_regex = re.compile(
    r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)
```

- Reviewed files: `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/test_function_call_parser.py`
- Validation implications: GLM-4.6/4.7 parser changes should keep escaped array/object arguments, Windows paths, literal `\n`, and normal XML output in the same regression suite.

### PR #13786 - Overlap GLM MoE GEMMs in two CUDA streams

- Link: https://github.com/sgl-project/sglang/pull/13786
- State: merged at `2025-11-25T18:41:36Z`, merge commit `4b45d556a7e66d1d978e6df14098a8ba87606a4b`
- Diff stats: `1` file, `+47/-3`
- Diff coverage: full diff reviewed.
- Motivation: GLM-4.6 decode was leaving shared-expert and routed-expert GEMMs serialized in the same stream. The PR body reports single-concurrency output speed improving from `60.40` to `66.31 tok/s` and GSM8K accuracy `0.952`.
- Key implementation: `Glm4MoeSparseMoeBlock.forward()` dispatches to `forward_normal_dual_stream()` during CUDA graph capture when an alternate stream exists and the token batch is nonempty. Shared experts run on one stream while routed experts compute top-k and MoE outputs on the current stream, then the streams synchronize and add the shared output in place.
- Key code excerpts:

```python
if (
    self.alt_stream is not None
    and hidden_states.shape[0] > 0
    and get_is_capture_mode()
):
    return self.forward_normal_dual_stream(...)
```

```python
torch.add(final_hidden_states, shared_output, out=final_hidden_states)
```

- Reviewed files: `python/sglang/srt/models/glm4_moe.py`
- Validation implications: this optimization is capture-mode-sensitive. Validate CUDA graph decode, empty-token guard, output equivalence, and throughput before enabling it together with shared-expert fusion or FP4 all-gather paths.

### PR #13873 - GLM-4.6 shared-expert fusion

- Link: https://github.com/sgl-project/sglang/pull/13873
- State: merged at `2025-12-01T19:24:26Z`, merge commit `982db4ebac260ef4b0597796541724c81a78fe94`
- Diff stats: `7` files, `+252/-24`
- Diff coverage: full diff reviewed.
- Motivation: GLM-4.6 has shared experts plus routed experts. Running them as separate MLP/MoE paths adds extra GEMMs and synchronization. The production blog later cites this shared-expert fusion as a core GLM4-MoE optimization, and the PR body reports MMLU score `0.770` plus latency/ITL improvements.
- Key implementation: SGLang represents shared experts as additional fused expert slots after `config.n_routed_experts`. The fused MoE instance increases `num_experts` and `top_k`, weight loading remaps `mlp.shared_experts` into `mlp.experts.{n_routed_experts}`, and the fused-MoE config lookup is extended for per-channel quant metadata.
- Key code excerpts:

```python
self.experts = get_moe_impl_class(quant_config)(
    num_experts=config.n_routed_experts + self.num_fused_shared_experts,
    num_fused_shared_experts=self.num_fused_shared_experts,
    top_k=self.top_k + self.num_fused_shared_experts,
)
```

```python
name = name.replace(
    "mlp.shared_experts",
    f"mlp.experts.{self.config.n_routed_experts}",
)
```

- Reviewed files: `python/sglang/srt/models/glm4_moe.py`, fused-MoE config helpers, GLM tests/docs touched by the PR.
- Validation implications: fusion must be tested independently from dual-stream overlap. Run logits/accuracy comparisons for BF16 and quantized checkpoints, then profile token throughput with `--enable-shared-experts-fusion`.

### PR #13989 - GLM-4.6 streaming tool-call arguments

- Link: https://github.com/sgl-project/sglang/pull/13989
- State: merged at `2025-12-13T21:48:20Z`, merge commit `80554598d33b68636be645856fce43403c7be1cb`
- Diff stats: `2` files, `+527/-81`
- Diff coverage: full diff reviewed.
- Motivation: GLM-4.6 tool calls were held until the closing `</tool_call>` tag, so users saw function arguments arrive all at once after a long wait instead of progressively streaming.
- Key implementation: `glm4_moe_detector.py` grows a streaming state machine. It emits the function name first, tracks how much raw argument content has already been streamed, converts XML `<arg_key>/<arg_value>` chunks into incremental JSON fragments, and shares the same state reset semantics with complete parsing.
- Key code excerpts:

```python
class StreamState(str, Enum):
    INIT = "INIT"
    BETWEEN = "BETWEEN"
    IN_KEY = "IN_KEY"
    WAITING_VALUE = "WAITING_VALUE"
    IN_VALUE = "IN_VALUE"
```

```python
raw_increment = func_args_raw[self._streamed_raw_length :]
json_increment = self._process_xml_to_json_streaming(
    raw_increment, func_name, tools
)
```

- Reviewed files: `python/sglang/srt/function_call/glm4_moe_detector.py`, parser tests.
- Validation implications: any future parser edit must cover streaming name-only chunks, streaming argument deltas, completed blocks, malformed partial XML, and multiple tool calls.

### PR #14585 - GLM-4.6V launch/accuracy fix with shared GLM4-MoE changes

- Link: https://github.com/sgl-project/sglang/pull/14585
- State: merged at `2025-12-08T15:46:16Z`, merge commit `cf0478d602ce3259e24bc17a463575484920e166`
- Diff stats: `12` files, `+308/-29`
- Diff coverage: GLM4V, GLM4V-MoE, shared GLM4-MoE, docs, logit processor, and FA3 registration hunks reviewed.
- Motivation: GLM-4.6V had accuracy and server-launch failures. Although the visible target is VLM, the PR touches shared GLM4-MoE text code used by GLM-4.6-style MoE paths, including shared-expert fusion and pipeline-parallel behavior.
- Key implementation: the VLM path adds `attn_qkv_bias`, corrects video grid flattening, registers GLM4V for FA3 defaults, and adds GLM4-MoE thinking-budget token IDs. The MoE path adds PP group handling, DP encoder wiring for visual features, `lm_head` only on the final PP rank, and shared-expert fusion weight remap logic.
- Key code excerpts:

```python
class Glm4MoeThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    THINKING_START_TOKEN_ID: int = 151350
    THINKING_END_TOKEN_ID: int = 151351
    NEW_LINE_TOKEN_ID: int = 198
```

```python
if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
    name = name.replace(
        "mlp.shared_experts",
        f"mlp.experts.{self.config.n_routed_experts}",
    )
```

- Reviewed files: `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm4_moe.py`, GLM docs/tests.
- Validation implications: keep VLM-specific checks separate, but when shared GLM4-MoE code changes, also run GLM-4.6 text MoE launch and fusion tests.

### PR #14668 - FlashInfer A2A MoE dispatcher

- Link: https://github.com/sgl-project/sglang/pull/14668
- State: merged at `2026-01-24T04:38:44Z`, merge commit `2c2c4e446b99c529896b3377b24e1b48b6a52e61`
- Diff stats: `14` files, `+723/-16`
- Diff coverage: dispatcher factory, FlashInfer dispatcher, GLM4-MoE server-args/backend selection, and environment knobs reviewed.
- Motivation: GLM4-MoE quantized serving needed a FlashInfer all-to-all MoE dispatcher path, especially for FP4/NVFP4-style token dispatch where the generic path was not the fastest or most compatible route.
- Key implementation: the MoE dispatcher factory adds a `flashinfer` backend. GLM4-MoE treats that backend as a specialized EP/A2A mode, sets EP size to TP size, disables shared-expert fusion where necessary, and enables the NVFP4 dispatch env flag for the dispatcher.
- Key code excerpts:

```python
elif a2a_backend.is_flashinfer():
    return FlashinferDispatcher(...)
```

```python
if self.moe_a2a_backend == "flashinfer":
    self.ep_size = self.tp_size
    self.disable_shared_experts_fusion = True
    envs.SGLANG_MOE_NVFP4_DISPATCH.set(True)
```

- Reviewed files: `python/sglang/srt/layers/moe/token_dispatcher/*`, `python/sglang/srt/models/glm4_moe.py`, server-args/env handling.
- Validation implications: do not combine FlashInfer A2A with shared-expert fusion unless the compatibility guard says it is supported. Validate token dispatch correctness and MoE output shape under TP-sized EP.

### PR #15333 - GLM-4.7 tool parser and docs

- Link: https://github.com/sgl-project/sglang/pull/15333
- State: merged at `2025-12-20T03:24:17Z`, merge commit `b82c7a0ae7444d4fa5a44185643f7c1cc6f372eb`
- Diff stats: `7` files, `+809/-394`
- Diff coverage: full parser, registration, docs, and tests reviewed.
- Motivation: GLM-4.7 changed tool-call format by removing the newline after the tool name. Reusing the GLM-4.5/4.6 parser would misparse `<tool_call>name<arg_key>...` outputs, while reasoning could still use the `glm45` parser.
- Key implementation: SGLang registers a new `glm47` tool-call parser. `glm47_moe_detector.py` accepts function name followed directly by `<arg_key>`, while docs tell users to launch GLM-4.7 with `--tool-call-parser glm47 --reasoning-parser glm45`.
- Key code excerpts:

```python
"glm45": Glm4MoeDetector,
"glm47": Glm47MoeDetector,
```

```python
self.func_detail_regex = re.compile(
    r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
)
```

- Reviewed files: `python/sglang/srt/function_call/glm47_moe_detector.py`, parser registry, GLM-4.7 docs/snippets, tests.
- Validation implications: GLM-4.7 and GLM-4.7-Flash launch recipes must use `glm47` for tools and `glm45` for reasoning. Parser tests should explicitly cover no-newline tool names.

### PR #15520 - model-gateway tool parser for GLM-4.7

- Link: https://github.com/sgl-project/sglang/pull/15520
- State: merged at `2025-12-21T18:25:38Z`, merge commit `26704c23c056e426c6bc86ea1289e82b5fd37e59`
- Diff stats: `8` files, `+179/-26`
- Diff coverage: Rust parser registration, model mapping, parser implementation, and tests reviewed.
- Motivation: the Rust model-gateway path needed the same GLM-4.7 parser split as the Python server. Without it, OpenAI-compatible gateway traffic could use the wrong GLM4-MoE tool-call regex.
- Key implementation: model-gateway registers `glm45_moe` and `glm47_moe` parsers, maps `glm-4.5*` and `glm-4.6*` to `glm45_moe`, maps `glm-4.7*` to `glm47_moe`, and keeps GLM-4.7 reasoning mapped to `glm45`.
- Key code excerpts:

```rust
registry.register_parser("glm45_moe", || Box::new(Glm4MoeParser::glm45()));
registry.register_parser("glm47_moe", || Box::new(Glm4MoeParser::glm47()));
registry.map_model("glm-4.6*", "glm45_moe");
registry.map_model("glm-4.7*", "glm47_moe");
```

```rust
pub fn glm45() -> Self {
    Self::new(r"(?s)<tool_call>([^\n]*)\n(.*)</tool_call>")
}
pub fn glm47() -> Self {
    Self::new(r"(?s)<tool_call>\s*([^<\s]+)\s*(.*?)</tool_call>")
}
```

- Reviewed files: `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `sgl-router/src/tool_parser/parsers/glm47_moe_parser.rs`, registry/model mapping, Rust tests.
- Validation implications: if Python parser behavior changes, mirror tests in model-gateway so hosted GLM-4.7 traffic does not diverge.

### PR #15753 - GLM detector support for complex JSON Schema

- Link: https://github.com/sgl-project/sglang/pull/15753
- State: merged at `2026-01-09T23:23:41Z`, merge commit `8ef5b9052825c2624e3ac91852b16998f6f6ee3c`
- Diff stats: `4` files, `+869/-20`
- Diff coverage: GLM4 and GLM47 detector schema handling plus tests reviewed.
- Motivation: real tool schemas contain arrays, nested objects, `anyOf`, nullable values, enums, and non-string argument types. The earlier parser relied on simple value parsing and could produce incorrect argument JSON under complex schemas.
- Key implementation: the detectors resolve argument type metadata from the declared tool schema before parsing each `<arg_value>`, then apply schema-aware parsing instead of treating everything as a raw string.
- Key code excerpts:

```python
arg_type = get_argument_type(func_name, arg_key, tools)
parsed_value, is_good_json = parse_arguments(arg_value, arg_type)
```

```python
if arg_type != "string":
    parsed_value, is_good_json = parse_arguments(arg_value, arg_type)
```

- Reviewed files: `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, function-call parser tests.
- Validation implications: GLM-4.7 parser tests should include complex schemas, not just scalar string arguments.

### PR #15754 - GLM detector empty function name and None values

- Link: https://github.com/sgl-project/sglang/pull/15754
- State: merged at `2025-12-30T21:42:16Z`, merge commit `bc8b526edad7cb0b53658a6d230d4f4f5a1d1949`
- Diff stats: `4` files, `+1513/-140`
- Diff coverage: GLM4/GLM47 detectors and parser tests reviewed.
- Motivation: tool-call streams can contain empty or malformed function names, `None`-like values, and partially emitted arguments. The prior parser could raise or emit invalid tool calls instead of gracefully skipping bad fragments.
- Key implementation: parser paths now validate the extracted function name against declared tools, handle empty names as non-calls, and normalize Python/JSON null-like values through the shared argument parser.
- Key code excerpts:

```python
if not func_name:
    return StreamingParseResult(normal_text=text)
```

```python
if func_name not in tool_indices:
    logger.warning("Invalid tool name ...")
    return StreamingParseResult()
```

- Reviewed files: `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, function-call parser tests.
- Validation implications: malformed GLM tool-call tests are part of the production parser contract, especially for streaming where incomplete XML is common.

### PR #17166 - GLM-4.7 NVFP4 and MTP fixes

- Link: https://github.com/sgl-project/sglang/pull/17166
- State: merged at `2026-01-21T06:15:27Z`, merge commit `2ff0880a0ed1b81f0dc34e45fbccaa244cf80cf8`
- Diff stats: `6` files, `+114/-9`
- Diff coverage: model config quant verification, loader source, MTP safetensors handling, GLM4-MoE routing, and server-args backend auto-selection reviewed.
- Motivation: GLM-4.7 FP4/NVFP4 plus MTP had three practical failures: draft model quantization could be overridden incorrectly, `mtp.safetensors` existed but was missing from the index, and Blackwell users needed a FlashInfer TensorRT-LLM MoE backend by default for modelopt FP4.
- Key implementation: compatible CLI/HF quant methods are preserved, GLM4-MoE NextN automatically appends `mtp.safetensors` when the HF config needs NextN layers, GLM MoE passes the DeepSeekV3 routing method type, and server args auto-select `flashinfer_trtllm` on SM100 when `modelopt_fp4` and FlashInfer `>=0.6.2` are available.
- Key code excerpts:

```python
is_compatible = (
    self.quantization in compatible_quantization_methods
    and quant_method in compatible_quantization_methods[self.quantization]
)
if is_compatible:
    logger.info("Using CLI-specified quantization ...")
elif self.is_draft_model:
    self.quantization = quant_method
```

```python
if (
    arch in ["Glm4MoeForCausalLM", "Glm4MoeForCausalLMNextN"]
    and num_nextn_layers > 0
):
    return hf_weights_files + [mtp_path]
```

```python
if self.quantization == "modelopt_fp4" and self.moe_runner_backend == "auto":
    if check_pkg_version_at_least("flashinfer-python", "0.6.2"):
        self.moe_runner_backend = "flashinfer_trtllm"
```

- Reviewed files: `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`.
- Validation implications: validate GLM-4.7-FP4/NVFP4 launch, `mtp.safetensors` inclusion, draft acceptance length, and backend selection on Blackwell separately.

### PR #17247 - GLM-4.7-Flash model support

- Link: https://github.com/sgl-project/sglang/pull/17247
- State: merged at `2026-01-20T20:04:43Z`, merge commit `76b06bee03e8d5e5fbd57dfbdbc80688705988ac`
- Diff stats: `6` files, `+842/-12`
- Diff coverage: new GLM4-MoE-Lite model file, model-config draft rewrite, model-shape derivation, serving chat template handling, MHA fast-path guard, and server-args speculative defaults reviewed.
- Motivation: GLM-4.7-Flash uses the `Glm4MoeLiteForCausalLM` architecture and needed an SGLang model implementation, MTP/NextN wiring, chat-template compatibility, and shape derivation before it could serve like the larger GLM-4.7 model.
- Key implementation: GLM4-MoE-Lite reuses DeepSeek MLA/MoE building blocks but defines its own gate, sparse MoE block, shared-expert fusion logic, and `EntryClass`. The model-config path rewrites Lite draft models to `Glm4MoeForCausalLMNextN`, sets Lite scaling to `1`, clears rope scaling, and guards the MHA concat fast path behind power-of-two dimensions.
- Key code excerpts:

```python
if is_draft_model and self.hf_config.architectures[0] in [
    "Glm4MoeForCausalLM",
    "Glm4MoeLiteForCausalLM",
]:
    self.hf_config.architectures[0] = "Glm4MoeForCausalLMNextN"
```

```python
if "Glm4MoeLiteForCausalLM" in self.hf_config.architectures:
    self.scaling = 1
    self.hf_config.rope_scaling = None
```

```python
class Glm4MoeLiteSparseMoeBlock(DeepseekV2MoE):
    self.topk = TopK(
        top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
        output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
    )
```

- Reviewed files: `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`.
- Validation implications: GLM-4.7-Flash needs separate smoke coverage for BF16, quantized, MTP, parser flags, and chat template output.

### PR #19246 - NPU optimize GLM-4.7

- Link: https://github.com/sgl-project/sglang/pull/19246
- State: merged at `2026-04-03T07:44:08Z`, merge commit `ad0516d9c1f8235edf594f14b76106dcc8b7e469`
- Diff stats: `4` files, `+146/-15`
- Diff coverage: NPU stream helpers, ModelSlim RMSNorm path, GLM4-MoE NPU attention/MoE changes, and NextN unquant draft path reviewed.
- Motivation: GLM-4.7 on NPU needed higher throughput and correctness for draft models. The PR body reports GSM8K accuracy `0.915`, latency `86.270s`, and output throughput `318.951 tok/s`.
- Key implementation: NPU utility streams split shared-expert and routed-expert work. GLM4-MoE uses a fused `split_qkv_rmsnorm_rope` op for decode instead of separate split, QK norm, and RoPE ops. NextN supports unquantized speculative draft models by temporarily setting BF16 dispatch-related env vars.
- Key code excerpts:

```python
def process_shared_expert(hidden_states, forward_func):
    stream = get_share_stream()
    if stream is None:
        stream = torch.get_device_module().Stream()
        set_share_stream(stream)
    stream.wait_stream(torch.get_device_module().current_stream())
    with torch.get_device_module().stream(stream):
        shared_output = forward_func(hidden_states)
    return shared_output
```

```python
q, k, v = split_qkv_rmsnorm_rope(
    qkv,
    self.rotary_emb.position_sin,
    self.rotary_emb.position_cos,
    self.q_size,
    self.kv_size,
    self.head_dim,
    eps=self.q_norm.variance_epsilon,
)
```

```python
unquant_patch = {
    "SGLANG_DEEPEP_BF16_DISPATCH": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "0",
}
with temp_set_env(allow_sglang=True, **unquant_patch):
    hidden_states = self.model(input_ids, positions, forward_batch)
```

- Reviewed files: `python/sglang/srt/hardware_backend/npu/utils.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`.
- Validation implications: this is NPU-only optimization. Check NPU decode accuracy, stream synchronization, fused QKNorm/RoPE output, and MTP draft behavior; also re-check GPU draft quantization because later PRs fixed a regression from this change.

### PR #20543 - Do not strip whitespace from GLM tool-call values

- Link: https://github.com/sgl-project/sglang/pull/20543
- State: merged at `2026-04-09T18:14:15Z`, merge commit `8eb235ab512528de4c55200c09e2cbc3159a94ba`
- Diff stats: `3` files, `+66/-2`
- Diff coverage: full diff reviewed.
- Motivation: GLM tool calls are often used to emit code edits, diffs, or structured text where leading indentation matters. Calling `strip()` on `<arg_value>` silently destroyed leading/trailing whitespace.
- Key implementation: remove `arg_value = arg_value.strip()` from both GLM4 and GLM47 detectors, while leaving `arg_key.strip()` intact. Regression tests assert that indented strings survive round-trip in both parsers.
- Key code excerpts:

```diff
 for arg_key, arg_value in pairs:
     arg_key = arg_key.strip()
-    arg_value = arg_value.strip()
     arg_type = get_argument_type(func_name, arg_key, tools)
```

```python
self.assertEqual(params["old_string"], "    indented code")
self.assertEqual(params["new_string"], "        also indented")
```

- Reviewed files: `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `test/registered/unit/function_call/test_function_call_parser.py`.
- Validation implications: parser tests for code-edit tools must preserve whitespace exactly. This is especially important for agentic coding workloads mentioned in the production blog.

### PR #21135 - Use `get_rope_config()` for configs without `rope_parameters`

- Link: https://github.com/sgl-project/sglang/pull/21135
- State: merged at `2026-03-26T18:22:13Z`, merge commit `646573e4e8d10c2684e0563bc40915b4bef874f4`
- Diff stats: `18` files, `+44/-42`
- Diff coverage: GLM4, GLM4-MoE, Qwen, DeepSeek, and other trust-remote-code model hunks reviewed.
- Motivation: a Transformers upgrade changed many models to read `config.rope_parameters["rope_theta"]` directly. Trust-remote-code configs, including GLM4-MoE, may expose `rope_theta`, `rope_scaling`, or partial rotary fields differently, causing startup failures.
- Key implementation: GLM4 and GLM4-MoE now call shared `get_rope_config(config)`. Partial rotary factor is read from `rope_scaling` when available and falls back to `config.partial_rotary_factor`.
- Key code excerpts:

```python
rope_theta, rope_scaling = get_rope_config(config)
partial_rotary_factor = (rope_scaling or {}).get("partial_rotary_factor")
if partial_rotary_factor is None:
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
```

- Reviewed files: `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/utils/hf_transformers_utils.py`, additional model files touched by the mass fix.
- Validation implications: any GLM-4.6/4.7 config-loading failure after Transformers changes should first check the `get_rope_config` path.

### PR #21403 - AMD fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8

- Link: https://github.com/sgl-project/sglang/pull/21403
- State: merged at `2026-04-11T00:04:08Z`, merge commit `7e4e1dcd7ac85f20e48e442515c352aa201049fb`
- Diff stats: `3` files, `+149/-13`
- Diff coverage: communicator, FP8 utility, and GLM4-MoE quant-format detection reviewed.
- Motivation: on AMD GLM-4.7-FP8, RMSNorm followed by dynamic per-token FP8 quantization caused extra global-memory traffic. The PR fuses these operations and reports roughly `+1%` decode ITL speedup on MI355X TP8, with GSM8K moving from `0.948` to `0.943`, within margin.
- Key implementation: `LayerCommunicator.prepare_attn()` accepts `quant_format="fp8_per_token"` and invokes AITER `rmsnorm_quant` / `add_rmsnorm_quant`. `apply_fp8_ptpc_linear()` recognizes tuple input `(q_input, x_scale)` and avoids redundant per-token quantization. GLM4-MoE detects `CompressedTensorsW8A8Fp8` with channel strategy and stores `attn_quant_format`.
- Key code excerpts:

```python
def _fused_rmsnorm_fp8_per_token_quant(...):
    out_fp8 = torch.empty((M, N), dtype=_aiter_fp8_dtype, device=hidden_states.device)
    scale = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
    _aiter_rmsnorm_quant(out_fp8, hidden_states, scale, weight, epsilon, 0)
    return (out_fp8, scale.unsqueeze(1))
```

```python
if isinstance(input, tuple):
    q_input, x_scale = input
    output = aiter.gemm_a8w8_bpreshuffle(
        q_input, weight, x_scale, weight_scale, None, torch.bfloat16
    )
```

```python
if (
    isinstance(scheme, CompressedTensorsW8A8Fp8)
    and scheme.strategy == QuantizationStrategy.CHANNEL
):
    return "fp8_per_token"
```

- Reviewed files: `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/glm4_moe.py`.
- Validation implications: AMD FP8 validation should compare accuracy and ITL with and without fused RMSNorm quant. Tuple hidden-state handling is the key correctness risk.

### PR #21534 - AMD GLM-4.7-FP8 accuracy CI on MI35x

- Link: https://github.com/sgl-project/sglang/pull/21534
- State: merged at `2026-03-28T02:29:37Z`, merge commit `7078e385ea137e380b091caf41f460444867ba85`
- Diff stats: `2` files, `+96/-0`
- Diff coverage: workflow and registered AMD test reviewed.
- Motivation: GLM-4.7-FP8 needed a nightly accuracy gate on AMD MI35x so future ROCm/AITER/GLM MoE changes do not silently regress.
- Key implementation: the ROCm nightly workflow adds an 8-GPU MI35x GLM-4.7-FP8 job. The test runs `zai-org/GLM-4.7-FP8` with TP8 and parser flags `--tool-call-parser=glm47 --reasoning-parser=glm45`, with baseline GSM8K accuracy `0.92`.
- Key code excerpts:

```yaml
- nightly-8-gpu-mi35x-glm47-fp8-rocm720
```

```python
base_args = [
    "--trust-remote-code",
    "--tool-call-parser=glm47",
    "--reasoning-parser=glm45",
]
```

- Reviewed files: `.github/workflows/nightly-test-amd-rocm720.yml`, `test/registered/amd/test_glm47_fp8.py`.
- Validation implications: AMD performance changes should update this test only with an explicit accuracy rationale; parser flags are part of the model contract.

### PR #21660 - Cast GLM gate projection to FP32

- Link: https://github.com/sgl-project/sglang/pull/21660
- State: merged at `2026-03-30T19:25:28Z`, merge commit `ad064c2f4e33e1ad2f5ad50b40bb1ab2fb3e4657`
- Diff stats: `1` file, `+6/-1`
- Diff coverage: full diff reviewed.
- Motivation: GLM routing quality depends on FP32 gate projection. Lower-precision gate logits can perturb expert selection and degrade accuracy; the PR body calls this one feature of `#21258`.
- Key implementation: `Glm4MoeGate` caches a FP32 copy of the gate weight in a non-persistent buffer and casts hidden states to FP32 before the linear projection.
- Key code excerpts:

```python
self.register_buffer("_weight_fp32", None, persistent=False)
```

```python
if self._weight_fp32 is None:
    self._weight_fp32 = self.weight.data.to(torch.float32)
logits = F.linear(hidden_states.to(torch.float32), self._weight_fp32, None)
```

- Reviewed files: `python/sglang/srt/models/glm4_moe.py`
- Validation implications: any runtime that updates gate weights must invalidate `_weight_fp32`; otherwise accuracy tests should focus on routing-sensitive GLM-4.7-FP8 and GLM-4.7-Flash.

### PR #21851 - GLM-4.7 and GLM-4.7-Flash loading/import format

- Link: https://github.com/sgl-project/sglang/pull/21851
- State: merged at `2026-04-04T20:42:02Z`, merge commit `b7ae3b5a9a57236c64e513276ab15bbabad4c4e7`
- Diff stats: `2` files, `+139/-86`
- Diff coverage: GLM4-MoE and GLM4-MoE-Lite hunks reviewed.
- Motivation: GLM-4.7-Flash had no EAGLE implementation, import/comment paths were stale, and `glm4_moe.py` had drifted from `deepseek_v2.py`. The result was fragile loading across Flash/lite, A2A backends, shared-expert fusion, and rope config.
- Key implementation: GLM4-MoE expands A2A backend awareness to include Mori, Ascend FuseEP, and FlashInfer; shared experts run with `tp_size=1` in A2A/FP4-allgather paths; shared-expert fusion allows AMD gfx942 and disables W4AFP8; GLM4-MoE-Lite uses `get_rope_config(config)`, removes EAGLE-specific ignore logic, and no longer pretends GLM-4.7-Flash has EAGLE.
- Key code excerpts:

```python
**(
    dict(tp_rank=0, tp_size=1)
    if get_moe_a2a_backend().is_deepep()
    or get_moe_a2a_backend().is_mooncake()
    or get_moe_a2a_backend().is_nixl()
    or get_moe_a2a_backend().is_mori()
    or get_moe_a2a_backend().is_ascend_fuseep()
    or get_moe_a2a_backend().is_flashinfer()
    or should_use_flashinfer_cutlass_moe_fp4_allgather()
    else {}
)
```

```python
elif self.quant_config and self.quant_config.get_name() == "w4afp8":
    disable_reason = (
        "GLM-4.5 W4AFP8 model uses different quant method for "
        "routed experts and shared experts."
    )
```

```python
rope_theta, rope_scaling = get_rope_config(config)
```

- Reviewed files: `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`.
- Validation implications: GLM-4.7-Flash should not be validated through an EAGLE path unless a later PR explicitly adds one. Check each A2A backend guard before turning on shared-expert fusion.

### PR #22509 - NPU fix for GLM-4.7-Flash

- Link: https://github.com/sgl-project/sglang/pull/22509
- State: merged at `2026-04-22T00:13:19Z`, merge commit `92f28e9ba80b81bba9f82a4c0a69dccf81ff581c`
- Diff stats: `2` files, `+4/-2`
- Diff coverage: full diff reviewed.
- Motivation: GPU-specific GLM-4.7-Flash optimizations caused NPU startup/runtime failures because imports and attributes assumed GPU-only kernels or AMD-only quant-format state.
- Key implementation: remove the module-level `sgl_kernel.dsv3_router_gemm` import from GLM4-MoE-Lite and use a safe default for `_gfx95_quant_format` when calling the DeepSeek V2 attention path.
- Key code excerpts:

```diff
-from sgl_kernel import dsv3_router_gemm
```

```diff
-            self._gfx95_quant_format,
+            getattr(self, "_gfx95_quant_format", ""),
```

- Reviewed files: `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_v2.py`
- Validation implications: NPU smoke tests should import GLM-4.7-Flash without CUDA-only kernels and run one prefill/decode path with parser flags.

### PR #22720 - Detect `gfx95_quant_format` for GLM-4.7-Flash

- Link: https://github.com/sgl-project/sglang/pull/22720
- State: merged at `2026-04-13T16:51:37Z`, merge commit `6b2bf66cd9cd0448b0e9f3af8a54e9e10686fdf2`
- Diff stats: `1` file, `+2/-0`
- Diff coverage: full diff reviewed.
- Motivation: GLM-4.7-Flash failed because `Glm4MoeLiteDecoderLayer` did not define `_gfx95_quant_format`, while the DeepSeek V2 path expected that attribute.
- Key implementation: initialize `_gfx95_quant_format` by calling `_detect_gfx95_quant_format()` before creating the layer communicator.
- Key code excerpt:

```python
self._gfx95_quant_format = self._detect_gfx95_quant_format()
```

- Reviewed files: `python/sglang/srt/models/glm4_moe_lite.py`
- Validation implications: this is a tiny but critical attribute-ordering fix. Regression is immediate startup failure on AMD GLM-4.7-Flash quantized paths.

### PR #22823 - Preserve auto-detected `quant_config` for GLM NextN draft

- Link: https://github.com/sgl-project/sglang/pull/22823
- State: merged at `2026-04-15T20:25:37Z`, merge commit `28e915b474eba6d132a65b28c8325b1bbc3f572a`
- Diff stats: `1` file, `+2/-1`
- Diff coverage: full diff reviewed.
- Motivation: PR `#19246` made `Glm4MoeForCausalLMNextN` gate draft quantization on `server_args.speculative_draft_model_quantization`. For auto-detected compressed-tensors FP8 checkpoints, users often do not pass `--quantization`, so the draft model dropped `quant_config`, loaded as BF16, and speculative acceptance collapsed from about `2.0` to `1.0`. The PR body reports throughput recovering from `489.22 tok/s` to `1018.8 tok/s`.
- Key implementation: treat an incoming loader-provided `quant_config` as proof that the draft should remain quantized, even when `speculative_draft_model_quantization` is not explicitly set.
- Key code excerpt:

```python
self.needs_quant_draft = (
    get_global_server_args().speculative_draft_model_quantization is not None
    or quant_config is not None
)
quant_config = quant_config if self.needs_quant_draft else None
```

- Reviewed files: `python/sglang/srt/models/glm4_moe_nextn.py`
- Validation implications: GLM-4.7-FP8/GLM-4.6-FP8 with EAGLE/NEXTN must check draft quant config and average accept length, not just server startup.

## Open / Unmerged PR Radar

### PR #11951 - WIP GLM-4.6 tool-call streaming parse

- Link: https://github.com/sgl-project/sglang/pull/11951
- State: open
- Diff stats: `3` files, `+450/-105`
- Diff coverage: Python detector, Rust parser, and Rust tests reviewed.
- Motivation: this WIP attempted to solve the same "tool arguments arrive only after complete `</tool_call>`" problem later addressed by the merged GLM parser work.
- Key implementation: it adds `current_tool_name_sent`, partial parser methods, and argument diffing in both Python and Rust. It is useful historical context but should not be treated as the active implementation because the merged detector evolved into a richer state machine.
- Key code excerpt:

```python
if not self.current_tool_name_sent:
    self.current_tool_name_sent = True
    calls.append(ToolCallItem(tool_index=tool_id, name=func_name, parameters=""))
```

- Validation implications: use this PR only as design background. For production, validate against merged `#13989`, `#15333`, `#15753`, `#15754`, and `#20543`.

### PR #17869 - NPU GLM-4.7-Flash support

- Link: https://github.com/sgl-project/sglang/pull/17869
- State: open
- Diff stats: `4` files, `+86/-5`
- Diff coverage: NPU attention backend, NPU MLA module, Ascend test utils, and GLM-4.7-Flash registered test reviewed.
- Motivation: GLM-4.7-Flash was not supported on NPU. The PR body reports `81%` accuracy and provides a launch recipe requiring Transformers 5.0.0 and `--tool-call-parser glm47 --reasoning-parser glm45`.
- Key implementation: NPU forward extend handles the `qk_head_dim == v_head_dim` case by iterating per request and calling `torch.ops.npu.npu_fused_infer_attention_score`. MLA KV buffer comments are extended to GLM-4.7-Flash, and a nightly NPU GSM8K test is added with TP4.
- Key code excerpts:

```python
if layer.qk_head_dim == layer.v_head_dim:
    q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
    ...
    torch.ops.npu.npu_fused_infer_attention_score(...)
```

```python
other_args = [
    "--attention-backend", "ascend",
    "--tp-size", "4",
    "--tool-call-parser", "glm47",
    "--reasoning-parser", "glm45",
]
```

- Validation implications: open NPU work should be reconciled with merged `#19246`, `#22509`, and `#22801` before copying launch flags.

### PR #18930 - AMD MTP tests for GLM-4.7

- Link: https://github.com/sgl-project/sglang/pull/18930
- State: open
- Diff stats: `2` files, `+120/-1`
- Diff coverage: AITER backend guard and AMD registered test reviewed.
- Motivation: GLM-4.7-FP8 MTP/speculative decoding on MI300 produced garbage outputs with `spec_accept_rate` near zero. The PR intentionally adds a failing test to capture the unknown issue.
- Key implementation: the test launches `zai-org/GLM-4.7-FP8` with TP8 and EAGLE settings, then checks GSM8K accuracy, `/get_server_info` average spec accept length, and `/v1/loads?include=spec` accept rate. A small AITER guard restricts MLA page-split kernel metadata to `self.use_mla`.
- Key code excerpts:

```python
other_args = [
    "--tp", "8",
    "--speculative-algorithm", "EAGLE",
    "--speculative-num-steps", "3",
    "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "4",
]
```

```python
self.assertGreater(spec_accept_rate, 0.5)
self.assertGreater(avg_spec_accept_length, 2.0)
```

- Validation implications: keep this as a canary for AMD MTP. The merged `#22823` quant-config fix directly addresses the same accept-length failure mode.

### PR #19040 - `Glm4MoeLiteConfig` and `enable_a2a_moe` for GLM-4.7-Flash

- Link: https://github.com/sgl-project/sglang/pull/19040
- State: open
- Diff stats: `4` files, `+52/-0`
- Diff coverage: config registration, new config class, GLM4-MoE-Lite model init, and HF transformer utils registry reviewed.
- Motivation: GLM-4.7-Flash uses `model_type="glm4_moe_lite"`, which Transformers did not natively register in the expected path. Runtime also hit `AttributeError` because `Glm4MoeLiteModel` bypassed `DeepseekV2Model.__init__` and did not set `enable_a2a_moe`.
- Key implementation: add `Glm4MoeLiteConfig` extending `Glm4MoeConfig`, register it with SGLang's local config registry, and set `self.enable_a2a_moe = False` in `Glm4MoeLiteModel.__init__`.
- Key code excerpts:

```python
class Glm4MoeLiteConfig(Glm4MoeConfig):
    model_type = "glm4_moe_lite"
    def __init__(self, q_lora_rank=768, kv_lora_rank=512, ...):
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        super().__init__(**kwargs)
```

```python
self.enable_a2a_moe = False  # Glm4MoeLite does not use all-to-all MoE dispatch
```

- Validation implications: if this lands, re-test GLM-4.7-Flash without `trust_remote_code`, with and without A2A MoE paths.

### PR #19106 - GLM4 MoE Lite CompressedTensors serving and version checks

- Link: https://github.com/sgl-project/sglang/pull/19106
- State: open
- Diff stats: `12` files, `+505/-37`
- Diff coverage: model config, MLA dispatch, DeepSeek weight loader, GLM4-MoE, GLM4-MoE-Lite, packed mapping, shared-expert fusion guards, and targeted unit tests reviewed.
- Motivation: `GLM-4.7-Flash-REAP-23B-A3B-AWQ-4bit` failed with `AttributeError: 'ReplicatedLinear' object has no attribute 'weight'`, while SGLang also printed an incorrect Transformers downgrade warning for `glm4_moe_lite`.
- Key implementation: guard MLA fused paths when packed quant modules do not expose `.weight`; dequantize compressed-tensors WNA16 `kv_b_proj` for MLA `w_kc`/`w_vc`; add packed module mappings for `fused_qkv_a_proj_with_mqa` and `gate_up_proj`; disable shared-expert fusion when checkpoint quant config marks shared experts ignored/non-quantized; and treat `glm4_moe_lite` as Transformers >=5.
- Key code excerpts:

```python
fused_qkv_a_proj = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
if (
    fused_qkv_a_proj is not None
    and getattr(fused_qkv_a_proj, "weight", None) is not None
    and use_intel_amx_backend(attn)
):
    return AttnForwardMethod.MLA_FUSED_ROPE_CPU
```

```python
packed_modules_mapping = {
    "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

```python
elif (
    self.quant_config is not None
    and hasattr(self.quant_config, "ignore")
    and any(".mlp.shared_experts." in item for item in self.quant_config.ignore)
):
    disable_reason = "Shared experts are marked as ignored/non-quantized ..."
```

- Validation implications: this is the main open compressed-tensors/AWQ GLM-4.7-Flash risk. Validate with regular AWQ and REAP/CT WNA16 checkpoints before merging any adjacent Lite changes.

### PR #22315 - GLM-4.7-FP8 EAGLE accept length draft quant-config fix

- Link: https://github.com/sgl-project/sglang/pull/22315
- State: open
- Diff stats: `1` file, `+7/-5`
- Diff coverage: full diff reviewed.
- Motivation: the NPU unquant draft change from `#19246` broke GPU GLM-4.7-FP8 EAGLE by setting `quant_config=None` when `--speculative-draft-model-quantization` was omitted, making all draft predictions effectively rejected. The PR body reports accept length recovering from `1.00` to around `2.75-3.08`.
- Key implementation: this alternative fix keeps GPU draft quantization always enabled and only applies the unquant draft override under `is_npu()`. The later merged `#22823` solved the same class more generally by preserving loader-provided `quant_config`.
- Key code excerpt:

```python
self.needs_quant_draft = True
if is_npu():
    self.needs_quant_draft = (
        get_global_server_args().speculative_draft_model_quantization
    )
    quant_config = quant_config if self.needs_quant_draft else None
```

- Validation implications: keep as historical context and compare against `#22823`. The invariant is: auto-detected FP8/compressed-tensors draft models must not silently become BF16 on GPU.

### PR #22801 - NPU dual-stream and DeepEP support for GLM-4.7-Flash

- Link: https://github.com/sgl-project/sglang/pull/22801
- State: open
- Diff stats: `2` files, `+14/-3`
- Diff coverage: DeepEP dispatcher and GLM4-MoE-Lite diff reviewed.
- Motivation: GLM-4.7-Flash on NPU needs dual-stream execution and DeepEP integration. The PR body includes accuracy and speed screenshots.
- Key implementation: DeepEP avoids forcing FP8 dispatch when `SGLANG_DEEPEP_BF16_DISPATCH` is set, GLM4-MoE-Lite gate accepts `forward_batch`, and NPU can create `alt_stream` when `SGLANG_NPU_USE_MULTI_STREAM` is enabled.
- Key code excerpts:

```python
elif not envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
    use_fp8 = True
```

```python
def forward(
    self,
    hidden_states,
    gemm_output_zero_allocator: BumpAllocator = None,
    forward_batch: ForwardBatch = None,
):
```

```python
self.alt_stream = (
    torch.cuda.Stream()
    if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
    else None
)
```

- Validation implications: if this lands, run NPU GLM-4.7-Flash with and without `SGLANG_NPU_USE_MULTI_STREAM` and verify DeepEP BF16 dispatch does not regress GPU/AMD paths.

### PR #23067 - Forward `continue_final_message` kwargs in `Glm45Detector`

- Link: https://github.com/sgl-project/sglang/pull/23067
- State: open
- Diff stats: `2` files, `+66/-1`
- Diff coverage: reasoning parser and unit tests reviewed.
- Motivation: `ReasoningParser` passes `continue_final_message` and `previous_content` kwargs when OpenAI chat requests ask to continue a trailing assistant message. `Glm45Detector` did not accept these kwargs, so GLM-4.5/4.6/4.7/GLM-5 requests with `--reasoning-parser glm45` could raise HTTP 500.
- Key implementation: add `continue_final_message` and `previous_content` parameters to `Glm45Detector.__init__` and forward them to `BaseReasoningFormatDetector`.
- Key code excerpt:

```python
def __init__(
    self,
    stream_reasoning: bool = True,
    force_reasoning: bool = False,
    continue_final_message: bool = False,
    previous_content: str = "",
):
    super().__init__(
        "<think>",
        "</think>",
        continue_final_message=continue_final_message,
        previous_content=previous_content,
    )
```

- Validation implications: GLM-4.7 uses `--reasoning-parser glm45`; continue-final-message tests belong in GLM-4.7 and GLM-5 parser validation even though the class name says GLM45.

## Production Optimization Evidence

The LMSYS / Novita production blog reports GLM4-MoE production optimizations around:

- shared-expert fusion: upstream reference `#13873`
- QK-Norm-RoPE fusion: cited as SGLang PR `#15141` in the blog
- async transfer: cited as SGLang PR `#14782` in the blog
- suffix decoding: optional speculative path for repeated agentic coding patterns
- EAGLE/NEXTN flags: `--speculative-algorithm NEXTN`, `--speculative-num-steps 3`, `--speculative-eagle-topk 1`, `--speculative-num-draft-tokens 4`

Treat these as separate toggles. Do not attribute a speedup to "GLM-4.7" in general unless the benchmark matrix states which toggles, hardware, quantization, TP/EP, and parser flags were enabled.

## Completeness Notes

- `#11951` and `#22315` are open alternatives or earlier attempts; the currently merged behavior is primarily `#13989` for GLM-4.6 parser streaming and `#22823` for draft quant preservation.
- `#17869`, `#19040`, `#19106`, and `#22801` are open GLM-4.7-Flash/NPU/CompressedTensors tracks. Keep them in the watchlist because they can change launch recipes and validation gates after merge.
- GLM-4.7-Flash is not the same as GLM-4.7 full MoE. Flash/lite changes must mention `glm4_moe_lite`, `Glm4MoeLiteForCausalLM`, or the GLM-4.7-Flash docs/tests.
- Parser changes often affect GLM-4.5, GLM-4.6, GLM-4.7, and GLM-5 together because they share `glm45` reasoning and GLM XML tool-call conventions.
