# GLM-4.6 / GLM-4.7 Model Optimization PR History

This document records the SGLang PR history for GLM-4.6, GLM-4.7, and GLM-4.7-Flash. Each PR listed here was reviewed through its GitHub diff before this write-up was filled in. Every card includes motivation, key implementation idea, code excerpt, reviewed files, and validation implications.

Evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Manual diff review date: `2026-04-23`
- Related skill: `skills/model-optimization/sglang/sglang-glm46-glm47-optimization`
- Full PR dossier: `skills/model-optimization/sglang/sglang-glm46-glm47-optimization/references/pr-history.md`

## Runtime Surfaces

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

## Main Tracks

- GLM-4.6: shared-expert fusion, dual-stream routed/shared expert overlap, GLM4 XML tool-call streaming.
- GLM-4.7: `glm47` tool parser, `glm45` reasoning parser, NVFP4/FP8/MTP/NextN correctness.
- GLM-4.7-Flash: `glm4_moe_lite`, `Glm4MoeLiteForCausalLM`, Lite config/loading, packed quant modules, no EAGLE implementation, AMD/NPU paths.
- Parser: GLM-4.6, GLM-4.7, and GLM-5 share GLM XML tool/reasoning behavior, so parser PRs frequently affect multiple model families.
- Hardware: AMD AITER FP8, NPU fused attention/QKNorm/RoPE/dual stream, FlashInfer A2A, and Blackwell FP4 need separate validation.

## Merged PRs

### #12456 - Escaped characters in GLM tool calls

- Link: https://github.com/sgl-project/sglang/pull/12456
- State: merged, commit `44da737770e4bcd9bfa27751f0a0751c9b5c06e1`
- Diff: `2` files, `+127/-13`
- Motivation: GLM tool calls may contain literal `\n`, escaped quotes, and JSON values inside `<arg_value>`. The old parser could miss the block and double-serialize array/object arguments.
- Key implementation: first try direct JSON parsing, then wrap the value in a JSON field to unescape JSON escape sequences and parse again. Regexes now accept real newlines and literal `\\n`.
- Code:

```python
wrapped = json.loads('{"tmp": "' + json_value + '"}')
parsed_value = json.loads(wrapped["tmp"])
```

```python
self.func_detail_regex = re.compile(
    r"<tool_call>(.*?)(?:\\n|\n)(.*)</tool_call>", re.DOTALL
)
```

- Reviewed files: `glm4_moe_detector.py`, `test/srt/test_function_call_parser.py`
- Validation: keep escaped JSON, literal newline, paths, and array arguments in GLM parser tests.

### #13786 - Dual-stream GLM MoE GEMM overlap

- Link: https://github.com/sgl-project/sglang/pull/13786
- State: merged, commit `4b45d556a7e66d1d978e6df14098a8ba87606a4b`
- Diff: `1` file, `+47/-3`
- Motivation: GLM-4.6 decode serialized shared-expert and routed-expert GEMMs. The PR reports single-concurrency output speed improving from `60.40` to `66.31 tok/s`, with GSM8K accuracy `0.952`.
- Key implementation: under CUDA graph capture, nonempty batches use `forward_normal_dual_stream()` so shared experts and routed experts can run on different streams before adding shared output back.
- Code:

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

- Reviewed files: `glm4_moe.py`
- Validation: test CUDA graph decode, empty-token guard, output equivalence, and throughput before combining with shared-expert fusion.

### #13873 - GLM-4.6 shared-expert fusion

- Link: https://github.com/sgl-project/sglang/pull/13873
- State: merged, commit `982db4ebac260ef4b0597796541724c81a78fe94`
- Diff: `7` files, `+252/-24`
- Motivation: GLM-4.6 shared experts and routed experts were executed as separate paths, adding GEMMs and synchronization. The LMSYS/Novita production blog later identifies shared-expert fusion as a core GLM4-MoE optimization.
- Key implementation: shared experts become extra fused expert slots after `n_routed_experts`; `num_experts` and `top_k` are increased, and weight loading remaps `mlp.shared_experts` to the fused expert index.
- Code:

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

- Reviewed files: `glm4_moe.py`, fused-MoE config, related tests/docs
- Validation: run logits/accuracy before throughput profiling; keep this toggle separate from dual-stream overlap.

### #13989 - GLM-4.6 streaming tool-call arguments

- Link: https://github.com/sgl-project/sglang/pull/13989
- State: merged, commit `80554598d33b68636be645856fce43403c7be1cb`
- Diff: `2` files, `+527/-81`
- Motivation: GLM-4.6 tool-call arguments were buffered until `</tool_call>`, making streaming responses appear stalled.
- Key implementation: add a streaming state machine, emit the function name first, track raw streamed length, and convert XML arg fragments into JSON increments.
- Code:

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

- Reviewed files: `glm4_moe_detector.py`, parser tests
- Validation: test name-only chunks, argument deltas, complete blocks, malformed partial XML, and multiple tool calls.

### #14585 - GLM-4.6V launch/accuracy fix with shared GLM4-MoE changes

- Link: https://github.com/sgl-project/sglang/pull/14585
- State: merged, commit `cf0478d602ce3259e24bc17a463575484920e166`
- Diff: `12` files, `+308/-29`
- Motivation: GLM-4.6V had accuracy and server-launch failures. The PR is VLM-facing but touches shared GLM4-MoE text paths, shared-expert fusion, and PP/DP behavior.
- Key implementation: add attention bias and video grid fixes for VLM, register GLM4V with FA3 defaults, add GLM thinking-budget tokens, and keep shared-expert weight remapping aligned.
- Code:

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

- Reviewed files: `glm4v.py`, `glm4v_moe.py`, `glm4_moe.py`, GLM docs/tests
- Validation: VLM checks are separate, but shared `glm4_moe.py` changes require GLM-4.6 text MoE regression.

### #14668 - FlashInfer A2A MoE dispatcher

- Link: https://github.com/sgl-project/sglang/pull/14668
- State: merged, commit `2c2c4e446b99c529896b3377b24e1b48b6a52e61`
- Diff: `14` files, `+723/-16`
- Motivation: GLM4-MoE FP4/NVFP4-style serving needed a FlashInfer A2A dispatcher path rather than relying only on generic dispatch.
- Key implementation: add a `flashinfer` dispatcher backend; GLM4-MoE sets EP size to TP size, disables shared-expert fusion for that path, and enables the NVFP4 dispatch env.
- Code:

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

- Reviewed files: MoE token dispatcher, `glm4_moe.py`, server args/env
- Validation: do not combine FlashInfer A2A with shared-expert fusion unless the guard explicitly allows it.

### #15333 - GLM-4.7 tool parser and docs

- Link: https://github.com/sgl-project/sglang/pull/15333
- State: merged, commit `b82c7a0ae7444d4fa5a44185643f7c1cc6f372eb`
- Diff: `7` files, `+809/-394`
- Motivation: GLM-4.7 removed the newline after tool name, so the GLM-4.5/4.6 parser could misparse `<tool_call>name<arg_key>...`.
- Key implementation: add `glm47` parser while keeping GLM-4.7 reasoning on `glm45`.
- Code:

```python
"glm45": Glm4MoeDetector,
"glm47": Glm47MoeDetector,
```

```python
self.func_detail_regex = re.compile(
    r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
)
```

- Reviewed files: `glm47_moe_detector.py`, parser registry, GLM-4.7 docs/snippets, tests
- Validation: GLM-4.7 recipes must include `--tool-call-parser glm47 --reasoning-parser glm45`.

### #15520 - model-gateway GLM-4.7 parser

- Link: https://github.com/sgl-project/sglang/pull/15520
- State: merged, commit `26704c23c056e426c6bc86ea1289e82b5fd37e59`
- Diff: `8` files, `+179/-26`
- Motivation: Rust model-gateway needed the same GLM-4.7 parser split as the Python server.
- Key implementation: register `glm45_moe` and `glm47_moe`, map `glm-4.5*` and `glm-4.6*` to the former, and `glm-4.7*` to the latter.
- Code:

```rust
registry.register_parser("glm45_moe", || Box::new(Glm4MoeParser::glm45()));
registry.register_parser("glm47_moe", || Box::new(Glm4MoeParser::glm47()));
registry.map_model("glm-4.6*", "glm45_moe");
registry.map_model("glm-4.7*", "glm47_moe");
```

```rust
pub fn glm47() -> Self {
    Self::new(r"(?s)<tool_call>\s*([^<\s]+)\s*(.*?)</tool_call>")
}
```

- Reviewed files: Rust GLM parser files, registry, Rust tests
- Validation: mirror Python parser behavior in model-gateway tests.

### #15753 - Complex JSON Schema in GLM detectors

- Link: https://github.com/sgl-project/sglang/pull/15753
- State: merged, commit `8ef5b9052825c2624e3ac91852b16998f6f6ee3c`
- Diff: `4` files, `+869/-20`
- Motivation: real tool schemas include arrays, objects, nullable values, enums, and `anyOf`; scalar-only parsing was insufficient.
- Key implementation: resolve argument type from the declared tool schema before parsing each `<arg_value>`.
- Code:

```python
arg_type = get_argument_type(func_name, arg_key, tools)
parsed_value, is_good_json = parse_arguments(arg_value, arg_type)
```

- Reviewed files: `glm4_moe_detector.py`, `glm47_moe_detector.py`, function-call parser tests
- Validation: GLM-4.7 parser tests need complex schemas, not only string arguments.

### #15754 - Empty function name and None values

- Link: https://github.com/sgl-project/sglang/pull/15754
- State: merged, commit `bc8b526edad7cb0b53658a6d230d4f4f5a1d1949`
- Diff: `4` files, `+1513/-140`
- Motivation: models can emit empty names, invalid names, `None`-like values, and partial XML; the parser should not raise or emit bad tool calls.
- Key implementation: validate function names, safely skip invalid names, and normalize Python/JSON null-like values through common parsing.
- Code:

```python
if not func_name:
    return StreamingParseResult(normal_text=text)
```

```python
if func_name not in tool_indices:
    logger.warning("Invalid tool name ...")
    return StreamingParseResult()
```

- Reviewed files: `glm4_moe_detector.py`, `glm47_moe_detector.py`, parser tests
- Validation: malformed GLM tool-call tests are part of the production contract.

### #17166 - GLM-4.7 NVFP4 and MTP fixes

- Link: https://github.com/sgl-project/sglang/pull/17166
- State: merged, commit `2ff0880a0ed1b81f0dc34e45fbccaa244cf80cf8`
- Diff: `6` files, `+114/-9`
- Motivation: GLM-4.7 FP4/NVFP4 + MTP had draft quantization, `mtp.safetensors`, and Blackwell backend-selection issues.
- Key implementation: preserve compatible CLI/HF quant methods, auto-add `mtp.safetensors` for GLM4-MoE NextN, and select `flashinfer_trtllm` for modelopt FP4 on Blackwell when available.
- Code:

```python
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

- Reviewed files: model config, loader, weight utils, `glm4_moe.py`, server args
- Validation: check MTP weights, draft accept length, and Blackwell backend auto-selection.

### #17247 - GLM-4.7-Flash model support

- Link: https://github.com/sgl-project/sglang/pull/17247
- State: merged, commit `76b06bee03e8d5e5fbd57dfbdbc80688705988ac`
- Diff: `6` files, `+842/-12`
- Motivation: GLM-4.7-Flash uses `Glm4MoeLiteForCausalLM`; SGLang needed a Lite model implementation, MTP/NextN wiring, chat-template compatibility, and shape derivation.
- Key implementation: add `glm4_moe_lite.py`, implement Lite gate/SparseMoeBlock/shared-expert fusion, rewrite Lite draft architecture to NextN, set Lite scaling to `1`.
- Code:

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

- Reviewed files: `glm4_moe_lite.py`, `model_config.py`, server args, serving chat, attention backend
- Validation: GLM-4.7-Flash needs independent BF16, quantized, MTP, parser, and chat-template coverage.

### #19246 - NPU optimize GLM-4.7

- Link: https://github.com/sgl-project/sglang/pull/19246
- State: merged, commit `ad0516d9c1f8235edf594f14b76106dcc8b7e469`
- Diff: `4` files, `+146/-15`
- Motivation: GLM-4.7 on NPU needed better decode performance and draft behavior. PR body reports GSM8K accuracy `0.915`, latency `86.270s`, and output throughput `318.951 tok/s`.
- Key implementation: add NPU shared/routed streams, fuse split+QKNorm+RoPE through `split_qkv_rmsnorm_rope`, and support unquantized speculative draft through temporary BF16 dispatch envs.
- Code:

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

- Reviewed files: NPU utils, ModelSlim RMSNorm, `glm4_moe.py`, `glm4_moe_nextn.py`
- Validation: test NPU fused QKNorm/RoPE, stream sync, MTP draft, and GPU draft quant regressions fixed later by #22823.

### #20543 - Preserve whitespace in GLM tool-call values

- Link: https://github.com/sgl-project/sglang/pull/20543
- State: merged, commit `8eb235ab512528de4c55200c09e2cbc3159a94ba`
- Diff: `3` files, `+66/-2`
- Motivation: tool calls often carry code edits or diffs where indentation matters. `arg_value.strip()` destroyed leading/trailing whitespace.
- Key implementation: remove `arg_value.strip()` from GLM4 and GLM47 detectors while keeping key trimming.
- Code:

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

- Reviewed files: `glm4_moe_detector.py`, `glm47_moe_detector.py`, parser tests
- Validation: exact whitespace preservation is required for agentic coding workloads.

### #21135 - `get_rope_config()` for configs without `rope_parameters`

- Link: https://github.com/sgl-project/sglang/pull/21135
- State: merged, commit `646573e4e8d10c2684e0563bc40915b4bef874f4`
- Diff: `18` files, `+44/-42`
- Motivation: direct `config.rope_parameters["rope_theta"]` broke trust-remote-code configs such as GLM4-MoE.
- Key implementation: use shared `get_rope_config(config)` and fall back partial rotary factor from config.
- Code:

```python
rope_theta, rope_scaling = get_rope_config(config)
partial_rotary_factor = (rope_scaling or {}).get("partial_rotary_factor")
if partial_rotary_factor is None:
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
```

- Reviewed files: `glm4.py`, `glm4_moe.py`, `hf_transformers_utils.py`, mass model fixes
- Validation: config-loading failures after Transformers changes should check the rope path first.

### #21403 - AMD fused RMSNorm + per-token FP8 quant for GLM-4.7-FP8

- Link: https://github.com/sgl-project/sglang/pull/21403
- State: merged, commit `7e4e1dcd7ac85f20e48e442515c352aa201049fb`
- Diff: `3` files, `+149/-13`
- Motivation: AMD GLM-4.7-FP8 had extra global-memory traffic between RMSNorm and per-token FP8 quant. PR reports about `+1%` decode ITL speedup on MI355X TP8.
- Key implementation: communicator supports `quant_format="fp8_per_token"` and calls AITER fused RMSNorm quant; FP8 linear consumes `(q_input, x_scale)` tuple; GLM4-MoE detects channel-strategy compressed-tensors W8A8 FP8.
- Code:

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

- Reviewed files: `communicator.py`, `fp8_utils.py`, `glm4_moe.py`
- Validation: compare accuracy and ITL with fused RMSNorm quant on/off; tuple hidden states are the key correctness risk.

### #21534 - AMD GLM-4.7-FP8 MI35x accuracy CI

- Link: https://github.com/sgl-project/sglang/pull/21534
- State: merged, commit `7078e385ea137e380b091caf41f460444867ba85`
- Diff: `2` files, `+96/-0`
- Motivation: GLM-4.7-FP8 needed a ROCm nightly accuracy gate on MI35x.
- Key implementation: add TP8 GLM-4.7-FP8 job with baseline GSM8K accuracy `0.92` and parser flags.
- Code:

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

- Reviewed files: AMD ROCm workflow, registered AMD test
- Validation: AMD GLM-4.7 changes should watch this gate.

### #21660 - FP32 GLM gate projection

- Link: https://github.com/sgl-project/sglang/pull/21660
- State: merged, commit `ad064c2f4e33e1ad2f5ad50b40bb1ab2fb3e4657`
- Diff: `1` file, `+6/-1`
- Motivation: GLM expert routing is sensitive to gate-logit precision.
- Key implementation: cache FP32 gate weight in a non-persistent buffer and cast hidden states before linear projection.
- Code:

```python
self.register_buffer("_weight_fp32", None, persistent=False)
```

```python
if self._weight_fp32 is None:
    self._weight_fp32 = self.weight.data.to(torch.float32)
logits = F.linear(hidden_states.to(torch.float32), self._weight_fp32, None)
```

- Reviewed files: `glm4_moe.py`
- Validation: runtime gate-weight updates must invalidate `_weight_fp32`; otherwise validate routing-sensitive accuracy.

### #21851 - GLM-4.7 and GLM-4.7-Flash loading/import format

- Link: https://github.com/sgl-project/sglang/pull/21851
- State: merged, commit `b7ae3b5a9a57236c64e513276ab15bbabad4c4e7`
- Diff: `2` files, `+139/-86`
- Motivation: GLM-4.7-Flash has no EAGLE implementation, import comments were stale, and GLM4-MoE had drifted from DeepSeek V2 behavior.
- Key implementation: expand A2A backend guards, use `tp_size=1` shared experts in A2A/FP4 all-gather paths, support AMD gfx942 shared-expert fusion, disable W4AFP8 fusion, use `get_rope_config`, and remove EAGLE-specific Lite logic.
- Code:

```python
dict(tp_rank=0, tp_size=1)
if get_moe_a2a_backend().is_deepep()
or get_moe_a2a_backend().is_flashinfer()
or should_use_flashinfer_cutlass_moe_fp4_allgather()
else {}
```

```python
rope_theta, rope_scaling = get_rope_config(config)
```

- Reviewed files: `glm4_moe.py`, `glm4_moe_lite.py`
- Validation: GLM-4.7-Flash should not use EAGLE unless a later PR adds it; validate A2A fusion guards per backend.

### #22509 - NPU GLM-4.7-Flash fix

- Link: https://github.com/sgl-project/sglang/pull/22509
- State: merged, commit `92f28e9ba80b81bba9f82a4c0a69dccf81ff581c`
- Diff: `2` files, `+4/-2`
- Motivation: GPU-only imports and AMD-only quant-format attributes caused GLM-4.7-Flash failures on NPU.
- Key implementation: remove module-level `dsv3_router_gemm` import and use `getattr` default for `_gfx95_quant_format`.
- Code:

```diff
-from sgl_kernel import dsv3_router_gemm
```

```diff
-            self._gfx95_quant_format,
+            getattr(self, "_gfx95_quant_format", ""),
```

- Reviewed files: `glm4_moe_lite.py`, `deepseek_v2.py`
- Validation: NPU smoke tests should cover import plus one prefill/decode path with GLM-4.7 parser flags.

### #22720 - `gfx95_quant_format` for GLM-4.7-Flash

- Link: https://github.com/sgl-project/sglang/pull/22720
- State: merged, commit `6b2bf66cd9cd0448b0e9f3af8a54e9e10686fdf2`
- Diff: `1` file, `+2/-0`
- Motivation: `Glm4MoeLiteDecoderLayer` lacked `_gfx95_quant_format`, causing startup failures when the DeepSeek V2 path expected it.
- Key implementation: initialize it before creating the layer communicator.
- Code:

```python
self._gfx95_quant_format = self._detect_gfx95_quant_format()
```

- Reviewed files: `glm4_moe_lite.py`
- Validation: AMD quantized GLM-4.7-Flash startup is the target regression.

### #22823 - Preserve auto-detected `quant_config` for GLM NextN draft

- Link: https://github.com/sgl-project/sglang/pull/22823
- State: merged, commit `28e915b474eba6d132a65b28c8325b1bbc3f572a`
- Diff: `1` file, `+2/-1`
- Motivation: auto-detected compressed-tensors FP8 checkpoints often do not pass explicit `--quantization`; the draft model dropped `quant_config`, loaded BF16, and accept length collapsed to about `1.0`.
- Key implementation: preserve loader-provided `quant_config` even when `speculative_draft_model_quantization` is unset.
- Code:

```python
self.needs_quant_draft = (
    get_global_server_args().speculative_draft_model_quantization is not None
    or quant_config is not None
)
quant_config = quant_config if self.needs_quant_draft else None
```

- Reviewed files: `glm4_moe_nextn.py`
- Validation: GLM-4.7-FP8 and GLM-4.6-FP8 EAGLE/NEXTN tests must check draft quant config and accept length.

## Open PR Radar

### #11951 - WIP GLM-4.6 streaming parser

- Link: https://github.com/sgl-project/sglang/pull/11951
- State: open, `3` files, `+450/-105`
- Motivation: early attempt to stream GLM-4.6 tool-call arguments.
- Key implementation: add `current_tool_name_sent`, partial parsers, and argument diffing in Python/Rust.
- Code:

```python
if not self.current_tool_name_sent:
    self.current_tool_name_sent = True
    calls.append(ToolCallItem(tool_index=tool_id, name=func_name, parameters=""))
```

- Note: merged #13989 is the active implementation; #11951 is design history.

### #17869 - NPU GLM-4.7-Flash support

- Link: https://github.com/sgl-project/sglang/pull/17869
- State: open, `4` files, `+86/-5`
- Motivation: GLM-4.7-Flash was not supported on NPU; PR body reports `81%` accuracy.
- Key implementation: handle `qk_head_dim == v_head_dim` in NPU attention and add Ascend GLM-4.7-Flash GSM8K test.
- Code:

```python
if layer.qk_head_dim == layer.v_head_dim:
    q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
    torch.ops.npu.npu_fused_infer_attention_score(...)
```

- Validation: reconcile with merged #19246/#22509 and open #22801 before copying launch flags.

### #18930 - AMD GLM-4.7 MTP tests

- Link: https://github.com/sgl-project/sglang/pull/18930
- State: open, `2` files, `+120/-1`
- Motivation: GLM-4.7-FP8 speculative decoding on MI300 produced garbage with `spec_accept_rate` near zero.
- Key implementation: add a canary test for TP8 EAGLE, checking GSM8K accuracy, accept rate, and average accept length.
- Code:

```python
self.assertGreater(spec_accept_rate, 0.5)
self.assertGreater(avg_spec_accept_length, 2.0)
```

- Validation: this is a useful AMD MTP canary for the same class fixed by #22823.

### #19040 - `Glm4MoeLiteConfig` and `enable_a2a_moe`

- Link: https://github.com/sgl-project/sglang/pull/19040
- State: open, `4` files, `+52/-0`
- Motivation: `glm4_moe_lite` config loading failed because the model type was not registered, and Lite model lacked `enable_a2a_moe`.
- Key implementation: add `Glm4MoeLiteConfig`, register it, and set `self.enable_a2a_moe = False`.
- Code:

```python
class Glm4MoeLiteConfig(Glm4MoeConfig):
    model_type = "glm4_moe_lite"
```

```python
self.enable_a2a_moe = False
```

- Validation: re-test Flash config loading without `trust_remote_code` and A2A guards if this lands.

### #19106 - GLM4 MoE Lite CompressedTensors/AWQ

- Link: https://github.com/sgl-project/sglang/pull/19106
- State: open, `12` files, `+505/-37`
- Motivation: `GLM-4.7-Flash-REAP-23B-A3B-AWQ-4bit` failed because packed modules lacked `.weight`, and `glm4_moe_lite` got wrong Transformers version guidance.
- Key implementation: guard `.weight` access, dequantize CT WNA16 `kv_b_proj`, add packed module mappings, disable shared-expert fusion for ignored shared experts, and treat Lite as TF>=5.
- Code:

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

- Validation: main open risk for GLM-4.7-Flash compressed-tensors/AWQ.

### #22315 - GLM-4.7-FP8 EAGLE accept length fix attempt

- Link: https://github.com/sgl-project/sglang/pull/22315
- State: open, `1` file, `+7/-5`
- Motivation: #19246 caused GPU GLM-4.7-FP8 draft to lose `quant_config`, reducing accept length to `1.00`.
- Key implementation: only allow the unquant draft override on NPU; GPU keeps draft quantization.
- Code:

```python
self.needs_quant_draft = True
if is_npu():
    self.needs_quant_draft = (
        get_global_server_args().speculative_draft_model_quantization
    )
    quant_config = quant_config if self.needs_quant_draft else None
```

- Note: merged #22823 solves this more generally by preserving loader-provided `quant_config`.

### #22801 - NPU dual-stream / DeepEP for GLM-4.7-Flash

- Link: https://github.com/sgl-project/sglang/pull/22801
- State: open, `2` files, `+14/-3`
- Motivation: GLM-4.7-Flash needs NPU dual-stream and DeepEP support.
- Key implementation: avoid forced FP8 DeepEP dispatch under BF16 env, pass `forward_batch` to Lite gate, and create `alt_stream` when `SGLANG_NPU_USE_MULTI_STREAM` is set.
- Code:

```python
elif not envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
    use_fp8 = True
```

```python
self.alt_stream = (
    torch.cuda.Stream()
    if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
    else None
)
```

- Validation: test NPU GLM-4.7-Flash with and without `SGLANG_NPU_USE_MULTI_STREAM`.

### #23067 - `continue_final_message` kwargs for `Glm45Detector`

- Link: https://github.com/sgl-project/sglang/pull/23067
- State: open, `2` files, `+66/-1`
- Motivation: GLM-4.7 uses `glm45` reasoning parser. `continue_final_message=true` could pass kwargs that `Glm45Detector` did not accept, causing HTTP 500.
- Key implementation: add `continue_final_message` and `previous_content` to `Glm45Detector.__init__` and forward them to the base detector.
- Code:

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

- Validation: GLM-4.7 and GLM-5 parser tests should include `continue_final_message` because both use `glm45` reasoning.

## Recommended Validation Matrix

- GLM-4.6 BF16 with `glm45` tool/reasoning parser.
- GLM-4.6 shared-expert fusion as a standalone toggle.
- GLM-4.6 CUDA graph dual-stream decode.
- GLM-4.7 BF16 with `glm47` tool parser and `glm45` reasoning parser.
- GLM-4.7-FP8 TP8 without MTP.
- GLM-4.7-FP8 TP8 with EAGLE/NEXTN, checking average accept length.
- GLM-4.7 NVFP4/modelopt FP4 on Blackwell, checking `flashinfer_trtllm` auto-selection.
- GLM-4.7-Flash BF16, quantized, and compressed-tensors/AWQ checkpoints.
- AMD MI35x/MI355X FP8 path, checking AITER fused RMSNorm quant.
- NPU GLM-4.7/Flash path, checking fused QKNorm/RoPE, dual stream, and parser flags.
