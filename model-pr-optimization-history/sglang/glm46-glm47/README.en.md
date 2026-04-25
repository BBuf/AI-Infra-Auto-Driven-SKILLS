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

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `GLM-4.6 / GLM-4.7` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-10-22 | [#11951](https://github.com/sgl-project/sglang/pull/11951) | open | WIP: Fix glm-4.6 tool call streaming parse | MoE/router, tests/benchmarks | `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs` |
| 2025-10-31 | [#12456](https://github.com/sgl-project/sglang/pull/12456) | merged | [fix] Handle escaped characters in GLM tool call parser to prevent double serialization | MoE/router, tests/benchmarks | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-11-23 | [#13786](https://github.com/sgl-project/sglang/pull/13786) | merged | Overlap glm moe gemms in two cuda streams | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-11-25 | [#13873](https://github.com/sgl-project/sglang/pull/13873) | merged | Feat: GLM-4.6 supports shared experts fusion | model wrapper, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` |
| 2025-11-26 | [#13989](https://github.com/sgl-project/sglang/pull/13989) | merged | Fix GLM-4.6 tool calls don't support streaming output for arguments i… | MoE/router, tests/benchmarks | `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/registered/function_call/test_function_call_parser.py` |
| 2025-12-07 | [#14585](https://github.com/sgl-project/sglang/pull/14585) | merged | [Glm46v] Bug fix for accuracy drop and unable to launch server | model wrapper, MoE/router, multimodal/processor, docs/config | `docs/basic_usage/glmv.md`, `python/sglang/srt/models/glm4v_moe.py`, `docs/basic_usage/glm45.md` |
| 2025-12-08 | [#14668](https://github.com/sgl-project/sglang/pull/14668) | merged | [NVIDIA] Add flashinfer all-to-all MOE dispatcher | model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/test/test_flashinfer_dispatcher.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` |
| 2025-12-17 | [#15333](https://github.com/sgl-project/sglang/pull/15333) | merged | [GLM-4.7] GLM-4.7 Tool Parser and Doc Update | model wrapper, MoE/router, tests/benchmarks, docs/config | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-12-20 | [#15520](https://github.com/sgl-project/sglang/pull/15520) | merged | [model-gateway]: Tool parser for glm47 | MoE/router, tests/benchmarks | `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs` |
| 2025-12-24 | [#15753](https://github.com/sgl-project/sglang/pull/15753) | merged | Fix GLM-4.7 MoE Detector complex JSON Schema type parsing | MoE/router, tests/benchmarks | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/utils.py`, `python/sglang/srt/function_call/glm47_moe_detector.py` |
| 2025-12-24 | [#15754](https://github.com/sgl-project/sglang/pull/15754) | merged | Fix: Handle empty func_name and None values in GLM MoE detectors | MoE/router, tests/benchmarks | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-01-15 | [#17166](https://github.com/sgl-project/sglang/pull/17166) | merged | [Fix] GLM 4.7 + NVFP4 + MTP | model wrapper, MoE/router, docs/config | `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py` |
| 2026-01-17 | [#17247](https://github.com/sgl-project/sglang/pull/17247) | merged | [New Model] GLM4.7-Flash | model wrapper, attention/backend, MoE/router, docs/config | `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` |
| 2026-01-28 | [#17869](https://github.com/sgl-project/sglang/pull/17869) | open | [NPU]Support model GLM-4.7-Flash for npu, accuracy 81% | model wrapper, attention/backend, tests/benchmarks | `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-02-17 | [#18930](https://github.com/sgl-project/sglang/pull/18930) | open | [AMD] Unit tests for mtp in GLM-4.7 | attention/backend, quantization, tests/benchmarks | `test/registered/amd/test_glm4v_fp8_mtp.py`, `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-02-20 | [#19040](https://github.com/sgl-project/sglang/pull/19040) | open | feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash | model wrapper, MoE/router, docs/config | `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/utils/hf_transformers_utils.py` |
| 2026-02-21 | [#19106](https://github.com/sgl-project/sglang/pull/19106) | open | Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks | model wrapper, attention/backend, MoE/router, tests/benchmarks, docs/config | `test/registered/core/test_deepseek_weight_loader.py`, `test/registered/core/test_model_config_transformers_version.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-24 | [#19246](https://github.com/sgl-project/sglang/pull/19246) | merged | [NPU] optimize glm4.7 | model wrapper, MoE/router, quantization | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/hardware_backend/npu/utils.py`, `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2026-03-13 | [#20543](https://github.com/sgl-project/sglang/pull/20543) | merged | fix: do not strip whitespace from GLM tool call values | MoE/router, tests/benchmarks | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-03-22 | [#21135](https://github.com/sgl-project/sglang/pull/21135) | merged | fix: use get_rope_config() to support models without rope_parameters | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py` |
| 2026-03-25 | [#21403](https://github.com/sgl-project/sglang/pull/21403) | merged | [AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8 | model wrapper, MoE/router, quantization | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-03-27 | [#21534](https://github.com/sgl-project/sglang/pull/21534) | merged | [AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x | quantization, tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml` |
| 2026-03-30 | [#21660](https://github.com/sgl-project/sglang/pull/21660) | merged | [GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model. | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2026-04-01 | [#21851](https://github.com/sgl-project/sglang/pull/21851) | merged | GLM-4.7 and GLM-4.7-Flash Loading and import format | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-04-08 | [#22315](https://github.com/sgl-project/sglang/pull/22315) | open | [Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2026-04-10 | [#22509](https://github.com/sgl-project/sglang/pull/22509) | merged | [NPU]Fix GLM-4.7-Flash failed on NPU | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-13 | [#22720](https://github.com/sgl-project/sglang/pull/22720) | merged | fix[glm4.7 flash]: properly detect `gfx95_quant_format` | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-04-14 | [#22801](https://github.com/sgl-project/sglang/pull/22801) | open | [NPU]add dual-stream and deepep support for GLM-4.7-Flash | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` |
| 2026-04-14 | [#22823](https://github.com/sgl-project/sglang/pull/22823) | merged | [Bugfix] Preserve auto-detected quant_config for GLM NextN draft model | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2026-04-17 | [#23067](https://github.com/sgl-project/sglang/pull/23067) | open | Fix: forward continue_final_message kwargs in Glm45Detector | tests/benchmarks | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |

### File-level PR diff reading notes

### PR #11951 - WIP: Fix glm-4.6 tool call streaming parse

- Link: https://github.com/sgl-project/sglang/pull/11951
- Status/date: `open`, created 2025-10-22; author `tonylt`.
- Diff scope read: `3` files, `+450/-105`; areas: MoE/router, tests/benchmarks; keywords: moe, router, config, test.
- Code diff details:
  - `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs` modified +198/-86 (284 lines); hunks: pub struct Glm4MoeParser {; impl Glm4MoeParser {; symbols: Glm4MoeParser
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +180/-19 (199 lines); hunks: from sglang.srt.entrypoints.openai.protocol import Tool; def parse_streaming_increment(; symbols: parse_streaming_increment, parse_streaming_increment, _parse_partial_tool_call, _find_common_prefix
  - `sgl-router/tests/tool_parser_glm4_moe.rs` modified +72/-0 (72 lines); hunks: async fn test_glm4_nested_json_in_arg_values() {
- Optimization/support interpretation: The concrete diff surface is `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs`; keywords observed in patches: moe, router, config, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12456 - [fix] Handle escaped characters in GLM tool call parser to prevent double serialization

- Link: https://github.com/sgl-project/sglang/pull/12456
- Status/date: `merged`, created 2025-10-31, merged 2025-11-05; author `soaringk`.
- Diff scope read: `2` files, `+127/-13`; areas: MoE/router, tests/benchmarks; keywords: moe, test.
- Code diff details:
  - `test/srt/test_function_call_parser.py` modified +103/-0 (103 lines); hunks: def test_partial_tool_call(self):; symbols: test_partial_tool_call, test_array_argument_with_escaped_json, check_params, check_single_todos
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13 (37 lines); hunks: def get_argument_type(func_name: str, arg_key: str, defined_tools: list):; def __init__(self):; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`; keywords observed in patches: moe, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13786 - Overlap glm moe gemms in two cuda streams

- Link: https://github.com/sgl-project/sglang/pull/13786
- Status/date: `merged`, created 2025-11-23, merged 2025-11-25; author `Qiaolin-Yu`.
- Diff scope read: `1` files, `+47/-3`; areas: model wrapper, MoE/router; keywords: cuda, deepep, expert, flash, fp4, moe, router, topk.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +47/-3 (50 lines); hunks: def forward(; symbols: forward, forward_normal_dual_stream, forward_normal
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: cuda, deepep, expert, flash, fp4, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13873 - Feat: GLM-4.6 supports shared experts fusion

- Link: https://github.com/sgl-project/sglang/pull/13873
- Status/date: `merged`, created 2025-11-25, merged 2025-12-01; author `UranusSeven`.
- Diff scope read: `7` files, `+252/-24`; areas: model wrapper, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, moe, quant, triton, expert, topk, benchmark, cuda, deepep, fp8.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/models/glm4_moe.py` modified +74/-19 (93 lines); hunks: is_cuda,; def __init__(; symbols: __init__, __init__, __init__, forward
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +19/-2 (21 lines); hunks: def try_get_optimal_moe_config(; def try_get_optimal_moe_config(; symbols: try_get_optimal_moe_config, try_get_optimal_moe_config, try_get_optimal_moe_config
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +7/-3 (10 lines); hunks: def get_model_config(; def get_model_config(; symbols: get_model_config, get_model_config
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-0 (4 lines); hunks: def __init__(; symbols: __init__, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`; keywords observed in patches: config, moe, quant, triton, expert, topk. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13989 - Fix GLM-4.6 tool calls don't support streaming output for arguments i…

- Link: https://github.com/sgl-project/sglang/pull/13989
- Status/date: `merged`, created 2025-11-26, merged 2025-12-13; author `cynial`.
- Diff scope read: `2` files, `+527/-81`; areas: MoE/router, tests/benchmarks; keywords: cache, moe, test.
- Code diff details:
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +498/-66 (564 lines); hunks: import json; def get_argument_type(func_name: str, arg_key: str, defined_tools: list):; symbols: get_argument_type, StreamState, get_argument_type, get_argument_type
  - `test/registered/function_call/test_function_call_parser.py` modified +29/-15 (44 lines); hunks: def test_streaming_tool_call(self):; def test_streaming_multiple_tool_calls(self):; symbols: test_streaming_tool_call, test_streaming_multiple_tool_calls, test_invalid_tool_call, test_partial_tool_call
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/registered/function_call/test_function_call_parser.py`; keywords observed in patches: cache, moe, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/registered/function_call/test_function_call_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14585 - [Glm46v] Bug fix for accuracy drop and unable to launch server

- Link: https://github.com/sgl-project/sglang/pull/14585
- Status/date: `merged`, created 2025-12-07, merged 2025-12-08; author `byjiang1996`.
- Diff scope read: `12` files, `+308/-29`; areas: model wrapper, MoE/router, multimodal/processor, docs/config; keywords: moe, config, attention, processor, doc, quant, vision, cache, cuda, eagle.
- Code diff details:
  - `docs/basic_usage/glmv.md` added +136/-0 (136 lines); hunks: +# GLM-4.6V / GLM-4.5V Usage
  - `python/sglang/srt/models/glm4v_moe.py` modified +68/-15 (83 lines); hunks: import torch.nn as nn; def __init__(; symbols: __init__, __init__, determine_num_fused_shared_experts, load_weights
  - `docs/basic_usage/glm45.md` added +70/-0 (70 lines); hunks: +## Launch GLM-4.5 / GLM-4.6 with SGLang
  - `python/sglang/srt/models/glm4v.py` modified +19/-3 (22 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, get_video_feature
  - `python/sglang/srt/sampling/custom_logit_processor.py` modified +8/-0 (8 lines); hunks: def __call__(self, logits, custom_param_list: list[dict[str, Any]]):; symbols: __call__, Glm4MoeThinkingBudgetLogitProcessor, Qwen3ThinkingBudgetLogitProcessor
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/glmv.md`, `python/sglang/srt/models/glm4v_moe.py`, `docs/basic_usage/glm45.md`; keywords observed in patches: moe, config, attention, processor, doc, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/glmv.md`, `python/sglang/srt/models/glm4v_moe.py`, `docs/basic_usage/glm45.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14668 - [NVIDIA] Add flashinfer all-to-all MOE dispatcher

- Link: https://github.com/sgl-project/sglang/pull/14668
- Status/date: `merged`, created 2025-12-08, merged 2026-01-24; author `trevor-m`.
- Diff scope read: `14` files, `+723/-16`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: flash, moe, expert, config, deepep, fp4, router, topk, quant, attention.
- Code diff details:
  - `python/sglang/test/test_flashinfer_dispatcher.py` added +322/-0 (322 lines); hunks: +import unittest; symbols: TestFlashinferDispatcher, setUpClass, tearDownClass, create_dispatcher
  - `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` added +263/-0 (263 lines); hunks: +from __future__ import annotations; symbols: FlashinferDispatchOutput, format, FlashinferCombineInput, format
  - `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` added +47/-0 (47 lines); hunks: +import torch.distributed as dist; symbols: CommBackend:, when, TorchDistributedCommBackend, __init__
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +23/-14 (37 lines); hunks: MoeRunner,; def _slice_scale(w):; symbols: _slice_scale, apply, apply, apply
  - `python/sglang/srt/server_args.py` modified +23/-2 (25 lines); hunks: "cutlass",; class ServerArgs:; symbols: ServerArgs:, _handle_a2a_moe, _handle_eplb_and_dispatch
- Optimization/support interpretation: The concrete diff surface is `python/sglang/test/test_flashinfer_dispatcher.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`; keywords observed in patches: flash, moe, expert, config, deepep, fp4. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/test/test_flashinfer_dispatcher.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15333 - [GLM-4.7] GLM-4.7 Tool Parser and Doc Update

- Link: https://github.com/sgl-project/sglang/pull/15333
- Status/date: `merged`, created 2025-12-17, merged 2025-12-20; author `zRzRzRzRzRzRzR`.
- Diff scope read: `7` files, `+809/-394`; areas: model wrapper, MoE/router, tests/benchmarks, docs/config; keywords: moe, kv, cache, doc, spec, config, fp8, processor, test.
- Code diff details:
  - `test/registered/function_call/test_function_call_parser.py` modified +212/-388 (600 lines); hunks: from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector; def setUp(self):; symbols: setUp, test_detect_and_parse_xml_format, test_streaming_xml_format, test_streaming_json_format
  - `python/sglang/srt/function_call/glm47_moe_detector.py` added +584/-0 (584 lines); hunks: +import ast; symbols: StreamState, get_argument_type, _convert_to_number, parse_arguments
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +5/-2 (7 lines); hunks: def get_argument_type(; symbols: get_argument_type, _convert_to_number
  - `docs/basic_usage/glm45.md` modified +4/-2 (6 lines); hunks: -## Launch GLM-4.5 / GLM-4.6 with SGLang; python3 -m sglang.launch_server \
  - `docs/advanced_features/server_arguments.md` modified +1/-1 (2 lines); hunks: Please consult the documentation below and [server_args.py](https://github.com/s
- Optimization/support interpretation: The concrete diff surface is `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`; keywords observed in patches: moe, kv, cache, doc, spec, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15520 - [model-gateway]: Tool parser for glm47

- Link: https://github.com/sgl-project/sglang/pull/15520
- Status/date: `merged`, created 2025-12-20, merged 2025-12-21; author `UbeCc`.
- Diff scope read: `8` files, `+179/-26`; areas: MoE/router, tests/benchmarks; keywords: moe, spec, test, config, benchmark, cache, doc, router.
- Code diff details:
  - `sgl-model-gateway/tests/tool_parser_glm47_moe.rs` added +132/-0 (132 lines); hunks: +//! GLM-4.7 MoE Parser Integration Tests
  - `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs` modified +22/-8 (30 lines); hunks: use crate::{; pub struct Glm4MoeParser {; symbols: Glm4MoeParser
  - `sgl-model-gateway/tests/tool_parser_glm4_moe.rs` modified +7/-7 (14 lines); hunks: use common::create_test_tools;; The weather will be..."#;
  - `sgl-model-gateway/src/tool_parser/factory.rs` modified +5/-3 (8 lines); hunks: impl ParserFactory {; impl ParserFactory {
  - `sgl-model-gateway/benches/tool_parser_benchmark.rs` modified +5/-2 (7 lines); hunks: Let me examine the scan results and provide recommendations."#;; analyze_customer_behavior
- Optimization/support interpretation: The concrete diff surface is `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs`; keywords observed in patches: moe, spec, test, config, benchmark, cache. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15753 - Fix GLM-4.7 MoE Detector complex JSON Schema type parsing

- Link: https://github.com/sgl-project/sglang/pull/15753
- Status/date: `merged`, created 2025-12-24, merged 2026-01-09; author `Leoyzen`.
- Diff scope read: `4` files, `+869/-20`; areas: MoE/router, tests/benchmarks; keywords: moe, spec, config, test.
- Code diff details:
  - `test/registered/function_call/test_glm47_moe_detector.py` modified +678/-3 (681 lines); hunks: from sglang.srt.entrypoints.openai.protocol import Function, Tool; def test_streamed_raw_length_multiple_empty_returns(self):; symbols: test_streamed_raw_length_multiple_empty_returns, TestGlm4ComplexJsonSchema, setUp, test_get_argument_type_simple_type
  - `python/sglang/srt/function_call/utils.py` modified +104/-1 (105 lines); hunks: from json import JSONDecodeError, JSONDecoder; def _get_tool_schema(tool: Tool) -> dict:; symbols: _get_tool_schema, infer_type_from_json_schema, get_json_schema_constraint
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +43/-10 (53 lines); hunks: ToolCallItem,; def get_argument_type(; symbols: get_argument_type, get_argument_type, _get_value_type, _format_value_complete
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +44/-6 (50 lines); hunks: ToolCallItem,; def get_argument_type(; symbols: get_argument_type, get_argument_type, _convert_to_number, _get_value_type
- Optimization/support interpretation: The concrete diff surface is `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/utils.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`; keywords observed in patches: moe, spec, config, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/utils.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15754 - Fix: Handle empty func_name and None values in GLM MoE detectors

- Link: https://github.com/sgl-project/sglang/pull/15754
- Status/date: `merged`, created 2025-12-24, merged 2025-12-30; author `Leoyzen`.
- Diff scope read: `4` files, `+1513/-140`; areas: MoE/router, tests/benchmarks; keywords: moe, spec, test, cache, config.
- Code diff details:
  - `test/registered/function_call/test_glm47_moe_detector.py` added +1176/-0 (1176 lines); hunks: +import json; symbols: TestGlm47MoeDetector, setUp, test_single_tool_call, test_multiple_tool_calls
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +303/-132 (435 lines); hunks: def get_argument_type(; def __init__(self):; symbols: get_argument_type, _convert_to_number, __init__, _reset_streaming_state
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +19/-8 (27 lines); hunks: def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult; def parse_streaming_increment(; symbols: detect_and_parse, parse_streaming_increment, parse_streaming_increment, parse_streaming_increment
  - `test/registered/function_call/test_function_call_parser.py` modified +15/-0 (15 lines); hunks: def check_single_todos(tool_result, expected):; symbols: check_single_todos, test_empty_function_name_handling, TestGlm47MoeDetector, setUp
- Optimization/support interpretation: The concrete diff surface is `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`; keywords observed in patches: moe, spec, test, cache, config. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17166 - [Fix] GLM 4.7 + NVFP4 + MTP

- Link: https://github.com/sgl-project/sglang/pull/17166
- Status/date: `merged`, created 2026-01-15, merged 2026-01-21; author `b8zhong`.
- Diff scope read: `6` files, `+114/-9`; areas: model wrapper, MoE/router, docs/config; keywords: config, quant, fp4, moe, cache, flash, spec, attention, expert, fp8.
- Code diff details:
  - `python/sglang/srt/model_loader/weight_utils.py` modified +38/-0 (38 lines); hunks: def filter_duplicate_safetensors_files(; symbols: filter_duplicate_safetensors_files, maybe_add_mtp_safetensors, filter_files_not_needed_for_inference
  - `python/sglang/srt/configs/model_config.py` modified +17/-8 (25 lines); hunks: def _verify_quantization(self) -> None:; symbols: _verify_quantization
  - `python/sglang/srt/server_args.py` modified +22/-0 (22 lines); hunks: from sglang.srt.utils.common import (; def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `python/sglang/srt/utils/common.py` modified +18/-0 (18 lines); hunks: def assert_pkg_version(pkg: str, min_version: str, message: str):; symbols: assert_pkg_version, check_pkg_version_at_least, kill_process_tree
  - `python/sglang/srt/model_loader/loader.py` modified +14/-0 (14 lines); hunks: get_quant_config,; class Source:; symbols: Source:, init_new, __init__, _get_weights_iterator
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: config, quant, fp4, moe, cache, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17247 - [New Model] GLM4.7-Flash

- Link: https://github.com/sgl-project/sglang/pull/17247
- Status/date: `merged`, created 2026-01-17, merged 2026-01-20; author `Qiaolin-Yu`.
- Diff scope read: `6` files, `+842/-12`; areas: model wrapper, attention/backend, MoE/router, docs/config; keywords: kv, moe, attention, config, cuda, flash, fp8, mla, quant, spec.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe_lite.py` added +808/-0 (808 lines); hunks: +# Copyright 2025-2026 SGLang Team; symbols: Glm4MoeLiteMLP, __init__, forward, Glm4MoeLiteGate
  - `python/sglang/srt/configs/model_config.py` modified +19/-9 (28 lines); hunks: def _config_draft_model(self):; def _derive_model_shapes(self):; symbols: _config_draft_model, _derive_model_shapes, _derive_model_shapes
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +7/-2 (9 lines); hunks: _use_aiter_gfx95,; def _concat_and_cast_mha_k(; symbols: _concat_and_cast_mha_k
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
  - `python/sglang/srt/server_args.py` modified +3/-0 (3 lines); hunks: def _handle_model_specific_adjustments(self):; def _handle_speculative_decoding(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding, auto_choose_speculative_params
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`; keywords observed in patches: kv, moe, attention, config, cuda, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17869 - [NPU]Support model GLM-4.7-Flash for npu, accuracy 81%

- Link: https://github.com/sgl-project/sglang/pull/17869
- Status/date: `open`, created 2026-01-28; author `McZyWu`.
- Diff scope read: `4` files, `+86/-5`; areas: model wrapper, attention/backend, tests/benchmarks; keywords: attention, flash, cache, kv, lora, test, cuda, mla, quant, topk.
- Code diff details:
  - `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py` added +54/-0 (54 lines); hunks: +import os; symbols: TestGLM47Flash
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +30/-4 (34 lines); hunks: def forward_extend(; symbols: forward_extend
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1 (2 lines); hunks: def forward_mha_prepare_npu(; symbols: forward_mha_prepare_npu
  - `python/sglang/test/ascend/test_ascend_utils.py` modified +1/-0 (1 lines); hunks: DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH = os.path.join(
- Optimization/support interpretation: The concrete diff surface is `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; keywords observed in patches: attention, flash, cache, kv, lora, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18930 - [AMD] Unit tests for mtp in GLM-4.7

- Link: https://github.com/sgl-project/sglang/pull/18930
- Status/date: `open`, created 2026-02-17; author `almaslof`.
- Diff scope read: `2` files, `+120/-1`; areas: attention/backend, quantization, tests/benchmarks; keywords: attention, cache, config, cuda, eagle, fp8, kv, mla, spec, test.
- Code diff details:
  - `test/registered/amd/test_glm4v_fp8_mtp.py` added +118/-0 (118 lines); hunks: +import unittest; symbols: TestGLM47FP8TPMTP, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-1 (3 lines); hunks: def init_forward_metadata_capture_cuda_graph(; symbols: init_forward_metadata_capture_cuda_graph
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/test_glm4v_fp8_mtp.py`, `python/sglang/srt/layers/attention/aiter_backend.py`; keywords observed in patches: attention, cache, config, cuda, eagle, fp8. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/test_glm4v_fp8_mtp.py`, `python/sglang/srt/layers/attention/aiter_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19040 - feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash

- Link: https://github.com/sgl-project/sglang/pull/19040
- Status/date: `open`, created 2026-02-20; author `lujangus`.
- Diff scope read: `4` files, `+52/-0`; areas: model wrapper, MoE/router, docs/config; keywords: config, moe, attention, flash, kv, lora, mla, spec.
- Code diff details:
  - `python/sglang/srt/configs/glm4_moe_lite.py` added +47/-0 (47 lines); hunks: +# Copyright 2025-2026 SGLang Team; symbols: Glm4MoeLiteConfig, with, __init__
  - `python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunks: from sglang.srt.configs.dots_vlm import DotsVLMConfig; "DotsVLMConfig",
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0 (2 lines); hunks: DotsVLMConfig,; KimiLinearConfig,
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/utils/hf_transformers_utils.py`; keywords observed in patches: config, moe, attention, flash, kv, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/utils/hf_transformers_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19106 - Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks

- Link: https://github.com/sgl-project/sglang/pull/19106
- Status/date: `open`, created 2026-02-21; author `lesj0610`.
- Diff scope read: `12` files, `+505/-37`; areas: model wrapper, attention/backend, MoE/router, tests/benchmarks, docs/config; keywords: kv, quant, config, moe, test, expert, mla, awq, cuda, attention.
- Code diff details:
  - `test/registered/core/test_deepseek_weight_loader.py` added +86/-0 (86 lines); hunks: +from types import SimpleNamespace; symbols: _pack_int4_row, test_dequantize_ct_wna16_weight, test_post_load_weights_dequantizes_ct_kv_b_proj, _DummyLoader
  - `test/registered/core/test_model_config_transformers_version.py` added +84/-0 (84 lines); hunks: +import logging; symbols: _build_model_config_stub, _mock_transformers_version, test_verify_transformers_version_glm4_moe_lite_no_downgrade_warning, test_verify_transformers_version_glm4_moe_lite_requires_tf5
  - `python/sglang/srt/models/deepseek_v2.py` modified +52/-27 (79 lines); hunks: def __init__(; def forward(; symbols: __init__, forward, DeepseekV2ForCausalLM, __init__
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +52/-8 (60 lines); hunks: def forward(; def __init__(; symbols: forward, __init__, Glm4MoeLiteForCausalLM, __init__
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +54/-0 (54 lines); hunks: normalize_e4m3fn_to_e4m3fnuz,; class DeepseekV2WeightLoaderMixin:; symbols: DeepseekV2WeightLoaderMixin:, _dequantize_ct_wna16_weight, do_load_weights, post_load_weights
- Optimization/support interpretation: The concrete diff surface is `test/registered/core/test_deepseek_weight_loader.py`, `test/registered/core/test_model_config_transformers_version.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: kv, quant, config, moe, test, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/registered/core/test_deepseek_weight_loader.py`, `test/registered/core/test_model_config_transformers_version.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19246 - [NPU] optimize glm4.7

- Link: https://github.com/sgl-project/sglang/pull/19246
- Status/date: `merged`, created 2026-02-24, merged 2026-04-03; author `randgun`.
- Diff scope read: `4` files, `+146/-15`; areas: model wrapper, MoE/router, quantization; keywords: expert, config, deepep, moe, quant, topk, attention, cuda, kv, processor.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +61/-11 (72 lines); hunks: from sglang.srt.distributed.device_communicators.pynccl_allocator import (; is_cuda,; symbols: Glm4MoeMLP, __init__, forward_prepare, forward_deepep
  - `python/sglang/srt/hardware_backend/npu/utils.py` modified +64/-0 (64 lines); hunks: def get_indexer_weight_stream():; symbols: get_indexer_weight_stream, get_share_stream, set_share_stream, get_routed_stream
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +19/-2 (21 lines); hunks: """Inference-only GLM-4.5, GLM-4.6 Speculative Decoding."""; from transformers import PretrainedConfig; symbols: __init__, forward
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +2/-2 (4 lines); hunks: def _rmsnorm_forward_oot(; symbols: _rmsnorm_forward_oot
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/hardware_backend/npu/utils.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; keywords observed in patches: expert, config, deepep, moe, quant, topk. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/hardware_backend/npu/utils.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20543 - fix: do not strip whitespace from GLM tool call values

- Link: https://github.com/sgl-project/sglang/pull/20543
- Status/date: `merged`, created 2026-03-13, merged 2026-04-09; author `lawrence-harmonic`.
- Diff scope read: `3` files, `+66/-2`; areas: MoE/router, tests/benchmarks; keywords: moe, test.
- Code diff details:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +66/-0 (66 lines); hunks: def test_empty_function_name_handling(self):; def check_single_todos(tool_result, expected):; symbols: test_empty_function_name_handling, test_whitespace_preserved_in_arg_values, TestGlm47MoeDetector, setUp
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +0/-1 (1 lines); hunks: def _parse_argument_pairs(; symbols: _parse_argument_pairs
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-1 (1 lines); hunks: def _parse_argument_pairs(; symbols: _parse_argument_pairs
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`; keywords observed in patches: moe, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21135 - fix: use get_rope_config() to support models without rope_parameters

- Link: https://github.com/sgl-project/sglang/pull/21135
- Status/date: `merged`, created 2026-03-22, merged 2026-03-26; author `alphabetc1`.
- Diff scope read: `18` files, `+44/-42`; areas: model wrapper, MoE/router; keywords: config, attention, kv, cuda, moe, cache, expert, lora.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +5/-5 (10 lines); hunks: log_info_on_rank0,; def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4.py` modified +5/-3 (8 lines); hunks: kv_cache_scales_loader,; def __init__(; symbols: __init__
  - `python/sglang/srt/models/grok.py` modified +2/-5 (7 lines); hunks: from sglang.srt.model_loader.loader import DefaultModelLoader; def __init__(; symbols: __init__
  - `python/sglang/srt/models/llada2.py` modified +4/-2 (6 lines); hunks: is_npu,; def __init__(; symbols: __init__
  - `python/sglang/srt/models/deepseek.py` modified +2/-2 (4 lines); hunks: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py`; keywords observed in patches: config, attention, kv, cuda, moe, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21403 - [AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8

- Link: https://github.com/sgl-project/sglang/pull/21403
- Status/date: `merged`, created 2026-03-25, merged 2026-04-11; author `Jacob0226`.
- Diff scope read: `3` files, `+149/-13`; areas: model wrapper, MoE/router, quantization; keywords: fp8, quant, cache, moe, attention, config, expert, flash, fp4, kv.
- Code diff details:
  - `python/sglang/srt/layers/communicator.py` modified +76/-8 (84 lines); hunks: _is_npu = is_npu(); def model_input_output():; symbols: _fused_rmsnorm_fp8_per_token_quant, model_input_output, AttentionInputs:, __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +58/-3 (61 lines); hunks: def forward_prepare(; def __init__(; symbols: forward_prepare, __init__, _detect_fp8_per_token_quant, _detect_attn_quant_format
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +15/-2 (17 lines); hunks: import logging; def can_auto_enable_marlin_fp8() -> bool:; symbols: can_auto_enable_marlin_fp8, apply_fp8_ptpc_linear, apply_fp8_ptpc_linear
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; keywords observed in patches: fp8, quant, cache, moe, attention, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21534 - [AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x

- Link: https://github.com/sgl-project/sglang/pull/21534
- Status/date: `merged`, created 2026-03-27, merged 2026-03-28; author `Jacob0226`.
- Diff scope read: `2` files, `+96/-0`; areas: quantization, tests/benchmarks; keywords: fp8, test, benchmark, cache, doc, fp4.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` added +61/-0 (61 lines); hunks: +"""MI35x GLM-4.7-FP8 GSM8K Accuracy Evaluation Test (8-GPU); symbols: TestGLM47FP8EvalMI35x, test_glm_47_fp8
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +35/-0 (35 lines); hunks: on:; jobs:
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; keywords observed in patches: fp8, test, benchmark, cache, doc, fp4. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21660 - [GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model.

- Link: https://github.com/sgl-project/sglang/pull/21660
- Status/date: `merged`, created 2026-03-30, merged 2026-03-30; author `zRzRzRzRzRzRzR`.
- Diff scope read: `1` files, `+6/-1`; areas: model wrapper, MoE/router; keywords: cache, config, expert, moe.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: def __init__(; symbols: __init__, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: cache, config, expert, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21851 - GLM-4.7 and GLM-4.7-Flash Loading and import format

- Link: https://github.com/sgl-project/sglang/pull/21851
- Status/date: `merged`, created 2026-04-01, merged 2026-04-04; author `zRzRzRzRzRzRzR`.
- Diff scope read: `2` files, `+139/-86`; areas: model wrapper, MoE/router; keywords: config, cuda, eagle, expert, flash, kv, moe, router, spec, attention.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +130/-57 (187 lines); hunks: """Inference-only GLM-4.5, GLM-4.6 and GLM-4.7 model compatible with HuggingFace weights"""; ); symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +9/-29 (38 lines); hunks: # limitations under the License.; get_tensor_model_parallel_world_size,; symbols: forward, __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`; keywords observed in patches: config, cuda, eagle, expert, flash, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22315 - [Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config

- Link: https://github.com/sgl-project/sglang/pull/22315
- Status/date: `open`, created 2026-04-08; author `maodoudou168`.
- Diff scope read: `1` files, `+7/-5`; areas: model wrapper, MoE/router; keywords: config, moe, quant, spec.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +7/-5 (12 lines); hunks: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe_nextn.py`; keywords observed in patches: config, moe, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22509 - [NPU]Fix GLM-4.7-Flash failed on NPU

- Link: https://github.com/sgl-project/sglang/pull/22509
- Status/date: `merged`, created 2026-04-10, merged 2026-04-22; author `Todobe`.
- Diff scope read: `2` files, `+4/-2`; areas: model wrapper, MoE/router; keywords: config, cuda, moe, quant, router.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +3/-1 (4 lines); hunks: import torch; _is_cuda = is_cuda()
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: config, cuda, moe, quant, router. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22720 - fix[glm4.7 flash]: properly detect `gfx95_quant_format`

- Link: https://github.com/sgl-project/sglang/pull/22720
- Status/date: `merged`, created 2026-04-13, merged 2026-04-13; author `ishandhanani`.
- Diff scope read: `1` files, `+2/-0`; areas: model wrapper, MoE/router; keywords: config, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +2/-0 (2 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe_lite.py`; keywords observed in patches: config, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe_lite.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22801 - [NPU]add dual-stream and deepep support for GLM-4.7-Flash

- Link: https://github.com/sgl-project/sglang/pull/22801
- Status/date: `open`, created 2026-04-14; author `Estrella-xx`.
- Diff scope read: `2` files, `+14/-3`; areas: model wrapper, MoE/router; keywords: config, moe, attention, cuda, deepep, expert, fp4, fp8, kv, mla.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +13/-2 (15 lines); hunks: get_pp_group,; ParallelLMHead,; symbols: __init__, forward, forward, __init__
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunks: def _dispatch_core(; symbols: _dispatch_core
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`; keywords observed in patches: config, moe, attention, cuda, deepep, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22823 - [Bugfix] Preserve auto-detected quant_config for GLM NextN draft model

- Link: https://github.com/sgl-project/sglang/pull/22823
- Status/date: `merged`, created 2026-04-14, merged 2026-04-15; author `Jiminator`.
- Diff scope read: `1` files, `+2/-1`; areas: model wrapper, MoE/router; keywords: config, moe, quant, spec.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-1 (3 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe_nextn.py`; keywords observed in patches: config, moe, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23067 - Fix: forward continue_final_message kwargs in Glm45Detector

- Link: https://github.com/sgl-project/sglang/pull/23067
- Status/date: `open`, created 2026-04-17; author `huwwds`.
- Diff scope read: `2` files, `+66/-1`; areas: tests/benchmarks; keywords: test.
- Code diff details:
  - `test/registered/unit/parser/test_reasoning_parser.py` modified +57/-0 (57 lines); hunks: def test_forced_reasoning_mode(self):; def test_continue_final_message_with_request(self):; symbols: test_forced_reasoning_mode, test_continue_final_message_accepts_kwargs, test_continue_final_message_think_start_in_previous, test_continue_final_message_think_end_in_previous
  - `python/sglang/srt/parser/reasoning_parser.py` modified +9/-1 (10 lines); hunks: class Glm45Detector(BaseReasoningFormatDetector):; symbols: Glm45Detector, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`; keywords observed in patches: test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 30; open PRs: 8.
- Open PRs to keep tracking: [#11951](https://github.com/sgl-project/sglang/pull/11951), [#17869](https://github.com/sgl-project/sglang/pull/17869), [#18930](https://github.com/sgl-project/sglang/pull/18930), [#19040](https://github.com/sgl-project/sglang/pull/19040), [#19106](https://github.com/sgl-project/sglang/pull/19106), [#22315](https://github.com/sgl-project/sglang/pull/22315), [#22801](https://github.com/sgl-project/sglang/pull/22801), [#23067](https://github.com/sgl-project/sglang/pull/23067)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
