# SGLang GLM-4.5 Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on `2026-04-22` and sgl-cookbook `origin/main` `816bad5` on `2026-04-21`.

Scope: GLM-4.5, GLM-4.5-Air, GLM4-MoE, NextN/MTP, A2A/DeepEP/Mooncake/FlashInfer, reduce-scatter, shared experts fusion, FP8/NVFP4/compressed-tensors, GLM45 reasoning parser, GLM45 tool parser, and shared GLM4-MoE runtime paths.

## Summary

GLM-4.5 is the baseline GLM MoE lane. Later GLM-4.6, GLM-4.7, and GLM-5.x work inherits many model, parser, quantization, and platform decisions from `glm4_moe.py`, `glm4_moe_detector.py`, and `reasoning_parser.py`. GLM-4.5V belongs to the VLM/OCR lane unless a PR touches shared text MoE, fused-MoE, quantization, or parser code.

## Code Surfaces

- `python/sglang/srt/models/glm4.py`
- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/function_call/glm4_moe_detector.py`
- `python/sglang/srt/function_call/glm47_moe_detector.py`
- `python/sglang/srt/parser/reasoning_parser.py`
- `docs/basic_usage/glm45.md`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5.mdx`
- `docs_new/src/snippets/autoregressive/glm-45-deployment.jsx`

## Diff-Reviewed PR Cards

### PR #8224 - Initial GLM-4.5 model support

- Link: https://github.com/sgl-project/sglang/pull/8224
- State: merged on `2025-07-28`, merge commit `6d6a8bc278eac424214e73544ae010bde3fb99cb`
- Diff coverage: `14` files, `+1673/-7`; reviewed GLM model, NextN, detector, model config, server args, and parser-registration hunks.
- Motivation: SGLang needed a complete GLM-4.5 MoE lane: text MoE runtime, NextN/MTP draft model, XML tool detector, parser aliases, and draft architecture rewriting.
- Key implementation: add `Glm4MoeForCausalLM`, `Glm4MoeForCausalLMNextN`, GLM MoE gate/sparse block/model logic, `Glm4MoeDetector`, and the `glm45` parser mapping.
- Key code excerpts:

```python
if is_draft_model and self.hf_config.architectures[0] == "Glm4MoeForCausalLM":
    self.hf_config.architectures[0] = "Glm4MoeForCausalLMNextN"
```

```python
class Glm4MoeForCausalLM(DeepseekV2ForCausalLM):
    ...
EntryClass = [Glm4MoeForCausalLM]
```

```python
class Glm4MoeForCausalLMNextN(Glm4MoeForCausalLM):
    def load_weights(self, weights, is_nextn=True):
        super().load_weights(weights, is_nextn=True)
```

- Validation implications: every GLM-4.5 change should smoke-test base MoE, NextN/MTP, `glm45` reasoning/tool parsers, TP/EP, and quantized loading.

### PR #8456 - compressed_tensors launch support

- Link: https://github.com/sgl-project/sglang/pull/8456
- State: merged on `2025-07-28`, merge commit `25f73c6cf3c2b20441266693ad12030157c1cbef`
- Diff coverage: `1` file, fully reviewed.
- Motivation: GLM-4.5-Air compressed-tensors checkpoints failed because shared-expert fusion accepted `fp8` and `blockwise_int8`, but not `compressed_tensors`.
- Key implementation: add `compressed_tensors` to the shared-expert fusion quantization allowlist. PR body reports GSM8K accuracy `0.935` and output throughput `1582.320 tok/s`.
- Key code excerpt:

```diff
 elif (
     self.quant_config.get_name() == "fp8"
     or self.quant_config.get_name() == "blockwise_int8"
+    or self.quant_config.get_name() == "compressed_tensors"
 ):
```

- Validation implications: include compressed-tensors in the GLM-4.5-Air quantized smoke matrix.

### PR #8647 - disable shared-expert TP under EP

- Link: https://github.com/sgl-project/sglang/pull/8647
- State: merged on `2025-08-01`, merge commit `2ae95d17e80710d5ed1189398f36905ad43f5baa`
- Diff coverage: `1` file, fully reviewed.
- Motivation: `--tp 8 --enable-ep-moe` failed for GLM-4.5 FP8 per-block because shared experts were still TP-sharded; output size `192` was not divisible by FP8 block size `128`.
- Key implementation: when EP is active, construct shared experts with `tp_rank=0,tp_size=1` and use EP-aware all-reduce/shared-output ordering. PR body reports GSM8K accuracy `0.955`, throughput `479.302 tok/s`.
- Key code excerpts:

```python
self.ep_size = get_moe_expert_parallel_world_size()
**(dict(tp_rank=0, tp_size=1) if self.ep_size > 1 else {})
```

```python
if self.ep_size > 1:
    if self.tp_size > 1 and not can_fuse_mlp_allreduce:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    if shared_output is not None:
        final_hidden_states += shared_output
```

- Validation implications: EP+TP GLM-4.5-Air FP8 per-block is the target regression lane.

### PR #8729 - keep router correction bias FP32

- Link: https://github.com/sgl-project/sglang/pull/8729
- State: merged on `2025-08-03`, merge commit `760286e3d378780546b88c6d9e932bc178d39669`
- Diff coverage: `1` file, fully reviewed.
- Motivation: `e_score_correction_bias` controls router/top-k correction and should stay FP32 for numerical stability.
- Key implementation: allocate the parameter with `dtype=torch.float32`.
- Key code excerpt:

```python
self.e_score_correction_bias = nn.Parameter(
    torch.empty((config.n_routed_experts), dtype=torch.float32)
)
```

- Validation implications: compare routing stability on BF16 and FP8 GLM-4.5.

### PR #8804 - support both GLM-4.5 and GLM-4.5-Air

- Link: https://github.com/sgl-project/sglang/pull/8804
- State: merged on `2025-08-05`, merge commit `a4b0d5c9e5cb2b36eacdc30bc9259a213cd17a16`
- Diff coverage: `1` file, fully reviewed.
- Motivation: shared-expert fusion was tied to one routed-expert count, which incorrectly disabled GLM-4.5-Air.
- Key implementation: default the shared-fusion architecture to `Glm4MoeForCausalLM` and remove the hard-coded expert-count guard.
- Key code excerpt:

```diff
-        self, architecture: str = "DeepseekV3ForCausalLM"
+        self, architecture: str = "Glm4MoeForCausalLM"
...
-            or self.config.n_routed_experts != 128
```

- Validation implications: test shared-expert fusion on both GLM-4.5 and GLM-4.5-Air.

### PR #8883 - reduce-scatter interface compatibility

- Link: https://github.com/sgl-project/sglang/pull/8883
- State: merged on `2025-08-07`, merge commit `5b6acc1495f4c4d44bfdb0ce8090426de280b002`
- Diff coverage: `1` file, fully reviewed.
- Motivation: GLM4 inherited DeepSeek MoE logic but had not mirrored reduce-scatter and communicator changes, causing wrong logits in DP/TP communicator modes.
- Key implementation: add `use_reduce_scatter` to GLM forward paths and skip TP all-reduce when reduce-scatter owns the reduction. This historical enablement is superseded by #11665.
- Key code excerpt:

```python
if (
    self.tp_size > 1
    and not can_fuse_mlp_allreduce
    and not use_reduce_scatter
):
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
```

```python
allow_reduce_scatter=True
```

- Validation implications: use #11665 for current reduce-scatter behavior.

### PR #9136 - DP attention buffer and flag utilities

- Link: https://github.com/sgl-project/sglang/pull/9136
- State: merged on `2025-08-14`, merge commit `b87aacb5c55d673ead0a2bc501a58f7d02a5e2cd`
- Diff coverage: `21` files; reviewed DP attention buffer/flag, logits, and GLM/DeepSeek call sites.
- Motivation: GLM MoE with DP attention needed gathered buffers and DP state through structured helpers instead of direct global server-args access.
- Key implementation: expose global/local DP buffer getters and initialized `is_dp_attention_enabled()`.
- Key code excerpts:

```python
def get_global_dp_buffer() -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_global_dp_buffer()
```

```python
def is_dp_attention_enabled():
    assert _ENABLE_DP_ATTENTION_FLAG is not None, "dp attention not initialized!"
    return _ENABLE_DP_ATTENTION_FLAG
```

- Validation implications: test DP attention, cuda graph, and logits gather for GLM MoE.

### PR #9223 - MoE TopKOutput cleanup

- Link: https://github.com/sgl-project/sglang/pull/9223
- State: merged on `2025-08-15`, merge commit `84b006b27833d93045ae5552e2cebb13f5140ab5`
- Diff coverage: `3` files, fully reviewed.
- Motivation: GLM4 MoE had to follow the refactored `TopKOutput` object contract rather than passing top-k tensors ad hoc.
- Key implementation: compute `topk_output = self.topk(...)` and pass the object into `self.experts`.
- Key code excerpt:

```python
topk_output = self.topk(hidden_states, router_logits)
final_hidden_states = self.experts(hidden_states, topk_output)
```

```python
assert TopKOutputChecker.format_is_bypassed(topk_output)
top_k = topk_output.topk_config.top_k
```

- Validation implications: inspect `TopKOutput` first for GLM4 MoE top-k format breakages.

### PR #9264 - GLM PP metadata quick fix

- Link: https://github.com/sgl-project/sglang/pull/9264
- State: merged on `2025-08-17`, merge commit `e47800e176b86d7d95309ab23d6cb3bd76d6c2be`
- Diff coverage: `2` files, fully reviewed.
- Motivation: GLM inference lacked PP metadata after a pipeline-parallel change; nightly GLM-4.5-Air-FP8 GSM8K threshold also used the wrong benchmark basis.
- Key implementation: set `pp_group`, `start_layer`, and `end_layer`; adjust GLM-4.5-Air-FP8 threshold from `0.94` to `0.78`.
- Key code excerpts:

```python
self.pp_group = get_pp_group()
self.start_layer = 0
self.end_layer = config.num_hidden_layers
```

```diff
-"zai-org/GLM-4.5-Air-FP8": 0.94,
+"zai-org/GLM-4.5-Air-FP8": 0.78,
```

- Validation implications: run PP metadata smoke after GLM wrapper edits.

### PR #10008 - MXFP4/AITER compatibility with GLM

- Link: https://github.com/sgl-project/sglang/pull/10008
- State: merged on `2025-09-04`, merge commit `e96973742c326a129da772a115bdeb925643d95a`
- Diff coverage: `8` files; reviewed DeepSeek MXFP4/AITER changes and GLM4 MoE signature compatibility hunk.
- Motivation: the PR mainly optimized DeepSeek MXFP4, but shared DeepSeek/GLM quant signatures had broken GLM-4.5-Air.
- Key implementation: activation supports fused MXFP4 prequant, and GLM4 MoE forward accepts `gemm_output_zero_allocator`.
- Key code excerpts:

```python
if fused_mxfp4_prequant:
    out = act_mul_and_mxfp4_quant(x, "silu")
```

```python
def forward_normal(..., gemm_output_zero_allocator: BumpAllocator = None) -> torch.Tensor:
```

- Validation implications: GLM-4.5-Air must be part of shared DeepSeek quant/kernel signature regressions.

### PR #11017 - GLM-4.5/4.6 docs and router mapping

- Link: https://github.com/sgl-project/sglang/pull/11017
- State: merged on `2025-09-28`, merge commit `abb6781573a86c7e7b22e41fd2924094a7d4a135`
- Diff coverage: `5` files, fully reviewed.
- Motivation: GLM-4.6 reused the GLM-4.5 runtime/parser shape, so docs and router mapping needed to describe the shared lane.
- Key implementation: map both `glm-4.5*` and `glm-4.6*` to `glm4_moe`.
- Key code excerpt:

```rust
self.map_model("glm-4.5*", "glm4_moe");
self.map_model("glm-4.6*", "glm4_moe");
```

- Validation implications: GLM-4.5/4.6 stay on `glm45`; GLM-4.7 uses a separate parser.

### PR #11665 - disable reduce-scatter for GLM45

- Link: https://github.com/sgl-project/sglang/pull/11665
- State: merged on `2025-10-18`, merge commit `f7ab9554554fbd3d07ffa4ad34c5fcbef69591b6`
- Diff coverage: `1` file, fully reviewed.
- Motivation: GLM45 MoE still did not safely support reduce-scatter.
- Key implementation: set `allow_reduce_scatter=False`.
- Key code excerpt:

```diff
-            allow_reduce_scatter=True,
+            allow_reduce_scatter=False,
```

- Validation implications: do not re-enable GLM45 reduce-scatter without a correctness matrix.

### PR #11692 - initialize GLM4.5 MoE A2A backend

- Link: https://github.com/sgl-project/sglang/pull/11692
- State: merged on `2025-10-16`, merge commit `476c67d7fcfea316f23d24afe90a8f679f0490a4`
- Diff coverage: `1` file, fully reviewed.
- Motivation: GLM-4.5 MoE block missed A2A backend initialization after the backend abstraction expanded beyond DeepEP.
- Key implementation: check DeepEP or Mooncake and store `_enable_a2a_moe`.
- Key code excerpt:

```python
self._enable_a2a_moe = (
    get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
)
```

- Validation implications: test both DeepEP and Mooncake A2A paths.

### PR #11800 - refactor GLM-4.5/4.5V implementations

- Link: https://github.com/sgl-project/sglang/pull/11800
- State: merged on `2025-10-24`, merge commit `4060ed37cb67262b0cc7af2bcbbdf37ba12d3501`
- Diff coverage: `4` files; reviewed text MoE, GLM4V MoE, PP, shared-expert, and NextN hunks.
- Motivation: GLM-4.5/4.5V inherited too much directly from DeepSeek-V2, making GLM-specific PP, VLM, and shared-expert changes fragile.
- Key implementation: define GLM-specific sparse MoE block, decoder layer, and model; use `make_layers`, `PPMissingLayer`, and `PPProxyTensors` for PP.
- Key code excerpts:

```python
self.layers, self.start_layer, self.end_layer = make_layers(
    config.num_hidden_layers,
    lambda idx, prefix: Glm4MoeDecoderLayer(...),
    pp_rank=self.pp_group.rank_in_group,
    pp_size=self.pp_group.world_size,
)
```

```python
if self.pp_group.is_last_rank:
    self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
else:
    self.norm = PPMissingLayer(return_tuple=True)
```

- Validation implications: test text GLM-4.5, GLM-4.5V shared text path, PP first/last rank, NextN, and shared-expert fusion.

### PR #11847 - dispatcher interface cleanup

- Link: https://github.com/sgl-project/sglang/pull/11847
- State: merged on `2025-10-20`, merge commit `bfc3b3f786829b3ba73504cda07b6ec74908564f`
- Diff coverage: `24` files; reviewed dispatcher, DP state, DeepEP/Mooncake, and GLM call-site hunks.
- Motivation: dispatcher signatures were inconsistent across Standard, DeepEP, and Mooncake paths.
- Key implementation: dispatcher paths now operate on `TopKOutput`, and extend/decode state is stored in DP attention helpers.
- Key code excerpt:

```python
def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput, ...):
    return single_batch_overlap.execute_sbo(
        hidden_states=hidden_states,
        topk_output=topk_output,
        experts=self,
    )
```

- Validation implications: test GLM4 MoE with DeepEP, Mooncake, TBO/SBO, and decode/extend transitions.

### PR #12162 - return routed experts

- Link: https://github.com/sgl-project/sglang/pull/12162
- State: merged on `2025-12-21`, merge commit `bed301a5acaa9577c9aa706468bdf242f6a43051`
- Diff coverage: `27` files; reviewed capturer, scheduler/model runner, FusedMoE capture, detokenizer, and GLM4 patch hunks.
- Motivation: RL/training workflows need routed expert IDs from inference to align training and serving behavior.
- Key implementation: create `RoutedExpertsCapturer`, capture `topk_output` in FusedMoE, and return base64 int32 expert IDs.
- Key code excerpt:

```python
self.routed_experts_capturer.capture(
    layer_id=self.layer_id,
    topk_output=topk_output,
)
```

- Validation implications: GLM MoE `layer_id`, `num_experts_per_tok`, and fused shared-expert top-k must stay aligned.

### PR #12456 - escaped GLM tool-call values

- Link: https://github.com/sgl-project/sglang/pull/12456
- State: merged on `2025-11-05`, merge commit `44da737770e4bcd9bfa27751f0a0751c9b5c06e1`
- Diff coverage: `2` files, fully reviewed.
- Motivation: literal escaped chars such as `\n` and `\"` broke GLM tool-call parsing and caused double serialization.
- Key implementation: regex accepts real newline and literal `\\n`; argument parsing falls back through direct JSON, JSON-string unescape, reparse, and `ast.literal_eval`.
- Key code excerpt:

```python
self.func_detail_regex = re.compile(
    r"<tool_call>(.*?)(?:\\n|\n)(.*)</tool_call>", re.DOTALL
)
```

```python
wrapped = json.loads('{"tmp": "' + json_value + '"}')
parsed_value = json.loads(wrapped["tmp"])
```

- Validation implications: test escaped arrays, Windows paths, literal `\n`, quotes, streaming, and non-streaming.

### PR #12497 - NVFP4 weight-scale padding assertion

- Link: https://github.com/sgl-project/sglang/pull/12497
- State: merged on `2026-01-15`, merge commit `4346db5fafee11513799ebb57ec3e6ad5d95f6e9`
- Diff coverage: `1` file, fully reviewed.
- Motivation: GLM-4.5-NVFP4 failed because a strict scale alignment assertion rejected layouts that swizzle padding can handle.
- Key implementation: replace the hard assertion with a warning. PR body reports TP8 GSM8K accuracy `0.945`, throughput `2362.182 tok/s`.
- Key code excerpt:

```python
if weight_scale.shape[assert_dim] % 4 != 0:
    logger.warning(
        "NVFP4 %s_weight_scale K' not multiple of 4: shape=%s, group_size=%s",
        name,
        tuple(weight_scale.shape),
    )
```

- Validation implications: check NVFP4 load, warnings, and GSM8K rather than assuming perfect scale alignment.

### PR #12572 - symmetric-memory collective buffers

- Link: https://github.com/sgl-project/sglang/pull/12572
- State: merged on `2025-11-05`, merge commit `2340798353bc58398b6d45f582c7c79b670d0256`
- Diff coverage: `19` files; reviewed symmetric-memory context, PyNccl collectives, DP buffer, and GLM MoE allocation.
- Motivation: all-gather/reduce-scatter buffers need symmetric-memory registration for overlap/communication paths.
- Key implementation: add `use_symmetric_memory()` and allocate GLM shared-output buffers under that context.
- Key code excerpt:

```python
with use_symmetric_memory(
    parallel_state.get_tp_group(), disabled=not is_allocation_symmetric()
):
    final_hidden_states_out = torch.empty_like(final_hidden_states)
```

- Validation implications: test TP collectives, cuda graph, and shared-output allocation.

### PR #12834 - KTransformers heterogeneous compute

- Link: https://github.com/sgl-project/sglang/pull/12834
- State: merged on `2025-11-10`, merge commit `ddd1440d0f027e85af6be53bbb309683ed7ea2c4`
- Diff coverage: `10` files; reviewed KT wrapper, server args, quant fallback, and GLM routed-scaling branch.
- Motivation: KTransformers CPU/GPU expert execution needed a unified wrapper rather than scattered hard-coded paths.
- Key implementation: `KTEPWrapperMethod` wraps GPU MoE quant methods and CPU AMX/AVX experts; GLM applies routed scaling correctly when this wrapper is active.
- Key code excerpt:

```python
if not _is_cuda or isinstance(self.experts.quant_method, KTEPWrapperMethod):
    final_hidden_states *= self.routed_scaling_factor
```

- Validation implications: test CPU/GPU expert correctness and routed-scaling behavior.

### PR #12957 - remove redundant code from #12834

- Link: https://github.com/sgl-project/sglang/pull/12957
- State: merged on `2025-11-10`, merge commit `9cfe78dd3076749c9ac1eec0a91d941d3d3a76c7`
- Diff coverage: `1` file, fully reviewed.
- Motivation: #12834 left duplicate GLM dual-stream forward code, increasing the risk of future one-branch fixes.
- Key implementation: delete the redundant method and unused `KTEPWrapperMethod` import.
- Key code excerpt:

```diff
-from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
-    def forward_normal_dual_stream(...):
-        ...
```

- Validation implications: current dual-stream behavior comes from #13786, not this deleted copy.

### PR #13786 - two-stream GLM MoE GEMM overlap

- Link: https://github.com/sgl-project/sglang/pull/13786
- State: merged on `2025-11-25`, merge commit `4b45d556a7e66d1d978e6df14098a8ba87606a4b`
- Diff coverage: `1` file, fully reviewed.
- Motivation: shared experts and routed experts can overlap on two CUDA streams during graph capture. PR body reports GLM-4.6-FP8 speed improving from `60.40` to `66.31 token/s`.
- Key implementation: route to `forward_normal_dual_stream()` when `alt_stream` exists, hidden states are non-empty, and capture mode is active.
- Key code excerpt:

```python
if (
    self.alt_stream is not None
    and hidden_states.shape[0] > 0
    and get_is_capture_mode()
):
    return self.forward_normal_dual_stream(...)
```

- Validation implications: test cuda graph on/off and fused/non-fused shared experts.

### PR #13873 - GLM shared-expert fusion

- Link: https://github.com/sgl-project/sglang/pull/13873
- State: merged on `2025-12-01`, merge commit `982db4ebac260ef4b0597796541724c81a78fe94`
- Diff coverage: `7` files; reviewed fused-MoE config, lookup, `glm4_moe.py`, and NextN changes.
- Motivation: although titled GLM-4.6, the PR changes the shared GLM4-MoE baseline by fusing shared experts into routed experts.
- Key implementation: shared expert becomes an extra expert slot; `num_experts` and `top_k` increase, and `mlp.shared_experts` weights remap to the last expert ID.
- Key code excerpt:

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

- Validation implications: expert count, top-k, weight remap, quant ignore, and fused-MoE config lookup must be validated together.

### PR #13989 - streaming GLM tool-call arguments

- Link: https://github.com/sgl-project/sglang/pull/13989
- State: merged on `2025-12-13`, merge commit `80554598d33b68636be645856fce43403c7be1cb`
- Diff coverage: `2` files; fully reviewed `glm4_moe_detector.py` and tests.
- Motivation: the old parser waited for the full `</tool_call>` and reparsed the entire XML block, so argument streaming did not work.
- Key implementation: introduce a `StreamState` XML-to-JSON streaming state machine and track `_streamed_raw_length`.
- Key code excerpt:

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
json_increment = self._process_xml_to_json_streaming(raw_increment, func_name, tools)
```

- Validation implications: split tags/values, multiple tools, quotes, unknown types, streaming, and non-streaming.

### PR #14668 - FlashInfer A2A MoE dispatcher

- Link: https://github.com/sgl-project/sglang/pull/14668
- State: merged on `2026-01-24`, merge commit `2c2c4e446b99c529896b3377b24e1b48b6a52e61`
- Diff coverage: `14` files; reviewed dispatcher, server args, env vars, modelopt quant, DeepSeek/GLM call sites, and tests.
- Motivation: FlashInfer one-sided NVLink all-to-all kernels needed an SGLang `--moe-a2a-backend=flashinfer` path.
- Key implementation: add `FlashinferDispatcher`, `FlashinferDispatchOutput`, backend enum, server-arg handling that ties EP size to TP size, disables shared experts fusion, and auto-enables `SGLANG_MOE_NVFP4_DISPATCH`.
- Key code excerpt:

```python
elif a2a_backend.is_flashinfer():
    return FlashinferDispatcher(
        group=get_tp_group().device_group,
        router_topk=moe_runner_config.top_k,
        num_experts=moe_runner_config.num_experts,
    )
```

```python
if self.moe_a2a_backend == "flashinfer":
    self.ep_size = self.tp_size
    self.disable_shared_experts_fusion = True
    envs.SGLANG_MOE_NVFP4_DISPATCH.set(True)
```

- Validation implications: FlashInfer A2A must be tested with FlashInfer CUTLASS, NVFP4 dispatch, dummy tokens, empty ranks, and disabled shared-expert fusion.

### PR #15333 - GLM-4.7 parser split

- Link: https://github.com/sgl-project/sglang/pull/15333
- State: merged on `2025-12-20`, merge commit `b82c7a0ae7444d4fa5a44185643f7c1cc6f372eb`
- Diff coverage: `7` files; reviewed docs, parser registry, `glm47_moe_detector.py`, and tests.
- Motivation: GLM-4.7 tool-call format differs from GLM-4.5/4.6, so it needs `glm47` rather than overloading `glm45`.
- Key implementation: register `Glm47MoeDetector`; docs state GLM-4.7 uses `glm47`, GLM-4.5/4.6 use `glm45`.
- Key code excerpt:

```python
"glm45": Glm4MoeDetector,
"glm47": Glm47MoeDetector,
```

- Validation implications: keep GLM45 and GLM47 parser tests separate.

### PR #15753 - complex JSON Schema type inference

- Link: https://github.com/sgl-project/sglang/pull/15753
- State: merged on `2026-01-09`, merge commit `8ef5b9052825c2624e3ac91852b16998f6f6ee3c`
- Diff coverage: `4` files; reviewed GLM45/GLM47 detectors, shared utils, and tests.
- Motivation: `anyOf`, `oneOf`, `allOf`, enums, and type arrays caused arrays/objects to be parsed as strings.
- Key implementation: add `infer_type_from_json_schema()` and use it in both GLM45 and GLM47 detectors.
- Key code excerpt:

```python
return infer_type_from_json_schema(properties[arg_key])
```

```python
if "properties" in schema:
    return "object"
if "items" in schema:
    return "array"
```

- Validation implications: GLM45 needs complex schema coverage too, not only GLM47.

### PR #15754 - robust partial GLM tool calls

- Link: https://github.com/sgl-project/sglang/pull/15754
- State: merged on `2025-12-30`, merge commit `bc8b526edad7cb0b53658a6d230d4f4f5a1d1949`
- Diff coverage: `4` files; reviewed GLM45/GLM47 error handling and boundary tests.
- Motivation: streaming can produce `<tool_call>` or partial tags before the function name exists; old code raised `AssertionError` or `None.strip()` errors.
- Key implementation: send tool name only after completion, safely extract groups, and update `_streamed_raw_length` even when no JSON is emitted.
- Key code excerpt:

```python
is_func_name_complete = has_arg_key or is_tool_end == self.eot_token
if not is_func_name_complete:
    return None
if not func_name:
    logger.warning("Empty function name detected, skipping tool call")
    return None
```

```python
self._streamed_raw_length = current_raw_length
if not json_increment:
    return None
```

- Validation implications: fuzz partial chunks, split tags, no-arg tools, undefined tools, and incomplete streams.

### PR #17714 - GLM45 reasoning tool interruption

- Link: https://github.com/sgl-project/sglang/pull/17714
- State: merged on `2026-03-02`, merge commit `da2a0240f7784fa8e4c7e978e4357a5908a4ee64`
- Diff coverage: `2` files, fully reviewed.
- Motivation: GLM-4.5 can emit `<think>...<tool_call>` without `</think>`; the old parser treated the tool call as truncated reasoning.
- Key implementation: add `tool_start_token` support to the base reasoning detector and implement `Glm45Detector`.
- Key code excerpt:

```python
if in_reasoning and self.tool_start_token is not None and self.tool_start_token in processed_text:
    tool_idx = processed_text.find(self.tool_start_token)
    reasoning_text = processed_text[:tool_idx].strip()
    normal_text = processed_text[tool_idx:]
```

```python
super().__init__("<think>", "</think>", tool_start_token="<tool_call>", ...)
```

- Validation implications: test normal reasoning, truncated reasoning, tool interruption, split token, and forced reasoning.

### PR #20543 - preserve GLM tool-call whitespace

- Link: https://github.com/sgl-project/sglang/pull/20543
- State: merged on `2026-04-09`, merge commit `8eb235ab512528de4c55200c09e2cbc3159a94ba`
- Diff coverage: `3` files, fully reviewed.
- Motivation: code-edit tool values can intentionally contain leading indentation; `arg_value.strip()` corrupted exact strings.
- Key implementation: strip only keys, not values, and test GLM45/GLM47 exact indentation.
- Key code excerpt:

```diff
 for arg_key, arg_value in pairs:
     arg_key = arg_key.strip()
-    arg_value = arg_value.strip()
     arg_type = get_argument_type(func_name, arg_key, tools)
```

- Validation implications: compare exact whitespace in parser tests.

## Diff-Reviewed Open PR Risk Cards

### PR #13711 - RTX Pro 6000 fused-MoE TP2 configs

- Link: https://github.com/sgl-project/sglang/pull/13711
- State: open as of `2026-04-23`
- Diff coverage: `5` files, `+585/-0`; reviewed fused-MoE benchmark utility and config JSON.
- Motivation: add FP8 W8A8 TP2 fused-MoE configs for GLM-4.5-Air/GLM-4.5V on 2x RTX Pro 6000 Blackwell.
- Key implementation: add `Glm4vMoeForConditionalGeneration` and Triton `3.4.0`/`3.5.1` configs for `E=128,N=704` and `E=129,N=704`.
- Key code excerpt:

```python
"Glm4vMoeForConditionalGeneration",
```

```json
{ "1": { "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64 } }
```

- Validation implications: treat as hardware-tuning radar until merged and validated on RTX Pro 6000.

### PR #19106 - GLM4 MoE Lite CompressedTensors and TF version

- Link: https://github.com/sgl-project/sglang/pull/19106
- State: open as of `2026-04-23`
- Diff coverage: `12` files; reviewed model config, attention backend, weight loader, DeepSeek/GLM mapping, GLM4 MoE Lite, and tests.
- Motivation: CompressedTensors GLM4 MoE Lite checkpoints failed because quantized linears may not expose `.weight`; `glm4_moe_lite` also incorrectly suggested downgrading transformers.
- Key implementation: guard MLA fast path on `.weight`, dequantize CT WNA16 `kv_b_proj`, add fused-module mappings, disable shared-expert fusion when shared experts are ignored/non-quantized, and require TF>=5.
- Key code excerpt:

```python
fused_qkv_a_proj = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
if fused_qkv_a_proj is not None and getattr(fused_qkv_a_proj, "weight", None) is not None:
    ...
```

```python
qweight = unpack_from_int32(qweight, num_bits=4, packed_dim=1)
return (qweight * scales).reshape(out_features, in_features)
```

- Validation implications: test AWQ baseline, CT WNA16, shared-expert ignore, and TF>=5 warning behavior.

### PR #19728 - ROCm GLM-4.5V-FP8 startup

- Link: https://github.com/sgl-project/sglang/pull/19728
- State: open as of `2026-04-23`
- Diff coverage: `4` files; reviewed fused-MoE padding guard, HIP FP8 fallback copy helper, and tests.
- Motivation: MI300X GLM-4.5V-FP8 startup failed with unpadded MoE weights plus global padding and padded HIP FP8 fallback buffers.
- Key implementation: skip padding subtraction when runtime hidden size already matches weight K, and copy/fill padded HIP FP8 output rows safely.
- Key code excerpt:

```python
elif hidden_states.shape[1] == w1.shape[2]:
    padded_size = 0
```

```python
dst[: src.shape[0]].copy_(src)
if dst.shape[0] > src.shape[0]:
    dst[src.shape[0] :].fill_(pad_value)
```

- Validation implications: VLM-specific but shared fused-MoE/FP8 code can affect text GLM MoE.

### PR #20917 - `/v1/responses` enable_thinking

- Link: https://github.com/sgl-project/sglang/pull/20917
- State: open as of `2026-04-23`
- Diff coverage: reviewed `/v1/responses` reasoning gating hunk and PR description; unrelated dependency/attention hunks are not GLM motivation.
- Motivation: `/v1/responses` did not honor `chat_template_kwargs.enable_thinking=false` for GLM45/Qwen3-style reasoning models.
- Key implementation: disable reasoning parsing only when `enable_thinking` is explicitly `False`.
- Key code excerpt:

```python
if self.reasoning_parser in ["qwen3", "glm45", "nemotron_3", "interns1"]:
    enable_reasoning = (
        not request.chat_template_kwargs
        or request.chat_template_kwargs.get("enable_thinking") is not False
    )
```

- Validation implications: `/v1/responses` and `/v1/chat/completions` must match for GLM45 with `enable_thinking` unset, true, and false.

### PR #23067 - forward `continue_final_message` in Glm45Detector

- Link: https://github.com/sgl-project/sglang/pull/23067
- State: open as of `2026-04-23`
- Diff coverage: `2` files, fully reviewed.
- Motivation: `ReasoningParser` forwards `continue_final_message` and `previous_content`, but `Glm45Detector.__init__` did not accept them, causing HTTP 500 for GLM-4.5/GLM-5 chat continuation.
- Key implementation: extend `Glm45Detector.__init__` and forward kwargs to the base detector.
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
        tool_start_token="<tool_call>",
        continue_final_message=continue_final_message,
        previous_content=previous_content,
    )
```

- Validation implications: keep this on GLM45/GLM5 radar; detector subclasses selected by `ReasoningParser` should accept base-path kwargs.

## Cookbook and Public Evidence

- `sgl-cookbook#92`: GLM-4.5 AMD MI300X/MI325X/MI355X deployment evidence.
- `sgl-cookbook#95`: GLM-4.5V AMD deployment evidence; keep in the VLM lane unless shared text MoE/fused-MoE/quant/parser code changes.
- Official SGLang docs cover GLM-4.5/4.6/4.7 launch, `glm45` vs `glm47`, EAGLE/MTP, and thinking budget via custom logit processor.
- LMSYS GLM-4.5 launch material documents day-one SGLang support, 128k context, native function calling, and MTP. Treat this as deployment background, not a substitute for PR diff review.

## Next Work

1. Build fixed BF16, FP8, compressed-tensors, NVFP4, A2A/DeepEP/Mooncake/FlashInfer, and parser-streaming smoke tests.
2. Keep GLM45 reasoning parser and tool parser changes in separate validation lanes.
3. Validate shared-expert fusion with expert count, top-k, weight remap, quant ignore, and fused-MoE config together.
4. Treat open PRs as risk cards until merged, not current-main behavior.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `GLM-4.5` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-21 | [#8224](https://github.com/sgl-project/sglang/pull/8224) | merged | GLM-4.5 Model Support | model wrapper, MoE/router, kernel, tests/benchmarks, docs/config | `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2025-07-28 | [#8456](https://github.com/sgl-project/sglang/pull/8456) | merged | fix GLM4_MOE launch with compressed_tensor quant model | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-01 | [#8647](https://github.com/sgl-project/sglang/pull/8647) | merged | Disable tp for shared experts under expert parallelism for GLM4.5 model (#8647) | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-03 | [#8729](https://github.com/sgl-project/sglang/pull/8729) | merged | use fp32 for e_score_correction_bias in GLM-4.5 | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-05 | [#8804](https://github.com/sgl-project/sglang/pull/8804) | merged | GLM-4.5 and GLM-4.5-Air both support | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-07 | [#8883](https://github.com/sgl-project/sglang/pull/8883) | merged | fix glm4 moe | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-13 | [#9136](https://github.com/sgl-project/sglang/pull/9136) | merged | [DP Attention] Refactor: adding some utility functions | model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, scheduler/runtime | `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| 2025-08-15 | [#9223](https://github.com/sgl-project/sglang/pull/9223) | merged | Cleanup MoE Refactor | model wrapper, MoE/router, quantization | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-17 | [#9264](https://github.com/sgl-project/sglang/pull/9264) | merged | Quick Fix GLM | model wrapper, MoE/router, tests/benchmarks | `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_nightly_gsm8k_eval.py` |
| 2025-09-04 | [#10008](https://github.com/sgl-project/sglang/pull/10008) | merged | Optimized deepseek-v3/r1 model performance on mxfp4 run | model wrapper, MoE/router, quantization | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` |
| 2025-09-28 | [#11017](https://github.com/sgl-project/sglang/pull/11017) | merged | Update GLM-4.5 Model Doc | model wrapper, MoE/router | `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`, `sgl-router/src/tool_parser/registry.rs` |
| 2025-10-15 | [#11665](https://github.com/sgl-project/sglang/pull/11665) | merged | fix(glm45): disable reduce scatter | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-16 | [#11692](https://github.com/sgl-project/sglang/pull/11692) | merged | Fix missing a2a backend init of GLM4.5 MoE Block | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-18 | [#11800](https://github.com/sgl-project/sglang/pull/11800) | merged | Refactoring GLM-4.5 and GLM-4.5V related implementations | model wrapper, MoE/router, multimodal/processor | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2025-10-20 | [#11847](https://github.com/sgl-project/sglang/pull/11847) | merged | [9/N] MoE Refactor: cleanup dispatcher interfaces | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-10-26 | [#12162](https://github.com/sgl-project/sglang/pull/12162) | merged | [Feature] Enable return routed experts | model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks | `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `test/srt/rl/test_return_routed_experts.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-10-31 | [#12456](https://github.com/sgl-project/sglang/pull/12456) | merged | [fix] Handle escaped characters in GLM tool call parser to prevent double serialization | MoE/router, tests/benchmarks | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-11-01 | [#12497](https://github.com/sgl-project/sglang/pull/12497) | merged | [Fix] Remove assertion for padding for NVFP4 weight scales to fix GLM 4.5 NVFP4 | quantization | `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2025-11-04 | [#12572](https://github.com/sgl-project/sglang/pull/12572) | merged | Register allgather/reducescatter buffers with symm memory | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime | `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/dp_attention.py` |
| 2025-11-07 | [#12834](https://github.com/sgl-project/sglang/pull/12834) | merged | Refactor KTransformers heterogeneous compute with unified GPU-quantization backend | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`, `python/sglang/srt/server_args.py` |
| 2025-11-10 | [#12957](https://github.com/sgl-project/sglang/pull/12957) | merged | clean redundant code in previous PR | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-11-21 | [#13711](https://github.com/sgl-project/sglang/pull/13711) | open | [fused-moe] Add TP2 RTX Pro 6000 for GLM-4.5-Air and GLM-4.5V | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` |
| 2025-11-23 | [#13786](https://github.com/sgl-project/sglang/pull/13786) | merged | Overlap glm moe gemms in two cuda streams | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-11-25 | [#13873](https://github.com/sgl-project/sglang/pull/13873) | merged | Feat: GLM-4.6 supports shared experts fusion | model wrapper, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` |
| 2025-11-26 | [#13989](https://github.com/sgl-project/sglang/pull/13989) | merged | Fix GLM-4.6 tool calls don't support streaming output for arguments i… | MoE/router, tests/benchmarks | `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/registered/function_call/test_function_call_parser.py` |
| 2025-12-08 | [#14668](https://github.com/sgl-project/sglang/pull/14668) | merged | [NVIDIA] Add flashinfer all-to-all MOE dispatcher | model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/test/test_flashinfer_dispatcher.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` |
| 2025-12-17 | [#15333](https://github.com/sgl-project/sglang/pull/15333) | merged | [GLM-4.7] GLM-4.7 Tool Parser and Doc Update | model wrapper, MoE/router, tests/benchmarks, docs/config | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-12-24 | [#15753](https://github.com/sgl-project/sglang/pull/15753) | merged | Fix GLM-4.7 MoE Detector complex JSON Schema type parsing | MoE/router, tests/benchmarks | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/utils.py`, `python/sglang/srt/function_call/glm47_moe_detector.py` |
| 2025-12-24 | [#15754](https://github.com/sgl-project/sglang/pull/15754) | merged | Fix: Handle empty func_name and None values in GLM MoE detectors | MoE/router, tests/benchmarks | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-01-25 | [#17714](https://github.com/sgl-project/sglang/pull/17714) | merged | Add GLM45 tool interruption support | tests/benchmarks | `test/registered/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-02-21 | [#19106](https://github.com/sgl-project/sglang/pull/19106) | open | Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks | model wrapper, attention/backend, MoE/router, tests/benchmarks, docs/config | `test/registered/core/test_deepseek_weight_loader.py`, `test/registered/core/test_model_config_transformers_version.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-03 | [#19728](https://github.com/sgl-project/sglang/pull/19728) | open | Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback | MoE/router, quantization, kernel, tests/benchmarks | `test/registered/moe/test_fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_custom_ops.py` |
| 2026-03-13 | [#20543](https://github.com/sgl-project/sglang/pull/20543) | merged | fix: do not strip whitespace from GLM tool call values | MoE/router, tests/benchmarks | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-03-19 | [#20917](https://github.com/sgl-project/sglang/pull/20917) | open | fix(serving_responses): check enable_thinking for qwen3/glm45 models | attention/backend, multimodal/processor | `PR_DESCRIPTION.md`, `python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py`, `python/sglang/srt/layers/attention/flashattention_backend.py` |
| 2026-04-17 | [#23067](https://github.com/sgl-project/sglang/pull/23067) | open | Fix: forward continue_final_message kwargs in Glm45Detector | tests/benchmarks | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |

### File-level PR diff reading notes

### PR #8224 - GLM-4.5 Model Support

- Link: https://github.com/sgl-project/sglang/pull/8224
- Status/date: `merged`, created 2025-07-21, merged 2025-07-28; author `zRzRzRzRzRzRzR`.
- Diff scope read: `14` files, `+1673/-7`; areas: model wrapper, MoE/router, kernel, tests/benchmarks, docs/config; keywords: moe, kv, config, expert, test, attention, fp4, processor, quant, spec.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` added +1034/-0 (1034 lines); hunks: +# Copyright 2025-2026 SGLang Team; symbols: Glm4MoeMLP, __init__, forward, Glm4MoeAttention
  - `test/srt/test_function_call_parser.py` modified +184/-0 (184 lines); hunks: from sglang.srt.entrypoints.openai.protocol import Function, Tool; def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_glm45_detector_ebnf
  - `python/sglang/srt/models/glm4_moe_nextn.py` added +167/-0 (167 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: Glm4MoeModelNextN, __init__, forward, Glm4MoeForCausalLMNextN
  - `python/sglang/srt/function_call/glm4_moe_detector.py` added +165/-0 (165 lines); hunks: +import ast; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
  - `test/srt/openai_server/features/test_enable_thinking.py` modified +55/-0 (55 lines); hunks: def setUpClass(cls):; def test_chat_completion_with_reasoning(self):; symbols: setUpClass, tearDownClass, test_chat_completion_with_reasoning, test_chat_completion_without_reasoning
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; keywords observed in patches: moe, kv, config, expert, test, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8456 - fix GLM4_MOE launch with compressed_tensor quant model

- Link: https://github.com/sgl-project/sglang/pull/8456
- Status/date: `merged`, created 2025-07-28, merged 2025-07-28; author `zminglei`.
- Diff scope read: `1` files, `+1/-0`; areas: model wrapper, MoE/router; keywords: config, fp8, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-0 (1 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: config, fp8, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8647 - Disable tp for shared experts under expert parallelism for GLM4.5 model (#8647)

- Link: https://github.com/sgl-project/sglang/pull/8647
- Status/date: `merged`, created 2025-08-01, merged 2025-08-01; author `zminglei`.
- Diff scope read: `1` files, `+73/-5`; areas: model wrapper, MoE/router; keywords: config, cuda, deepep, expert, kv, moe, quant, router, topk.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +73/-5 (78 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, forward_normal_dual_stream
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: config, cuda, deepep, expert, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8729 - use fp32 for e_score_correction_bias in GLM-4.5

- Link: https://github.com/sgl-project/sglang/pull/8729
- Status/date: `merged`, created 2025-08-03, merged 2025-08-03; author `zRzRzRzRzRzRzR`.
- Diff scope read: `1` files, `+1/-1`; areas: model wrapper, MoE/router; keywords: config, expert, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: config, expert, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8804 - GLM-4.5 and GLM-4.5-Air both support

- Link: https://github.com/sgl-project/sglang/pull/8804
- Status/date: `merged`, created 2025-08-05, merged 2025-08-05; author `zRzRzRzRzRzRzR`.
- Diff scope read: `1` files, `+1/-2`; areas: model wrapper, MoE/router; keywords: config, cuda, expert, kv, moe.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-2 (3 lines); hunks: def __init__(; def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts, determine_num_fused_shared_experts
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: config, cuda, expert, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8883 - fix glm4 moe

- Link: https://github.com/sgl-project/sglang/pull/8883
- Status/date: `merged`, created 2025-08-07, merged 2025-08-07; author `ch-wan`.
- Diff scope read: `1` files, `+19/-4`; areas: model wrapper, MoE/router; keywords: attention, cuda, deepep, expert, moe.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +19/-4 (23 lines); hunks: def __init__(; def forward_normal_dual_stream(; symbols: __init__, forward_normal_dual_stream, forward_normal_dual_stream, forward_normal
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: attention, cuda, deepep, expert, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9136 - [DP Attention] Refactor: adding some utility functions

- Link: https://github.com/sgl-project/sglang/pull/9136
- Status/date: `merged`, created 2025-08-13, merged 2025-08-14; author `ch-wan`.
- Diff scope read: `21` files, `+216/-159`; areas: model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, scheduler/runtime; keywords: attention, config, cuda, cache, processor, spec, eagle, moe, expert, triton.
- Code diff details:
  - `python/sglang/srt/layers/dp_attention.py` modified +114/-27 (141 lines); hunks: import logging; tensor_model_parallel_all_reduce,; symbols: DPPaddingMode, DpPaddingMode, is_max_len, is_sum_len
  - `python/sglang/srt/layers/logits_processor.py` modified +12/-18 (30 lines); hunks: tensor_model_parallel_all_gather,; class LogitsMetadata:; symbols: LogitsMetadata:, from_forward_batch, compute_dp_attention_metadata, compute_dp_attention_metadata
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +8/-21 (29 lines); hunks: ); def __init__(self, model_runner: ModelRunner):; symbols: __init__, capture_one_batch_size, capture_one_batch_size, capture_one_batch_size
  - `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` modified +7/-21 (28 lines); hunks: import torch; def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__, capture_one_batch_size, capture_one_batch_size, capture_one_batch_size
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +7/-21 (28 lines); hunks: import torch; def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__, capture_one_batch_size, capture_one_batch_size, capture_one_batch_size
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`; keywords observed in patches: attention, config, cuda, cache, processor, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9223 - Cleanup MoE Refactor

- Link: https://github.com/sgl-project/sglang/pull/9223
- Status/date: `merged`, created 2025-08-15, merged 2025-08-15; author `ch-wan`.
- Diff scope read: `3` files, `+18/-16`; areas: model wrapper, MoE/router, quantization; keywords: expert, router, topk, cuda, moe, config, flash, fp4, fp8, quant.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-7 (13 lines); hunks: def forward_normal_dual_stream(; def forward_normal(; symbols: forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +8/-3 (11 lines); hunks: def apply(; def apply(; symbols: apply, apply
  - `python/sglang/srt/models/glm4_moe.py` modified +4/-6 (10 lines); hunks: def forward_normal_dual_stream(; def forward_normal(; symbols: forward_normal_dual_stream, forward_normal
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: expert, router, topk, cuda, moe, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9264 - Quick Fix GLM

- Link: https://github.com/sgl-project/sglang/pull/9264
- Status/date: `merged`, created 2025-08-17, merged 2025-08-17; author `hebiao064`.
- Diff scope read: `2` files, `+6/-1`; areas: model wrapper, MoE/router, tests/benchmarks; keywords: config, expert, fp8, moe, quant, test.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +5/-0 (5 lines); hunks: from sglang.srt.distributed import (; def __init__(; symbols: __init__, __init__
  - `test/srt/test_nightly_gsm8k_eval.py` modified +1/-1 (2 lines); hunks: "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8": 0.83,
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_nightly_gsm8k_eval.py`; keywords observed in patches: config, expert, fp8, moe, quant, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_nightly_gsm8k_eval.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10008 - Optimized deepseek-v3/r1 model performance on mxfp4 run

- Link: https://github.com/sgl-project/sglang/pull/10008
- Status/date: `merged`, created 2025-09-04, merged 2025-09-04; author `kkHuang-amd`.
- Diff scope read: `8` files, `+489/-67`; areas: model wrapper, MoE/router, quantization; keywords: cuda, fp4, quant, triton, expert, moe, cache, flash, fp8, kv.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +228/-32 (260 lines); hunks: is_cpu,; _is_cpu_amx_available = cpu_has_amx_support(); symbols: forward, __init__, forward, forward
  - `python/sglang/srt/layers/quantization/quark/utils.py` modified +97/-0 (97 lines); hunks: from types import MappingProxyType; def _is_equal_or_regex_match(; symbols: deep_compare, _is_equal_or_regex_match, b_dynamic_mxfp4_quant, mxfp4_to_f32
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +49/-30 (79 lines); hunks: from aiter.ops.gemm_op_a4w4 import gemm_a4w4; def get_min_capability(cls) -> int:; symbols: get_min_capability, process_weights_after_loading, create_weights, apply_weights
  - `python/sglang/srt/layers/rocm_linear_utils.py` added +44/-0 (44 lines); hunks: +import torch; symbols: aiter_dsv3_router_gemm, get_dsv3_gemm_output_zero_allocator_size
  - `python/sglang/srt/layers/communicator.py` modified +36/-4 (40 lines); hunks: from sglang.srt.managers.schedule_batch import global_server_args_dict; def prepare_attn(; symbols: prepare_attn, prepare_attn
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`; keywords observed in patches: cuda, fp4, quant, triton, expert, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11017 - Update GLM-4.5 Model Doc

- Link: https://github.com/sgl-project/sglang/pull/11017
- Status/date: `merged`, created 2025-09-28, merged 2025-09-28; author `zRzRzRzRzRzRzR`.
- Diff scope read: `5` files, `+11/-12`; areas: model wrapper, MoE/router; keywords: moe, config, router, expert, fp4, quant, spec.
- Code diff details:
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +3/-3 (6 lines); hunks: def parse_arguments(json_value):; def __init__(self):; symbols: parse_arguments, Glm4MoeDetector, __init__, has_tool_call
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-3 (6 lines); hunks: # limitations under the License.; def determine_num_fused_shared_experts(; symbols: determine_num_fused_shared_experts
  - `sgl-router/src/tool_parser/registry.rs` modified +2/-3 (5 lines); hunks: impl ParserRegistry {
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-2 (4 lines); hunks: # limitations under the License.; def __init__(; symbols: __init__
  - `sgl-router/src/reasoning_parser/README.md` modified +1/-1 (2 lines); hunks: classDiagram
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`, `sgl-router/src/tool_parser/registry.rs`; keywords observed in patches: moe, config, router, expert, fp4, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`, `sgl-router/src/tool_parser/registry.rs`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11665 - fix(glm45): disable reduce scatter

- Link: https://github.com/sgl-project/sglang/pull/11665
- Status/date: `merged`, created 2025-10-15, merged 2025-10-18; author `jimmy-evo`.
- Diff scope read: `1` files, `+1/-1`; areas: model wrapper, MoE/router; keywords: attention, moe.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: attention, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11692 - Fix missing a2a backend init of GLM4.5 MoE Block

- Link: https://github.com/sgl-project/sglang/pull/11692
- Status/date: `merged`, created 2025-10-16, merged 2025-10-16; author `ShangmingCai`.
- Diff scope read: `1` files, `+4/-2`; areas: model wrapper, MoE/router; keywords: config, deepep, expert, moe.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +4/-2 (6 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, forward_normal_dual_stream
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: config, deepep, expert, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11800 - Refactoring GLM-4.5 and GLM-4.5V related implementations

- Link: https://github.com/sgl-project/sglang/pull/11800
- Status/date: `merged`, created 2025-10-18, merged 2025-10-24; author `zRzRzRzRzRzRzR`.
- Diff scope read: `4` files, `+356/-565`; areas: model wrapper, MoE/router, multimodal/processor; keywords: config, spec, expert, moe, processor, quant, attention, awq, cache, cuda.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +322/-354 (676 lines); hunks: """Inference-only GLM-4.5, GLM-4.6 model compatible with HuggingFace weights"""; get_pp_group,; symbols: __init__, forward, __init__, forward
  - `python/sglang/srt/models/glm4v_moe.py` modified +29/-196 (225 lines); hunks: import torch.nn as nn; def __init__(; symbols: __init__, __init__, determine_num_fused_shared_experts, load_weights
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-14 (18 lines); hunks: # limitations under the License.; from sglang.srt.model_executor.forward_batch_info import ForwardBatch; symbols: forward, forward, forward, Glm4MoeForCausalLMNextN
  - `python/sglang/srt/multimodal/processors/glm4v.py` modified +1/-1 (2 lines); hunks: class Glm4vImageProcessor(SGLangBaseProcessor):; symbols: Glm4vImageProcessor, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; keywords observed in patches: config, spec, expert, moe, processor, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11847 - [9/N] MoE Refactor: cleanup dispatcher interfaces

- Link: https://github.com/sgl-project/sglang/pull/11847
- Status/date: `merged`, created 2025-10-20, merged 2025-10-20; author `ch-wan`.
- Diff scope read: `24` files, `+394/-428`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime; keywords: expert, moe, topk, deepep, config, quant, router, triton, flash, fp4.
- Code diff details:
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91 (177 lines); hunks: from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder; DispatchOutput,; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, __init__
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99 (168 lines); hunks: tma_align_input_scale,; def __init__(; symbols: __init__, __init__, forward, forward
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35 (79 lines); hunks: get_moe_expert_parallel_world_size,; from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod; symbols: _get_tile_tokens_dim, create_moe_dispatcher, FusedMoeWeightScaleSupported, __init__
  - `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39 (76 lines); hunks: from typing import NamedTuple, Optional, Tuple; import torch; symbols: MooncakeDispatchOutput, __init__, dispatch_a, dispatch_a
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-46 (60 lines); hunks: ); from sglang.srt.server_args import get_global_server_args; symbols: __init__, forward_deepep, forward_deepep, _forward_shared_experts_and_put_results
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; keywords observed in patches: expert, moe, topk, deepep, config, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12162 - [Feature] Enable return routed experts

- Link: https://github.com/sgl-project/sglang/pull/12162
- Status/date: `merged`, created 2025-10-26, merged 2025-12-21; author `ocss884`.
- Diff scope read: `27` files, `+646/-10`; areas: model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks; keywords: expert, moe, attention, cache, config, cuda, topk, processor, scheduler, spec.
- Code diff details:
  - `python/sglang/srt/layers/moe/routed_experts_capturer.py` added +289/-0 (289 lines); hunks: +import logging; symbols: get_tensor_size_bytes, _RoutedExpertsDeviceCache:, __init__, get_buffer_size_bytes
  - `test/srt/rl/test_return_routed_experts.py` added +187/-0 (187 lines); hunks: +import asyncio; symbols: TestReturnRoutedExperts, setUpClass, test_return_routed_experts, fetch_result
  - `python/sglang/srt/model_executor/model_runner.py` modified +61/-0 (61 lines); hunks: set_is_extend_in_batch,; def initialize(self, min_per_gpu_memory: float):; symbols: initialize, initialize, init_routed_experts_capturer, remote_instance_init_transfer_engine
  - `python/sglang/srt/managers/detokenizer_manager.py` modified +18/-0 (18 lines); hunks: from typing import Dict, List, Union; def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):; symbols: _decode_batch_token_id_output, _extract_routed_experts, handle_batch_token_id_out, handle_batch_token_id_out
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +18/-0 (18 lines); hunks: from sglang.srt.disaggregation.utils import DisaggregationMode; def process_batch_result_prebuilt(self: Scheduler, batch: ScheduleBatch):; symbols: process_batch_result_prebuilt, maybe_collect_routed_experts, process_batch_result_prefill, process_batch_result_prefill
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `test/srt/rl/test_return_routed_experts.py`, `python/sglang/srt/model_executor/model_runner.py`; keywords observed in patches: expert, moe, attention, cache, config, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `test/srt/rl/test_return_routed_experts.py`, `python/sglang/srt/model_executor/model_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12456 - [fix] Handle escaped characters in GLM tool call parser to prevent double serialization

- Link: https://github.com/sgl-project/sglang/pull/12456
- Status/date: `merged`, created 2025-10-31, merged 2025-11-05; author `soaringk`.
- Diff scope read: `2` files, `+127/-13`; areas: MoE/router, tests/benchmarks; keywords: moe, test.
- Code diff details:
  - `test/srt/test_function_call_parser.py` modified +103/-0 (103 lines); hunks: def test_partial_tool_call(self):; symbols: test_partial_tool_call, test_array_argument_with_escaped_json, check_params, check_single_todos
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13 (37 lines); hunks: def get_argument_type(func_name: str, arg_key: str, defined_tools: list):; def __init__(self):; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`; keywords observed in patches: moe, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12497 - [Fix] Remove assertion for padding for NVFP4 weight scales to fix GLM 4.5 NVFP4

- Link: https://github.com/sgl-project/sglang/pull/12497
- Status/date: `merged`, created 2025-11-01, merged 2026-01-15; author `b8zhong`.
- Diff scope read: `1` files, `+7/-4`; areas: quantization; keywords: config, fp4, fp8, quant.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +7/-4 (11 lines); hunks: def _slice_scale(w):; symbols: _slice_scale
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/modelopt_quant.py`; keywords observed in patches: config, fp4, fp8, quant. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/modelopt_quant.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12572 - Register allgather/reducescatter buffers with symm memory

- Link: https://github.com/sgl-project/sglang/pull/12572
- Status/date: `merged`, created 2025-11-04, merged 2025-11-05; author `nvcastet`.
- Diff scope read: `19` files, `+250/-114`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime; keywords: attention, moe, expert, topk, config, router, cuda, flash, fp4, fp8.
- Code diff details:
  - `python/sglang/srt/distributed/parallel_state.py` modified +76/-19 (95 lines); hunks: def reg_all_gather_into_tensor_fake(; def __init__(; symbols: reg_all_gather_into_tensor_fake, reg_reduce_scatter_tensor, reg_reduce_scatter_tensor_fake, GroupCoordinator:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +30/-22 (52 lines); hunks: from sglang.srt.distributed.device_communicators.pynccl_allocator import (; def apply(; symbols: apply, apply, apply, apply
  - `python/sglang/srt/layers/dp_attention.py` modified +34/-11 (45 lines); hunks: get_tp_group,; class _DpGatheredBufferWrapper:; symbols: _DpGatheredBufferWrapper:, set_dp_buffer_len, get_global_dp_buffer, get_local_dp_buffer
  - `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py` modified +16/-21 (37 lines); hunks: import os; def is_symmetric_memory_enabled():; symbols: is_symmetric_memory_enabled, get_nccl_mem_pool, use_symmetric_memory:, SymmetricMemoryContext:
  - `python/sglang/srt/layers/moe/topk.py` modified +21/-9 (30 lines); hunks: import torch.nn.functional as F; def forward_cuda(; symbols: forward_cuda, forward_cpu, forward_npu, empty_topk_output
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/dp_attention.py`; keywords observed in patches: attention, moe, expert, topk, config, router. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/dp_attention.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12834 - Refactor KTransformers heterogeneous compute with unified GPU-quantization backend

- Link: https://github.com/sgl-project/sglang/pull/12834
- Status/date: `merged`, created 2025-11-07, merged 2025-11-10; author `Atream`.
- Diff scope read: `10` files, `+494/-507`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime; keywords: expert, moe, config, quant, cuda, fp8, triton, topk, deepep, cache.
- Code diff details:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +1/-411 (412 lines); hunks: import enum; except ImportError:; symbols: _mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState, GPTQMarlinState
  - `python/sglang/srt/layers/moe/kt_ep_wrapper.py` added +393/-0 (393 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: KTConfig:, create_kt_config_from_server_args, mask_cpu_expert_ids, KTEPWrapperMethod
  - `python/sglang/srt/server_args.py` modified +6/-43 (49 lines); hunks: class ServerArgs:; def __post_init__(self):; symbols: ServerArgs:, __post_init__, _handle_grammar_backend, _handle_ktransformers_configs
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +24/-19 (43 lines); hunks: get_moe_a2a_backend,; FusedMoEMethodBase,; symbols: __init__, __init__, _weight_loader_physical, _weight_loader_impl
  - `python/sglang/srt/models/glm4_moe.py` modified +37/-0 (37 lines); hunks: ); def forward(; symbols: forward, forward_normal_dual_stream, forward_normal
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: expert, moe, config, quant, cuda, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12957 - clean redundant code in previous PR

- Link: https://github.com/sgl-project/sglang/pull/12957
- Status/date: `merged`, created 2025-11-10, merged 2025-11-10; author `Atream`.
- Diff scope read: `1` files, `+0/-37`; areas: model wrapper, MoE/router; keywords: config, cuda, deepep, expert, flash, fp4, fp8, moe, quant, router.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +0/-37 (37 lines); hunks: ); def forward(; symbols: forward, forward_normal_dual_stream, forward_normal
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: config, cuda, deepep, expert, flash, fp4. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13711 - [fused-moe] Add TP2 RTX Pro 6000 for GLM-4.5-Air and GLM-4.5V

- Link: https://github.com/sgl-project/sglang/pull/13711
- Status/date: `open`, created 2025-11-21; author `mratsim`.
- Diff scope read: `5` files, `+585/-0`; areas: MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: config, moe, triton, fp8, quant, benchmark, kv.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunks: +{
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: def get_model_config(; symbols: get_model_config
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json`; keywords observed in patches: config, moe, triton, fp8, quant, benchmark. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

### PR #17714 - Add GLM45 tool interruption support

- Link: https://github.com/sgl-project/sglang/pull/17714
- Status/date: `merged`, created 2026-01-25, merged 2026-03-02; author `Leoyzen`.
- Diff scope read: `2` files, `+238/-3`; areas: tests/benchmarks; keywords: test.
- Code diff details:
  - `test/registered/parser/test_reasoning_parser.py` modified +182/-0 (182 lines); hunks: from sglang.srt.parser.reasoning_parser import (; def test_streaming_kimi_format(self):; symbols: test_streaming_kimi_format, TestGlm45Detector, setUp, test_init
  - `python/sglang/srt/parser/reasoning_parser.py` modified +56/-3 (59 lines); hunks: def __init__(; def detect_and_parse(self, text: str) -> StreamingParseResult:; symbols: __init__, detect_and_parse, parse_streaming_increment, parse_streaming_increment
- Optimization/support interpretation: The concrete diff surface is `test/registered/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`; keywords observed in patches: test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

### PR #19728 - Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback

- Link: https://github.com/sgl-project/sglang/pull/19728
- Status/date: `open`, created 2026-03-03; author `andyluo7`.
- Diff scope read: `4` files, `+104/-4`; areas: MoE/router, quantization, kernel, tests/benchmarks; keywords: fp8, quant, expert, moe, test, triton, cache, config, cuda, mla.
- Code diff details:
  - `test/registered/moe/test_fused_moe.py` modified +66/-0 (66 lines); hunks: import unittest; def test_various_configurations(self):; symbols: test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4 (25 lines); hunks: def per_token_group_quant_mla_deep_gemm_masked_fp8(; def _native_dynamic_per_token_quant_fp8(output, input, scale):; symbols: per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_token_quant_fp8
  - `python/sglang/test/test_custom_ops.py` modified +11/-0 (11 lines); hunks: import pytest; def test_scaled_fp8_quant_with_padding(dtype) -> None:; symbols: test_scaled_fp8_quant_with_padding
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0 (6 lines); hunks: def fused_experts_impl(; symbols: fused_experts_impl
- Optimization/support interpretation: The concrete diff surface is `test/registered/moe/test_fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_custom_ops.py`; keywords observed in patches: fp8, quant, expert, moe, test, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/moe/test_fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_custom_ops.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

### PR #20917 - fix(serving_responses): check enable_thinking for qwen3/glm45 models

- Link: https://github.com/sgl-project/sglang/pull/20917
- Status/date: `open`, created 2026-03-19; author `xueliangyang-oeuler`.
- Diff scope read: `8` files, `+130/-19`; areas: attention/backend, multimodal/processor; keywords: spec, test, attention, cache, cuda, doc, flash, fp8, kv, quant.
- Code diff details:
  - `PR_DESCRIPTION.md` added +69/-0 (69 lines); hunks: +## PR Motivation
  - `python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py` modified +19/-8 (27 lines); hunks: from sglang.multimodal_gen.runtime.server_args import ServerArgs; def _apply_torch_compile(self, pipe: Any, server_args: ServerArgs) -> Any:; symbols: _apply_torch_compile
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +18/-9 (27 lines); hunks: def forward_extend(; def forward_decode(; symbols: forward_extend, forward_decode
  - `python/sglang/srt/entrypoints/openai/serving_responses.py` modified +10/-1 (11 lines); hunks: def _make_response_output_items(; symbols: _make_response_output_items
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/hunyuan3d_paint.py` modified +7/-1 (8 lines); hunks: ); def _do_load_paint(self, server_args: ServerArgs) -> None:; symbols: _do_load_paint, _convert_pil_list_to_tensor
- Optimization/support interpretation: The concrete diff surface is `PR_DESCRIPTION.md`, `python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`; keywords observed in patches: spec, test, attention, cache, cuda, doc. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `PR_DESCRIPTION.md`, `python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

- Covered PRs: 35; open PRs: 5.
- Open PRs to keep tracking: [#13711](https://github.com/sgl-project/sglang/pull/13711), [#19106](https://github.com/sgl-project/sglang/pull/19106), [#19728](https://github.com/sgl-project/sglang/pull/19728), [#20917](https://github.com/sgl-project/sglang/pull/20917), [#23067](https://github.com/sgl-project/sglang/pull/23067)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
