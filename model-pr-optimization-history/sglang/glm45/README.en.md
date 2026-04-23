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
