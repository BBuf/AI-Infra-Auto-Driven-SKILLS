# SGLang Qwen3 Core PR-Diff Optimization History

This document covers Qwen3 dense, Qwen3 MoE, Qwen3-30B-A3B, Qwen3-235B-A22B, embeddings, pooled-output variants, parsers, quantization, PP/DP/EP/CP, EAGLE3, NPU/XPU/MLX, and low-latency docs. It is based on SGLang mainline around `b3e6cf60a` and sgl-cookbook around `816bad5`.

The full skill-side source dossier is `skills/model-optimization/sglang/sglang-qwen3-core-optimization/references/pr-history.md`. This README keeps the model-history view in sync and keeps a per-PR motivation/implementation/code-fragment summary instead of a title-only list.

## Code Surfaces

- `python/sglang/srt/models/qwen3.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `python/sglang/srt/layers/moe/`
- `python/sglang/srt/layers/quantization/`
- `python/sglang/srt/layers/attention/`
- `python/sglang/srt/distributed/`
- `python/sglang/srt/function_call/qwen25_detector.py`
- `test/registered/models/test_qwen_models.py`
- `test/registered/4-gpu-models/test_qwen3_30b.py`
- `test/registered/stress/test_stress_qwen3_235b.py`
- `test/srt/models/test_lora_qwen3.py`
- `test/registered/backends/test_qwen3_fp4_trtllm_gen_moe.py`
- NPU Qwen3 tests under `test/registered/npu/`
- `docs/basic_usage/qwen3.md`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx`

## Merged PR Cards

### Bring-Up and Config Compatibility

- [#4693](https://github.com/sgl-project/sglang/pull/4693) added native Qwen3 and Qwen3MoE support. Motivation: SGLang needed `Qwen3ForCausalLM` and `Qwen3MoeForCausalLM` instead of Qwen2 fallback. Implementation: added `qwen3.py` and `qwen3_moe.py`, split packed QKV, applied Q/K RMSNorm before RoPE, and routed MoE through gate plus `FusedMoE`. Key fragment: `q, k, v = qkv.split(...)`, `q, k = self._apply_qk_norm(q, k)`, `self.experts = FusedMoE(...)`.
- [#6990](https://github.com/sgl-project/sglang/pull/6990) added Qwen3 embedding support. Motivation: Qwen3 embedding checkpoints used model-prefixed names. Implementation: prefix unprefixed names for embedding models. Key fragment: `if "Embedding" in self.config.name_or_path: name = add_prefix(name, "model")`.
- [#17535](https://github.com/sgl-project/sglang/pull/17535) fixed Qwen3 embedding rename heuristics. Motivation: fine-tuned embedding checkpoints may not contain `"Embedding"` in the model name. Implementation: only prefix root weights beginning with `layers.`, `embed_tokens.`, or `norm.`. Key fragment: `if not name.startswith("model.") and (name.startswith("layers.") or name.startswith("embed_tokens.") or name.startswith("norm."))`.
- [#17784](https://github.com/sgl-project/sglang/pull/17784) upgraded Transformers compatibility. Motivation: HF config fields moved around `rope_parameters` and nested text configs. Implementation: read `config.rope_parameters`, normalize legacy `rope_scaling["type"]`, and resolve text configs from thinker/llm/language/text subconfigs. Key fragment: `rope_theta = config.rope_parameters.get("rope_theta", 1000000.0)`.
- [#20931](https://github.com/sgl-project/sglang/pull/20931) fixed Qwen3 MoE RoPE compatibility. Motivation: old checkpoints can keep top-level `rope_theta`/`rope_scaling`. Implementation: use `get_rope_config(config)` and store `self.rope_theta` for fused qk_norm_rope. Key fragment: `rope_theta, rope_scaling = get_rope_config(config)`.
- [#22739](https://github.com/sgl-project/sglang/pull/22739) restored dense Qwen3 RoPE fallback. Motivation: JSON overrides could create `rope_parameters` without `rope_theta`. Implementation: check for `"rope_theta"` before using `rope_parameters`, else fall back to top-level fields. Key fragment: `"rope_theta" in config.rope_parameters`.

### MoE, DeepEP, EPLB, and Dispatch

- [#5917](https://github.com/sgl-project/sglang/pull/5917) enabled Qwen3 EP MoE. Motivation: Qwen3-235B-A22B-FP8 needed expert parallel serving. Implementation: choose `EPMoE` when EP is enabled and reuse the same class for expert mapping. Key fragment: `MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE`.
- [#6120](https://github.com/sgl-project/sglang/pull/6120) added Qwen3 DeepEP. Motivation: Qwen3 MoE needed DeepEP all-to-all dispatch. Implementation: select `DeepEPMoE`, build `DeepEPDispatcher`, run `select_experts`, dispatch, and combine. Key fragment: `self.deepep_dispatcher = DeepEPDispatcher(...)`.
- [#6121](https://github.com/sgl-project/sglang/pull/6121) added DP attention for Qwen2/3 MoE. Motivation: EP MoE deployments need DP attention. Implementation: use attention TP rank/size, introduce full/scattered FFN input modes, and communicate through `dp_gather_partial`/`dp_scatter`. Key fragment: `self.num_heads = self.total_num_heads // attn_tp_size`.
- [#6533](https://github.com/sgl-project/sglang/pull/6533) added EPLB for Qwen3. Motivation: Qwen3 MoE needed redundant experts and expert-location metadata. Implementation: create MoE with redundant experts, collect per-layer MoE weights, and pass `ExpertLocationDispatchInfo` into top-k. Key fragment: `ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)`.
- [#6709](https://github.com/sgl-project/sglang/pull/6709) fixed Qwen3 MoE PP plus EPLB. Motivation: non-local PP layers are `PPMissingLayer`. Implementation: collect routed expert weights only for `range(self.start_layer, self.end_layer)`. Key fragment: `for layer_id in range(self.start_layer, self.end_layer)`.
- [#6818](https://github.com/sgl-project/sglang/pull/6818) fixed dynamic EPLB weight references. Motivation: EPLB could reference weights too early or on wrong local layers. Implementation: use lazy expert-weight collection. Key fragment: `self._routed_experts_weights_of_layer = LazyValue(lambda: {...})`.
- [#6964](https://github.com/sgl-project/sglang/pull/6964) added approximate and exact expert distribution collection. Motivation: EPLB needed exact top-k stats and DeepEP approximate stats. Implementation: exact mode uses `scatter_add_`; Qwen3 wraps top-k in the global recorder. Key fragment: `self._data[layer_idx, :].scatter_add_(...)`.
- [#7580](https://github.com/sgl-project/sglang/pull/7580) moved EPLB files. Motivation: expert location/distribution became a subsystem. Implementation: move helpers under `sglang.srt.eplb`. Key fragment: `from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo`.
- [#8448](https://github.com/sgl-project/sglang/pull/8448) added EPLB support in FusedMoE. Motivation: FusedMoE loader did not understand expert-location metadata. Implementation: map logical expert ids to all physical expert ids. Key fragment: `logical_to_all_physical(self.layer_id, expert_id)`.
- [#13715](https://github.com/sgl-project/sglang/pull/13715) fixed EPLB plus FP4 compatibility. Motivation: ModelOpt FP4 scale and swizzled tensors are not local expert weights. Implementation: filter MoE weight params by local-expert leading dimension. Key fragment: `x.data.shape[0] == num_local_experts`.
- [#6820](https://github.com/sgl-project/sglang/pull/6820) restored token padding optimization. Motivation: Qwen3 MoE failed to pass non-padded token counts to top-k. Implementation: thread `num_token_non_padded` into `select_experts` and `fused_topk`. Key fragment: `num_token_non_padded=forward_batch.num_token_non_padded`.
- [#7222](https://github.com/sgl-project/sglang/pull/7222) enabled DP attention with DeepEP auto mode. Motivation: DeepEP auto had been blocked with DP attention. Implementation: resolve mode from `forward_batch.is_extend_in_batch` and pass full `forward_batch` to experts. Key fragment: `self.deepep_mode.resolve(forward_batch.is_extend_in_batch)`.
- [#7723](https://github.com/sgl-project/sglang/pull/7723) fixed FlashInfer MoE flag wiring. Motivation: Qwen MoE did not pass `enable_flashinfer_moe`. Implementation: pass FlashInfer kwargs only when enabled. Key fragment: `dict(enable_flashinfer_moe=True, enable_ep_moe=...)`.
- [#7966](https://github.com/sgl-project/sglang/pull/7966) refactored `select_experts`. Motivation: routing logic was duplicated and hard to extend. Implementation: introduce `TopKOutput` and `TopK`; MoE runners consume `topk_output`. Key fragment: `class TopKOutput(NamedTuple): ...`.
- [#8421](https://github.com/sgl-project/sglang/pull/8421) simplified DeepEP output. Motivation: model files owned too much dispatch/combine code. Implementation: `DeepEPMoE.forward` owns dispatch, expert compute, and combine. Key fragment: `dispatch_output = self.dispatch(...)`, `hidden_states = self.moe_impl(dispatch_output)`.
- [#8658](https://github.com/sgl-project/sglang/pull/8658) updated MoE parallelism arguments. Motivation: old EP/DeepEP booleans did not scale to multiple A2A backends. Implementation: add `MoeA2ABackend` and map deprecated flags. Key fragment: `class MoeA2ABackend(Enum): ... DEEPEP = "deepep"`.
- [#8751](https://github.com/sgl-project/sglang/pull/8751) reduced Slime update-weight overhead. Motivation: repeated parameter traversal and non-local expert loads were expensive. Implementation: cache `params_dict`, skip non-local expert weights, and lazily initialize expert maps. Key fragment: `self._cached_params_dict = dict(self.named_parameters())`.
- [#9338](https://github.com/sgl-project/sglang/pull/9338) refactored TopK output selection. Motivation: Qwen3 MoE needed clean backend-specific output formats. Implementation: choose `TRITON_KERNEL`, `BYPASSED`, or `STANDARD` based on backend/quantization. Key fragment: `output_format = TopKOutputFormat.BYPASSED`.

### PP and Tied Embeddings

- [#6250](https://github.com/sgl-project/sglang/pull/6250) added PP for Qwen2/Qwen3. Motivation: large Qwen3 models need pipeline layer partitioning. Implementation: add `PPMissingLayer`, `PPProxyTensors`, layer ranges, and PP-aware loader skips. Key fragment: `self.layers, self.start_layer, self.end_layer = make_layers(..., pp_rank=..., pp_size=...)`.
- [#6546](https://github.com/sgl-project/sglang/pull/6546) added tied-weight support in Qwen PP. Motivation: last PP rank owns LM head but not embeddings. Implementation: send embedding weights from first rank to last rank and copy into `lm_head`. Key fragment: `self.lm_head.weight.copy_(emb_token_weight)`.
- [#15223](https://github.com/sgl-project/sglang/pull/15223) fixed Qwen3 PP load. Motivation: send/recv rank and shape were wrong. Implementation: send to `world_size - 1`, recv into `self.lm_head.weight.shape`. Key fragment: `dst=self.pp_group.world_size - 1`.
- [#15890](https://github.com/sgl-project/sglang/pull/15890) fixed tied embedding logic under PP. Motivation: last PP rank filtered out `model.embed_tokens.weight`. Implementation: load embedding weight directly into `lm_head.weight` on last tied PP rank. Key fragment: `if name == "model.embed_tokens.weight" and self.pp_group.is_last_rank`.

### DP Attention, TBO, CP, and Speculative Paths

- [#6598](https://github.com/sgl-project/sglang/pull/6598) added Qwen3 MoE TBO. Motivation: Qwen3-235B needed overlap across DP attention and DeepEP normal. Implementation: split layer execution into `op_*` stages and use `MaybeTboDeepEPDispatcher`. Key fragment: `_compute_moe_qwen3_layer_operations_strategy_tbo(...)`.
- [#6652](https://github.com/sgl-project/sglang/pull/6652) fixed Qwen3 TBO and DP LM-head. Motivation: logits had to use the attention TP group. Implementation: `ParallelLMHead(..., use_attn_tp_group=...)`. Key fragment: `use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"]`.
- [#7681](https://github.com/sgl-project/sglang/pull/7681) added dense Qwen3 DP attention. Motivation: dense Qwen3 needed TP8 DP8 support. Implementation: QKV and output projections use attention TP rank/size, and decoder uses `LayerCommunicator`. Key fragment: `tp_rank=attn_tp_rank, tp_size=attn_tp_size`.
- [#8280](https://github.com/sgl-project/sglang/pull/8280) enhanced DP attention. Motivation: padding, buffer allocation, and communication needed cleanup. Implementation: `DPPaddingMode`, lazy gathered buffers, and DP+EAGLE graph sizing. Key fragment: `if sum_len * 2 > max_len * get_attention_dp_size(): return cls.MAX_LEN`.
- [#9101](https://github.com/sgl-project/sglang/pull/9101) added reduce-scatter for DP attention padding. Motivation: max-padding DP attention needed post-MoE reduce-scatter. Implementation: `LayerCommunicator.should_use_reduce_scatter` controls Qwen3 MoE MLP communication. Key fragment: `hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)`.
- [#12002](https://github.com/sgl-project/sglang/pull/12002) added EAGLE3 DP attention for Qwen3 MoE. Motivation: large EP deployments needed EAGLE3 target capture. Implementation: gather/clone captured residuals and use attention TP context in EAGLE worker. Key fragment: `captured_last_layer_outputs.append(gathered_last_layer_output)`.
- [#18233](https://github.com/sgl-project/sglang/pull/18233) added Qwen3 MoE context parallel. Motivation: long-context prefill needed attention CP and MoE topology integration. Implementation: allgather/rerange KV cache and split `q` into prev/next chunks. Key fragment: `cp_all_gather_rerange_kv_cache(...)`.
- [#21195](https://github.com/sgl-project/sglang/pull/21195) enabled Qwen3 tests. Motivation: Qwen3-30B CP test could return to CI. Implementation: restore EP all-reduce for `ep_size > 1` before TP all-reduce. Key fragment: `moe_expert_parallel_all_reduce(final_hidden_states)`.
- [#22003](https://github.com/sgl-project/sglang/pull/22003) supported `moe_dp_size=1` with different attention CP sizes. Motivation: production wants CP only for attention. Implementation: map `_MOE_DP` to `_ATTN_CP`, add `ScatterMode.MOE_FULL`, gather before MoE and slice after. Key fragment: `hidden_states.narrow(0, moe_cp_rank * max_tokens_per_rank, actual_local_tokens)`.
- [#22358](https://github.com/sgl-project/sglang/pull/22358) added DFLASH capture. Motivation: explicit aux-hidden capture was needed for DFLASH data collection. Implementation: `set_dflash_layers_to_capture` on dense/MoE Qwen3. Key fragment: `self.model.layers_to_capture = [val + 1 for val in layer_ids]`.

### Quantization and FlashInfer/TRTLLM

- [#7912](https://github.com/sgl-project/sglang/pull/7912) added Qwen ModelOpt FP8/NVFP4 support. Motivation: one-line ModelOpt launch. Implementation: recursive `common_group_size` and KV scale remap. Key fragment: `name = maybe_remap_kv_scale_name(name, params_dict)`.
- [#8036](https://github.com/sgl-project/sglang/pull/8036) added FlashInfer blockscale FP8 MoE. Motivation: low-latency FP8 MoE. Implementation: `FlashInferEPMoE` calls `trtllm_fp8_block_scale_moe`. Key fragment: `return trtllm_fp8_block_scale_moe(..., routing_method_type=2)`.
- [#8450](https://github.com/sgl-project/sglang/pull/8450) added FlashInfer blockscale FP8 for TP MoE. Motivation: `#8036` only covered EP. Implementation: add `FlashInferFusedMoE` and version-gated backend selection. Key fragment: `return FlashInferFusedMoE if should_use_flashinfer_trtllm_moe() else FusedMoE`.
- [#9973](https://github.com/sgl-project/sglang/pull/9973) optimized Qwen3 MoE with FlashInfer fused all-reduce. Motivation: AllReduce plus FusedNormAdd were major profile costs. Implementation: SM90/SM100 fusion for <=4096 tokens and `_sglang_needs_allreduce_fusion` marker. Key fragment: `hidden_states._sglang_needs_allreduce_fusion = True`.
- [#13489](https://github.com/sgl-project/sglang/pull/13489) enabled FlashInfer TRTLLM-GEN-MoE for Qwen3. Motivation: Qwen3 FP8 MoE should use TRTLLM-GEN on SM100 when suitable. Implementation: pass `RoutingMethodType.Renormalize` and auto-select `flashinfer_trtllm`. Key fragment: `self.moe_runner_backend = "flashinfer_trtllm"`.
- [#14093](https://github.com/sgl-project/sglang/pull/14093) fused FP8 KV-cache write for TRTLLM MHA. Motivation: remove four tiny FP8 KV kernels. Implementation: Triton fused quant+write kernel and skip generic cache write. Key fragment: `self._fused_fp8_set_kv_buffer(...); k = None; v = None`.
- [#18189](https://github.com/sgl-project/sglang/pull/18189) fixed Qwen3-235B NVFP4 launch. Motivation: ignored q/k/v BF16 modules needed packed mapping. Implementation: add `packed_modules_mapping`. Key fragment: `"qkv_proj": ["q_proj", "k_proj", "v_proj"]`.

### QK-Norm, RoPE, and KV Store Fusion

- [#7740](https://github.com/sgl-project/sglang/pull/7740) added two-stream Q/K norm. Motivation: overlap Q and K RMSNorm in CUDA graph capture. Implementation: pass `alt_stream` and run K norm on the alternate stream. Key fragment: `with torch.cuda.stream(self.alt_stream): k_by_head = self.k_norm(k_by_head)`.
- [#10749](https://github.com/sgl-project/sglang/pull/10749) fused KV write into RoPE. Motivation: remove separate BF16 KV-cache write. Implementation: pass `FusedSetKVBufferArg` into RoPE and call attention with `save_kv_cache=False`. Key fragment: `save_kv_cache=not enable_fused_set_kv_buffer(forward_batch)`.
- [#13998](https://github.com/sgl-project/sglang/pull/13998) added fused QK-norm/RoPE for Qwen3 MoE. Motivation: many decode layers made separate qk_norm and RoPE expensive. Implementation: CUDA fused kernel gated on non-MRoPE and head dim 64/128/256. Key fragment: `self.use_fused_qk_norm_rope = ...`.
- [#15835](https://github.com/sgl-project/sglang/pull/15835) added JIT fused QK norm. Motivation: shared, lightweight fused q/k norm was needed across models. Implementation: `fused_inplace_qknorm` and shared `apply_qk_norm`. Key fragment: `fused_inplace_qknorm(...); return q, k`.
- [#19059](https://github.com/sgl-project/sglang/pull/19059) added JIT fused qknorm_rope. Motivation: replace AOT fused kernel and fix NeoX mask behavior. Implementation: register `fused_qk_norm_rope_out` custom op and gate Qwen3 MoE init. Key fragment: `@register_custom_op(op_name="fused_qk_norm_rope_out", mutates_args=["qkv"])`.
- [#21654](https://github.com/sgl-project/sglang/pull/21654) optimized fused qknorm_rope. Motivation: remove duplicate `__sincosf` and `powf`. Implementation: template on head dim/interleave/YaRN and recurse frequency. Key fragment: `template <int head_dim, bool interleave, bool yarn>`.

### LoRA, EAGLE3, Prefill, and Shared Plumbing

- [#7312](https://github.com/sgl-project/sglang/pull/7312) added Qwen3 LoRA hidden dims. Motivation: packed projections needed correct adapter dimensions. Implementation: temporary model-local `get_hidden_dim`. Key fragment: `elif module_name == "gate_up_proj": return self.config.hidden_size, self.config.intermediate_size`.
- [#8987](https://github.com/sgl-project/sglang/pull/8987) fixed default LoRA hidden-dim logic. Motivation: duplicated model overrides were wrong. Implementation: centralize in `lora/utils.py`. Key fragment: `if module_name == "qkv_proj": return (config.hidden_size, None)`.
- [#7634](https://github.com/sgl-project/sglang/pull/7634) added layer-wise prefill. Motivation: PD multiplexing needed partial decoder execution. Implementation: `ForwardMode.SPLIT_PREFILL` and Qwen3 `forward_split_prefill`. Key fragment: `self.model.forward_split_prefill(..., (forward_batch.split_index, next_split_index))`.
- [#7745](https://github.com/sgl-project/sglang/pull/7745) added EAGLE3 for Qwen. Motivation: Qwen draft models needed aux hidden capture. Implementation: save `hidden_states + residual` at configured layers and pass to `LogitsProcessor`. Key fragment: `aux_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)`.
- [#10975](https://github.com/sgl-project/sglang/pull/10975) generalized `--mem-fraction-static` heuristics. Motivation: defaults needed GPU-memory-aware sizing. Implementation: reserve memory for chunked prefill, CUDA graph, DP attention, and speculative modes. Key fragment: `reserved_mem += self.cuda_graph_max_bs * self.dp_size * 3`.
- [#10911](https://github.com/sgl-project/sglang/pull/10911) added Qwen3-Omni thinker plumbing. Motivation: Omni reused Qwen3 MoE language model code. Implementation: MRoPE dispatch for `qwen3_omni_moe` and `decoder_layer_type` parameter. Key fragment: `decoder_layer_type=Qwen3MoeDecoderLayer`.

### Ascend NPU, XPU, and MLX

- [#10574](https://github.com/sgl-project/sglang/pull/10574) optimized Qwen3 on Ascend. Motivation: NPU memory format and CMO prefetch. Implementation: format-cast W8A8 weights, add CMO prefetch, and pass MLP weights as cache. Key fragment: `torch_npu.npu_format_cast(layer.weight.data, 29)`.
- [#12078](https://github.com/sgl-project/sglang/pull/12078) added broader Ascend Qwen optimization. Motivation: fix W8A8 memory, CMO deadlock, EPLB, NPU graph, and fuseEP. Implementation: add `ASCEND_FUSEEP`, `NpuFuseEPMoE`, and NPU `split_qkv_rmsnorm_rope`. Key fragment: `ASCEND_FUSEEP = "ascend_fuseep"`.
- [#15203](https://github.com/sgl-project/sglang/pull/15203) added NPU GPTQ quantization. Motivation: Qwen3 GPTQ on Ascend. Implementation: `GPTQLinearAscendMethod` and `npu_weight_quant_batchmatmul`; MoE GPTQ remains unsupported. Key fragment: `return GPTQLinearAscendMethod(self)`.
- [#15390](https://github.com/sgl-project/sglang/pull/15390) fixed NPU Qwen3 PP. Motivation: local PP first layer is not always layer 0. Implementation: generate RoPE cos/sin when `layer_id == token_to_kv_pool.start_layer`. Key fragment: `self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer`.
- [#16115](https://github.com/sgl-project/sglang/pull/16115) fixed NPU DP LM-head. Motivation: split-qkv-rmsnorm-rope args and rotary dtype fallback were wrong. Implementation: native fallback for BF16 query plus float cache; LM-head uses attention TP group. Key fragment: `use_attn_tp_group=get_global_server_args().enable_dp_lm_head`.
- [#19532](https://github.com/sgl-project/sglang/pull/19532) fixed NPU speculative inference. Motivation: EAGLE3 target verify made decode appear extend-like. Implementation: use `is_extend_or_draft_extend_or_mixed()`. Key fragment: `forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()`.

## Open Radar Cards

- [#9147](https://github.com/sgl-project/sglang/pull/9147) Qwen3-MoE W4AFP8. Motivation: load W4AFP8 block-quantized Qwen3 MoE. Implementation draft: choose W4AFP8 TP/EP MoE method, interleave scales, call `cutlass_w4a8_moe`. Key fragment: `return cutlass_w4a8_moe(..., w1_q=layer.w13_weight, w2_q=layer.w2_weight, topk_ids_=topk_ids)`. Risk: stale against current TopK/MoeA2ABackend.
- [#20127](https://github.com/sgl-project/sglang/pull/20127) tied embeddings for Qwen MoE and Qwen3Next. Motivation: tied checkpoints without `lm_head.weight` can leave random heads. Implementation draft: single-rank tied models set `self.lm_head = self.model.embed_tokens`; PP last rank copies embedding weight. Key fragment: `self.lm_head = self.model.embed_tokens`.
- [#20474](https://github.com/sgl-project/sglang/pull/20474) Intel XPU Qwen3. Motivation: XPU layernorm/MRoPE. Implementation draft: use XPU EU count and Triton RoPE. Key fragment: `return torch.xpu.get_device_properties(device).gpu_eu_count`.
- [#20520](https://github.com/sgl-project/sglang/pull/20520) NPU TP communication compression. Motivation: reduce Qwen3 NPU prefill communication. Implementation draft: dynamic int8 quantize, allgather tensor and scale, dequantize/reduce. Key fragment: `x_q, scale = npu_dynamic_quant(x, dst_type=torch.int8)`.
- [#21412](https://github.com/sgl-project/sglang/pull/21412) dense Qwen3 old-style RoPE compatibility. Motivation: dense path had the same fallback issue as Qwen3 MoE. Implementation draft: use `get_rope_config(config)`. Key fragment: `rope_theta, rope_scaling = get_rope_config(config)`.
- [#21770](https://github.com/sgl-project/sglang/pull/21770) Apple MLX Qwen3 tests. Motivation: initial MLX coverage. Implementation draft: launch with `SGLANG_USE_MLX=1` and `enable_thinking=False`. Key fragment: `env["SGLANG_USE_MLX"] = "1"`.
- [#22529](https://github.com/sgl-project/sglang/pull/22529) Qwen3 sliding-window attention. Motivation: alternating sliding/full attention in Qwen3-like models. Implementation draft: convert HF inclusive window to SGLang exclusive window and read `layer_types`. Key fragment: `is_sliding = layer_types[layer_id] == "sliding_attention"`.
- [#22674](https://github.com/sgl-project/sglang/pull/22674) shared Qwen NPU quant mappings. Motivation: GDN packed names were missing. Implementation draft: add `in_proj_qkvz` and `in_proj_ba`. Key fragment: `"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"]`.
- [#22837](https://github.com/sgl-project/sglang/pull/22837) Qwen3 reasoning detector tool-call fix. Motivation: `<tool_call>` before `</think>` was swallowed into reasoning content. Implementation draft: pass `tool_start_token="<tool_call>"` to the base detector. Key fragment: `tool_start_token="<tool_call>"`.
- [#23372](https://github.com/sgl-project/sglang/pull/23372) NPU speculative decoding CI. Motivation: validate EAGLE3/NEXTN and `ascend_fuseep`. Implementation draft: Qwen3-32B W8A8 + EAGLE3 PD test. Key fragment: `"--speculative-attention-mode", "decode"`.
- [#23397](https://github.com/sgl-project/sglang/pull/23397) dense deterministic math. Motivation: alignment rollout logprobs should match Megatron scoring. Implementation draft: disable fusion, force BF16 dense math, use FP32 q/k norm weights. Key fragment: `get_on_policy_rms_norm_kwargs(weight_dtype=torch.float32)`.
- [#23434](https://github.com/sgl-project/sglang/pull/23434) Qwen3 pooled-output embeddings. Motivation: reranker/sequence-classification variants lacked `get_input_embeddings`. Implementation draft: forward to wrapped model. Key fragment: `return self.model.get_input_embeddings()`.

## Docs and Cookbook Cards

- [#22429](https://github.com/sgl-project/sglang/pull/22429) added Qwen3-32B/8B Ascend low-latency docs. Motivation: tested A3/A2 launch recipes. Key command fragment: `--attention-backend ascend --device npu --quantization modelslim --speculative-algorithm EAGLE3 --dtype bfloat16`.
- [#22446](https://github.com/sgl-project/sglang/pull/22446) added Qwen3-30B-A3B low-latency docs. Motivation: Ascend low-latency recipe for A3B. Key command fragment: `--tp-size 2 --mem-fraction-static 0.6 --attention-backend ascend`.
- [#22687](https://github.com/sgl-project/sglang/pull/22687) fixed Qwen3-8B/32B docs. Motivation: remove wrong launch lines. Key diff fragment: `-export HCCL_BUFFSIZE=400`.
- [#22450](https://github.com/sgl-project/sglang/pull/22450) open Qwen3-14B Ascend low-latency docs. Motivation: add A3 Qwen3-14B recipes. Key command fragment: `--quantization modelslim --sampling-backend ascend --schedule-conservativeness 0.01`.
- [sgl-cookbook #74](https://github.com/sgl-project/sgl-cookbook/pull/74) refreshed Qwen3 AMD and tool-calling docs. Motivation: cookbook reproduction context. Implementation: markdown/command updates, not runtime support.
- [sgl-cookbook #245](https://github.com/sgl-project/sgl-cookbook/pull/245) refreshed Qwen cookbook content after Qwen3/Qwen3.5/Qwen3-Next changes.

## Next Priorities

1. Track `#22837`, `#22529`, `#20127`, `#20520`, and `#9147`.
2. Regression-test Qwen3.5 and Qwen3-Next whenever shared Qwen3 helpers change.
3. Keep using `skills/model-optimization/sglang/sglang-model-pr-diff-dossier` for any new model PR history.
