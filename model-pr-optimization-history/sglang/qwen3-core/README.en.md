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
3. Keep using `skills/model-optimization/model-pr-diff-dossier` for any new model PR history.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Qwen3 Core` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-03-23 | [#4693](https://github.com/sgl-project/sglang/pull/4693) | merged | [Model] Adding Qwen3 and Qwen3MoE | model wrapper, attention/backend, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2025-04-30 | [#5917](https://github.com/sgl-project/sglang/pull/5917) | merged | [qwen3] support qwen3 ep moe | model wrapper, MoE/router | `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-05-08 | [#6120](https://github.com/sgl-project/sglang/pull/6120) | merged | Support qwen3 deepep | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2025-05-08 | [#6121](https://github.com/sgl-project/sglang/pull/6121) | merged | feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088 | model wrapper, attention/backend, MoE/router, tests/benchmarks | `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/dp_attention.py` |
| 2025-05-13 | [#6250](https://github.com/sgl-project/sglang/pull/6250) | merged | Add pipeline parallelism for Qwen2 and Qwen3 Model | model wrapper, MoE/router, tests/benchmarks | `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3.py` |
| 2025-05-22 | [#6533](https://github.com/sgl-project/sglang/pull/6533) | merged | support eplb for qwen3 | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/managers/expert_distribution.py` |
| 2025-05-23 | [#6546](https://github.com/sgl-project/sglang/pull/6546) | merged | added support for tied weights in qwen pipeline parallelism | model wrapper, tests/benchmarks | `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py` |
| 2025-05-25 | [#6598](https://github.com/sgl-project/sglang/pull/6598) | merged | qwen3moe support two batch overlap | model wrapper, MoE/router, tests/benchmarks | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/operations_strategy.py`, `test/srt/test_two_batch_overlap.py` |
| 2025-05-27 | [#6652](https://github.com/sgl-project/sglang/pull/6652) | merged | Fix qwen3 tbo/dp-lm-head | model wrapper, MoE/router | `python/sglang/srt/two_batch_overlap.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-05-28 | [#6709](https://github.com/sgl-project/sglang/pull/6709) | merged | Fix PP for Qwen3 MoE | model wrapper, MoE/router, tests/benchmarks | `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-06-02 | [#6818](https://github.com/sgl-project/sglang/pull/6818) | merged | Fix wrong weight reference in dynamic EPLB | model wrapper, MoE/router | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-06-03 | [#6820](https://github.com/sgl-project/sglang/pull/6820) | merged | Fix Qwen3MoE missing token padding optimization | model wrapper, MoE/router | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-06-08 | [#6964](https://github.com/sgl-project/sglang/pull/6964) | merged | Support both approximate and exact expert distribution collection | model wrapper, MoE/router | `python/sglang/srt/managers/expert_distribution.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-06-09 | [#6990](https://github.com/sgl-project/sglang/pull/6990) | merged | support qwen3 emebedding | model wrapper, tests/benchmarks | `python/sglang/srt/models/qwen3.py`, `test/srt/models/test_embedding_models.py` |
| 2025-06-16 | [#7222](https://github.com/sgl-project/sglang/pull/7222) | merged | DP Attention with Auto DeepEP Dispatch | model wrapper, MoE/router, scheduler/runtime, tests/benchmarks | `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-06-18 | [#7312](https://github.com/sgl-project/sglang/pull/7312) | merged | Add get_hidden_dim to qwen3.py for correct lora | model wrapper, scheduler/runtime, tests/benchmarks | `test/srt/models/lora/test_lora_qwen3.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/test/runners.py` |
| 2025-06-27 | [#7580](https://github.com/sgl-project/sglang/pull/7580) | merged | Move files related to EPLB | model wrapper, MoE/router, scheduler/runtime, tests/benchmarks | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-29 | [#7634](https://github.com/sgl-project/sglang/pull/7634) | merged | [Feature] Layer-wise Prefill | model wrapper, MoE/router, scheduler/runtime | `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma.py` |
| 2025-07-01 | [#7681](https://github.com/sgl-project/sglang/pull/7681) | merged | support qwen3 dense model dp attention | model wrapper | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py` |
| 2025-07-02 | [#7723](https://github.com/sgl-project/sglang/pull/7723) | merged | [Bug] add flashinfer bool check for fusedmoe in Qwen moe models | model wrapper, MoE/router | `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-07-03 | [#7740](https://github.com/sgl-project/sglang/pull/7740) | merged | [optimize] add two stream norm for qwen3 | model wrapper, MoE/router | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2.py` |
| 2025-07-03 | [#7745](https://github.com/sgl-project/sglang/pull/7745) | merged | [feat] Support EAGLE3 for Qwen | model wrapper, MoE/router | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2025-07-09 | [#7912](https://github.com/sgl-project/sglang/pull/7912) | merged | Qwen FP8/NVFP4 ModelOPT Quantization support | model wrapper, quantization | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py` |
| 2025-07-11 | [#7966](https://github.com/sgl-project/sglang/pull/7966) | merged | [1/N] MoE Refactor: refactor `select_experts` | model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-07-15 | [#8036](https://github.com/sgl-project/sglang/pull/8036) | merged | [NVIDIA] Add Flashinfer MoE blockscale fp8 backend | model wrapper, MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-07-23 | [#8280](https://github.com/sgl-project/sglang/pull/8280) | merged | DP Enhancement | model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks | `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2025-07-27 | [#8421](https://github.com/sgl-project/sglang/pull/8421) | merged | [3/N] MoE Refactor: Simplify DeepEP Output | model wrapper, MoE/router | `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-07-28 | [#8448](https://github.com/sgl-project/sglang/pull/8448) | merged | Support EPLB in FusedMoE | model wrapper, MoE/router, kernel | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/eplb/expert_location.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-07-28 | [#8450](https://github.com/sgl-project/sglang/pull/8450) | merged | [NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE | model wrapper, MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/fp8.py` |
| 2025-08-01 | [#8658](https://github.com/sgl-project/sglang/pull/8658) | merged | [5/N] MoE Refactor: Update MoE parallelism arguments | model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config | `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-08-04 | [#8751](https://github.com/sgl-project/sglang/pull/8751) | merged | [1/3] Optimize Slime Update Weights: Remove QWen3MOE Load Weight Overhead | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-08-08 | [#8987](https://github.com/sgl-project/sglang/pull/8987) | merged | Fix incorrect default get_hidden_dim logic | model wrapper | `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/lora/utils.py`, `python/sglang/srt/models/granite.py` |
| 2025-08-12 | [#9101](https://github.com/sgl-project/sglang/pull/9101) | merged | Feature: support qwen and llama4 reducescatter for dp attention padding | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/llama4.py` |
| 2025-08-13 | [#9147](https://github.com/sgl-project/sglang/pull/9147) | open | support Qwen3-MoE-w4afp8 | model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `sgl-router/src/routers/pd_router.rs`, `python/sglang/srt/models/phi4mm_utils.py`, `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` |
| 2025-08-19 | [#9338](https://github.com/sgl-project/sglang/pull/9338) | merged | Refactor TopK to ensure readability and extensibility | model wrapper, MoE/router, kernel | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-09-03 | [#9973](https://github.com/sgl-project/sglang/pull/9973) | merged | Optimize Qwen3-moe model by using flashinfer fused allreduce | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2025-09-17 | [#10574](https://github.com/sgl-project/sglang/pull/10574) | merged | [Ascend]optimize Qwen3 on Ascend | model wrapper, quantization, scheduler/runtime | `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/layers/communicator.py` |
| 2025-09-22 | [#10749](https://github.com/sgl-project/sglang/pull/10749) | merged | Fuse write kv buffer into rope for qwen3 moe & bailing moe | model wrapper, MoE/router | `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py` |
| 2025-09-25 | [#10911](https://github.com/sgl-project/sglang/pull/10911) | merged | model: qwen3-omni (thinker-only) | model wrapper, MoE/router, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-09-27 | [#10975](https://github.com/sgl-project/sglang/pull/10975) | merged | Use more general heuristics to set the default value of --mem-fraction-static | model wrapper, attention/backend, tests/benchmarks | `python/sglang/srt/server_args.py`, `python/sglang/srt/managers/io_struct.py`, `.github/workflows/pr-test.yml` |
| 2025-10-23 | [#12002](https://github.com/sgl-project/sglang/pull/12002) | merged | Eagle3 DP attention for Qwen3 MoE | model wrapper, attention/backend, MoE/router, scheduler/runtime, tests/benchmarks | `test/srt/test_eagle_dp_attention.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/communicator.py` |
| 2025-10-24 | [#12078](https://github.com/sgl-project/sglang/pull/12078) | merged | [Ascend] qwen optimization | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime | `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` |
| 2025-11-18 | [#13489](https://github.com/sgl-project/sglang/pull/13489) | merged | Flashinfer TRTLLM-GEN-MoE + Qwen3 | model wrapper, MoE/router | `python/sglang/srt/server_args.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-11-21 | [#13715](https://github.com/sgl-project/sglang/pull/13715) | merged | Fix EPLB + FP4 Quantization Compatibility Issue | model wrapper, MoE/router | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2025-11-26 | [#13998](https://github.com/sgl-project/sglang/pull/13998) | merged | [apply][2/2] Fused qk_norm_rope for Qwen3-MoE | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/server_args.py` |
| 2025-11-28 | [#14093](https://github.com/sgl-project/sglang/pull/14093) | merged | Add fused FP8 KV cache write kernel for TRTLLM MHA backend | model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py`, `test/manual/test_trtllm_fp8_kv_kernel.py`, `python/sglang/srt/layers/attention/trtllm_mha_backend.py` |
| 2025-12-15 | [#15203](https://github.com/sgl-project/sglang/pull/15203) | merged | [NPU] support GPTQ quantization on npu | model wrapper, quantization, tests/benchmarks | `python/sglang/srt/layers/quantization/gptq.py`, `test/srt/ascend/test_ascend_gptq.py`, `python/sglang/srt/models/qwen3.py` |
| 2025-12-16 | [#15223](https://github.com/sgl-project/sglang/pull/15223) | merged | [bug fix][pp] fix qwen3 model load | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2025-12-18 | [#15390](https://github.com/sgl-project/sglang/pull/15390) | merged | [NPU]qwen3 pp bugfix | model wrapper, MoE/router | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-12-25 | [#15835](https://github.com/sgl-project/sglang/pull/15835) | merged | [Feature] JIT Fused QK norm + qk norm clean up | model wrapper, MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/norm.cuh`, `python/sglang/jit_kernel/utils.py`, `python/sglang/jit_kernel/benchmark/bench_qknorm.py` |
| 2025-12-26 | [#15890](https://github.com/sgl-project/sglang/pull/15890) | merged | [PP] fix wrong weight logic for tie_word_embeddings model | model wrapper | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py` |
| 2025-12-30 | [#16115](https://github.com/sgl-project/sglang/pull/16115) | merged | [NPU][Bugfix] Fix qwen3 error when enable-dp-lm-head | model wrapper, MoE/router | `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-01-21 | [#17535](https://github.com/sgl-project/sglang/pull/17535) | merged | Update weight rename check for Qwen3 Embeddings | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-01-26 | [#17784](https://github.com/sgl-project/sglang/pull/17784) | merged | Upgrade transformers==5.3.0 | model wrapper, MoE/router, quantization, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/utils/hf_transformers_utils.py`, `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/models/gemma3_causal.py` |
| 2026-02-03 | [#18189](https://github.com/sgl-project/sglang/pull/18189) | merged | [ModelOpt] Fix broken Qwen3-235B-A22B-Instruct-2507-NVFP4 launch | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-02-04 | [#18233](https://github.com/sgl-project/sglang/pull/18233) | merged | Support Qwen3 MoE context parallel | model wrapper, attention/backend, MoE/router, scheduler/runtime, tests/benchmarks | `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/test/attention/test_flashattn_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py` |
| 2026-02-20 | [#19059](https://github.com/sgl-project/sglang/pull/19059) | merged | [jit_kernel] Add fused_qknorm_rope JIT kernel | model wrapper, MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` |
| 2026-02-28 | [#19532](https://github.com/sgl-project/sglang/pull/19532) | merged | [NPU] bugs fix: fix a condition bug when using speculative inference on Qwen3 and Qwen3 moe | model wrapper, MoE/router | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-08 | [#20127](https://github.com/sgl-project/sglang/pull/20127) | open | [Qwen] Handle tie_word_embeddings for Qwen MoE and Qwen3Next | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-12 | [#20474](https://github.com/sgl-project/sglang/pull/20474) | open | Intel XPU: Qwen3 support (layernorm/MRoPE) + test_qwen3 | attention/backend, tests/benchmarks | `test/srt/xpu/test_qwen3.py`, `docker/xpu.Dockerfile`, `python/sglang/srt/layers/rotary_embedding/mrope.py` |
| 2026-03-13 | [#20520](https://github.com/sgl-project/sglang/pull/20520) | open | [NPU]TP Communications compression For Qwen3 models for NPU | model wrapper, quantization, tests/benchmarks, docs/config | `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py`, `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py`, `python/sglang/srt/distributed/device_communicators/npu_communicator.py` |
| 2026-03-19 | [#20931](https://github.com/sgl-project/sglang/pull/20931) | merged | [Bugifx] qwen3 rope parameter compatibility | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-23 | [#21195](https://github.com/sgl-project/sglang/pull/21195) | merged | Enable the qwen3 test | model wrapper, MoE/router, tests/benchmarks | `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-25 | [#21412](https://github.com/sgl-project/sglang/pull/21412) | open | [Bugfix] Fix Qwen3 RoPE config compatibility for old-style checkpoints | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-03-30 | [#21654](https://github.com/sgl-project/sglang/pull/21654) | merged | [jit_kernel] Optimize fused_qknorm_rope: deduplicate sincosf for interleave RoPE | model wrapper, MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`, `python/sglang/jit_kernel/fused_qknorm_rope.py` |
| 2026-03-31 | [#21770](https://github.com/sgl-project/sglang/pull/21770) | open | [Apple][MLX][Test] Add Qwen3 correctness and accuracy tests for Apple Silicon | model wrapper, tests/benchmarks | `test/registered/models/test_qwen3_mlx_correctness.py`, `test/registered/models/test_qwen3_mlx_accuracy.py` |
| 2026-04-03 | [#22003](https://github.com/sgl-project/sglang/pull/22003) | merged | Support moe_dp_size = 1 for various attention_cp_size | model wrapper, attention/backend, MoE/router, tests/benchmarks | `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py` |
| 2026-04-08 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-09 | [#22429](https://github.com/sgl-project/sglang/pull/22429) | merged | [NPU]add Qwen3-32b and Qwen3-8b low latency md | docs/config | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-09 | [#22446](https://github.com/sgl-project/sglang/pull/22446) | merged | [NPU] add qwen3-30b-a3b low latency example | docs/config | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-09 | [#22450](https://github.com/sgl-project/sglang/pull/22450) | open | [NPU] Add Qwen3-14B low latency doc | docs/config | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-10 | [#22529](https://github.com/sgl-project/sglang/pull/22529) | open | [Model] Support sliding window attention for Qwen3 | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-04-13 | [#22674](https://github.com/sgl-project/sglang/pull/22674) | open | [NPU] Support Qwen3.5-MoE and Qwen3-Next quantization | misc | `python/sglang/srt/model_loader/loader.py` |
| 2026-04-13 | [#22687](https://github.com/sgl-project/sglang/pull/22687) | merged | [NPU]qwen3-8b and 32b md bugfix | docs/config | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-14 | [#22739](https://github.com/sgl-project/sglang/pull/22739) | merged | Restore Qwen3 rope config fallback | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-04-15 | [#22837](https://github.com/sgl-project/sglang/pull/22837) | open | [Bug] Qwen3 reasoning detector silently swallows tool_call when </think> is missing | tests/benchmarks | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-04-21 | [#23372](https://github.com/sgl-project/sglang/pull/23372) | open | [NPU] Add CI tests for Speculative Decoding | attention/backend, MoE/router, tests/benchmarks | `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` |
| 2026-04-21 | [#23397](https://github.com/sgl-project/sglang/pull/23397) | open | [alignment-sglang] PR3: Dense Deterministic Math | model wrapper, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks | `test/registered/core/test_tp_invariant_ops.py`, `test/registered/core/test_on_policy_wiring.py`, `test/registered/core/test_dense_deterministic_math.py` |
| 2026-04-22 | [#23434](https://github.com/sgl-project/sglang/pull/23434) | open | [Model] Qwen3ForPooledOutput: forward get_input_embeddings to inner model | model wrapper | `python/sglang/srt/models/qwen3_classification.py` |

### File-level PR diff reading notes

### PR #4693 - [Model] Adding Qwen3 and Qwen3MoE

- Link: https://github.com/sgl-project/sglang/pull/4693
- Status/date: `merged`, created 2025-03-23, merged 2025-04-18; author `yhyang201`.
- Diff scope read: `5` files, `+780/-14`; areas: model wrapper, attention/backend, MoE/router; keywords: config, quant, attention, expert, kv, moe, processor, spec, cache, cuda.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` added +423/-0 (423 lines); hunks: +# Adapted from qwen2_moe.py; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, Qwen3MoeAttention
  - `python/sglang/srt/models/qwen3.py` added +335/-0 (335 lines); hunks: +# Adapted from qwen2.py; symbols: Qwen3Attention, __init__, _apply_qk_norm, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +13/-11 (24 lines); hunks: from sglang.srt.managers.expert_distribution import ExpertDistributionRecorder; def __init__(; symbols: __init__, __init__
  - `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +5/-2 (7 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen2.py` modified +4/-1 (5 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2_moe.py`; keywords observed in patches: config, quant, attention, expert, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5917 - [qwen3] support qwen3 ep moe

- Link: https://github.com/sgl-project/sglang/pull/5917
- Status/date: `merged`, created 2025-04-30, merged 2025-04-30; author `laixinn`.
- Diff scope read: `2` files, `+16/-6`; areas: model wrapper, MoE/router; keywords: attention, config, expert, moe, processor, quant, topk, triton.
- Code diff details:
  - `python/sglang/srt/models/qwen2_moe.py` modified +8/-3 (11 lines); hunks: RowParallelLinear,; VocabParallelEmbedding,; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen3_moe.py` modified +8/-3 (11 lines); hunks: RowParallelLinear,; ParallelLMHead,; symbols: __init__, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: attention, config, expert, moe, processor, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6120 - Support qwen3 deepep

- Link: https://github.com/sgl-project/sglang/pull/6120
- Status/date: `merged`, created 2025-05-08, merged 2025-05-22; author `sleepcoo`.
- Diff scope read: `2` files, `+125/-8`; areas: model wrapper, MoE/router; keywords: moe, attention, config, deepep, expert, fp8, processor, quant, router, topk.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +121/-7 (128 lines); hunks: get_pp_group,; RowParallelLinear,; symbols: __init__, __init__, __init__, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +4/-1 (5 lines); hunks: def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`; keywords observed in patches: moe, attention, config, deepep, expert, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6121 - feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088

- Link: https://github.com/sgl-project/sglang/pull/6121
- Status/date: `merged`, created 2025-05-08, merged 2025-05-16; author `Fr4nk1inCs`.
- Diff scope read: `4` files, `+449/-70`; areas: model wrapper, attention/backend, MoE/router, tests/benchmarks; keywords: attention, moe, config, deepep, expert, kv, processor, quant, triton, cuda.
- Code diff details:
  - `python/sglang/srt/models/qwen2_moe.py` modified +227/-32 (259 lines); hunks: # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py; tensor_model_parallel_all_reduce,; symbols: __init__, forward, __init__, __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +221/-28 (249 lines); hunks: """Inference-only Qwen3MoE model compatible with HuggingFace weights."""; tensor_model_parallel_all_reduce,; symbols: __init__, __init__, __init__, forward
  - `python/sglang/srt/layers/dp_attention.py` modified +0/-10 (10 lines); hunks: def get_local_attention_dp_size():; symbols: get_local_attention_dp_size, get_local_attention_dp_rank, get_local_attention_dp_size, disable_dp_size
  - `python/sglang/bench_one_batch.py` modified +1/-0 (1 lines); hunks: def _maybe_prepare_dp_attn_batch(batch: ScheduleBatch, model_runner):; symbols: _maybe_prepare_dp_attn_batch
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/dp_attention.py`; keywords observed in patches: attention, moe, config, deepep, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/dp_attention.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6250 - Add pipeline parallelism for Qwen2 and Qwen3 Model

- Link: https://github.com/sgl-project/sglang/pull/6250
- Status/date: `merged`, created 2025-05-13, merged 2025-05-18; author `libratiger`.
- Diff scope read: `5` files, `+340/-73`; areas: model wrapper, MoE/router, tests/benchmarks; keywords: attention, config, processor, quant, cache, moe, expert, kv, test.
- Code diff details:
  - `python/sglang/srt/models/qwen2.py` modified +95/-26 (121 lines); hunks: # Adapted from llama2.py; from sglang.srt.layers.quantization.base_config import QuantizationConfig; symbols: Qwen2MLP, __init__, __init__, get_input_embedding
  - `python/sglang/srt/models/qwen2_moe.py` modified +89/-27 (116 lines); hunks: # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py; from sglang.srt.layers.quantization.base_config import QuantizationCon; symbols: Qwen2MoeMLP, __init__, __init__, forward
  - `python/sglang/srt/models/qwen3.py` modified +52/-10 (62 lines); hunks: # Adapted from qwen2.py; from sglang.srt.layers.quantization.base_config import QuantizationConfig; symbols: Qwen3Attention, __init__, __init__, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +49/-10 (59 lines); hunks: """Inference-only Qwen3MoE model compatible with HuggingFace weights."""; from transformers.configuration_utils import PretrainedConfig; symbols: Qwen3MoeSparseMoeBlock, __init__, __init__, forward
  - `test/srt/test_pp_single_node.py` modified +55/-0 (55 lines); hunks: """; def test_gsm8k(self):; symbols: test_gsm8k, TestQwenPPAccuracy, setUpClass, run_gsm8k_test
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3.py`; keywords observed in patches: attention, config, processor, quant, cache, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6533 - support eplb for qwen3

- Link: https://github.com/sgl-project/sglang/pull/6533
- Status/date: `merged`, created 2025-05-22, merged 2025-05-24; author `yizhang2077`.
- Diff scope read: `3` files, `+46/-25`; areas: model wrapper, MoE/router; keywords: expert, moe, router, topk, config, deepep, fp8, processor, quant, triton.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +39/-22 (61 lines); hunks: RowParallelLinear,; ParallelLMHead,; symbols: Qwen3MoeSparseMoeBlock, __init__, __init__, forward
  - `python/sglang/srt/layers/moe/topk.py` modified +4/-2 (6 lines); hunks: def fused_topk(; def fused_topk(; symbols: fused_topk, fused_topk, select_experts
  - `python/sglang/srt/managers/expert_distribution.py` modified +3/-1 (4 lines); hunks: def _convert_global_physical_count_to_logical_count(; symbols: _convert_global_physical_count_to_logical_count
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/managers/expert_distribution.py`; keywords observed in patches: expert, moe, router, topk, config, deepep. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/managers/expert_distribution.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6546 - added support for tied weights in qwen pipeline parallelism

- Link: https://github.com/sgl-project/sglang/pull/6546
- Status/date: `merged`, created 2025-05-23, merged 2025-05-25; author `FrankLeeeee`.
- Diff scope read: `4` files, `+134/-20`; areas: model wrapper, tests/benchmarks; keywords: config, processor, quant, test, vision, attention.
- Code diff details:
  - `test/srt/test_pp_single_node.py` modified +56/-0 (56 lines); hunks: def test_pp_consistency(self):; symbols: test_pp_consistency, TestQwenPPTieWeightsAccuracy, setUpClass, run_gsm8k_test
  - `python/sglang/srt/models/qwen3.py` modified +39/-10 (49 lines); hunks: from sglang.srt.layers.quantization.base_config import QuantizationConfig; def __init__(; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen2.py` modified +38/-9 (47 lines); hunks: def __init__(; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: __init__, load_weights
  - `.github/workflows/pr-test.yml` modified +1/-1 (2 lines); hunks: jobs:
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`; keywords observed in patches: config, processor, quant, test, vision, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6598 - qwen3moe support two batch overlap

- Link: https://github.com/sgl-project/sglang/pull/6598
- Status/date: `merged`, created 2025-05-25, merged 2025-05-26; author `yizhang2077`.
- Diff scope read: `5` files, `+351/-28`; areas: model wrapper, MoE/router, tests/benchmarks; keywords: moe, deepep, expert, attention, config, cuda, kv, mla, processor, router.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +200/-11 (211 lines); hunks: ParallelLMHead,; from sglang.srt.model_loader.weight_utils import default_weight_loader; symbols: __init__, forward_deepep, forward_deepep, op_gate
  - `python/sglang/srt/operations_strategy.py` modified +98/-7 (105 lines); hunks: def init_new_tbo(; def _compute_moe_deepseek_blog_decode(layer):; symbols: init_new_tbo, _assert_all_same, _compute_layer_operations_strategy_tbo, _compute_moe_deepseek_layer_operations_strategy_tbo
  - `test/srt/test_two_batch_overlap.py` modified +28/-0 (28 lines); hunks: from sglang.srt.utils import kill_process_tree; def test_compute_split_seq_index(self):; symbols: test_compute_split_seq_index, TestQwen3TwoBatchOverlap, setUpClass
  - `python/sglang/srt/models/qwen2_moe.py` modified +17/-6 (23 lines); hunks: from sglang.srt.managers.schedule_batch import global_server_args_dict; def forward(; symbols: forward
  - `python/sglang/srt/two_batch_overlap.py` modified +8/-4 (12 lines); hunks: def model_forward_maybe_tbo(; def _model_forward_tbo_split_inputs(; symbols: model_forward_maybe_tbo, _model_forward_tbo_split_inputs, _model_forward_tbo_split_inputs
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/operations_strategy.py`, `test/srt/test_two_batch_overlap.py`; keywords observed in patches: moe, deepep, expert, attention, config, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/operations_strategy.py`, `test/srt/test_two_batch_overlap.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6652 - Fix qwen3 tbo/dp-lm-head

- Link: https://github.com/sgl-project/sglang/pull/6652
- Status/date: `merged`, created 2025-05-27, merged 2025-05-27; author `yizhang2077`.
- Diff scope read: `3` files, `+3/-1`; areas: model wrapper, MoE/router; keywords: config, moe, processor, quant.
- Code diff details:
  - `python/sglang/srt/two_batch_overlap.py` modified +1/-1 (2 lines); hunks: def model_forward_maybe_tbo(; symbols: model_forward_maybe_tbo
  - `python/sglang/srt/models/qwen2_moe.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/two_batch_overlap.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: config, moe, processor, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/two_batch_overlap.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6709 - Fix PP for Qwen3 MoE

- Link: https://github.com/sgl-project/sglang/pull/6709
- Status/date: `merged`, created 2025-05-28, merged 2025-05-29; author `jinyouzhi`.
- Diff scope read: `2` files, `+60/-4`; areas: model wrapper, MoE/router, tests/benchmarks; keywords: moe, expert, test.
- Code diff details:
  - `test/srt/test_pp_single_node.py` modified +57/-1 (58 lines); hunks: def test_pp_consistency(self):; def test_pp_consistency(self):; symbols: test_pp_consistency, TestQwenPPTieWeightsAccuracy, setUpClass, test_pp_consistency
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: moe, expert, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6818 - Fix wrong weight reference in dynamic EPLB

- Link: https://github.com/sgl-project/sglang/pull/6818
- Status/date: `merged`, created 2025-06-02, merged 2025-06-03; author `fzyzcjy`.
- Diff scope read: `3` files, `+27/-13`; areas: model wrapper, MoE/router; keywords: config, expert, moe, attention, deepep, kv, processor, triton.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-8 (21 lines); hunks: from sglang.srt.utils import (; def __init__(; symbols: __init__, routed_experts_weights_of_layer, determine_n_share_experts_fusion, post_load_weights
  - `python/sglang/srt/utils.py` modified +13/-0 (13 lines); hunks: def support_triton(backend: str) -> bool:; symbols: support_triton, cpu_has_amx_support, LazyValue:, __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-5 (6 lines); hunks: """Inference-only Qwen3MoE model compatible with HuggingFace weights."""; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: config, expert, moe, attention, deepep, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6820 - Fix Qwen3MoE missing token padding optimization

- Link: https://github.com/sgl-project/sglang/pull/6820
- Status/date: `merged`, created 2025-06-03, merged 2025-06-05; author `fzyzcjy`.
- Diff scope read: `2` files, `+5/-3`; areas: model wrapper, MoE/router; keywords: expert, moe, topk, deepep, router.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +3/-3 (6 lines); hunks: def fused_topk(; def fused_topk(; symbols: fused_topk, fused_topk, select_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: def forward_deepep(; def op_select_experts(self, state):; symbols: forward_deepep, op_select_experts
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: expert, moe, topk, deepep, router. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6964 - Support both approximate and exact expert distribution collection

- Link: https://github.com/sgl-project/sglang/pull/6964
- Status/date: `merged`, created 2025-06-08, merged 2025-06-10; author `fzyzcjy`.
- Diff scope read: `4` files, `+101/-71`; areas: model wrapper, MoE/router; keywords: expert, topk, moe, router, cuda, deepep.
- Code diff details:
  - `python/sglang/srt/managers/expert_distribution.py` modified +67/-43 (110 lines); hunks: def init_new(; def on_forward_pass_start(self, forward_batch: ForwardBatch):; symbols: init_new, __init__, on_forward_pass_start, on_select_experts
  - `python/sglang/srt/models/deepseek_v2.py` modified +19/-16 (35 lines); hunks: def op_select_experts(self, state):; symbols: op_select_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +14/-11 (25 lines); hunks: def op_select_experts(self, state):; symbols: op_select_experts
  - `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: class ServerArgs:; symbols: ServerArgs:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/managers/expert_distribution.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: expert, topk, moe, router, cuda, deepep. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/managers/expert_distribution.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6990 - support qwen3 emebedding

- Link: https://github.com/sgl-project/sglang/pull/6990
- Status/date: `merged`, created 2025-06-09, merged 2025-06-09; author `Titan-p`.
- Diff scope read: `2` files, `+3/-0`; areas: model wrapper, tests/benchmarks; keywords: config, test.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +2/-0 (2 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
  - `test/srt/models/test_embedding_models.py` modified +1/-0 (1 lines); hunks: ("Alibaba-NLP/gte-Qwen2-1.5B-instruct", 1, 1e-5),
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`, `test/srt/models/test_embedding_models.py`; keywords observed in patches: config, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`, `test/srt/models/test_embedding_models.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7222 - DP Attention with Auto DeepEP Dispatch

- Link: https://github.com/sgl-project/sglang/pull/7222
- Status/date: `merged`, created 2025-06-16, merged 2025-07-05; author `ch-wan`.
- Diff scope read: `13` files, `+136/-90`; areas: model wrapper, MoE/router, scheduler/runtime, tests/benchmarks; keywords: deepep, moe, topk, expert, attention, cuda, quant, spec, fp8, scheduler.
- Code diff details:
  - `test/srt/test_hybrid_dp_ep_tp_mtp.py` modified +80/-40 (120 lines); hunks: def setUpClass(cls):; def setUpClass(cls):; symbols: setUpClass, setUpClass, setUpClass, setUpClass
  - `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +15/-13 (28 lines); hunks: deepep_post_reorder_triton_kernel,; def dispatch_a(; symbols: dispatch_a, dispatch_b, combine, combine_a
  - `python/sglang/srt/models/qwen3_moe.py` modified +7/-9 (16 lines); hunks: def forward_deepep(; def forward_deepep(; symbols: forward_deepep, forward_deepep, op_dispatch_a, op_experts
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-7 (14 lines); hunks: def forward_deepep(; def forward_deepep(; symbols: forward_deepep, forward_deepep, op_dispatch_a, op_experts
  - `python/sglang/srt/two_batch_overlap.py` modified +7/-3 (10 lines); hunks: ); def replay_prepare(; symbols: replay_prepare, TboDPAttentionPreparer:, prepare_all_gather, prepare_all_gather
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: deepep, moe, topk, expert, attention, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7312 - Add get_hidden_dim to qwen3.py for correct lora

- Link: https://github.com/sgl-project/sglang/pull/7312
- Status/date: `merged`, created 2025-06-18, merged 2025-07-20; author `logachevpa`.
- Diff scope read: `5` files, `+240/-2`; areas: model wrapper, scheduler/runtime, tests/benchmarks; keywords: lora, test, config, attention, cache, cuda, kv, spec, triton.
- Code diff details:
  - `test/srt/models/lora/test_lora_qwen3.py` added +209/-0 (209 lines); hunks: +# Copyright 2023-2025 SGLang Team; symbols: TestLoRA, _run_lora_multiple_batch_on_model_cases, test_ci_lora_models, test_all_lora_models
  - `python/sglang/srt/models/qwen3.py` modified +24/-0 (24 lines); hunks: def __init__(; symbols: __init__, get_input_embeddings, get_hidden_dim, forward
  - `python/sglang/test/runners.py` modified +6/-1 (7 lines); hunks: def __init__(; def start_model_process(self, in_queue, out_queue, model_path, torch_dtype):; symbols: __init__, start_model_process, forward_generation_raw, forward_generation_raw
  - `test/srt/models/lora/test_lora.py` modified +0/-1 (1 lines); hunks: def ensure_reproducibility(self):; symbols: ensure_reproducibility, _run_lora_multiple_batch_on_model_cases
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: class TestFile:; symbols: TestFile:
- Optimization/support interpretation: The concrete diff surface is `test/srt/models/lora/test_lora_qwen3.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/test/runners.py`; keywords observed in patches: lora, test, config, attention, cache, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/models/lora/test_lora_qwen3.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/test/runners.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7580 - Move files related to EPLB

- Link: https://github.com/sgl-project/sglang/pull/7580
- Status/date: `merged`, created 2025-06-27, merged 2025-06-29; author `fzyzcjy`.
- Diff scope read: `22` files, `+42/-54`; areas: model wrapper, MoE/router, scheduler/runtime, tests/benchmarks; keywords: expert, moe, config, attention, cuda, quant, deepep, fp8, kv, lora.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +13/-13 (26 lines); hunks: set_mscclpp_all_reduce,; from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
  - `python/sglang/srt/models/qwen2_moe.py` modified +5/-5 (10 lines); hunks: get_tensor_model_parallel_world_size,; ParallelLMHead,
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-5 (8 lines); hunks: parallel_state,; ParallelLMHead,
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-5 (8 lines); hunks: tensor_model_parallel_all_gather,; ParallelLMHead,
  - `python/sglang/srt/eplb/eplb_manager.py` renamed +2/-4 (6 lines); hunks: import torch.cuda
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: expert, moe, config, attention, cuda, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7634 - [Feature] Layer-wise Prefill

- Link: https://github.com/sgl-project/sglang/pull/7634
- Status/date: `merged`, created 2025-06-29, merged 2025-07-16; author `jason-fxz`.
- Diff scope read: `13` files, `+464/-2`; areas: model wrapper, MoE/router, scheduler/runtime; keywords: config, processor, expert, moe, cuda, kv, spec.
- Code diff details:
  - `python/sglang/srt/models/gemma3_causal.py` modified +63/-0 (63 lines); hunks: def forward(; symbols: forward, forward_split_prefill, load_weights
  - `python/sglang/srt/models/gemma2.py` modified +51/-0 (51 lines); hunks: def forward(; symbols: forward, forward_split_prefill, get_hidden_dim
  - `python/sglang/srt/models/gemma.py` modified +48/-0 (48 lines); hunks: def forward(; symbols: forward, forward_split_prefill, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +44/-0 (44 lines); hunks: def __init__(; def forward(; symbols: __init__, forward, forward_split_prefill, start_layer
  - `python/sglang/srt/models/qwen3_moe.py` modified +43/-0 (43 lines); hunks: def forward(; symbols: forward, forward_split_prefill, start_layer
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma.py`; keywords observed in patches: config, processor, expert, moe, cuda, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7681 - support qwen3 dense model dp attention

- Link: https://github.com/sgl-project/sglang/pull/7681
- Status/date: `merged`, created 2025-07-01, merged 2025-07-03; author `yizhang2077`.
- Diff scope read: `2` files, `+49/-17`; areas: model wrapper; keywords: attention, config, kv, quant, cache, processor.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +42/-16 (58 lines); hunks: split_tensor_along_last_dim,; def __init__(; symbols: __init__, __init__, __init__, forward
  - `python/sglang/srt/models/qwen2.py` modified +7/-1 (8 lines); hunks: ParallelLMHead,; def __init__(; symbols: __init__, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`; keywords observed in patches: attention, config, kv, quant, cache, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7723 - [Bug] add flashinfer bool check for fusedmoe in Qwen moe models

- Link: https://github.com/sgl-project/sglang/pull/7723
- Status/date: `merged`, created 2025-07-02, merged 2025-07-03; author `yilian49`.
- Diff scope read: `2` files, `+18/-0`; areas: model wrapper, MoE/router; keywords: flash, moe, config, deepep, expert, quant, topk.
- Code diff details:
  - `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-0 (9 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: flash, moe, config, deepep, expert, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7740 - [optimize] add two stream norm for qwen3

- Link: https://github.com/sgl-project/sglang/pull/7740
- Status/date: `merged`, created 2025-07-03, merged 2025-07-03; author `yizhang2077`.
- Diff scope read: `4` files, `+54/-10`; areas: model wrapper, MoE/router; keywords: config, cuda, quant, attention, moe, deepep.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +24/-5 (29 lines); hunks: from sglang.srt.layers.rotary_embedding import get_rope; def __init__(; symbols: Qwen3Attention, __init__, __init__, _apply_qk_norm
  - `python/sglang/srt/models/qwen3_moe.py` modified +24/-5 (29 lines); hunks: VocabParallelEmbedding,; from sglang.srt.models.qwen2_moe import Qwen2MoeMLP as Qwen3MoeMLP; symbols: Qwen3MoeSparseMoeBlock, __init__, __init__, _apply_qk_norm
  - `python/sglang/srt/models/qwen2.py` modified +3/-0 (3 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__
  - `python/sglang/srt/models/qwen2_moe.py` modified +3/-0 (3 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2.py`; keywords observed in patches: config, cuda, quant, attention, moe, deepep. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7745 - [feat] Support EAGLE3 for Qwen

- Link: https://github.com/sgl-project/sglang/pull/7745
- Status/date: `merged`, created 2025-07-03, merged 2025-07-05; author `Ximingwang-09`.
- Diff scope read: `4` files, `+81/-6`; areas: model wrapper, MoE/router; keywords: eagle, config, cache, kv, moe, processor, spec, expert, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +28/-2 (30 lines); hunks: import logging; def __init__(; symbols: __init__, get_input_embeddings, forward, set_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +25/-2 (27 lines); hunks: """Inference-only Qwen3MoE model compatible with HuggingFace weights."""; def __init__(; symbols: __init__, forward, forward, start_layer
  - `python/sglang/srt/models/qwen2_moe.py` modified +15/-1 (16 lines); hunks: def __init__(; def forward(; symbols: __init__, forward, forward, forward
  - `python/sglang/srt/models/qwen2.py` modified +13/-1 (14 lines); hunks: def __init__(; def forward(; symbols: __init__, get_input_embedding, forward, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`; keywords observed in patches: eagle, config, cache, kv, moe, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7912 - Qwen FP8/NVFP4 ModelOPT Quantization support

- Link: https://github.com/sgl-project/sglang/pull/7912
- Status/date: `merged`, created 2025-07-09, merged 2025-09-03; author `jingyu-ml`.
- Diff scope read: `2` files, `+43/-4`; areas: model wrapper, quantization; keywords: kv, cache, config, cuda, fp4, quant, vision.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +35/-2 (37 lines); hunks: def get_min_capability(cls) -> int:; def from_config(cls, config: Dict[str, Any]) -> ModelOptFp4Config:; symbols: get_min_capability, get_config_filenames, common_group_size, from_config
  - `python/sglang/srt/models/qwen3.py` modified +8/-2 (10 lines); hunks: from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py`; keywords observed in patches: kv, cache, config, cuda, fp4, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7966 - [1/N] MoE Refactor: refactor `select_experts`

- Link: https://github.com/sgl-project/sglang/pull/7966
- Status/date: `merged`, created 2025-07-11, merged 2025-07-19; author `ch-wan`.
- Diff scope read: `39` files, `+557/-872`; areas: model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: expert, moe, router, topk, triton, quant, cuda, fp8, config, attention.
- Code diff details:
  - `python/sglang/srt/layers/quantization/unquant.py` modified +55/-152 (207 lines); hunks: +from __future__ import annotations; use_intel_amx_backend,; symbols: __init__, create_weights, apply, forward_cuda
  - `python/sglang/srt/layers/moe/topk.py` modified +171/-5 (176 lines); hunks: # limitations under the License.; except ImportError:; symbols: TopKOutput, TopK, __init__, forward_native
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +21/-71 (92 lines); hunks: # Adapted from https://github.com/vllm-project/vllm/tree/v0.8.2/vllm/model_executor/layers/quantization/compressed_tensors; ); symbols: GPTQMarlinState, CompressedTensorsMoEMethod:, CompressedTensorsMoEMethod, __new__
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +14/-75 (89 lines); hunks: import importlib; use_intel_amx_backend,; symbols: get_quant_method, apply, apply, create_weights
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +13/-74 (87 lines); hunks: import logging; tma_align_input_scale,; symbols: __init__, __init__, determine_expert_map, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; keywords observed in patches: expert, moe, router, topk, triton, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8036 - [NVIDIA] Add Flashinfer MoE blockscale fp8 backend

- Link: https://github.com/sgl-project/sglang/pull/8036
- Status/date: `merged`, created 2025-07-15, merged 2025-07-27; author `kaixih`.
- Diff scope read: `8` files, `+179/-47`; areas: model wrapper, MoE/router, quantization, kernel; keywords: flash, moe, quant, deepep, expert, config, fp4, router, fp8, topk.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +102/-7 (109 lines); hunks: get_bool_env_var,; from aiter.fused_moe import fused_moe; symbols: forward, _get_tile_tokens_dim, EPMoE, _weight_loader_physical
  - `python/sglang/srt/models/deepseek_v2.py` modified +44/-20 (64 lines); hunks: RowParallelLinear,; def __init__(; symbols: __init__, __init__, forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-7 (16 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__
  - `python/sglang/srt/server_args.py` modified +13/-3 (16 lines); hunks: class ServerArgs:; def __post_init__(self):; symbols: ServerArgs:, __post_init__, add_cli_args
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +5/-5 (10 lines); hunks: def __init__(self, quant_config: ModelOptFp4Config):; def process_weights_after_loading(self, layer: torch.nn.Module) -> None:; symbols: __init__, create_weights, process_weights_after_loading, process_weights_after_loading
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; keywords observed in patches: flash, moe, quant, deepep, expert, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8280 - DP Enhancement

- Link: https://github.com/sgl-project/sglang/pull/8280
- Status/date: `merged`, created 2025-07-23, merged 2025-07-25; author `ch-wan`.
- Diff scope read: `20` files, `+665/-1116`; areas: model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks; keywords: spec, attention, config, cuda, eagle, cache, processor, topk, kv, triton.
- Code diff details:
  - `test/srt/test_hybrid_dp_ep_tp_mtp.py` modified +70/-850 (920 lines); hunks: ); def test_mmlu(self):; symbols: Test0, Test00, setUpClass, test_mmlu
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +193/-22 (215 lines); hunks: import triton; if TYPE_CHECKING:; symbols: ForwardBatch:, ForwardBatch:, ForwardBatch:, init_new
  - `python/sglang/srt/speculative/eagle_worker.py` modified +59/-44 (103 lines); hunks: def draft_model_runner(self):; def forward_batch_speculative_generation(; symbols: draft_model_runner, forward_batch_speculative_generation, forward_batch_speculative_generation, forward_batch_speculative_generation
  - `python/sglang/srt/layers/dp_attention.py` modified +72/-24 (96 lines); hunks: import functools; _LOCAL_ATTN_DP_RANK = None; symbols: DPPaddingMode, is_max_len, is_sum_len, get_dp_padding_mode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +61/-25 (86 lines); hunks: from sglang.srt.custom_op import CustomOp; def get_batch_sizes_to_capture(model_runner: ModelRunner):; symbols: get_batch_sizes_to_capture, __init__, __init__, can_run
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_worker.py`; keywords observed in patches: spec, attention, config, cuda, eagle, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_worker.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8421 - [3/N] MoE Refactor: Simplify DeepEP Output

- Link: https://github.com/sgl-project/sglang/pull/8421
- Status/date: `merged`, created 2025-07-27, merged 2025-07-28; author `ch-wan`.
- Diff scope read: `8` files, `+319/-276`; areas: model wrapper, MoE/router; keywords: moe, deepep, expert, topk, config, router, fp8, quant, triton, eagle.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +69/-118 (187 lines); hunks: +# NOTE(ch-wan): this file will be moved to sglang/srt/layers/moe/token_dispatcher/deepep.py; use_deepep = False; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, format
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +150/-30 (180 lines); hunks: +from __future__ import annotations; next_power_of_2,; symbols: __init__, forward, dispatch, moe_impl
  - `python/sglang/srt/models/qwen3_moe.py` modified +12/-69 (81 lines); hunks: def __init__(; def forward_deepep(; symbols: __init__, forward, forward_deepep, op_gate
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-56 (69 lines); hunks: def forward_deepep(; def op_select_experts(self, state):; symbols: forward_deepep, op_select_experts, op_dispatch_a, op_dispatch_b
  - `python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py` added +48/-0 (48 lines); hunks: +from __future__ import annotations; symbols: DispatchOutputFormat, is_standard, is_deepep_normal, is_deepep_ll
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: moe, deepep, expert, topk, config, router. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8448 - Support EPLB in FusedMoE

- Link: https://github.com/sgl-project/sglang/pull/8448
- Status/date: `merged`, created 2025-07-28, merged 2025-07-29; author `ch-wan`.
- Diff scope read: `15` files, `+107/-11`; areas: model wrapper, MoE/router, kernel; keywords: config, expert, moe, quant, deepep, flash, router, topk, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-1 (45 lines); hunks: get_tensor_model_parallel_world_size,; def __init__(; symbols: __init__, __init__, weight_loader, _weight_loader_physical
  - `python/sglang/srt/eplb/expert_location.py` modified +17/-6 (23 lines); hunks: def __post_init__(self):; def init_by_mapping(; symbols: __post_init__, init_trivial, init_by_mapping, init_by_eplb
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +16/-3 (19 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, weight_loader, __init__
  - `python/sglang/srt/eplb/expert_distribution.py` modified +5/-0 (5 lines); hunks: def init_new(; symbols: init_new
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: def __init__(; def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/eplb/expert_location.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; keywords observed in patches: config, expert, moe, quant, deepep, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/eplb/expert_location.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8450 - [NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE

- Link: https://github.com/sgl-project/sglang/pull/8450
- Status/date: `merged`, created 2025-07-28, merged 2025-08-01; author `kaixih`.
- Diff scope read: `6` files, `+131/-46`; areas: model wrapper, MoE/router, quantization, kernel; keywords: flash, moe, expert, topk, config, quant, deepep, fp8, router, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +54/-1 (55 lines); hunks: # Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/layer.py; logger = logging; symbols: should_use_flashinfer_trtllm_moe, FusedMoeWeightScaleSupported, _weight_loader_impl, make_expert_input_scale_params_mapping
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +19/-34 (53 lines); hunks: silu_and_mul_triton_kernel,; get_bool_env_var,; symbols: _get_tile_tokens_dim, EPMoE, __init__, forward
  - `python/sglang/srt/layers/quantization/fp8.py` modified +52/-0 (52 lines); hunks: def dummy_func(*args, **kwargs):; def apply(; symbols: dummy_func, apply, get_tile_tokens_dim, Fp8MoEMethod
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-4 (7 lines); hunks: from sglang.srt.layers.moe.ep_moe.layer import (; def __init__(; symbols: __init__, __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-3 (6 lines); hunks: from sglang.srt.layers.moe.ep_moe.layer import (; def __init__(; symbols: __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`; keywords observed in patches: flash, moe, expert, topk, config, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8658 - [5/N] MoE Refactor: Update MoE parallelism arguments

- Link: https://github.com/sgl-project/sglang/pull/8658
- Status/date: `merged`, created 2025-08-01, merged 2025-08-01; author `ch-wan`.
- Diff scope read: `38` files, `+342/-299`; areas: model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: deepep, moe, config, expert, flash, quant, attention, cuda, fp8, topk.
- Code diff details:
  - `test/srt/test_hybrid_dp_ep_tp_mtp.py` modified +80/-80 (160 lines); hunks: def setUpClass(cls):; def setUpClass(cls):; symbols: setUpClass, setUpClass, setUpClass, setUpClass
  - `python/sglang/srt/server_args.py` modified +47/-20 (67 lines); hunks: class ServerArgs:; class ServerArgs:; symbols: ServerArgs:, ServerArgs:, __post_init__, print_deprecated_warning
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +9/-35 (44 lines); hunks: from __future__ import annotations; should_use_flashinfer_trtllm_moe,; symbols: __init__, __init__, __init__, forward
  - `python/sglang/srt/layers/moe/utils.py` added +43/-0 (43 lines); hunks: +from enum import Enum; symbols: MoeA2ABackend, _missing_, is_deepep, is_standard
  - `python/sglang/srt/models/deepseek_v2.py` modified +10/-15 (25 lines); hunks: from transformers import PretrainedConfig; get_moe_impl_class,; symbols: __init__, __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; keywords observed in patches: deepep, moe, config, expert, flash, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8751 - [1/3] Optimize Slime Update Weights: Remove QWen3MOE Load Weight Overhead

- Link: https://github.com/sgl-project/sglang/pull/8751
- Status/date: `merged`, created 2025-08-04, merged 2025-08-06; author `hebiao064`.
- Diff scope read: `1` files, `+26/-6`; areas: model wrapper, MoE/router; keywords: cache, config, expert, moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +26/-6 (32 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights, load_weights, load_weights, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: cache, config, expert, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8987 - Fix incorrect default get_hidden_dim logic

- Link: https://github.com/sgl-project/sglang/pull/8987
- Status/date: `merged`, created 2025-08-08, merged 2025-08-09; author `lifuhuang`.
- Diff scope read: `7` files, `+36/-143`; areas: model wrapper; keywords: attention, config, kv, lora, spec.
- Code diff details:
  - `python/sglang/srt/models/gemma2.py` modified +0/-34 (34 lines); hunks: def forward_split_prefill(; symbols: forward_split_prefill, get_hidden_dim, get_module_name, get_attention_sliding_window_size
  - `python/sglang/srt/lora/utils.py` modified +24/-5 (29 lines); hunks: def get_hidden_dim(; symbols: get_hidden_dim, if
  - `python/sglang/srt/models/granite.py` modified +0/-25 (25 lines); hunks: def forward(; symbols: forward, get_hidden_dim, get_module_name, get_module_name_from_weight_name
  - `python/sglang/srt/models/llama.py` modified +0/-25 (25 lines); hunks: def end_layer(self):; symbols: end_layer, get_input_embeddings, get_hidden_dim, get_module_name
  - `python/sglang/srt/models/qwen3.py` modified +0/-24 (24 lines); hunks: def __init__(; symbols: __init__, get_input_embeddings, get_hidden_dim, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/lora/utils.py`, `python/sglang/srt/models/granite.py`; keywords observed in patches: attention, config, kv, lora, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/lora/utils.py`, `python/sglang/srt/models/granite.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9101 - Feature: support qwen and llama4 reducescatter for dp attention padding

- Link: https://github.com/sgl-project/sglang/pull/9101
- Status/date: `merged`, created 2025-08-12, merged 2025-08-14; author `Misaka9468`.
- Diff scope read: `5` files, `+68/-16`; areas: model wrapper, MoE/router; keywords: attention, moe, config, expert, topk, deepep, lora, router.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +18/-5 (23 lines); hunks: def __init__(; def get_moe_weights(self):; symbols: __init__, forward, get_moe_weights, forward_normal
  - `python/sglang/srt/models/qwen2_moe.py` modified +18/-4 (22 lines); hunks: def __init__(; def __init__(; symbols: __init__, forward, forward, __init__
  - `python/sglang/srt/models/llama4.py` modified +16/-3 (19 lines); hunks: def __init__(; def __init__(; symbols: __init__, forward, forward, __init__
  - `python/sglang/srt/models/llama.py` modified +10/-2 (12 lines); hunks: def __init__(; symbols: __init__, forward, forward
  - `python/sglang/srt/lora/layers.py` modified +6/-2 (8 lines); hunks: def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor; def forward(self, input_: torch.Tensor):; symbols: apply_lora, forward, forward, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/llama4.py`; keywords observed in patches: attention, moe, config, expert, topk, deepep. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/llama4.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9147 - support Qwen3-MoE-w4afp8

- Link: https://github.com/sgl-project/sglang/pull/9147
- Status/date: `open`, created 2025-08-13; author `zhilingjiang`.
- Diff scope read: `636` files, `+16172/-71705`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, cache, test, router, spec, attention, doc, flash, kv, cuda.
- Code diff details:
  - `sgl-router/src/routers/pd_router.rs` removed +0/-2180 (2180 lines); hunks: -// PD (Prefill-Decode) Router Implementation; symbols: PDRouter
  - `python/sglang/srt/models/phi4mm_utils.py` removed +0/-1917 (1917 lines); hunks: -# Copyright 2024 SGLang Team; symbols: BlockBase, __init__, get_activation, adaptive_enc_mask
  - `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` removed +0/-1700 (1700 lines); hunks: -# SPDX-License-Identifier: Apache-2.0; symbols: DualChunkFlashAttentionMetadata:, DualChunkFlashAttentionBackend, __init__, get_sparse_attention_config
  - `sgl-router/tests/api_endpoints_test.rs` removed +0/-1644 (1644 lines); hunks: -mod common;; symbols: TestContext
  - `sgl-router/src/core/worker.rs` modified +16/-1387 (1403 lines); hunks: -use super::{CircuitBreaker, CircuitBreakerConfig, WorkerError, WorkerResult};; pub trait Worker: Send + Sync + fmt::Debug {; symbols: BasicWorker, DPAwareWorker
- Optimization/support interpretation: The concrete diff surface is `sgl-router/src/routers/pd_router.rs`, `python/sglang/srt/models/phi4mm_utils.py`, `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`; keywords observed in patches: config, cache, test, router, spec, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `sgl-router/src/routers/pd_router.rs`, `python/sglang/srt/models/phi4mm_utils.py`, `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9338 - Refactor TopK to ensure readability and extensibility

- Link: https://github.com/sgl-project/sglang/pull/9338
- Status/date: `merged`, created 2025-08-19, merged 2025-09-15; author `ch-wan`.
- Diff scope read: `14` files, `+52/-47`; areas: model wrapper, MoE/router, kernel; keywords: moe, quant, config, expert, triton, flash, fp4, fp8, topk, deepep.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +30/-9 (39 lines); hunks: from dataclasses import dataclass; is_npu,; symbols: TopKConfig:, __init__, __init__, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-12 (19 lines); hunks: get_deepep_mode,; def __init__(; symbols: __init__
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10 (10 lines); hunks: logger = logging.getLogger(__name__); symbols: _is_fp4_quantization_enabled, selection, _get_tile_tokens_dim
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4 (8 lines); hunks: def _forward_ll(dispatch_output: DeepEPLLOutput):; def get_moe_impl_class(quant_config: Optional[QuantizationConfig] = None):; symbols: _forward_ll, get_moe_impl_class, get_moe_impl_class, get_moe_impl_class
  - `python/sglang/srt/models/longcat_flash.py` modified +2/-2 (4 lines); hunks: def __init__(; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: __init__, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; keywords observed in patches: moe, quant, config, expert, triton, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9973 - Optimize Qwen3-moe model by using flashinfer fused allreduce

- Link: https://github.com/sgl-project/sglang/pull/9973
- Status/date: `merged`, created 2025-09-03, merged 2025-09-04; author `yuan-luo`.
- Diff scope read: `3` files, `+52/-12`; areas: model wrapper, MoE/router; keywords: cuda, flash, moe, attention, config, deepep, expert, fp4, processor, router.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +39/-8 (47 lines); hunks: RowParallelLinear,; from sglang.srt.model_loader.weight_utils import default_weight_loader; symbols: forward, get_moe_weights, forward_normal, forward_normal
  - `python/sglang/srt/layers/communicator.py` modified +9/-3 (12 lines); hunks: ); def _gather_hidden_states_and_residual(; symbols: _gather_hidden_states_and_residual
  - `python/sglang/srt/models/qwen2_moe.py` modified +4/-1 (5 lines); hunks: def __init__(; symbols: __init__, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2_moe.py`; keywords observed in patches: cuda, flash, moe, attention, config, deepep. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10574 - [Ascend]optimize Qwen3 on Ascend

- Link: https://github.com/sgl-project/sglang/pull/10574
- Status/date: `merged`, created 2025-09-17, merged 2025-09-23; author `ping1jing2`.
- Diff scope read: `6` files, `+81/-2`; areas: model wrapper, quantization, scheduler/runtime; keywords: attention, cache, config, cuda, flash, mla, quant.
- Code diff details:
  - `python/sglang/srt/utils.py` modified +44/-0 (44 lines); hunks: def make_layers(; symbols: make_layers, get_cmo_stream, prepare_weight_cache, wait_cmo_stream
  - `python/sglang/srt/models/qwen3.py` modified +18/-2 (20 lines); hunks: ); def forward(; symbols: Qwen3Attention, forward
  - `python/sglang/srt/layers/communicator.py` modified +8/-0 (8 lines); hunks: is_hip,; def prepare_mlp(; symbols: prepare_mlp, CommunicateContext:, is_same_group_size, _gather_hidden_states_and_residual
  - `python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunks: def add_mla_attention_backend(backend_name):; symbols: add_mla_attention_backend, RankZeroFilter
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +2/-0 (2 lines); hunks: def process_weights_after_loading(self, layer):; def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading, NPU_W8A8LinearMethodMTImpl:, process_weights_after_loading, NPU_W8A8DynamicLinearMethod
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/layers/communicator.py`; keywords observed in patches: attention, cache, config, cuda, flash, mla. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/layers/communicator.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10749 - Fuse write kv buffer into rope for qwen3 moe & bailing moe

- Link: https://github.com/sgl-project/sglang/pull/10749
- Status/date: `merged`, created 2025-09-22, merged 2025-09-26; author `yuan-luo`.
- Diff scope read: `4` files, `+105/-34`; areas: model wrapper, MoE/router; keywords: cache, cuda, kv, attention, moe, config, lora, spec.
- Code diff details:
  - `python/sglang/srt/models/utils.py` added +51/-0 (51 lines); hunks: +# Copyright 2023-2025 SGLang Team; symbols: enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg
  - `python/sglang/srt/models/gpt_oss.py` modified +7/-30 (37 lines); hunks: from sglang.srt.managers.schedule_batch import global_server_args_dict; def forward_normal(; symbols: forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, GptOssAttention
  - `python/sglang/srt/models/bailing_moe.py` modified +25/-2 (27 lines); hunks: from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode; def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: from sglang.srt.model_loader.weight_utils import default_weight_loader; def forward_prepare(; symbols: forward_prepare, forward_core
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py`; keywords observed in patches: cache, cuda, kv, attention, moe, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10911 - model: qwen3-omni (thinker-only)

- Link: https://github.com/sgl-project/sglang/pull/10911
- Status/date: `merged`, created 2025-09-25, merged 2025-10-16; author `mickqian`.
- Diff scope read: `16` files, `+1947/-328`; areas: model wrapper, MoE/router, multimodal/processor, tests/benchmarks, docs/config; keywords: vision, attention, moe, config, cache, quant, expert, processor, spec, test.
- Code diff details:
  - `python/sglang/srt/models/qwen3_omni_moe.py` added +661/-0 (661 lines); hunks: +# Copyright 2025 Qwen Team; symbols: Qwen3OmniMoeAudioEncoderLayer, __init__, forward, SinusoidsPositionEmbedding
  - `python/sglang/srt/configs/qwen3_omni.py` added +613/-0 (613 lines); hunks: +from transformers import PretrainedConfig; symbols: Qwen3OmniMoeAudioEncoderConfig, __init__, Qwen3OmniMoeVisionEncoderConfig, __init__
  - `python/sglang/srt/layers/rotary_embedding.py` modified +357/-2 (359 lines); hunks: def get_rope_index(; def get_rope_index(; symbols: get_rope_index, get_rope_index, get_rope_index, get_rope_index_qwen3_omni
  - `test/srt/test_vision_openai_server_common.py` modified +132/-96 (228 lines); hunks: import base64; AUDIO_BIRD_SONG_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/bird_song.mp3"; symbols: TestOpenAIOmniServerBase, TestOpenAIMLLMServerBase, setUpClass, get_or_download_file
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +53/-168 (221 lines); hunks: # ==============================================================================; class Qwen3MoeLLMModel(Qwen3MoeModel):; symbols: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py`; keywords observed in patches: vision, attention, moe, config, cache, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10975 - Use more general heuristics to set the default value of --mem-fraction-static

- Link: https://github.com/sgl-project/sglang/pull/10975
- Status/date: `merged`, created 2025-09-27, merged 2025-09-29; author `merrymercy`.
- Diff scope read: `9` files, `+157/-141`; areas: model wrapper, attention/backend, tests/benchmarks; keywords: cache, test, cuda, attention, kv, lora, mla, config, eagle, moe.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +116/-82 (198 lines); hunks: def _handle_missing_default_values(self):; symbols: _handle_missing_default_values, _handle_gpu_memory_settings, _generate_cuda_graph_batch_sizes
  - `python/sglang/srt/managers/io_struct.py` modified +22/-13 (35 lines); hunks: Image = Any; class GenerateReqInput:; symbols: SessionParams:, GenerateReqInput:, GenerateReqInput:, contains_mm_input
  - `.github/workflows/pr-test.yml` modified +0/-26 (26 lines); hunks: jobs:; jobs:
  - `python/sglang/srt/model_loader/weight_utils.py` modified +10/-10 (20 lines); hunks: def find_local_hf_snapshot_dir(; def download_weights_from_hf(; symbols: find_local_hf_snapshot_dir, download_weights_from_hf
  - `test/srt/test_multi_instance_release_memory_occupation.py` modified +5/-2 (7 lines); hunks: import multiprocessing; TEST_SUITE = dict(; symbols: _run_sglang_subprocess
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `python/sglang/srt/managers/io_struct.py`, `.github/workflows/pr-test.yml`; keywords observed in patches: cache, test, cuda, attention, kv, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `python/sglang/srt/managers/io_struct.py`, `.github/workflows/pr-test.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12002 - Eagle3 DP attention for Qwen3 MoE

- Link: https://github.com/sgl-project/sglang/pull/12002
- Status/date: `merged`, created 2025-10-23, merged 2025-10-29; author `qhsc`.
- Diff scope read: `9` files, `+219/-27`; areas: model wrapper, attention/backend, MoE/router, scheduler/runtime, tests/benchmarks; keywords: eagle, spec, moe, attention, config, cuda, processor, test, cache, topk.
- Code diff details:
  - `test/srt/test_eagle_dp_attention.py` added +129/-0 (129 lines); hunks: +import unittest; symbols: TestEAGLE3EngineDPAttention, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/srt/models/qwen2_moe.py` modified +30/-15 (45 lines); hunks: def forward(; def __init__(; symbols: forward, __init__, set_eagle3_layers_to_capture, forward
  - `python/sglang/srt/layers/communicator.py` modified +23/-1 (24 lines); hunks: from dataclasses import dataclass; def __init__(; symbols: __init__, prepare_attn_and_capture_last_layer_outputs, prepare_attn
  - `python/sglang/srt/models/qwen3_moe.py` modified +16/-8 (24 lines); hunks: def forward(; def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):; symbols: forward, set_eagle3_layers_to_capture, load_weights
  - `python/sglang/srt/models/llama_eagle3.py` modified +11/-1 (12 lines); hunks: # https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py; def forward(; symbols: forward, __init__
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_eagle_dp_attention.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/communicator.py`; keywords observed in patches: eagle, spec, moe, attention, config, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_eagle_dp_attention.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/communicator.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12078 - [Ascend] qwen optimization

- Link: https://github.com/sgl-project/sglang/pull/12078
- Status/date: `merged`, created 2025-10-24, merged 2025-11-25; author `Liwansi`.
- Diff scope read: `16` files, `+561/-108`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime; keywords: cache, config, cuda, kv, moe, attention, deepep, expert, mla, quant.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +137/-0 (137 lines); hunks: logger = logging.getLogger(__name__); def npu_fused_moe_without_routing_weights_bf16(; symbols: DeepEPMoE, npu_fused_moe_without_routing_weights_bf16, NpuFuseEPMoE, __init__
  - `python/sglang/srt/layers/attention/ascend_backend.py` modified +85/-45 (130 lines); hunks: def forward_decode_graph(; symbols: forward_decode_graph
  - `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` added +97/-0 (97 lines); hunks: +from __future__ import annotations; symbols: FuseEPDispatchOutput, format, FuseEPCombineInput, format
  - `python/sglang/srt/models/qwen3_moe.py` modified +56/-4 (60 lines); hunks: is_cuda,; logger = logging.getLogger(__name__); symbols: Qwen3MoeSparseMoeBlock, forward, op_core, forward_prepare
  - `python/sglang/srt/models/qwen3.py` modified +40/-4 (44 lines); hunks: _is_cuda = is_cuda(); def _apply_qk_norm(; symbols: Qwen3Attention, __init__, _apply_qk_norm, forward_prepare_native
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py`; keywords observed in patches: cache, config, cuda, kv, moe, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13489 - Flashinfer TRTLLM-GEN-MoE + Qwen3

- Link: https://github.com/sgl-project/sglang/pull/13489
- Status/date: `merged`, created 2025-11-18, merged 2025-11-18; author `b8zhong`.
- Diff scope read: `2` files, `+43/-1`; areas: model wrapper, MoE/router; keywords: attention, config, moe, quant, cache, expert, flash, fp8, spec, topk.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +41/-1 (42 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class; def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: attention, config, moe, quant, cache, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13715 - Fix EPLB + FP4 Quantization Compatibility Issue

- Link: https://github.com/sgl-project/sglang/pull/13715
- Status/date: `merged`, created 2025-11-21, merged 2026-01-10; author `shifangx`.
- Diff scope read: `8` files, `+49/-3`; areas: model wrapper, MoE/router; keywords: expert, moe, quant, config, topk, triton, attention, fp8, deepep, flash.
- Code diff details:
  - `python/sglang/srt/layers/moe/utils.py` modified +12/-0 (12 lines); hunks: def get_tbo_token_distribution_threshold() -> float:; symbols: get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: DispatchOutput,; def get_moe_weights(self):; symbols: get_moe_weights, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +7/-1 (8 lines); hunks: from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class; def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunks: from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class; def get_moe_weights(self):; symbols: get_moe_weights, forward_normal
  - `python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunks: from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE; def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py`; keywords observed in patches: expert, moe, quant, config, topk, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13998 - [apply][2/2] Fused qk_norm_rope for Qwen3-MoE

- Link: https://github.com/sgl-project/sglang/pull/13998
- Status/date: `merged`, created 2025-11-26, merged 2025-12-07; author `yuan-luo`.
- Diff scope read: `2` files, `+199/-22`; areas: model wrapper, MoE/router; keywords: attention, cache, config, cuda, expert, flash, kv, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +193/-22 (215 lines); hunks: """Inference-only Qwen3MoE model compatible with HuggingFace weights."""; is_npu,; symbols: compute_yarn_parameters, get_mscale, find_correction_dim, find_correction_range
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs:, add_cli_args
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: attention, cache, config, cuda, expert, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14093 - Add fused FP8 KV cache write kernel for TRTLLM MHA backend

- Link: https://github.com/sgl-project/sglang/pull/14093
- Status/date: `merged`, created 2025-11-28, merged 2025-12-05; author `harvenstar`.
- Diff scope read: `4` files, `+854/-7`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks; keywords: cache, kv, attention, fp8, cuda, quant, triton, flash, moe, test.
- Code diff details:
  - `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py` added +467/-0 (467 lines); hunks: +"""; symbols: _process_kv_tensor, _fused_fp8_set_kv_buffer_kernel, fused_fp8_set_kv_buffer, _naive_fp8_set_kv_buffer
  - `test/manual/test_trtllm_fp8_kv_kernel.py` added +306/-0 (306 lines); hunks: +"""; symbols: TestTRTLLMFP8KVKernel, setUpClass, _test_kernel_correctness, test_basic_3d_input_3d_cache
  - `python/sglang/srt/layers/attention/trtllm_mha_backend.py` modified +72/-6 (78 lines); hunks: The kernel supports sm100 only, with sliding window and attention sink features.; FlashInferAttnBackend,; symbols: get_cuda_graph_seq_len_fill_value, _should_use_fused_fp8_path, _fused_fp8_set_kv_buffer, init_forward_metadata
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-1 (10 lines); hunks: def forward_prepare_npu(; def forward_prepare_native(; symbols: forward_prepare_npu, forward_prepare_native, forward_core
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py`, `test/manual/test_trtllm_fp8_kv_kernel.py`, `python/sglang/srt/layers/attention/trtllm_mha_backend.py`; keywords observed in patches: cache, kv, attention, fp8, cuda, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py`, `test/manual/test_trtllm_fp8_kv_kernel.py`, `python/sglang/srt/layers/attention/trtllm_mha_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15203 - [NPU] support GPTQ quantization on npu

- Link: https://github.com/sgl-project/sglang/pull/15203
- Status/date: `merged`, created 2025-12-15, merged 2026-01-29; author `22dimensions`.
- Diff scope read: `5` files, `+259/-6`; areas: model wrapper, quantization, tests/benchmarks; keywords: cache, cuda, quant, test, attention, awq, config, fp4, fp8, marlin.
- Code diff details:
  - `python/sglang/srt/layers/quantization/gptq.py` modified +178/-5 (183 lines); hunks: replace_parameter,; if _is_cuda:; symbols: __init__, __init__, __repr__, get_scaled_act_names
  - `test/srt/ascend/test_ascend_gptq.py` added +73/-0 (73 lines); hunks: +import unittest; symbols: TestAscendGPTQInt8, setUpClass, test_a_gsm8k
  - `python/sglang/srt/models/qwen3.py` modified +6/-1 (7 lines); hunks: def forward(; symbols: forward
  - `python/sglang/srt/layers/linear.py` modified +1/-0 (1 lines); hunks: "TPUInt8LinearMethod",
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: # NOTE: please sort the test cases alphabetically by the test file name
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/gptq.py`, `test/srt/ascend/test_ascend_gptq.py`, `python/sglang/srt/models/qwen3.py`; keywords observed in patches: cache, cuda, quant, test, attention, awq. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/gptq.py`, `test/srt/ascend/test_ascend_gptq.py`, `python/sglang/srt/models/qwen3.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15223 - [bug fix][pp] fix qwen3 model load

- Link: https://github.com/sgl-project/sglang/pull/15223
- Status/date: `merged`, created 2025-12-16, merged 2025-12-17; author `XucSh`.
- Diff scope read: `1` files, `+3/-3`; areas: model wrapper; keywords: config.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +3/-3 (6 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`; keywords observed in patches: config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15390 - [NPU]qwen3 pp bugfix

- Link: https://github.com/sgl-project/sglang/pull/15390
- Status/date: `merged`, created 2025-12-18, merged 2025-12-24; author `Liwansi`.
- Diff scope read: `2` files, `+4/-3`; areas: model wrapper, MoE/router; keywords: kv, moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +3/-2 (5 lines); hunks: def forward_prepare_native(self, positions, hidden_states):; def forward(; symbols: forward_prepare_native, forward_prepare_npu, forward_prepare_npu, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: def forward_prepare_npu(; symbols: forward_prepare_npu
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15835 - [Feature] JIT Fused QK norm + qk norm clean up

- Link: https://github.com/sgl-project/sglang/pull/15835
- Status/date: `merged`, created 2025-12-25, merged 2025-12-28; author `DarkSharpness`.
- Diff scope read: `15` files, `+827/-127`; areas: model wrapper, MoE/router, kernel, tests/benchmarks; keywords: cuda, kv, cache, test, flash, spec, triton, attention, benchmark, config.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/norm.cuh` added +202/-0 (202 lines); hunks: +#include <sgl_kernel/runtime.cuh>; symbols: QKNormParams, auto, uint32_t, uint32_t
  - `python/sglang/jit_kernel/utils.py` modified +149/-1 (150 lines); hunks: from __future__ import annotations; def load_jit(; symbols: load_jit, cache_once, wrapper, is_arch_support_pdl
  - `python/sglang/jit_kernel/benchmark/bench_qknorm.py` added +130/-0 (130 lines); hunks: +import itertools; symbols: sglang_aot_qknorm, sglang_jit_qknorm, flashinfer_qknorm, torch_impl_qknorm
  - `python/sglang/jit_kernel/tests/test_qknorm.py` added +85/-0 (85 lines); hunks: +import torch; symbols: sglang_aot_qknorm, sglang_jit_qknorm, flashinfer_qknorm, torch_impl_qknorm
  - `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunks: # See the License for the specific language governing permissions and; def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/csrc/norm.cuh`, `python/sglang/jit_kernel/utils.py`, `python/sglang/jit_kernel/benchmark/bench_qknorm.py`; keywords observed in patches: cuda, kv, cache, test, flash, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/csrc/norm.cuh`, `python/sglang/jit_kernel/utils.py`, `python/sglang/jit_kernel/benchmark/bench_qknorm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15890 - [PP] fix wrong weight logic for tie_word_embeddings model

- Link: https://github.com/sgl-project/sglang/pull/15890
- Status/date: `merged`, created 2025-12-26, merged 2026-01-27; author `XucSh`.
- Diff scope read: `2` files, `+19/-48`; areas: model wrapper; keywords: config, processor, vision, cache, eagle.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +10/-24 (34 lines); hunks: def __init__(; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: __init__, load_weights, load_weights
  - `python/sglang/srt/models/qwen2.py` modified +9/-24 (33 lines); hunks: def __init__(; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: __init__, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`; keywords observed in patches: config, processor, vision, cache, eagle. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16115 - [NPU][Bugfix] Fix qwen3 error when enable-dp-lm-head

- Link: https://github.com/sgl-project/sglang/pull/16115
- Status/date: `merged`, created 2025-12-30, merged 2026-01-08; author `chenxu214`.
- Diff scope read: `8` files, `+52/-16`; areas: model wrapper, MoE/router; keywords: kv, doc, cache, config, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/llama.py` modified +37/-4 (41 lines); hunks: maybe_remap_kv_scale_name,; def __init__(; symbols: LlamaMLP, __init__, forward_prepare_native, forward_prepare_npu
  - `python/sglang/srt/models/qwen3.py` modified +4/-3 (7 lines); hunks: def forward_prepare_npu(self, positions, hidden_states, forward_batch):; def __init__(; symbols: forward_prepare_npu, __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: def forward_prepare_npu(; symbols: forward_prepare_npu
  - `python/sglang/srt/layers/rotary_embedding.py` modified +2/-2 (4 lines); hunks: def forward_npu(; symbols: forward_npu
  - `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +3/-1 (4 lines); hunks: cpu_has_amx_support,; def __post_init__(self):; symbols: __post_init__, get_masked_input_and_mask
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: kv, doc, cache, config, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17535 - Update weight rename check for Qwen3 Embeddings

- Link: https://github.com/sgl-project/sglang/pull/17535
- Status/date: `merged`, created 2026-01-21, merged 2026-02-03; author `satyamk7054`.
- Diff scope read: `1` files, `+5/-1`; areas: model wrapper; keywords: config.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +5/-1 (6 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`; keywords observed in patches: config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17784 - Upgrade transformers==5.3.0

- Link: https://github.com/sgl-project/sglang/pull/17784
- Status/date: `merged`, created 2026-01-26, merged 2026-03-18; author `JustinTong0323`.
- Diff scope read: `95` files, `+1136/-343`; areas: model wrapper, MoE/router, quantization, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, attention, kv, processor, vision, cache, cuda, moe, spec, test.
- Code diff details:
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +546/-21 (567 lines); hunks: def download_from_hf(; def get_hf_text_config(config: PretrainedConfig):; symbols: download_from_hf, get_rope_config, _patch_text_config, get_hf_text_config
  - `test/registered/vlm/test_vlm_input_format.py` modified +122/-17 (139 lines); hunks: def forward(self, x):; def setUpClass(cls):; symbols: forward, setUpClass, TestQwenVLUnderstandsImage, _init_visual
  - `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: def __init__(; class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, __init__, __init__
  - `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: from __future__ import annotations; from sglang.srt.layers.rotary_embedding.yarn import YaRNScalingRotaryEmbedding; symbols: _get_rope_param, get_rope, get_rope, get_rope
  - `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: class ModelImpl(str, Enum):; def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, is_deepseek_nsa, is_deepseek_nsa
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/utils/hf_transformers_utils.py`, `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/models/gemma3_causal.py`; keywords observed in patches: config, attention, kv, processor, vision, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/utils/hf_transformers_utils.py`, `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/models/gemma3_causal.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18189 - [ModelOpt] Fix broken Qwen3-235B-A22B-Instruct-2507-NVFP4 launch

- Link: https://github.com/sgl-project/sglang/pull/18189
- Status/date: `merged`, created 2026-02-03, merged 2026-02-08; author `vincentzed`.
- Diff scope read: `1` files, `+8/-0`; areas: model wrapper, MoE/router; keywords: config, fp4, kv, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +8/-0 (8 lines); hunks: def __init__(; symbols: __init__, Qwen3MoeForCausalLM, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: config, fp4, kv, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18233 - Support Qwen3 MoE context parallel

- Link: https://github.com/sgl-project/sglang/pull/18233
- Status/date: `merged`, created 2026-02-04, merged 2026-03-22; author `Shunkangz`.
- Diff scope read: `19` files, `+968/-73`; areas: model wrapper, attention/backend, MoE/router, scheduler/runtime, tests/benchmarks; keywords: attention, config, cuda, flash, moe, cache, expert, kv, spec, test.
- Code diff details:
  - `python/sglang/srt/layers/utils/cp_utils.py` added +460/-0 (460 lines); hunks: +from dataclasses import dataclass; symbols: ContextParallelMetadata:, is_prefill_context_parallel_enabled, is_prefill_cp_in_seq_split, can_cp_split
  - `python/sglang/test/attention/test_flashattn_backend.py` modified +106/-22 (128 lines); hunks: from sglang.srt.layers.radix_attention import RadixAttention; def __init__(; symbols: __init__, __init__, _verify_output, _create_forward_batch
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +73/-20 (93 lines); hunks: from sglang.srt.configs.model_config import AttentionArch; def __init__(; symbols: __init__, forward_extend, forward_extend, forward_extend
  - `test/registered/4-gpu-models/test_qwen3_30b.py` added +77/-0 (77 lines); hunks: +import unittest; symbols: TestQwen330B, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +36/-5 (41 lines); hunks: import torch; def ensure_workspace_initialized(; symbols: ensure_workspace_initialized, ensure_workspace_initialized, fake_flashinfer_allreduce_residual_rmsnorm, flashinfer_allreduce_residual_rmsnorm
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/test/attention/test_flashattn_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`; keywords observed in patches: attention, config, cuda, flash, moe, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/test/attention/test_flashattn_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19059 - [jit_kernel] Add fused_qknorm_rope JIT kernel

- Link: https://github.com/sgl-project/sglang/pull/19059
- Status/date: `merged`, created 2026-02-20, merged 2026-03-27; author `Johnsonms`.
- Diff scope read: `5` files, `+1127/-3`; areas: model wrapper, MoE/router, kernel, tests/benchmarks; keywords: cuda, kv, attention, config, moe, test, benchmark, cache, spec, triton.
- Code diff details:
  - `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` added +444/-0 (444 lines); hunks: +"""; symbols: _compute_inv_freq_yarn, fused_qk_norm_rope_ref, rms_norm_heads, apply_interleave
  - `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` added +307/-0 (307 lines); hunks: +/*; symbols: void, int, parameters, arguments
  - `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` added +183/-0 (183 lines); hunks: +"""; symbols: bench_fused_qknorm_rope, calculate_diff
  - `python/sglang/jit_kernel/fused_qknorm_rope.py` added +181/-0 (181 lines); hunks: +from __future__ import annotations; symbols: _jit_fused_qknorm_rope_module, fused_qk_norm_rope_out, can_use_fused_qk_norm_rope, fused_qk_norm_rope
  - `python/sglang/srt/models/qwen3_moe.py` modified +12/-3 (15 lines); hunks: _is_cuda = is_cuda(); def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`; keywords observed in patches: cuda, kv, attention, config, moe, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19532 - [NPU] bugs fix: fix a condition bug when using speculative inference on Qwen3 and Qwen3 moe

- Link: https://github.com/sgl-project/sglang/pull/19532
- Status/date: `merged`, created 2026-02-28, merged 2026-03-03; author `shengzhaotian`.
- Diff scope read: `2` files, `+8/-2`; areas: model wrapper, MoE/router; keywords: moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +4/-1 (5 lines); hunks: def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-1 (5 lines); hunks: def forward_prepare(; symbols: forward_prepare
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20127 - [Qwen] Handle tie_word_embeddings for Qwen MoE and Qwen3Next

- Link: https://github.com/sgl-project/sglang/pull/20127
- Status/date: `open`, created 2026-03-08; author `xingsy97`.
- Diff scope read: `3` files, `+66/-25`; areas: model wrapper, MoE/router; keywords: config, processor, quant, eagle, moe, attention, cache, cuda.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +25/-8 (33 lines); hunks: from sglang.srt.layers.quantization.base_config import QuantizationConfig; def __init__(; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +24/-7 (31 lines); hunks: def __init__(; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen3_next.py` modified +17/-10 (27 lines); hunks: def __init__(; def get_embed_and_head(self):; symbols: __init__, get_embed_and_head, set_embed_and_head, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: config, processor, quant, eagle, moe, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20474 - Intel XPU: Qwen3 support (layernorm/MRoPE) + test_qwen3

- Link: https://github.com/sgl-project/sglang/pull/20474
- Status/date: `open`, created 2026-03-12; author `jmunetong`.
- Diff scope read: `6` files, `+159/-7`; areas: attention/backend, tests/benchmarks; keywords: test, triton, attention, doc, cache, cuda, kv, processor, spec.
- Code diff details:
  - `test/srt/xpu/test_qwen3.py` added +133/-0 (133 lines); hunks: +"""; symbols: TestQwen3, setUpClass, tearDownClass, get_request_json
  - `docker/xpu.Dockerfile` modified +11/-6 (17 lines); hunks: ARG SG_LANG_KERNEL_BRANCH=main; RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda-forge/miniforge/re
  - `python/sglang/srt/layers/rotary_embedding/mrope.py` modified +9/-0 (9 lines); hunks: def forward_npu(; symbols: forward_npu, forward_xpu, get_rope_index
  - `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +4/-0 (4 lines); hunks: device_context,; def _layer_norm_fwd_1pass_kernel(; symbols: _layer_norm_fwd_1pass_kernel, _get_sm_count
  - `.github/workflows/pr-test-xpu.yml` modified +1/-1 (2 lines); hunks: jobs:
- Optimization/support interpretation: The concrete diff surface is `test/srt/xpu/test_qwen3.py`, `docker/xpu.Dockerfile`, `python/sglang/srt/layers/rotary_embedding/mrope.py`; keywords observed in patches: test, triton, attention, doc, cache, cuda. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/xpu/test_qwen3.py`, `docker/xpu.Dockerfile`, `python/sglang/srt/layers/rotary_embedding/mrope.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20520 - [NPU]TP Communications compression For Qwen3 models for NPU

- Link: https://github.com/sgl-project/sglang/pull/20520
- Status/date: `open`, created 2026-03-13; author `egvenediktov`.
- Diff scope read: `12` files, `+172/-10`; areas: model wrapper, quantization, tests/benchmarks, docs/config; keywords: quant, attention, config, cuda, moe, test, cache, spec.
- Code diff details:
  - `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py` added +37/-0 (37 lines); hunks: +import unittest; symbols: TestLlama
  - `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py` added +37/-0 (37 lines); hunks: +import os; symbols: TestQwen38BCommQuantization
  - `python/sglang/srt/distributed/device_communicators/npu_communicator.py` modified +29/-1 (30 lines); hunks: from sglang.srt.utils import is_npu; def all_reduce(self, x: torch.Tensor) -> torch.Tensor:; symbols: NpuCommunicator:, __init__, all_reduce, quant_all_reduce
  - `python/sglang/srt/server_args.py` modified +21/-0 (21 lines); hunks: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs:, __post_init__, add_cli_args, from_cli_args
  - `python/sglang/srt/distributed/parallel_state.py` modified +14/-0 (14 lines); hunks: def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:; symbols: all_reduce, quant_all_reduce, fused_allreduce_rmsnorm
- Optimization/support interpretation: The concrete diff surface is `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py`, `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py`, `python/sglang/srt/distributed/device_communicators/npu_communicator.py`; keywords observed in patches: quant, attention, config, cuda, moe, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py`, `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py`, `python/sglang/srt/distributed/device_communicators/npu_communicator.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20931 - [Bugifx] qwen3 rope parameter compatibility

- Link: https://github.com/sgl-project/sglang/pull/20931
- Status/date: `merged`, created 2026-03-19, merged 2026-03-20; author `lviy`.
- Diff scope read: `1` files, `+4/-3`; areas: model wrapper, MoE/router; keywords: attention, config, cuda, kv, moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: is_non_idle_and_non_empty,; def forward_prepare_native(; symbols: forward_prepare_native, apply_qk_norm_rope, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: attention, config, cuda, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21195 - Enable the qwen3 test

- Link: https://github.com/sgl-project/sglang/pull/21195
- Status/date: `merged`, created 2026-03-23, merged 2026-03-24; author `Shunkangz`.
- Diff scope read: `2` files, `+6/-5`; areas: model wrapper, MoE/router, tests/benchmarks; keywords: cuda, expert, fp8, moe, router, test, topk.
- Code diff details:
  - `test/registered/4-gpu-models/test_qwen3_30b.py` modified +2/-5 (7 lines); hunks: popen_launch_server,
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-0 (4 lines); hunks: get_moe_tensor_parallel_world_size,; def forward_normal(; symbols: forward_normal
- Optimization/support interpretation: The concrete diff surface is `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/models/qwen3_moe.py`; keywords observed in patches: cuda, expert, fp8, moe, router, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/models/qwen3_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21412 - [Bugfix] Fix Qwen3 RoPE config compatibility for old-style checkpoints

- Link: https://github.com/sgl-project/sglang/pull/21412
- Status/date: `open`, created 2026-03-25; author `rbqlsquf`.
- Diff scope read: `1` files, `+2/-2`; areas: model wrapper; keywords: attention, config, cuda.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +2/-2 (4 lines); hunks: from sglang.srt.models.utils import apply_qk_norm; def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`; keywords observed in patches: attention, config, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21654 - [jit_kernel] Optimize fused_qknorm_rope: deduplicate sincosf for interleave RoPE

- Link: https://github.com/sgl-project/sglang/pull/21654
- Status/date: `merged`, created 2026-03-30, merged 2026-04-01; author `Johnsonms`.
- Diff scope read: `5` files, `+208/-77`; areas: model wrapper, MoE/router, kernel, tests/benchmarks; keywords: kv, attention, cuda, config, moe, benchmark, cache, spec, test.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` modified +94/-55 (149 lines); hunks: namespace {; compute_freq_yarn(float base, int rotary_dim, int half_dim, float factor, float; symbols: void, void, void, void
  - `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` modified +85/-4 (89 lines); hunks: """; ci_range=[64, 512],; symbols: bench_fused_qknorm_rope, bench_fused_qknorm_rope_production, calculate_diff, calculate_diff
  - `python/sglang/jit_kernel/fused_qknorm_rope.py` modified +25/-16 (41 lines); hunks: @cache_once; def fused_qk_norm_rope_out(; symbols: _jit_fused_qknorm_rope_module, _jit_fused_qknorm_rope_module, fused_qk_norm_rope_out, fused_qk_norm_rope_out
  - `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` modified +2/-2 (4 lines); hunks: def apply_interleave(x):; def test_fused_qknorm_rope_partial_rotary(head_dim, is_neox):; symbols: apply_interleave, apply_neox, test_fused_qknorm_rope_partial_rotary
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`, `python/sglang/jit_kernel/fused_qknorm_rope.py`; keywords observed in patches: kv, attention, cuda, config, moe, benchmark. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`, `python/sglang/jit_kernel/fused_qknorm_rope.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21770 - [Apple][MLX][Test] Add Qwen3 correctness and accuracy tests for Apple Silicon

- Link: https://github.com/sgl-project/sglang/pull/21770
- Status/date: `open`, created 2026-03-31; author `linzhonghong`.
- Diff scope read: `2` files, `+159/-0`; areas: model wrapper, tests/benchmarks; keywords: cache, cuda, test.
- Code diff details:
  - `test/registered/models/test_qwen3_mlx_correctness.py` added +89/-0 (89 lines); hunks: +import os; symbols: TestQwen3MlxCorrectness, setUpClass, tearDownClass, _chat
  - `test/registered/models/test_qwen3_mlx_accuracy.py` added +70/-0 (70 lines); hunks: +import os; symbols: TestQwen3MlxAccuracy, setUpClass, tearDownClass, test_gsm8k_accuracy
- Optimization/support interpretation: The concrete diff surface is `test/registered/models/test_qwen3_mlx_correctness.py`, `test/registered/models/test_qwen3_mlx_accuracy.py`; keywords observed in patches: cache, cuda, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/models/test_qwen3_mlx_correctness.py`, `test/registered/models/test_qwen3_mlx_accuracy.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22003 - Support moe_dp_size = 1 for various attention_cp_size

- Link: https://github.com/sgl-project/sglang/pull/22003
- Status/date: `merged`, created 2026-04-03, merged 2026-04-20; author `Shunkangz`.
- Diff scope read: `8` files, `+276/-25`; areas: model wrapper, attention/backend, MoE/router, tests/benchmarks; keywords: moe, attention, cuda, config, expert, flash, fp4, spec, test.
- Code diff details:
  - `python/sglang/srt/layers/communicator.py` modified +164/-10 (174 lines); hunks: get_dp_global_num_tokens,; class ScatterMode(Enum):; symbols: ScatterMode, model_input_output, _compute_layer_input_mode, _compute_mlp_mode
  - `test/registered/4-gpu-models/test_qwen3_30b.py` modified +55/-0 (55 lines); hunks: def test_gsm8k(self):; symbols: test_gsm8k, TestQwen330BCP, setUpClass, tearDownClass
  - `python/sglang/srt/layers/dp_attention.py` modified +28/-0 (28 lines); hunks: get_attn_tensor_model_parallel_rank,; def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):; symbols: attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank, get_moe_cp_size
  - `python/sglang/srt/distributed/parallel_state.py` modified +13/-7 (20 lines); hunks: def initialize_model_parallel(; def initialize_model_parallel(; symbols: initialize_model_parallel, initialize_model_parallel, destroy_model_parallel, destroy_model_parallel
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: def __init__(; symbols: __init__, get_input_embeddings
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py`; keywords observed in patches: moe, attention, cuda, config, expert, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22358 - Enable DFLASH support for additional model backends

- Link: https://github.com/sgl-project/sglang/pull/22358
- Status/date: `merged`, created 2026-04-08, merged 2026-04-09; author `mmangkad`.
- Diff scope read: `8` files, `+152/-5`; areas: model wrapper, MoE/router; keywords: flash, eagle, config, expert, kv, moe, processor, spec.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: def forward(; def forward(; symbols: forward, forward, get_layer, get_input_embeddings
  - `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):; def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: def __init__(; def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture
  - `python/sglang/srt/models/qwen3_vl.py` modified +16/-0 (16 lines); hunks: def __init__(; def forward(; symbols: __init__, forward, set_dflash_layers_to_capture, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: flash, eagle, config, expert, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22429 - [NPU]add Qwen3-32b and Qwen3-8b low latency md

- Link: https://github.com/sgl-project/sglang/pull/22429
- Status/date: `merged`, created 2026-04-09, merged 2026-04-09; author `Liwansi`.
- Diff scope read: `1` files, `+296/-0`; areas: docs/config; keywords: attention, benchmark, cache, config, cuda, doc, eagle, quant, spec, test.
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +296/-0 (296 lines); hunks: you encounter issues or have any questions, please [open an issue](https://githu; We tested it based on the `RANDOM` dataset.
- Optimization/support interpretation: The concrete diff surface is `docs/platforms/ascend/ascend_npu_best_practice.md`; keywords observed in patches: attention, benchmark, cache, config, cuda, doc. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/platforms/ascend/ascend_npu_best_practice.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22446 - [NPU] add qwen3-30b-a3b low latency example

- Link: https://github.com/sgl-project/sglang/pull/22446
- Status/date: `merged`, created 2026-04-09, merged 2026-04-11; author `heziiop`.
- Diff scope read: `1` files, `+130/-0`; areas: docs/config; keywords: attention, benchmark, cache, config, cuda, doc, eagle, quant, spec, test.
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +130/-0 (130 lines); hunks: you encounter issues or have any questions, please [open an issue](https://githu; We tested it based on the `RANDOM` dataset.
- Optimization/support interpretation: The concrete diff surface is `docs/platforms/ascend/ascend_npu_best_practice.md`; keywords observed in patches: attention, benchmark, cache, config, cuda, doc. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/platforms/ascend/ascend_npu_best_practice.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22450 - [NPU] Add Qwen3-14B low latency doc

- Link: https://github.com/sgl-project/sglang/pull/22450
- Status/date: `open`, created 2026-04-09; author `LinyuanLi0046`.
- Diff scope read: `1` files, `+323/-0`; areas: docs/config; keywords: attention, benchmark, cache, config, cuda, doc, eagle, quant, scheduler, spec.
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +323/-0 (323 lines); hunks: you encounter issues or have any questions, please [open an issue](https://githu; you encounter issues or have any questions, please [open an issue](https://git
- Optimization/support interpretation: The concrete diff surface is `docs/platforms/ascend/ascend_npu_best_practice.md`; keywords observed in patches: attention, benchmark, cache, config, cuda, doc. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/platforms/ascend/ascend_npu_best_practice.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22529 - [Model] Support sliding window attention for Qwen3

- Link: https://github.com/sgl-project/sglang/pull/22529
- Status/date: `open`, created 2026-04-10; author `bzantium`.
- Diff scope read: `1` files, `+29/-0`; areas: model wrapper; keywords: attention, config, cuda, eagle, kv, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +29/-0 (29 lines); hunks: Qwen3Config = None; def __init__(; symbols: get_attention_sliding_window_size, __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`; keywords observed in patches: attention, config, cuda, eagle, kv, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22674 - [NPU] Support Qwen3.5-MoE and Qwen3-Next quantization

- Link: https://github.com/sgl-project/sglang/pull/22674
- Status/date: `open`, created 2026-04-13; author `Dmovic`.
- Diff scope read: `1` files, `+2/-0`; areas: misc; keywords: config, kv, quant.
- Code diff details:
  - `python/sglang/srt/model_loader/loader.py` modified +2/-0 (2 lines); hunks: def _get_quantization_config(; symbols: _get_quantization_config
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_loader/loader.py`; keywords observed in patches: config, kv, quant. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_loader/loader.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22687 - [NPU]qwen3-8b and 32b md bugfix

- Link: https://github.com/sgl-project/sglang/pull/22687
- Status/date: `merged`, created 2026-04-13, merged 2026-04-13; author `Liwansi`.
- Diff scope read: `1` files, `+4/-8`; areas: docs/config; keywords: cache, cuda, doc, eagle, quant, spec, topk.
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +4/-8 (12 lines); hunks: LOCAL_HOST2=`hostname -I\|awk -F " " '{print$2}'`; python -m sglang.launch_server --model-path $MODEL_PATH \
- Optimization/support interpretation: The concrete diff surface is `docs/platforms/ascend/ascend_npu_best_practice.md`; keywords observed in patches: cache, cuda, doc, eagle, quant, spec. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/platforms/ascend/ascend_npu_best_practice.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22739 - Restore Qwen3 rope config fallback

- Link: https://github.com/sgl-project/sglang/pull/22739
- Status/date: `merged`, created 2026-04-14, merged 2026-04-14; author `ishandhanani`.
- Diff scope read: `1` files, `+10/-2`; areas: model wrapper; keywords: attention, config.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +10/-2 (12 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`; keywords observed in patches: attention, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22837 - [Bug] Qwen3 reasoning detector silently swallows tool_call when </think> is missing

- Link: https://github.com/sgl-project/sglang/pull/22837
- Status/date: `open`, created 2026-04-15; author `gucasbrg`.
- Diff scope read: `2` files, `+43/-0`; areas: tests/benchmarks; keywords: test.
- Code diff details:
  - `test/registered/unit/parser/test_reasoning_parser.py` modified +42/-0 (42 lines); hunks: def test_streaming_qwen3_forced_reasoning_format(self):; symbols: test_streaming_qwen3_forced_reasoning_format, test_detect_and_parse_tool_call_without_think_close, test_streaming_tool_call_without_think_close, TestKimiDetector
  - `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`; keywords observed in patches: test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23372 - [NPU] Add CI tests for Speculative Decoding

- Link: https://github.com/sgl-project/sglang/pull/23372
- Status/date: `open`, created 2026-04-21; author `EdwardXuy`.
- Diff scope read: `7` files, `+729/-14`; areas: attention/backend, MoE/router, tests/benchmarks; keywords: eagle, spec, test, cache, attention, quant, topk, cuda, config, deepep.
- Code diff details:
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py` added +185/-0 (185 lines); hunks: +import os; symbols: TestAscendSpeculativeAttentionMode, setUpClass, start_prefill, start_decode
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py` added +159/-0 (159 lines); hunks: +import os; symbols: TestNpuSpeculativeDraftParams, setUpClass, tearDownClass, test_draft_params_via_server_info
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` added +156/-0 (156 lines); hunks: +import os; symbols: TestNpuSpeculativeTokenMap, test_eagle3_ignores_token_map_gsm8k, test_eagle_with_valid_token_map_gsm8k
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_draft_attention_backend.py` added +105/-0 (105 lines); hunks: +import os; symbols: TestAscendSpeculativeDraftAttentionAndMoeRunner, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_moe_a2a_backend.py` added +97/-0 (97 lines); hunks: +import os; symbols: TestAscendSpeculativeMoeA2ABackend, setUpClass, test_a_gsm8k
- Optimization/support interpretation: The concrete diff surface is `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py`; keywords observed in patches: eagle, spec, test, cache, attention, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23397 - [alignment-sglang] PR3: Dense Deterministic Math

- Link: https://github.com/sgl-project/sglang/pull/23397
- Status/date: `open`, created 2026-04-21; author `maocheng23`.
- Diff scope read: `16` files, `+2285/-50`; areas: model wrapper, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks; keywords: attention, cuda, moe, spec, config, flash, processor, quant, test, cache.
- Code diff details:
  - `test/registered/core/test_tp_invariant_ops.py` added +866/-0 (866 lines); hunks: +"""Tests for TP-invariant kernels (PR1).; symbols: _simulate_tp_matmul, TestTPInvariantMode, tearDown, test_mode_context_restores_previous_state
  - `test/registered/core/test_on_policy_wiring.py` added +527/-0 (527 lines); hunks: +import json; symbols: _run_server_args_script, install_openai_stubs, _mock_model_config, TestOnPolicyServerArgs
  - `test/registered/core/test_dense_deterministic_math.py` added +293/-0 (293 lines); hunks: +import json; symbols: _run_dense_math_script, install_openai_stubs, TestDenseOnPolicyHelpers, test_default_dense_math_helpers_are_inactive
  - `python/sglang/srt/layers/on_policy_utils.py` added +222/-0 (222 lines); hunks: +from __future__ import annotations; symbols: _get_server_args, get_rl_on_policy_target, is_true_on_policy_enabled, is_tp_invariant_target
  - `python/sglang/srt/tp_invariant_ops/tp_invariant_ops.py` added +219/-0 (219 lines); hunks: +import contextlib; symbols: is_tp_invariant_mode_enabled, enable_tp_invariant_mode, disable_tp_invariant_mode, set_tp_invariant_mode
- Optimization/support interpretation: The concrete diff surface is `test/registered/core/test_tp_invariant_ops.py`, `test/registered/core/test_on_policy_wiring.py`, `test/registered/core/test_dense_deterministic_math.py`; keywords observed in patches: attention, cuda, moe, spec, config, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/core/test_tp_invariant_ops.py`, `test/registered/core/test_on_policy_wiring.py`, `test/registered/core/test_dense_deterministic_math.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23434 - [Model] Qwen3ForPooledOutput: forward get_input_embeddings to inner model

- Link: https://github.com/sgl-project/sglang/pull/23434
- Status/date: `open`, created 2026-04-22; author `fortunecookiee`.
- Diff scope read: `1` files, `+3/-0`; areas: model wrapper; keywords: config.
- Code diff details:
  - `python/sglang/srt/models/qwen3_classification.py` modified +3/-0 (3 lines); hunks: def __init__(; symbols: __init__, get_input_embeddings, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_classification.py`; keywords observed in patches: config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_classification.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 79; open PRs: 13.
- Open PRs to keep tracking: [#9147](https://github.com/sgl-project/sglang/pull/9147), [#20127](https://github.com/sgl-project/sglang/pull/20127), [#20474](https://github.com/sgl-project/sglang/pull/20474), [#20520](https://github.com/sgl-project/sglang/pull/20520), [#21412](https://github.com/sgl-project/sglang/pull/21412), [#21770](https://github.com/sgl-project/sglang/pull/21770), [#22450](https://github.com/sgl-project/sglang/pull/22450), [#22529](https://github.com/sgl-project/sglang/pull/22529), [#22674](https://github.com/sgl-project/sglang/pull/22674), [#22837](https://github.com/sgl-project/sglang/pull/22837), [#23372](https://github.com/sgl-project/sglang/pull/23372), [#23397](https://github.com/sgl-project/sglang/pull/23397), [#23434](https://github.com/sgl-project/sglang/pull/23434)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
