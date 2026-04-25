# Qwen3 Core PR Diff Dossier

Evidence snapshot:

- SGLang mainline checked around `2026-04-22`: `b3e6cf60a`
- sgl-cookbook mainline checked around `2026-04-21`: `816bad5`
- Scope: `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, Qwen3 dense/MoE shared parser, quantization, PP/DP/EP/CP, EAGLE3, NPU/XPU/MLX, and Qwen3 low-latency docs.
- Runtime files reviewed repeatedly: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/moe/`, `python/sglang/srt/layers/quantization/`, `python/sglang/srt/distributed/`, `python/sglang/srt/function_call/qwen25_detector.py`, Qwen3 registered/manual tests, NPU tests, and Qwen docs/cookbook.

This file intentionally uses per-PR diff cards. Do not replace these cards with generated summaries; update a card only after reading the PR diff or final merge commit.

## Initial Model Bring-Up

### [#4693](https://github.com/sgl-project/sglang/pull/4693) - Add Qwen3 and Qwen3MoE

- Motivation: add first-class support for the newly released Qwen3 dense and Qwen3 MoE architectures instead of relying on Qwen2 compatibility. The diff references upstream HF/vLLM work and fixes the missing `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM` dispatch path.
- Key implementation: introduced `qwen3.py` and `qwen3_moe.py`; split Q/K/V from a packed `QKVParallelLinear`; applied Q/K RMSNorm before RoPE; reused `RadixAttention`; added MoE routing with `ReplicatedLinear` gate and `FusedMoE`; extended Qwen2 base models to accept a `decoder_layer_type`; increased FlashInfer workspace for Qwen3.
- Key diff excerpt:

```python
self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q, k = self._apply_qk_norm(q, k)
q, k = self.rotary_emb(positions, q, k)
```

```python
self.experts = FusedMoE(
    num_experts=config.num_experts,
    top_k=config.num_experts_per_tok,
    hidden_size=config.hidden_size,
    intermediate_size=config.moe_intermediate_size,
    reduce_results=False,
    renormalize=config.norm_topk_prob,
)
```

- Validation/risk: this is the root compatibility PR. Any later Qwen3 regression should first check whether dense and MoE still preserve the initial QK-norm-before-RoPE and packed-weight mapping assumptions.

### [#6990](https://github.com/sgl-project/sglang/pull/6990) - Support Qwen3 embedding

- Motivation: Qwen3 embedding checkpoints in issue `#6917` used a model-prefixed weight layout that did not match the early Qwen3 loader.
- Key implementation: renamed unprefixed embedding checkpoint names by adding `model.` only for Qwen3 embedding models; added `Qwen/Qwen3-Embedding-8B` coverage.
- Key diff excerpt:

```python
if "Embedding" in self.config.name_or_path:
    name = add_prefix(name, "model")
```

- Validation/risk: later PR `#17535` tightens this logic because checking only the model name was too narrow.

### [#17535](https://github.com/sgl-project/sglang/pull/17535) - Update weight rename check for Qwen3 embeddings

- Motivation: fine-tuned Qwen3 embedding models may not contain `"Embedding"` in `name_or_path`; the previous rename heuristic could trigger a `KeyError` such as `layers.0.mlp.gate_up_proj.weight`.
- Key implementation: changed the loader to rename only unprefixed root model weights that start with `layers.`, `embed_tokens.`, or `norm.`.
- Key diff excerpt:

```python
if not name.startswith("model.") and (
    name.startswith("layers.")
    or name.startswith("embed_tokens.")
    or name.startswith("norm.")
):
    name = add_prefix(name, "model")
```

- Validation/risk: validate both Qwen3 embedding and normal Qwen3 causal checkpoints because the heuristic sits in the shared loader.

### [#17784](https://github.com/sgl-project/sglang/pull/17784) - Upgrade transformers compatibility

- Motivation: newer HF configuration objects moved or normalized RoPE fields, and Qwen-family model configs could arrive as dict subconfigs after the transformers upgrade.
- Key implementation: Qwen3 reads `rope_parameters` when present; shared helpers normalize legacy `rope_scaling["type"]`; `get_hf_text_config` converts dict subconfigs to `PretrainedConfig` and prioritizes `thinker_config`, `llm_config`, `language_config`, then `text_config`.
- Key diff excerpt:

```python
rope_theta = config.rope_parameters.get("rope_theta", 1000000.0)
rope_scaling = config.rope_parameters.get("rope_scaling")
```

```python
if isinstance(rs, dict) and "rope_type" in rs and "type" not in rs:
    rs["type"] = rs["rope_type"]
```

- Validation/risk: Qwen3 dense, Qwen3 MoE, Qwen3 Omni, and later Qwen3.5/Next checkpoints all depend on this config normalization.

### [#20931](https://github.com/sgl-project/sglang/pull/20931) - Qwen3 RoPE parameter compatibility

- Motivation: some Qwen3 MoE checkpoints kept top-level `rope_theta` and `rope_scaling` without `rope_parameters`; direct `config.rope_parameters[...]` access caused launch failures.
- Key implementation: imported the shared `get_rope_config(config)` helper and stored `self.rope_theta` so fused QK-norm/RoPE could use the same fallback path as native RoPE.
- Key diff excerpt:

```python
from sglang.srt.utils.hf_transformers_utils import get_rope_config

rope_theta, rope_scaling = get_rope_config(config)
self.rope_theta = rope_theta
```

- Validation/risk: cover both old-style and new-style config JSON files, especially with `--enable-fused-qk-norm-rope`.

### [#22739](https://github.com/sgl-project/sglang/pull/22739) - Restore Qwen3 RoPE config fallback

- Motivation: overriding JSON with `rope_scaling` could produce `config.rope_parameters` without `rope_theta`, reintroducing a `KeyError`.
- Key implementation: dense Qwen3 now checks that `rope_parameters` exists and contains `rope_theta`; otherwise it falls back to top-level `rope_theta` and `rope_scaling`.
- Key diff excerpt:

```python
if (
    hasattr(config, "rope_parameters")
    and config.rope_parameters
    and "rope_theta" in config.rope_parameters
):
    rope_theta = config.rope_parameters["rope_theta"]
    rope_scaling = config.rope_parameters
else:
    rope_theta = getattr(config, "rope_theta", 1000000)
    rope_scaling = getattr(config, "rope_scaling", None)
```

- Validation/risk: dense Qwen3 and MoE Qwen3 need matching fallback behavior; otherwise fused and non-fused attention paths diverge.

## MoE Parallelism, DeepEP, EPLB, and Dispatch

### [#5917](https://github.com/sgl-project/sglang/pull/5917) - Support Qwen3 EP MoE

- Motivation: Qwen3-235B-A22B-FP8 needed expert parallel MoE via `--enable-ep-moe`, not just tensor-parallel `FusedMoE`.
- Key implementation: selected `EPMoE` when EP is enabled and reused the selected implementation for expert weight mapping.
- Key diff excerpt:

```python
MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
self.experts = MoEImpl(...)
expert_params_mapping = MoEImpl.make_expert_params_mapping(...)
```

- Validation/risk: validated on 8xH200 Qwen/Qwen3-235B-A22B-FP8 TP8 with `--enable-ep-moe`; later refactors replaced the flag with `--moe-a2a-backend`.

### [#6120](https://github.com/sgl-project/sglang/pull/6120) - Support Qwen3 DeepEP

- Motivation: Qwen3 MoE needed DeepEP all-to-all dispatch for high-throughput expert serving; the initial PR copied the proven DeepSeek path and verified Qwen3-235B accuracy.
- Key implementation: chose `DeepEPMoE` when DeepEP is enabled; created `DeepEPDispatcher`; called `select_experts`; dispatched hidden states to local experts and combined the outputs.
- Key diff excerpt:

```python
MoEImpl = (
    DeepEPMoE
    if global_server_args_dict["enable_deepep_moe"]
    else (EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE)
)
```

```python
self.deepep_dispatcher = DeepEPDispatcher(
    group=parallel_state.get_tp_group().device_group,
    router_topk=self.top_k,
    permute_fusion=True,
    num_experts=config.num_experts,
    num_local_experts=config.num_experts // self.tp_size,
    hidden_size=config.hidden_size,
)
```

- Validation/risk: Qwen/Qwen3-235B-A22B-FP8 TP4 DeepEP normal reported GSM8K `0.970`; always verify BF16 and FP8 separately.

### [#6121](https://github.com/sgl-project/sglang/pull/6121) - DP attention for Qwen2/3 MoE

- Motivation: EP MoE deployments require data-parallel attention; issue `#6088` left Qwen MoE unsupported in that topology.
- Key implementation: attention uses `get_attention_tp_rank/size`; dense and sparse FFN layers choose between full and scattered input modes; `LayerCommunicator` gathers/scatters around attention and MLP.
- Key diff excerpt:

```python
attn_tp_rank = get_attention_tp_rank()
attn_tp_size = get_attention_tp_size()
self.num_heads = self.total_num_heads // attn_tp_size
```

```python
class _FFNInputMode(Enum):
    SCATTERED = auto()
    FULL = auto()
```

```python
dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
dp_scatter(residual, hidden_states, forward_batch)
```

- Validation/risk: Qwen3-30B-A3B 4xA40 MMLU TP4 baseline `0.798`, TP=DP=EP=4 `0.796`.

### [#6533](https://github.com/sgl-project/sglang/pull/6533) - Support EPLB for Qwen3

- Motivation: Qwen3 MoE needed Expert Parallel Load Balancing with redundant physical experts, matching the DeepSeek EPLB machinery.
- Key implementation: switched Qwen3 MoE to `get_moe_impl_class()`; added redundant experts; collected per-layer MoE weights; passed `ExpertLocationDispatchInfo` into routing; exposed model metadata for logical expert placement.
- Key diff excerpt:

```python
self.experts = get_moe_impl_class()(
    num_experts=config.num_experts
    + global_server_args_dict["ep_num_redundant_experts"],
    top_k=config.num_experts_per_tok,
    ...
)
```

```python
topk_weights, topk_idx = select_experts(
    ...,
    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
        layer_id=self.layer_id
    ),
)
```

- Validation/risk: Qwen3-235B-A22B-FP8 TP8 DP2 DP attention DeepEP normal EPLB reported accuracy `0.949`, latency `216.240s`, output throughput `941.167 tok/s`.

### [#6709](https://github.com/sgl-project/sglang/pull/6709) - Fix PP for Qwen3 MoE

- Motivation: the EPLB collection in `#6533` walked layers that are `PPMissingLayer` on non-local PP ranks.
- Key implementation: only collects routed expert weights for `self.start_layer <= layer_id < self.end_layer`.
- Key diff excerpt:

```python
self.routed_experts_weights_of_layer = {
    layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
    for layer_id in range(self.start_layer, self.end_layer)
    if isinstance(self.model.layers[layer_id].mlp, Qwen3MoeSparseMoeBlock)
}
```

- Validation/risk: Qwen3-30B-A3B PP2 GSM8K reported accuracy `0.791`; keep PP and EPLB tests together.

### [#6818](https://github.com/sgl-project/sglang/pull/6818) - Fix wrong weight reference in dynamic EPLB

- Motivation: dynamic EPLB referenced expert weights before the correct model-local layers were ready.
- Key implementation: introduced lazy expert-weight collection in the shared EPLB pattern and kept Qwen3 MoE aligned with the local-layer-only collection introduced by `#6709`.
- Key diff excerpt:

```python
self._routed_experts_weights_of_layer = LazyValue(
    lambda: {
        layer_id: layer.mlp.get_moe_weights()
        for layer_id, layer in enumerate(self.model.layers)
        if isinstance(layer.mlp, DeepseekV2MoE)
    }
)
```

- Validation/risk: if Qwen3 EPLB weight maps look stale, check lazy initialization and PP-local layer bounds before changing router math.

### [#6964](https://github.com/sgl-project/sglang/pull/6964) - Approximate and exact expert distribution collection

- Motivation: EPLB needed both exact top-k distribution statistics and faster approximate statistics from DeepEP normal dispatch.
- Key implementation: added GPU/CPU gatherers; exact mode uses `scatter_add_`; approximate mode reads DeepEP normal stats; Qwen3 and DeepSeek wrap top-k selection in the global expert-distribution recorder.
- Key diff excerpt:

```python
self._data[layer_idx, :].scatter_add_(
    dim=0, index=topk_ids.masked_fill(~mask, 0).long(), src=mask.int()
)
```

```python
with get_global_expert_distribution_recorder().with_current_layer(self.layer_id):
    state.topk_weights_local, state.topk_idx_local = select_experts(...)
```

- Validation/risk: use exact mode for correctness/debugging and approximate mode for production-scale DeepEP telemetry.

### [#7580](https://github.com/sgl-project/sglang/pull/7580) - Move EPLB files

- Motivation: EPLB helpers had grown into a subsystem and needed a dedicated package layout.
- Key implementation: moved expert distribution, location metadata, dispatch, and updater helpers under `python/sglang/srt/eplb/`; updated Qwen3 and DeepSeek imports.
- Key diff excerpt:

```python
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
```

- Validation/risk: import-only changes still matter for downstream skill docs because old import paths appear in older PR diffs.

### [#8448](https://github.com/sgl-project/sglang/pull/8448) - EPLB in FusedMoE

- Motivation: issue `#8398` showed FusedMoE weight loading did not understand expert-location metadata.
- Key implementation: `FusedMoE` accepts `layer_id`; logical expert IDs map to all physical expert IDs via `logical_to_all_physical`; shared experts bypass redundant mapping.
- Key diff excerpt:

```python
physical_expert_ids = global_expert_location_metadata.logical_to_all_physical(
    self.layer_id, expert_id
)
for physical_expert_id in physical_expert_ids:
    self._weight_loader_physical(..., expert_id=physical_expert_id)
```

- Validation/risk: verify both normal and redundant expert weight loading, especially FP4/NVFP4 where scales may not have local-expert leading dimensions.

### [#13715](https://github.com/sgl-project/sglang/pull/13715) - Fix EPLB + FP4 compatibility

- Motivation: ModelOpt FP4 parameters include global expert scales, swizzled blockscale tensors, and scalar metadata that do not look like local expert weights; EPLB tried to remap them and crashed.
- Key implementation: added `filter_moe_weight_param_global_expert`; Qwen2/Qwen3 MoE `get_moe_weights()` filters only tensors that truly have a local-expert leading dimension.
- Key diff excerpt:

```python
def filter_moe_weight_param_global_expert(name, x, num_local_experts):
    return (
        not getattr(x, "_sglang_require_global_experts", False)
        and not name.endswith("_blockscale_swizzled")
        and x.data.ndim > 0
        and x.data.shape[0] == num_local_experts
    )
```

- Validation/risk: EPLB plus ModelOpt FP4 must test both routed expert weights and scale tensors.

### [#6820](https://github.com/sgl-project/sglang/pull/6820) - Fix Qwen3 MoE token padding optimization

- Motivation: Qwen3 MoE did not pass non-padded token counts into top-k selection, so token padding optimization was ineffective.
- Key implementation: forwarded `forward_batch.num_token_non_padded` through `select_experts` and `fused_topk`.
- Key diff excerpt:

```python
topk_weights, topk_idx = select_experts(
    ...,
    num_token_non_padded=forward_batch.num_token_non_padded,
)
```

- Validation/risk: profile prefill with padded batches; missing this argument causes silent throughput loss, not necessarily wrong answers.

### [#7222](https://github.com/sgl-project/sglang/pull/7222) - DP attention with auto DeepEP dispatch

- Motivation: DeepEP `auto` mode had been blocked with DP attention even though Qwen3 MoE deployments needed automatic prefill/decode dispatch selection.
- Key implementation: resolves DeepEP mode using `forward_batch.is_extend_in_batch`; removes the DP-attention assertion; Qwen3 MoE calls experts with the full `forward_batch`.
- Key diff excerpt:

```python
resolved_deepep_mode = self.deepep_mode.resolve(
    forward_batch.is_extend_in_batch
)
```

```python
state.hidden_states_mlp_output = self.mlp(hidden_states, state.forward_batch)
```

- Validation/risk: test mixed prefill/decode batches because `auto` mode no longer follows only `ForwardMode`.

### [#7723](https://github.com/sgl-project/sglang/pull/7723) - FlashInfer bool check for FusedMoE in Qwen MoE

- Motivation: Qwen MoE models failed to pass `enable_flashinfer_moe`; `FusedMoE` therefore used the default false path even when the server flag was set.
- Key implementation: passes FlashInfer MoE kwargs only when the global flag is enabled.
- Key diff excerpt:

```python
**(
    dict(
        enable_flashinfer_moe=True,
        enable_ep_moe=global_server_args_dict["enable_ep_moe"],
    )
    if global_server_args_dict["enable_flashinfer_moe"]
    else {}
),
```

- Validation/risk: check server args, constructor kwargs, and backend selection together; the bug is a wiring mismatch.

### [#7966](https://github.com/sgl-project/sglang/pull/7966) - Refactor `select_experts`

- Motivation: routing logic was duplicated and difficult to extend for gate-router fusion; MoE inputs were too broad and inconsistent across FusedMoE/EPMoE/DeepEPMoE.
- Key implementation: introduced `TopKOutput`; added a `TopK` custom op; changed FusedMoE/EPMoE forward signatures to consume `topk_output`; Qwen3 MoE owns `self.topk`.
- Key diff excerpt:

```python
class TopKOutput(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor
```

```python
topk_output = self.topk(hidden_states, router_logits)
final_hidden_states = self.experts(hidden_states, topk_output)
```

- Validation/risk: later PRs rely on the new top-k object for FlashInfer, ModelOpt, and DeepEP output refactors.

### [#8421](https://github.com/sgl-project/sglang/pull/8421) - Simplify DeepEP output

- Motivation: model files owned too much DeepEP dispatch/combine logic; MoE layers needed a common dispatch-output abstraction.
- Key implementation: introduced `DispatchOutputFormat` and DeepEP output classes; `DeepEPMoE.forward` now performs dispatch, expert compute, and combine; Qwen3 TBO uses `self.experts.deepep_dispatcher` and `self.experts.moe_impl`.
- Key diff excerpt:

```python
dispatch_output = self.dispatch(
    hidden_states, topk_idx, topk_weights, forward_batch
)
hidden_states = self.moe_impl(dispatch_output)
hidden_states = self.combine(
    hidden_states,
    dispatch_output.topk_idx,
    dispatch_output.topk_weights,
    forward_batch,
)
```

- Validation/risk: if a later Qwen3 MoE change tries to reimplement DeepEP dispatch in `qwen3_moe.py`, compare against this refactor first.

### [#8658](https://github.com/sgl-project/sglang/pull/8658) - Update MoE parallelism arguments

- Motivation: `--enable-ep-moe` and `--enable-deepep-moe` were too narrow once standard EP, DeepEP, and vendor A2A backends shared the same model paths.
- Key implementation: introduced `MoeA2ABackend`; deprecated flags mutate the new backend field; Qwen3 MoE checks `global_server_args_dict["moe_a2a_backend"].is_deepep()` and uses `get_moe_expert_parallel_world_size()`.
- Key diff excerpt:

```python
class MoeA2ABackend(Enum):
    STANDARD = ("standard", "none")
    DEEPEP = "deepep"
```

```python
if self.enable_deepep_moe:
    self.moe_a2a_backend = "deepep"
```

- Validation/risk: docs and launch examples after this PR should prefer `--moe-a2a-backend` over old booleans.

### [#8751](https://github.com/sgl-project/sglang/pull/8751) - Remove Qwen3 MoE load-weight overhead for Slime

- Motivation: Slime weight update overhead in Qwen3 MoE came partly from repeatedly traversing parameters and trying to load non-local expert weights.
- Key implementation: cached `params_dict`; skipped expert weights that do not map to the rank; lazily initialized `routed_experts_weights_of_layer`.
- Key diff excerpt:

```python
if not hasattr(self, "_cached_params_dict"):
    self._cached_params_dict = dict(self.named_parameters())
params_dict = self._cached_params_dict
```

```python
if name not in params_dict:
    continue
...
if is_expert_weight:
    continue
```

- Validation/risk: this changes update-weight hot paths; verify both initial load and repeated update calls under EP.

### [#9338](https://github.com/sgl-project/sglang/pull/9338) - Refactor TopK readability and extensibility

- Motivation: top-k fixes were hardcoded in DeepSeek-specific code; Qwen3 MoE needed extensible output-format selection for Triton, FlashInfer TRTLLM, and FP4/MXFP4 paths.
- Key implementation: added `TopKOutputFormat`; TopK chooses Triton, bypassed, or standard format; Qwen3 MoE passes `get_moe_impl_class(quant_config)`.
- Key diff excerpt:

```python
if self.topk_config.output_format is not None:
    output_format = self.topk_config.output_format
elif get_moe_runner_backend().is_triton_kernel():
    output_format = TopKOutputFormat.TRITON_KERNEL
elif should_use_flashinfer_trtllm_moe():
    output_format = TopKOutputFormat.BYPASSED
else:
    output_format = TopKOutputFormat.STANDARD
```

- Validation/risk: top-k output format must match the selected MoE runner backend; mismatches usually surface as shape or routing-logit errors.

## Pipeline Parallelism and Tied Embeddings

### [#6250](https://github.com/sgl-project/sglang/pull/6250) - Add pipeline parallelism for Qwen2/Qwen3

- Motivation: large Qwen3 dense/MoE models need PP in addition to TP/EP, with layers and final head split across ranks.
- Key implementation: added `get_pp_group`, `PPMissingLayer`, `PPProxyTensors`, and `get_layer_id`; first rank owns embeddings; last rank owns norm/logits; weight loading skips layers outside the local PP range.
- Key diff excerpt:

```python
self.layers, self.start_layer, self.end_layer = make_layers(
    config.num_hidden_layers,
    ...,
    pp_rank=self.pp_group.rank_in_group,
    pp_size=self.pp_group.world_size,
)
```

```python
if not self.pp_group.is_last_rank:
    return PPProxyTensors(
        {"hidden_states": hidden_states, "residual": residual}
    )
```

- Validation/risk: PP bugs are often load-time or LM-head bugs, so include a tiny Qwen3 dense checkpoint and a Qwen3 MoE checkpoint.

### [#6546](https://github.com/sgl-project/sglang/pull/6546) - Tied weights in Qwen PP

- Motivation: `tie_word_embeddings=True` failed under PP because the last rank owned `lm_head` but not `embed_tokens`.
- Key implementation: first PP rank sends embedding weights to the last rank; last rank copies them into `lm_head`; loader maps `lm_head.weight` to `model.embed_tokens.weight` for PP+tied.
- Key diff excerpt:

```python
if self.pp_group.world_size > 1 and config.tie_word_embeddings:
    if self.pp_group.is_first_rank:
        self.pp_group.send(
            self.model.embed_tokens.weight, dst=self.pp_group.last_rank
        )
    else:
        emb_token_weight = self.pp_group.recv(
            size=(config.vocab_size, config.hidden_size),
            dtype=next(self.model.parameters()).dtype,
            src=self.pp_group.first_rank,
        )
        self.lm_head.weight.copy_(emb_token_weight)
```

- Validation/risk: later PRs fix rank indexing and eventually remove this runtime send/recv path.

### [#15223](https://github.com/sgl-project/sglang/pull/15223) - Fix Qwen3 model load under PP

- Motivation: Qwen3-0.6B TP2 PP4 startup failed because tied embedding PP send/recv used incorrect global rank assumptions and receive shape.
- Key implementation: PP send targets `world_size - 1`; recv uses `self.lm_head.weight.shape` and `src=0`.
- Key diff excerpt:

```python
if self.pp_group.is_first_rank:
    self.pp_group.send(
        self.model.embed_tokens.weight, dst=self.pp_group.world_size - 1
    )
elif self.pp_group.is_last_rank:
    emb_token_weight = self.pp_group.recv(
        size=self.lm_head.weight.shape,
        dtype=next(self.model.parameters()).dtype,
        src=0,
    )
```

- Validation/risk: this fixes the old tied-weight mechanism but does not eliminate its fragility.

### [#15890](https://github.com/sgl-project/sglang/pull/15890) - Fix tied embedding weight logic under PP

- Motivation: Qwen3-4B PP=2 produced bad output because the last PP rank filtered out `model.embed_tokens.weight`; Qwen3-0.6B only worked when safetensors also contained `lm_head.weight`.
- Key implementation: removed runtime PP send/recv; when the loader sees `model.embed_tokens.weight` on the last PP rank and `tie_word_embeddings=True`, it loads that tensor directly into `lm_head.weight`.
- Key diff excerpt:

```python
if name == "model.embed_tokens.weight":
    if self.pp_group.is_last_rank and self.config.tie_word_embeddings:
        if "lm_head.weight" in params_dict:
            param = params_dict["lm_head.weight"]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
```

- Validation/risk: validate tied and untied checkpoints; do not assume `lm_head.weight` exists in the checkpoint.

### [#20127](https://github.com/sgl-project/sglang/pull/20127) - Open: handle tied embeddings for Qwen MoE and Qwen3Next

- Motivation: Qwen3 MoE, Qwen2 MoE, and Qwen3Next still create separate `ParallelLMHead` even when `tie_word_embeddings=True`; checkpoints without `lm_head.weight` leave a random output head.
- Key implementation in open diff: PP-aware LM-head creation ties directly to `model.embed_tokens` when world size is one; PP last rank creates a real head and loader copies `model.embed_tokens.weight`.
- Key diff excerpt:

```python
if self.pp_group.is_last_rank:
    if self.pp_group.world_size == 1 and config.tie_word_embeddings:
        self.lm_head = self.model.embed_tokens
    else:
        self.lm_head = ParallelLMHead(...)
else:
    self.lm_head = PPMissingLayer()
```

- Validation/risk: open PR; re-check current diff before relying on it. It is relevant to Qwen3 Core because it closes the MoE side of the dense PP fixes.

## DP Attention, TBO, Context Parallel, and Speculative Paths

### [#6598](https://github.com/sgl-project/sglang/pull/6598) - Qwen3 MoE two-batch overlap

- Motivation: Qwen3-235B serving needed two-batch overlap (TBO) to overlap communication and compute across DP attention and DeepEP normal mode.
- Key implementation: rewrote Qwen3 MoE attention/layer/MLP into schedulable `op_*` stages; used `MaybeTboDeepEPDispatcher`; added Qwen3-specific TBO operation strategy.
- Key diff excerpt:

```python
self.deepep_dispatcher = MaybeTboDeepEPDispatcher(...)
```

```python
elif layer_name == "Qwen3MoeDecoderLayer":
    return OperationsStrategy.concat(
        [
            _compute_moe_qwen3_layer_operations_strategy_tbo(
                layer, forward_mode
            )
            for layer in layers
        ]
    )
```

- Validation/risk: Qwen3-235B-A22B-FP8 TP8 DP8 DP attention DeepEP normal TBO reported GSM8K `0.945`, latency `210.649s`, output throughput `966.423 tok/s`.

### [#6652](https://github.com/sgl-project/sglang/pull/6652) - Fix Qwen3 TBO and DP LM-head

- Motivation: Qwen3 TBO parameter wiring needed correction and Qwen2/Qwen3 needed DP LM-head support.
- Key implementation: fixed zero-allocator TBO parameter; `ParallelLMHead` uses the attention TP group when `enable_dp_lm_head` is set.
- Key diff excerpt:

```python
self.lm_head = ParallelLMHead(
    config.vocab_size,
    config.hidden_size,
    quant_config=quant_config,
    prefix=add_prefix("lm_head", prefix),
    use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
)
```

- Validation/risk: DP attention plus logits processing must use the same TP group as attention, not the MoE TP group.

### [#7681](https://github.com/sgl-project/sglang/pull/7681) - Dense Qwen3 DP attention

- Motivation: dense Qwen3 needed DP attention parity with Qwen3 MoE; Qwen3-8B TP8 DP8 was the target validation shape.
- Key implementation: attention projections use attention TP rank/size; output projection sets `reduce_results=False`; embedding disables TP when DP attention is enabled; decoder layer uses `LayerCommunicator`.
- Key diff excerpt:

```python
self.qkv_proj = QKVParallelLinear(
    ...,
    tp_rank=attn_tp_rank,
    tp_size=attn_tp_size,
)
self.o_proj = RowParallelLinear(
    ...,
    tp_rank=attn_tp_rank,
    tp_size=attn_tp_size,
    reduce_results=False,
)
```

```python
hidden_states, residual = self.layer_communicator.prepare_attn(
    hidden_states, residual, forward_batch
)
```

- Validation/risk: reported Qwen3-8B TP8 DP8 accuracy `0.909`, throughput `3595 tok/s`; DP attention regressions should test dense and MoE separately.

### [#8280](https://github.com/sgl-project/sglang/pull/8280) - DP enhancement

- Motivation: DP attention needed better padding, memory allocation, and communication behavior across Qwen3 MoE and related models.
- Key implementation: added `DPPaddingMode`; lazily allocated gathered buffers; used max padded length for DP+EAGLE CUDA graphs; moved DP gather/scatter into `LayerCommunicator`; added tensor collectives.
- Key diff excerpt:

```python
class DPPaddingMode(IntEnum):
    MAX_LEN = auto()
    SUM_LEN = auto()
```

```python
if sum_len * 2 > max_len * get_attention_dp_size():
    return cls.MAX_LEN
else:
    return cls.SUM_LEN
```

- Validation/risk: performance depends on global token distribution; benchmark both balanced and skewed batches.

### [#9101](https://github.com/sgl-project/sglang/pull/9101) - Reduce-scatter for DP attention padding

- Motivation: Qwen2 MoE, Qwen3 MoE, and Llama4 needed reduce-scatter after MoE/MLP when DP attention uses max padding.
- Key implementation: `LayerCommunicator` decides whether reduce-scatter is profitable; Qwen3 MoE forwards `use_reduce_scatter`; normal MoE skips its all-reduce in the reduce-scatter path.
- Key diff excerpt:

```python
use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
    forward_batch
)
hidden_states = self.mlp(
    hidden_states, forward_batch, use_reduce_scatter
)
```

```python
def forward(..., use_reduce_scatter: bool = False) -> torch.Tensor:
    if not global_server_args_dict["moe_a2a_backend"].is_deepep():
        return self.forward_normal(hidden_states, use_reduce_scatter)
```

- Validation/risk: Qwen3-235B-A22B-Instruct-2507-FP8 TP8 EP8 DP8 reported random serving throughput `11367 -> 12692 tok/s`.

### [#12002](https://github.com/sgl-project/sglang/pull/12002) - EAGLE3 DP attention for Qwen3 MoE

- Motivation: large Qwen3 MoE deployments needed EAGLE3 with DP attention and EP, including target-side auxiliary hidden-state capture.
- Key implementation: `LayerCommunicator.prepare_attn_and_capture_last_layer_outputs` gathers captured residuals; Qwen3 MoE marks capture layers; EAGLE worker uses attention TP group context under DP attention.
- Key diff excerpt:

```python
def prepare_attn_and_capture_last_layer_outputs(...):
    hidden_states, residual = self.prepare_attn(...)
    if captured_last_layer_outputs is not None:
        gathered_last_layer_output = self._communicate_simple_fn(
            hidden_states=residual, ...
        )
        if gathered_last_layer_output is residual:
            gathered_last_layer_output = residual.clone()
        captured_last_layer_outputs.append(gathered_last_layer_output)
    return hidden_states, residual
```

- Validation/risk: 8xH100 Qwen3-235B EAGLE3 DP attention GSM8K `0.970`; 3-node PD decode `0.965`.

### [#18233](https://github.com/sgl-project/sglang/pull/18233) - Qwen3 MoE context parallel

- Motivation: long-context Qwen3 MoE prefill needed context parallelism across attention and MoE topology boundaries.
- Key implementation: FlashAttention backend allgathers/reranges KV cache for CP; attention splits `q` into previous/next chunks; Qwen3 MoE uses MoE tensor-parallel all-reduce in the correct group.
- Key diff excerpt:

```python
key_cache_full = cp_all_gather_rerange_kv_cache(
    k, self.attn_cp_size, forward_batch, torch.cuda.current_stream()
)
value_cache_full = cp_all_gather_rerange_kv_cache(
    v, self.attn_cp_size, forward_batch, torch.cuda.current_stream()
)
forward_batch.token_to_kv_pool.set_kv_buffer(
    layer, cache_loc, key_cache_full, value_cache_full, layer.k_scale, layer.v_scale
)
```

```python
q_prev, q_next = torch.chunk(
    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
    2,
    dim=0,
)
```

- Validation/risk: Qwen3-30B-FP8 TP4 moe-dp2 ep2 attn-cp2 reported GSM8K `0.785`, latency `43.704s`.

### [#21195](https://github.com/sgl-project/sglang/pull/21195) - Enable the Qwen3 test

- Motivation: a previously disabled Qwen3-30B CP test needed the model path fixed enough to run in CI.
- Key implementation: restored EP all-reduce for `ep_size > 1` before the TP all-reduce path; registered the 4-GPU H100 test.
- Key diff excerpt:

```python
if self.ep_size > 1 and not should_allreduce_fusion:
    final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)
```

```python
register_cuda_ci(est_time=300, suite="stage-c-test-4-gpu-h100")
```

- Validation/risk: test enablement PRs often encode the final invariant; keep this all-reduce ordering when refactoring Qwen3 MoE.

### [#22003](https://github.com/sgl-project/sglang/pull/22003) - Support `moe_dp_size = 1` with various attention CP sizes

- Motivation: earlier CP required `attention_cp_size == moe_dp_size`; production wants CP for attention while keeping MoE DP at one.
- Key implementation: maps `_MOE_DP` to `_ATTN_CP` when `attn_cp_size > moe_dp_size`; adds `ScatterMode.MOE_FULL`; gathers hidden states across MoE CP ranks and slices them back after MoE.
- Key diff excerpt:

```python
if attn_cp_size > moe_dp_size:
    _MOE_DP = _ATTN_CP
elif moe_dp_size == tensor_model_parallel_size:
    _MOE_DP = _TP
```

```python
class ScatterMode(Enum):
    SCATTERED = auto()
    TP_ATTN_FULL = auto()
    FULL = auto()
    MOE_FULL = auto()
```

```python
hidden_states = hidden_states.narrow(
    0, moe_cp_rank * max_tokens_per_rank, actual_local_tokens
).contiguous()
```

- Validation/risk: benchmark cited `--tp-size 4 --moe-dp-size 1 --ep-size 4 --attn-cp-size 2` latency `73.150s` versus old `171.566s`.

### [#22358](https://github.com/sgl-project/sglang/pull/22358) - DFLASH support for Qwen model backends

- Motivation: z-lab DFLASH collection needed explicit aux-hidden capture before the newer DFLASH spec was finalized.
- Key implementation: Qwen3 dense and MoE expose `set_dflash_layers_to_capture`; dense maps HF-style "after layer k" to SGLang "before layer k+1"; MoE marks layers for capture.
- Key diff excerpt:

```python
def set_dflash_layers_to_capture(self, layer_ids: List[int]):
    if not self.pp_group.is_last_rank:
        return
    if layer_ids is None:
        raise ValueError("DFLASH requires explicit layer_ids for aux hidden capture.")
    self.capture_aux_hidden_states = True
    self.model.layers_to_capture = [val + 1 for val in layer_ids]
```

- Validation/risk: explicit layer IDs are required; do not silently choose EAGLE3 default capture layers for DFLASH.

## Quantization and FlashInfer/TRTLLM MoE

### [#7912](https://github.com/sgl-project/sglang/pull/7912) - Qwen FP8/NVFP4 ModelOpt support

- Motivation: ModelOpt exports for Qwen dense/MoE should launch with one-line quantization flags on B100/H100.
- Key implementation: `ModelOptFp4Config.common_group_size` searches nested config dictionaries and enforces consistency; Qwen3 loader remaps KV scale names and skips unmatched scale tensors.
- Key diff excerpt:

```python
@staticmethod
def common_group_size(cfg: dict) -> int:
    sizes = set()
    ...
    if not sizes:
        raise ValueError("No group_size found in config.")
    if len(sizes) > 1:
        raise ValueError(f"Inconsistent group_size values: {sorted(sizes)}")
    return next(iter(sizes))
```

```python
if "scale" in name:
    name = maybe_remap_kv_scale_name(name, params_dict)
    if name is None:
        continue
```

- Validation/risk: PR reports Qwen3-1.7B and Qwen3-30B-A3B pass FP8/NVFP4 on B100, FP8 on H100.

### [#8036](https://github.com/sgl-project/sglang/pull/8036) - FlashInfer MoE blockscale FP8 backend

- Motivation: FlashInfer TRTLLM-style blockscale FP8 MoE was needed for lower Qwen3 MoE latency; PR reports up to 3x E2E improvement in target cases.
- Key implementation: added `--enable-flashinfer-trtllm-moe`; `FlashInferEPMoE` calls `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`; weight loader swaps w1/w3 because FlashInfer expects w31 layout.
- Key diff excerpt:

```python
return trtllm_fp8_block_scale_moe(
    routing_logits=router_logits.to(torch.float32),
    hidden_states=a_q,
    hidden_states_scale=a_sf_t,
    tile_tokens_dim=get_tile_tokens_dim(
        hidden_states.shape[0], self.top_k, self.num_experts
    ),
    routing_method_type=2,
)
```

- Validation/risk: check FlashInfer version and expert weight layout before turning this on by default.

### [#8450](https://github.com/sgl-project/sglang/pull/8450) - FlashInfer MoE blockscale FP8 for TP MoE

- Motivation: `#8036` covered EP; TP-only MoE needed the same FlashInfer TRTLLM backend.
- Key implementation: moved TRTLLM support into `FusedMoE` with `FlashInferFusedMoE`; added `should_use_flashinfer_trtllm_moe()` version gating; returns FlashInfer impl for both EP and TP when enabled.
- Key diff excerpt:

```python
@lru_cache(maxsize=1)
def should_use_flashinfer_trtllm_moe():
    return global_server_args_dict["enable_flashinfer_trtllm_moe"] and (
        not importlib.util.find_spec("flashinfer")
        or pkg_version.parse(__import__("flashinfer").__version__)
        >= pkg_version.parse("0.2.9rc1")
    )
```

```python
return FlashInferEPMoE if should_use_flashinfer_trtllm_moe() else EPMoE
return FlashInferFusedMoE if should_use_flashinfer_trtllm_moe() else FusedMoE
```

- Validation/risk: TP and EP use different communication assumptions; validate both with FP8 blockscale checkpoints.

### [#9973](https://github.com/sgl-project/sglang/pull/9973) - FlashInfer fused all-reduce for Qwen3 MoE

- Motivation: profiling found AllReduce at `13.26%` and FusedNormAdd at `6.45%`; fusing all-reduce with RMSNorm/residual add improved E2E throughput.
- Key implementation: communicator enables FlashInfer all-reduce fusion on SM90/SM100 for token counts <=4096; Qwen3 MoE MLP marks tensors that need post-layer all-reduce fusion and skips redundant all-reduce.
- Key diff excerpt:

```python
if (
    (_is_sm100_supported or _is_sm90_supported)
    and _is_flashinfer_available
    and hasattr(layernorm, "forward_with_allreduce_fusion")
    and global_server_args_dict["enable_flashinfer_allreduce_fusion"]
    and hidden_states.shape[0] <= 4096
):
    hidden_states, residual = layernorm.forward_with_allreduce_fusion(...)
```

```python
if should_allreduce_fusion:
    hidden_states._sglang_needs_allreduce_fusion = True
```

- Validation/risk: sample reported input throughput +2.2%; verify accuracy because the fusion changes residual/norm ordering boundaries.

### [#13489](https://github.com/sgl-project/sglang/pull/13489) - FlashInfer TRTLLM-GEN-MoE plus Qwen3

- Motivation: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 with `--moe-runner-backend flashinfer_trtllm --quantization fp8` should work and should become a good default on SM100.
- Key implementation: Qwen3 MoE passes `RoutingMethodType.Renormalize`; server args infer FP8 from `hf_config.quantization_config.quant_method`; on SM100 and no A2A, auto backend becomes `flashinfer_trtllm`.
- Key diff excerpt:

```python
routing_method_type=RoutingMethodType.Renormalize
```

```python
if (
    self.quantization == "fp8"
    and self.moe_a2a_backend == "none"
    and self.moe_runner_backend == "auto"
):
    self.moe_runner_backend = "flashinfer_trtllm"
```

- Validation/risk: PR reports GSM8K `0.942 -> 0.946` and latency `2.426s -> 1.832s` for a bench sample.

### [#14093](https://github.com/sgl-project/sglang/pull/14093) - Fused FP8 KV-cache write for TRTLLM MHA

- Motivation: TRTLLM FP8 KV path launched four tiny kernels for quant K/V and write K/V; fusing quantization plus paged cache write reduces kernel overhead.
- Key implementation: added Triton `_fused_fp8_set_kv_buffer_kernel`; TRTLLM backend writes FP8 KV cache in one path and sets `k = v = None` to skip the generic write.
- Key diff excerpt:

```python
@triton.jit
def _fused_fp8_set_kv_buffer_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_loc_ptr,
    k_scale,
    v_scale,
    use_provided_scale,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    ...
):
    token_id = tl.program_id(0)
```

```python
if use_fused_fp8_path:
    self._fused_fp8_set_kv_buffer(
        q=q, k=k, v=v, layer=layer, forward_batch=forward_batch
    )
    k = None
    v = None
```

- Validation/risk: Qwen3 MoE FP8 KV cache should be checked against the naive path for bit-level correctness.

### [#18189](https://github.com/sgl-project/sglang/pull/18189) - Fix Qwen3-235B NVFP4 launch

- Motivation: `nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4` failed because the ignore list left q/k/v in BF16 while Qwen3 MoE had no packed-module mapping for the fused `qkv_proj`.
- Key implementation: added `packed_modules_mapping` to `Qwen3MoeForCausalLM` so ModelOpt can map fused modules back to HF component names.
- Key diff excerpt:

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

- Validation/risk: PR reports Qwen3-235B-A22B-Instruct-2507-NVFP4 GSM8K Platinum accuracy `0.980`; also check 30B NVFP4 remains unchanged.

### [#9147](https://github.com/sgl-project/sglang/pull/9147) - Open: Qwen3-MoE W4AFP8

- Motivation: add static-calibration W4AFP8 block quantization for Qwen3 MoE.
- Key implementation in open diff: `W4AFp8Config` selects TP or EP MoE method; int4 weights and FP8 activation scales are packed/interleaved to the CUTLASS layout; calls `cutlass_w4a8_moe`.
- Key diff excerpt:

```python
elif isinstance(layer, FusedMoE):
    if global_server_args_dict["enable_ep_moe"]:
        return W4AFp8EPMoEMethod(self)
    else:
        return W4AFp8TPMoEMethod(self)
```

```python
return cutlass_w4a8_moe(
    start_expert_id=0,
    end_expert_id=self.num_experts - 1,
    total_num_experts=self.num_experts,
    a=x,
    w1_q=layer.w13_weight,
    w2_q=layer.w2_weight,
    topk_weights=topk_weights,
    topk_ids_=topk_ids,
)
```

- Validation/risk: open PR appears stale against current TopK/MoeA2ABackend APIs; rebase is required before using the design.

## QK-Norm, RoPE, KV-Store, and Kernel Fusion

### [#7740](https://github.com/sgl-project/sglang/pull/7740) - Two-stream norm for Qwen3

- Motivation: Qwen3 Q/K RMSNorm can overlap on separate CUDA streams; this targets small but measurable TPOT improvement on Qwen3-235B.
- Key implementation: plumbed `alt_stream` into Qwen2/Qwen3 dense/MoE layers; in CUDA graph capture, Q norm runs on current stream and K norm on alt stream.
- Key diff excerpt:

```python
if self.alt_stream is not None and get_is_capture_mode():
    current_stream = torch.cuda.current_stream()
    self.alt_stream.wait_stream(current_stream)
    q_by_head = q.reshape(-1, self.head_dim)
    q_by_head = self.q_norm(q_by_head)
    with torch.cuda.stream(self.alt_stream):
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
    current_stream.wait_stream(self.alt_stream)
```

- Validation/risk: reported Qwen3-235B-A22B-FP8 GSM8K `0.948`; overlap should be gated to CUDA capture mode.

### [#10749](https://github.com/sgl-project/sglang/pull/10749) - Fuse write-KV-buffer into RoPE for Qwen3 MoE

- Motivation: Qwen3 MoE decode can avoid a separate KV-cache write by fusing KV store into the RoPE path for BF16 KV cache.
- Key implementation: `enable_fused_set_kv_buffer` gates CUDA BF16 KV cache; `create_fused_set_kv_buffer_arg` passes value/cache metadata into RoPE; attention call skips saving KV cache when fused write was used.
- Key diff excerpt:

```python
def enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
    return _is_cuda and forward_batch.token_to_kv_pool.dtype == torch.bfloat16
```

```python
q, k = self.rotary_emb(
    positions,
    q,
    k,
    fused_set_kv_buffer_arg=(
        create_fused_set_kv_buffer_arg(
            value=v, layer=self.attn, forward_batch=forward_batch
        )
        if enable_fused_set_kv_buffer(forward_batch)
        else None
    ),
)
attn_output = self.attn(
    *inner_state,
    save_kv_cache=not enable_fused_set_kv_buffer(forward_batch),
)
```

- Validation/risk: Qwen3-30B-A3B random 4000/1 input throughput `91961 -> 94401 tok/s`.

### [#13998](https://github.com/sgl-project/sglang/pull/13998) - Fused QK-norm/RoPE for Qwen3 MoE

- Motivation: Qwen3 MoE has many decode layers and pays noticeable overhead for separate qk_norm plus RoPE kernels.
- Key implementation: imports CUDA `sgl_kernel.fused_qk_norm_rope`; computes YaRN parameters; gates on non-MRoPE and head dim 64/128/256; fused path handles BF16 qkv and falls back otherwise.
- Key diff excerpt:

```python
self.compatible_with_fused_qk_norm_rope = (
    not isinstance(self.rotary_emb, MRotaryEmbedding)
) and self.head_dim in (64, 128, 256)
self.use_fused_qk_norm_rope = (
    get_global_server_args().enable_fused_qk_norm_rope
    and self.compatible_with_fused_qk_norm_rope
)
```

```python
fused_qk_norm_rope(
    qkv,
    self.num_heads,
    self.num_kv_heads,
    self.num_kv_heads,
    self.head_dim,
    self.q_norm.variance_epsilon,
    self.q_norm.weight,
    self.k_norm.weight,
    theta,
    self.rotary_emb.is_neox_style,
    positions,
    factor,
    low,
    high,
    attention_factor,
)
```

- Validation/risk: Qwen3-235B-A22B H20 TTFT `990.23ms -> 935.69ms`; fused path must preserve YaRN behavior.

### [#15835](https://github.com/sgl-project/sglang/pull/15835) - JIT fused QK norm cleanup

- Motivation: newer FlashInfer QK norm was not available in sgl-kernel, small batches underused bandwidth, and each model had redundant `_apply_qk_norm`.
- Key implementation: added JIT `fused_inplace_qknorm`; registered custom op helper; introduced shared `apply_qk_norm`; replaced model-local QK-norm logic in Qwen3/Qwen3 MoE and others.
- Key diff excerpt:

```python
def apply_qk_norm(q, k, q_norm, k_norm, head_dim, alt_stream=None, allow_inplace=True):
    if (
        _is_cuda
        and allow_inplace
        and (q_eps == k_eps)
        and not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
        and can_use_fused_inplace_qknorm(head_dim)
    ):
        fused_inplace_qknorm(...)
        return q, k
```

- Validation/risk: PR claims Qwen3 E2E around 1-2%; deterministic inference disables this path.

### [#19059](https://github.com/sgl-project/sglang/pull/19059) - Add fused QK-norm/RoPE JIT kernel

- Motivation: migrate AOT fused qknorm-rope into lightweight JIT and fix NeoX active-mask undefined behavior.
- Key implementation: registered `fused_qk_norm_rope_out`; JIT kernel mutates packed QKV; `can_use_fused_qk_norm_rope` gates Qwen3 MoE init.
- Key diff excerpt:

```python
@register_custom_op(
    op_name="fused_qk_norm_rope_out",
    mutates_args=["qkv"],
)
def fused_qk_norm_rope_out(...):
    module = _jit_fused_qknorm_rope_module(head_dim, is_neox)
    module.fused_qk_norm_rope(...)
```

```python
self.use_fused_qk_norm_rope = (
    get_global_server_args().enable_fused_qk_norm_rope
    and self.compatible_with_fused_qk_norm_rope
    and _is_cuda
    and can_use_fused_qk_norm_rope(
        self.head_dim, self.rotary_emb.is_neox_style, torch.bfloat16
    )
)
```

- Validation/risk: JIT/AOT bit-identical in tested configs; keep head-dim and RoPE-style gates explicit.

### [#21654](https://github.com/sgl-project/sglang/pull/21654) - Optimize fused QK-norm/RoPE

- Motivation: JIT fused kernel still duplicated `__sincosf`, used `powf`, and carried array overhead.
- Key implementation: templated kernel on head dim, interleave, and YaRN; computes sin/cos once per pair; updates frequency recursively; only compiles YaRN code when needed.
- Key diff excerpt:

```cpp
template <int head_dim, bool interleave, bool yarn>
__global__ void fusedQKNormRopeKernel(...)
```

```cpp
for (int i = 0; i < numElemsPerThread; i += 2) {
    float e0 = elements[i];
    float e1 = elements[i + 1];
    float s, c;
    __sincosf(pos_id * f, &s, &c);
    elements[i] = (e0 * c - e1 * s) * attention_factor;
    elements[i + 1] = (e1 * c + e0 * s) * attention_factor;
    freq *= freq_ratio;
}
```

- Validation/risk: H100 kernel speedup 7-15% for interleave cases; preserve YaRN compile key in the JIT cache.

## LoRA, Layer-Wise Prefill, EAGLE3, Memory Heuristics, and Shared Plumbing

### [#7312](https://github.com/sgl-project/sglang/pull/7312) - Add `get_hidden_dim` for Qwen3 LoRA

- Motivation: issue `#7271` showed Qwen3 LoRA adapters could not infer correct hidden dimensions for packed projections.
- Key implementation: Qwen3 model returned per-module LoRA input/output dimensions for qkv, q, kv, o, gate_up, and down projections; tests compared Qwen3-4B LoRA against HF with ROUGE-L.
- Key diff excerpt:

```python
def get_hidden_dim(self, module_name: str) -> Tuple[int]:
    if module_name in ["q_proj", "qkv_proj"]:
        return (
            self.config.hidden_size,
            self.config.head_dim * self.config.num_attention_heads,
        )
    elif module_name in ["o_proj"]:
        return (
            self.config.head_dim * self.config.num_attention_heads,
            self.config.hidden_size,
        )
```

- Validation/risk: later `#8987` centralizes this default to avoid model-specific drift.

### [#8987](https://github.com/sgl-project/sglang/pull/8987) - Fix default LoRA hidden-dim logic

- Motivation: issue `#8939` found wrong default hidden-dim logic and duplicated model overrides.
- Key implementation: centralized LoRA hidden-dim logic in `lora/utils.py`; `qkv_proj` is LoRA-A only, `q_proj`/`kv_proj` are LoRA-B only; removed Qwen3-specific override.
- Key diff excerpt:

```python
if module_name == "qkv_proj":
    return (config.hidden_size, None)
elif module_name == "kv_proj":
    return (None, head_dim * config.num_key_value_heads)
elif module_name == "q_proj":
    return (None, head_dim * config.num_attention_heads)
elif module_name == "o_proj":
    return (head_dim * config.num_attention_heads, config.hidden_size)
```

- Validation/risk: Qwen3 LoRA tests should run after shared LoRA helper changes, even when `qwen3.py` itself is untouched.

### [#7634](https://github.com/sgl-project/sglang/pull/7634) - Layer-wise prefill

- Motivation: PD multiplexing needed to run intervals of decoder layers and store intermediate states in `ForwardBatch`.
- Key implementation: added `ForwardMode.SPLIT_PREFILL`; `ModelRunner.forward_split_prefill()` dispatches layer intervals; Qwen3/Qwen3 MoE implement `forward_split_prefill` with correct embedding, layer loop, final norm/logits, and expert-distribution recording.
- Key diff excerpt:

```python
class ForwardMode(IntEnum):
    ...
    SPLIT_PREFILL = auto()
```

```python
ret = self.model.forward_split_prefill(
    forward_batch.input_ids,
    forward_batch.positions,
    forward_batch,
    (forward_batch.split_index, next_split_index),
)
forward_batch.split_index = next_split_index
```

- Validation/risk: Qwen3 MoE split prefill must preserve residual and expert recorder context across partial layer intervals.

### [#7745](https://github.com/sgl-project/sglang/pull/7745) - EAGLE3 for Qwen

- Motivation: Qwen and Qwen3 EAGLE3 draft models needed auxiliary hidden-state capture for target/draft agreement.
- Key implementation: Qwen2/Qwen3 model loops capture aux hidden states before configured layers; top-level forwards pass `aux_hidden_states` into `LogitsProcessor`; default layers use `[2, num_layers // 2, num_layers - 3]` or user-specified IDs plus one.
- Key diff excerpt:

```python
if i in self.layers_to_capture:
    aux_hidden_states.append(
        hidden_states + residual if residual is not None else hidden_states
    )
```

```python
return self.logits_processor(
    input_ids,
    hidden_states,
    self.lm_head,
    forward_batch,
    aux_hidden_states,
)
```

- Validation/risk: EAGLE3 capture layers differ from DFLASH explicit layer IDs; do not reuse the default blindly.

### [#10975](https://github.com/sgl-project/sglang/pull/10975) - General heuristics for `--mem-fraction-static`

- Motivation: default chunked prefill, CUDA graph, and memory-fraction settings needed GPU-memory-aware heuristics instead of hardcoded scattered defaults.
- Key implementation: computes GPU memory once, chooses `chunked_prefill_size` and `cuda_graph_max_bs` by buckets, and subtracts reserved memory for chunked prefill, CUDA graph, DP attention, and speculative algorithms.
- Key diff excerpt:

```python
reserved_mem += max(self.chunked_prefill_size, 2048) * 1.5
reserved_mem += self.cuda_graph_max_bs * 2
if self.enable_dp_attention:
    reserved_mem += self.cuda_graph_max_bs * self.dp_size * 3
self.mem_fraction_static = round((gpu_mem - reserved_mem) / gpu_mem, 3)
```

- Validation/risk: this PR only lightly touches Qwen3 imports, but Qwen3-235B capacity planning depends on the server-side heuristic.

### [#10911](https://github.com/sgl-project/sglang/pull/10911) - Qwen3-Omni thinker-only plumbing

- Motivation: Qwen3-Omni thinker-only support needed to reuse Qwen3 MoE language-model code with a custom multimodal layer and MRoPE indexing.
- Key implementation: added Qwen3 Omni configs and thinker model; `MRotaryEmbedding.get_rope_index` dispatches `qwen3_omni_moe`; `Qwen3MoeModel` accepts `decoder_layer_type`.
- Key diff excerpt:

```python
if model_type == "qwen3_omni_moe":
    return MRotaryEmbedding.get_rope_index_qwen3_omni(...)
```

```python
def __init__(
    self,
    config: Qwen3MoeConfig,
    quant_config: Optional[QuantizationConfig] = None,
    prefix: str = "",
    decoder_layer_type=Qwen3MoeDecoderLayer,
) -> None:
```

- Validation/risk: mostly belongs to Qwen VLM/Omni docs, but the `decoder_layer_type` extensibility is Qwen3 Core shared infrastructure.

## Ascend NPU, XPU, MLX, and Platform Work

### [#10574](https://github.com/sgl-project/sglang/pull/10574) - Optimize Qwen3 on Ascend

- Motivation: Ascend Qwen3 serving needed NPU-native memory format and communication prefetch improvements.
- Key implementation: enables internal NPU format; casts W8A8 weights to format 29; adds CMO stream prefetch helpers; Qwen3 decoder passes MLP weights as cache on NPU and waits after MLP.
- Key diff excerpt:

```python
layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)
```

```python
cache=(
    [self.mlp.gate_up_proj.weight, self.mlp.down_proj.weight]
    if _is_npu
    else None
)
```

- Validation/risk: NPU optimizations should remain behind `_is_npu`; paged attention attempts were removed until packages were ready.

### [#12078](https://github.com/sgl-project/sglang/pull/12078) - Ascend Qwen optimization

- Motivation: follow-up to the Ascend roadmap; fixed W8A8 memory duplication, CMO deadlock, EPLB static-index, fused split-qkv-rmsnorm-rope, l1-norm top-k, NPU graph, and Ascend fused EP.
- Key implementation: added `ascend_fuseep`; introduced `NpuFuseEPMoE`; Qwen3/Qwen3 MoE use `split_qkv_rmsnorm_rope`; top-k uses `sgl_kernel_npu.norm.l1_norm`.
- Key diff excerpt:

```python
class MoeA2ABackend(Enum):
    ASCEND_FUSEEP = "ascend_fuseep"
```

```python
if get_moe_a2a_backend().is_ascend_fuseep():
    return NpuFuseEPMoE
```

```python
q, k, v = split_qkv_rmsnorm_rope(
    qkv,
    self.rotary_emb.position_sin,
    self.rotary_emb.position_cos,
    self.q_norm.weight,
    self.k_norm.weight,
    self.q_size,
    self.kv_size,
    self.head_dim,
    self.q_norm.variance_epsilon,
)
```

- Validation/risk: PR reports Qwen3-235B about +10% with fuseep; validate graph mode, bs1/bs2, and EPLB separately.

### [#15203](https://github.com/sgl-project/sglang/pull/15203) - NPU GPTQ quantization

- Motivation: issue `#15202` and NPU roadmap needed GPTQ for Qwen3 on Ascend, including GPTQv2 zero-point behavior.
- Key implementation: added `GPTQLinearAscendMethod`; NPU linear layers use `npu_weight_quant_batchmatmul`; GPTQv1 increments zeros while v2 does not; Qwen3 CMO cache checks that projections have `.weight`.
- Key diff excerpt:

```python
if _is_npu:
    if isinstance(layer, LinearBase):
        return GPTQLinearAscendMethod(self)
    elif isinstance(layer, FusedMoE):
        raise NotImplementedError("GPTQ Method does not support MoE yet.")
```

```python
out = torch_npu.npu_weight_quant_batchmatmul(
    reshaped_x,
    qweight,
    antiquant_scale=scales,
    antiquant_offset=qzeros,
    antiquant_group_size=self.quant_config.group_size,
    bias=bias,
)
```

- Validation/risk: PR reports Qwen3-1.7B fp16 GSM8K `0.710`, GPTQ Int8 `0.690`, Int4 `0.190`; note MoE GPTQ remains unsupported.

### [#15390](https://github.com/sgl-project/sglang/pull/15390) - NPU Qwen3 PP bugfix

- Motivation: Qwen3 PP on NPU generated RoPE sin/cos only on layer 0; under PP, the local first layer may be `token_to_kv_pool.start_layer`.
- Key implementation: `forward_prepare_npu` accepts `forward_batch` and checks `self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer`.
- Key diff excerpt:

```python
def forward_prepare_npu(self, positions, hidden_states, forward_batch):
    qkv, _ = self.qkv_proj(hidden_states)
    if self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer:
        self.rotary_emb.get_cos_sin_with_position(positions)
```

- Validation/risk: validate PP-size > 1 on NPU with dense Qwen3.

### [#16115](https://github.com/sgl-project/sglang/pull/16115) - Fix Qwen3 NPU with DP LM-head

- Motivation: `--enable-dp-lm-head` errored on Qwen3 NPU because fused split-qkv-rmsnorm-rope arguments and rotary dtype fallback were misaligned.
- Key implementation: NPU rotary falls back native when BF16 query meets float cos/sin cache; Qwen3 NPU `split_qkv_rmsnorm_rope` calls named eps/weight/bias args; LM-head uses attention TP group when DP LM-head is enabled.
- Key diff excerpt:

```python
if query.dtype == torch.bfloat16 and self.cos_sin_cache.dtype == torch.float:
    return self.forward_native(positions, query, key, offsets)
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
    q_weight=self.q_norm.weight,
    k_weight=self.k_norm.weight,
)
```

- Validation/risk: test NPU DP LM-head with both dense and MoE paths because LM-head TP group choice is shared.

### [#19532](https://github.com/sgl-project/sglang/pull/19532) - NPU speculative inference bugfix

- Motivation: previous NPU logic avoided fused split-qkv-rmsnorm-rope during prefill using `forward_mode.is_extend()`, but EAGLE3 target verification can make decode appear as extend.
- Key implementation: changed the condition to `is_extend_or_draft_extend_or_mixed()` in dense and MoE Qwen3 NPU prepare paths.
- Key diff excerpt:

```python
if (
    not _is_npu
    or forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
):
    q, k, v = self.forward_prepare_native(...)
```

- Validation/risk: speculative decoding validation must include target verify and draft extend/mixed modes.

### [#20474](https://github.com/sgl-project/sglang/pull/20474) - Open: Intel XPU Qwen3 support

- Motivation: enable Qwen3 on Intel XPU, including layernorm gated kernels and MRoPE path.
- Key implementation in open diff: `layernorm_gated._get_sm_count` returns XPU EU count; `MRotaryEmbedding.forward_xpu` delegates to Triton; tests launch Qwen/Qwen3-0.6B with `--device xpu --attention-backend intel_xpu`.
- Key diff excerpt:

```python
if _is_xpu:
    return torch.xpu.get_device_properties(device).gpu_eu_count
```

```python
def forward_xpu(self, positions, query, key, fused_set_kv_buffer_arg=None):
    return self.forward_triton(positions, query, key)
```

- Validation/risk: open PR; re-check its latest state before documenting XPU as supported.

### [#20520](https://github.com/sgl-project/sglang/pull/20520) - Open: NPU TP communication compression for Qwen3

- Motivation: INT8 TP communication compression on Qwen3 NPU prefill targets around 5% prefill speedup without average accuracy degradation.
- Key implementation in open diff: adds `--enable-quant-communications`; NPU communicator dynamically quantizes tensors, all-gathers int8 plus scale, dequantizes, and reduces; Qwen3 MLP receives `forward_batch` so row-parallel layers can inspect mode.
- Key diff excerpt:

```python
def tensor_model_parallel_quant_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().quant_all_reduce(input_)
```

```python
x_q, scale = npu_dynamic_quant(x, dst_type=torch.int8)
dist.all_gather_into_tensor(output_tensor, x_q, group=self.group)
dist.all_gather_into_tensor(output_scale, scale, group=self.group)
output_tensor = output_tensor.to(x.dtype) * output_scale.unsqueeze(-1).to(x.dtype)
```

- Validation/risk: open PR; only enable in non-decode/non-idle modes and NPU TP size > 1.

### [#21770](https://github.com/sgl-project/sglang/pull/21770) - Open: Apple MLX Qwen3 tests

- Motivation: add initial correctness and GSM8K coverage for Qwen3 on Apple Silicon MLX.
- Key implementation in open diff: launches server with `SGLANG_USE_MLX=1`; uses `enable_thinking=False`; adds lightweight GSM8K thresholds.
- Key diff excerpt:

```python
env = os.environ.copy()
env["SGLANG_USE_MLX"] = "1"
cls.process = popen_launch_server(cls.model, cls.base_url, env=env)
```

```python
"chat_template_kwargs": {"enable_thinking": False}
```

- Validation/risk: open PR is test coverage, not full feature support; keep wording precise.

## Parser, Sliding Window, and Alignment Radar

### [#21412](https://github.com/sgl-project/sglang/pull/21412) - Open: dense Qwen3 old-style RoPE compatibility

- Motivation: dense Qwen3 still had a counterpart to `#20931`; old-style checkpoints with top-level `rope_theta`/`rope_scaling` but no `rope_parameters` can raise `KeyError`.
- Key implementation in open diff: imports `get_rope_config(config)` and replaces direct access to `config.rope_parameters["rope_theta"]`.
- Key diff excerpt:

```python
from sglang.srt.utils.hf_transformers_utils import get_rope_config

rope_theta, rope_scaling = get_rope_config(config)
```

- Validation/risk: open PR; compare with `#22739` before deciding whether the issue is fully fixed on main.

### [#22529](https://github.com/sgl-project/sglang/pull/22529) - Open: sliding window attention for Qwen3

- Motivation: new Qwen3-architecture models can use alternating sliding/full attention via HF `layer_types`; SGLang needed equivalent per-layer window support.
- Key implementation in open diff: converts HF inclusive `sliding_window` to SGLang exclusive size; determines per-layer sliding status from `layer_types` or `max_window_layers`; passes `sliding_window_size` to `RadixAttention`.
- Key diff excerpt:

```python
def get_attention_sliding_window_size(config):
    if getattr(config, "sliding_window", None) is not None:
        return config.sliding_window - 1
    else:
        return None
```

```python
if layer_types is not None and layer_id < len(layer_types):
    is_sliding = layer_types[layer_id] == "sliding_attention"
```

- Validation/risk: open PR; test alternating patterns and legacy `max_window_layers`.

### [#22837](https://github.com/sgl-project/sglang/pull/22837) - Open: Qwen3 reasoning detector and tool calls

- Motivation: `Qwen3Detector` did not pass `tool_start_token` to the base detector; if `<tool_call>` arrived before `</think>`, the parser swallowed the tool call into `reasoning_content`.
- Key implementation in open diff: passes `tool_start_token="<tool_call>"` and adds streaming/non-streaming regression tests that force parser state out of reasoning when a tool call starts.
- Key diff excerpt:

```python
super().__init__(
    "<think>",
    "</think>",
    force_reasoning=force_reasoning,
    stream_reasoning=stream_reasoning,
    tool_start_token="<tool_call>",
    ...
)
```

```python
result = self.detector.parse_streaming_increment(
    "<tool_call>\n<function=foo>\n</function>\n</tool_call>"
)
self.assertEqual(
    result.normal_text,
    "<tool_call>\n<function=foo>\n</function>\n</tool_call>",
)
self.assertFalse(self.detector._in_reasoning)
```

- Validation/risk: parser tests must include streaming chunks where reasoning and tool-call tokens interleave.

### [#23397](https://github.com/sgl-project/sglang/pull/23397) - Open: dense deterministic math

- Motivation: alignment/on-policy training wanted dense Qwen3 rollout math numerically compatible with Megatron scoring, targeting near-zero rollout/training logprob diff.
- Key implementation in open diff: adds on-policy helpers to disable reduce-scatter/all-reduce fusion, force BF16 dense math/LM-head, and use TP-invariant row-linear/tree all-reduce; Qwen3 q/k norm uses FP32 weights in this mode.
- Key diff excerpt:

```python
norm_kwargs = get_on_policy_rms_norm_kwargs(
    weight_dtype=torch.float32,
)
self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
```

```python
if should_force_bfloat16_dense_tensor_math():
    hidden_states = hidden_states.to(torch.bfloat16)
```

- Validation/risk: open alignment PR; do not mix deterministic math mode with performance-default fusion assumptions.

### [#23434](https://github.com/sgl-project/sglang/pull/23434) - Open: Qwen3 pooled output embeddings

- Motivation: Qwen3 sequence-classification/reward variants lacked `get_input_embeddings`, breaking score API embedding override for Qwen3-Reranker-style models.
- Key implementation in open diff: forwards `get_input_embeddings()` from `Qwen3ForPooledOutput` to the wrapped model.
- Key diff excerpt:

```python
def get_input_embeddings(self) -> nn.Embedding:
    return self.model.get_input_embeddings()
```

- Validation/risk: open PR; validation should use score/reranker API rather than only causal generation.

## Qwen3.5/Next Shared Quantization Radar

### [#22674](https://github.com/sgl-project/sglang/pull/22674) - Open: NPU Qwen3.5-MoE and Qwen3-Next quantization

- Motivation: Qwen3.5 and Qwen3Next GDN linear attention pack `in_proj_qkv+in_proj_z` and `in_proj_b+in_proj_a`; default loader mappings missed these fused names on NPU quantized checkpoints.
- Key implementation in open diff: adds GDN packed entries to default `packed_modules_mapping` in the loader.
- Key diff excerpt:

```python
"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
"in_proj_ba": ["in_proj_b", "in_proj_a"],
```

- Validation/risk: this is not a Qwen3 Core runtime PR, but it belongs in the radar because it extends the shared Qwen quant loader assumptions.

## SGLang Low-Latency Docs and CI Radar

### [#22429](https://github.com/sgl-project/sglang/pull/22429) - Qwen3-32B and Qwen3-8B Ascend low-latency docs

- Motivation: document tested A3/A2 low-latency serving recipes for Qwen3 dense models.
- Key implementation: adds model tables and launch commands using Ascend backend, ModelSlim quantization, EAGLE3 draft model, BF16 dtype, and NPU device flags.
- Key diff excerpt:

```bash
--attention-backend ascend \
--device npu \
--quantization modelslim \
--speculative-algorithm EAGLE3 \
--dtype bfloat16
```

- Validation/risk: docs-only PR; treat table latency numbers as hardware/config-specific, not universal defaults.

### [#22446](https://github.com/sgl-project/sglang/pull/22446) - Qwen3-30B-A3B low-latency docs

- Motivation: add low-latency serving recipes for Qwen3-30B-A3B on Ascend.
- Key implementation: documents `--tp-size 2`, memory fraction settings, Ascend backend, ModelSlim quantization, and EAGLE3 paths.
- Key diff excerpt:

```bash
--tp-size 2 \
--mem-fraction-static 0.6 \
--attention-backend ascend \
--speculative-algorithm EAGLE3
```

- Validation/risk: docs-only; use it as a reproduction starting point and re-benchmark on the target machine.

### [#22687](https://github.com/sgl-project/sglang/pull/22687) - Qwen3-8B/32B docs bugfix

- Motivation: clean up incorrect low-latency doc commands.
- Key implementation: removed stale `HCCL_BUFFSIZE=400` and duplicate `--speculative-draft-model-quantization unquant`.
- Key diff excerpt:

```diff
-export HCCL_BUFFSIZE=400
...
---speculative-algorithm EAGLE3 --speculative-draft-model-path xxx --speculative-draft-model-quantization unquant
+--speculative-algorithm EAGLE3 --speculative-draft-model-path xxx
```

- Validation/risk: keep docs commands minimal; duplicate flags can hide real launch failures.

### [#22450](https://github.com/sgl-project/sglang/pull/22450) - Open: Qwen3-14B Ascend low-latency docs

- Motivation: add Qwen3-14B low-latency recipes for A3.
- Key implementation in open diff: documents ModelSlim quantization, Ascend sampling backend, EAGLE3, and schedule conservativeness settings.
- Key diff excerpt:

```bash
--quantization modelslim \
--sampling-backend ascend \
--speculative-algorithm EAGLE3 \
--schedule-conservativeness 0.01
```

- Validation/risk: open docs PR; verify latest command before copying into a runbook.

### [#23372](https://github.com/sgl-project/sglang/pull/23372) - Open: NPU speculative decoding CI

- Motivation: validate NPU speculative decoding and expert parallelism on A2/A3, including EAGLE3/NEXTN, draft attention backend, token map, and `ascend_fuseep`.
- Key implementation in open diff: adds Qwen3-32B W8A8 + EAGLE3 PD test with `--speculative-attention-mode decode`; registers nightly 8-NPU A3 suite.
- Key diff excerpt:

```python
"--speculative-algorithm", "EAGLE3",
"--speculative-draft-model-path", QWEN3_32B_EAGLE3_WEIGHTS_PATH,
"--speculative-attention-mode", "decode",
"--tp-size", "4",
```

```python
register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)
```

- Validation/risk: open CI PR; useful for coverage planning even before merge.

## sgl-cookbook Evidence

### [sgl-cookbook #74](https://github.com/sgl-project/sgl-cookbook/pull/74) - AMD Qwen3 support and tool-calling doc fixes

- Motivation: cookbook examples needed to cover Qwen3 on AMD and fix Qwen tool-calling instructions.
- Key implementation: updated cookbook commands/docs rather than SGLang runtime files.
- Key excerpt: cookbook-level launch and parser docs around Qwen3/AMD were refreshed; use the PR diff for exact command strings before copying.
- Validation/risk: docs evidence only. Always pair cookbook commands with SGLang runtime PRs and a current `sglang --version`.

### [sgl-cookbook #245](https://github.com/sgl-project/sgl-cookbook/pull/245) - Qwen cookbook refresh

- Motivation: refresh Qwen cookbook content after Qwen3/Qwen3.5/Qwen3-Next runtime changes.
- Key implementation: updates cookbook pages, examples, and links for Qwen-family usage.
- Key excerpt: command/documentation changes live in cookbook markdown rather than SGLang source.
- Validation/risk: cookbook PRs provide reproduction context but should not be cited as runtime support by themselves.

## Practical Completeness Checklist

Before declaring Qwen3 Core documentation complete for a new cycle:

1. Search `qwen3.py`, `qwen3_moe.py`, `qwen25_detector.py`, Qwen tests, NPU tests, and Qwen docs with `git log --follow --oneline -- <path>`.
2. Search merged PRs by `Qwen3`, `qwen3_moe`, `Qwen3MoeForCausalLM`, `Qwen3ForCausalLM`, `qwen25_detector`, and `qwen3 rope`.
3. Search open PRs with the same terms and mark them as radar, not current-main support.
4. For every PR, read the diff and update motivation, key implementation, key code excerpt, and validation/risk.
5. If a PR is docs-only, quote the exact launch/config line that changed and explain the serving implication.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG Qwen3 Core PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

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

## Diff Cards

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


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
