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
