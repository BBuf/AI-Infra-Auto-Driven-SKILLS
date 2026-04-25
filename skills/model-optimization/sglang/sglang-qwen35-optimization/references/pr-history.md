# Qwen3.5 PR Diff Dossier

本档案按“读过 PR diff 后再手写”的标准维护。每个条目都记录实际打开过的 diff/source、PR 动机、关键实现思路、核心代码片段和验证含义；不要再把 PR 只写成一句话清单。

Evidence baseline:

- SGLang main snapshot used in the earlier sweep: `b3e6cf60a` on 2026-04-22.
- sgl-cookbook main snapshot used in the earlier sweep: `816bad5` on 2026-04-21.
- Primary source surface: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, Qwen3.5 config, Mamba/KV cache pools, speculative decode, NIXL disaggregation, registered tests, docs and cookbook snippets.

## Runtime Surfaces

- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/models/qwen3_5_mtp.py`
- `python/sglang/srt/configs/qwen3_5.py`
- `python/sglang/srt/models/qwen2_moe.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/disaggregation/nixl/conn.py`
- `python/sglang/jit_kernel/triton/gdn_fused_proj.py`
- `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`
- `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`
- `docs/basic_usage/qwen3_5.md`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`
- `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`
- `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`
- `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`
- `test/registered/8-gpu-models/test_qwen35.py`
- `test/registered/gb300/test_qwen35_fp8.py`
- `test/registered/gb300/test_qwen35_nvfp4.py`

## Diff-Reviewed PR Cards

### PR #18489 - Initial Qwen3.5 dense/MoE/VL support

- Status: merged 2026-02-09, merge commit `27c447653d9cf0f63aea1c190b931be4875cbf86`.
- Diff reviewed: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`, `qwen_vl.py`, `server_args.py`, `eagle_worker.py`, model runner/config registration.
- Motivation: SGLang needed day-one support for the new Qwen3.5 family, including dense and MoE text models plus multimodal conditional-generation classes. The PR body explicitly references the upstream HF implementation and states the new `Qwen3_5MoeForConditionalGeneration` and `Qwen3_5ForConditionalGeneration` classes.
- Key implementation: adds a dedicated Qwen3.5 model file, config registration, Qwen3.5 MTP wrapper, multimodal processor hooks, and server/speculative decode wiring. The initial model brought in hybrid Gated Delta Net linear attention, full attention layers, MoE routing, deepstack multimodal embeddings, packed weight loading and Qwen3 parser deployment flags.
- Key code excerpt:

```python
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    ...

class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    ...

EntryClass = [
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
]
```

- Validation implication: this PR is the baseline for every later Qwen3.5 optimization. Any later change to GDN projection, MTP, PP, VLM or quantization must be checked against the class names and loader mappings introduced here.

### PR #18538 - Refactor Qwen3.5 MTP body

- Status: merged; diff reviewed in `python/sglang/srt/models/qwen3_5_mtp.py`.
- Motivation: the first MTP implementation carried a separate `Qwen3_5MultiTokenPredictor` style body, which duplicated model code and made checkpoint loading brittle.
- Key implementation: the PR replaces the custom predictor body with a nested `Qwen3_5ForCausalLM`, adds the MTP fusion projection `fc`, and uses two `GemmaRMSNorm` pre-fc norms before concatenating token embeddings and target hidden states.
- Key code excerpt:

```python
hidden_states = self.enorm(input_embeds)
target_hidden_states = self.hnorm(target_hidden_states)
hidden_states = torch.cat([hidden_states, target_hidden_states], dim=-1)
hidden_states, _ = self.fc(hidden_states)
return self.model(..., input_embeds=hidden_states, ...)
```

- Validation implication: MTP weight mapping must translate `model.fc`, `model.pre_fc` and nested model names consistently. This is why later PRs around quant prefixes and spec-v2 all touch `qwen3_5_mtp.py`.

### PR #18544 - Qwen3.5 follow-up for NPU, ModelSlim and EPLB

- Status: merged; diff reviewed in `hybrid_linear_attn_backend.py`, `modelslim.py`, `qwen3_5.py`.
- Motivation: the fresh model path still had CUDA-centric assumptions and prefix mismatches that blocked Ascend/NPU and ModelSlim quantized checkpoints.
- Key implementation: skips CUDA JIT/Triton assertions on NPU, normalizes ModelSlim prefixes such as `language_model.`, fixes Qwen3.5 MLP prefix handling for `.linear_attn`, and exposes EPLB expert-location config.
- Key code excerpt:

```python
if not is_cpu() and not is_npu():
    from sglang.srt.layers.attention.fla import ...
```

```python
return ModelConfigForExpertLocation(
    num_layers=config.num_hidden_layers,
    num_logical_experts=config.num_experts,
)
```

- Validation implication: NPU and ModelSlim support are not side notes. They constrain how later fused projection and quantization mappings may rename Qwen3.5 modules.

### PR #18926 - Block-wise FP8 quantization and prefix alignment

- Status: merged 2026-02-18, merge commit `fa5698d7916497288af8fe5a5b57bc4ee7e6fb37`.
- Diff reviewed: `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_vl.py`.
- Motivation: Qwen3.5 FP8 checkpoints need block-wise scale loading for merged column layers; at the same time, the MTP quant prefix was still `model` while the checkpoint used an MTP-specific hierarchy.
- Key implementation: adds `_load_merged_block_scale()` for `MergedColumnParallelLinear`, dispatches `BlockQuantScaleParameter` in `weight_loader_v2`, changes Qwen3.5 MTP prefix from `model` to `mtp`, and limits the old `model.` prefix stripping hack to Mistral-3.
- Key code excerpt:

```python
elif isinstance(param, BlockQuantScaleParameter):
    self._load_merged_block_scale(param, loaded_weight)
    return
```

```python
self.model = Qwen3_5ForCausalLM(
    config,
    quant_config,
    prefix=add_prefix("mtp", prefix),
)
```

- Validation implication: any Qwen3.5 FP8/NVFP4/MTP load failure should first inspect block scale slicing and `mtp.` prefix mapping before blaming kernels.

### PR #18937 - NVFP4 checkpoint support

- Status: merged; diff reviewed in `qwen3_5.py`, `qwen3_5_mtp.py` and RoPE/loader related hunks.
- Motivation: NVIDIA ModelOpt FP4/NVFP4 checkpoints cannot quantize every Qwen3.5 module uniformly. Linear attention, full attention and MTP layers need explicit quant guards.
- Key implementation: disables `modelopt_fp4` quant config in the Qwen3.5 linear-attention/full-attention/MTP paths, tightens expert name checks, and improves unknown RoPE scaling errors to include the actual config dict.
- Key code excerpt:

```python
linear_attn_quant_config = (
    None if quant_config and quant_config.get_name() == "modelopt_fp4"
    else quant_config
)
```

```python
if quant_config and quant_config.get_name() == "modelopt_fp4":
    quant_config = None
```

- Validation implication: NVFP4 Qwen3.5 accuracy depends on intentionally leaving some hybrid modules unquantized; later shared-expert and GDN fusion must preserve those exclusions.

### PR #19070 - Dense Qwen3.5 TP>1 precision fix

- Status: merged; diff reviewed in Qwen3.5 dense/MoE MLP call path.
- Motivation: dense Qwen3.5 with tensor parallelism greater than one had a precision regression from applying the MoE-style all-reduce fusion path too broadly.
- Key implementation: separates MoE and dense MLP invocation, passes `should_allreduce_fusion` only where it is valid, marks dense hidden states for deferred communicator postprocessing, and avoids premature all-reduce.
- Key code excerpt:

```python
hidden_states = self.mlp(
    hidden_states,
    should_allreduce_fusion=should_allreduce_fusion,
)
hidden_states._sglang_needs_allreduce_fusion = True
```

- Validation implication: dense 27B/4B lanes should not inherit every MoE communication optimization automatically.

### PR #19220 - PCG fix for Qwen3.5

- Status: merged; diff reviewed in `qwen3_next.py`, `qwen3_5.py`, `fp8_utils.py`.
- Motivation: the PCG path added a custom `gdn_with_output` split wrapper that conflicted with Qwen3.5 GDN execution and compile/fake registration expectations.
- Key implementation: removes the custom GDN PCG wrapper, uses the regular attention call directly, adds fake registration for `sgl_kernel::fp8_blockwise_scaled_mm`, and restores `@torch.no_grad()` on model forward.
- Key code excerpt:

```python
hidden_states = self.attn(
    positions=positions,
    hidden_states=hidden_states,
    forward_batch=forward_batch,
)
```

- Validation implication: PCG/compile fixes can change graph-capture behavior without changing model math; Qwen3.5 regression tests need both compile and non-compile coverage.

### PR #19391 - Enable Qwen3.5 MTP spec-v2 and add NVFP4 tests

- Status: merged 2026-03-04, merge commit `9457c049e19e1cfa75833ef4351ac5aa26941c2c`.
- Diff reviewed: `decode.py`, `memory_pool.py`, `forward_batch_info.py`, `qwen3_5_mtp.py`, `server_args.py`, `eagle_worker_v2.py`, `test_qwen35_models.py`.
- Motivation: MTP v2 needed to work for multimodal Qwen3.5 by carrying `mm_input_embeds` into the draft model; Qwen3.5 NVFP4 also needed real accuracy tests with chat template and acceptance length checks.
- Key implementation: passes `mm_input_embeds` into `_draft_extend_for_prefill`, treats draft extend v2 correctly in Qwen3.5 MTP, removes incorrect global extra-buffer assertions, and changes radix-cache/spec/no-buffer handling from silent disabling to explicit error.
- Key code excerpt:

```python
if mm_input_embeds is not None:
    forward_batch.mm_input_embeds = mm_input_embeds
```

```python
and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
```

- Validation implication: Qwen3.5 speculative decoding must be validated with `SGLANG_ENABLE_SPEC_V2`, reasoning parser `qwen3`, chat template, and `avg_spec_accept_length > 3.3`.

### PR #19411 - Last-layer communicator flag for Qwen3.5-27B repeat bug

- Status: merged; diff reviewed in Qwen3.5 decoder layer construction.
- Motivation: Qwen3.5-27B hit a repeat/output issue caused by layer communicator state not knowing when a decoder layer is the final layer.
- Key implementation: passes `is_last_layer=(layer_id == config.num_hidden_layers - 1)` into Qwen3.5 layer communicator setup.
- Key code excerpt:

```python
is_last_layer=(layer_id == config.num_hidden_layers - 1)
```

- Validation implication: small one-line communicator changes can materially affect dense Qwen3.5 output quality; keep a 27B decode smoke test in the matrix.

### PR #19670 - Pipeline parallel support

- Status: merged; diff reviewed in Qwen3.5 PP model construction and `test_pp_single_node.py`.
- Motivation: Qwen3.5 could not be split across pipeline stages without missing-layer placeholders and first/last-rank embedding/head handling.
- Key implementation: adds `PPMissingLayer`, `start_layer`/`end_layer`, `get_pp_indices`, first/last rank embedding/head handling, and a Qwen3.5 PP accuracy test.
- Key code excerpt:

```python
self.start_layer, self.end_layer, self.layers = make_layers(
    config.num_hidden_layers,
    get_layer,
    prefix="model.layers",
)
```

```python
def get_embed_and_head(self):
    embed = self.model.embed_tokens.weight if self.pp_group.is_first_rank else None
    head = self.lm_head.weight if self.pp_group.is_last_rank else None
    return embed, head
```

- Validation implication: every later weight-loading change must skip layers outside the local PP stage and preserve tied embedding semantics.

### PR #19767 - Qwen3.5 MTP and EPLB fixes

- Status: merged; diff reviewed in `qwen2_moe.py`, `qwen3_5_mtp.py`, `qwen3_next` MTP hunks.
- Motivation: MTP/NEXTN layers should not participate in EPLB expert-location dispatch like normal target-model MoE layers, and MTP forward should not pollute expert distribution recording.
- Key implementation: adds `is_nextn` to `Qwen2MoeSparseMoeBlock`, disables DeepEP `ExpertLocationDispatchInfo` for nextn, wraps MTP forward with expert-distribution recorder disable, and creates lazy expert-distribution values for normal layers.
- Key code excerpt:

```python
if self.is_nextn:
    self.expert_location_dispatch_info = None
```

```python
with get_global_expert_distribution_recorder().disable_this_region():
    hidden_states = self.model(...)
```

- Validation implication: MTP speedups and EPLB balancing must be validated together; otherwise speculative draft layers can distort routing telemetry.

### PR #19889 - TRTLLM/FlashInfer all-reduce fusion

- Status: merged; diff reviewed in layernorm all-reduce helper, Qwen2 MoE forward, `server_args.py`.
- Motivation: Qwen3.5 MoE needed all-reduce fusion with TRTLLM/FlashInfer paths to reduce communication overhead without breaking Gemma-style RMSNorm semantics.
- Key implementation: introduces `_forward_with_allreduce_fusion`, supports Gemma `weight + 1.0`, adds `should_allreduce_fusion` to `Qwen2MoeSparseMoeBlock.forward`, and declares Qwen3.5 architectures eligible for TRTLLM all-reduce fusion.
- Key code excerpt:

```python
return _forward_with_allreduce_fusion(
    hidden_states,
    residual,
    self.weight + 1.0,
    self.variance_epsilon,
)
```

```python
"Qwen3_5MoeForConditionalGeneration",
"Qwen3_5ForConditionalGeneration",
```

- Validation implication: all-reduce fusion changes the communication schedule; compare TP/EP/MTP acceptance and dense accuracy when toggling FlashInfer/TRTLLM backends.

### PR #19961 - Keep GDN `A_log` in FP32

- Status: merged; diff reviewed in Qwen3.5 GDN initialization.
- Motivation: `A_log` controls linear-attention recurrent dynamics and should not inherit lower precision from BF16/FP8 checkpoint paths.
- Key implementation: explicitly initializes `A_log` as `torch.float32` even when the surrounding model uses lower precision.
- Key code excerpt:

```python
self.A_log = nn.Parameter(
    torch.empty(self.num_v_heads // self.attn_tp_size, dtype=torch.float32),
)
```

- Validation implication: if GDN accuracy regresses after quant or dtype changes, check state parameters such as `A_log` before tuning kernels.

### PR #20386 - Replace `einops.rearrange` with native flatten

- Status: merged; diff reviewed in Qwen3.5 GDN output path.
- Motivation: `einops.rearrange` added measurable overhead in a hot GDN path.
- Key implementation: replaces the rearrange call with a native flatten/reshape operation. The PR body reports the operation dropping from about `12.67us` to `4.74us` over 720 calls on H100.
- Key code excerpt:

```python
core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
```

- Validation implication: small Python/tensor-layout simplifications are worth recording when they sit inside decode loops.

### PR #20736 - AMD shared-expert fusion with router experts

- Status: merged; diff reviewed in `qwen2_moe.py`, `qwen3_5.py` shared-expert loader hunks.
- Motivation: Qwen3.5 MoE has a shared expert whose intermediate size can match routed experts. On AMD/AITER, fusing the shared expert into the routed expert tensor avoids a separate shared-expert MLP path.
- Key implementation: `Qwen2MoeSparseMoeBlock` computes `num_fused_shared_experts`, expands top-k and expert count by one, appends the shared expert id/weight to `StandardTopKOutput`, and remaps `mlp.shared_expert.*` weights to `mlp.experts.{num_experts_base}.*` in Qwen3.5 loading.
- Key code excerpt:

```python
shared_expert_id = self.num_experts
shared_ids = torch.full((M, self.num_fused_shared_experts), shared_expert_id, ...)
fused_topk_ids = torch.cat([topk_output.topk_ids, shared_ids], dim=-1)
fused_topk_weights = torch.cat([topk_output.topk_weights, shared_weights], dim=-1)
```

```python
if num_fused_shared_experts > 0 and "mlp.shared_expert." in name:
    name = name.replace("mlp.shared_expert.", f"mlp.experts.{num_experts_base}.")
```

- Validation implication: this is a major AMD performance lane but it is fragile for quantized checkpoints. PR #22948 later disables it when MXFP4 shared experts are excluded from quantization.

### PR #20864 - Remove H2D/D2H overhead in Qwen3.5 SpecV2

- Status: merged 2026-03-31, merge commit `03e4f2858d5f164636ffa310f3296f7b5faac209`.
- Diff reviewed: `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`.
- Motivation: Qwen3.5 SpecV2 had avoidable host-device overhead in `prepare_v2_verify`, especially for text-only verify batches and Mamba track index construction.
- Key implementation: uses `torch.stack(...).to(torch.int64)` for Mamba track indices instead of constructing a tensor from a Python list of CUDA scalars, and adds a text-only fast path that creates the mrope delta tensor directly on device.
- Key code excerpt:

```python
batch.mamba_track_indices = torch.stack(
    [req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx] for req in batch.reqs]
).to(torch.int64)
```

```python
if all(mm_input is None for mm_input in mm_inputs):
    mrope_delta_tensor = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
```

- Validation implication: SpecV2 performance investigations should inspect Python list/tensor construction, not only CUDA kernels.

### PR #21019 - Fuse Qwen3.5 GDN split/reshape/cat with Triton

- Status: merged 2026-03-23, merge commit `5bdc07d974f6cf236fa765a685453ea5e587a838`.
- Diff reviewed: `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`.
- Motivation: Qwen3-Next and Qwen3.5 checkpoint layouts differ. Qwen3-Next stores fused/interleaved `in_proj_qkvz`; Qwen3.5 stores `in_proj_qkv` and `in_proj_z` separately, so Qwen3.5 needs a contiguous-layout fused projection kernel.
- Key implementation: introduces `fused_qkvzba_split_reshape_cat`, replaces four projection layers (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) with two fused layers (`in_proj_qkvz`, `in_proj_ba`), adds `_make_packed_weight_loader` for fused and split checkpoint formats, and adds mapping from split checkpoint names to fused parameters.
- Key code excerpt:

```python
self.in_proj_qkvz = self.create_qkvz_proj(...)
self.in_proj_ba = self.create_ba_proj(...)
self.in_proj_qkvz.weight.weight_loader = self._make_packed_weight_loader(self.in_proj_qkvz)
self.in_proj_ba.weight.weight_loader = self._make_packed_weight_loader(self.in_proj_ba)
```

```python
("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
("in_proj_qkvz.", "in_proj_z.", 3),
("in_proj_ba.", "in_proj_b.", 0),
("in_proj_ba.", "in_proj_a.", 1),
```

- Validation implication: the PR body reported about +7.4% output/token throughput and lower TTFT/TPOT on H200. Later PR #22312 is a direct correctness follow-up caused by non-contiguous B/A views after this fusion.

### PR #21070 - PP layer splitting fix

- Status: merged; diff reviewed in Qwen3.5 PP layer construction and loader skip logic.
- Motivation: Qwen3.5 PP could instantiate or load layers outside the local pipeline stage, causing memory pressure and missing-parameter behavior.
- Key implementation: passes `pp_rank` and `pp_size` into `make_layers`, and makes fused expert weight loading skip names not present in `params_dict`.
- Key code excerpt:

```python
self.start_layer, self.end_layer, self.layers = make_layers(
    config.num_hidden_layers,
    get_layer,
    prefix="model.layers",
    pp_rank=self.pp_group.rank_in_group,
    pp_size=self.pp_group.world_size,
)
```

```python
if name not in params_dict:
    continue
```

- Validation implication: PP tests must verify both runtime memory behavior and weight loading, especially for MoE and fused projections.

### PR #21234 - AMD MXFP4 Qwen3.5-397B support

- Status: merged; diff reviewed in Qwen3.5 quant mapping and VL subclass mapping.
- Motivation: AMD gfx950 needed MXFP4 Qwen3.5-397B support, including fused QKV/Gate/GDN projection names.
- Key implementation: under `_is_gfx95`, declares `packed_modules_mapping` for `qkv_proj`, `gate_up_proj`, `in_proj_qkvz` and `in_proj_ba`; Qwen3.5 VL subclasses reuse the same mapping and disable the HF-to-SGLang mapper.
- Key code excerpt:

```python
if _is_gfx95:
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
```

- Validation implication: AMD MXFP4 and NPU ModelSlim fixes both rely on model-local packed mappings; avoid pushing Qwen3.5 fused mapping back into a global loader-only hack.

### PR #21347 - PP tied-embedding weight loading

- Status: merged; diff reviewed in Qwen3.5 dense and MoE load paths.
- Motivation: Qwen3.5 4B dense uses tied embeddings, and the last PP rank needs `lm_head.weight` even when `model.embed_tokens.weight` belongs to the first rank.
- Key implementation: if `tie_word_embeddings` is enabled and the last PP rank sees `model.embed_tokens.weight`, the loader redirects it to `lm_head.weight`.
- Key code excerpt:

```python
if self.config.tie_word_embeddings and name == "model.embed_tokens.weight":
    name = "lm_head.weight"
```

- Validation implication: tied embedding PP bugs show up as load-time missing heads or decode garbage, not as obvious architecture errors.

### PR #21448 - MoE loading and Mamba cache sharding in PP

- Status: merged; diff reviewed in `memory_pool.py`, Qwen3.5 load paths, Qwen3-VL stage properties.
- Motivation: pipeline parallel Qwen3.5 should only allocate Mamba state for local layers and only load weights for layers assigned to the current PP stage.
- Key implementation: Mamba pool receives local `mamba_layer_ids`, maps layer ids relative to `start_layer`, and Qwen3.5 `load_weights` skips layers outside `[start_layer, end_layer)`.
- Key code excerpt:

```python
mamba_layer_ids = [
    layer_id for layer_id in cache_params.layers
    if start_layer <= layer_id < end_layer
]
```

```python
layer_id = get_layer_id(name)
if layer_id is not None and (layer_id < self.model.start_layer or layer_id >= self.model.end_layer):
    continue
```

- Validation implication: Qwen3.5 PP cannot be validated by attention KV only; Mamba state shape and layer locality must be included.

### PR #21487 - GB300 nightly benchmark suites

- Status: merged 2026-03-29, merge commit `9d64a821735430ad7e3bd2f19e6c87f6ee057c3e`.
- Diff reviewed: `test/registered/gb300/test_qwen35_fp8.py`, `test_qwen35_nvfp4.py`, GLM/DeepSeek/Kimi GB300 tests, `run_suite.py`.
- Motivation: GB300/4x B200 NVL4 needed per-model nightly performance coverage for FP8 and NVFP4 lanes, including Qwen3.5 and GLM-5.
- Key implementation: adds Qwen3.5 FP8 and NVFP4 GB300 registered tests with TP4, MTP/spec-v2, `trtllm_mha`, FlashInfer all-reduce fusion, and Qwen parsers.
- Key code excerpt:

```python
QWEN35_FP8_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B-FP8"
QWEN35_NVFP4_MODEL_PATH = "nvidia/Qwen3.5-397B-A17B-NVFP4"
```

```python
extra_args=base_args + mtp_args,
env={"SGLANG_ENABLE_SPEC_V2": "1"},
```

- Validation implication: GB300 is now a separate deployment lane; cookbook updates for B200/GB300 must stay aligned with these tests.

### PR #21669 - AMD Qwen3.5 FP8 nightly performance

- Status: merged; diff reviewed in AMD nightly workflow and perf test files.
- Motivation: AMD CI had accuracy coverage but lacked Qwen3.5-397B FP8 performance tracking on MI30x/MI35x.
- Key implementation: adds non-blocking AMD perf jobs with `SGLANG_USE_AITER=1`, default model `Qwen/Qwen3.5-397B-A17B-FP8`, TP=8, Triton attention backend, static memory fraction and multithread load config.
- Key code excerpt:

```python
QWEN35_FP8_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
other_args=[
    "--tp", "8",
    "--attention-backend", "triton",
    "--model-loader-extra-config", '{"enable_multithread_load": true}',
]
```

- Validation implication: AMD perf tests should be treated as optimization guardrails, not just platform smoke tests.

### PR #21692 - NPU Qwen3.5 quantization fix

- Status: merged 2026-04-08, merge commit `cd373667cdfab2677dad680697a954e59473d7f6`.
- Diff reviewed: `modelslim.py`, `model_loader/loader.py`, `qwen3_5.py`.
- Motivation: after #21019 fused `in_proj_qkv + in_proj_z` into `in_proj_qkvz` and `in_proj_b + in_proj_a` into `in_proj_ba`, NPU/ModelSlim quantization no longer found the right module mapping.
- Key implementation: extends Qwen3.5 packed module mapping to NPU, refactors ModelSlim `get_linear_scheme()` to mirror MoE scheme lookup, and checks both per-subset and global packed mappings when skipping layers.
- Key code excerpt:

```python
if _is_gfx95 or _is_npu:
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
```

```python
if self.is_layer_skipped(prefix, packed_modules_mapping_subset) or self.is_layer_skipped(prefix, self.packed_modules_mapping):
    return UnquantizedLinearMethod()
```

- Validation implication: quantization bugs after GDN fusion often live in naming/mapping, not math kernels.

### PR #21849 - Allow Qwen3.5 models for encoder disaggregation

- Status: merged 2026-04-06, merge commit `7f2fcc0b08592fbcccedcc9f27225e1acc0198d9`.
- Diff reviewed: `server_args.py`, `encode_server.py`, `qwen_vl.py`, `test_epd_disaggregation.py`.
- Motivation: Qwen3.5 multimodal runtime worked, but encoder disaggregation startup rejected `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration` because the allowlist was stale.
- Key implementation: adds both Qwen3.5 architectures to the encoder-disaggregation allowlist, extends video metadata/timestamp handling to `qwen3_5` and `qwen3_5_moe`, and adds an EPD image/video regression test with `Qwen/Qwen3.5-27B`.
- Key code excerpt:

```python
"Qwen3_5ForConditionalGeneration",
"Qwen3_5MoeForConditionalGeneration",
```

```python
self.model_type in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
```

- Validation implication: Qwen3.5-VL support includes split encoder/language deployments, not only single-process multimodal serving.

### PR #22145 - NIXL heterogeneous TP KV transfer fix

- Status: merged 2026-04-07, merge commit `3148742ddb2ccd3478065c91f0222bb4817903f0`.
- Diff reviewed: `python/sglang/srt/disaggregation/nixl/conn.py`.
- Motivation: NIXL disaggregated serving with heterogeneous TP on non-MLA models hung because RDMA notification keys used `pp_rank` and collapsed all prefill ranks when PP=1; GQA head distribution also used per-rank KV head counts and lost information when total KV heads were fewer than TP ranks.
- Key implementation: derives head distribution from `total_kv_head_num`, adds GQA replication/unique-head handling, and uses `engine_rank` in KV/state notifications.
- Key code excerpt:

```python
total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
dst_heads_per_rank = max(1, total_kv_heads // decode_tp_size)
```

```python
notif = f"{req.room}_kv_{chunk_id}_{int(is_last)}_{self.kv_args.engine_rank}"
```

- Validation implication: this is Step 1 for Qwen3.5 PD/NIXL hetero TP. Without it, decode can hang forever even though prefill drained requests.

### PR #22240 - NIXL Mamba state slice transfer for heterogeneous TP

- Status: merged 2026-04-07, merge commit `5ae00ecd48b11f943d8ba319f3b0e828d8a41116`.
- Diff reviewed: `python/sglang/srt/disaggregation/nixl/conn.py`.
- Motivation: Mooncake supported Mamba state slice transfer for heterogeneous TP, but NIXL raised `RuntimeError`, blocking hybrid Mamba models such as Qwen3.5 under PD with different prefill/decode TP layouts.
- Key implementation: extends `KVArgsRegisterInfo` with destination state metadata, adds `_send_mamba_state_slice()` to slice `conv_state` and `temporal_state` along the TP-sharded dimension, and sends `state_item_lens`/`state_dim_per_tensor` during registration.
- Key code excerpt:

```python
dst_state_item_lens: list[int] = dataclasses.field(default_factory=list)
dst_state_dim_per_tensor: list[int] = dataclasses.field(default_factory=list)
```

```python
src_dim_offset = src_dim_start * src_bytes_per_dim
dst_dim_offset = dst_dim_start * dst_bytes_per_dim
bytes_to_send = num_dims_to_send * src_bytes_per_dim
```

- Validation implication: Qwen3.5 PD tests need both KV cache and Mamba state transfer checks. The PR body validated Qwen3.5-397B-A17B-FP8 GSM8K/GPQA on GB200.

### PR #22312 - GDN non-contiguous B/A tensor correctness fix

- Status: merged 2026-04-10, merge commit `8ba96460440c6f83ed523b33ed07b05e302ad690`.
- Diff reviewed: `fused_gdn_gating.py`, `fused_sigmoid_gating_recurrent.py`, `test_gdn_noncontiguous_stride.py`.
- Motivation: after #21019, Qwen3.5-27B could take a fallback BA path where `mixed_ba.split()` returns non-contiguous B/A views. GDN Triton kernels assumed contiguous layout and hardcoded token/batch strides, causing accuracy to collapse from 49/50 to 3/50 in the reported regression test.
- Key implementation: passes explicit `stride_a` and `stride_b` to `fused_gdn_gating`, computes token-axis `stride_a` in `fused_sigmoid_gating_delta_rule_update`, and adds CUDA regression tests comparing contiguous and split-view non-contiguous inputs.
- Key code excerpt:

```python
stride_a = a.stride(0)
stride_b = b.stride(0)
blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
```

```python
stride_a = a.stride()[-2]
p_a += stride_a
```

- Validation implication: every fused layout optimization must include stride-aware tests, especially for split views returned by PyTorch.

### PR #22358 - DFLASH support for Qwen3.5 and sibling backends

- Status: merged 2026-04-09, merge commit `c3833ba929ee3a36e75b8d7b58d91e2c86c49d40`.
- Diff reviewed: `qwen3_5.py`, `qwen3.py`, `qwen3_moe.py`, `qwen3_next.py`, `qwen3_vl.py`, and other backend files.
- Motivation: DFLASH training/support needed aux hidden-state capture across z-lab model backends without waiting for the later DFLASH spec v2.
- Key implementation: Qwen3.5 decoder layers call `prepare_attn_and_capture_last_layer_outputs`, Qwen3.5 model tracks `layers_to_capture`, returns `(hidden_states, aux_hidden_states)` when needed, and exposes `set_dflash_layers_to_capture()`.
- Key code excerpt:

```python
self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
    hidden_states,
    residual,
    forward_batch,
    captured_last_layer_outputs=captured_last_layer_outputs,
)
```

```python
def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):
    self.layers_to_capture = layers_to_capture
    for layer_id in self.layers_to_capture:
        setattr(self.layers[layer_id], "_is_layer_to_capture", True)
```

- Validation implication: DFLASH changes alter forward return shape when aux states are captured, so serving and logits paths must explicitly unwrap auxiliary hidden states.

### PR #22431 - Qwen3.5 processor-output video fix

- Status: merged 2026-04-18, merge commit `2a327f08772f6b9ada7f2f4792f9b7d0e16a5fa1`.
- Diff reviewed: `python/sglang/srt/multimodal/processors/qwen_vl.py`.
- Motivation: when Qwen3.5 video input used `processor_output` format, `preprocess_video()` returned a single object instead of `(video, metadata)`, while Qwen3.5 processing expected two values and raised `ValueError: too many values to unpack`.
- Key implementation: makes non-`VideoDecoderWrapper` inputs return `(vr, None)` so raw decoded video and processor-output paths share the same tuple contract.
- Key code excerpt:

```python
if not is_video_obj:
    return vr, None
```

- Validation implication: Qwen3.5 multimodal docs should cover both raw video URL/path input and pre-tokenized processor-output input.

### PR #22493 - MambaPool CPU offload during retraction

- Status: merged 2026-04-22, merge commit `415f64e763750e3350d87b03d554171937279ce7`.
- Diff reviewed: `schedule_batch.py`, `scheduler.py`, `allocator.py`, `memory_pool.py`, `test_mamba_unittest.py`.
- Motivation: during request retraction, only attention KV cache was copied to CPU. Qwen3.5 hybrid Mamba state (`conv` + `temporal`) was dropped, corrupting generation after retracted requests resumed.
- Key implementation: adds `MambaPool.get_cpu_copy()`/`load_cpu_copy()`, extends `HybridLinearKVPool` CPU offload to include optional `mamba_indices`, passes `mamba_pool_idx` from `Req.offload_kv_cache()`/`load_kv_cache()`, and logs `#mamba_num_gained`.
- Key code excerpt:

```python
self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(
    token_indices, mamba_indices=self.mamba_pool_idx
)
```

```python
def get_cpu_copy(self, indices, mamba_indices=None, **kwargs):
    kv_cpu = self.full_kv_pool.get_cpu_copy(indices)
    mamba_cpu = self.mamba_pool.get_cpu_copy(mamba_indices) if mamba_indices is not None else None
    return kv_cpu, mamba_cpu
```

- Validation implication: retraction and memory pressure tests for Qwen3.5 must assert both token KV and Mamba state round-trips.

### PR #22908 - AMD radix-cache/speculative decoding conflict resolution

- Status: merged 2026-04-21, merge commit `ac08ebed6510ae600e50c77ffd5484601f264c3f`.
- Diff reviewed: `python/sglang/srt/server_args.py`.
- Motivation: Qwen3.5 MoE speculative decoding with `no_buffer` and radix cache raised a hard error. The suggested `extra_buffer` workaround is CUDA/FLA-only and fails on ROCm.
- Key implementation: review settled on ROCm-specific handling: if `is_hip()` and radix cache is enabled under incompatible Qwen3.5 speculative settings, automatically disable radix cache; CUDA/other platforms keep the explicit error telling users to use `extra_buffer` and `SGLANG_ENABLE_SPEC_V2=1`.
- Key code excerpt:

```python
if is_hip():
    logger.warning(
        f"Speculative decoding for {model_arch} is not compatible "
        "with radix cache on ROCm devices. Automatically disabling radix cache."
    )
    self.disable_radix_cache = True
else:
    raise ValueError(...)
```

- Validation implication: AMD commands can omit `--disable-radix-cache` for this conflict, but CUDA commands should still use `--mamba-scheduler-strategy extra_buffer` for spec-v2.

### PR #22913 - Split Qwen3.5 B200 FP4 tests and bump partitions

- Status: merged 2026-04-17, merge commit `005209317888bb497e4c10b2a793b4d1f666c533`.
- Diff reviewed: `.github/workflows/pr-test.yml`, `test_qwen35_fp4_triton.py`, `test_qwen35_fp4_mtp_v2.py`, deleted `test_qwen35_models.py`.
- Motivation: all Qwen3.5 NVFP4 B200 tests lived in one file and launched multiple 234GB model servers, causing slower B200 partitions to hit the 30-minute timeout.
- Key implementation: splits Triton and MTP-v2 tests into separate files, removes v1 MTP per review, and increases `stage-c-test-4-gpu-b200` partitions from 5 to 6.
- Key code excerpt:

```yaml
matrix:
  part: [0, 1, 2, 3, 4, 5]
```

```python
register_cuda_ci(est_time=540, suite="stage-c-test-4-gpu-b200")
envs.SGLANG_ENABLE_SPEC_V2.set(True)
```

- Validation implication: Qwen3.5 CI stability is part of model optimization history; without split tests, real regressions are hidden behind suite timeouts.

### PR #22948 - Disable shared-expert fusion for MXFP4 excluded shared expert

- Status: merged 2026-04-16, merge commit `52f0b86f5d639e2cf376e12d699d44ec67da460d`.
- Diff reviewed: `python/sglang/srt/models/qwen2_moe.py`.
- Motivation: after #20736 enabled shared-expert fusion, Qwen3.5 MXFP4 checkpoints broke because routed experts were MXFP4 but the shared expert remained BF16/FP32 via quant exclusion. Fusing that shared expert into the quantized MoE tensor would require online quantization that did not exist.
- Key implementation: `can_fuse_shared_expert()` now receives `quant_config` and returns false when `exclude_layers` contains `shared_expert` but not `shared_expert_gate` and not MTP shared-expert paths.
- Key code excerpt:

```python
if quant_config is not None:
    exclude_layers = getattr(quant_config, "exclude_layers", [])
    if any(
        "shared_expert" in layer
        and "shared_expert_gate" not in layer
        and not layer.startswith("mtp.")
        for layer in exclude_layers
    ):
        return False
```

- Validation implication: shared-expert fusion must inspect quant exclusion lists, not just model shape.

### PR #23034 - Qwen3.6 docs with Qwen3.5 deployment rule updates

- Status: merged 2026-04-20, merge commit `59b1d86669853d308d484351c1542faa4285e27f`.
- Diff reviewed: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`, intro cards and docs JSON.
- Motivation: Qwen3.6 docs replaced the Qwen card target, but the deployment component also encoded Qwen3.5-derived MTP/Mamba behavior.
- Key implementation: when MTP/speculative decoding is enabled, the snippet disables Mamba scheduler V1 and forces V2/`extra_buffer`.
- Key code excerpt:

```jsx
const mtpEnabled = values.speculative === 'enabled';
if (mtpEnabled) {
  return [
    { id: 'v1', label: 'V1', default: false, disabled: true },
    { id: 'v2', label: 'V2', default: true },
  ];
}
```

- Validation implication: Qwen3.5 and Qwen3.6 cookbook snippets share runtime assumptions; docs changes can affect user-visible recommended Qwen3.5 commands.

### PR #23467 - FP8 `modules_to_not_convert` dot-boundary fix

- Status: merged 2026-04-22, merge commit `bfec013403ad117bdb0da9486b399a9c87c569d6`.
- Diff reviewed: `python/sglang/srt/layers/quantization/utils.py`.
- Motivation: substring matching in FP8 ignored-layer handling caused collisions such as Qwen3.5 `in_proj_a` matching `in_proj_ba`, and Qwen3.6 `mlp.gate` matching `mlp.gate_up_proj`.
- Key implementation: introduces `_module_path_match()` with exact and dot-boundary matching, and lists fused fallback shards for `qkv_proj`, `gate_up_proj`, `in_proj_ba`, and `in_proj_qkvz`.
- Key code excerpt:

```python
def _module_path_match(ignored: str, prefix: str) -> bool:
    if ignored == prefix:
        return True
    if prefix.startswith(ignored + "."):
        return True
    return ("." + ignored + ".") in ("." + prefix + ".")
```

```python
"_FALLBACK_FUSED_SHARDS": {
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
}
```

- Validation implication: this is a direct guard for Qwen3.5 fused GDN projection quant loading.

### PR #23474 - Hybrid linear-attention CPU offload fix

- Status: open at review time; diff reviewed in `OffloaderV1` and `test_offloader_tied_params.py`.
- Motivation: hybrid linear-attention models with tied/view parameters can lose aliasing when CPU offload materializes tensors independently, breaking Qwen3.5/Qwen3.6-style fused/view-heavy weights.
- Key implementation: records view aliases before offload with `state_dict(keep_vars=True)`, shares device tensors for tied parameters via `src_to_dev`, and recreates cached views using `as_strided`.
- Key code excerpt:

```python
for name, tensor in module.state_dict(keep_vars=True).items():
    ...
    view_aliases[name] = (src_name, tensor.size(), tensor.stride(), tensor.storage_offset())
```

```python
dev_tensor = src_to_dev[src_name].as_strided(size, stride, storage_offset)
```

- Validation implication: this is not merged history yet, but it is relevant radar for Qwen3.5/Qwen3.6 CPU-offload correctness. Keep it separate from merged PR cards until merged.

## sgl-cookbook / Docs Evidence

- `sgl-cookbook#164`: initial Qwen3.5 cookbook.
- `sgl-cookbook#168`: FP8 and NVFP4 deployment updates.
- `sgl-cookbook#169`: B200 configs.
- `sgl-cookbook#177` and `#214`: H200 FP8 and MTP command updates.
- `sgl-cookbook#179`: AMD MI300X/MI325X/MI355X deployment notes.
- `sgl-cookbook#207`: B200 FP8 FlashInfer all-reduce fusion tip.
- `sgl-cookbook#230`: B200 FP4/NVFP4 generator update.
- `sgl-cookbook#237`: FP8 KV accuracy caution and command generation cleanup.
- Official SGLang Qwen3.5 docs describe hybrid GDN + full-attention, shared experts, DeepStack Vision/Conv3d, AMD `--attention-backend triton`, `SGLANG_USE_AITER=1`, `--reasoning-parser qwen3`, and `--tool-call-parser qwen3_coder`.
- AMD's Qwen3.5 day-0 article confirms the ROCm optimization path: GDN through Triton, shared-expert MoE through hipBLASLt/AITER, and native multimodal kernels through MIOpen/PyTorch.

## Reviewer Checklist For Future Updates

- Do not add a PR number to this file unless its diff/source was opened and reviewed.
- For every new Qwen3.5 PR, record the actual modified files and at least one concrete code excerpt.
- Keep merged PRs separate from open radar.
- Re-check Qwen3.5 with this matrix: dense vs MoE, text vs VLM, BF16 vs FP8 vs NVFP4/MXFP4, CUDA vs ROCm vs NPU, TP/PP/EP, MTP spec-v1/v2, PD/NIXL, and retraction.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG Qwen3.5 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-09 | [#18489](https://github.com/sgl-project/sglang/pull/18489) | merged | [MODEL] Adding Support for Qwen3.5 Models | model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py` |
| 2026-02-10 | [#18538](https://github.com/sgl-project/sglang/pull/18538) | merged | [Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation | model wrapper | `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-10 | [#18544](https://github.com/sgl-project/sglang/pull/18544) | merged | [Ascend]Support qwen3.5 | model wrapper, attention/backend, quantization | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2026-02-17 | [#18926](https://github.com/sgl-project/sglang/pull/18926) | merged | feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation | model wrapper, quantization | `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-02-17 | [#18937](https://github.com/sgl-project/sglang/pull/18937) | merged | [Qwen3.5] Enable nvfp4 checkpoint | model wrapper | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-02-20 | [#19070](https://github.com/sgl-project/sglang/pull/19070) | merged | fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1 | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-24 | [#19220](https://github.com/sgl-project/sglang/pull/19220) | merged | [PCG] fix piecewise cuda graph for Qwen3.5 | model wrapper, quantization | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-02-26 | [#19391](https://github.com/sgl-project/sglang/pull/19391) | merged | [Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4 | model wrapper, scheduler/runtime, tests/benchmarks | `test/registered/4-gpu-models/test_qwen35_models.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/disaggregation/decode.py` |
| 2026-02-26 | [#19411](https://github.com/sgl-project/sglang/pull/19411) | merged | [Qwen3.5] Qwen3.5-27B inference repeat bug fix | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-02 | [#19670](https://github.com/sgl-project/sglang/pull/19670) | merged | [Qwen3.5] Support Qwen3.5 Pipeline Parallelism | model wrapper, tests/benchmarks | `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py` |
| 2026-03-03 | [#19767](https://github.com/sgl-project/sglang/pull/19767) | merged | Fix qwen3.5 mtp eplb related issues | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2026-03-04 | [#19889](https://github.com/sgl-project/sglang/pull/19889) | merged | Use TRTLLM allreduce fusion for Qwen 3.5 | model wrapper, MoE/router | `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-03-05 | [#19961](https://github.com/sgl-project/sglang/pull/19961) | merged | fix: change qwen 3.5 linear attention a_log to fp32 | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-11 | [#20386](https://github.com/sgl-project/sglang/pull/20386) | merged | perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe… | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-17 | [#20736](https://github.com/sgl-project/sglang/pull/20736) | merged | [AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8 | model wrapper, MoE/router | `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-18 | [#20864](https://github.com/sgl-project/sglang/pull/20864) | merged | [Perf]Remove H2D for Qwen3.5 SpecV2 | scheduler/runtime | `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py` |
| 2026-03-20 | [#21019](https://github.com/sgl-project/sglang/pull/21019) | merged | [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel | model wrapper, kernel | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-21 | [#21070](https://github.com/sgl-project/sglang/pull/21070) | merged | [Qwen3.5] Fix broken pipeline parallelism layer splitting | model wrapper, tests/benchmarks | `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py` |
| 2026-03-23 | [#21234](https://github.com/sgl-project/sglang/pull/21234) | merged | [AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-24 | [#21347](https://github.com/sgl-project/sglang/pull/21347) | merged | [Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-26 | [#21448](https://github.com/sgl-project/sglang/pull/21448) | merged | [Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-03-26 | [#21487](https://github.com/sgl-project/sglang/pull/21487) | merged | feat(ci): add GB300 nightly benchmark test suites | quantization, scheduler/runtime, tests/benchmarks | `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py` |
| 2026-03-30 | [#21669](https://github.com/sgl-project/sglang/pull/21669) | merged | [AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x | quantization, tests/benchmarks | `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` |
| 2026-03-30 | [#21692](https://github.com/sgl-project/sglang/pull/21692) | merged | [Bugfix] [NPU] Qwen3.5 with quantization fix | model wrapper, quantization | `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_loader/loader.py` |
| 2026-04-01 | [#21849](https://github.com/sgl-project/sglang/pull/21849) | merged | [VLM]: allow Qwen3.5 models for encoder disaggregation | multimodal/processor, tests/benchmarks | `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-04-05 | [#22145](https://github.com/sgl-project/sglang/pull/22145) | merged | [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support) | misc | `python/sglang/srt/disaggregation/nixl/conn.py` |
| 2026-04-07 | [#22240](https://github.com/sgl-project/sglang/pull/22240) | merged | [Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5) | misc | `python/sglang/srt/disaggregation/nixl/conn.py` |
| 2026-04-08 | [#22312](https://github.com/sgl-project/sglang/pull/22312) | merged | Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B | attention/backend, tests/benchmarks | `test/registered/attention/test_gdn_noncontiguous_stride.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` |
| 2026-04-08 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-09 | [#22431](https://github.com/sgl-project/sglang/pull/22431) | merged | Fix Qwen3.5 video processing when passing video_data in "processor_output" format | multimodal/processor | `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-04-10 | [#22493](https://github.com/sgl-project/sglang/pull/22493) | merged | Add MambaPool kvcache offloading during retraction | scheduler/runtime, tests/benchmarks | `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py` |
| 2026-04-15 | [#22908](https://github.com/sgl-project/sglang/pull/22908) | merged | [AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict. | misc | `python/sglang/srt/server_args.py` |
| 2026-04-15 | [#22913](https://github.com/sgl-project/sglang/pull/22913) | merged | test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6 | model wrapper, quantization, kernel, tests/benchmarks | `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` |
| 2026-04-16 | [#22948](https://github.com/sgl-project/sglang/pull/22948) | merged | [AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled | model wrapper, MoE/router | `python/sglang/srt/models/qwen2_moe.py` |
| 2026-04-17 | [#23034](https://github.com/sgl-project/sglang/pull/23034) | merged | docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs | model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, docs/config | `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` |
| 2026-04-22 | [#23467](https://github.com/sgl-project/sglang/pull/23467) | merged | fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert | quantization | `python/sglang/srt/layers/quantization/utils.py` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | tests/benchmarks | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |

## Diff Cards

### PR #18489 - [MODEL] Adding Support for Qwen3.5 Models

- Link: https://github.com/sgl-project/sglang/pull/18489
- Status/date: `merged`, created 2026-02-09, merged 2026-02-09; author `zju-stu-lizheng`.
- Diff scope read: `17` files, `+1923/-9`; areas: model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, config, processor, spec, attention, vision, cache, cuda, expert, kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` added +1310/-0 (1310 lines); hunks: +# Copyright 2025 Qwen Team; symbols: Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, forward
  - `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0 (415 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/configs/qwen3_5.py` added +113/-0 (113 lines); hunks: +from transformers import PretrainedConfig; symbols: Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__, Qwen3_5Config
  - `python/sglang/srt/model_executor/model_runner.py` modified +14/-3 (17 lines); hunks: Lfm2Config,; def qwen3_next_config(self):; symbols: qwen3_next_config, hybrid_gdn_config, compute_logprobs_only, model_is_mrope
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +16/-1 (17 lines); hunks: import numpy as np; from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem; symbols: preprocess_video, QwenVLImageProcessor, process_mm_data_async
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`; keywords observed in patches: moe, config, processor, spec, attention, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18538 - [Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation

- Link: https://github.com/sgl-project/sglang/pull/18538
- Status/date: `merged`, created 2026-02-10, merged 2026-02-12; author `zju-stu-lizheng`.
- Diff scope read: `2` files, `+62/-118`; areas: model wrapper; keywords: config, moe, quant, attention, expert, processor, spec, triton.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +44/-112 (156 lines); hunks: from sglang.srt.layers.layernorm import GemmaRMSNorm; def __init__(; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/models/qwen3_5.py` modified +18/-6 (24 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: config, moe, quant, attention, expert, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18544 - [Ascend]Support qwen3.5

- Link: https://github.com/sgl-project/sglang/pull/18544
- Status/date: `merged`, created 2026-02-10, merged 2026-02-12; author `chenxu214`.
- Diff scope read: `3` files, `+23/-4`; areas: model wrapper, attention/backend, quantization; keywords: attention, quant, cache, config, cuda, expert, moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: # Distributed; def __init__(; symbols: __init__, load_fused_expert_weights, get_model_config_for_expert_location
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +9/-0 (9 lines); hunks: def is_layer_skipped(; symbols: is_layer_skipped
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +2/-2 (4 lines); hunks: from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_npu; def __init__(self, model_runner: ModelRunner):; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; keywords observed in patches: attention, quant, cache, config, cuda, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18926 - feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation

- Link: https://github.com/sgl-project/sglang/pull/18926
- Status/date: `merged`, created 2026-02-17, merged 2026-02-18; author `zju-stu-lizheng`.
- Diff scope read: `4` files, `+57/-12`; areas: model wrapper, quantization; keywords: config, quant, kv, attention, awq, expert, fp8, test, vision.
- Code diff details:
  - `python/sglang/srt/layers/linear.py` modified +48/-0 (48 lines); hunks: def _load_fused_module_from_checkpoint(; def weight_loader_v2(; symbols: _load_fused_module_from_checkpoint, _load_merged_block_scale, weight_loader_v2, weight_loader_v2
  - `python/sglang/srt/layers/quantization/fp8.py` modified +5/-2 (7 lines); hunks: def from_config(cls, config: Dict[str, Any]) -> Fp8Config:; symbols: from_config
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-6 (7 lines); hunks: def __init__(; def load_fused_expert_weights(; symbols: __init__, load_fused_expert_weights
  - `python/sglang/srt/models/qwen3_vl.py` modified +3/-4 (7 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; keywords observed in patches: config, quant, kv, attention, awq, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18937 - [Qwen3.5] Enable nvfp4 checkpoint

- Link: https://github.com/sgl-project/sglang/pull/18937
- Status/date: `merged`, created 2026-02-17, merged 2026-02-19; author `hlu1`.
- Diff scope read: `3` files, `+26/-8`; areas: model wrapper; keywords: config, fp4, quant, expert, kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +19/-7 (26 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, load_weights
  - `python/sglang/srt/layers/rotary_embedding.py` modified +3/-1 (4 lines); hunks: def get_rope(; symbols: get_rope
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0 (4 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; keywords observed in patches: config, fp4, quant, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19070 - fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1

- Link: https://github.com/sgl-project/sglang/pull/19070
- Status/date: `merged`, created 2026-02-20, merged 2026-02-25; author `zju-stu-lizheng`.
- Diff scope read: `1` files, `+32/-6`; areas: model wrapper; keywords: moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +32/-6 (38 lines); hunks: def forward(; def forward(; symbols: forward, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19220 - [PCG] fix piecewise cuda graph for Qwen3.5

- Link: https://github.com/sgl-project/sglang/pull/19220
- Status/date: `merged`, created 2026-02-24, merged 2026-02-26; author `zminglei`.
- Diff scope read: `4` files, `+9/-46`; areas: model wrapper, quantization; keywords: config, attention, cuda, eagle, expert, fp8, kv, lora, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-25 (25 lines); hunks: import torch; make_layers,; symbols: set_eagle3_layers_to_capture, gdn_with_output
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-21 (22 lines); hunks: import torch.nn as nn; from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock; symbols: forward, _forward, _forward
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +7/-0 (7 lines); hunks: def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=No; symbols: _fp8_scaled_mm_abstract, _fp8_blockwise_scaled_mm_abstract
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: def get_input_embeddings(self):; symbols: get_input_embeddings, should_apply_lora, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; keywords observed in patches: config, attention, cuda, eagle, expert, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19391 - [Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4

- Link: https://github.com/sgl-project/sglang/pull/19391
- Status/date: `merged`, created 2026-02-26, merged 2026-03-04; author `hlu1`.
- Diff scope read: `8` files, `+252/-16`; areas: model wrapper, scheduler/runtime, tests/benchmarks; keywords: scheduler, spec, cache, eagle, test, attention, config, cuda, fp4, kv.
- Code diff details:
  - `test/registered/4-gpu-models/test_qwen35_models.py` added +240/-0 (240 lines); hunks: +import unittest; symbols: TestQwen35FP4, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/server_args.py` modified +5/-4 (9 lines); hunks: def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache, _handle_sampling_backend
  - `python/sglang/srt/disaggregation/decode.py` modified +0/-5 (5 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +0/-5 (5 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/speculative/eagle_worker_v2.py` modified +4/-0 (4 lines); hunks: def _draft_extend_for_prefill(; def _draft_extend_for_prefill(; symbols: _draft_extend_for_prefill, _draft_extend_for_prefill, forward_batch_generation
- Optimization/support interpretation: The concrete diff surface is `test/registered/4-gpu-models/test_qwen35_models.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/disaggregation/decode.py`; keywords observed in patches: scheduler, spec, cache, eagle, test, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/4-gpu-models/test_qwen35_models.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/disaggregation/decode.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19411 - [Qwen3.5] Qwen3.5-27B inference repeat bug fix

- Link: https://github.com/sgl-project/sglang/pull/19411
- Status/date: `merged`, created 2026-02-26, merged 2026-02-26; author `AlfredYyong`.
- Diff scope read: `1` files, `+2/-0`; areas: model wrapper; keywords: attention, config.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: def __init__(; def __init__(; symbols: __init__, forward, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: attention, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19670 - [Qwen3.5] Support Qwen3.5 Pipeline Parallelism

- Link: https://github.com/sgl-project/sglang/pull/19670
- Status/date: `merged`, created 2026-03-02, merged 2026-03-07; author `yuan-luo`.
- Diff scope read: `2` files, `+114/-13`; areas: model wrapper, tests/benchmarks; keywords: attention, cache, config, cuda, expert, test.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +60/-13 (73 lines); hunks: ); from sglang.srt.layers.radix_attention import RadixAttention; symbols: __init__, get_layer, get_layer, get_input_embeddings
  - `test/registered/distributed/test_pp_single_node.py` modified +54/-0 (54 lines); hunks: def test_pp_consistency(self):; symbols: test_pp_consistency, TestQwen35PPAccuracy, setUpClass, run_gsm8k_test
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py`; keywords observed in patches: attention, cache, config, cuda, expert, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19767 - Fix qwen3.5 mtp eplb related issues

- Link: https://github.com/sgl-project/sglang/pull/19767
- Status/date: `merged`, created 2026-03-03, merged 2026-03-09; author `luoyuyan`.
- Diff scope read: `5` files, `+79/-16`; areas: model wrapper, MoE/router; keywords: config, quant, expert, moe, cuda, processor, attention, deepep, router, triton.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-1 (35 lines); hunks: from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration; def __init__(; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6 (25 lines); hunks: from transformers import PretrainedConfig; def __init__(; symbols: __init__, __init__, get_model_config_for_expert_location, get_embed_and_head
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +12/-7 (19 lines); hunks: from transformers import PretrainedConfig; def __init__(; symbols: __init__, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +8/-2 (10 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, get_moe_weights, _forward_deepep
  - `python/sglang/srt/models/qwen3_next.py` modified +6/-0 (6 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py`; keywords observed in patches: config, quant, expert, moe, cuda, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19889 - Use TRTLLM allreduce fusion for Qwen 3.5

- Link: https://github.com/sgl-project/sglang/pull/19889
- Status/date: `merged`, created 2026-03-04, merged 2026-03-18; author `b8zhong`.
- Diff scope read: `4` files, `+88/-52`; areas: model wrapper, MoE/router; keywords: moe, flash, attention, fp4, processor, spec, topk, triton.
- Code diff details:
  - `python/sglang/srt/layers/layernorm.py` modified +63/-48 (111 lines); hunks: import torch_npu; def forward_with_allreduce_fusion(; symbols: _forward_with_allreduce_fusion, RMSNorm, __init__, forward_with_allreduce_fusion
  - `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: def forward(; def forward(; symbols: forward, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +11/-2 (13 lines); hunks: RowParallelLinear,; def forward(; symbols: forward, forward
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`; keywords observed in patches: moe, flash, attention, fp4, processor, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19961 - fix: change qwen 3.5 linear attention a_log to fp32

- Link: https://github.com/sgl-project/sglang/pull/19961
- Status/date: `merged`, created 2026-03-05, merged 2026-03-18; author `shiyu7`.
- Diff scope read: `1` files, `+1/-1`; areas: model wrapper; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: n/a. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20386 - perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe…

- Link: https://github.com/sgl-project/sglang/pull/20386
- Status/date: `merged`, created 2026-03-11, merged 2026-03-12; author `vedantjh2`.
- Diff scope read: `1` files, `+1/-2`; areas: model wrapper; keywords: config.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-2 (3 lines); hunks: import torch; def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20736 - [AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8

- Link: https://github.com/sgl-project/sglang/pull/20736
- Status/date: `merged`, created 2026-03-17, merged 2026-04-15; author `zhentaocc`.
- Diff scope read: `2` files, `+218/-8`; areas: model wrapper, MoE/router; keywords: config, cuda, expert, moe, deepep, fp8, quant, router, topk, triton.
- Code diff details:
  - `python/sglang/srt/models/qwen2_moe.py` modified +108/-5 (113 lines); hunks: ); from sglang.srt.utils import (; symbols: can_fuse_shared_expert, Qwen2MoeMLP, __init__, __init__
  - `python/sglang/srt/models/qwen3_5.py` modified +110/-3 (113 lines); hunks: LazyValue,; _is_npu = is_npu(); symbols: __init__, __init__, __init__, _get_num_fused_shared_experts
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: config, cuda, expert, moe, deepep, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20864 - [Perf]Remove H2D for Qwen3.5 SpecV2

- Link: https://github.com/sgl-project/sglang/pull/20864
- Status/date: `merged`, created 2026-03-18, merged 2026-03-31; author `Chen-0210`.
- Diff scope read: `2` files, `+17/-13`; areas: scheduler/runtime; keywords: spec, cache, eagle.
- Code diff details:
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +14/-8 (22 lines); hunks: def _compute_spec_mrope_positions(; symbols: _compute_spec_mrope_positions
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +3/-5 (8 lines); hunks: def prepare_for_v2_verify(; symbols: prepare_for_v2_verify
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`; keywords observed in patches: spec, cache, eagle. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21019 - [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel

- Link: https://github.com/sgl-project/sglang/pull/21019
- Status/date: `merged`, created 2026-03-20, merged 2026-03-23; author `yuan-luo`.
- Diff scope read: `3` files, `+597/-202`; areas: model wrapper, kernel; keywords: kv, triton, attention, config, cuda, cache, fp8, moe, processor, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +285/-65 (350 lines); hunks: import torch; RowParallelLinear,; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/jit_kernel/triton/gdn_fused_proj.py` added +310/-0 (310 lines); hunks: +from __future__ import annotations; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_qkvzba_split_reshape_cat_contiguous_kernel, fused_qkvzba_split_reshape_cat_contiguous
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-137 (139 lines); hunks: from typing import Any, Iterable, Optional, Set, Tuple; logger = logging.getLogger(__name__); symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, Qwen3GatedDeltaNet, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: kv, triton, attention, config, cuda, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21070 - [Qwen3.5] Fix broken pipeline parallelism layer splitting

- Link: https://github.com/sgl-project/sglang/pull/21070
- Status/date: `merged`, created 2026-03-21, merged 2026-03-21; author `alisonshao`.
- Diff scope read: `2` files, `+15/-20`; areas: model wrapper, tests/benchmarks; keywords: config, expert, moe, test.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +8/-15 (23 lines); hunks: ); def get_layer(idx: int, prefix: str):; symbols: get_layer, load_fused_expert_weights, load_fused_expert_weights
  - `test/registered/distributed/test_pp_single_node.py` modified +7/-5 (12 lines); hunks: def setUpClass(cls):; def run_gsm8k_test(self, pp_size):; symbols: setUpClass, run_gsm8k_test, run_gsm8k_test, run_gsm8k_test
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py`; keywords observed in patches: config, expert, moe, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21234 - [AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model

- Link: https://github.com/sgl-project/sglang/pull/21234
- Status/date: `merged`, created 2026-03-23, merged 2026-03-30; author `hubertlu-tw`.
- Diff scope read: `1` files, `+18/-0`; areas: model wrapper; keywords: config, cuda, expert, kv, moe, vision.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +18/-0 (18 lines); hunks: cpu_has_amx_support,; _is_cuda = is_cuda(); symbols: forward, Qwen3_5ForCausalLM, __init__, load_fused_expert_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: config, cuda, expert, kv, moe, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21347 - [Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model

- Link: https://github.com/sgl-project/sglang/pull/21347
- Status/date: `merged`, created 2026-03-24, merged 2026-04-01; author `edwingao28`.
- Diff scope read: `1` files, `+22/-0`; areas: model wrapper; keywords: config, expert.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +22/-0 (22 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; def load_fused_expert_weights(; symbols: load_weights, load_fused_expert_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: config, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21448 - [Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode

- Link: https://github.com/sgl-project/sglang/pull/21448
- Status/date: `merged`, created 2026-03-26, merged 2026-03-30; author `sufeng-buaa`.
- Diff scope read: `6` files, `+78/-8`; areas: model wrapper, attention/backend, scheduler/runtime, tests/benchmarks; keywords: cache, spec, attention, config, kv, mla, cuda, expert, lora, test.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +31/-1 (32 lines); hunks: from sglang.srt.layers.radix_attention import RadixAttention; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights, load_fused_expert_weights, load_weights, load_fused_expert_weights
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +17/-0 (17 lines); hunks: def _init_pools(self: ModelRunner):; def _init_pools(self: ModelRunner):; symbols: _init_pools, _init_pools, _init_pools
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +11/-5 (16 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +13/-0 (13 lines); hunks: def separate_deepstack_embeds(self, embedding):; symbols: separate_deepstack_embeds, start_layer, end_layer, pad_input_ids
  - `python/sglang/srt/disaggregation/decode.py` modified +4/-2 (6 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`, `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: cache, spec, attention, config, kv, mla. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`, `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21487 - feat(ci): add GB300 nightly benchmark test suites

- Link: https://github.com/sgl-project/sglang/pull/21487
- Status/date: `merged`, created 2026-03-26, merged 2026-03-29; author `Kangyan-Zhou`.
- Diff scope read: `11` files, `+874/-4`; areas: quantization, scheduler/runtime, tests/benchmarks; keywords: test, attention, cuda, eagle, spec, topk, flash, cache, fp4, kv.
- Code diff details:
  - `python/sglang/test/accuracy_test_runner.py` modified +296/-3 (299 lines); hunks: def _run_simple_eval(; def run_accuracy_test(; symbols: _run_simple_eval, _get_nemo_venv, _ensure_nemo_data_prepared, _run_nemo_skills_eval
  - `test/registered/gb300/test_deepseek_v32_nvfp4.py` added +82/-0 (82 lines); hunks: +import unittest; symbols: TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4
  - `test/registered/gb300/test_deepseek_v32.py` added +79/-0 (79 lines); hunks: +import unittest; symbols: TestDeepseekV32, test_deepseek_v32
  - `test/registered/gb300/test_qwen35_nvfp4.py` added +79/-0 (79 lines); hunks: +import unittest; symbols: TestQwen35Nvfp4, test_qwen35_nvfp4
  - `test/registered/gb300/test_qwen35_fp8.py` added +75/-0 (75 lines); hunks: +import unittest; symbols: TestQwen35Fp8, test_qwen35_fp8
- Optimization/support interpretation: The concrete diff surface is `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`; keywords observed in patches: test, attention, cuda, eagle, spec, topk. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21669 - [AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/21669
- Status/date: `merged`, created 2026-03-30, merged 2026-04-07; author `michaelzhang-ai`.
- Diff scope read: `6` files, `+408/-8`; areas: quantization, tests/benchmarks; keywords: test, attention, config, fp8, triton, cache, benchmark, moe.
- Code diff details:
  - `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py` added +139/-0 (139 lines); hunks: +"""Nightly performance benchmark for Qwen3.5-397B-A17B FP8.; symbols: generate_simple_markdown_report, TestNightlyQwen35Fp8Performance, setUpClass, test_bench_qwen35_fp8
  - `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` added +139/-0 (139 lines); hunks: +"""MI35x Nightly performance benchmark for Qwen3.5-397B-A17B FP8.; symbols: generate_simple_markdown_report, TestQwen35Fp8PerfMI35x, setUpClass, test_qwen35_fp8_perf
  - `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` modified +42/-1 (43 lines); hunks: import os; def setUpClass(cls):; symbols: setUpClass, setUpClass, tearDownClass, test_lm_eval
  - `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py` modified +36/-3 (39 lines); hunks: import os; def setUpClass(cls):; symbols: setUpClass, test_lm_eval, test_lm_eval
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +26/-2 (28 lines); hunks: jobs:; jobs:
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`; keywords observed in patches: test, attention, config, fp8, triton, cache. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21692 - [Bugfix] [NPU] Qwen3.5 with quantization fix

- Link: https://github.com/sgl-project/sglang/pull/21692
- Status/date: `merged`, created 2026-03-30, merged 2026-04-08; author `OrangeRedeng`.
- Diff scope read: `3` files, `+29/-42`; areas: model wrapper, quantization; keywords: config, moe, quant, vision, expert, kv, triton.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +25/-39 (64 lines); hunks: FusedMoEMethodBase,; def get_quant_method(; symbols: get_quant_method, get_quant_method, _get_scheme_from_parts, get_linear_scheme
  - `python/sglang/srt/models/qwen3_5.py` modified +3/-3 (6 lines); hunks: def forward(; def load_fused_expert_weights(; symbols: forward, Qwen3_5ForCausalLM, load_fused_expert_weights, Qwen3_5ForConditionalGeneration
  - `python/sglang/srt/model_loader/loader.py` modified +1/-0 (1 lines); hunks: def _get_quantization_config(; symbols: _get_quantization_config
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_loader/loader.py`; keywords observed in patches: config, moe, quant, vision, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_loader/loader.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21849 - [VLM]: allow Qwen3.5 models for encoder disaggregation

- Link: https://github.com/sgl-project/sglang/pull/21849
- Status/date: `merged`, created 2026-04-01, merged 2026-04-06; author `Ratish1`.
- Diff scope read: `4` files, `+190/-3`; areas: multimodal/processor, tests/benchmarks; keywords: moe, processor, config, cuda, scheduler, test.
- Code diff details:
  - `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0 (184 lines); hunks: # Omni model for local testing; override via env var EPD_OMNI_MODEL; def test_mmmu(self):; symbols: test_mmmu, TestEPDDisaggregationQwen35, setUpClass, start_encode
  - `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2 (5 lines); hunks: async def _process_mm_items(self, mm_items, modality):; symbols: _process_mm_items
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: def _handle_encoder_disaggregation(self):; symbols: _handle_encoder_disaggregation
- Optimization/support interpretation: The concrete diff surface is `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`; keywords observed in patches: moe, processor, config, cuda, scheduler, test. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22145 - [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support)

- Link: https://github.com/sgl-project/sglang/pull/22145
- Status/date: `merged`, created 2026-04-05, merged 2026-04-07; author `YAMY1234`.
- Diff scope read: `1` files, `+20/-8`; areas: misc; keywords: cache, config, kv, mla.
- Code diff details:
  - `python/sglang/srt/disaggregation/nixl/conn.py` modified +20/-8 (28 lines); hunks: def send_kvcache_slice(; def add_transfer_request(; symbols: send_kvcache_slice, add_transfer_request, add_transfer_request
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/disaggregation/nixl/conn.py`; keywords observed in patches: cache, config, kv, mla. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/disaggregation/nixl/conn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22240 - [Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5)

- Link: https://github.com/sgl-project/sglang/pull/22240
- Status/date: `merged`, created 2026-04-07, merged 2026-04-07; author `YAMY1234`.
- Diff scope read: `1` files, `+143/-2`; areas: misc; keywords: kv, spec.
- Code diff details:
  - `python/sglang/srt/disaggregation/nixl/conn.py` modified +143/-2 (145 lines); hunks: class KVArgsRegisterInfo:; def from_zmq(cls, msg: List[bytes]):; symbols: KVArgsRegisterInfo:, from_zmq, from_zmq, from_zmq
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/disaggregation/nixl/conn.py`; keywords observed in patches: kv, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/disaggregation/nixl/conn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22312 - Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B

- Link: https://github.com/sgl-project/sglang/pull/22312
- Status/date: `merged`, created 2026-04-08, merged 2026-04-10; author `cs-cat`.
- Diff scope read: `3` files, `+272/-8`; areas: attention/backend, tests/benchmarks; keywords: attention, triton, cache, config, cuda, test.
- Code diff details:
  - `test/registered/attention/test_gdn_noncontiguous_stride.py` added +255/-0 (255 lines); hunks: +"""; symbols: _make_noncontiguous_ab, TestFusedGdnGatingNonContiguous, _run_test, test_small
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +9/-6 (15 lines); hunks: def fused_sigmoid_gating_delta_rule_update_kernel(; def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update
  - `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` modified +8/-2 (10 lines); hunks: def fused_gdn_gating_kernel(; def fused_gdn_gating_kernel(; symbols: fused_gdn_gating_kernel, fused_gdn_gating_kernel, fused_gdn_gating, fused_gdn_gating
- Optimization/support interpretation: The concrete diff surface is `test/registered/attention/test_gdn_noncontiguous_stride.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`; keywords observed in patches: attention, triton, cache, config, cuda, test. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/attention/test_gdn_noncontiguous_stride.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

### PR #22431 - Fix Qwen3.5 video processing when passing video_data in "processor_output" format

- Link: https://github.com/sgl-project/sglang/pull/22431
- Status/date: `merged`, created 2026-04-09, merged 2026-04-18; author `lkhl`.
- Diff scope read: `1` files, `+1/-1`; areas: multimodal/processor; keywords: processor.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: async def preprocess_video(; symbols: preprocess_video
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`; keywords observed in patches: processor. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/multimodal/processors/qwen_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22493 - Add MambaPool kvcache offloading during retraction

- Link: https://github.com/sgl-project/sglang/pull/22493
- Status/date: `merged`, created 2026-04-10, merged 2026-04-22; author `hlu1`.
- Diff scope read: `5` files, `+193/-16`; areas: scheduler/runtime, tests/benchmarks; keywords: cache, kv, test, attention, cuda, mla, scheduler, triton.
- Code diff details:
  - `test/registered/unit/mem_cache/test_mamba_unittest.py` modified +123/-0 (123 lines); hunks: def make_dummy_req():; symbols: make_dummy_req, test_mamba_pool_cpu_offload, test_hybrid_kv_pool_cpu_offload, test_insert_prev_prefix_len
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +43/-6 (49 lines); hunks: def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:; def set_kv_buffer(; symbols: fork_from, get_cpu_copy, load_cpu_copy, get_contiguous_buf_infos
  - `python/sglang/srt/mem_cache/allocator.py` modified +8/-8 (16 lines); hunks: def free(self, free_index: torch.Tensor):; def clear(self):; symbols: free, get_cpu_copy, get_cpu_copy, load_cpu_copy
  - `python/sglang/srt/managers/scheduler.py` modified +11/-0 (11 lines); hunks: def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:; def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch; symbols: update_running_batch, update_running_batch
  - `python/sglang/srt/managers/schedule_batch.py` modified +8/-2 (10 lines); hunks: def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):; symbols: offload_kv_cache, load_kv_cache, log_time_stats
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py`; keywords observed in patches: cache, kv, test, attention, cuda, mla. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22908 - [AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict.

- Link: https://github.com/sgl-project/sglang/pull/22908
- Status/date: `merged`, created 2026-04-15, merged 2026-04-21; author `ChangLiu0709`.
- Diff scope read: `1` files, `+14/-4`; areas: misc; keywords: cache, scheduler, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +14/-4 (18 lines); hunks: def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache, _handle_sampling_backend
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: cache, scheduler, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22913 - test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6

- Link: https://github.com/sgl-project/sglang/pull/22913
- Status/date: `merged`, created 2026-04-15, merged 2026-04-17; author `alisonshao`.
- Diff scope read: `4` files, `+184/-247`; areas: model wrapper, quantization, kernel, tests/benchmarks; keywords: cuda, test, attention, config, fp4, quant, scheduler, eagle, flash, spec.
- Code diff details:
  - `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-245 (245 lines); hunks: -import unittest; symbols: TestQwen35FP4, test_gsm8k, TestQwen35FP4MTP, setUpClass
  - `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` added +105/-0 (105 lines); hunks: +import unittest; symbols: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k
  - `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` added +77/-0 (77 lines); hunks: +import unittest; symbols: TestQwen35FP4, test_gsm8k
  - `.github/workflows/pr-test.yml` modified +2/-2 (4 lines); hunks: jobs:; jobs:
- Optimization/support interpretation: The concrete diff surface is `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`; keywords observed in patches: cuda, test, attention, config, fp4, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22948 - [AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled

- Link: https://github.com/sgl-project/sglang/pull/22948
- Status/date: `merged`, created 2026-04-16, merged 2026-04-16; author `mqhc2020`.
- Diff scope read: `1` files, `+17/-1`; areas: model wrapper, MoE/router; keywords: config, deepep, expert, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen2_moe.py` modified +17/-1 (18 lines); hunks: def can_fuse_shared_expert(; def can_fuse_shared_expert(; symbols: can_fuse_shared_expert, can_fuse_shared_expert, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2_moe.py`; keywords observed in patches: config, deepep, expert, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23034 - docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs

- Link: https://github.com/sgl-project/sglang/pull/23034
- Status/date: `merged`, created 2026-04-17, merged 2026-04-17; author `zijiexia`.
- Diff scope read: `73` files, `+2214/-215`; areas: model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, docs/config; keywords: doc, spec, attention, config, cuda, cache, moe, quant, eagle, expert.
- Code diff details:
  - `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunks: +---
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunks: +---
  - `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0 (299 lines); hunks: +---; symbols: per_token_group_quant_8bit, add
  - `docs_new/docs/advanced_features/server_arguments.mdx` modified +241/-45 (286 lines); hunks: Please consult the documentation below and [server_args.py](https://github.com/s; Please consult the documentation below and [server_args.py](https://github.com
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` added +219/-0 (219 lines); hunks: +export const Qwen36Deployment = () => {
- Optimization/support interpretation: The concrete diff surface is `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx`; keywords observed in patches: doc, spec, attention, config, cuda, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23467 - fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert

- Link: https://github.com/sgl-project/sglang/pull/23467
- Status/date: `merged`, created 2026-04-22, merged 2026-04-22; author `mickqian`.
- Diff scope read: `1` files, `+31/-4`; areas: quantization; keywords: config, fp8, kv, moe, quant.
- Code diff details:
  - `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunks: def __getattr__(self, name):; def is_layer_skipped(; symbols: __getattr__, _module_path_match, names, is_layer_skipped
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/utils.py`; keywords observed in patches: config, fp8, kv, moe, quant. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- Link: https://github.com/sgl-project/sglang/pull/23474
- Status/date: `open`, created 2026-04-22; author `kawaruko`.
- Diff scope read: `2` files, `+284/-8`; areas: tests/benchmarks; keywords: attention, cache, cuda, spec, test.
- Code diff details:
  - `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: +"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).; symbols: _TiedChild, __init__, forward, _TiedParent
  - `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: import logging; def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:; symbols: maybe_offload_to_cpu, maybe_offload_to_cpu, forward
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`; keywords observed in patches: attention, cache, cuda, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
