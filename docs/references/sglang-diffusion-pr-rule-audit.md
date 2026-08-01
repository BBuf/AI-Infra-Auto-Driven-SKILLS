# SGLang Diffusion Historical Optimization Rule Audit

## Method and boundary

This dossier backs `sgl-engine-sglang-diffusion/knowledge/history-rules.toml`.
The repository scan covered all 1,001 PR-numbered commits touching
`python/sglang/multimodal_gen` through SGLang commit
`fd96a35fb087a30e55a17bca28b198028b689f5f` (2026-08-01). Subject and path
classification found 365 unique performance-related PRs across residency,
compile/graph, kernel/fusion/layout, parallel communication, VAE/output, exact
reuse, runtime/I/O, precision, and sparse/approximate families. Categories
overlap. That broad scan establishes search-space coverage; only the thirteen PRs
below are encoded as executable hypotheses because their full merge diffs were
read and their generalization boundaries were recorded.

For every card, GitHub PR metadata was fetched on 2026-08-01 and the complete
merge diff was read from the local SGLang object database. “Diff lines” is
`git show --format= --no-ext-diff <merge_commit> | wc -l`. A historical memory
cutoff, model flag, shape, or speedup is never an acceptance rule. It can only
seed a candidate that must pass the active profile, frozen workload, lane
evidence, and integrated remeasurement.

Lossless here means unchanged logical model work and an implementation-level
equivalence argument. Backend accumulation order may move outputs. Precision
reduction, approximate cache, sparse attention, progressive resolution, or
reduced tokens remain quality-gated even when an old PR reports good images.

## Residency and memory lifetime

### PR #21248 — skip automatic Wan/MOVA layerwise offload on high-memory GPUs

- PR: [sgl-project/sglang#21248](https://github.com/sgl-project/sglang/pull/21248)
- State/merge: MERGED, 2026-03-25 10:45:30 UTC
- Merge commit: `e4ad10520b8d409c6d32079a9c46ec7bdc0463ed`
- Size: +54/-8, 1 file, 95 full-diff lines
- Reviewed file: `runtime/server_args.py`

Motivation: H200 layerwise offload traded memory for large latency regressions:
Wan2.2 A14B 4.22→6.77 s, Wan2.2 5B 5.91→9.86 s, and MOVA 3.12→5.62 s.
The implementation inspected total memory and disabled the automatic Wan/MOVA
offload default above a documented 130 GiB cutoff while preserving explicit
flags and lower-memory behavior.

Representative diff excerpt:

```python
if device_total_memory_gb >= WAN_LAYERWISE_OFFLOAD_AUTO_DISABLE_MEM_GB:
    auto_enable_layerwise_offload = False
    self.dit_layerwise_offload = False
```

Validation in the PR included no-offload peak-reserved measurements up to about
127.2 GiB and the three full workload timings above. Safe generalization: high
VRAM should route a residency experiment. Unsafe generalization: copying 130
GiB, disabling every offload component, or ignoring concurrent free memory,
compile peaks, FSDP, Cache-DiT, and model stage lifetime.

### PR #31538 — partial DiT resident layers

- PR: [sgl-project/sglang#31538](https://github.com/sgl-project/sglang/pull/31538)
- State/merge: MERGED, 2026-07-29 08:52:56 UTC
- Merge commit: `227dadd79a17dc16c2a5e0c33ad7ea68f1e7093f`
- Size: +240/-5, 4 files, 413 full-diff lines
- Reviewed files: `component_manager.py`, `layerwise_offload.py`, split
  `server_args.py`, and `test_layerwise_offload.py`

Motivation: pure layerwise streaming retransfers every layer at each denoise
step, while a large prefetch window can consume similar VRAM without retaining
layers across steps. The change added `resident_layers`, armed residency at the
first denoise forward, retained leading layers across steps, and force-released
them at stage teardown. It also prevented a large resident set from being
prefetched during a prior component stage.

Representative diff excerpt:

```python
if not force and layer_idx < self._retained_layers:
    return
```

The PR reported byte-identical output and showed Qwen-Image on RTX 5090 improve
from 56.60 s pure streaming to 33.97 s with a measured resident subset; FLUX
also improved from 16.96 s to 13.06 s. Safe generalization: sweep a
profile-justified resident-layer count when full residency does not fit. The
layer count is model/shape/GPU-specific and must not be copied. Partial
residency requires layerwise offload and is not equivalent to prefetch depth.

### PR #25457 — memory-aware component load order

- PR: [sgl-project/sglang#25457](https://github.com/sgl-project/sglang/pull/25457)
- State/merge: MERGED, 2026-05-17 05:22:55 UTC
- Merge commit: `c1d9e37a52e6d09e3db239a5388bc198fcc9b255`
- Size: +404/-13, 6 files, 526 full-diff lines
- Reviewed files: `component_loading_order.py`, `component_manager.py`,
  `component_resident_strategies.py`, `composed_pipeline_base.py`, and both
  associated unit-test modules

Motivation: loading a large DiT after small helpers can hit a startup peak even
when the final pipeline placement is viable. The implementation inferred
checkpoint payload sizes from safetensors/indexes, fell back to component risk
classes, kept stable tie-breaking, and loaded large/high-risk components first.

Representative diff excerpt:

```python
return sorted(component_specs, key=lambda spec: (
    -(infer_component_weight_size_bytes(spec.component_model_path) or 0),
    component_load_risk_rank(spec.load_module_name),
    spec.index,
))
```

Validation was unit coverage for size inference, aliases, numbered variants,
stable ordering, and FSDP handling. Safe generalization: inspect load/release
lifetime when startup peak blocks a candidate. This is a feasibility rule, not
a load-excluded E2E speed claim; it must not be credited as latency speedup.

### PR #29862 — transient offload during compile

- PR: [sgl-project/sglang#29862](https://github.com/sgl-project/sglang/pull/29862)
- State/merge: MERGED, 2026-07-02 11:39:20 UTC
- Merge commit: `119b76567daf36af7cc27ad062f9500d30a800ea`
- Size: +156/-8, 7 files, 343 full-diff lines
- Reviewed files: denoising stage, Hunyuan3D/Ideogram stage adaptations,
  progressive Ideogram stage, server args, warmup logging, and server tests

Motivation: max-autotune compilation had a much larger transient peak than
resident inference. The change layerwise-offloaded DiT weights for compile
warmup, temporarily moved resident non-DiT components to CPU, and restored both
components and the user-requested steady-state DiT placement before real work.
It explicitly skipped user layerwise offload, FSDP, Cache-DiT, no-warmup, and
subclasses whose forward could not run the restoration.

Representative diff excerpt:

```python
for module in self._offloaded_dit_modules_for_compile:
    module.disable_offload()
self._offloaded_dit_modules_for_compile.clear()
```

FLUX.2-dev TP=2 max-autotune went from OOM at about 123 GB/GPU to fitting at
about 41 GB/GPU on 2×H200, then served resident at about 12.8 s. A Z-Image
steady-state check was 1.082 versus 1.078 s. Safe generalization: model compile
peak and steady-state placement separately. Never leave transient offload
enabled during the scored inference placement.

## Compile, graph, VAE, and output

### PR #32696 — regional torch compile

- PR: [sgl-project/sglang#32696](https://github.com/sgl-project/sglang/pull/32696)
- State/merge: MERGED, 2026-07-29 13:57:06 UTC
- Merge commit: `917e900d4d53da6b35f67e0dd0c494ac8107c65a`
- Size: +191/-5, 4 files, 249 full-diff lines
- Reviewed files: denoising stage, split server args, `torch_compile.py`, and
  `test_regional_torch_compile.py`

Motivation: whole-DiT compile can create oversized graphs and long startup,
while repeated blocks provide stable boundaries. The opt-in path uses each
model's `_compile_conditions`, fails if no region matches, compiles matching
submodules once, and leaves eager/whole-model defaults unchanged.

Representative diff excerpt:

```python
conditions = getattr(module, "_compile_conditions", ())
matches = [submodule for name, submodule in module.named_modules()
           if name and any(condition(name, submodule) for condition in conditions)]
```

On H100 LTX-2.3, 48-block regional compile reduced warmup 42.15→24.20 s and
two-step denoise 2261.95→1248.59 ms versus whole-transformer compile. CI and
unit suites passed. Safe generalization: route regional compile when the locked
model declares tested regions; compilation once is not enough to close graph,
VAE/output, or scheduler/runtime search families.

### PR #27431 — layout-preserving LTX-2 VAE decode

- PR: [sgl-project/sglang#27431](https://github.com/sgl-project/sglang/pull/27431)
- State/merge: MERGED, 2026-06-09 15:26:40 UTC
- Merge commit: `aa18a68ac52cd3d4cb56bac551be4e43fe0d7516`
- Size: +223/-49, 4 files, 368 full-diff lines
- Reviewed files: VAE loader, `ltx_2_vae.py`, channels-last VAE tests, and VAE
  loader tests

Motivation: Hopper Conv3d favored NDHWC, but repeat-plus-concatenate temporal
padding converted the input back to NCDHW before every convolution. The change
allocated padded tensors directly in `channels_last_3d`, copied edge frames
without an intermediate conversion, and enabled the layout for single-GPU
LTX-2 while preserving the original path when weights use another layout.

Representative diff excerpt:

```python
out = torch.empty(shape, dtype=x.dtype, device=x.device,
                  memory_format=torch.channels_last_3d)
out[:, :, left:left + t].copy_(x)
```

H100 Conv3d microbenchmarks improved 31.3→8.5 ms; warm E2E decode improved
5.41→3.84 s and peak reserved fell 71.81→62.12 GiB. Tests cover causal,
non-causal, cached, edge-replication, and loader gates. Safe generalization:
propagate a profitable format through the full VAE path. An isolated conversion
or a layout unsupported by the target backend may regress. Lowering VAE
precision remains quality-gated.

### PR #32784 — CUDA video output finalization

- PR: [sgl-project/sglang#32784](https://github.com/sgl-project/sglang/pull/32784)
- State/merge: MERGED, 2026-07-29 14:04:21 UTC
- Merge commit: `22151edca162da06d913478c96bd8e3db84f4380`
- Size: +514/-13, 2 files, 608 full-diff lines
- Reviewed files: `runtime/entrypoints/utils.py` and `test_output_saving.py`

Motivation: video output paid pageable CPU staging and, with audio, separate
encode/remux passes. The change added a shape-aware cached CUDA-registered
memfd, pinned fallback materialization, one-pass H.264/AAC, bounded x264
threads, a 1 GiB cache cap, cleanup, and compatibility fallback.

Representative diff excerpt:

```python
buffer.tensor.copy_(frames, non_blocking=True)
torch.cuda.current_stream(frames.device).synchronize()
subprocess.run(command, pass_fds=(buffer.fd,), check=True)
```

A representative 124-frame 1344×768 output improved 1.591→0.621 s (2.56×),
and the output was fully decoded for validation; 19 output tests passed. Safe
generalization: profile finalization inside the actual timing scope and verify
container, codecs, dimensions, frame count, duration, audio, and fallback.

## Exact reuse, synchronization, and frozen-topology communication

### PR #29755 — exact Helios cross-attention K/V reuse

- PR: [sgl-project/sglang#29755](https://github.com/sgl-project/sglang/pull/29755)
- State/merge: MERGED, 2026-07-06 20:56:37 UTC
- Merge commit: `ca73c7705592e2c175c2a5d53f1b195a364bd95b`
- Size: +90/-12, 1 file, 169 full-diff lines
- Reviewed file: `runtime/models/dits/helios.py`

Motivation: prompt conditioning remained constant while text projection and
every block's cross-attention K/V projection repeated across steps/chunks. The
change split `project_kv`, cached one list per request on `forward_batch.extra`,
keyed it by tensor identity/shape/stride/dtype/device, and disabled caching when
gradients or request storage were unavailable.

Representative diff excerpt:

```python
key = self._tensor_key(encoder_hidden_states) if cache is not None else None
kvs = cache.get(key) if key is not None else None
```

Ascend 910B×4 SP=4 measurements reduced E2E 517.5→512.3 s and denoise
514.9→509.8 s. Safe generalization: exact reuse needs a proof that all cache-key
inputs, lifetime, and invalidation conditions are complete. Similar-looking
cross-step reuse without that proof belongs to an approximate/quality gate.

### PR #27440 — remove UniPC scheduler host synchronization

- PR: [sgl-project/sglang#27440](https://github.com/sgl-project/sglang/pull/27440)
- State/merge: MERGED, 2026-06-06 14:01:41 UTC
- Merge commit: `7c6f9542c711133b5021c5fb9a7e200d60690ca8`
- Size: +6/-6, 1 file, 44 full-diff lines
- Reviewed file: `runtime/models/schedulers/scheduling_unipc_multistep.py`

Motivation: constructing a tensor from a list of GPU scalar tensors introduced
`aten::item`, `_local_scalar_dense`, and `cudaStreamSynchronize`. The change
stacked already resident scalars and constructed the constant one on device.

Representative diff excerpt:

```python
rks.append(torch.ones((), dtype=h.dtype, device=h.device))
rks = torch.stack(rks).to(device=device)
```

Cosmos3 H200 warm E2E improved 57457.98→56509.18 ms with unchanged 43792 MB
peak; 37 unit tests and 6 subtests passed. Safe generalization: profile scheduler
scalar extraction and sync before rewriting. Preserve dtype/device and the
algorithm's coefficient construction exactly.

### PR #31854 — two-rank CUDA-IPC zero-staging all-to-all

- PR: [sgl-project/sglang#31854](https://github.com/sgl-project/sglang/pull/31854)
- State/merge: MERGED, 2026-07-31 11:35:49 UTC
- Merge commit: `754b692afc2948137c4315989481e406acae0159`
- Size: +779/-8, 9 files, 960 full-diff lines
- Reviewed files: diffusion envs, base communicator, new `ipc_a2a.py`, attention
  layer, `usp.py`, GPU worker, Qwen-Image DiT, GPU test registry, and the new
  two-GPU parity test

Motivation: at Ulysses degree two, separate Q/K/V and output NCCL all-to-alls
paid rendezvous plus layout/staging around each transfer. The implementation
mapped peer CUDA IPC buffers, used bounded GPU-side sequence synchronization,
combined Q/K/V exchange, wrote segmented joint attention directly into the
consumption layout, capped shape buffers, checked timeouts at request
boundaries, and fell back when the narrow gate failed.

Representative diff excerpt:

```python
if world_size == 2 and scatter_dim in (1, 2):
    fast = _ipc_all_to_all_4d(group, input_, scatter_dim)
    if fast is not None:
        return fast
```

On 2×H100, Qwen-Image improved 4.41→3.95 s across the feature ladder and FLUX
improved about 8.4%; the two-rank parity test requires bitwise equality and
positive IPC engagement. The PR's rank-scaling experiment found the analogous
four-rank design 25–45% slower. Safe generalization: keep baseline ranks and
parallel degrees frozen and try the transport only for its exact two-rank,
same-node, peer-access, shape, capture, and watchdog contract. Never infer an
N-rank rule or change topology to obtain the fast path.

## Quality-gated cache, precision, and reduced-token rules

### PR #25328 — mount Cache-DiT before torch.compile

- PR: [sgl-project/sglang#25328](https://github.com/sgl-project/sglang/pull/25328)
- State/merge: MERGED, 2026-05-15 10:08:40 UTC
- Merge commit: `20123e0b165556cf2b8ec05fac866a3bd2a29ccd`
- Size: +33/-11, 1 file, 98 full-diff lines
- Reviewed file: `runtime/pipelines_core/stages/denoising.py`

Motivation: compile warmup traced the unwrapped transformer, then the first real
request mounted Cache-DiT and compiled the actual path again. The change made
cache and compile installation state explicit, deferred compile until cache was
mounted, preserved ordinary non-compile warmup behavior, and installed each
compiled module once.

Representative diff excerpt:

```python
self._maybe_enable_cache_dit(num_inference_steps, batch)
for transformer in filter(None, [self.transformer, self.transformer_2]):
    self._maybe_enable_torch_compile(transformer)
```

Across five H200 runs, first-real-request denoise fell 5.2370→2.9449 s; client
latency including request-based warmup improved about 2.89%. Safe
generalization: order interacting wrappers before tracing and measure the
frozen timing scope. Cache-DiT remains approximate and must pass its engagement,
LPIPS, visual, and integrated quality gates.

### PR #32697 — decode-only Wan VAE BF16

- PR: [sgl-project/sglang#32697](https://github.com/sgl-project/sglang/pull/32697)
- State/merge: MERGED, 2026-07-29 13:57:51 UTC
- Merge commit: `4f5b50c576e8118614174d5762c6818709227f58`
- Size: +99/-12, 10 files, 298 full-diff lines
- Reviewed files: base/LongLive2/Wan pipeline configs, model HTTP metadata,
  decoding stage, precision utility, and four unit-test modules

Motivation: BF16 accelerated Wan VAE decode, but lowering global VAE precision
also changed image-to-video encoding. The implementation added a decode-only
policy, resolved it at residency and execution use-sites, kept encode precision
unchanged, exposed the resolved policy, defaulted Wan2.1/LongLive2 appropriately,
and kept standard Wan2.2 FP32 for its ModelOpt quality boundary.

Representative diff excerpt:

```python
decode_precision = getattr(pipeline_config, "vae_decode_precision", None)
if decode_precision is not None:
    return precision_to_dtype(decode_precision, "vae_decode_precision")
```

H100 Wan2.1 decode improved 441.08→326.00 ms (1.353×), with reported PSNR,
SSIM, MS-SSIM, CLIP, and NVIDIA diffusion CI. Safe generalization: search the
smallest stage-specific precision boundary and gate quality. Do not copy
Wan2.1's BF16 default to Wan2.2 or another checkpoint family.

### PR #27736 — Ideogram 4 progressive resolution

- PR: [sgl-project/sglang#27736](https://github.com/sgl-project/sglang/pull/27736)
- State/merge: MERGED, 2026-06-11 15:16:53 UTC
- Merge commit: `7f57b344c9e1caa65505e72f6f2029591caf5ad7`
- Size: +886/-25, 7 files, 1,056 full-diff lines
- Reviewed files: progressive-resolution documentation, Ideogram pipeline,
  progressive base/FLUX/Ideogram/Qwen stages, and progressive unit tests

Motivation: early coarse latent grids reduce quadratic attention work, but
Ideogram 4 has conditional and unconditional transformers plus resolution-bound
positions, masks, indicators, LLM features, cache context, and schedule state.
The implementation routed full-resolution at zero extra work, added the
model-specific pack/unpack and transition rebuild, refreshed both transformers,
and documented SP incompatibility.

Representative diff excerpt:

```python
self._refresh_cache_dit_context(n_remaining, _get_scm_preset())
```

RTX A6000 denoise improved 1.24–1.56× for 20 steps and 1.19–1.54× for 48 steps
across transition deltas, with visual comparisons and transition unit tests.
Safe generalization: route a quality-gated schedule search only when the locked
model implements every resolution-dependent state transition. Never copy
spectrum constants, delta, stage split, or a sequence-parallel assumption.

## Flow rules derived from the audit

1. Route component placement/offload to a default lossless residency lane.
2. Measure per-GPU free/peak/safety memory and transfers; never accept a total
   VRAM cutoff alone.
3. Treat partial DiT residency, prefetch depth, transient compile placement,
   load order, and auxiliary-component residency as separate hypotheses.
4. Search compiler/graph, VAE/decode/output, scheduler/exact reuse, and
   communication/layout inside the complete load-excluded E2E kernel lane.
5. Route Cache-DiT/compile ordering, decode-only precision, and progressive
   resolution to their existing quality-gated lanes with active-lane history.
6. Preserve the baseline's GPUs, rank map, and parallel degrees throughout.
7. Require active engagement, fallback accounting, equivalence reasoning, and
   a full frozen-workload measurement for every retained point.
8. Register compatible positive candidates and remeasure their cumulative
   stack; no single failed rule closes a lane or proves the integrated target
   unreachable.
