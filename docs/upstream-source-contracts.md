# Upstream source contracts

Source inspection: **2026-11-08**. These frozen heads support the specific
contracts below, not a full audit of every upstream model or a GPU benchmark.
Record the deployed image digest, framework commit, CUDA/Triton/FlashInfer
versions and actual dispatch before applying a source-level conclusion.

| Source | Ref | Inspected commit |
|---|---|---|
| SGLang | `main` | [`c24285bde97c`](https://github.com/sgl-project/sglang/commit/c24285bde97c3e6424cd5d8cb35788889034d8ea) |
| vLLM | `main` | [`b022bb7a4873`](https://github.com/vllm-project/vllm/commit/b022bb7a4873a47cdcdf474edf3bd499c221f40d) |
| TensorRT-LLM | `main` | [`b433473ba0d0`](https://github.com/NVIDIA/TensorRT-LLM/commit/b433473ba0d0d49764f4d2fb3d45c82716b946dc) |
| TokenSpeed | `main` | [`7b21034e72bb`](https://github.com/lightseekorg/tokenspeed/commit/7b21034e72bb45d1edef3d3b5483b50b58c4c492) |
| FlashInfer | `main` | [`31c146b16789`](https://github.com/flashinfer-ai/flashinfer/commit/31c146b16789f112e9b9861bd79247d4701e437a) |

The [October audit](refresh-2026-10-05.md), model PR dossiers and GPU experiment
tables keep their original dates. The deleted SGLang `dsv4.1` branch at
[`f238b3cae10d`](https://github.com/sgl-project/sglang/commit/f238b3cae10db1575a632855fc69b2d1fe003142)
is historical evidence; current V4.1 code lives on main.

## Profiler control and output

| Framework | Current contract | Capture consequence |
|---|---|---|
| SGLang | [SchedulerProfilerManager](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/managers/scheduler_components/profiler_manager.py) is v1/default. Opt-in v2 moved to [observability/profile_utils.py](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/observability/profile_utils.py); it remains stage-auto only, with `profile_by_stage=true`, no `start_step`, merge or manual stop. | The helper targets v1. `with_stack` defaults true; `record_shapes` defaults false in both v1 and v2. Set shapes explicitly for an attribution capture that needs them. Label `profile_id` for distributed merging. The staged helper returns a unique capture directory, excluding prior runs under the same root. `SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS` can defer native stage capture until sufficient admission; check the actual batch. |
| vLLM | [StartProfileRequest](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/entrypoints/serve/profile/api_router.py) accepts optional `profile_prefix`, `delay_iterations`, `max_iterations`, `profiler_kwargs`. Routes still require a configured profiler. [ProfilerConfig](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/config/profiler.py) distinguishes iteration limits from warmup/wait/active schedules. | The helper negotiates prefix/limit fields through the running server's OpenAPI and sends only advertised overrides. Older or OpenAPI-disabled servers use bodyless launch-configured control. Output directory remains launch-configured. `capture_torch_profiler` is deprecated and has no effect; model runner V1 was removed. CPU-only frontend traces are not GPU evidence. |
| TensorRT-LLM | [StartProfileRequest](https://github.com/NVIDIA/TensorRT-LLM/blob/b433473ba0d0d49764f4d2fb3d45c82716b946dc/tensorrt_llm/serve/openai_protocol.py) supports `output_dir`, `num_steps`, `start_step`, `activities`. [PyExecutorProfileManager](https://github.com/NVIDIA/TensorRT-LLM/blob/b433473ba0d0d49764f4d2fb3d45c82716b946dc/tensorrt_llm/_torch/pyexecutor/profiling.py) claims a pending window atomically, rejects overlapping starts, and starts/stops Kineto on the executor thread. | The helper sends native directory and CPU/GPU activities. No rank0 patch is needed. The inspected profiler still omits `with_stack`; unknown Python attribution stays unknown or needs a recorded mapping build. Do not use parallel capture clients on one engine. |
| TokenSpeed | [worker Profiler](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/engine/profiler.py) is driven by the Rust host's forward thread; `note_forward` counts committed forwards. [Rust start_profile dispatcher](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-engine/crates/ts-control/src/calls.rs) translates known options into [ProfileReq](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-engine/crates/ts-wire/src/control.rs) and rejects unknown fields. | The old Python `request_handler`, `scheduler_control_client`, and `control_server` paths are gone. [serve orchestration](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/cli/serve_smg.py) fronts the headless engine with external pinned SMG wheels. Verify the installed gateway's HTTP forwarding; do not infer an old `port + 1` control sidecar. |

SGLang's [Rust HTTP ingress](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/rust/sglang-server/src/api_server/http/info.rs)
now implements GET/POST start/stop and fans control out to scheduler ranks.
[Rust authorization](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/rust/sglang-server/src/api_server/http/auth.rs)
uses the admin key for profile routes when configured, otherwise the API key;
health/ready/metrics probes remain open. Older Rust builds can return 404.
The helper no longer rejects an ingress merely because `frontend=rust`.

TokenSpeed's current worker validates profiler prerequisites before changing
state: AMD GPU/Proton requires `HSA_TOOLS_DISABLE_REGISTER=0`; Proton cannot
combine with GPU/CUDA_PROFILER and AMD Proton uses `ROCR_VISIBLE_DEVICES`,
not `HIP_VISIBLE_DEVICES`. Torch CPU/GPU requires process-global
`profile_all_threads` support because inference runs on a Rust-owned thread.
`EXPERT_LOAD` requires counters/device and refuses online EPLB. Native Proton,
MEM and expert-load artifacts are not Chrome traces.

Trace names remain framework-specific:

- SGLang: `[prefix-]<profile_id>-TP-<t>[-DP-<d>][-PP-<p>][-EP-<e>][-EXTEND|-DECODE].trace.json.gz`.
- vLLM workers: `[prefix_]dp<d>_pp<p>_tp<t>_dcp<c>_ep<e>_rank<g>.<ts>.pt.trace.json.gz`;
  frontend: `<host>_<pid>.async_llm.<ts>.pt.trace.json.gz`.
- TensorRT-LLM: `trtllm-trace-<id>-rank-<rank>.json`.
- TokenSpeed: `<profile_id>-[DP<d>-]TP<t>[-EXTEND|-DECODE].trace.json.gz`.

A successful response does not prove a fresh GPU trace exists. Verify nonzero
GPU events, actual phases, ranks and warmup. Read from a shared mount or copy
artifacts back explicitly; never substitute an older trace. Keep all ranks
for collective/skew diagnosis.

## DS V4.1 dispatch and numerical boundaries

[deepseek_v4_mhc.py](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/deepseek_v4_mhc.py)
owns current mHC kernel dispatch;
[deepseek_v4_boundary.py](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/deepseek_v4_boundary.py)
connects it to the general layer-boundary flow. V4.1 carries previous-sublayer
pre coefficients; V4's fused post+pre formula is not interchangeable. Current
V4.1 nevertheless fuses post, next combine, norm, quantization and sometimes
next coefficient statistics. Read the
[mHC dispatch matrix](../skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md#residual-mixing-and-collective-epilogues)
for row/platform/consumer guards and state invalidation.

The [merged mHC refactor review](../skills/llm-torch-profiler-analysis/references/dsv41-mhc-source-review.md)
traces #42245 through the current boundary transport. The old combine-only
kernels and `MhcPostFusion` context hook are deleted; later TP2 support, cached
next statistics and decode/Mega-prefill fusions have their own current gates.
The [Hopper indexer review](../skills/llm-torch-profiler-analysis/references/dsv41-hopper-indexer-review.md)
separates the 1 GiB prefill score budget, fused 32-head epilogue and live-prefix
paged decode from pending dense/compact scoring proposals.

[WO-A dispatch](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/deepseek_v4.py)
selects GEMV, small-batch GEMM or token-major BMM using phase, shape, strides,
platform and consumer format. Target and DSpark draft need separate checks;
request concurrency does not equal kernel M. Preserve the intermediate BF16
rounding and consumer's MXFP8 scale layout when moving quantization upstream.

[Main defaults](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/environ.py) still set
`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=False`, `SGLANG_DSV4_KV_LAYOUT=v4`, and
`SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=False`. The historical dsv4.1 branch's
fused-finalize default was True; that accuracy evidence does not transfer to a
different ref or dispatch. The [paged KV contract](../skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md#paged-kv-storage-contract)
retains its October source audit: 528 B/288 B include payload/scales per stored
token before page padding, not whole-model KV capacity.

The [TRT-LLM sparse-attention review](../skills/llm-torch-profiler-analysis/references/dsv41-trtllm-source-review.md)
traces three merged changes through current callers: uniform-FP8 KV stores,
decoder bounded replay with breakable prefill, and explicit persistent counters
through FlashInfer's public API. KV norm/RoPE/store is fused; Q still preserves
BF16 intermediate rounding before a separate FP8 conversion. Counter capacity
is bounded by query rows, including speculative rows, rather than requests.
Encoder bounded replay remains unsupported on this pinned mainline backend.

[Detailed step annotations](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/model_executor/step_span_utils.py)
carry context/generation sums of query lengths, KV lengths, squared query
lengths and query×KV lengths. Verify/Draft widths come from speculative shape,
not acceptance length. These aggregates help check an analytical template;
the bundled simulator does not automatically ingest them.

## Compiler and dependency dispatch

TokenSpeed's [merged V4.1 mHC review](../skills/llm-torch-profiler-analysis/references/tokenspeed-mhc-source-review.md)
separates MNNVL communication fusion from eager MegaMHC. The former admits
prepared 2/4/8/16-rank BF16 `[T,5120]` inputs through 2048 rows, subject to
workspace and numerical-policy gates. MegaMHC remains unsupported under CUDA
graph replay; the V4.1 model excludes it from breakable capture, while its
low-level eligibility function does not enforce that restriction for new
direct callers. Current main also has an optional FP8-producing MegaMHC join.
Do not transfer eager fusion timing or kernel presence to the captured path.

The [TokenSpeed Hopper indexer review](../skills/llm-torch-profiler-analysis/references/tokenspeed-indexer-source-review.md)
traces merged #1857's fused cleaner/block maxima through current candidate
selection. Only bounded consumers may read its partially initialized outputs;
scratch still scales with capacity. The standalone block-max helper retains a
full-width read, and the native Hopper logits budgets remain 32/128 MiB for
paged/broadcast scoring. Do not apply Blackwell's newer scoring policy to it.
The same review traces merged portable DSA empty-tile skipping: it preserves
the dense logits grid and Top-K scan, unlike providers that accept compact page
plans. The KPool prefill TP row-split proposal is closed without merge; current
GLM-5.3-Flash still passes full query rows through that runtime call. Its long
prefill results and small-batch gather regressions remain branch evidence.

The [TokenSpeed operator review](../skills/llm-torch-profiler-analysis/references/tokenspeed-operator-source-review.md)
reconciles merged routing, sampling and attention changes. BF16 precomputed
MoE routing still rounds weights to BF16 before passing the unpacked tuple;
the implementation moved to `ops/moe/flashinfer.py`. The chain sampler skips
an unused fallback-index reduction, while six other callers retain it; the
current chain entry point supports two acceptance rules. The runtime TRT-LLM
MHA leaf reuses completion counters across calls and KV-pool rebinds, with
tree-binding growth and an oversize fallback. Generic registered wrappers
still omit the buffer. Attribute removed fills to the actual caller, and keep
historical wheel-specific microbenchmarks separate from current serving results.
The FP32 row-CTA projection has moved from `decode_gemv` to `gemm.mv`, with
N≤1024, M≤16 and separate single-/multi-row K limits. Beam now calls this
dtype-aware path directly; ordinary FP32 `gemm.mm` is not redirected through
the BF16-only MV shortcut. Preserve the four-element BF16 load cap and disabled
FMA contraction when reasoning about equality to explicit FP32 widening.

- SGLang kernel roots are `python/sglang/kernels/{aot,jit,ops}`. The current
  [dependency pin](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/pyproject.toml) uses FlashInfer 0.7.3 (cu13).
  Check the actual wheel/image, not just source main.
- vLLM's [pass manager](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/passes/pass_manager.py) and
  [PassConfig](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/config/compilation.py) no longer contain `enable_sp`
  or `fuse_gemm_comms`; the old sequence-parallel/collective pass files are gone.
  Model/backend MoE sequence parallelism is a separate runtime decision.
  O2/O3 now conditionally enable attention-output quant for quantized models.
  The [compile guide](../skills/llm-torch-profiler-analysis/references/vllm-torch-compile-fusions.md)
  covers current registration, provider gates and Muse-specific patterns.
- [FlashInfer block-scaled cluster split-K](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_splitk.py)
  retains M≤32, two/four K slices, `swap_ab=True`, R128C4 scales and
  K divisibility by `128 * split_k_slices`. DSM reduction combines FP32 partials
  without a separate global reduction kernel. Per-token alpha is applied in
  FP32 after that reduction. [Tactic selection](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/gemm/gemm_base.py)
  still decides dispatch; source availability is not measured benefit.

## Memory sizing

[KV sizing](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/mem_cache/kv_cache_configurator.py)
reserves `pre_model_load_memory * (1 - mem_fraction_static)`, subtracting it
and model-specific reservations from currently free memory. It does not
multiply post-weight free memory by the fraction.

[Post-capture eligibility](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/arg_groups/overrides.py)
and [physical pool resizing](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/model_executor/model_runner_components/kv_pool_runtime.py)
now cover eligible DFlash-family draft pools with the target. Distinct backing
owners are deduplicated; the final budget includes their physical bytes and
reserves canary workspace. Target summary starts a resize round, followed by
one `Post-capture KV sizing: draft KV cache allocated.` line per draft owner.
Sum these per rank; never let the last draft overwrite the target. VA upper
bounds are not allocations; independent pools and Mamba state remain separate.

The feature remains opt-in and CUDA-only. Gates exclude logical-page KV
sharding, memory saver, incompatible Mooncake/EFA pools, prefill-only cache
disable, insufficient role-specific graph coverage and compressed-KV models.
DCP/MLA need unified memory. There is no blanket exclusion of every speculative
or mamba-ish model; uncovered decode/mamba activation reserves still matter.
Use final logs rather than extrapolating ordinary dense-KV arithmetic to
compressed pools or speculative buffers.

## Maintaining this evidence

Read caller and implementation at the same SHA. New PR cards require full
diff review; source links here do not claim such a review of all historical
PRs. Keep numerical/performance claims tied to their actual experiment.
Use [dispatch principles](../skills/llm-torch-profiler-analysis/references/heuristics.md#establish-dispatch-and-numerical-contracts)
and [paired validation](../skills/llm-serving-auto-benchmark/references/paired-validation.md)
when testing a candidate.
