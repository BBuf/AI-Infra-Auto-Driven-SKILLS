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
owns mHC dispatch, while
[deepseek_v4_boundary.py](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/deepseek_v4_boundary.py)
connects it to layer boundaries. V4.1 carries previous-sublayer pre coefficients;
V4's fused post+pre formula is not interchangeable. Current V4.1 can fuse post,
next combine, norm, quantization and next coefficient statistics. The old
combine-only kernels and `MhcPostFusion` context hook have been removed.
Use the [mHC dispatch matrix](../skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md#residual-mixing-and-collective-epilogues)
for row limits, collective ownership and consumer-format gates.

Cached state is valid only for the residual and rows that produced it.
`HcState.with_residual` drops cached normalized/quantized input and statistics;
`take_rows` narrows ordinary per-row state but drops swizzled MXFP8 pairs.
Engram seams and missing next boundaries can terminate a fused handoff.
For overlap, `use_stats_stream` selects decode or nonempty target verify;
Hopper restricts this to M=1 or decode M=2–64. `_compute_triplet` joins before
post consumes coefficients. Mega mHC can supply cached next statistics, so
a missing side-stream producer does not by itself indicate lost overlap.

[WO-A dispatch](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/deepseek_v4.py)
selects GEMV, small-batch GEMM or token-major BMM using phase, shape, strides,
platform and consumer format. Check target and DSpark draft separately;
request concurrency does not equal kernel M. Preserve intermediate BF16
rounding and the next consumer's MXFP8 scale layout when moving quantization.

[Main defaults](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/environ.py)
set `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=False`, `SGLANG_DSV4_KV_LAYOUT=v4`
and `SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=False`. The historical dsv4.1 branch
enabled fused finalize by default; its accuracy results retain that setting.
The [paged KV contract](../skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md#paged-kv-storage-contract)
distinguishes split and interleaved payload/scale layouts, page padding and
reader compatibility. The 528 B/288 B figures are per stored token before
page padding, not whole-model KV capacity.

[Detailed step annotations](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/model_executor/step_span_utils.py)
carry context/generation sums of query lengths, KV lengths, squared query
lengths and query×KV lengths. Verify/Draft widths come from speculative shape,
not acceptance length. These aggregates help check an analytical template;
the bundled simulator does not automatically ingest them.

## Sparse indexing and cache writes

Use the concrete producer and selector when diagnosing DSA/V4.1 traces:

- Current V4.1 selects BF16 scores through DeepSelect. DSA's DCP route also
  needs cross-rank selection/merge, local slots and local counts. A selector
  replacement must preserve those outputs and graph-owned storage.
- The [DSA top-k backend](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py)
  already passes row mappings on eligible packed-v2 and FlashInfer paths;
  the legacy AOT fallback still gathers the wide page table. Attribute a
  visible gather to its actual branch before proposing its removal.
  Packed-v2 can mask score prefixes in place, so overlapping rows or another
  consumer of that buffer need separate handling.
- Hopper's 1 GiB prefill score budget bounds an intermediate tensor, not
  total model memory. Compact physical columns and logical sequence lengths
  are different coordinate systems. Preserve scheduler geometry and live
  lengths when reusing sparse plans; unwritten score tails require bounded
  consumers. See the [indexer catalog](../skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md).
- The current DeepGEMM pin has separate legacy and modern MQA callers.
  Schedule buffers, request indices and FP32/BF16 conversion ranges must
  match the selected API. Historical full-width FP32-copy baselines do not
  represent current prefix-only conversion or native BF16 selection.
- SGLang's fused FP8 prefix commit requires equal K/V row widths, E4M3
  storage and resolvable scales. Preserve tensor-versus-host division and
  source-dtype rounding before saturation. Prefix commit can write slot zero;
  an ordinary cache writer may reserve that slot for padding. The unified
  MHA view and legacy pool are separate dispatch paths.
- Qwen3 may produce Q with model-specific fused norm/rotary kernels, bypassing
  the generic rotary entry point. Follow the actual producer before attributing
  a remaining Q cast to a missing RoPE fusion. Removing a BF16/FP16 intermediate
  before FP8 conversion changes the numerical contract.

## Compiler and dependency dispatch

- SGLang kernels live under `python/sglang/kernels/{aot,jit,ops}`. Its
  [dependency pin](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/pyproject.toml)
  uses FlashInfer 0.7.3 (cu13); check the deployed wheel as well as source.
- vLLM's [pass manager](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/passes/pass_manager.py)
  and [PassConfig](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/config/compilation.py)
  no longer contain `enable_sp` or `fuse_gemm_comms`; runtime MoE sequence
  parallelism is a separate decision. O2/O3 conditionally enable attention-output
  quantization for quantized models. Resolved flags, graph splitting, custom-op
  registration and compile range all affect whether a pattern can match.
  Matched RoPE/cache fusion can still leave Q quantization as a separate op.
  The [compile guide](../skills/llm-torch-profiler-analysis/references/vllm-torch-compile-fusions.md)
  records current registration and provider gates.
- CUDA Graph cache binding must use the deployed runner's real metadata,
  persistent slot addresses and write owner. Current vLLM MRV2 builds metadata
  for PIECEWISE; `for_capture=False` alone does not prove stale writes.
- [FlashInfer block-scaled cluster split-K](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_splitk.py)
  retains M≤32, two/four K slices, `swap_ab=True`, R128C4 scales and
  K divisibility by `128 * split_k_slices`. DSM reduction combines FP32
  partials; per-token alpha is applied afterwards. The
  [tactic selector](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/gemm/gemm_base.py)
  still decides whether it runs.
- FlashInfer Cake collective/finalize selection depends on TP, token count,
  expert layout, output dtype and available workspace. Preserve routing-weight
  rounding and the location of shared-expert addition. PDL permits early
  residence but dependency waits can remain inside the kernel duration.
  Scratch whose addresses were captured by a graph must remain alive.
- TokenSpeed MNNVL post/norm fusion and MegaMHC have different admission.
  MegaMHC is excluded from V4.1 breakable capture; its low-level eligibility
  function alone does not guarantee graph safety. Deferred routed/shared
  finalization also needs BF16 activations and the original hidden width.
  DSpark cycle timing includes draft and verify work, so it is not TPOT per
  accepted token. Use the [fusion catalog](../skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md)
  and [overlap catalog](../skills/llm-torch-profiler-analysis/references/overlap-catalog.md)
  for concrete symbols and backend gates.


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

Read caller and implementation at the same SHA. Keep numerical/performance
claims tied to their actual experiment and distinguish pending proposals from
code available in the deployed revision.
Use [dispatch principles](../skills/llm-torch-profiler-analysis/references/heuristics.md#establish-dispatch-and-numerical-contracts)
and [paired validation](../skills/llm-serving-auto-benchmark/references/paired-validation.md)
when testing a candidate.
