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

The [DeepGEMM LiteTopK producer review](../skills/llm-torch-profiler-analysis/references/deepgemm-litetopk-producer-review.md)
adds a November 9 comparison against `dev` commit
`909f68c19cc23c6afd9cc039c0493b00586b66b2`. Open #94 emits an additive
1024-bin histogram alongside the stored BF16 logits, with a 170-split
shared-counter budget and seven-stage MXFP4 D128 KV ring. It does not remove
the score matrix. Include the reported 5/6-token producer overhead when
evaluating selector savings. Its graph test zeroes the histogram inside
capture; the sanitizer path skips that graph section. Current `dev` still
lacks the histogram argument and has a compatibility wrapper plus newer
CC 10.7 chunk/ring and TMEM-base handling, which a rebase must preserve.
The complete SGLang consumer/migration reviews remain pending.

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

## FP8 KV store dispatch

The [SGLang FP8 store review](../skills/llm-torch-profiler-analysis/references/sglang-fp8-kv-store-source-review.md)
separates merged #40785 prefix commits, merged #42982 saturation, and open
#31652 ordinary-store fusion. Current DFlash/DSpark prefix calls reach a
shared helper through the layer router. Its fused FP8 branch requires equal
K/V row widths, E4M3 storage and resolvable scales; GPU scale parameters use float shadows.
Preserve tensor-versus-host division semantics and source-dtype rounding
before clamping to the representable FP8 limit. Prefix commit can write slot
zero; the proposed ordinary writer must skip that reserved padding slot.

The ordinary proposal admits CUDA SM89+, dense NHD sources, independent K/V
widths, no DCP mask, and host/absent/scalar FP32 device scales. Its device
scales are read during graph replay. It only changes the legacy MHA pool;
current unified `MHAView.set_kv_buffer` is independent and remains unfused.
Retain caller copies where fallback stores can mutate inputs. Include layout
copies in prefix-cluster timing, and distinguish graph-level L2 flushing from
cold destination lines on every ordinary-store call. Historical microbenchmarks
and small serial-serving reports do not establish current all-path speedups.

The [SGLang Q-quantization review](../skills/llm-torch-profiler-analysis/references/sglang-query-quantization-source-review.md)
tracks closed, unmerged #35142. Its compile path exposes a native Q cast to
Inductor; its eager kernel stores FP32 rotation directly to FP8, omitting the
separate CUDA path's BF16/FP16 rounding boundary. Current FA admission depends
on the selected per-layer version and phase. Current Qwen3 can produce Q with
its own fused norm/rotary kernels, bypassing the modified rotary entry point.
Check that actual producer before attributing a remaining Q cast to a missing
RoPE fusion; historical throughput reports do not validate a current port.

## Compiler and dependency dispatch

The [vLLM compiler review](../skills/llm-torch-profiler-analysis/references/vllm-compiler-source-review.md)
distinguishes current splitting/custom-op ordering from closed #57349 and its
open successor #57900. Ordinary/QK RoPE-cache flags can be disabled after
default resolution; an MLA flag can survive while its cache op is split out.
Matched FP8 Q quantization remains a separate operation after the fused
RoPE/cache writer. Current QK/MRoPE registration also changes V-cache layout
and validates head dimensions and K consumers. The pending fix's MI355X
measurements include 128/256-token regressions within the unchanged default
256-token range; pattern matches on other models are not measured speedups.
The same review distinguishes #49142's platform-only MLA default proposal
from #54851's metadata-free DSA capture fix. Current MRV2 builds metadata even
for PIECEWISE; `for_capture=False` is not proof of an empty forward context.
Check the captured cache/slot addresses and current independent indexer,
HiSparse and PCP ownership before diagnosing stale writes or porting the patch.

The [CUDA RoPE/cache reviews](../skills/llm-torch-profiler-analysis/references/vllm-rope-cache-kernel-review.md)
distinguish open #57271's full-width equal-head GQA writer from open #58798's
partial-RoPE, V-scaling and unequal-width MiMo DiffKV path. Check short slot
maps and the FP32-to-cache versus intermediate-BF16 rounding boundary before
accepting the ordinary FP8 variant. Current MiMo can retain padded checkpoint
chunks in QKV projection output; the proposed plain split needs adaptation or
an explicit layout gate. Historical B200/SM80 attention timings do not validate
that later mainline layout or establish service throughput.
The same review covers open #52363's manual Llama/FlashAttention path, which
keeps CUDA opt-in and checks the token threshold inside the compiled custom
op. Preserve its bypass/rotation contracts and the above-threshold Q/K-copy
cost; earlier hardware reports and approval predate the final refactor.

The [CUDA QK-norm/cache review](../skills/llm-torch-profiler-analysis/references/vllm-qknorm-rope-cache-source-review.md)
covers open #57682's complete diff and stacked partial-RoPE shuffle fix. Its
FlashAttention writer requires an unquantized decoder cache, equal supported
head widths and no KV sharing/DCP; opaque layer names additionally require
homogeneous eligible layers. Preserve compile-range and graph-visibility gates.
Bitwise parity is against the already-fused QK-norm/RoPE baseline; the separate
eager chain and Step5 kernel have different intermediate rounding. Current
Step5 and Muse Spark use explicit model-specific paths, so the generic CUDA
pass being unavailable does not establish a model-wide fusion gap. The reported
H200 service results use an old-wheel overlay and dummy weights, with final-head
GPU validation pending.

The [FlashInfer QK-norm/append review](../skills/llm-torch-profiler-analysis/references/flashinfer-qknorm-rope-append-source-review.md)
replaces the obsolete claim that mainline lacks RMSNorm before fused RoPE/KV
append. Current dedicated CAKE BF16/FP8 APIs accept packed BF16 QKV, D=128,
head pairs 8/1 or 64/8 and contiguous NHD caches on SM90/100/103. Their stage
selection and optional-output contracts differ from closed #5405. Current
FP8 integrates indptr validation/status writes into the compute launch; the
old dynamic-prefill path launched a separate validator and zeroed flags in
Python. Padded Q scales remain unwritten. PDL waits protect QKV reads and
stores, while metadata, weights and scales must already be ready. Explicit
API availability does not prove a serving model invokes it, and #5405's
5.75–6.37× BF16 pipeline timings do not measure current CAKE or service gains.
The full #5952 review establishes the initial CAKE BF16 implementation:
head/architecture-only JIT selection, register-resident FP32 norm/rotation,
one ordinary launch and independent K/V redirects with optional tail clearing.
Its 21-case test source and historical H100/H200/B200 reports do not cover
changing-metadata graph replay or B300 execution. The committed 17-shape
benchmark saves medians only, so the body's 28-row confidence-interval claim
cannot be reproduced from that output. Current shape/PDL policy is a later
contract and must be checked independently.
The full #5956 FP8 review confirms all-mode indptr validation within the
single launch, eight-lane shuffle masks and raw zero-word tail stores.
Sequence/page bounds and valid scale contents remain caller obligations.
Final-head H100 CI ran the 51-case source suite successfully on CUDA 12.9;
the tests use numerical tolerance for tail zeros despite a bitwise claim in
the PR text. Historical 18.2×/14.9× baseline ratios compare different
numerical/tail-write behavior, and the external paired protocol differs from
the committed fixed-order, eager benchmark. Neither establishes service gains.
The complete #6008 BF16 review traces direct guesses, twenty-warps-per-row
ownership and paired K/V clearing units. Its 42 passing H100 cases include
21 CPU selector checks, so they do not establish every generated program's
numerical coverage. The historical page-1024 `w20 + u2` missing-build error
was reproduced from the pure selector; current main separates that rule and
has later twenty-warp programs. Check the selected stage and actual grid,
including clearing CTAs retained when tail writes are disabled in #6008.
The full HY3 #4890 review retains its open status and November 3 suggestion
to move the large-batch schedule into CAKE. Its explicit uniform-one-token
promise, SM100-only FP8 dynamic-Q/norm2/64Q8KV/B>=256 fast path and unwritten
empty-request flags differ from current CAKE's contract. The 59-case author
report includes a meaningful delayed-producer stream check and exact
fast/general equality, but no graph-replay test. Static comparator rounding
and side effects differ; historical operator ratios do not establish a current
CAKE or serving advantage. Trace tags do not prove the fast branch executed.

The [SGLang MoE collective review](../skills/llm-torch-profiler-analysis/references/sglang-moe-collective-source-review.md)
separates closed #39214 from current CuTe-DSL boundary fusion. The proposal
gathers dense expert-major rows and converts scales before the pure Cake API;
those launches and temporary bytes are part of its cost. Its helper accepts
FP16/BF16, but the actual deferred producer requires BF16. Current main passes
native permuted rows/indices to `kMoEFinalizeARResidualRMSNorm`, with independent
producer and consumer eligibility. It prepares one graph-stable workspace and
withholds early shared-expert loads under speculative decoding. The proposal's
deterministic-mode exception is not current behavior. Reported TP8 payload
checks pass tolerance while the per-question score-equality criterion fails;
neither model-output parity nor serving speedup is established.

The [FlashInfer MoE collective review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-collective-source-review.md)
fully reviews merged #4730, including all 28 original generated programs.
Its reduction API requires dense expert-major rows and preserves activation
rounding after each contribution/addition. The Lamport entry counter is not a
generic output-completion event. Current union selection also depends on expert
count, PDL and token ranges, and a missing all-reduce output now writes to
graph-stable scratch rather than selecting a no-store kernel. Final-head
H100/T4/A10G CI passes twelve CPU contracts and skips all six distributed
cases; the author's native B200/B300 reports remain separate. Runtime bytes
match the measured candidate, while source-inventory tests were subsequently
corrected. The historical 1.12x operator ratio is not a current serving result.
The #5856 consolidation routes omitted outputs into that sink and removes
the legacy bundle; its paired timings compare different schedules, with L2
flush disabled to limit rank arrival skew. H100 CI passes 69 CPU cases and
skips all six distributed parameters. The old graph test ends before cache
growth and therefore did not cover the surviving-graph lifetime problem.
The same review covers #5894's graph-address retention fix: live plus retired
scratch payload is below twice the live capacity per device/dtype cache,
excluding graph-private allocations and allocator overhead. H100 final-head CI
passes 70 cases including the capture test; the canary test retains old tensor
references, so it is not an independent reproduction of freed-storage reuse.

The [indexed MoE finalize review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-source-review.md)
completes merged #4875, including all generated device/binding variants.
Its permuted rows, Int32 inverse map and activation-dtype route scales differ
from the reduction API. Preserve rounding before residual/RMSNorm and FP4
quantization, and do not read unwritten scale padding. Current code uses one
module per architecture, C++ shape/occupancy dispatch and byte-sized quant
buffers rather than the original manifest/16-bit-container requirements.
Final-head H100 CI passes seven CPU cases and skips 43 GPU cases across the
three relevant files. Historical B200/B300 timing mixes protocols/cohorts;
the committed harness lacks the external v4 per-iteration host rendezvous.
Current two-shot/deferred successor device programs require their own review.
The same review completes #5852's host consolidation: retained device files
are unchanged, quant/scales use byte capacity, and only the public API keeps
the Lamport payload cap. Its trimmed benchmark can fall back to CUDA events
despite reporting CUPTI enabled and no longer preserves raw timing samples.
Retain the individual warmed-run regressions and the incomplete TP4 timing;
six CPU passes/43 skips in H100 CI do not verify Blackwell execution.
The #5870 template consolidation retains the numerical/protocol contracts
and fixes compiler-sensitive poll expansion. Its reported SASS identity is
within each nvcc version. A/A spread and rank-min analysis do not erase the
remaining rank-max regressions or TP4 hang; the historical template file has
since been removed from current source registration.
The #6057/#6128/#6166 reviews trace paired loads, the alternating DSM slot
ring, local owner polling and TP4-only residual prefetch. TP8's earlier local
poll regression was attributed to rolled code generation; TP2 quad and TP8
residual hoisting were rejected for specific throughput regressions. The
current launcher caps occupancy at five and also queries active-cluster
capacity; its shape-selected successor programs differ from that historical
template. Keep external rank-min timing, missing completed Blackwell CI and
reported large-token timeouts separate from the six CPU passes/43 GPU skips
verified for each final-head H100 run.

The [#7392 scheduling review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-schedule-review.md)
traces all 32 generated programs, TP2 lookahead, small-token oct gathering,
the TP4/TP8 publish/clean/consume pipeline and mbarrier RMS exchange. The
active-cluster query bounds the grid separately from block occupancy. Its
historical large gate is literally `T > 2*C`, not a guarantee of three tokens
on every cluster. The committed T<=128 tests miss the large paths on normal
B200/B300 devices; final-head H100 CI passes nine CPU tests and skips six GPU
tests. The external 720-row rank-min table retains three 0.997x B300 rows.
Current two-shot/deferred selection has different gates: twelve historical
source paths are gone and all twenty surviving files have changed.

The [#7511 two-shot review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-two-shot-review.md)
traces owner reduce-scatter/all-gather, the BF16 negative-zero sideband and
TP2's shared-memory own-partial ring. Protocol selection must agree across
ranks even when local capacity selects different compatible one-shot programs.
Its expanded boundary tests inspect kernel names only under specified PDL,
weight-bias and device conditions. Final-head CI ran six AOT build/import
jobs, with no targeted CPU/GPU test jobs or JUnit artifacts; green status
does not validate the new numerical paths. Keep the external 0.977x rows
and 3.422x/3.468x operator maxima distinct from serving claims. All 32
historical device files differ at the current pin. The same review covers
#7650's in-band BF16 negative-zero escape: current corresponding helpers
retain it, replacing the separate mask. Exhaustive CPU mapping checks preserve
all finite communicated values but expose the `0xFFFF` NaN collision, so the
body's bitwise claim needs a domain qualification. Its 72 changed timing rows
are separated from 648 unchanged rows, and final-head CI again has AOT only.

The same review covers #7813's eight deferred programs and six existing
two-shot load changes. Keep one token's activation-rounded residual sum in
thread-owned shared carry until its RMS wait and normalized/quantized stores
finish; drain the last token explicitly. The historical TP4 T88 threshold is
now T72. Its timing table omits the newly admitted threshold ranges, retains
two changed-program B300 regressions, and reports operator timing only.
Final-head CI again ran AOT only. All 40 historical device files differ from
the current pin; the data-lifetime review does not certify later bodies.

The [#7905 prefetch review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-prefetch-review.md)
traces the TP4 crossover to T72 and sixteen one-shot packed-load rewrites.
Residual and norm-weight words cross the poll before conversion, with an
empty assembly operand dependency rather than a new synchronization barrier.
Retain 62/59 changed-shape regressions on B300/B200, mostly TP8, and separate
the reported T72 aggregate from the 720-row table. The revised test mirror
reuses existing T71/72 cases; final-head CI again has no targeted GPU results.

The same review covers #8073's twelve small-token input-prefetch programs.
Route/shared-expert reads remain after PDL wait and precede the epoch/clear
reads; the buffer-size word is still read earlier. Publication and reduction
bases use local copies of the same control values, not newly sampled epochs.
Full-file transformations preserve the remaining arithmetic and protocol.
Retain twelve/two changed-shape B300/B200 regressions, all TP8, and distinguish
the external rank-min timings from the unchanged rank-max harness. The final
green CI again has AOT builds but no targeted numerical test jobs.

The review also traces #8139's TP4/K4/no-shared extension: with the default
limit, T65–71 selects the small program only when resident-cluster capacity
holds every token. T72 still takes the earlier two-shot branch; TP2/TP8 rules
do not change. All forty regenerated bodies differ only by unused-macro
removal and bijective temporary renaming. The new CPU predicate checks pass
locally, but neither those checks nor the final AOT-only CI prove GPU occupancy
or numerical execution. The sixteen measured changed shapes are T65/T71;
the standard 720-row matrix exercises unchanged programs.

The same review covers #8321's eight early two-shot programs. Default TP4
T72–128 and TP8 T56–128 read the first four routing indices after PDL wait,
before the current/clear words; the buffer-byte word is still read earlier.
Scatter and clearing addresses reuse those sampled words. K8's second index
group and shared-expert data are not newly hoisted. Original CPU table and
boundary functions pass locally, but final-head CI runs only AOT builds.
The 64 published changed timing rows are a subset of the reported 320-shape
cohort; the other 256 rows and smaller-limit fallbacks lack individual data.
Current early dispatch matches, while all 48 later whole bodies differ.

The [TP2 quad-route review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-quad-review.md)
reconciles that subsequent regeneration in #8503: all 48 retained bodies and
four new quad bodies match current bytes. Quad selects eligible TP2 batches
through T142 after the small/large branches; actual cluster capacity still
caps the launch. Shared-expert values are read early but added at the original
position. Both NVFP4 dtype variants still read clamped row-zero data for
negative routes and skip their contribution. Preserve the identity-only GPU
inputs, AOT-only final CI and one B300/two B200 regressions in the broader
176-shape cohort. Only 32 changed timing rows are published individually.
Eleven additional current device bodies remain outside that review.

The [shared-expert prefetch review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-shared-prefetch-review.md)
covers four of those bodies in #8697. TP4 shared-expert T≥1536 selects
`_tsds`: one thread in eight issues an L2 hint before routing, after PDL wait;
the actual non-coherent vector load and shared addition keep their original
position. The two-shot protocol and dtype rounding remain unchanged. All 56
historical device files match current bytes. CPU tests cover mixed shared-presence selectors, while
final-head CI passes 26 CPU cases and skips all six multi-GPU nodes. The
reported 32 changed measurements repeat sixteen shapes, all at T2048;
retain that limitation when using the roughly 3% kernel timing result.

The same review covers #8864's remaining seven bodies: TP8/K8/shared T≥768
uses `_tsds` except FP16 ordinary output; TP2/K4/shared T>1280 uses `_lts`.
The omitted TP8 slot falls back to `_tsd`; TP2 retains its one-shot pipeline.
All 63 current device files, binding, loader and CPU/GPU tests now match the
reviewed source. The launcher uses small-token limit 64; an arbitrary larger
direct-FFI override can still intercept TP2 before large-token selection.
Final-head CI passes 29 CPU tests and skips six GPU nodes. Mixed-rank tests
remain selector-only. The fourteen published changed rows and fourteen extra
reported measurements cover 22 distinct shapes, with no new serving result.

The [FlashInfer source-consolidation review](../skills/llm-torch-profiler-analysis/references/flashinfer-comm-source-consolidation-review.md)
traces #7054 across five communication families. Shared source paths retain
distinct launch bounds, pipeline depth and module compile definitions; source
deduplication is not performance evidence. Ulysses's included header is outside
the manifest digest check at both the PR head and current pin, although the
JIT path separately invokes Ninja with compiler-generated dependencies. The
current norm-combine host has six programs: T=1, 2–3, 4–37, 38–479 and a
T≥480 pipeline band whose early-load choice also depends on SM count. Its
historical shared binding has been replaced. Preserve these revision and
validation boundaries when mapping trace symbols to a proposed optimization.

The [vLLM routing review](../skills/llm-torch-profiler-analysis/references/vllm-routing-source-review.md)
traces merged #59737's fused softmax/sigmoid gating PDL. Current main enables
the host attribute only on SM90+, while row-padded logits and unsupported
expert counts still use ordinary score/top-k launches. The producer's runtime
GEMM cutovers are shape-specific; the old M=32/33 boundary is not universal.
Early residence includes dependency waits and does not prove saved time.
Historical GB200 reports show modest decode gains and some prefill regressions;
their kernel overlays and two accuracy repeats do not validate the current build.

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

The [TokenSpeed selected-slot attention review](../skills/llm-torch-profiler-analysis/references/tokenspeed-dsa-source-review.md)
traces merged #1780 through current portable dispatch. Gathered latent tiles
feed both QK and PV; BF16 residuals preserve score precision, while FP32 split
partials are merged with base-2 weights and public LSE remains natural-log.
Split count follows grid geometry rather than phase names. This provider
does not advertise batch invariance or sorted-slot order, and ignores physical
KV lengths: selection must already encode causal validity. Current KVP record
output is a separate, gated evolution of the unsplit dense core. Historical
H20 results predate the final stride-preserving rebase and need that qualifier.

The [TokenSpeed routing review](../skills/llm-torch-profiler-analysis/references/tokenspeed-routing-source-review.md)
traces merged #1770 through V4.1's current MoE plan and CUDA finalizer.
Route scoring, normalization and scale stay FP32 until the BF16 store; default
MXFP8 quantization selects CuTe only for contiguous inputs already at padded
width. Deferred routed/shared finalization requires the original hidden width,
BF16 activations and NVIDIA/provider admission. Main now restores missing
precomputed weights in `MoELayer.forward`; MegaMoE has a separate finalized
path. Historical full-active-batch DSpark cycles include draft and verify work
and must not be quoted as latency per accepted token.

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
The closed gfx1250 persistent MoE proposal is not installed by those merged
changes: current `ops/moe/gluon.py` still calls regular decode/prefill, with
separate FP8 and dynamic-MXFP4 activation contracts. Its proposed two-routes-
per-expert gate and unresolved public activation defaults remain branch facts.

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
