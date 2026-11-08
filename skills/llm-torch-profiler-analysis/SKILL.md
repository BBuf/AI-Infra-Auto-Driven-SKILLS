---
name: llm-torch-profiler-analysis
description: "Analyze existing Torch Profiler Chrome traces or capture from SGLang, vLLM, TensorRT-LLM and TokenSpeed servers. Return kernel attribution, overlap opportunities and fusion patterns; use for model dispatch, CUDA Graph gaps, stream contention or exposed kernel tails. Includes existing SGLang Omni traces."
---

# LLM Torch Profiler Analysis

Produce three tables: kernel/source attribution, overlap opportunities, and
fusion patterns. Start from the user's trace or running server. Inspect the
actual model, framework revision, phase, device/rank and parallelism before
choosing a fast path. Existing traces need no GPU or framework installation.
The Python analyzers use the standard library.

Read [source contracts](../../docs/upstream-source-contracts.md) for dated
source inspections and version-sensitive profiler APIs. Apply the general
[dispatch and numerical principles](references/heuristics.md#establish-dispatch-and-numerical-contracts)
to every model. The [fusion catalog](references/fuse-overlap-catalog.md)
contains concrete operator families, eligibility examples and storage contracts;
the [overlap catalog](references/overlap-catalog.md) covers scheduling.
Consult the model-history knowledge base when detailed PR evidence is needed.
Source checks and historical captures are not fresh GPU validation.

## Choose the evidence

| Mode | Evidence and limits |
|---|---|
| Existing trace | All four frameworks; SGLang Omni Python roots are also recognized. Use Chrome JSON or gzip with GPU events. A generic trace may need `--framework` explicitly. |
| Single live capture | Use a shared server/client output directory and a supported HTTP profiler. Save actual launch arguments and package revisions. |
| Mapping + formal | Eager/graph-off trace supplies call-site context; warmed graph trace supplies timing. Match signatures, shapes and stage before transferring attribution. Never transfer eager timing to the formal run. |

For a diffusion workload, record whether it uses the framework's native backend
or a Diffusers fallback. A fallback trace can be analyzed, but does not establish
native-kernel performance or dispatch.

## Read a trace

Commands below are relative to this skill directory.

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework sglang --input /path/to/TP-0-DECODE.trace.json.gz

python3 scripts/analyze_llm_torch_profile.py \
  --framework vllm \
  --mapping-input /path/to/eager-mapping \
  --formal-input /path/to/warmed-graph
```

Do not collapse a distributed run to rank0 when investigating rank skew or
collective waits. Keep the original capture; a GPU-only or compact export is
for sharing, not replacement evidence. Use
[torch-profiler-layer-track](../torch-profiler-layer-track/SKILL.md) for verified
L0/L1 guides and synthetic display lanes, or
[llm-pipeline-analysis](../llm-pipeline-analysis/SKILL.md) for layer intervals.

## Capture against a running server

The common helper defaults to separately driven prefill and decode workloads:
nominal input/output lengths 4090/1 and 1/2048, with 10 warmup steps and 5
requested capture steps. Override these to match the slow serving scenario.
Synthetic prompt lengths are not a tokenizer-independent guarantee; save actual
usage and batch/graph dimensions. A long-context decode bottleneck needs a
representative KV length, not automatically the one-token-input default.

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework sglang --url http://127.0.0.1:30000 \
  --output-dir /shared/profiles/run-001 \
  --profile-workload both --warmup-steps 10 --num-steps 5 \
  --prefill-input-len 4090 --prefill-output-len 1 \
  --decode-input-len 4090 --decode-output-len 2048
```

`--profile-workload` accepts `both`, `prefill`, `decode`, or `legacy` (explicit
probe prompt/token settings). Workload labels alone do not make a trace purely
decode: inspect stage annotations and exclude request prefill when needed.
The helper targets SGLang profiler v1 (default). Declare `--sglang-profile-v2`
if the remote server enables `SGLANG_PROFILE_V2=1`; the helper refuses this
incompatible flow before issuing control requests. `/server_info` does not
expose remote environment variables. Native v2 needs `profile_by_stage:true`
and `num_steps`, without `start_step`, `merge_profiles` or `/stop_profile`.
Current Rust ingress also supports start/stop with the same scheduler control.
Older images may lack these routes; check the actual response. Keyed servers
need the configured admin/API credential; the bundled helper has no auth flag,
so use authenticated native control and offline analysis for those captures.
The host launcher explicitly sets `SGLANG_PROFILE_V2=0`.
The helper puts its label in `profile_id`, so merging can discover rank files.
SGLang can emit stage-specific traces; its scheduler stop counter may require
a final forward to close the window. The helper's SGLang workload adds a guard
request/step. Check actual trace counts instead of assuming identical windows
across frameworks.

For other frameworks add `--no-profile-by-stage`. Their HTTP windows and
iteration controls differ:

- **vLLM:** launch with
  `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/shared/profiles/run-001","max_iterations":5,"ignore_frontend":true}'`.
  Current start accepts per-session `profile_prefix`, `max_iterations`,
  `delay_iterations` and `profiler_kwargs`. The helper checks `/openapi.json`
  and sends a stage-qualified prefix and `--num-steps` as `max_iterations`
  only when advertised. Older/OpenAPI-disabled servers retain bodyless control
  and launch-configured limits. Output directory and optional profiler warmup
  remain launch settings for this helper; probe steps do not override a
  warmup/wait/active schedule or guarantee a pure decode window.
  `torch_profiler_with_stack` defaults true in the inspected source. Frontend
  traces can contain no kernels; select GPU-worker traces for GPU analysis.
  `active_iterations` only takes effect with a warmup/wait schedule; use
  `max_iterations` to bound workers. The host option is `--profiler-max-iterations`
  (`--profiler-active-iterations` remains an alias). Two capture rounds require
  [vLLM #57460](https://github.com/vllm-project/vllm/pull/57460), merged
  2026-09-21; older images need separate launches for prefill and decode.
  Keep CPU in `torch_profiler_activities` for source attribution and step
  annotations; CUDA-only activities omit both Python stacks and annotations.
- **TensorRT-LLM:** the inspected PyTorch server accepts runtime start with
  `output_dir` and `activities`; the helper sends both and stops after probes.
  Native traces are uniquely named per request and rank. There is no need for
  the retired `py_executor.py` rank0-only patch or host-specific matrix runner.
  Current setup is in `_torch/pyexecutor/profiling.py`, still without
  `with_stack=True`. Report unresolved Python attribution, or make a separate,
  recorded mapping build against that installed file. Older images without
  HTTP support must use their documented capture method plus `--input`.
- **TokenSpeed:** the helper sends `output_dir`, CPU/GPU activities,
  `with_stack=true`, `record_shapes=false`, and a stage-qualified `profile_id`.
  Current servers expose start/stop; verify the deployed image. Proton's
  native Hatchet artifacts are not Torch Chrome traces. Current `serve` uses
  SMG plus a headless Rust-hosted engine; the old Python control sidecar was
  removed. Resolve the actual gateway/control URL and validate forwarding on
  the installed pinned wheels. Do not assume serving port + 1.
  `EXPERT_LOAD` writes `.expert-load.pt`, not Chrome traces.
  `TOKENSPEED_CUPTI_GRAPH_WARMUP=1` can leave zero GPU events in later captures.
  Current Torch capture requires process-global `profile_all_threads` support
  for Rust-owned forward threads. AMD capture needs
  `HSA_TOOLS_DISABLE_REGISTER=0`; Proton cannot combine with GPU/CUDA_PROFILER,
  and expert-load recording refuses online EPLB. See the source contracts.

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework trtllm --url http://127.0.0.1:8000 \
  --output-dir /shared/profiles/run-002 \
  --no-profile-by-stage --profile-workload both
```

A server-side directory must be mounted here or copied back for analysis.
The helper rejects a capture with no new artifact, recognizes overwritten
filenames, and copies fresh remote traces into a capture-specific subdirectory
without moving the server's originals. Preserve that returned directory rather
than analyzing every historical run under a shared root. Concurrent profiling
sessions still need separate output directories.

Native alternatives include SGLang graph-capture traces under
`graph_capture_profile/`, `MEM` snapshots, and `CUDA_PROFILER` with nsys;
vLLM CUDA/NVTX and Proton Chrome export (the old `capture_torch_profiler`
option is deprecated and has no effect);
TRT-LLM `TLLM_PROFILE_START_STOP`/`TLLM_TORCH_PROFILE_TRACE`; and TokenSpeed
VizTracer/Proton plus `tokenspeed merge-traces`. Graph-capture and VizTracer
artifacts are excluded from default runtime-trace discovery. Pass a file
explicitly when intentionally studying one of these artifacts. Native Hatchet,
MEM and expert-load outputs require their own readers.

## Interpret before optimizing

1. Confirm nonzero GPU events, a complete warmed forward, actual dispatch and
   target/draft separation (`step[VERIFY ...]` versus `step[DRAFT ...]` in
   SGLang; vLLM uses `execute_context_*_generation_*`). BS=1 need not mean kernel M=1 during speculation.
2. Read all relevant GPU streams. A CPU `cudaGraphLaunch` span is host time;
   missing lanes or dropped CUPTI records can look like GPU idle time.
3. Separate sum of kernel durations, union of GPU busy intervals, and end-to-end
   wall time. Overlapping kernels can sum to more than wall time. A table's
   duration share is not removable latency or an additive speedup estimate.
4. Identify the producer/consumer join and exposed tail. PDL consumers may
   include dependency wait in kernel duration. Compare PDL on/off with the same
   warp configuration before attributing a bimodal duration to arithmetic.
5. Consult source dispatch before proposing fusion. Existing routed-expert
   activation/quant does not establish shared-expert coverage. Match rounding,
   clamp, scale/layout, padding and graph reuse contracts. Check a quantization
   oracle against an independent formula: a kernel and its reference can share
   the same scale-floor error.
6. Follow state ownership across fused boundaries: cached normalized rows,
   quantized values/scales and next coefficients can become invalid after a
   residual rewrite or row selection. V4.1 Engram/late-layer selection are
   concrete examples; fewer launches do not prove equivalent semantics.
7. Time changes without profiling; run kernel correctness and actual-model
   accuracy on the candidate path. NCU replay/resource metrics help explain a
   kernel but can perturb or serialize the multi-stream behavior being studied.

For source mapping, prefer model/layer frames over dispatch/runtime wrappers.
Retain unknown locations as unknown. Kernel names and eager similarities are
hints; inspect the concrete producer, consumer and backend. The script's three
tables are observations and candidates, not proof that an optimization is safe
or already active in a different framework.

Capture labels must be filename components: the helper rejects path separators
and NUL before contacting the server. Live capture requires positive `num_steps`.
SGLang v1/v2 default to stacks on and shapes off; request shapes explicitly when
needed. The [profiler PR review](references/profiler-open-pr-review.md) distinguishes
current behavior from pending stage-counter, filename and teardown fixes.

## References and output

Load only relevant references:

- [source-map.md](references/source-map.md): immutable profiler and worker paths.
- [heuristics.md](references/heuristics.md): attribution and overlap labels,
  dispatch, numerical contracts, validation and their limitations.
- [vllm-torch-compile-fusions.md](references/vllm-torch-compile-fusions.md):
  current pass registration, platform gates and source patterns.
- [dsv41-open-pr-review.md](references/dsv41-open-pr-review.md): reviewed mHC,
  KV-scale and Hopper indexer proposals, actual gates, branch-specific flags
  and validation gaps. Tile budgets bound score tensors, not whole-model peaks.
- [dsv41-trtllm-source-review.md](references/dsv41-trtllm-source-review.md):
  merged sparse-attention integration, breakable prefill and public persistent
  counter API, plus the pending per-token/metadata rewrite. Distinguish main's
  remaining Q conversion from the proposal's fused output and PDL gates.
- [dsv41-hopper-indexer-review.md](references/dsv41-hopper-indexer-review.md):
  current visible-tile scheduling and fused head reduction, pending dense/compact
  scoring proposals, and the closed grouped-WGMMA prototype. Preserve logical
  versus compact scan lengths, numerical order and current prefill interfaces.
- [dsv41-mhc-source-review.md](references/dsv41-mhc-source-review.md): merged
  mHC state/collective ownership refactor, deleted combine-only kernels, cache
  invalidation and current boundary transport. Check who still owes the sum.
- [tokenspeed-mhc-source-review.md](references/tokenspeed-mhc-source-review.md):
  merged MNNVL post/norm fusion, eager MegaMHC and later FP8 output. Admission,
  graph-replay restrictions and stream contention determine the observed path.
- [fuse-overlap-catalog.md](references/fuse-overlap-catalog.md) and
  [overlap-catalog.md](references/overlap-catalog.md): historical precedents;
  recheck a PR's current state and target revision before borrowing code.
  The overlap catalog also covers GPU idle gaps, scheduler overlap, phase
  graphs, metadata glue graphs, replay streams, PDL and collective dispatch.

Return the trace/report path, framework and available model/server arguments;
the kernel, overlap-opportunity and fusion-pattern tables; the main exposed
bottleneck; attribution confidence and whether mapping/formal traces were used.
For a new capture also state which phases/ranks contain GPU events. Do not
claim a speedup or accuracy equivalence without the corresponding experiment.
