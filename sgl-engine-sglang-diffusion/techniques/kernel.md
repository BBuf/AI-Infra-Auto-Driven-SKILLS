# Kernel executor scope

## Identity and correctness

`kernel` owns mathematically lossless optimization of every hot region inside
SGLang Diffusion's frozen load-excluded end-to-end path. It has a hard budget
of **60 candidate rounds**.
One round is one profiled hypothesis, one isolated implementation, one real run,
and one gate.

Correctness is established by reasoning about the actual method and code, not
by output similarity. Preserve the global logical denoising-step and
DiT/model-call counts and introduce no approximation, step skipping, sparsity,
sub-16-bit quantization, rank reduction, or changed logical model work. Numeric
movement caused by fusion, reduction order, or equivalent backend behavior is
not a rejection reason. Do not compute or gate on bit identity, tensor/latent
differences, floating-point tolerances, LPIPS, PSNR, or visual quality.

## Owned optimization surface

Profile the warm end-to-end path before selecting work. Candidate families
include:

- equivalent local attention, GEMM, normalization, activation, and convolution
  backends;
- AdaLN, residual-gate, attention-output-gate, and residual/modulation fusion;
- QK-normalization plus RoPE fusion and equivalent QKV projection merging;
- GEMM epilogues and FFN glue;
- exact caching of provably invariant local values such as conditioning K/V,
  position tensors, masks, transformed weights, compiled artifacts, and
  allocator buffers;
- local layout and data-movement improvements;
- launch and synchronization reduction; and
- warm `torch.compile` or other compiler fusion for stable repeated regions.
- regional compile, breakable graph capture, persistent compiler caches, and
  graph-break reduction;
- VAE decode layout propagation, exact decode parallelism, halo/copy removal,
  postprocess, and media-finalization fast paths; and
- scheduler synchronization removal, invariant precomputation, exact K/V
  reuse, inference-mode overhead, and other measured runtime glue.

The coverage ledger must disposition all nine registry IDs. In particular,
`layout-copy-launch` includes permute/contiguous/reshape traffic around
distributed attention, and `custom-or-upstream-kernel` requires an active
search for a Triton, CUDA/CuTe, JIT, AOT, or pinned upstream implementation.
Trying `torch.compile` once does not exhaust this lane. A compile-only attempt
cannot disposition VAE/output or scheduler/runtime coverage.

The executor must first audit SGLang's existing fast paths. Auxiliary SGLang,
KDA-Pilot, and FastVideo knowledge may suggest implementations, but cannot
expand correctness or evidence policy.

Use KernelWiki, NCU report analysis, and the warp-specialization report skill
according to `KERNEL-EVIDENCE.json`. NCU is mandatory for an implemented
Triton, CUDA/CuTe, or upstream kernel. Warp-specialization timeline
instrumentation is mandatory only when the CUDA/CuTe candidate actually uses
warp specialization; otherwise record why it does not apply.
On Hopper, preserve the NCU workflow but query architecture-valid metrics
instead of reusing Blackwell-only metric identifiers.

## Ownership boundaries

Residency owns component placement/offload and transfer prefetch. Baseline
CP/SP/TP/EP/FSDP/CFG degrees, process groups, rank maps, and selected GPUs are
frozen. Kernel may optimize equivalent collective implementation, layout, and
overlap without changing that topology. Cache owns approximate cross-step reuse; PISA owns
approximate attention; quantization owns sub-16-bit behavior; token pruning owns
reduced token work. Do not claim their gains in this lane.

Exact dead-computation or common-subexpression elimination is allowed only with
a method argument proving identical inputs/results and a provably zero removed
contribution. It is not permission to skip work that contributes to the result.

## Required evidence and frontier

Before searching, record the active model path, block mix, tensor shapes, dtype,
dominant kernel families, launch/layout costs, and a warm repeated full-DiT
profile. Each candidate has an OFF guard, OFF-identity proof, implementation
manifest and source hashes, engagement/fallback counters, warmup policy, timing
scope, compile/init cost, and a real full-workload run.

Microbenchmarks screen implementations but cannot establish end-to-end speedup.
Use the current target gap to avoid spending a five-prompt run on a candidate
whose targeted screen shows negligible impact, unless it is essentially free,
is needed as a composable primitive, or closes required coverage. Record the
screened hypothesis instead of repeatedly measuring equivalent flag variants.
The candidate commit must change production-consumed source; run artifacts,
evidence JSON, or a configuration file that is not wired into execution cannot
serve as the implementation patch.
Retain only measurable latency improvements or non-dominated peak-memory points.
Maintain a cumulative ON stack and periodically rerun its full-DiT and complete
frozen-workload gate. Delivery contains the fastest measured exact composed
stack plus its method-equivalence argument and unchanged logical counts.

Deliver at round 60, after a genuine multi-hypothesis plateau, or after a target
is reached and fully gated. A target is search pressure, not an acceptance rule
or proof of reachability.

Use the controller-generated `DELIVERY-CONTRACT.json` for exact schemas,
required artifacts, profile/baseline bindings, and pinned KernelWiki citation
paths. Run its static preflight command before returning control.
