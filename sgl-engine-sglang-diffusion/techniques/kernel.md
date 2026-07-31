# Kernel work-order scope

## Identity and correctness

`kernel` owns mathematically lossless optimization of SGLang Diffusion's
repeated transformer/DiT path. It has a hard budget of **40 candidate rounds**.
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

The root agent must first audit SGLang's existing fast paths. Auxiliary SGLang,
KDA-Pilot, and FastVideo knowledge may suggest implementations, but cannot
expand correctness or evidence policy.

## Knowledge and candidate coverage

Read the `kernel` projection in the bound `SEARCH-SPACE.json`, then query the
locked SGLang root kernel skills, SGLang Diffusion skills and source,
KDA-Pilot's diffusion kernel corpus, KernelWiki, NCU/warp-specialization
guidance, and FastVideo kernel material. Start with exact live shapes and
hotspots; a kernel that was successful on another model or GPU is only a
reference until its SGLang dispatch and end-to-end gain are measured.

Cover all applicable Sol kernel directions before declaring the lane exhausted:
GEMM epilogues, norm/modulation/residual fusion, attention-adjacent fusion,
compile/graph capture, layout/copy elimination, launch reduction,
stream/communication overlap, and decode/VAE/postprocess work. Every
implementation manifest cites exact source, commit, path, and SHA-256 knowledge
origins.

## Ownership boundaries

Topology owns CP/SP/TP/EP/FSDP/CFG degrees, process groups, rank maps,
collectives, placement, and multi-device scheduling. Kernel candidates preserve
the frozen topology. Cache owns approximate cross-step reuse; sparse attention
owns attention approximation; quantization owns sub-16-bit behavior; token
pruning owns reduced token work. Do not claim their gains in this lane.

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
Retain only measurable latency improvements or non-dominated peak-memory points.
Maintain a cumulative ON stack and periodically rerun its full-DiT and complete
frozen-workload gate. Delivery contains the fastest measured exact composed
stack plus its method-equivalence argument and unchanged logical counts.

Deliver at round 40, after a genuine multi-hypothesis plateau, or after a target
is reached and fully gated. A target is search pressure, not an acceptance rule
or proof of reachability.
