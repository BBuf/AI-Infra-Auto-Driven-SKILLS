# Topology work-order scope

## Identity and correctness

`topology` is an optional Sol-Engine-compatible work-order technique with a
hard budget of **20 candidate rounds**. It losslessly optimizes how the
unchanged SGLang Diffusion inference program is partitioned, placed,
communicated, and scheduled within the exact frozen GPU resource envelope.

The global mathematical function and logical work remain unchanged. Preserve
global logical denoising-step and DiT/model-call counts. Floating-point
reduction order may move outputs and is not a correctness failure; no output
difference, LPIPS, PSNR, tolerance, or visual-quality metric may reject a
candidate.

## Owned optimization surface

Freeze both the semantic workload and resource/measurement envelope in
`TOPOLOGY-PREFLIGHT.json` before searching. Profile first, then vary one measured
bottleneck per candidate. In scope are:

- context/sequence parallelism including Ulysses, Ring, or justified hybrids;
- tensor and expert parallelism;
- FSDP, reduce-scatter/all-gather forms, replication, and residency;
- equivalent batched versus parallel CFG branches;
- device meshes, rank ordering, process groups, and their nesting/reuse;
- activation, parameter, expert, and stage placement; and
- exact collective or P2P scheduling, chunking, prefetch, and overlap when
  covered by the frozen timing scope.

Kernel owns local backend, fusion, compilation, and local kernels. Cache owns
cross-step approximation, PISA owns attention approximation, quantization owns
precision reduction, and token pruning owns reduced token work. Preserve the
frozen GPU count and dtype policy.

## Candidate and evidence contract

Every ON path has a candidate-specific OFF guard that restores the frozen
topology and fails closed for unknown combinations. State every rank coordinate,
every process group's members, and whether axes are orthogonal, nested, or
reuse ranks. Distributed preflight must prove complete token/head/feature,
parameter/expert, and CFG coverage; resolved reductions; consistent collective
ordering; valid async lifetimes; all-rank participation; and no silent fallback.

Every scored run preserves four run-local artifacts with identical
`candidate_id` and `run_id`:

1. `topology_preflight.json`, with all checks explicitly passing;
2. `topology_manifest.json`, with declared topology and real source hashes;
3. `topology_trace.json`, with exactly one participating record per rank,
   collectives, bytes/timing, positive total time and peak memory, and fallback
   counters; and
4. `equivalence.json`, with equal global logical step/model-call counts and the
   method argument.

Process groups, rank map, placement, and collectives are non-empty descriptions
of the actual ON run. Observed trace activity must agree with the manifest. A
label or environment variable without dispatch and participation evidence is
not an implementation.

Only a complete frozen-workload run establishes a frontier point. Retain a
mathematically equivalent candidate that measurably improves latency or creates
a non-dominated peak-memory point. Delivery includes exact reproduction,
activation, source hashes, all four topology artifacts, per-rank evidence,
authoritatively recomputed speedup, and no projected or fallback points.
