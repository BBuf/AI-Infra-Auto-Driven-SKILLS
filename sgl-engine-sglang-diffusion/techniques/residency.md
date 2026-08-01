# Residency executor scope

## Identity and correctness

`residency` owns mathematically lossless optimization of component placement,
offload, prefetch, and lifetime on the exact GPUs selected by the frozen
baseline. It has a hard budget of **20 candidate rounds**. One scientific round
is one profile-driven hypothesis, one isolated implementation, one complete
frozen-workload run, and one gate. Capability checks, memory probes, rejected
incompatibilities, and build failures are preflight and do not consume a round.

Preserve model inputs, dtype policy, global denoising-step and DiT/model-call
counts, GPU UUIDs, GPU count, and every CP/SP/TP/EP/FSDP/CFG degree. Residency
may change only declared component placement and transfer scheduling. Numeric
movement from an equivalent transfer/compute schedule is not a rejection
reason; output-difference or visual metrics never gate this lossless lane.

## Owned optimization surface

Profile the complete load-excluded E2E path and record selected-GPU total and
minimum-free memory before choosing a candidate. The five required coverage
families are:

- `component-residency`: DiT, text/image encoders, VAE, vocoder, bridges, and
  other repeatedly used auxiliaries;
- `partial-dit-residency`: keep a measured subset of DiT layers resident while
  streaming the remainder;
- `layerwise-prefetch`: tune transfer depth and copy/compute overlap without
  mistaking prefetched layers for persistent layers;
- `compile-time-residency`: use transient placement to survive compilation and
  restore the declared steady-state map before scored inference; and
- `load-order-lifetime`: order construction, release, and reuse according to
  measured component size and stage lifetime.

The executor must test relevant components independently before stacking them.
High VRAM is a signal to search, not a universal rule to disable all offload.
A historical threshold may seed preflight but cannot establish applicability or
acceptance. Search partial residency when full residency does not fit, and
search prefetch independently because prefetched layers may be transferred on
every denoise step.

## Conflicts and ownership

Resolve behavior against the source-current locked SGLang checkout. DiT
layerwise offload is incompatible with FSDP or Cache-DiT unless that checkout
has an explicit verified path. Explicit user memory flags are part of the
baseline: a candidate may change them only in its declared activation and its
OFF path must restore them. Compiler/kernel fusion belongs to `kernel`;
parallel degrees and rank maps are frozen; approximate cache, precision, sparse
attention, and pruning remain in their quality-gated lanes.

## Required evidence and frontier

Every retained frontier point includes `RESIDENCY-EVIDENCE.json`, bound to the
candidate, run, raw profile digest, and controller-owned `GPU-INVENTORY.json`.
That inventory resolves the frozen command's `CUDA_VISIBLE_DEVICES` (or the
controller-visible default order) to exact GPU UUIDs and total memory before
any executor starts. Record baseline and
candidate memory snapshots (total, minimum-free, peak, safety margin), H2D copy
count and duration, resolved component maps and footprints, DiT layer/prefetch
counts where applicable, separate compile and steady-state placement, conflict
checks, positive engagement counters, and hashed complete-run/equivalence
artifacts.

Candidate peak plus the declared safety margin must fit the measured GPU
envelope. A no-op, stale profile, GPU drift, contradictory strategy, un-restored
compile placement, or copied threshold fails closed. Retain only measured
latency improvements or non-dominated memory points. Register positive
residency candidates for cumulative integration with every compatible lane;
one rejected residency idea never closes the lane. Delivery requires a complete
disposition of all five coverage IDs.
