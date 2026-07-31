# Token-pruning work-order scope

## Adaptation status and identity

`token_pruning` brings Sol-Engine's full pruning/merging family into a
standalone SGLang work-order lane. It exists in Sol-Engine's paper,
implementation, and technique documentation but not in the reviewed
lightweight registry. The **20-round hard budget is an SGLang adaptation
default**, not an upstream lightweight setting.

This is a quality-gated lane. It reduces work on low-salience image/video tokens
in selected layers and denoising steps, then restores the full positional and
shape contract before downstream consumers.

Read every method and candidate in the `token_pruning` projection of the bound
`SEARCH-SPACE.json`, then query locked SGLang model/layout source, Sol
implementations, and transferable FastVideo/KDA evidence. Preserve the
documented, referenced, adapted, and validated status of each direction and
cite exact knowledge origins in every implementation manifest.

## Method families and boundaries

Profile and map the live token layout before selecting a pruning site. Preserve
prompt/context/control boundaries, spatial-temporal ordering, position/RoPE
state, attention masks and K/V layout, guidance branches, packed metadata,
sequence-parallel partitions, and downstream shapes.

In-scope families include feature-norm or dynamics-aware pruning, ToMe-style
merging and restoration, importance/region/attention-guided selection,
shape-stable compute masking, cluster representatives, and dynamic
layer/step/region density. Tune criterion, keep/prune ratio, layer and timestep
schedule, refresh policy, reconstruction/restoration, and dense fallback.

Do not claim cache-family reuse, sparse-attention approximation, quantization,
kernel fusion, topology, scheduler, step-count, resolution, prompt, or decode
gains. Token-wise feature reuse belongs here only when token selection is the
primary mechanism and is labeled explicitly.

Before closing the lane as `no_gain`, compare at least five applicable,
distinct reduction families: direct pruning, merging, compute masking,
region/dynamics-aware selection, and one of cluster, context, token-wise cache,
or dynamic-density policies. Preflight failures consume no scientific round.

## Required full-workload gate

Before quality runs, prove the OFF guard restores the source-current path and
that gathered, masked, or merged tokens restore positional/layout contracts.
Every scored point runs the complete frozen five-prompt workload.

Retained points require:

- original and reduced token counts, per-region density, selection/refresh
  pattern, and positive saved-work engagement;
- gather/scatter/merge/restoration overhead, attention/FFN timing, compile-cache
  and dense-fallback counters;
- exact criterion, layer/step schedule, restoration rule, activation,
  implementation manifest, source hashes, and run provenance;
- end-to-end performance rather than projected operator savings;
- aligned prompt-level LPIPS; and
- a passing built-in multimodal visual verdict with no external vision API.

The visual review emphasizes identity or object disappearance, prompt
misalignment, motion popping, temporal inconsistency, patch boundaries,
smearing, background drift, faces/hands/text, and restoration artifacts. A
scalar metric or nominal mask without actual token compute reduction cannot
pass.
