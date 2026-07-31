# Sparse-attention work-order scope

## Identity and complete method space

`sparse_attention` owns Sol-Engine's complete quality-gated, training-free
sparse-attention space and has a hard budget of **20 candidate rounds**. PISA
is one candidate family, not the boundary of this lane.

Read the `sparse_attention` projection in the work order's
`SEARCH-SPACE.json`. Consider all nine canonical Sol directions:

1. Piecewise/PISA exact-block selection with an approximate remainder;
2. Sparse VideoGen-style spatial/temporal head routing;
3. Sparse VideoGen2 semantic-aware token permutation;
4. AdaSpa-style online precise search and mask reuse;
5. SpargeAttn-style proxy or universal mask prediction;
6. LVSA-style rotating anchors and long-video windows;
7. SVOO-style layer profiling and QK co-clustering;
8. HASTE-style head-wise adaptive budgets; and
9. MInference-style dynamic pattern selection.

Also inspect the locked Sol-Attn backends and policy code when the target GPU,
attention shape, dtype, token layout, and SGLang backend seam are compatible.
Some Sol manifests are `env_only` or `blocker_probe`; preserve that status and
do not present them as complete public CUDA ports.

Before closing the family as `no_gain`, review all applicable method entries
and compare at least five genuinely distinct sparse policies when model and
hardware capabilities permit. Unsupported token layouts, unavailable backend
seams, or incompatible GPU architectures are preflight findings and consume no
scientific round.

## Knowledge and adaptation

Use the bound sources in this order:

1. locked SGLang attention implementations, model token layout, distributed
   partitions, attention backend registry, and Diffusion skills;
2. Sol search documents, structured candidates, policy implementations,
   Sol-Attn kernels, capability requirements, and composition rules;
3. KDA-Pilot/KernelWiki/NCU material for kernel implementation and measured
   bottleneck analysis;
4. FastVideo attention and sparse-routing evidence.

Every implementation manifest cites exact knowledge origins. The final patch
must be self-contained in SGLang and must prove that its activation reaches the
adapted SGLang path. A Sol dry run, environment variable, manifest, or policy
name is reference evidence only.

## Algorithm and ownership rules

Map spatial, temporal, text, reference, conditioning, and guidance tokens
before choosing a pattern. Record head/layer/step routing, block or window
shape, token permutation and inverse mapping, selection signal, mask reuse,
dense guards, and fallback policy.

For PISA, preserve Q/K/V chunk reduction, Taylor-error block selection, exact
selected-block softmax attention, and the approximate remainder. For every
method, define exact density and effective sparsity from work actually
dispatched:

```text
density = fraction handled by the declared exact/dense path
sparsity = fraction whose dense attention work is actually avoided
```

Do not claim cache, token-pruning, quantization, VAE/text-encoder, scheduler,
prompt, step-count, shape, topology, or unrelated kernel gains. Kernel may
implement the sparse primitive, but this lane owns its approximation,
activation, quality, and end-to-end claim.

## Fixed full-workload gate

An explicit OFF guard restores SGLang's source-current dense attention path and
must pass OFF identity before scoring. Every recipe point runs the complete
frozen five-prompt workload with unchanged model, checkpoint, VAE, scheduler,
prompt text, seed policy, resolution, duration, frames, fps, steps, guidance,
flow shift, motion score, decode, hardware, and timing scope.

Retained points require real sparse dispatch, positive avoided-work
engagement, zero disallowed silent fallback, durable outputs and aligned
frames, benchmark and run provenance, implementation manifest and source
hashes, aligned prompt-level LPIPS, and a passing built-in multimodal visual
review. No external Gemini or vision API may produce the verdict.

For every run report end-to-end time, denoise/DiT time, sparse kernel time,
selection/permutation/mask overhead, exact density, effective sparsity,
head/layer/step schedule, dispatch and dense-fallback counts, peak memory,
LPIPS, prompt-level visual status, and artifact severity. Isolated attention or
shortened-video tests are screening evidence only.
