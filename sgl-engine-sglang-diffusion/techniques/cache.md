# Cache work-order scope

## Identity and search-space boundary

`cache` owns Sol-Engine's complete quality-gated cross-step reuse and
prediction space, adapted to the locked SGLang Diffusion model. It has a hard
budget of **20 candidate rounds**.

Do not inherit the lightweight Sol workflow's closed three-family boundary.
Read the `cache` projection in the work order's `SEARCH-SPACE.json` before
selecting a hypothesis. Consider every applicable family:

1. whole-step denoiser output reuse;
2. TeaCache-style timestep-aware reuse;
3. EasyCache-style adaptive transform/residual reuse;
4. PAB-style spatial, temporal, cross, joint, or model-specific attention
   broadcast;
5. block/layer feature caching, including Cache-DiT/DeepCache-style placement;
6. FORA-style transformer attention and MLP intermediate reuse;
7. token-wise feature caching;
8. CFG-aware branch reuse;
9. content- or motion-adaptive schedules;
10. predictive, delta, and Taylor-style forecasting; and
11. architecture-aware DiT, U-Net, stage, and resolution reuse.

Method names are discovery anchors, not portable implementations. Inspect the
live denoising loop, block boundaries, guidance branches, token layout, and
existing SGLang cache hooks before adapting a candidate. Do not copy
Cosmos3-specific Sol paths or environment flags without proving that the
locked SGLang code consumes them.

Before closing the family as `no_gain`, compare at least five distinct,
applicable directions: timestep-aware reuse, adaptive transform reuse,
attention broadcast, block/layer caching, and one of token-wise, CFG-aware,
motion/content-adaptive, or predictive caching. Record unsupported candidates
as preflight findings without consuming a scientific round.

## Knowledge and provenance

Use the bound `KNOWLEDGE.json` snapshots in this order:

1. locked SGLang cache/model/pipeline source and SGLang Diffusion skills;
2. the full Sol search document, candidate manifests, techniques, and site
   documentation;
3. KDA-Pilot and KernelWiki when cache bookkeeping exposes a kernel hotspot;
4. FastVideo cache/attention implementation evidence.

Every implementation manifest cites at least one exact source, commit, relative
path, and raw SHA-256 from those snapshots. Classify a method as documented,
referenced, adapted, or validated; never infer validation from a Sol manifest.

## Search axes and ownership

Within a family, discover signal, payload, placement, decision rule, refresh
policy, layer/step/region scope, guidance policy, forecast correction, and
dense fallback from target-model traces. Measure cache lookup, copy,
gather/scatter, prediction, and refresh overhead rather than assuming skipped
model work becomes end-to-end speedup.

Kernel owns same-step mathematically exact fusion and backend work.
Sparse-attention owns approximate attention routing. Quantization owns reduced
precision. Token pruning owns selection-driven reduced token work. Topology
owns multi-device partitioning and communication. A cache candidate may
interact with those lanes, but it must attribute only cross-step reuse or
prediction.

## Fixed full-workload gate

Every scored point runs the complete frozen workload: the same five prompts,
seeds, checkpoint, VAE, scheduler, resolution, duration, frames, fps,
denoising steps, guidance, flow shift, motion score, decode, hardware, and
timing scope. Module or single-DiT evidence is diagnostic only.

Retained points require real cache engagement, an OFF guard and OFF identity,
run provenance, output videos and aligned frames, end-to-end benchmark,
implementation manifest, source hashes, aligned prompt-level LPIPS, and a
passing built-in multimodal visual review. No external Gemini or vision API may
provide the verdict. Report signal source, reused payload, refresh rule,
hit/recompute pattern, fallback counters, full parameter point, parent trial,
and failure mode.

## Matched-time objective

Compare quality between families only at matched measured end-to-end time:

```text
time_ratio = candidate_total_s / frozen_baseline_total_s
speedup = frozen_baseline_total_s / candidate_total_s
```

Two family points are matched only when their measured `time_ratio` differs by
at most 2% relative. Otherwise tune and rerun at the shared target; never
extrapolate from a microbenchmark. Repeat noisy or close points and use their
median complete-run time.

Build an evidence-backed speed/quality curve for useful families. At every
target report LPIPS mean/max, the built-in multimodal verdict, and every
prompt-level failure. Never hide a bad prompt in an aggregate. Delivery
identifies the best exact recipe, feasible range, and overall Pareto frontier.
