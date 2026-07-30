# Cache executor scope

## Identity and method boundary

`cache` is Sol-Engine's quality-gated, closed compatibility comparison of
exactly three cross-step cache families:

1. TeaCache;
2. EasyCache; and
3. TaylorSeer.

It has a hard budget of **20 candidate rounds**. Cache-DiT, PAB, DeepCache,
FasterCache, generic fixed-step reuse, attention broadcast, token pruning,
kernel fusion, precision changes, scheduler changes, and cross-family hybrids
are outside this compatibility lane. SGLang cache families outside the closed
set must run in separately labeled extension lanes under the same quality-gated
semantics.

Within one family the executor may faithfully integrate the method, reduce its
bookkeeping overhead, choose cache payload and placement, and tune native
refresh, threshold, history, correction, layer, and timestep controls.

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

Compare quality only at matched measured end-to-end time:

```text
time_ratio = candidate_total_s / frozen_baseline_total_s
speedup = frozen_baseline_total_s / candidate_total_s
```

Two family points are matched only when their measured `time_ratio` differs by
at most 2% relative. Otherwise tune and rerun at the shared target; never
extrapolate from a microbenchmark. Repeat noisy or close points and use their
median complete-run time.

Build an evidence-backed speed/quality curve for each family over shared
feasible time targets. At every target report LPIPS mean/max, the built-in
multimodal verdict, and every prompt-level failure. Never hide a bad prompt in
an aggregate. Delivery identifies the quality winner and exact recipe at each
matched target, feasible ranges for unmatched families, and the overall Pareto
frontier.
