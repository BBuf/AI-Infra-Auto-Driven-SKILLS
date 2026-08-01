# PISA executor scope

## Identity and implementation rule

`pisa` owns the quality-gated PISA (Piecewise Sparse Attention) compatibility
lane and has a hard budget of **20 candidate rounds**. It is attention
approximation and routing only: do not claim cache, token-pruning,
quantization, VAE/text-encoder, scheduler, prompt, step-count, shape, topology,
or unrelated kernel gains.

PISA evaluates critical query-key blocks with exact softmax attention and
handles the remainder through its block-wise Taylor approximation. Preserve
the validated algorithmic components: Q/K/V chunk reduction, Taylor-error
block selection, exact selected-block attention, and
`approx_remainder = true`. Simply dropping unselected blocks, setting unused
metadata, or running dense attention with inactive routing is not PISA.

Record both:

```text
density = fraction handled by exact selected-block attention
sparsity = fraction handled by the approximate Taylor remainder
density = 1 - sparsity
```

The upstream validated block size 64 is an initial measured point, not a
mandatory final recipe. Adapt the implementation to SGLang's current attention
API and actual model shapes. Record source commit, source path, and digest for
the knowledge used; the final patch must be self-contained in SGLang.

## Fixed full-workload gate

An explicit OFF guard restores SGLang's source-current dense attention path and
must pass OFF identity before scoring. Every recipe point runs the complete
frozen five-prompt workload with unchanged model, checkpoint, VAE, scheduler,
prompt text, seed policy, resolution, duration, frames, fps, steps, guidance,
flow shift, motion score, decode, hardware, and timing scope.

Retained points require real PISA dispatch, positive exact and approximate
engagement statistics, zero disallowed silent fallback, durable output and
aligned frames, benchmark and run provenance, implementation manifest and
source hashes, aligned prompt-level LPIPS, and a passing built-in multimodal
visual review. No external Gemini or vision API may produce the verdict.

For every run report end-to-end time, isolated denoise/DiT time when available,
PISA kernel time, selection/approximation overhead, exact density and schedule,
dispatch and dense-fallback counts, peak memory, LPIPS, prompt-level visual
status, and artifact severity. Isolated attention or shortened-video tests are
screening evidence and cannot populate the delivered recipe frontier.
