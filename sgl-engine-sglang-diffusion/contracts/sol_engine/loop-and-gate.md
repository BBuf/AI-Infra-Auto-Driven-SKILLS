# Loop and gate contract

This is an SGLang-oriented semantic snapshot of Sol-Engine's reviewed
orchestration contract. The upstream source and digest are recorded in
`source-lock.json`; this document adapts the contract without changing its
acceptance semantics.

- The baseline is frozen and never remeasured by an executor.
- One round is one hypothesis, one candidate, one real run, and one gate.
- Every candidate edits only its isolated experiment worktree, preserves the
  frozen workload and timing scope, and has an explicit OFF path that restores
  the source-current behavior.
- Lossless residency/kernel candidates (and any imported legacy topology
  artifact) are judged by mathematical and
  algorithmic equivalence, unchanged global logical denoising-step and
  DiT/model-call counts, and no approximation, sparsity, step skipping,
  sub-16-bit quantization, rank reduction, or changed logical work.
- Lossless candidates are never rejected using output differences, LPIPS,
  PSNR, floating-point tolerances, or visual quality. Output frames are used
  only to establish run authenticity.
- Quality-gated cache, PISA, quantization, and token-pruning candidates require
  aligned LPIPS, built-in multimodal visual review, real engagement, and the
  complete frozen workload.
- A quality-gated visual review uses the agent's built-in multimodal capability;
  it does not call Gemini or another external vision API.
- Every retained point names a real run directory and durable output, frame,
  benchmark, assessment, implementation-manifest, and provenance artifacts.
- An activation without positive engagement evidence, or with a disallowed
  silent fallback, is a no-op and cannot be retained.
- The master independently recomputes performance and verifies provenance,
  authenticity, quality, and the actual method before integration.
- A failed delivery is returned to the same executor with exact problems. It is
  never relabeled or accepted on the executor's assertion alone.
- Integration uses only independently verified frontier points and reruns the
  complete frozen workload. If any selected technique is quality-gated, the
  integrated recipe receives the quality gate; an all-lossless recipe never
  receives an output-difference, LPIPS, or visual-quality gate.

The technique scope's hard round budget governs one executor session. Meeting a
stretch target is not a substitute for passing the applicable gate, and an
unmet stretch target is not evidence that the target is theoretically
unreachable.
