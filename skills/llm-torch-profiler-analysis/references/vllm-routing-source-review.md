# vLLM MoE routing launch and dispatch review

Inspected on 2026-11-08 against vLLM
`b022bb7a4873a47cdcdf474edf3bd499c221f40d`. Read the complete final diff,
body and discussion, then the current gating translation unit, Python routing
callers, router GEMM dispatch and relevant regression tests. Historical GPU
reports were not rerun here.

## PR #59737 — Enable PDL for the fused softmax/sigmoid gating consumer

- Link: https://github.com/vllm-project/vllm/pull/59737
- State: merged 2026-10-19; head `8470dd48e75ca8ec392d689bbb49fa8dc76fbeef`,
  base `bde4d001130e5112d984ba99f0546e22c09e1a89`,
  merge `3c9beb82d1a92eb5a84cca4408a0a1b479d4dd03`. Fresh API head/base matched.
- Coverage: all 44 diff lines, one file, +24/-0; body, 23 issue comments,
  nine reviews and ten inline comments. The changed file is now
  `csrc/moe/topk_softmax_kernels.cu`; older review/body paths contain the
  retired `csrc/libtorch_stable/` prefix.
- Motivation: the FP32 router GEMM already supported programmatic dependent
  launch, but its fused top-k consumer used an ordinary launch. Enabling the
  consumer lets it become resident earlier while waiting for producer data.

The final diff adds the dependency wait and launch-completion signal before
row-dependent exits and reads. The shared template serves both softmax and
sigmoid, including bias and normalization; it does not change routing math,
output layout, block geometry or the Python/operator API.

```diff
+#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
+    // PDL permits early launch, but logits/metadata still depend on the producer.
+    cudaGridDependencySynchronize();
+    cudaTriggerProgrammaticLaunchCompletion();
+#endif
```

The CUDA host launch changes to `cudaLaunchKernelEx`, sets the programmatic
stream-serialization attribute and checks its return code. ROCm retains the
ordinary launch. In this PR's final head the attribute is set without a host
architecture query; only device instructions are guarded by SM90. Earlier
review requests removed a template flag and redundant CUDA-version checks.

The signal permits scheduling a dependent kernel; it does not certify that
gating output has been written. Consumers still need the dependency wait
before reading it. CUDA documents that the wait covers producer completion
and visibility of its global writes; concurrent execution is opportunistic.
See the [CUDA PDL contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

### Current admission and subsequent changes

Current main has restored a cached host `major >= 9` check and sets
`config.numAttrs = use_pdl ? 1 : 0`, with a comment explaining that the
attribute otherwise adds a programmatic CUDA Graph edge on pre-SM90 devices.
Do not copy the historical review's unconditional-host-policy conclusion
into current guidance. The device wait/signal remain before dependent reads.

`FusedTopKRouter` and `fused_topk_bias` call the custom softmax/sigmoid operators
on their NVIDIA path. The fused CUDA leaf requires packed logits rows and
one of its explicit expert-count specializations: powers of two through 512,
192/320/384/448/576, plus 352 for sigmoid only. Row-padded logits or unsupported
expert counts use the stride-aware score kernel followed by `moeTopK`; those
two ordinary launches did not acquire this PDL behavior. The outer wrapper
can allocate a workspace even when the selected expert count uses a fused
leaf, so an allocation alone does not prove fallback. ROCm AITER and the
separate sqrt-softplus/DSV4 routing providers need their own evidence.

The producing FP32 GEMM is also conditional. `GateLinear` requires an
unquantized FP32 weight, no bias, compatible architecture/shape and no
batch-invariant policy. Its runtime custom op avoids freezing the token-count
branch during compilation. The common token limit is 32. With BF16 input and
the BF16x3 alternative enabled, Hopper uses 24; on the non-Hopper path,
`(H,E)=(5120,256)` uses 48 for multiples of eight but otherwise 16, and
`(4096,352)` uses 11. `(8192,256)` is independently capped at 16. Larger calls
use BF16x3 or FP32 linear fallback. Thus the historical M=32/33 validation
boundary is not the universal current cutoff.

The producer kernel now signals immediately after its own dependency wait,
with subsequent shape-specific token grouping and launch geometry. These
later producer changes are not part of the one-file gating patch. A captured
router/gating residence overlap includes waiting time; inspect dispatch and
measure a synchronized chain rather than adding the overlapping durations.

### Validation and measured tradeoffs

The PR adds no tests or benchmark harness to the repository. It reports 866
unchanged upstream tests, 360 zero-tolerance operator comparisons per process,
1,200 preallocated-chain and 960 runtime-dispatch changing-input graph checks,
SM100a compilation and a standalone memcheck run. Equal NaNs were accepted;
NaN payload-bit identity was not asserted. These are author-reported results
from external harnesses, not reproduced checks of the final merged checkout.
Current `test_fused_topk.py` covers fused/fallback shapes, bias, dtypes,
renormalization, row padding and non-finite logits; those numerical tests do
not by themselves establish producer/consumer graph ordering.

The chain microbenchmark uses GB200, M=4/H=6144/E=128, top-4 sigmoid. Reported
hot-weight latency is 5.330→4.844 microseconds; rotating 64 distinct 3 MiB
weights saves 0.466 microseconds. Both sides include an approximately
0.682-microsecond non-PDL boundary to prevent overlap into the next repetition
and premature logits-buffer reuse. Isolated gating saves only 0.076
microseconds. Those measurements are not a model-level 9% speedup.

Serving uses MiniMax-M3 NVFP4, eight GB200s across two nodes, TP8 without EP,
FP8 KV/indexer cache, FlashInfer/TRT-LLM attention, no prefix caching or
speculation, and 64 requests in each of three valid A/B pairs per workload.
Mean TPOT improves 0.61–1.16% at concurrency 1/16/32 for the tested 128/1024
and 8192/1024 lengths; concurrency four is effectively unchanged. Pure-prefill
8192/1 TTFT regresses 0.54%/0.49% at concurrency 16/32. One pair with attention
JIT during measurement was excluded and replaced after increased warmup.
Finite bursts and three pairs do not establish statistical significance or
sustained-load capacity. Separate traces report overlap in all 6,720 patched
adjacent pairs, but residence overlap is not a direct latency measurement.

Accuracy uses a different setup: full 1,319-question GSM8K, five-shot,
temperature zero, concurrency eight, real EAGLE3 and three draft tokens.
Two repeats give baseline mean 94.81% and candidate 94.66%, a decrease of
0.152 percentage points versus 0.531 points between baseline repeats. This
does not establish equivalence, even though every request completed.

Serving/kernel checks overlay the PR CUDA file on runtime
`9ea8f3ffc354901b740f0b31988900897b7221d7` with PyTorch 2.13/CUDA 13;
accuracy compares official v0.29.0 with the candidate kernel at `713192bab`.
They are not full current-main builds. Later host-guard cleanup was checked
with formatting/pre-commit without rerunning GPU tests. Discussion records
approvals and repeated CI triggers, including a final-head retry of eight
previously failed steps; trigger comments are not final test results. Hopper,
ROCm and final-head full-build validation remain unestablished by these reports.

Current source:
[gating kernels and launcher](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/csrc/moe/topk_softmax_kernels.cu),
[router GEMM kernel](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/csrc/fp32_router_gemm.cu),
[router GEMM entry](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/csrc/fp32_router_gemm_entry.cu),
[GateLinear dispatch](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/model_executor/layers/fused_moe/router/gate_linear.py),
[standard routing](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py),
[biased routing](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py),
[custom operators](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/_custom_ops.py),
[current numerical tests](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/tests/kernels/moe/test_fused_topk.py).
