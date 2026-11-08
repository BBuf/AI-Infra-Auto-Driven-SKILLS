# TokenSpeed routing, sampling and attention launch overhead

Inspected on 2026-11-08. All three full diffs, PR bodies and available comments
and reviews were read. Fresh API checks matched each head/base below. Current
callers and implementations were checked at TokenSpeed
`7b21034e72bb45d1edef3d3b5483b50b58c4c492`, with FlashInfer API contracts at
`31c146b16789f112e9b9861bd79247d4701e437a`. These are source reviews; the
historical GPU measurements below were not rerun for this refresh.

## PR #1606 — Pass unpacked routes to FlashInfer

- Link: https://github.com/lightseekorg/tokenspeed/pull/1606
- State: merged 2026-10-06; head `5a541c2820734f38503d0c8ffe15733f2cc4c34e`,
  base `ae4a958664aa6085493ea612336f749aca532da0`,
  merge `9b818230a79c234e72a4927ebdb87d2731b43796`.
- Coverage: all 114 diff lines, two files, +46/-14; one issue comment,
  three reviews and no inline comments. Two approvals cover the final head;
  the earlier-head review was dismissed.
- Reviewed files: implementation
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer/trtllm_unquant.py`
  and behavioral test
  `tokenspeed-kernel/test/nvidia/thirdparty/test_moe_finalize_fuse_shared.py`.
- Motivation: every precomputed-route call used casts, masks, shifts and ORs
  to build packed expert-ID/weight words even though the pinned FlashInfer
  0.7.1rc2 accepted separate tensors. That setup is visible at small batches.

The wrapper retains contiguous int32 IDs and BF16-rounded weights, but passes
the pair directly to `trtllm_bf16_routed_moe`. This removes packing work;
it does not remove route-weight conversion or change top-k selection.

```diff
+            topk = (
+                topk_ids.to(torch.int32).contiguous(),
+                topk_weights.to(torch.bfloat16).contiguous(),
             )
```

In-kernel routing from logits and weight preparation stay separate. With
`do_finalize=False`, the result still carries GEMM2 output, expert weights and
the expanded-to-permuted index map. A shared-expert finalizer may deliberately
use the original FP32 route weights: equality of packed and unpacked BF16
finalized outputs does not establish bitwise equality to that FP32 finalize.
The test compares packed/unpacked finalized results exactly, changes strided
route weights before graph replay, and then checks the separate shared-expert
finalize with `atol=rtol=5e-2` to accommodate its different rounding.

### Current path and validation limits

The old implementation subdirectory is gone. The body and registrations now
live in `ops/moe/flashinfer.py`. The BF16 precomputed SwiGLU/SiLU and Relu2
registrations admit NVIDIA SM100–SM103, unquantized weights, deferred finalize
and EP, but not all-to-all EP. Kernel selection still checks activation,
alignment, numerical features and routing mode; a visible top-k tensor alone
does not prove this provider was selected. `MoELayer.forward` distinguishes
bypassed/in-kernel routing from precomputed routing and checks the plan's
capabilities before calling `moe_apply`.

The current shared body preserves the same BF16 pair. Gated intermediate
sizes may use a private 64-aligned launcher only when its installed-source
adapter and JIT compiler checks succeed; otherwise planning retains the stock
128 alignment. Relu2 retains 128. The private entry points clone FlashInfer's
API dispatch and forward the same arguments. FlashInfer main's public routed
API accepts both BF16 and FP32 weights, but this TokenSpeed BF16 path explicitly
rounds to BF16. Do not import the native-weight behavior of the separate NVFP4
path into this claim. Current TokenSpeed requirements pin FlashInfer 0.7.3;
that is distinct from the wheel named in the historical report.

The body reports 23 native tests on GB200, 15 kernel-selection tests with six
CPU-platform skips, and pre-commit success. The added regression selects
`(tokens=1, PDL=false)` and `(tokens=33, PDL=true)`; it is not a full Cartesian
product. Its GB200 benchmark uses 16 BF16 experts, top-4, hidden 4096,
intermediate 1024, strided FP32 routes and finalized output. Warmed graphs
contain 32 calls, measured old/new/new/old with medians of 15 runs. Per-call
latency is 25.22→20.59 microseconds at batch 1, 71.35→66.60 at batch 8, and
79.56→72.30 at batch 33. Outputs match exactly for those measurements. No full
serving throughput or latency was measured, and the body does not pin a
separate benchmark candidate SHA.

Current source:
[BF16 provider and registration](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer.py),
[plan/apply](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/__init__.py),
[runtime expert caller](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/layers/moe/expert.py),
[private alignment adapter](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/trtllm_bf16_moe.py),
[FlashInfer tuple routing ABI](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/fused_moe/core.py).

## PR #1607 — Skip an unused fallback-index reduction

- Link: https://github.com/lightseekorg/tokenspeed/pull/1607
- State: merged 2026-10-06; head `27ebc3d6a8ae2fefbdc39381288a7b7ee94650aa`,
  base `e83bc7743217658c5b87b0e8306eb59e1eae387e`,
  merge `ae4a958664aa6085493ea612336f749aca532da0`.
- Coverage: all 163 diff lines, two files, +62/-21; one issue comment,
  one final-head approval and no inline comments.
- Reviewed files: vendored kernel
  `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/include/flashinfer/sampling.cuh`
  and `tokenspeed-kernel/test/nvidia/thirdparty/test_cuda.py`.
- Motivation: the common probability sampler computes the last valid token
  index for each vocabulary chunk. The target-only chain kernel consumes
  `sampled_id` instead, so this extra integer block reduction is unused there.

`TRACK_LAST_VALID` is a compile-time template flag around the integer index
construction, max reduction and `last_valid_id` write. The probability sum,
CDF scan, sampling comparison, final barrier and aggregate update remain.
Only the target-only chain call passes false:

```diff
+    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
+                           DETERMINISTIC, false>(
```

The six other calls pass true: ordinary sampling, Top-K, Top-P, Min-P,
Top-K+Top-P and the separate general speculative sampler. Removing the reduction
globally would remove their fallback. The optimized chain initializes
`sampled_id` to `vocab_size-1` and reads that field for its residual draw;
it never substitutes `last_valid_id`.

### Current dispatch and acceptance rules

The current header retains the flag and all seven call-site choices. Its
`ChainSpeculativeSamplingKernelTargetOnlyFastPath` name is historical: the
dispatcher also specializes `USE_DRAFT_PROB`. With recorded draft probabilities
it uses rejection sampling and a residual distribution; without them it uses
the target-only threshold rule. Both specializations use the same optimized
CDF helper. Sentinel proposal rows and rejection handling have evolved after
this PR; the missing index reduction is not evidence of a new acceptance rule.

The Python adapter loads TokenSpeed's native `sampling_chain` library, with
FP32 probability tensors and int32 output/index conventions. The current API
requires explicit `use_draft_prob` and `reject_draft_prob_threshold`; recorded
probabilities are mandatory for the former. The native dispatcher launches one
1024-thread CTA per request, choosing vector width from vocabulary alignment,
deterministic mode, draft-probability mode and PDL. The serving FlashInfer
backends choose this chain path, a separate block verifier or a single-token
per-request draw according to configuration. Greedy verification can use the
same probability path with top-k=1 and a numerical-policy correction.
Synthetic acceptance can replace the chain's
outcome, so its reported lengths must not be mistaken for ordinary acceptance.

The new test uses sparse mass at IDs 0, 2048 and 4095 and coins at 0, 0.25,
0.75 and just below 1, checking exact tokens, positions and zero accepted-draft
counts with deterministic mode both on and off. The body reports 26 selected
tests plus 96 old/new exact comparisons, including draft-buffer writeback,
PDL and deterministic variants. Its performance report uses GB200, FP32,
deterministic mode, PDL off, no draft probabilities and unit thresholds:
batch 1/chain 1/vocabulary 131072 improves 47.44→39.77 microseconds;
batch 8/chain 4 improves 52.59→43.61. Each graph contains 256 calls, with
31-run medians and six alternating measurements per version. These are native
kernel timings, not full-serving or other-architecture measurements. No local
GPU run or final-head CI-log verification was performed in this refresh.

Current source:
[helper, chain kernel and dispatcher](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/include/flashinfer/sampling.cuh),
[Python API](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/sampling_chain.py),
[native binding](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/sampling_chain.cu),
[serving dispatch](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/sampling/backends/flashinfer.py).

## PR #1889 — Reuse the TRT-LLM MHA completion counters

- Link: https://github.com/lightseekorg/tokenspeed/pull/1889
- State: merged 2026-10-06; head `b3a864f4ecd00e3762845ec98a3ba311de3291d1`,
  base `93f0534f44ba9c9f5fd5d1379c114ee4c8d232e1`,
  merge `e83bc7743217658c5b87b0e8306eb59e1eae387e`.
- Coverage: all 277 diff lines, three files, +207/-0; no issue/inline comments,
  one final-head approval.
- Reviewed files: runtime
  `python/tokenspeed/runtime/layers/attention/backends/paged/trtllm.py`,
  public helper export
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mha/flashinfer.py`,
  and `test/runtime/test_trtllm_kv_counter_buffer.py`.
- Motivation: omitting FlashInfer's optional counter buffer allocated and
  zero-filled it at every attention layer. Graph capture retained that fill
  kernel, so it recurred during replay too.

The leaf allocates one zeroed uint8 buffer on CUDA, sized by FlashInfer's public
helper for maximum requests times DFlash block-decode expansion, TP query heads
and SM count. The helper uses `4 * round_up(max(requests * q_heads, SMs), 8)`
bytes. Under the API's self-reset contract, completed attention calls leave
the counters zero; the leaf can reuse them without another fill. This is
owner-local state, independent of KV-pool rebinding and attention scratch.

The per-call check returns `None` if capacity is insufficient, preserving
FlashInfer's fresh-allocation fallback. The decode call counts requests rather
than all query tokens:

```diff
+            multi_ctas_kv_counter_buffer=self._kv_counter_buffer_for(
+                q.shape[0] // metadata.max_seq_len_q, q.shape[1]
+            ),
```

Extend uses `cu_seqlens_q.shape[0]-1`. DFlash represents each block position
as a single-query request, so its expanded request count is intentional.
CPU leaves keep no buffer. Sequential reuse relies on completed/self-reset
launches; these checks do not authorize sharing one counter allocation across
concurrent attention calls on independent streams.

### Current evolution and validation limits

Current `bind_tree_draft` can grow the owner buffer to cover
`max_bs * tree_topk` lane requests. `_tree_cascade` also passes it to the
prefix-attention call, using `query_rows / q_len`. Thus the original body's
fixed-size description is no longer the complete lifetime contract: tree
binding can replace the allocation, and graph users must bind the correct
tree geometry before capturing references to it. Normal decode/extend retain
the capacity fallback and pool rebind preserves the buffer.

The generic registered `flashinfer_trtllm_mha_*_with_kvcache` kernel wrappers
in `ops/attention/mha/flashinfer.py` still omit this argument. This PR improves
the runtime `TRTLLMMHAAttnBackend` leaf, not every use of similarly named
attention entry points. Trace attribution must identify the caller.
FlashInfer's current resolver checks device, contiguity, workspace alignment
and byte capacity; it does not zero a supplied buffer. First-use initialization
and subsequent self-reset remain part of the caller/kernel contract.

The added tests check decode/extend identity reuse, oversize fallback, the
SM-count floor, cache-pool rebinding, CPU behavior and CUDA zero initialization.
The body reports 192 related CPU tests versus 188 on its baseline, with two
pre-existing CUDA-dependent FlashMLA rebind failures on both; 16 GPU runtime
tests, six MHA tests and 389 real-kernel checks on GB200. Reported checks cover
exact outputs, counters zero after calls, changed-input graph replay, and
`MultiCtasKv`/`MultiCtasKvCga` execution with fill launches reduced from one to
zero. They used FlashInfer **0.6.18**, while the PR's 0.7.0 pin was compared
only in source. Current TokenSpeed pins 0.7.3; none of those wheel versions
were executed locally for this refresh.

The attention-only graph benchmark uses BF16, head dimension 128, page 64,
4096 KV tokens and five sets of 100 replays. Qwen3-8B geometry (36 layers,
32/8 query/KV heads), batch 1 improves 412.1→336.8 microseconds per attention
step; Qwen3-32B geometry (64 layers, 64/8 heads) improves 731.0→597.1.
Reported savings over batches 1/8/32 range from 1.17 to 2.09 microseconds
per layer. Those geometries are not full model execution or TPOT evidence.

Current source:
[runtime leaf](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/layers/attention/backends/paged/trtllm.py),
[exports and generic wrappers](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mha/flashinfer.py),
[FlashInfer size/resolver contract](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/utils.py),
[decode argument and resolver](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/decode.py),
[context argument and resolver](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/prefill.py).
